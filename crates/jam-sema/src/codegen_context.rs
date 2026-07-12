/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The codegen context — the backend's hub. Owns the LLVM module/builder and
//! the shared frontend pools, caches the TypeIdx→LLVM-type lowering, and holds
//! the struct/enum/union registries. Ported (incrementally) from the
//! `JamCodegenContext` in `src/codegen.{h,cpp}`.
//!
//! ## Lifetime design
//!
//! The C++ context owns its `llvm::LLVMContext` AND caches `llvm::Type*`
//! handles into it. In Rust that shape is self-referential and illegal, so we
//! follow the inkwell pattern: the *driver* creates the [`Context`] first, and
//! `CodegenContext<'ctx>` borrows `&'ctx Context`, caching `Type<'ctx>` handles
//! that all share the context's lifetime. The owned `Module`/`Builder` are also
//! `'ctx`.
//!
//! ## Scope of this increment
//!
//! [`CodegenContext::get_llvm_type`] lowers the cases that do NOT need the
//! analyzer-filled layout passes or the generic-substitution engine:
//! primitives, pointers, slices (`{ptr,len}`), arrays, and registry-backed
//! struct/union/enum types (once a body fills the registries). The
//! substitution/alias/chained `Named` resolution order, generic-call and
//! array-expr resolution, and `requalify_type` land with `decl` + the analyzer
//! and return an explicit error here until then.

use std::cell::{Cell, Ref, RefCell};
use std::collections::HashMap;

use jam_core::diag::{Diagnostics, SrcLoc, Trace};
use jam_core::index::{NodeIdx, StringIdx, TypeIdx};
use jam_llvm::{Context, Module, Type};
use jam_syntax::ast::{EnumDeclAST, FunctionAST, StructDeclAST};
use jam_syntax::ast_flat::{AstTag, NodeStore, StringPool, TypeKind, TypePool};

use crate::comptime::{
    CompCallResolver, CompCtx, ComptimeEvaluator, ComptimeScope, ComptimeValue, DEFAULT_ITER_CAP,
    ExecResult,
};
use crate::generics::{generic_arg_spelling, substitute_type};
use crate::target::Os;

/// Round `off` up to the next multiple of `align` (matches the C++ `alignUp`,
/// `(off + align - 1) / align * align`, without its intermediate overflow).
fn align_up(off: u64, align: u64) -> u64 {
    off.div_ceil(align) * align
}

/// Module-qualified identity for a type/decl: `modulePath.name`, or just `name`
/// for the entry module. Two modules that each define `Thing` get distinct
/// identities (`a.Thing` vs `b.Thing`).
pub fn qualify_type_name(module_path: &str, name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        format!("{module_path}.{name}")
    }
}

/// A registered struct: its LLVM type and its (name, type) fields.
#[derive(Clone, Debug)]
pub struct StructInfo<'ctx> {
    pub name: String,
    pub ty: Type<'ctx>,
    pub fields: Vec<(String, TypeIdx)>,
}

/// A registered untagged union: every field shares the same address.
#[derive(Clone, Debug)]
pub struct UnionInfo<'ctx> {
    pub name: String,
    pub ty: Type<'ctx>,
    pub fields: Vec<(String, TypeIdx)>,
}

/// One enum variant: a name, positional payload types (empty for unit), and a
/// runtime discriminant.
#[derive(Clone, Debug)]
pub struct EnumVariantInfo {
    pub name: String,
    pub payload_types: Vec<TypeIdx>,
    pub discriminant: u32,
}

/// A registered enum. Unit-only enums lower to `i8`; payloaded enums lower to
/// `{ i8 tag, [N x i8] payload }` (the named LLVM type filled by the analyzer
/// via [`CodegenContext::set_enum_llvm_type`]).
#[derive(Clone, Debug)]
pub struct EnumInfo<'ctx> {
    pub name: String,
    pub ty: Option<Type<'ctx>>,
    pub variants: Vec<EnumVariantInfo>,
    pub has_payload_variant: bool,
    pub max_payload_size: u64,
    pub max_payload_align: u64,
}

/// A generic's deferred method-body lowering: its instantiated method-signature
/// clones, the substitution to lower them under, and the body module that was
/// active at stash time (so the drain lowers the bodies in the same scope the
/// non-deferred path would have, not the empty stack at drain time).
type DeferredMethodWork = (Vec<FunctionAST>, HashMap<String, TypeIdx>, String);

/// The codegen context. See the module docs for the lifetime rationale.
pub struct CodegenContext<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: jam_llvm::Builder<'ctx>,

    // Shared frontend pools (filled by the parser, then read by the backend).
    pub type_pool: TypePool,
    pub string_pool: StringPool,
    pub node_store: NodeStore,

    // TypeIdx -> LLVM type, memoized. Interior-mutable so the by-`&self`
    // lowering can fill it; recursion never holds the borrow across a call.
    llvm_type_cache: RefCell<Vec<Option<Type<'ctx>>>>,

    // TypeIdx -> size / alignment in bytes, memoized. (The substitution-context
    // bypass the C++ applies lands with the substitution engine; with no active
    // subst yet, unconditional memoization is correct.)
    size_memo: RefCell<HashMap<TypeIdx, u64>>,
    align_memo: RefCell<HashMap<TypeIdx, u64>>,

    // Type registries, keyed by module-qualified name. `RefCell` so the
    // demand-driven analyzer can register bodies through `&self` *during*
    // recursive lowering (`get_llvm_type` -> ensure-body -> `get_llvm_type`),
    // mirroring the C++ `mutable` registries. Lookups therefore return Copy /
    // cloned data, never a borrow that would outlive the `Ref` guard.
    structs: RefCell<HashMap<String, StructInfo<'ctx>>>,
    unions: RefCell<HashMap<String, UnionInfo<'ctx>>>,
    enums: RefCell<HashMap<String, EnumInfo<'ctx>>>,

    // Accumulated diagnostics. RefCell so the demand-driven analyzer can push
    // errors through `&self` during lowering.
    diagnostics: RefCell<Diagnostics>,
    // Display filename stamped onto astgen diagnostics (the C++ `currentFile_`,
    // read by `locOf`). Entry-global, temporarily swapped to `<key>.jam` while an
    // imported module's bodies lower (main.cpp:2253-2286).
    current_file: RefCell<String>,
    // Module-path stack — the analyzer pushes a type's OWNING module while
    // filling its body so requalify / privacy use the right scope (the C++
    // `BodyModuleGuard`). `current_body_module` reads the top (empty = entry).
    body_module_stack: RefCell<Vec<String>>,
    // (ctxModule -> (bareTypeName -> ownerModule)). Built up front by the
    // driver; consulted by `requalify_type`.
    type_module_of: RefCell<HashMap<String, HashMap<String, String>>>,

    // Registered function/method signatures, keyed by the lookup name the call
    // site uses (bare for entry-module free fns, qualified otherwise). Owns a
    // clone of each `FunctionAST` — cheap, since the body is just `NodeIdx`
    // indices into the shared `NodeStore`. Mirrors the C++ `fnRegistry` of
    // `const FunctionAST*`; AstGen reads it to lower `Call` (mangled name + ABI).
    function_asts: RefCell<HashMap<String, FunctionAST>>,

    // Withdrawn instantiated methods: qualified name
    // ("Vec__Counter.withCapacity") -> human-readable reason. A call to a
    // withdrawn method reports "not available for this instantiation" with the
    // recorded reason instead of a bare "unknown method" (the C++
    // `withdrawnMethods_`).
    withdrawn_methods: RefCell<HashMap<String, String>>,

    // Type name -> its `cfn drop`'s mangled LLVM symbol. Pre-mangled (owned
    // strings) so the registry doesn't borrow the module AST. Consulted by
    // `lookup_drop_fn_name` / `type_needs_drop` for scope-exit drop emission.
    drop_fns: RefCell<HashMap<String, String>>,

    // Struct name -> its TOP-LEVEL `cfn clone(self: T) T` (the C++ `CloneRegistry`).
    // In-struct clones register as `Name.clone` methods (ordinary dispatch); only
    // the top-level form lives here, routed through emit_clone_into's glue.
    clone_fns: RefCell<HashMap<String, FunctionAST>>,

    // (ctx_module, bare_name) -> pre-interned qualified `Named` TypeIdx. Lets
    // `requalify_type` map a module body's bare type reference to its qualified
    // identity as a pure `&self` lookup (interning happens in the register pass).
    requalify_map: RefCell<HashMap<(String, String), TypeIdx>>,

    // Active generic-substitution frames (innermost last). Each frame maps a
    // generic parameter name (`T`) to its concrete argument while a generic
    // instantiation's body is lowered (the C++ `currentSubst_`).
    current_subst: RefCell<Vec<HashMap<String, TypeIdx>>>,

    // Active comp-value substitution context (the C++ `currentCompSubst_`).
    // Parallel to `current_subst` but carries baked `ComptimeValue`s: set right
    // before a comp-instantiated fn clone's body is lowered, so a body
    // reference to a comp param (`k`) folds to the call-site constant.
    current_comp_subst: RefCell<HashMap<String, ComptimeValue>>,

    // module_path -> the module's anonymous struct bodies (the `return struct
    // {...}` of a generic factory), so generic instantiation can reach the body
    // a `StructExpr` node references (its `lhs` indexes this per-module list).
    anon_structs: RefCell<HashMap<String, Vec<StructDeclAST>>>,
    // module_path -> the module's anonymous enum bodies (`return enum {...}`).
    anon_enums: RefCell<HashMap<String, Vec<EnumDeclAST>>>,
    // Type-alias name -> target TypeIdx: `const Foo = Bar` and destructuring
    // imports (`const { Token } = import("m")` aliases Token -> m.Token).
    // Resolved regardless of body module (the C++ scoped-alias table).
    type_aliases: RefCell<HashMap<String, TypeIdx>>,
    // Import-handle name -> resolved module identity: `const lib = import("m")`
    // makes `lib.fn()` resolve to `m.fn`.
    import_handles: RefCell<HashMap<String, String>>,
    // (body_module, GenericCall TypeIdx) -> the instantiated `Named` TypeIdx.
    // Keyed by the body module too: the SAME bare `Pair(u64)` TypeIdx resolves
    // to mod_gen_a.Pair__u64 in one module and mod_gen_b.Pair__u64 in another.
    // A pure `&self` lookup; instantiation runs earlier with `&mut`.
    generic_resolutions: RefCell<HashMap<(String, TypeIdx), TypeIdx>>,
    // Module-scope `const NAME = expr;` declarations, by name (bare + qualified).
    // No LLVM global is emitted: each use site re-lowers the initializer.
    module_consts: RefCell<HashMap<String, ModuleConstInfo>>,
    // ArrayExpr TypeIdx (`[expr]T`) -> resolved `Array(T, n)` (n folded from the
    // length expr). A pure `&self` lookup; resolution runs earlier with `&mut`.
    array_expr_resolutions: RefCell<HashMap<TypeIdx, TypeIdx>>,
    // Per-module namespace for multi-dot chained access (`w.leaf.Point`):
    // module identity -> its pub types + pub re-export module aliases.
    module_namespaces: RefCell<HashMap<String, ModuleNamespace>>,
    // Hook into astgen: lower an instantiated generic's method bodies (Pass-2).
    // Set by the driver to `astgen::instantiate_methods` so instantiate_struct_expr
    // (here) can drive body lowering that lives in the astgen crate-module.
    method_instantiator: Cell<Option<MethodInstantiator>>,
    // The `bindDeclTypes` early-instantiation pass (the C++ Phase 1b) instantiates
    // a generic's TYPE only and DEFERS its method-body lowering, matching the
    // oracle's intern order (the type monomorph interns early; method names intern
    // in the later method pre-pass). When `defer_method_lowering` is set,
    // instantiate_struct_expr stashes (clones, body_subst) here instead of calling
    // the hook; the method pre-pass drains + lowers them.
    defer_method_lowering: Cell<bool>,
    pending_method_lowering: RefCell<Vec<DeferredMethodWork>>,
}

/// Signature of the astgen hook that lowers an instantiated generic's methods
/// (`ctx`, the original anon methods, the instance name, the body subst, the
/// defining module). Set via [`CodegenContext::set_method_instantiator`].
pub type MethodInstantiator =
    fn(&CodegenContext, &[FunctionAST], &HashMap<String, TypeIdx>) -> Result<(), String>;

/// A loaded module's public surface, for resolving multi-dot chained access
/// (`w.leaf.Point` / `w.leaf.makePoint`). Mirrors the C++ `ModuleNamespace`.
#[derive(Default)]
pub struct ModuleNamespace {
    /// pub type name -> its qualified `Named` TypeIdx (`Point` -> `leaf.Point`).
    pub types: HashMap<String, TypeIdx>,
    /// pub re-export name -> the resolved module identity it points at
    /// (`pub const leaf = import("mod_chain_leaf")` -> `leaf` -> `mod_chain_leaf`).
    pub module_aliases: HashMap<String, String>,
    /// NON-pub decl names (types + fns), recorded so a handle-qualified access
    /// distinguishes "private" from "does not exist" (the C++ `privateNames`,
    /// main.cpp:702/713 regPriv).
    pub private_names: std::collections::HashSet<String>,
}

/// A module-scope const: its initializer expression + declared type. A plain
/// const re-lowers `init_expr` (with `declared_type` as the expected type) at
/// every reference; a `comp` const folds to a single value instead.
#[derive(Clone)]
pub struct ModuleConstInfo {
    pub init_expr: NodeIdx,
    pub declared_type: TypeIdx,
    pub is_comp: bool,
    /// The const's bare source name (`SIZE`), used to seed the comptime scope.
    pub bare_name: String,
    /// The owning module (empty for the entry module). Scopes visibility so two
    /// modules' same-named consts never conflate.
    pub module_path: String,
}

impl<'ctx> CodegenContext<'ctx> {
    /// Create a context over a driver-owned [`Context`].
    pub fn new(ctx: &'ctx Context, module_name: &str) -> CodegenContext<'ctx> {
        CodegenContext {
            ctx,
            module: ctx.create_module(module_name),
            builder: ctx.create_builder(),
            type_pool: TypePool::new(),
            string_pool: StringPool::new(),
            node_store: NodeStore::new(),
            llvm_type_cache: RefCell::new(Vec::new()),
            size_memo: RefCell::new(HashMap::new()),
            align_memo: RefCell::new(HashMap::new()),
            structs: RefCell::new(HashMap::new()),
            unions: RefCell::new(HashMap::new()),
            enums: RefCell::new(HashMap::new()),
            diagnostics: RefCell::new(Diagnostics::new()),
            current_file: RefCell::new(String::new()),
            body_module_stack: RefCell::new(Vec::new()),
            type_module_of: RefCell::new(HashMap::new()),
            function_asts: RefCell::new(HashMap::new()),
            withdrawn_methods: RefCell::new(HashMap::new()),
            drop_fns: RefCell::new(HashMap::new()),
            clone_fns: RefCell::new(HashMap::new()),
            requalify_map: RefCell::new(HashMap::new()),
            current_subst: RefCell::new(Vec::new()),
            current_comp_subst: RefCell::new(HashMap::new()),
            anon_structs: RefCell::new(HashMap::new()),
            anon_enums: RefCell::new(HashMap::new()),
            type_aliases: RefCell::new(HashMap::new()),
            import_handles: RefCell::new(HashMap::new()),
            module_namespaces: RefCell::new(HashMap::new()),
            method_instantiator: Cell::new(None),
            defer_method_lowering: Cell::new(false),
            pending_method_lowering: RefCell::new(Vec::new()),
            generic_resolutions: RefCell::new(HashMap::new()),
            module_consts: RefCell::new(HashMap::new()),
            array_expr_resolutions: RefCell::new(HashMap::new()),
        }
    }

    // ---- comptime fold + array-expr resolution ----
    /// Fold `expr` to a compile-time value, with all module consts seeded into
    /// scope (so an initializer / array length can reference a `const`). Returns
    /// `ComptimeValue::None` when it can't be folded.
    pub fn fold_comptime_expr(&self, expr: NodeIdx) -> ComptimeValue {
        self.fold_comptime_expr_in(expr, &ComptimeScope::new())
    }

    /// Like [`fold_comptime_expr`] but with `local`'s bindings (function-local
    /// `comp const`/`comp var`) overlaid on the module-const seed.
    pub fn fold_comptime_expr_in(&self, expr: NodeIdx, local: &ComptimeScope) -> ComptimeValue {
        let ev = ComptimeEvaluator::new(&self.node_store, &self.string_pool, &self.type_pool);
        let mut scope = ComptimeScope::new();
        // Install a cfn resolver so `bufLen(8)` in a `[N]u8` length (or any
        // comptime position) folds by running the cfn body — the C++
        // `CompCallResolverImpl`. Depth / total-call budgets shared (via Cell)
        // across the recursion bound runaway cfn->cfn chains.
        let depth = Cell::new(0u32);
        let total = Cell::new(0u32);
        let mut resolver = CfnResolver {
            ctx: self,
            depth: &depth,
            total: &total,
        };
        // Int arithmetic / array sizes are OS-independent; a default host is
        // fine (pointer-width @sizeOf goes through type_size, not the evaluator).
        let mut ctx = CompCtx {
            resolver: Some(&mut resolver),
            emitter: None,
            diags: None,
            loc: SrcLoc::new("", 0),
            host_os: Os::MacOs,
        };
        // Fixpoint-seed: a const may reference siblings, so re-pass until stable.
        // Only consts VISIBLE to the current body module seed the scope, bound
        // by their bare name — so a module's `SIZE` never reads a sibling
        // module's same-named const. (Dedup via the bare-name fixpoint below.)
        let cur = self.current_body_module();
        let consts: Vec<(String, NodeIdx)> = self
            .module_consts
            .borrow()
            .values()
            .filter(|v| v.module_path == cur)
            .map(|v| (v.bare_name.clone(), v.init_expr))
            .collect();
        let mut progress = true;
        while progress {
            progress = false;
            for (name, init) in &consts {
                if scope.lookup(name).is_some() {
                    continue;
                }
                let v = ev.eval(*init, &scope, &mut ctx);
                if !v.is_none() {
                    scope.bind(name.clone(), v);
                    progress = true;
                }
            }
        }
        // Local comp bindings shadow module consts.
        local.copy_bindings_into(&mut scope);
        ev.eval(expr, &scope, &mut ctx)
    }

    /// Evaluate a VALUE-returning `cfn` call at compile time (the C++
    /// `astgenCompTimeFnCall`'s fold path): fold each `arg_exprs[i]` in
    /// `caller_scope`, bind to the callee's param names, then run the body —
    /// with a `CfnResolver` active throughout so a cfn calling another cfn
    /// (`doubled()` -> `base()`) and recursion (`fact`/`fib` via `comp if`)
    /// resolve. Returns `Ok(value)` (the body's `return` value) or `Err` on an
    /// arg that isn't comptime-constant / a body that fails to fold. Value cfns
    /// carry no `@emit*` intrinsics, so no emitter is threaded here (the void
    /// `@emit`-bearing cfns keep their recording path in astgen).
    pub fn eval_cfn_call(
        &self,
        fn_ast: &FunctionAST,
        arg_exprs: &[NodeIdx],
        caller_scope: &ComptimeScope,
        line: u32,
    ) -> Result<ComptimeValue, String> {
        let ev = ComptimeEvaluator::new(&self.node_store, &self.string_pool, &self.type_pool);
        let depth = Cell::new(0u32);
        let total = Cell::new(0u32);
        let loc = SrcLoc::new("", line);

        // 1. Fold each arg through `fold_comptime_expr_in`, which seeds the
        //    current module's consts (so a `square(SEED + 1)` arg referencing a
        //    `comp const SEED` folds) AND installs a CfnResolver (so a nested
        //    cfn-call arg `add1(square(5))` folds). `caller_scope` overlays the
        //    function-local comp bindings on top.
        let mut outer = ComptimeScope::new();
        for (i, &arg_idx) in arg_exprs.iter().enumerate() {
            let v = self.fold_comptime_expr_in(arg_idx, caller_scope);
            if v.is_none() {
                return Err(format!(
                    "argument to cfn `{}` (param `{}`) must be a compile-time constant",
                    fn_ast.name, fn_ast.args[i].name
                ));
            }
            outer.bind(fn_ast.args[i].name.clone(), v);
        }

        // 2. Run the body with the resolver (nested cfn calls). The returned
        //    value is materialized by the caller, narrowed to the return type.
        let mut diags = Diagnostics::new();
        let mut resolver = CfnResolver {
            ctx: self,
            depth: &depth,
            total: &total,
        };
        let mut ctx = CompCtx {
            resolver: Some(&mut resolver),
            emitter: None,
            diags: Some(&mut diags),
            loc,
            host_os: Os::MacOs,
        };
        let mut iter = 0u32;
        let mut ret = ComptimeValue::None;
        let r = ev.exec_block(
            &fn_ast.body,
            &mut outer,
            &mut iter,
            DEFAULT_ITER_CAP,
            &mut ret,
            &mut ctx,
        );
        if matches!(r, ExecResult::Error | ExecResult::IterationCap) {
            return Err(format!(
                "cfn `{}` failed to evaluate at compile time",
                fn_ast.name
            ));
        }
        Ok(ret)
    }

    /// Resolve an `ArrayExpr` (`[lenExpr]T`) to a concrete `Array(T, n)`,
    /// instantiating on first use by folding the length expression. Caches the
    /// result; returns `ty` unchanged for non-`ArrayExpr` inputs, and errors
    /// when the length isn't a comptime non-negative integer.
    pub fn resolve_array_expr_instantiate(&self, ty: TypeIdx) -> Result<TypeIdx, String> {
        if let Some(&r) = self.array_expr_resolutions.borrow().get(&ty) {
            return Ok(r);
        }
        let k = self.type_pool.get(ty);
        if k.kind != TypeKind::ArrayExpr {
            return Ok(ty);
        }
        let elem = TypeIdx::new(k.a);
        let len_node = NodeIdx::new(k.b);
        let len = self.fold_comptime_expr(len_node);
        if !len.is_int() {
            return Err("astgen: array length must be comptime-known".into());
        }
        let n = len.as_u64();
        if n > u32::MAX as u64 {
            return Err("astgen: array size out of range".into());
        }
        let resolved = self.type_pool.intern_array(elem, n as u32);
        self.array_expr_resolutions
            .borrow_mut()
            .insert(ty, resolved);
        Ok(resolved)
    }

    /// The resolved `Array` for an `ArrayExpr` TypeIdx, or `NONE` (pure lookup).
    pub fn resolve_array_expr(&self, ty: TypeIdx) -> TypeIdx {
        self.array_expr_resolutions
            .borrow()
            .get(&ty)
            .copied()
            .unwrap_or(TypeIdx::NONE)
    }

    // ---- module consts ----
    /// Register a module-scope const under `name` (the bare or qualified key).
    pub fn register_module_const(&self, name: impl Into<String>, info: ModuleConstInfo) {
        self.module_consts.borrow_mut().insert(name.into(), info);
    }
    /// The const registered under `name`, if any (cloned).
    pub fn get_module_const(&self, name: &str) -> Option<ModuleConstInfo> {
        self.module_consts.borrow().get(name).cloned()
    }
    /// Is a const already registered under the (bare or qualified) `name`?
    pub fn has_module_const(&self, name: &str) -> bool {
        self.module_consts.borrow().contains_key(name)
    }

    // ---- generic instantiation ----
    /// Register a module's anonymous struct bodies (its generic factories'
    /// `return struct {...}`), keyed by the module's identity (empty = entry).
    pub fn register_anon_structs(
        &self,
        module_path: impl Into<String>,
        structs: Vec<StructDeclAST>,
    ) {
        self.anon_structs
            .borrow_mut()
            .insert(module_path.into(), structs);
    }
    /// Register a module's anonymous enum bodies (its generic factories'
    /// `return enum {...}`), keyed by the module's identity.
    pub fn register_anon_enums(&self, module_path: impl Into<String>, enums: Vec<EnumDeclAST>) {
        self.anon_enums
            .borrow_mut()
            .insert(module_path.into(), enums);
    }

    /// Resolve a `GenericCall` type (e.g. `Pair(u64)`) to its monomorphized
    /// `Named` struct, instantiating it on first use: register the struct with
    /// substituted fields + clone each method's signature under `Inst.method`
    /// (resolution-only — method bodies are NOT lowered here, matching the
    /// `--emit-jir` cut before `jirDefineBody`). Caches `GenericCall -> Named`.
    /// Returns `ty` unchanged for non-`GenericCall` inputs.
    /// Instantiate the generic underlying `ty` when it is a GenericCall or a
    /// type-alias chain that bottoms out in one (`CounterI32 -> Counter(i32)`),
    /// so its fields/layout resolve. `ty`'s own spelling is left unchanged.
    pub fn resolve_alias_generic_instantiate(&self, ty: TypeIdx) -> Result<(), String> {
        let mut probe = ty;
        for _ in 0..8 {
            let k = self.type_pool.get(probe);
            let (kind, a) = (k.kind, k.a);
            if kind == TypeKind::GenericCall {
                self.resolve_generic_call_instantiate(probe)?;
                return Ok(());
            }
            if kind == TypeKind::Named {
                let aliased = self.lookup_type_alias(&self.str_name(a));
                if !aliased.is_none() && aliased != probe {
                    probe = aliased;
                    continue;
                }
            }
            break;
        }
        Ok(())
    }

    pub fn resolve_generic_call_instantiate(&self, ty: TypeIdx) -> Result<TypeIdx, String> {
        // Inside an instantiated body, apply the active substitution to the call's
        // type args (`Option(T)` -> `Option(u8)`) so we instantiate the concrete
        // type, not a `__T` garbage instance. No-op when no frame is active.
        let ty = if self.has_active_subst() {
            let frame = self
                .current_subst
                .borrow()
                .last()
                .cloned()
                .unwrap_or_default();
            substitute_type(ty, &frame, &self.type_pool, &self.string_pool)
        } else {
            ty
        };
        let key = (self.current_body_module(), ty);
        let cached = self.generic_resolutions.borrow().get(&key).copied();
        if let Some(r) = cached {
            return Ok(r);
        }
        let (callee, args) = {
            let k = self.type_pool.get(ty);
            if k.kind != TypeKind::GenericCall {
                return Ok(ty);
            }
            (
                self.str_name(k.a),
                self.type_pool.generic_args_at(k.b).to_vec(),
            )
        };
        // Canonicalize a handle-qualified callee (`c.Vec` -> `std/collections.Vec`)
        // so it resolves via get_function_ast AND its monomorph name (`Vec__i32`)
        // dedups with the bare/canonical spelling — the C++ resolves these through
        // its scoped handle-fn table (main.cpp:1609-1660). A canonical dotted callee
        // (`std/collections.Vec`), whose prefix is a module path not a handle, is
        // left unchanged.
        let callee = match callee.split_once('.') {
            Some((prefix, rest)) => match self.import_handle_module(prefix) {
                Some(module) => format!("{module}.{rest}"),
                None => callee,
            },
            None => callee,
        };
        // Resolve the CURRENT module's generic first (two modules may define the
        // same-named generic, e.g. mod_gen_a.Pair vs mod_gen_b.Pair).
        let bm = self.current_body_module();
        // Requalify each type arg against the body module so a bare arg written
        // in a module's own body (`Option(File).Some(..)` inside std/fs) gets the
        // qualified monomorph name (`Option__std/fs.File`, not a duplicate bare
        // `Option__File`). Idempotent for builtins / already-qualified args. This
        // is the body-level half of the C++ bindDeclTypes requalification.
        let args: Vec<TypeIdx> = args.iter().map(|&a| self.requalify_type(a, &bm)).collect();
        let generic = if !callee.contains('.') && !bm.is_empty() {
            self.get_function_ast(&format!("{bm}.{callee}"))
                .or_else(|| self.get_function_ast(&callee))
        } else {
            self.get_function_ast(&callee)
        }
        .ok_or_else(|| format!("astgen: unknown generic `{callee}`"))?;
        if !generic.is_generic() {
            return Err(format!("astgen: `{callee}` is not a generic"));
        }
        // Mirror the C++ `requalifyType(GenericCall)` side effect: intern the
        // qualified callee name at lookup time (keeps later StringIdxs aligned).
        // (Generics with RECURSIVE generic method signatures — Vec.pop ->
        // Option(T) — are deferred inside instantiate_struct_expr until the full
        // Pass-1 intern order lands; non-recursive ones instantiate cleanly.)
        if !generic.module_path.is_empty() {
            let qcallee = format!("{}.{}", generic.module_path, generic.name);
            self.string_pool.intern(qcallee.as_bytes());
        }
        if args.len() != generic.args.len() {
            return Err(format!(
                "astgen: generic `{callee}` expects {} type arg(s), got {}",
                generic.args.len(),
                args.len()
            ));
        }
        let mut subst: HashMap<String, TypeIdx> = HashMap::new();
        for (i, p) in generic.args.iter().enumerate() {
            subst.insert(p.name.clone(), args[i]);
        }

        // Walk the body for the `return` (return-T forwarding or return-struct).
        for &stmt in &generic.body {
            let n = *self.node_store.get(stmt);
            if n.tag != AstTag::Return {
                continue;
            }
            let v = *self.node_store.get(NodeIdx::new(n.lhs));
            let resolved = match v.tag {
                AstTag::Variable => {
                    let name = self.str_name(v.lhs);
                    if let Some(&c) = subst.get(&name) {
                        c
                    } else {
                        let sid = self.string_pool.intern(name.as_bytes());
                        let as_named = self.type_pool.intern_named(sid);
                        substitute_type(as_named, &subst, &self.type_pool, &self.string_pool)
                    }
                }
                AstTag::StructExpr => {
                    // The instance name starts from the generic's OWNER-qualified
                    // identity (`std/collections.Vec__u8`), not the call spelling,
                    // so cross-module instantiations carry their defining module.
                    let qbase = if generic.module_path.is_empty() {
                        generic.name.clone()
                    } else {
                        format!("{}.{}", generic.module_path, generic.name)
                    };
                    self.instantiate_struct_expr(
                        v.lhs,
                        &qbase,
                        &args,
                        &subst,
                        &generic.module_path,
                    )?
                }
                AstTag::EnumExpr => {
                    let qbase = if generic.module_path.is_empty() {
                        generic.name.clone()
                    } else {
                        format!("{}.{}", generic.module_path, generic.name)
                    };
                    self.instantiate_enum_expr(v.lhs, &qbase, &args, &subst, &generic.module_path)?
                }
                _ => {
                    return Err(format!(
                        "astgen: generic `{callee}` return shape not yet ported"
                    ));
                }
            };
            self.generic_resolutions.borrow_mut().insert(key, resolved);
            return Ok(resolved);
        }
        Err(format!("astgen: generic `{callee}` has no `return`"))
    }

    /// Instantiate a `return struct {...}` anon body for concrete `args`:
    /// register the monomorphized struct (substituted + requalified fields,
    /// `set_body`) and clone each method's *signature* under `Inst.method`.
    fn instantiate_struct_expr(
        &self,
        anon_idx: u32,
        callee: &str,
        args: &[TypeIdx],
        subst: &HashMap<String, TypeIdx>,
        defining_module: &str,
    ) -> Result<TypeIdx, String> {
        let anon = self
            .anon_struct(defining_module, anon_idx)
            .ok_or_else(|| format!("astgen: missing anon struct for generic `{callee}`"))?;

        let mut inst_name = callee.to_string();
        for &a in args {
            inst_name.push_str("__");
            inst_name.push_str(&generic_arg_spelling(a, &self.type_pool, &self.string_pool));
        }
        // Memoize on the instantiated struct name.
        if self.is_struct_name_registered(&inst_name) {
            let sid = self.string_pool.intern(inst_name.as_bytes());
            return Ok(self.type_pool.intern_named(sid));
        }
        let sid = self.string_pool.intern(inst_name.as_bytes());
        let inst_named = self.type_pool.intern_named(sid);

        // bodySubst = params + the anon's synthetic name + `Self` -> the instance.
        let mut body_subst = subst.clone();
        body_subst.insert(anon.name.clone(), inst_named);
        body_subst.insert("Self".to_string(), inst_named);

        // Fields: substitute, then requalify against the DEFINING module.
        let mut inst_fields: Vec<(String, TypeIdx)> = Vec::with_capacity(anon.fields.len());
        for (fname, fty) in &anon.fields {
            let subbed = substitute_type(*fty, &body_subst, &self.type_pool, &self.string_pool);
            inst_fields.push((fname.clone(), self.requalify_type(subbed, defining_module)));
        }
        let named = self.context().named_struct(&inst_name);
        self.register_struct(inst_name.clone(), named, inst_fields.clone());

        // Fill the LLVM body in the defining module's scope (sibling field types
        // resolve there). Best-effort: a field that can't lower leaves the body
        // unset (the struct is still registered for field-index resolution).
        self.push_body_module(defining_module.to_string());
        let mut field_llvm = Vec::with_capacity(inst_fields.len());
        let mut ok = true;
        for (_, fty) in &inst_fields {
            match self.get_llvm_type(*fty) {
                Ok(t) => field_llvm.push(t),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            named.set_body(&field_llvm, false);
        }
        self.pop_body_module();

        // Pass-1: clone each method's SIGNATURE under `Inst.method` (substitute +
        // requalify params/return; a GenericCall sig like `Option(T)` -> the
        // qualified-and-instantiated `std/option.Option__u8` so the recursive
        // type is interned here, before any body). Method NAMES are NOT interned
        // here — they intern lazily at the Pass-2 emit_call site, matching the
        // oracle (jirDeclarePrototype does not touch the string pool).
        let mut clones: Vec<FunctionAST> = Vec::with_capacity(anon.methods.len());
        for m in &anon.methods {
            let mut clone = m.clone();
            clone.name = format!("{inst_name}.{}", m.name);
            clone.module_path = String::new();
            clone.parent_struct = String::new();
            for p in &mut clone.args {
                p.ty = self.instantiate_sig_type(p.ty, &body_subst, defining_module)?;
            }
            if !clone.return_type.is_none() {
                clone.return_type =
                    self.instantiate_sig_type(clone.return_type, &body_subst, defining_module)?;
            }
            let is_drop = m.name == "drop"
                && m.args.len() == 1
                && m.args[0].name == "self"
                && m.args[0].mode == jam_core::param_mode::ParamMode::Mut;
            if is_drop {
                self.register_drop_fn(inst_name.clone(), clone.name.clone());
            }
            self.register_function_ast(clone.name.clone(), clone.clone());
            clones.push(clone);
        }

        // Pass-2: lower each method body (interns call-site method names +
        // recursive instantiations) via the astgen hook, under the body subst.
        // The early bindDeclTypes pass DEFERS this (type-only instantiation) so the
        // method names intern later (the method pre-pass), matching the oracle.
        if self.defer_method_lowering.get() {
            let bm = self.current_body_module();
            self.pending_method_lowering
                .borrow_mut()
                .push((clones, body_subst, bm));
        } else if let Some(lower) = self.method_instantiator() {
            lower(self, &clones, &body_subst)?;
        }
        Ok(inst_named)
    }

    /// Substitute + requalify a method signature type, and if the result is a
    /// GenericCall (`Option(T)` -> `Option(u8)`), qualify its callee and
    /// instantiate it so the nested type is interned at Pass-1 (the C++ ABI
    /// consultation order).
    fn instantiate_sig_type(
        &self,
        ty: TypeIdx,
        body_subst: &HashMap<String, TypeIdx>,
        defining_module: &str,
    ) -> Result<TypeIdx, String> {
        let subbed = substitute_type(ty, body_subst, &self.type_pool, &self.string_pool);
        let rq = self.requalify_type(subbed, defining_module);
        if self.type_pool.get(rq).kind == TypeKind::GenericCall {
            let q = self.qualify_generic_callee(rq, defining_module);
            self.resolve_generic_call_instantiate(q)?;
            return Ok(q);
        }
        Ok(rq)
    }

    /// Instantiate a `return enum {...}` anon body for concrete `args`: register
    /// the monomorphized enum with substituted + requalified variant payloads.
    fn instantiate_enum_expr(
        &self,
        anon_idx: u32,
        callee: &str,
        args: &[TypeIdx],
        subst: &HashMap<String, TypeIdx>,
        defining_module: &str,
    ) -> Result<TypeIdx, String> {
        let anon = self
            .anon_enum(defining_module, anon_idx)
            .ok_or_else(|| format!("astgen: missing anon enum for generic `{callee}`"))?;

        let mut inst_name = callee.to_string();
        for &a in args {
            inst_name.push_str("__");
            inst_name.push_str(&generic_arg_spelling(a, &self.type_pool, &self.string_pool));
        }
        if self.enums.borrow().contains_key(&inst_name) {
            let sid = self.string_pool.intern(inst_name.as_bytes());
            return Ok(self.type_pool.intern_named(sid));
        }
        let sid = self.string_pool.intern(inst_name.as_bytes());
        let inst_named = self.type_pool.intern_named(sid);

        let mut body_subst = subst.clone();
        body_subst.insert(anon.name.clone(), inst_named);
        body_subst.insert("Self".to_string(), inst_named);

        let mut variants: Vec<EnumVariantInfo> = Vec::with_capacity(anon.variants.len());
        for v in &anon.variants {
            let mut payload_types: Vec<TypeIdx> = Vec::with_capacity(v.payload_types.len());
            for &ty in &v.payload_types {
                let subbed = substitute_type(ty, &body_subst, &self.type_pool, &self.string_pool);
                payload_types.push(self.requalify_type(subbed, defining_module));
            }
            variants.push(EnumVariantInfo {
                name: v.name.clone(),
                payload_types,
                discriminant: v.discriminant,
            });
        }
        // Payload layout (max over variants; each payload field aligned), the
        // analyzer's `fillEnumBodies` equivalent for an instantiated enum.
        let has_payload = variants.iter().any(|v| !v.payload_types.is_empty());
        let (mut max_size, mut max_align) = (0u64, 1u64);
        if has_payload {
            for v in &variants {
                let (mut off, mut var_align) = (0u64, 1u64);
                for &t in &v.payload_types {
                    let s = self.type_size(t)?;
                    let a = self.type_align(t)?;
                    off = off.div_ceil(a) * a + s;
                    var_align = var_align.max(a);
                }
                if var_align > 1 {
                    off = off.div_ceil(var_align) * var_align;
                }
                max_size = max_size.max(off);
                max_align = max_align.max(var_align);
            }
        }
        self.register_enum(inst_name.clone(), variants);
        if has_payload {
            let named = self.context().named_struct(&inst_name);
            // Fill the LLVM struct body `{i8 tag, alignDriver, [extra x i8]}` so the
            // instantiated enum is not left `opaque` in --emit-ir — the analyzer's
            // enum-body layout (analyzer.rs:423-446) for an instantiated enum.
            let (align_driver, driver_size) = match max_align {
                1 => (self.context().i8_type(), 1u64),
                2 => (self.context().i16_type(), 2),
                4 => (self.context().i32_type(), 4),
                8 => (self.context().i64_type(), 8),
                _ => {
                    return Err(format!(
                        "enum `{inst_name}` requires alignment > 8, which is not yet supported"
                    ));
                }
            };
            let padded = max_size.div_ceil(max_align) * max_align;
            let extra = padded.saturating_sub(driver_size);
            let i8 = self.context().i8_type();
            let mut body = vec![i8, align_driver];
            if extra > 0 {
                body.push(i8.array_type(extra));
            }
            named.set_body(&body, false);
            self.set_enum_llvm_type(&inst_name, named, max_size, max_align, true);
        }
        Ok(inst_named)
    }

    /// A clone of `module_path`'s anonymous enum at `idx`, if registered.
    fn anon_enum(&self, module_path: &str, idx: u32) -> Option<EnumDeclAST> {
        self.anon_enums
            .borrow()
            .get(module_path)
            .and_then(|v| v.get(idx as usize).cloned())
    }

    /// Qualify a `GenericCall` signature type to its defining/owning modules:
    /// the callee to its generic's module (`Holder`->`mod_x.Holder`) and each
    /// arg requalified against `ctx_module` (`Inner`->`mod_x.Inner`). Used for
    /// SIGNATURE types (the oracle qualifies return/param generic types; local-
    /// slot ones stay bare). No-op for non-`GenericCall` / already-qualified.
    pub fn qualify_generic_callee(&self, ty: TypeIdx, ctx_module: &str) -> TypeIdx {
        let k = self.type_pool.get(ty);
        if k.kind != TypeKind::GenericCall {
            return ty;
        }
        let callee = self.str_name(k.a);
        let args = self.type_pool.generic_args_at(k.b).to_vec();
        let new_callee = if let Some((prefix, rest)) = callee.split_once('.') {
            // A handle-qualified generic callee (`c.Vec`): resolve the import handle
            // to its canonical module path so `c.Vec(i32)` interns the SAME
            // GenericCall as bare `Vec(i32)` and dedups to one monomorph (the C++
            // resolves these through its scoped handle-fn table, main.cpp:1609-1660).
            // An already-canonical dotted callee (`std/collections.Vec`) — whose
            // prefix is a module path, not a handle — passes through unchanged.
            match self.import_handle_module(prefix) {
                Some(module) => format!("{module}.{rest}"),
                None => callee.clone(),
            }
        } else if !ctx_module.is_empty()
            && let Some(f) = self.get_function_ast(&format!("{ctx_module}.{callee}"))
        {
            // A module's OWN generic wins over a same-named one elsewhere (two
            // modules may each define `Pair`); resolve against ctx_module first.
            format!("{}.{}", f.module_path, f.name)
        } else if let Some(f) = self.get_function_ast(&callee) {
            if f.module_path.is_empty() {
                callee.clone()
            } else {
                format!("{}.{}", f.module_path, f.name)
            }
        } else {
            callee.clone()
        };
        // Requalify args with the active-subst-keeps-literal rule (matches the
        // C++ requalifyTypeUncached GenericCall arm): a generic param `T` arg
        // stays literal so an instantiated body interns the literal
        // `Vec(T)`/`Option(T)` GenericCalls in the oracle's order.
        let new_args: Vec<TypeIdx> = args
            .iter()
            .map(|&a| self.requalify_generic_arg(a, ctx_module))
            .collect();
        if new_callee == callee && new_args == args {
            return ty;
        }
        let sid = self.string_pool.intern(new_callee.as_bytes());
        self.type_pool.intern_generic_call(sid, new_args)
    }

    /// A clone of `module_path`'s anonymous struct at `idx`, if registered.
    fn anon_struct(&self, module_path: &str, idx: u32) -> Option<StructDeclAST> {
        self.anon_structs
            .borrow()
            .get(module_path)
            .and_then(|v| v.get(idx as usize).cloned())
    }

    /// The instantiated `Named` for a `GenericCall` TypeIdx, or `NONE`.
    pub fn lookup_generic_resolution(&self, ty: TypeIdx) -> TypeIdx {
        let key = (self.current_body_module(), ty);
        self.generic_resolutions
            .borrow()
            .get(&key)
            .copied()
            .unwrap_or(TypeIdx::NONE)
    }

    // ---- generic substitution stack ----
    /// Enter a generic instantiation: bind parameter names to concrete args
    /// while its body is lowered.
    pub fn push_subst(&self, frame: HashMap<String, TypeIdx>) {
        self.current_subst.borrow_mut().push(frame);
    }
    /// Leave the current generic instantiation.
    pub fn pop_subst(&self) {
        self.current_subst.borrow_mut().pop();
    }
    /// Apply the active generic substitution to `ty` (recursing compounds /
    /// GenericCall args): `Vec(T)` -> `Vec(u8)`, `*mut[] T` -> `*mut[] u8`,
    /// `Self`/`T` -> the instance/arg. No-op when no frame is active.
    pub fn apply_current_subst(&self, ty: TypeIdx) -> TypeIdx {
        if !self.has_active_subst() {
            return ty;
        }
        let frame = self
            .current_subst
            .borrow()
            .last()
            .cloned()
            .unwrap_or_default();
        substitute_type(ty, &frame, &self.type_pool, &self.string_pool)
    }

    /// Rewrite a `Self.method` / `T.method` callee inside an instantiated body to
    /// the concrete `Inst.method` (the C++ `resolvePrefix` reads currentSubst).
    /// No-op outside an instantiation or for a non-subst prefix.
    pub fn resolve_subst_prefix(&self, callee: &str) -> String {
        if !self.has_active_subst() {
            return callee.to_string();
        }
        if let Some((prefix, rest)) = callee.split_once('.') {
            let sub = self.lookup_current_subst(prefix);
            if !sub.is_none() {
                let k = self.type_pool.get(sub);
                if k.kind == TypeKind::Named {
                    return format!("{}.{rest}", self.str_name(k.a));
                }
            }
        }
        callee.to_string()
    }

    /// The concrete TypeIdx bound to generic parameter `name` in the innermost
    /// frame, or `NONE`.
    pub fn lookup_current_subst(&self, name: &str) -> TypeIdx {
        self.current_subst
            .borrow()
            .last()
            .and_then(|f| f.get(name).copied())
            .unwrap_or(TypeIdx::NONE)
    }
    /// Whether any substitution frame is active (memoization in the type-layout
    /// methods is bypassed while a substitution is in flight).
    pub fn has_active_subst(&self) -> bool {
        !self.current_subst.borrow().is_empty()
    }

    /// Install the comp-value substitution map active while a comp-instantiated
    /// fn clone's body is lowered (the C++ `setCurrentCompSubst`).
    pub fn set_current_comp_subst(&self, s: HashMap<String, ComptimeValue>) {
        *self.current_comp_subst.borrow_mut() = s;
    }
    /// Clear the comp-value substitution map (the C++ `clearCurrentCompSubst`).
    pub fn clear_current_comp_subst(&self) {
        self.current_comp_subst.borrow_mut().clear();
    }
    /// Bind the active comp-value substitutions into `scope`, so a body
    /// reference to a comp param folds to its call-site constant (the C++
    /// `seedComptimeScope` tail loop). Bound last to shadow same-named consts.
    pub fn seed_comp_subst_into(&self, scope: &mut ComptimeScope) {
        for (name, value) in self.current_comp_subst.borrow().iter() {
            scope.bind(name.clone(), value.clone());
        }
    }

    // ---- function registry ----
    /// Register a function/method signature under `name` (the call-site lookup
    /// key). Last registration wins, matching the C++ `registerFunctionAST`.
    pub fn register_function_ast(&self, name: impl Into<String>, f: FunctionAST) {
        self.function_asts.borrow_mut().insert(name.into(), f);
    }
    /// Withdraw a conditionally-instantiated method whose body failed to compile
    /// for these type args (the C++ `withdraw`).
    pub fn unregister_function_ast(&self, name: &str) {
        self.function_asts.borrow_mut().remove(name);
    }
    /// Record why a conditionally-instantiated method was withdrawn (the C++
    /// `recordWithdrawnMethod`): a call to a withdrawn method reports "not
    /// available for this instantiation — <reason>" instead of a bare
    /// "unknown method".
    pub fn record_withdrawn_method(&self, name: impl Into<String>, reason: impl Into<String>) {
        self.withdrawn_methods
            .borrow_mut()
            .insert(name.into(), reason.into());
    }
    /// The recorded withdrawal reason for `name`, if it was withdrawn (the C++
    /// `getWithdrawnMethod`).
    pub fn get_withdrawn_method(&self, name: &str) -> Option<String> {
        self.withdrawn_methods.borrow().get(name).cloned()
    }
    /// Install the astgen hook that lowers instantiated generic method bodies.
    pub fn set_method_instantiator(&self, f: MethodInstantiator) {
        self.method_instantiator.set(Some(f));
    }
    fn method_instantiator(&self) -> Option<MethodInstantiator> {
        self.method_instantiator.get()
    }
    /// While set, `instantiate_struct_expr` instantiates a generic's TYPE only and
    /// stashes its method-body lowering (the C++ Phase-1b `bindDeclTypes`).
    pub fn set_defer_method_lowering(&self, on: bool) {
        self.defer_method_lowering.set(on);
    }
    /// Lower every method body stashed by the deferred (type-only) instantiation,
    /// in stash order (the method pre-pass, the C++ Phase-1k). Drains the queue
    /// with deferral OFF so nested generics encountered in a body instantiate
    /// immediately (the C++ resolveGenericCall during method codegen).
    pub fn drain_pending_method_lowering(&self) -> Result<(), String> {
        let Some(lower) = self.method_instantiator() else {
            return Ok(());
        };
        let was = self.defer_method_lowering.replace(false);
        let pending = std::mem::take(&mut *self.pending_method_lowering.borrow_mut());
        let mut result = Ok(());
        for (clones, body_subst, bm) in pending {
            self.push_body_module(bm);
            let r = lower(self, &clones, &body_subst);
            self.pop_body_module();
            if let Err(e) = r {
                result = Err(e);
                break;
            }
        }
        self.defer_method_lowering.set(was);
        result
    }
    /// Look up a registered function by call-site name, returning a clone (the
    /// body is `NodeIdx` indices, so this stays cheap).
    pub fn get_function_ast(&self, name: &str) -> Option<FunctionAST> {
        self.function_asts.borrow().get(name).cloned()
    }

    // ---- drop registry ----
    /// Register the (already-mangled) LLVM symbol of `T`'s `cfn drop`, keyed by
    /// the type name the drop site resolves through. The C++ stores a
    /// `FunctionAST*` and mangles on lookup; we pre-mangle so the registry holds
    /// owned `String`s (no module-AST borrow entangling the context lifetime).
    pub fn register_drop_fn(
        &self,
        type_name: impl Into<String>,
        mangled_symbol: impl Into<String>,
    ) {
        self.drop_fns
            .borrow_mut()
            .insert(type_name.into(), mangled_symbol.into());
    }

    /// The mangled drop-fn symbol for the type `ty` names (Struct/Named/Enum),
    /// or `None` when it has no registered drop. Resolves through `name_for_kinds`
    /// so an instantiated generic / type-alias (`CounterI32` -> `Counter__i32`)
    /// finds its drop — the C++ `lookupDropFnLLVMName` chase.
    pub fn lookup_drop_fn_name(&self, ty: TypeIdx) -> Option<String> {
        let name = self.name_for_kinds(ty, &[TypeKind::Struct, TypeKind::Named, TypeKind::Enum])?;
        self.drop_fns.borrow().get(&name).cloned()
    }

    /// Register a top-level `cfn clone(self: T) T` under its receiver struct name.
    pub fn register_clone_fn(&self, type_name: impl Into<String>, clone_fn: FunctionAST) {
        self.clone_fns
            .borrow_mut()
            .insert(type_name.into(), clone_fn);
    }

    /// The top-level `cfn clone` for the struct named `name`, if registered.
    pub fn lookup_clone_fn(&self, name: &str) -> Option<FunctionAST> {
        self.clone_fns.borrow().get(name).cloned()
    }

    /// Whether `ty` owns heap that must be released at scope exit: it has a
    /// registered `cfn drop`, OR it is an array/struct whose element/fields
    /// recursively need drop. (Payloaded-enum and generic resolution deferred;
    /// no memoization yet — correctness first.)
    pub fn type_needs_drop(&self, ty: TypeIdx) -> bool {
        if self.lookup_drop_fn_name(ty).is_some() {
            return true;
        }
        let k = self.type_pool.get(ty);
        match k.kind {
            TypeKind::Array => self.type_needs_drop(TypeIdx::new(k.a)),
            TypeKind::Struct | TypeKind::Named => {
                if let Some(fields) = self.struct_fields(ty) {
                    return fields.iter().any(|(_, fty)| self.type_needs_drop(*fty));
                }
                // An enum carries drop if any variant payload does (e.g.
                // `Item.Many(Vec(u64))` — the Vec owns its buffer).
                if let Some(name) = self.enum_name_of(ty)
                    && let Some(vs) = self.enum_variants_by_name(&name)
                {
                    return vs
                        .iter()
                        .any(|v| v.payload_types.iter().any(|pt| self.type_needs_drop(*pt)));
                }
                false
            }
            // A `Vec(u64)` payload/field resolves to its `Vec__u64` monomorph,
            // which carries the `cfn drop`.
            TypeKind::GenericCall => {
                let r = self.resolve_generic_call(ty);
                !r.is_none() && r != ty && self.type_needs_drop(r)
            }
            _ => false,
        }
    }

    // ---- diagnostics ----
    /// Push an error diagnostic (the analyzer reports through `&self`).
    pub fn push_error(&self, loc: SrcLoc, message: impl Into<String>) {
        self.diagnostics.borrow_mut().error(loc, message);
    }
    pub fn push_error_with_trace(
        &self,
        loc: SrcLoc,
        message: impl Into<String>,
        trace: Vec<Trace>,
    ) {
        self.diagnostics
            .borrow_mut()
            .error_with_trace(loc, message, trace);
    }
    pub fn has_errors(&self) -> bool {
        self.diagnostics.borrow().has_errors()
    }
    pub fn error_count(&self) -> usize {
        self.diagnostics.borrow().error_count()
    }
    /// Borrow the accumulated diagnostics (e.g. to render at the end of a run).
    pub fn diagnostics(&self) -> Ref<'_, Diagnostics> {
        self.diagnostics.borrow()
    }

    // ---- current display file (for astgen diagnostics) ----
    /// Set the display filename stamped onto astgen diagnostics (the C++
    /// `setCurrentFile`). Entry-global; swapped for imported-body lowering.
    pub fn set_current_file(&self, file: impl Into<String>) {
        *self.current_file.borrow_mut() = file.into();
    }
    /// The display filename for the body currently lowering (the C++
    /// `currentFile`). Empty until the driver sets it.
    pub fn current_file(&self) -> String {
        self.current_file.borrow().clone()
    }

    // ---- body-module stack ----
    /// Enter a type body's owning module (the C++ `BodyModuleGuard` ctor).
    pub fn push_body_module(&self, module_path: impl Into<String>) {
        self.body_module_stack.borrow_mut().push(module_path.into());
    }
    /// Leave the current body module (the guard's dtor).
    pub fn pop_body_module(&self) {
        self.body_module_stack.borrow_mut().pop();
    }
    /// The module whose body is currently being resolved (empty = entry module).
    pub fn current_body_module(&self) -> String {
        self.body_module_stack
            .borrow()
            .last()
            .cloned()
            .unwrap_or_default()
    }

    // ---- type-owner registry + requalify ----
    /// Record that within `ctx_module` the bare type `type_name` is owned by
    /// `owner_module` (declared there or imported). Built once up front.
    pub fn register_type_owner(&self, ctx_module: &str, type_name: &str, owner_module: &str) {
        self.type_module_of
            .borrow_mut()
            .entry(ctx_module.to_string())
            .or_default()
            .insert(type_name.to_string(), owner_module.to_string());
        // Ownership feeds the resolution memos; clear them (registration is an
        // up-front pass, so this is effectively free).
        self.size_memo.borrow_mut().clear();
        self.align_memo.borrow_mut().clear();
    }

    /// The owner module recorded for `(ctx_module, type_name)`, if any.
    pub fn type_owner(&self, ctx_module: &str, type_name: &str) -> Option<String> {
        self.type_module_of
            .borrow()
            .get(ctx_module)
            .and_then(|m| m.get(type_name))
            .cloned()
    }

    /// Record that within `ctx_module` the bare user type `bare_name` resolves
    /// to the (already-interned) qualified `qualified_ty`. Populated by the
    /// driver's register pass — interning happens there (with `&mut` pool
    /// access), so [`requalify_type`] stays a pure `&self` lookup.
    pub fn register_requalify(&self, ctx_module: &str, bare_name: &str, qualified_ty: TypeIdx) {
        self.requalify_map.borrow_mut().insert(
            (ctx_module.to_string(), bare_name.to_string()),
            qualified_ty,
        );
    }

    /// Rewrite a bare user-type `Named` leaf to its module-qualified TypeIdx as
    /// seen from `ctx_module` (a pure lookup into the pre-interned map; no-op for
    /// the entry module, already-qualified names, and non-`Named` types).
    /// Recursive requalification through ptr/array element types is deferred.
    pub fn requalify_type(&self, ty: TypeIdx, ctx_module: &str) -> TypeIdx {
        let k = self.type_pool.get(ty);
        // A GenericCall requalifies its callee + args (`Option(File)` ->
        // `std/option.Option(std/fs.File)`) so the monomorph name matches the
        // oracle's qualified spelling.
        if k.kind == TypeKind::GenericCall {
            return self.qualify_generic_callee(ty, ctx_module);
        }
        // Wrapper kinds carry the element TypeIdx in `a`; recurse into it so a
        // qualified element (`[3]timer.TimerChannel`, `*mut bus.Bus`) surfaces
        // through the wrapper. `b` (length / length-expr) is preserved. Mirrors
        // the C++ `requalifyTypeUncached` wrapper arm (src/codegen.cpp:430-442);
        // without this an array/pointer/slice field of an imported struct keeps
        // a bare element name that can't requalify under a foreign body module.
        match k.kind {
            TypeKind::PtrSingle
            | TypeKind::PtrMany
            | TypeKind::Slice
            | TypeKind::Array
            | TypeKind::ArrayExpr => {
                let elem = self.requalify_type(TypeIdx::new(k.a), ctx_module);
                if elem == TypeIdx::new(k.a) {
                    return ty;
                }
                let mut nk = k;
                nk.a = elem.raw();
                return self.type_pool.intern(nk);
            }
            _ => {}
        }
        if k.kind != TypeKind::Named {
            return ty;
        }
        let name = self.str_name(k.a);
        // Active generic substitution (a `Self`/`T` Named inside an instantiated
        // method body/signature) -> the concrete instance/type arg. No-op when no
        // frame is active (the common case).
        if self.has_active_subst() {
            let sub = self.lookup_current_subst(&name);
            if !sub.is_none() {
                return sub;
            }
        }
        if name.contains('.') {
            return ty;
        }
        // A module's OWN type wins over a same-named global alias (two modules
        // may each define `Pair`): try the body-module requalify map first.
        if !ctx_module.is_empty()
            && let Some(&q) = self
                .requalify_map
                .borrow()
                .get(&(ctx_module.to_string(), name.clone()))
        {
            return q;
        }
        // Then a destructuring-import alias (`Token` -> `m.Token`), which is a
        // qualified `Named`: resolve the SPELLING. A type-alias-const whose
        // target is a GenericCall (`CounterI32` -> `Counter(i32)`) keeps its own
        // spelling here (the oracle dumps `*CounterI32`); name_for_kinds chases
        // it only at lookup time.
        if let Some(&aliased) = self.type_aliases.borrow().get(&name)
            && self.type_pool.get(aliased).kind == TypeKind::Named
        {
            return aliased;
        }
        ty
    }

    /// Requalify a GenericCall's *argument* the way the C++ `requalifyType`
    /// does when called from `requalifyTypeUncached`'s GenericCall arm: an
    /// active-subst `Named` (the generic param `T`/`Self`) stays LITERAL rather
    /// than being substituted. `requalify_type` substitutes such a Named (it is
    /// the only requalify caller that does so); for args we must keep `T`
    /// un-substituted so a literal `Vec(T)` / `Option(T)` GenericCall interns
    /// during an instantiated body — matching the oracle's TypePool order.
    /// Recurses wrapper kinds (`*Vec(T)`) and nested GenericCalls.
    fn requalify_generic_arg(&self, ty: TypeIdx, ctx_module: &str) -> TypeIdx {
        let k = self.type_pool.get(ty);
        match k.kind {
            TypeKind::Named => {
                let name = self.str_name(k.a);
                // Active-subst param stays literal (the C++ rule); fall through
                // to the ordinary requalify otherwise.
                if self.has_active_subst() && !self.lookup_current_subst(&name).is_none() {
                    return ty;
                }
                self.requalify_type(ty, ctx_module)
            }
            TypeKind::PtrSingle | TypeKind::PtrMany | TypeKind::Slice | TypeKind::Array => {
                let elem = self.requalify_generic_arg(TypeIdx::new(k.a), ctx_module);
                if elem == TypeIdx::new(k.a) {
                    return ty;
                }
                let mut nk = k;
                nk.a = elem.raw();
                self.type_pool.intern(nk)
            }
            TypeKind::GenericCall => {
                let callee = self.str_name(k.a);
                let args = self.type_pool.generic_args_at(k.b).to_vec();
                let new_callee = if callee.contains('.') {
                    callee.clone()
                } else if !ctx_module.is_empty()
                    && let Some(f) = self.get_function_ast(&format!("{ctx_module}.{callee}"))
                {
                    format!("{}.{}", f.module_path, f.name)
                } else if let Some(f) = self.get_function_ast(&callee) {
                    if f.module_path.is_empty() {
                        callee.clone()
                    } else {
                        format!("{}.{}", f.module_path, f.name)
                    }
                } else {
                    callee.clone()
                };
                let new_args: Vec<TypeIdx> = args
                    .iter()
                    .map(|&a| self.requalify_generic_arg(a, ctx_module))
                    .collect();
                if new_callee == callee && new_args == args {
                    return ty;
                }
                let sid = self.string_pool.intern(new_callee.as_bytes());
                self.type_pool.intern_generic_call(sid, new_args)
            }
            _ => ty,
        }
    }

    // ---- accessors ----
    pub fn context(&self) -> &'ctx Context {
        self.ctx
    }
    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }
    pub fn builder(&self) -> &jam_llvm::Builder<'ctx> {
        &self.builder
    }

    /// If `ty` is a `Struct`/`Named` (or the given kinds), its registry name.
    fn name_for_kinds(&self, ty: TypeIdx, kinds: &[TypeKind]) -> Option<String> {
        // Requalify a bare imported-module type to its qualified registry name
        // (no-op for the entry module / already-qualified). This is the single
        // chokepoint every struct/union/enum registry lookup flows through, so a
        // module body's bare reference to its own type resolves correctly.
        let mut ty = self.requalify_type(ty, &self.current_body_module());
        // Chase, bounded: a GenericCall (`Pair(u64)`) resolves to its
        // instantiated `Named` struct; a type-alias Named (`CounterI32` ->
        // `Counter(i32)`) resolves to its target (which may itself be a
        // GenericCall, hence the loop) so struct/enum field lookups land on the
        // monomorphized type. Mirrors the C++ lookupStruct recursive chaser.
        for _ in 0..8 {
            let k = self.type_pool.get(ty);
            if k.kind == TypeKind::GenericCall {
                // On-demand: instantiate the GenericCall if not cached, so a
                // qualified callee (`std/option.Option(i32)`) resolves to its
                // monomorph even when only the bare form was cached earlier.
                let r = self.resolve_generic_call(ty);
                if r.is_none() || r == ty {
                    break;
                }
                ty = r;
                continue;
            }
            if k.kind == TypeKind::Named {
                let name = self.str_name(k.a);
                // Multi-dot chained type (`w.leaf.Point`): resolve through the
                // module-namespace re-export chain to the leaf's canonical Named.
                if name.matches('.').count() >= 2 {
                    let chained = self.resolve_chained_type(&name);
                    if !chained.is_none() && chained != ty {
                        ty = chained;
                        continue;
                    }
                }
                let aliased = self.lookup_type_alias(&name);
                if !aliased.is_none() && aliased != ty {
                    ty = self.requalify_type(aliased, &self.current_body_module());
                    continue;
                }
            }
            break;
        }
        let k = self.type_pool.get(ty);
        if kinds.contains(&k.kind) {
            Some(self.str_name(k.a))
        } else {
            None
        }
    }

    // ---- struct registry ----
    pub fn register_struct(
        &self,
        name: impl Into<String>,
        ty: Type<'ctx>,
        fields: Vec<(String, TypeIdx)>,
    ) {
        let name = name.into();
        self.structs
            .borrow_mut()
            .insert(name.clone(), StructInfo { name, ty, fields });
    }
    /// LLVM type of the struct a `Struct`/`Named` TypeIdx resolves to.
    pub fn struct_llvm_type(&self, ty: TypeIdx) -> Option<Type<'ctx>> {
        let name = self.name_for_kinds(ty, &[TypeKind::Struct, TypeKind::Named])?;
        self.structs.borrow().get(&name).map(|s| s.ty)
    }
    /// (name, type) fields of the struct a TypeIdx resolves to.
    pub fn struct_fields(&self, ty: TypeIdx) -> Option<Vec<(String, TypeIdx)>> {
        let name = self.name_for_kinds(ty, &[TypeKind::Struct, TypeKind::Named])?;
        self.structs.borrow().get(&name).map(|s| s.fields.clone())
    }
    /// Is the struct a `Struct`/`Named` TypeIdx names registered?
    pub fn is_struct_registered(&self, ty: TypeIdx) -> bool {
        self.name_for_kinds(ty, &[TypeKind::Struct, TypeKind::Named])
            .is_some_and(|n| self.structs.borrow().contains_key(&n))
    }
    /// Index of `field_name` within `struct_name`, or -1 if absent.
    pub fn field_index(&self, struct_name: &str, field_name: &str) -> i32 {
        match self.structs.borrow().get(struct_name) {
            Some(s) => s
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .map(|i| i as i32)
                .unwrap_or(-1),
            None => -1,
        }
    }

    // ---- union registry ----
    pub fn register_union(
        &self,
        name: impl Into<String>,
        ty: Type<'ctx>,
        fields: Vec<(String, TypeIdx)>,
    ) {
        let name = name.into();
        self.unions
            .borrow_mut()
            .insert(name.clone(), UnionInfo { name, ty, fields });
    }
    pub fn union_llvm_type(&self, ty: TypeIdx) -> Option<Type<'ctx>> {
        let name = self.name_for_kinds(ty, &[TypeKind::Union, TypeKind::Named])?;
        self.unions.borrow().get(&name).map(|u| u.ty)
    }
    pub fn union_fields(&self, ty: TypeIdx) -> Option<Vec<(String, TypeIdx)>> {
        let name = self.name_for_kinds(ty, &[TypeKind::Union, TypeKind::Named])?;
        self.unions.borrow().get(&name).map(|u| u.fields.clone())
    }
    pub fn is_union_registered(&self, ty: TypeIdx) -> bool {
        self.name_for_kinds(ty, &[TypeKind::Union, TypeKind::Named])
            .is_some_and(|n| self.unions.borrow().contains_key(&n))
    }
    pub fn union_field_type(&self, union_name: &str, field_name: &str) -> TypeIdx {
        match self.unions.borrow().get(union_name) {
            Some(u) => u
                .fields
                .iter()
                .find(|(n, _)| n == field_name)
                .map(|(_, t)| *t)
                .unwrap_or(TypeIdx::NONE),
            None => TypeIdx::NONE,
        }
    }

    // ---- enum registry ----
    pub fn register_enum(&self, name: impl Into<String>, variants: Vec<EnumVariantInfo>) {
        let name = name.into();
        // An enum is payloaded if any variant carries payload types; this gates
        // the by-ref classification and the LLVM-body fill (analyzer). The
        // analyzer recomputes the layout but reads this flag to decide whether
        // a body is needed at all.
        let has_payload_variant = variants.iter().any(|v| !v.payload_types.is_empty());
        self.enums.borrow_mut().insert(
            name.clone(),
            EnumInfo {
                name,
                ty: None,
                variants,
                has_payload_variant,
                max_payload_size: 0,
                max_payload_align: 1,
            },
        );
    }
    /// Fill an enum's lowered LLVM type + payload layout (the analyzer computes
    /// these once the field types are known).
    pub fn set_enum_llvm_type(
        &self,
        name: &str,
        llvm_type: Type<'ctx>,
        max_payload_size: u64,
        max_payload_align: u64,
        has_payload_variant: bool,
    ) {
        if let Some(e) = self.enums.borrow_mut().get_mut(name) {
            e.ty = Some(llvm_type);
            e.max_payload_size = max_payload_size;
            e.max_payload_align = max_payload_align;
            e.has_payload_variant = has_payload_variant;
        }
    }
    /// Whether the enum a `Enum`/`Named` TypeIdx names carries any payload
    /// variant (`None` if the enum isn't registered).
    pub fn enum_has_payload(&self, ty: TypeIdx) -> Option<bool> {
        let name = self.name_for_kinds(ty, &[TypeKind::Enum, TypeKind::Named])?;
        self.enums
            .borrow()
            .get(&name)
            .map(|e| e.has_payload_variant)
    }
    /// The enum's lowered LLVM type (only set for payloaded enums, post-fill).
    pub fn enum_llvm_type(&self, ty: TypeIdx) -> Option<Type<'ctx>> {
        let name = self.name_for_kinds(ty, &[TypeKind::Enum, TypeKind::Named])?;
        self.enums.borrow().get(&name).and_then(|e| e.ty)
    }
    /// `(max_payload_size, max_payload_align)` for the enum, or `None`.
    pub fn enum_payload_layout(&self, ty: TypeIdx) -> Option<(u64, u64)> {
        let name = self.name_for_kinds(ty, &[TypeKind::Enum, TypeKind::Named])?;
        self.enums
            .borrow()
            .get(&name)
            .map(|e| (e.max_payload_size, e.max_payload_align))
    }
    /// The registered enum name a `Enum`/`Named` TypeIdx resolves to, or `None`
    /// when it isn't a registered enum. (The `lookupEnum(ty)->name` of the C++.)
    pub fn enum_name_of(&self, ty: TypeIdx) -> Option<String> {
        let name = self.name_for_kinds(ty, &[TypeKind::Enum, TypeKind::Named])?;
        if self.enums.borrow().contains_key(&name) {
            Some(name)
        } else {
            None
        }
    }
    /// The registered struct name a `Struct`/`Named`/instantiated-`GenericCall`
    /// TypeIdx resolves to, or `None` (the `lookupStruct(ty)->name` of the C++).
    pub fn struct_name_of(&self, ty: TypeIdx) -> Option<String> {
        let name = self.name_for_kinds(ty, &[TypeKind::Struct, TypeKind::Named])?;
        if self.structs.borrow().contains_key(&name) {
            Some(name)
        } else {
            None
        }
    }
    /// Index of `variant_name` within `enum_name`, or -1.
    pub fn enum_variant_index(&self, enum_name: &str, variant_name: &str) -> i32 {
        match self.enums.borrow().get(enum_name) {
            Some(e) => e
                .variants
                .iter()
                .position(|v| v.name == variant_name)
                .map(|i| i as i32)
                .unwrap_or(-1),
            None => -1,
        }
    }

    // ---- by-name registry accessors (for the analyzer's body fill) ----
    pub fn struct_named_type(&self, name: &str) -> Option<Type<'ctx>> {
        self.structs.borrow().get(name).map(|s| s.ty)
    }
    pub fn is_struct_name_registered(&self, name: &str) -> bool {
        self.structs.borrow().contains_key(name)
    }
    pub fn union_named_type(&self, name: &str) -> Option<Type<'ctx>> {
        self.unions.borrow().get(name).map(|u| u.ty)
    }
    pub fn union_fields_by_name(&self, name: &str) -> Option<Vec<(String, TypeIdx)>> {
        self.unions.borrow().get(name).map(|u| u.fields.clone())
    }
    pub fn is_union_name_registered(&self, name: &str) -> bool {
        self.unions.borrow().contains_key(name)
    }
    pub fn enum_named_type(&self, name: &str) -> Option<Type<'ctx>> {
        self.enums.borrow().get(name).and_then(|e| e.ty)
    }
    pub fn enum_variants_by_name(&self, name: &str) -> Option<Vec<EnumVariantInfo>> {
        self.enums.borrow().get(name).map(|e| e.variants.clone())
    }
    pub fn enum_has_payload_by_name(&self, name: &str) -> Option<bool> {
        self.enums.borrow().get(name).map(|e| e.has_payload_variant)
    }
    pub fn is_enum_name_registered(&self, name: &str) -> bool {
        self.enums.borrow().contains_key(name)
    }

    fn str_name(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.string_pool.get(StringIdx::new(id))).into_owned()
    }

    // ---- deferred resolution hooks ----
    // These return `NONE` until the analyzer + substitution engine land. They
    // exist now so `abi`'s classifiers can be ported against a stable surface;
    // for non-generic, non-aliased types (the common case) they are never the
    // deciding path.

    /// Resolve a `GenericCall` TypeIdx to its instantiated concrete TypeIdx.
    /// DEFERRED — returns `NONE`.
    pub fn resolve_generic_call(&self, ty: TypeIdx) -> TypeIdx {
        let cached = self.lookup_generic_resolution(ty);
        if !cached.is_none() {
            return cached;
        }
        // ON-DEMAND INSTANTIATION (the C++ `const resolveGenericCall`): the pools +
        // registries are interior-mutable, so a `&self` layout query instantiates
        // an uninstantiated GenericCall on the spot.
        self.resolve_generic_call_instantiate(ty)
            .unwrap_or(TypeIdx::NONE)
    }
    /// Register a type alias (`const Foo = Bar` / destructuring import name).
    pub fn register_type_alias(&self, name: impl Into<String>, target: TypeIdx) {
        self.type_aliases.borrow_mut().insert(name.into(), target);
    }
    /// Register an import handle (`const lib = import("m")`) -> module identity.
    pub fn register_import_handle(&self, name: impl Into<String>, module: impl Into<String>) {
        self.import_handles
            .borrow_mut()
            .insert(name.into(), module.into());
    }
    /// The module identity an import-handle name resolves to, if any.
    pub fn import_handle_module(&self, name: &str) -> Option<String> {
        self.import_handles.borrow().get(name).cloned()
    }
    /// Format a qualified-name lookup miss the way the C++
    /// `JamCodegenContext::formatNamespaceLookupError` does (codegen.cpp:819).
    /// `qualified` is the dotted spelling that failed to resolve (`fmt.println`),
    /// `kind` is what we were looking for (`function`, `user-defined type`, ...).
    /// Splits at the FIRST dot: the leading segment is an import handle; if it
    /// resolves to a module, the trailing segment didn't exist there. The C++
    /// also distinguishes a PRIVATE trailing symbol ("is not exported"), but the
    /// Rust import-handle table carries only `handle -> modulePath` (no private-
    /// name set), and no corpus path exercises that branch, so it is omitted.
    pub fn format_namespace_lookup_error(&self, kind: &str, qualified: &str) -> String {
        let Some(dot_pos) = qualified.find('.') else {
            return format!("Unknown {kind}: {qualified}");
        };
        let handle = &qualified[..dot_pos];
        let bare = &qualified[dot_pos + 1..];
        match self.import_handle_module(handle) {
            None => format!("unknown module handle `{handle}` in `{qualified}`"),
            Some(module_path) => {
                // Distinguish private from nonexistent (the C++ privateNames
                // branch, codegen.cpp:831).
                let is_private = self
                    .module_namespaces
                    .borrow()
                    .get(&module_path)
                    .is_some_and(|ns| ns.private_names.contains(bare));
                if is_private {
                    format!("symbol `{bare}` is not exported from module `{module_path}`")
                } else {
                    format!("symbol `{bare}` does not exist in module `{module_path}`")
                }
            }
        }
    }

    /// Privacy gate for a dotted spelling `handle.Name` evaluated from OUTSIDE
    /// the handle's target module. The registries key types/functions by their
    /// qualified identity for internal resolution, and a handle spelled like
    /// the module identity (`const lib = import("lib")`) makes a user spelling
    /// collide with that internal key — so the pub check must happen at lookup
    /// (the C++ never registers handle-qualified keys for non-pub decls,
    /// main.cpp:701-721). Returns the diagnostic when `Name` is private.
    pub fn namespace_privacy_violation(&self, dotted: &str) -> Option<String> {
        let (handle, bare) = dotted.split_once('.')?;
        let target = self.import_handle_module(handle)?;
        // The defining module's own bodies keep using qualified identities.
        if self.current_body_module() == target {
            return None;
        }
        let violates = self
            .module_namespaces
            .borrow()
            .get(&target)
            .is_some_and(|ns| ns.private_names.contains(bare));
        if violates {
            Some(format!(
                "symbol `{bare}` is not exported from module `{target}`"
            ))
        } else {
            None
        }
    }
    /// Register a loaded module's public namespace (keyed by its identity).
    pub fn register_module_namespace(&self, module: impl Into<String>, ns: ModuleNamespace) {
        self.module_namespaces
            .borrow_mut()
            .insert(module.into(), ns);
    }

    /// Walk a 3+-segment dotted access (`w.leaf.Point`): the first segment is an
    /// import handle, each intermediate segment a pub re-export module alias.
    /// Returns `(leaf module identity, last segment)`. The C++ `walkChain`.
    fn walk_chain(&self, dotted: &str) -> Option<(String, String)> {
        let segs: Vec<&str> = dotted.split('.').collect();
        if segs.len() < 3 {
            return None;
        }
        let mut cur = self.import_handle_module(segs[0])?;
        for seg in &segs[1..segs.len() - 1] {
            let next = self
                .module_namespaces
                .borrow()
                .get(&cur)
                .and_then(|ns| ns.module_aliases.get(*seg).cloned())?;
            cur = next;
        }
        Some((cur, segs[segs.len() - 1].to_string()))
    }

    /// Resolve a chained TYPE access (`w.leaf.Point`) to the leaf module's
    /// canonical `Named` TypeIdx, or NONE. The C++ `resolveChainedType`.
    pub fn resolve_chained_type(&self, dotted: &str) -> TypeIdx {
        let Some((leaf, last)) = self.walk_chain(dotted) else {
            return TypeIdx::NONE;
        };
        self.module_namespaces
            .borrow()
            .get(&leaf)
            .and_then(|ns| ns.types.get(&last).copied())
            .unwrap_or(TypeIdx::NONE)
    }

    /// Resolve a chained FUNCTION access (`w.leaf.makePoint`) to the leaf
    /// module's function. The C++ `resolveChainedFunction`.
    pub fn resolve_chained_function(&self, dotted: &str) -> Option<FunctionAST> {
        let (leaf, last) = self.walk_chain(dotted)?;
        self.get_function_ast(&format!("{leaf}.{last}"))
    }
    /// Resolve a type-alias name to its target TypeIdx, or `NONE`.
    pub fn lookup_type_alias(&self, name: &str) -> TypeIdx {
        self.type_aliases
            .borrow()
            .get(name)
            .copied()
            .unwrap_or(TypeIdx::NONE)
    }

    // ---- type lowering ----

    /// Resolve a `TypeIdx` to its LLVM type, memoized per TypeIdx.
    ///
    /// `requalify_type` (which keys each module's same-named type separately) is
    /// deferred to the decl/analyzer increment; until then the caller is
    /// responsible for passing already-qualified TypeIdxs.
    pub fn get_llvm_type(&self, ty: TypeIdx) -> Result<Type<'ctx>, String> {
        // Requalify a bare imported-module type to its qualified registry name
        // against the module whose body is being lowered (no-op for the entry
        // module / already-qualified types); memoize by the resolved TypeIdx.
        let ty = self.requalify_type(ty, &self.current_body_module());
        let idx = ty.index();
        {
            let cache = self.llvm_type_cache.borrow();
            if let Some(Some(t)) = cache.get(idx) {
                return Ok(*t);
            }
        }
        // Compute without holding the cache borrow (the recursion re-enters).
        let result = self.compute_llvm_type(ty)?;
        {
            let mut cache = self.llvm_type_cache.borrow_mut();
            if cache.len() <= idx {
                cache.resize(idx + 1, None);
            }
            cache[idx] = Some(result);
        }
        Ok(result)
    }

    fn compute_llvm_type(&self, ty: TypeIdx) -> Result<Type<'ctx>, String> {
        let k = self.type_pool.get(ty);
        match k.kind {
            TypeKind::Invalid | TypeKind::Void | TypeKind::NoReturn => Ok(self.ctx.void_type()),
            TypeKind::Bool => Ok(self.ctx.i1_type()),
            TypeKind::Int => match k.a {
                8 => Ok(self.ctx.i8_type()),
                16 => Ok(self.ctx.i16_type()),
                32 => Ok(self.ctx.i32_type()),
                64 => Ok(self.ctx.i64_type()),
                w => Err(format!("Unsupported int width: {w}")),
            },
            TypeKind::Float => {
                if k.a == 32 {
                    Ok(self.ctx.f32_type())
                } else {
                    Ok(self.ctx.f64_type())
                }
            }
            TypeKind::PtrSingle | TypeKind::PtrMany => {
                // Pointee lowered for cache priming / validation; LLVM pointers
                // are opaque so the address space is what matters.
                let _elem = self.get_llvm_type(TypeIdx::new(k.a))?;
                Ok(self.ctx.pointer_type(0))
            }
            TypeKind::Slice => {
                let elem = self.get_llvm_type(TypeIdx::new(k.a))?;
                let _ = elem;
                let elem_ptr = self.ctx.pointer_type(0);
                let usize_ty = self.ctx.i64_type();
                Ok(self.ctx.struct_type(&[elem_ptr, usize_ty], false))
            }
            TypeKind::Array => {
                let elem = self.get_llvm_type(TypeIdx::new(k.a))?;
                Ok(elem.array_type(k.b as u64))
            }
            // Direct enum kind always lowers to the i8 tag.
            TypeKind::Enum => Ok(self.ctx.i8_type()),
            TypeKind::Struct => self
                .struct_llvm_type(ty)
                .ok_or_else(|| format!("Unknown struct type: {}", self.str_name(k.a))),
            TypeKind::Union => self
                .union_llvm_type(ty)
                .ok_or_else(|| format!("Unknown union type: {}", self.str_name(k.a))),
            TypeKind::Named => {
                // A handle-qualified spelling of a PRIVATE type is rejected
                // before the registry can serve the internal qualified key.
                if let Some(msg) = self.namespace_privacy_violation(&self.str_name(k.a)) {
                    return Err(msg);
                }
                // Registry-backed resolution. The substitution / alias /
                // chained-reexport / privacy resolution order lands with the
                // analyzer + substitution engine.
                if let Some(t) = self.struct_llvm_type(ty) {
                    Ok(t)
                } else if let Some(t) = self.union_llvm_type(ty) {
                    Ok(t)
                } else if let Some(has_payload) = self.enum_has_payload(ty) {
                    if has_payload {
                        self.enum_llvm_type(ty).ok_or_else(|| {
                            format!("enum `{}` has no lowered LLVM type yet", self.str_name(k.a))
                        })
                    } else {
                        Ok(self.ctx.i8_type())
                    }
                } else {
                    Err(format!(
                        "unresolved Named type `{}` (subst/alias/chained resolution \
                         not yet ported)",
                        self.str_name(k.a)
                    ))
                }
            }
            TypeKind::ArrayExpr => {
                let resolved = self.resolve_array_expr(ty);
                if !resolved.is_none() && resolved != ty {
                    self.get_llvm_type(resolved)
                } else {
                    Err("ArrayExpr lowering needs array-expr resolution (deferred)".into())
                }
            }
            TypeKind::GenericCall => {
                let resolved = self.resolve_generic_call(ty);
                if !resolved.is_none() && resolved != ty {
                    self.get_llvm_type(resolved)
                } else {
                    Err("GenericCall lowering needs the substitution engine (deferred)".into())
                }
            }
            TypeKind::Type => Err(
                "internal: cannot lower `type` to LLVM (generic was not instantiated before codegen)"
                    .into(),
            ),
            TypeKind::Module => Err("Module type has no LLVM representation".into()),
            // An opaque pointer — the per-call-site signature is rebuilt by the
            // CallIndirect path from the Fn TypeKey (the C++ codegen.cpp:244).
            TypeKind::Fn => Ok(self.ctx.pointer_type(0)),
        }
    }

    // ---- size / alignment ----

    /// Size of a type in bytes (memoized). Mirrors LLVM's aggregate layout so
    /// `{i64, i8}` reports 16, not 9.
    pub fn type_size(&self, ty: TypeIdx) -> Result<u64, String> {
        let ty = self.requalify_type(ty, &self.current_body_module());
        if let Some(&v) = self.size_memo.borrow().get(&ty) {
            return Ok(v);
        }
        let r = self.type_size_impl(ty)?;
        self.size_memo.borrow_mut().insert(ty, r);
        Ok(r)
    }

    fn type_size_impl(&self, ty: TypeIdx) -> Result<u64, String> {
        let k = self.type_pool.get(ty);
        match k.kind {
            TypeKind::Invalid
            | TypeKind::Void
            | TypeKind::NoReturn
            | TypeKind::Type
            | TypeKind::Module => Ok(0),
            TypeKind::Bool => Ok(1),
            TypeKind::Int | TypeKind::Float => Ok((k.a / 8) as u64),
            TypeKind::PtrSingle | TypeKind::PtrMany | TypeKind::Fn => Ok(8),
            TypeKind::Slice => Ok(16), // { ptr, len }
            TypeKind::Array => Ok(k.b as u64 * self.type_size(TypeIdx::new(k.a))?),
            TypeKind::Enum => Ok(1), // unit-only enums lower to u8
            TypeKind::Struct | TypeKind::Named => {
                // Deferred: substitution + alias resolution.
                if let Some(fields) = self.struct_fields(ty) {
                    let mut off = 0u64;
                    let mut max_align = 1u64;
                    for (_, fty) in &fields {
                        let a = self.type_align(*fty)?;
                        off = align_up(off, a);
                        off += self.type_size(*fty)?;
                        if a > max_align {
                            max_align = a;
                        }
                    }
                    return Ok(align_up(off, max_align));
                }
                if let Some(fields) = self.union_fields(ty) {
                    return self.union_size(fields);
                }
                if let Some(has_payload) = self.enum_has_payload(ty) {
                    if !has_payload {
                        return Ok(1);
                    }
                    let (mps, mpa) = self.enum_payload_layout(ty).unwrap();
                    let padded = align_up(mps, mpa);
                    return Ok(if padded == 0 { 2 * mpa } else { mpa + padded });
                }
                Err("type_size: unresolved user type (subst/alias resolution deferred)".into())
            }
            TypeKind::Union => match self.union_fields(ty) {
                Some(fields) => self.union_size(fields),
                None => Err("type_size: unknown union".into()),
            },
            TypeKind::ArrayExpr => {
                let resolved = self.resolve_array_expr(ty);
                if !resolved.is_none() && resolved != ty {
                    self.type_size(resolved)
                } else {
                    Err("type_size: ArrayExpr resolution deferred".into())
                }
            }
            TypeKind::GenericCall => {
                let resolved = self.resolve_generic_call(ty);
                if !resolved.is_none() && resolved != ty {
                    self.type_size(resolved)
                } else {
                    Err("type_size: GenericCall resolution deferred".into())
                }
            }
        }
    }

    fn union_size(&self, fields: Vec<(String, TypeIdx)>) -> Result<u64, String> {
        let mut max_size = 0u64;
        let mut max_align = 1u64;
        for (_, fty) in &fields {
            let s = self.type_size(*fty)?;
            let a = self.type_align(*fty)?;
            if s > max_size {
                max_size = s;
            }
            if a > max_align {
                max_align = a;
            }
        }
        Ok(align_up(max_size, max_align))
    }

    /// Alignment of a type in bytes (memoized). Equal to size for scalars; the
    /// max of constituents for aggregates.
    pub fn type_align(&self, ty: TypeIdx) -> Result<u64, String> {
        let ty = self.requalify_type(ty, &self.current_body_module());
        if let Some(&v) = self.align_memo.borrow().get(&ty) {
            return Ok(v);
        }
        let r = self.type_align_impl(ty)?;
        self.align_memo.borrow_mut().insert(ty, r);
        Ok(r)
    }

    /// Alignment for a *standalone storage slot* (alloca / byref-aggregate
    /// memcpy). Matches the C++ oracle's quirk: a payloaded, NON-generic enum
    /// referenced by a *module-qualified* (imported) name reports slot
    /// alignment 1, because the imported enum's `max_payload_align` is never
    /// lifted past its initial 1 even though its LLVM body is laid out fully.
    /// The entry module's own (bare-named) enums, every generic monomorph
    /// (`Enum__arg`, which `set_enum_llvm_type`s the real align at
    /// instantiation), and all structs/unions keep their real alignment.
    /// Use this ONLY at the standalone-storage sites, never for field layout.
    pub fn alloca_align(&self, ty: TypeIdx) -> Result<u64, String> {
        if self.enum_has_payload(ty) == Some(true)
            && let Some(name) = self.enum_name_of(ty)
        {
            let qualified = name.contains('.') || name.contains('/');
            let is_monomorph = name.contains("__");
            if qualified && !is_monomorph {
                return Ok(1);
            }
        }
        self.type_align(ty)
    }

    fn type_align_impl(&self, ty: TypeIdx) -> Result<u64, String> {
        let k = self.type_pool.get(ty);
        match k.kind {
            TypeKind::Invalid
            | TypeKind::Void
            | TypeKind::NoReturn
            | TypeKind::Bool
            | TypeKind::Type
            | TypeKind::Module
            | TypeKind::Enum => Ok(1),
            TypeKind::Int | TypeKind::Float => Ok((k.a / 8) as u64),
            TypeKind::PtrSingle | TypeKind::PtrMany | TypeKind::Slice | TypeKind::Fn => Ok(8),
            TypeKind::Array => self.type_align(TypeIdx::new(k.a)),
            TypeKind::Struct | TypeKind::Named => {
                if let Some(fields) = self.struct_fields(ty) {
                    return self.max_field_align(fields);
                }
                if let Some(fields) = self.union_fields(ty) {
                    return self.max_field_align(fields);
                }
                if let Some(has_payload) = self.enum_has_payload(ty) {
                    let (_mps, mpa) = self.enum_payload_layout(ty).unwrap();
                    return Ok(if has_payload { mpa } else { 1 });
                }
                Err("type_align: unresolved user type (subst/alias resolution deferred)".into())
            }
            TypeKind::Union => match self.union_fields(ty) {
                Some(fields) => self.max_field_align(fields),
                None => Err("type_align: unknown union".into()),
            },
            TypeKind::ArrayExpr => {
                let resolved = self.resolve_array_expr(ty);
                if !resolved.is_none() && resolved != ty {
                    self.type_align(resolved)
                } else {
                    Err("type_align: ArrayExpr resolution deferred".into())
                }
            }
            TypeKind::GenericCall => {
                let resolved = self.resolve_generic_call(ty);
                if !resolved.is_none() && resolved != ty {
                    self.type_align(resolved)
                } else {
                    Err("type_align: GenericCall resolution deferred".into())
                }
            }
        }
    }

    fn max_field_align(&self, fields: Vec<(String, TypeIdx)>) -> Result<u64, String> {
        let mut max_align = 1u64;
        for (_, fty) in &fields {
            let a = self.type_align(*fty)?;
            if a > max_align {
                max_align = a;
            }
        }
        Ok(max_align)
    }
}

/// Folds a `cfn` call encountered in a comptime position (`[bufLen(8)]u8`, a
/// `cfn` calling another `cfn`) by running the callee's body in the evaluator.
/// Mirrors the C++ `CompCallResolverImpl`. The depth / total-call budgets are
/// shared `Cell`s so the whole instantiation chain is bounded.
struct CfnResolver<'a, 'ctx> {
    ctx: &'a CodegenContext<'ctx>,
    depth: &'a Cell<u32>,
    total: &'a Cell<u32>,
}

/// Recursion cap (C++ `kMaxDepth`).
const CFN_MAX_DEPTH: u32 = 256;
/// Total-call cap across one fold (C++ `kMaxTotalCalls`).
const CFN_MAX_TOTAL_CALLS: u32 = 1_000_000;

impl CompCallResolver for CfnResolver<'_, '_> {
    fn resolve_call(
        &mut self,
        name: &str,
        args: &[ComptimeValue],
        diags: &mut Diagnostics,
        loc: &SrcLoc,
    ) -> ComptimeValue {
        // Resolve against the body module first (the C++ namespace walk), then
        // the bare name — two modules may each define a same-named cfn.
        let cur = self.ctx.current_body_module();
        let f = if !cur.is_empty() && !name.contains('.') {
            self.ctx
                .get_function_ast(&format!("{cur}.{name}"))
                .or_else(|| self.ctx.get_function_ast(name))
        } else {
            self.ctx.get_function_ast(name)
        };
        let f = match f {
            Some(f) if f.is_comp_time_fn => f,
            _ => return ComptimeValue::None,
        };
        if f.args.len() != args.len() {
            return ComptimeValue::None;
        }
        if self.total.get() >= CFN_MAX_TOTAL_CALLS || self.depth.get() >= CFN_MAX_DEPTH {
            return ComptimeValue::None;
        }
        self.total.set(self.total.get() + 1);
        self.depth.set(self.depth.get() + 1);

        // A fresh lexical scope: the callee sees only its bound params.
        let mut scope = ComptimeScope::new();
        for (p, v) in f.args.iter().zip(args.iter()) {
            scope.bind(p.name.clone(), v.clone());
        }
        let mut ret = ComptimeValue::None;
        let mut iter = 0u32;
        // Build a local evaluator from the context pools (the resolver no longer
        // borrows one), and a fresh resolver sharing the SAME budget Cells for
        // cfn->cfn calls.
        let ev = ComptimeEvaluator::new(
            &self.ctx.node_store,
            &self.ctx.string_pool,
            &self.ctx.type_pool,
        );
        let mut nested = CfnResolver {
            ctx: self.ctx,
            depth: self.depth,
            total: self.total,
        };
        let mut nctx = CompCtx {
            resolver: Some(&mut nested),
            emitter: None,
            diags: Some(diags),
            loc: loc.clone(),
            host_os: Os::MacOs,
        };
        ev.exec_block(
            &f.body,
            &mut scope,
            &mut iter,
            DEFAULT_ITER_CAP,
            &mut ret,
            &mut nctx,
        );
        self.depth.set(self.depth.get() - 1);
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jam_syntax::ast_flat::builtin;

    #[test]
    fn qualify_type_name_entry_vs_module() {
        assert_eq!(qualify_type_name("", "Thing"), "Thing");
        assert_eq!(qualify_type_name("lib/a", "Thing"), "lib/a.Thing");
    }

    #[test]
    fn body_module_stack_and_type_owner() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        assert_eq!(cg.current_body_module(), "");
        cg.push_body_module("lib/a");
        cg.push_body_module("lib/b");
        assert_eq!(cg.current_body_module(), "lib/b");
        cg.pop_body_module();
        assert_eq!(cg.current_body_module(), "lib/a");
        cg.pop_body_module();
        assert_eq!(cg.current_body_module(), "");

        cg.register_type_owner("main", "Thing", "lib/a");
        assert_eq!(cg.type_owner("main", "Thing").as_deref(), Some("lib/a"));
        assert!(cg.type_owner("main", "Other").is_none());
    }

    #[test]
    fn diagnostics_accumulate_through_shared_ref() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        assert!(!cg.has_errors());
        cg.push_error(jam_core::diag::SrcLoc::new("f.jam", 3), "boom");
        assert!(cg.has_errors());
        assert_eq!(cg.error_count(), 1);
    }

    #[test]
    fn lowers_primitive_types() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        assert!(cg.get_llvm_type(builtin::BOOL).unwrap().is_integer());
        assert_eq!(cg.get_llvm_type(builtin::BOOL).unwrap().int_width(), 1);
        assert_eq!(cg.get_llvm_type(builtin::I32).unwrap().int_width(), 32);
        assert_eq!(cg.get_llvm_type(builtin::I8).unwrap().int_width(), 8);
        assert_eq!(cg.get_llvm_type(builtin::I64).unwrap().int_width(), 64);
        assert!(cg.get_llvm_type(builtin::F64).unwrap().is_float());
        assert!(cg.get_llvm_type(builtin::F32).unwrap().is_float());
        assert!(cg.get_llvm_type(builtin::VOID).unwrap().is_void());
    }

    #[test]
    fn lowers_pointer_slice_array() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let p = cg.type_pool.intern_ptr_single(builtin::I32);
        assert!(cg.get_llvm_type(p).unwrap().is_pointer());

        let s = cg.type_pool.intern_slice(builtin::U8);
        assert!(cg.get_llvm_type(s).unwrap().is_struct()); // { ptr, i64 }

        let a = cg.type_pool.intern_array(builtin::I32, 4);
        let at = cg.get_llvm_type(a).unwrap();
        assert!(at.is_array());
        assert!(at.array_element_type().is_integer());
    }

    #[test]
    fn lowers_registered_struct() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        // Intern the name + a Struct TypeIdx, register its LLVM body.
        let nid = cg.string_pool.intern(b"Point");
        let sty = cg.type_pool.intern_struct(nid);
        let llvm_st = ctx.struct_type(&[ctx.i32_type(), ctx.i32_type()], false);
        cg.register_struct(
            "Point",
            llvm_st,
            vec![("x".into(), builtin::I32), ("y".into(), builtin::I32)],
        );

        let lowered = cg.get_llvm_type(sty).unwrap();
        assert!(lowered.is_struct());
        assert_eq!(cg.field_index("Point", "y"), 1);
        assert_eq!(cg.field_index("Point", "z"), -1);
        assert!(cg.is_struct_registered(sty));
    }

    #[test]
    fn unregistered_struct_errors() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let nid = cg.string_pool.intern(b"Ghost");
        let sty = cg.type_pool.intern_struct(nid);
        let err = cg.get_llvm_type(sty).unwrap_err();
        assert!(err.contains("Unknown struct type: Ghost"));
    }

    #[test]
    fn type_cache_is_memoized() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let a = cg.get_llvm_type(builtin::I32).unwrap();
        let b = cg.get_llvm_type(builtin::I32).unwrap();
        // Same handle returned (cache hit).
        assert_eq!(a, b);
    }

    #[test]
    fn primitive_sizes_and_aligns() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        assert_eq!(cg.type_size(builtin::I32).unwrap(), 4);
        assert_eq!(cg.type_size(builtin::BOOL).unwrap(), 1);
        assert_eq!(cg.type_size(builtin::F64).unwrap(), 8);
        assert_eq!(cg.type_align(builtin::I64).unwrap(), 8);
        let p = cg.type_pool.intern_ptr_single(builtin::I32);
        assert_eq!(cg.type_size(p).unwrap(), 8);
        let s = cg.type_pool.intern_slice(builtin::U8);
        assert_eq!(cg.type_size(s).unwrap(), 16); // { ptr, len }
        let a = cg.type_pool.intern_array(builtin::I32, 4);
        assert_eq!(cg.type_size(a).unwrap(), 16);
        assert_eq!(cg.type_align(a).unwrap(), 4);
    }

    #[test]
    fn struct_layout_matches_llvm_padding() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let nid = cg.string_pool.intern(b"S");
        let sty = cg.type_pool.intern_struct(nid);
        let llvm = ctx.struct_type(&[ctx.i64_type(), ctx.i8_type()], false);
        cg.register_struct(
            "S",
            llvm,
            vec![("a".into(), builtin::I64), ("b".into(), builtin::I8)],
        );
        // {i64, i8} => 16 (padded to align), NOT 9. align 8.
        assert_eq!(cg.type_size(sty).unwrap(), 16);
        assert_eq!(cg.type_align(sty).unwrap(), 8);

        // {i32, i8} => i32 at 0 (size 4), i8 at 4 (size 1) => 5, padded to
        // align 4 => 8.
        let nid2 = cg.string_pool.intern(b"S2");
        let sty2 = cg.type_pool.intern_struct(nid2);
        let llvm2 = ctx.struct_type(&[ctx.i32_type(), ctx.i8_type()], false);
        cg.register_struct(
            "S2",
            llvm2,
            vec![("a".into(), builtin::I32), ("b".into(), builtin::I8)],
        );
        assert_eq!(cg.type_size(sty2).unwrap(), 8);
        assert_eq!(cg.type_align(sty2).unwrap(), 4);
    }

    #[test]
    fn union_size_is_max_field() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let nid = cg.string_pool.intern(b"U");
        let uty = cg.type_pool.intern_named(nid); // Named "U" resolves via union registry
        let llvm = ctx.i64_type();
        cg.register_union(
            "U",
            llvm,
            vec![("a".into(), builtin::I64), ("b".into(), builtin::I8)],
        );
        assert_eq!(cg.type_size(uty).unwrap(), 8); // max(8, 1) padded to align 8
        assert_eq!(cg.type_align(uty).unwrap(), 8);
    }

    #[test]
    fn unit_enum_named_lowers_to_i8() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let nid = cg.string_pool.intern(b"Color");
        let nty = cg.type_pool.intern_named(nid);
        cg.register_enum(
            "Color",
            vec![EnumVariantInfo {
                name: "Red".into(),
                payload_types: vec![],
                discriminant: 0,
            }],
        );
        // No payload variant -> i8.
        let lowered = cg.get_llvm_type(nty).unwrap();
        assert!(lowered.is_integer());
        assert_eq!(lowered.int_width(), 8);
        assert_eq!(cg.enum_variant_index("Color", "Red"), 0);
    }
}
