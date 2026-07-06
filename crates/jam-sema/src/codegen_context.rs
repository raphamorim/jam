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

