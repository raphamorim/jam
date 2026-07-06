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

