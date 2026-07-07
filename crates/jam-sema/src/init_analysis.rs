/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Definite-initialization / move-ownership analysis — ported from
//! `src/init_analysis.{h,cpp}`.
//!
//! A tree-walking dataflow over a function body produces, at each program
//! point, a map from binding name to a 2-bit lattice state {Unknown, Init,
//! Uninit, MaybeInit}. The lattice merges by bitwise OR, exactly as Clang's
//! `UninitializedValues` pass does. Jam programs are structured (no goto), so a
//! recursive analyzer over the AST suffices — there is no separate CFG pass.
//!
//! ## Scope of this port
//!
//! Unlike the C++ standalone analyzer (which threaded raw callback function
//! pointers through an `AnalysisHooks` struct), this port lives *inside*
//! `jam_sema` and borrows the [`CodegenContext`] directly for the questions it
//! cannot answer from its own tables: `type_needs_drop`, `requalify_type`, and
//! the enum-variant-constructor registry. The pools (`NodeStore`, `StringPool`,
//! `TypePool`) are read off the context too.
//!
//! ## Why it exists
//!
//! The single consumer is generic method-instantiation withdrawal
//! ([`crate::astgen::instantiate_methods`]): when a generic method's body is
//! lowered under drop-bearing type args, the oracle runs this analysis and, if
//! it reports any move/ownership diagnostic, WITHDRAWS the method to a bare
//! `declare` (no body). The canonical case is `Vec(T).filled`, which moves a
//! by-value (`let`-mode) drop-bearing parameter into a slot inside a `while`
//! loop — a use-of-moved-binding on loop re-entry, and a move out of a borrowed
//! parameter besides.
//!
//! This port only needs to reproduce the oracle's *accept/reject* verdict, so
//! diagnostic message strings are intentionally terse — what matters is whether
//! [`analyze`] returns a non-empty diagnostics list.

use std::collections::{HashMap, HashSet};

use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};
use jam_core::param_mode::ParamMode;
use jam_syntax::ast::FunctionAST;
use jam_syntax::ast_flat::{AstTag, TypeKind};

use crate::codegen_context::CodegenContext;

/// Per-binding initialization state, a 2-bit lattice. Merge is bitwise OR.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
enum InitState {
    Unknown = 0b00,
    Init = 0b01,
    Uninit = 0b10,
    MaybeInit = 0b11,
}

fn merge_state(a: InitState, b: InitState) -> InitState {
    match (a as u8) | (b as u8) {
        0b00 => InitState::Unknown,
        0b01 => InitState::Init,
        0b10 => InitState::Uninit,
        _ => InitState::MaybeInit,
    }
}

/// One reportable move/ownership problem found during analysis.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub message: String,
    pub var_name: String,
    /// 1-based source line of the anchor node (`0` when none). Parse-time
    /// stamped, so it is correct for imported modules too. Mirrors the C++
    /// `Diagnostic::line` (`lineOf(anchor)`).
    pub line: u32,
}

type NameMap = HashMap<String, InitState>;

/// One step of a borrow path under a call arg: a field access, an index, or a
/// dereference. Used by the exclusivity check (MVS P5) to compare two arg
/// expressions for overlapping access. Mirrors the C++ `Analyzer::PathStep`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PathStep {
    /// `.field` — carries the field-name `StringIdx`.
    Field(u32),
    /// `[idx]` — `Some(n)` for a constant index, `None` for a dynamic one.
    Index(Option<u64>),
    /// `.*` dereference.
    Deref,
}

/// A borrow path rooted at a base binding. `base == None` when the arg isn't a
/// simple lvalue chain (the exclusivity check skips it). Mirrors the C++
/// `Analyzer::BorrowPath`; `steps` is in root->leaf order after extraction.
struct BorrowPath {
    base: Option<u32>,
    steps: Vec<PathStep>,
}

/// Result of analyzing a statement or expression.
struct StmtResult {
    state: NameMap,
    /// Control flow does not reach the next statement (return/break/continue,
    /// or such an exit on every path).
    terminated: bool,
}

/// Run the move/ownership analysis on `fn_ast`'s body, borrowing `ctx` for the
/// type/enum queries. Returns an empty vector on success; otherwise every
/// detected move/ownership problem. The instantiation-withdrawal caller only
/// inspects `is_empty()`.
pub fn analyze(fn_ast: &FunctionAST, ctx: &CodegenContext) -> Vec<Diagnostic> {
    let mut a = Analyzer::new(ctx);
    a.run(fn_ast)
}

struct Analyzer<'a, 'ctx> {
    ctx: &'a CodegenContext<'ctx>,
    diagnostics: Vec<Diagnostic>,

    /// Current function's parameter modes, prebuilt in `run`.
    param_modes: HashMap<String, ParamMode>,
    /// Current function's parameter list (name + mode), for the borrowed-move
    /// rejection message.
    params: Vec<(String, ParamMode)>,

    /// Names declared `comp const` / `comp var` — no runtime slot, skipped.
    comp_names: HashSet<String>,
    /// Static type per binding name (params + var decls).
    var_types: HashMap<String, TypeIdx>,
    /// Conditional nesting depth: 0 in the body, +1 per if-arm/loop/match-arm.
    cond_depth: i32,
    /// Conditional depth at each `var` local's declaration (params at depth 0
    /// only when `move`-mode).
    decl_depth: HashMap<String, i32>,
    /// Every binding moved anywhere (any type) — phrases read-errors as moves.
    moved_bindings: HashSet<String>,
    /// Drop-bearing bindings that were definitely moved.
    moved_drop_bearing: HashSet<String>,
}

impl<'a, 'ctx> Analyzer<'a, 'ctx> {
    fn new(ctx: &'a CodegenContext<'ctx>) -> Self {
        Analyzer {
            ctx,
            diagnostics: Vec::new(),
            param_modes: HashMap::new(),
            params: Vec::new(),
            comp_names: HashSet::new(),
            var_types: HashMap::new(),
            cond_depth: 0,
            decl_depth: HashMap::new(),
            moved_bindings: HashSet::new(),
            moved_drop_bearing: HashSet::new(),
        }
    }

    fn run(&mut self, fn_ast: &FunctionAST) -> Vec<Diagnostic> {
        self.param_modes.clear();
        self.params.clear();
        self.var_types.clear();
        self.decl_depth.clear();
        self.moved_bindings.clear();
        self.moved_drop_bearing.clear();
        self.comp_names.clear();
        self.cond_depth = 0;

        let mut state: NameMap = HashMap::new();
        for p in &fn_ast.args {
            self.param_modes.insert(p.name.clone(), p.mode);
            self.params.push((p.name.clone(), p.mode));
            state.insert(p.name.clone(), InitState::Init);
            self.var_types.insert(p.name.clone(), p.ty);
            // A `move` parameter is OWNED by the callee — it participates in the
            // move/drop rules like a depth-0 local. `let`/`mut` params are
            // borrowed: moving a drop-bearing value out of them is rejected.
            if p.mode == ParamMode::Move {
                self.decl_depth.insert(p.name.clone(), 0);
            }
        }

        let mut r = StmtResult {
            state,
            terminated: false,
        };
        for &stmt in &fn_ast.body {
            if r.terminated {
                break;
            }
            r = self.analyze_node(stmt, r.state);
        }

        // If control reaches the end of the body without an explicit return,
        // every in-scope drop-bearing local must be definitely initialized —
        // codegen synthesizes a drop at fall-off-end. (Functions that always
        // return skip this; the per-return checks cover them.) The C++
        // run()'s `if (!r.terminated)` guard, init_analysis.cpp:298.
        if !r.terminated {
            self.check_drop_bearing_locals_init(&r.state, NodeIdx::NONE);
        }

        std::mem::take(&mut self.diagnostics)
    }

    // ---- hooks into the codegen context ----

    /// Resolve a `StringIdx` to its (lossy-UTF-8) name.
    fn str_name(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.ctx.string_pool.get(StringIdx::new(id))).into_owned()
    }
    fn node_tag(&self, idx: NodeIdx) -> AstTag {
        self.ctx.node_store.get(idx).tag
    }
    fn extra(&self, i: u32) -> u32 {
        self.ctx.node_store.get_extra(ExtraIdx::new(i))
    }

    /// MATCH-MOVE oracle: does this type carry ownership (so matching it by
    /// value consumes the scrutinee)? The C++ `typeNeedsDrop` hook.
    fn type_owns_drops(&self, ty: TypeIdx) -> bool {
        self.ctx.type_needs_drop(ty)
    }

    /// True when `enum_name.variant_name` names a known enum-variant
    /// constructor (concrete enums + generic factories register by name). Tries
    /// the bare name first, then the body-module-qualified spelling so a
    /// module's own enum reference resolves to its registry key.
    fn is_enum_variant_ctor(&self, enum_name: &str, variant_name: &str) -> bool {
        let probe = |key: &str| {
            self.ctx
                .enum_variants_by_name(key)
                .map(|vs| vs.iter().any(|v| v.name == variant_name))
                .unwrap_or(false)
        };
        if probe(enum_name) {
            return true;
        }
        let bm = self.ctx.current_body_module();
        !bm.is_empty() && probe(&format!("{bm}.{enum_name}"))
    }

    /// Move-gate oracle: is `binding_name`'s type drop-bearing? Consults the
    /// context's `type_needs_drop` (which covers registered drops, structs whose
    /// fields drop, payloaded enums, arrays, and generic instantiations).
    fn binding_is_drop_bearing(&self, binding_name: &str) -> bool {
        match self.var_types.get(binding_name) {
            Some(&ty) => self.type_owns_drops(ty),
            None => false,
        }
    }

    // ---- tag dispatch ----

