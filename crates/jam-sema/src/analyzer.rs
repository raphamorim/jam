/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Demand-driven decl analysis. Ported from `src/analyzer.{h,cpp}`.
//!
//! Two state machines stack on the [`DeclTable`](crate::decl::DeclTable):
//!   * `Decl.analysis` (`Unreferenced`/`InProgress`/`Complete`/…) — the
//!     chokepoint at [`Analyzer::ensure_decl_analyzed`]. Re-entering an
//!     `InProgress` decl is a dependency loop.
//!   * `Struct/Enum/Union.status` — a separate body lifecycle, materialised on
//!     demand by `resolve_type_fields_*` (the next increment).
//!
//! ## Design
//!
//! Unlike the C++ (`Analyzer` holds `ctx&` + `DeclTable&` and the ctx has a
//! back-reference for `get_llvm_type` → `ensure_struct_body`), this is a
//! transient struct over `&mut CodegenContext` + `&mut DeclTable`. The analyzer
//! drives its own dependency order, so `get_llvm_type` stays a pure `&self`
//! reader (it returns the *named* struct handle; bodies are filled separately
//! via `set_body`) — no bidirectional borrow, no merge.
//!
//! ## Scope of this increment
//!
//! The chokepoint + cycle detection + `analyze_function` (the `FnSignature`
//! ABI cache) + the type-decl dispatch (each type resolves to its `Named`
//! TypeIdx). The intricate `resolve_type_fields_{struct,enum,union}` body-fill
//! — which materialises LLVM struct bodies and the enum/union layout math, and
//! needs the driver's register pass to have created the named types — lands
//! next.

use jam_core::diag::{SrcLoc, Trace, TraceKind};
use jam_core::index::{DeclIndex, StringIdx, TypeIdx};
use jam_syntax::ast_flat::TypeKind;

use crate::abi::{classify_param, classify_return};
use crate::codegen_context::CodegenContext;
use crate::decl::{DeclAnalysis, DeclKind, DeclTable, DeclValue};

/// Round `off` up to the next multiple of `align` (matches the C++ `alignUp`).
fn round_up(off: u64, align: u64) -> u64 {
    off.div_ceil(align) * align
}

/// Which body-fill stack a cycle error draws its trace from.
#[derive(Copy, Clone)]
enum BodyKind {
    Struct,
    Enum,
    Union,
}

pub struct Analyzer<'c, 'ctx> {
    ctx: &'c mut CodegenContext<'ctx>,
    decls: &'c mut DeclTable<'ctx>,
    /// Currently-analyzing decl indices; a decl already on the stack when asked
    /// again is a dependency cycle.
    analysis_stack: Vec<DeclIndex>,
    /// Parallel stacks for the body-fill lifecycles (struct/enum/union have
    /// their own `*Status::*Wip` cycle detection, separate from `analysis_stack`).
    struct_fill_stack: Vec<DeclIndex>,
    enum_fill_stack: Vec<DeclIndex>,
    union_fill_stack: Vec<DeclIndex>,
}

impl<'c, 'ctx> Analyzer<'c, 'ctx> {
    pub fn new(
        ctx: &'c mut CodegenContext<'ctx>,
        decls: &'c mut DeclTable<'ctx>,
    ) -> Analyzer<'c, 'ctx> {
        Analyzer {
            ctx,
            decls,
            analysis_stack: Vec::new(),
            struct_fill_stack: Vec::new(),
            enum_fill_stack: Vec::new(),
            union_fill_stack: Vec::new(),
        }
    }

    /// The current analysis stack (for trace formatting / tests).
    pub fn analysis_stack(&self) -> &[DeclIndex] {
        &self.analysis_stack
    }

    /// Ensure `idx`'s value has been computed; returns it. Idempotent — a
    /// `Complete` decl returns its cached value; an `InProgress` decl triggers
    /// the cycle detector and returns `None`.
    pub fn ensure_decl_analyzed(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        if idx.is_none() {
            return DeclValue::None;
        }
        match self.decls.get(idx).analysis {
            DeclAnalysis::Complete => return self.decls.get(idx).value.clone(),
            DeclAnalysis::InProgress => {
                self.push_cycle_error(idx);
                // Don't mark failed — the cycle is the caller's problem too. A
                // later call (after unwind) hits Complete cleanly.
                return DeclValue::None;
            }
            DeclAnalysis::AnalysisFailure | DeclAnalysis::DependencyFailure => {
                return DeclValue::None;
            }
            DeclAnalysis::Unreferenced => {}
        }

        self.decls.get_mut(idx).analysis = DeclAnalysis::InProgress;
        self.analysis_stack.push(idx);
        let value = self.analyze_decl(idx);
        self.analysis_stack.pop();

        // Re-fetch by index — `analyze_decl` may have created new decls and
        // reallocated the table (analyzer.cpp:108). Never hold a `Decl` borrow
        // across the recursion.
        let again = self.decls.get_mut(idx);
        if again.analysis == DeclAnalysis::InProgress {
            // No terminal state set by the branch: default to Complete with the
            // returned value (a branch that pushed a cycle error returns None
            // but stays InProgress, so this marks it Complete-with-None — a
            // resolved state, so retries don't recurse forever).
            again.analysis = DeclAnalysis::Complete;
            again.value = value.clone();
        }
        self.decls.get(idx).value.clone()
    }

    /// Demand-driven name lookup: find the decl, force its analysis, return the
    /// resolved value. `None`-kind if the name isn't in the table.
    pub fn resolve_decl(&mut self, name: &str) -> DeclValue<'ctx> {
        let idx = self.decls.find_by_name(name);
        if idx.is_none() {
            return DeclValue::None;
        }
        self.ensure_decl_analyzed(idx)
    }

    fn analyze_decl(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        match self.decls.get(idx).kind {
            DeclKind::Function => self.analyze_function(idx),
            DeclKind::Struct => self.analyze_struct(idx),
            DeclKind::Enum => self.analyze_enum(idx),
            DeclKind::Union => self.analyze_union(idx),
            // Const-or-alias decision is made inside `analyze_const`.
            DeclKind::Const | DeclKind::TypeAlias => self.analyze_const(idx),
            DeclKind::Invalid => DeclValue::None,
        }
    }

    fn analyze_function(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        let fn_ast = self.decls.get(idx).fn_ast;
        let f = match fn_ast {
            Some(f) => f,
            None => return DeclValue::None,
        };
        // Cache the ABI signature for non-generic functions so downstream
        // codegen / call sites read one source instead of re-classifying.
        // Generic functions skip the cache (per-instantiation TypeIdxs).
        let computed = self.decls.get(idx).signature.computed;
        if !f.is_generic() && !computed {
            let loc = self.loc_of(idx);
            let mut params = Vec::with_capacity(f.args.len());
            for p in &f.args {
                match classify_param(p.mode, p.ty, self.ctx) {
                    Ok(pa) => params.push(pa),
                    Err(e) => {
                        self.ctx
                            .push_error(loc, format!("fn `{}` param `{}`: {e}", f.name, p.name));
                        return DeclValue::Function(f);
                    }
                }
            }
            let return_abi = match classify_return(f.return_type, self.ctx) {
                Ok(r) => r,
                Err(e) => {
                    self.ctx
                        .push_error(loc, format!("fn `{}` return: {e}", f.name));
                    return DeclValue::Function(f);
                }
            };
            let sig = &mut self.decls.get_mut(idx).signature;
            sig.params = params;
            sig.return_abi = return_abi;
            sig.computed = true;
        }
        DeclValue::Function(f)
    }

    /// Resolve a type decl to its `Named` TypeIdx. (Body fill via
    /// `resolve_type_fields_*` lands next; the decl is still usable as a type.)
    fn type_named_value(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        let name = self.decls.get(idx).name.clone();
        let sid = self.ctx.string_pool.intern(name.as_bytes());
        DeclValue::Type(self.ctx.type_pool.intern_named(sid))
    }

