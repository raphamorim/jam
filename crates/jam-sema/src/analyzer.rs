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

    fn analyze_struct(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        // Body fill is part of the struct's own analysis: ensure_decl_analyzed
        // is the single entry point that both proves cycle-free status AND
        // materialises the LLVM body.
        self.resolve_type_fields_struct(idx);
        self.type_named_value(idx)
    }

    fn analyze_enum(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        self.resolve_type_fields_enum(idx);
        self.type_named_value(idx)
    }

    fn analyze_union(&mut self, idx: DeclIndex) -> DeclValue<'ctx> {
        self.resolve_type_fields_union(idx);
        self.type_named_value(idx)
    }

    // ---- body fill ----

    /// Resolve a `(field name -> TypeKey)` leaf to a registry name when it's a
    /// direct named type (Named/Struct/Enum/Union), or `None` for indirections
    /// (ptr/slice/array) — pointer fields don't make the aggregate
    /// size-dependent on itself, so they skip the cycle check.
    fn direct_named_type(&self, ty: TypeIdx) -> Option<String> {
        let k = self.ctx.type_pool.get(ty);
        if matches!(
            k.kind,
            TypeKind::Named | TypeKind::Struct | TypeKind::Enum | TypeKind::Union
        ) {
            Some(
                String::from_utf8_lossy(&self.ctx.string_pool.get(StringIdx::new(k.a)))
                    .into_owned(),
            )
        } else {
            None
        }
    }

    /// Ensure a direct-named field/payload's body is filled (routing to the
    /// matching registry). Returns true on success / not-a-known-type.
    fn ensure_nested_body(&mut self, name: &str) -> bool {
        if self.ctx.is_struct_name_registered(name) {
            self.ensure_struct_body(name)
        } else if self.ctx.is_enum_name_registered(name) {
            self.ensure_enum_body(name)
        } else if self.ctx.is_union_name_registered(name) {
            self.ensure_union_body(name)
        } else {
            true
        }
    }

    pub fn ensure_struct_body(&mut self, name: &str) -> bool {
        let idx = self.decls.find_by_name_and_kind(name, DeclKind::Struct);
        if idx.is_none() {
            return false;
        }
        self.resolve_type_fields_struct(idx)
    }
    pub fn ensure_enum_body(&mut self, name: &str) -> bool {
        let idx = self.decls.find_by_name_and_kind(name, DeclKind::Enum);
        if idx.is_none() {
            return false;
        }
        self.resolve_type_fields_enum(idx)
    }
    pub fn ensure_union_body(&mut self, name: &str) -> bool {
        let idx = self.decls.find_by_name_and_kind(name, DeclKind::Union);
        if idx.is_none() {
            return false;
        }
        self.resolve_type_fields_union(idx)
    }

    /// Materialise a struct's LLVM body. `FieldTypesWip` re-entry = self-cycle.
    pub fn resolve_type_fields_struct(&mut self, idx: DeclIndex) -> bool {
        use crate::decl::StructStatus as S;
        if idx.is_none() || self.decls.get(idx).kind != DeclKind::Struct {
            return false;
        }
        let module = self
            .decls
            .get(idx)
            .struct_ast
            .map(|s| s.module_path.clone())
            .unwrap_or_default();
        self.ctx.push_body_module(module);

        let result = (|| {
            match self.decls.get(idx).struct_status {
                S::HaveFieldTypes | S::HaveLayout | S::LayoutWip => return true,
                S::FieldTypesWip => {
                    self.push_body_cycle_error(idx, "struct", BodyKind::Struct);
                    return false;
                }
                S::None => {}
            }
            self.decls.get_mut(idx).struct_status = S::FieldTypesWip;
            self.struct_fill_stack.push(idx);

            let struct_ast = self.decls.get(idx).struct_ast;
            let name = self.decls.get(idx).name.clone();
            let fields = match struct_ast {
                Some(s) => s.fields.clone(),
                None => {
                    self.decls.get_mut(idx).struct_status = S::HaveFieldTypes;
                    self.struct_fill_stack.pop();
                    return true;
                }
            };

            let mut field_llvm = Vec::with_capacity(fields.len());
            let mut failed = false;
            for (_, fty) in &fields {
                // Fill a directly-named nested type's body BEFORE sizing the
                // field. A payloaded-enum (or struct/union) field needs its body
                // filled before `get_llvm_type` can resolve its aggregate type;
                // querying first errored and stubbed the whole struct to `{ i8 }`
                // (which then GEP'd out of bounds -> LLVM abort). The C++ ensures
                // the nested body the same way (analyzer.cpp:208-234) and only a
                // genuine cycle marks the fill failed.
                if let Some(fname) = self.direct_named_type(*fty)
                    && !self.ensure_nested_body(&fname)
                {
                    failed = true;
                }
                match self.ctx.get_llvm_type(*fty) {
                    Ok(t) => field_llvm.push(t),
                    Err(_) => {
                        failed = true;
                        field_llvm.push(self.ctx.context().i8_type());
                    }
                }
            }

            if let Some(named) = self.ctx.struct_named_type(&name) {
                if failed {
                    named.set_body(&[self.ctx.context().i8_type()], false);
                } else {
                    named.set_body(&field_llvm, false);
                }
            }
            self.decls.get_mut(idx).struct_status = S::HaveFieldTypes;
            self.struct_fill_stack.pop();
            !failed
        })();
        self.ctx.pop_body_module();
        result
    }

    /// Materialise a payloaded enum's `{ i8 tag, alignDriver, [extra x i8] }`
    /// body. Unit-only enums need no struct (they lower to `i8`).
    pub fn resolve_type_fields_enum(&mut self, idx: DeclIndex) -> bool {
        use crate::decl::EnumStatus as E;
        if idx.is_none() || self.decls.get(idx).kind != DeclKind::Enum {
            return false;
        }
        let module = self
            .decls
            .get(idx)
            .enum_ast
            .map(|e| e.module_path.clone())
            .unwrap_or_default();
        self.ctx.push_body_module(module);
        let result = (|| {
            match self.decls.get(idx).enum_status {
                E::HaveBody => return true,
                E::BodyWip => {
                    self.push_body_cycle_error(idx, "enum", BodyKind::Enum);
                    return false;
                }
                E::None => {}
            }
            self.decls.get_mut(idx).enum_status = E::BodyWip;
            self.enum_fill_stack.push(idx);

            let name = self.decls.get(idx).name.clone();
            let has_payload = self.ctx.enum_has_payload_by_name(&name).unwrap_or(false);
            if !has_payload {
                // Unit-only enum lowers to i8; nothing to materialise.
                self.decls.get_mut(idx).enum_status = E::HaveBody;
                self.enum_fill_stack.pop();
                return true;
            }
            let variants = self.ctx.enum_variants_by_name(&name).unwrap_or_default();

            let mut max_size = 0u64;
            let mut max_align = 1u64;
            let mut failed = false;
            'outer: for v in &variants {
                let mut off = 0u64;
                let mut var_align = 1u64;
                for &t in &v.payload_types {
                    if let Some(pname) = self.direct_named_type(t)
                        && !self.ensure_nested_body(&pname)
                    {
                        failed = true;
                        break 'outer;
                    }
                    let (s, a) = match (self.ctx.type_size(t), self.ctx.type_align(t)) {
                        (Ok(s), Ok(a)) => (s, a),
                        _ => {
                            failed = true;
                            break 'outer;
                        }
                    };
                    off = round_up(off, a) + s;
                    if a > var_align {
                        var_align = a;
                    }
                }
                if var_align > 1 {
                    off = round_up(off, var_align);
                }
                if off > max_size {
                    max_size = off;
                }
                if var_align > max_align {
                    max_align = var_align;
                }
            }

            // The enum's LLVM named struct uses the bare enum name (the C++
            // names it `%Op`, not `%enum.Op`); instantiated generic enums
            // already use the bare inst_name.
            let named = self.ctx.context().named_struct(&name);
            if failed {
                let i8 = self.ctx.context().i8_type();
                named.set_body(&[i8, i8], false);
                self.ctx.set_enum_llvm_type(&name, named, 0, 1, true);
                self.decls.get_mut(idx).enum_status = E::HaveBody;
                self.enum_fill_stack.pop();
                return false;
            }
            let (align_driver, driver_size) = match max_align {
                1 => (self.ctx.context().i8_type(), 1u64),
                2 => (self.ctx.context().i16_type(), 2),
                4 => (self.ctx.context().i32_type(), 4),
                8 => (self.ctx.context().i64_type(), 8),
                _ => {
                    let loc = self.loc_of(idx);
                    self.ctx.push_error(
                        loc,
                        format!("enum `{name}` requires alignment > 8, which is not yet supported"),
                    );
                    self.decls.get_mut(idx).enum_status = E::HaveBody;
                    self.enum_fill_stack.pop();
                    return false;
                }
            };
            let padded = round_up(max_size, max_align);
            let extra = padded.saturating_sub(driver_size);
            let i8 = self.ctx.context().i8_type();
            let mut body = vec![i8, align_driver];
            if extra > 0 {
                body.push(i8.array_type(extra));
            }
            named.set_body(&body, false);
            self.ctx
                .set_enum_llvm_type(&name, named, max_size, max_align, true);

            self.decls.get_mut(idx).enum_status = E::HaveBody;
            self.enum_fill_stack.pop();
            true
        })();
        self.ctx.pop_body_module();
        result
    }

    /// Materialise a union's `{ alignedField, [padding x i8] }` body.
    pub fn resolve_type_fields_union(&mut self, idx: DeclIndex) -> bool {
        use crate::decl::UnionStatus as U;
        if idx.is_none() || self.decls.get(idx).kind != DeclKind::Union {
            return false;
        }
        let module = self
            .decls
            .get(idx)
            .union_ast
            .map(|u| u.module_path.clone())
            .unwrap_or_default();
        self.ctx.push_body_module(module);
        let result = (|| {
            match self.decls.get(idx).union_status {
                U::HaveBody => return true,
                U::BodyWip => {
                    self.push_body_cycle_error(idx, "union", BodyKind::Union);
                    return false;
                }
                U::None => {}
            }
            self.decls.get_mut(idx).union_status = U::BodyWip;
            self.union_fill_stack.push(idx);

            let name = self.decls.get(idx).name.clone();
            let fields = self.ctx.union_fields_by_name(&name).unwrap_or_default();
            if fields.is_empty() {
                let loc = self.loc_of(idx);
                self.ctx
                    .push_error(loc, format!("union `{name}` must have at least one field"));
                self.decls.get_mut(idx).union_status = U::HaveBody;
                self.union_fill_stack.pop();
                return false;
            }

            let mut max_size = 0u64;
            let mut max_align = 1u64;
            let mut align_field = 0usize;
            let mut failed = false;
            for (i, (_, t)) in fields.iter().enumerate() {
                if let Some(fname) = self.direct_named_type(*t)
                    && !self.ensure_nested_body(&fname)
                {
                    failed = true;
                    break;
                }
                let (s, a) = match (self.ctx.type_size(*t), self.ctx.type_align(*t)) {
                    (Ok(s), Ok(a)) => (s, a),
                    _ => {
                        failed = true;
                        break;
                    }
                };
                if s > max_size {
                    max_size = s;
                }
                if a > max_align {
                    max_align = a;
                    align_field = i;
                }
            }
            let named = self.ctx.union_named_type(&name);
            if failed {
                if let Some(named) = named {
                    named.set_body(&[self.ctx.context().i8_type()], false);
                }
                self.decls.get_mut(idx).union_status = U::HaveBody;
                self.union_fill_stack.pop();
                return false;
            }
            let alloc_size = round_up(max_size, max_align);
            let aligned_ty = self.ctx.get_llvm_type(fields[align_field].1).ok();
            let aligned_sz = self.ctx.type_size(fields[align_field].1).unwrap_or(0);
            let padding = alloc_size.saturating_sub(aligned_sz);
            if let (Some(named), Some(aligned_ty)) = (named, aligned_ty) {
                let mut body = vec![aligned_ty];
                if padding > 0 {
                    body.push(self.ctx.context().i8_type().array_type(padding));
                }
                named.set_body(&body, false);
            }
            self.decls.get_mut(idx).union_status = U::HaveBody;
            self.union_fill_stack.pop();
            true
        })();
        self.ctx.pop_body_module();
        result
    }

