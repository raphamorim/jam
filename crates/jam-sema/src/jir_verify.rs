/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Structural verification of a finished [`JirFunction`]. Catches drift the
//! codegen would otherwise hit as a silent miscompile or a late LLVM-verifier
//! failure. Run after AstGen finishes a function; returns an empty vector when
//! the function is well-formed, otherwise human-readable diagnostics (with
//! source-line hints where available). Ported from `src/jir_verify.{h,cpp}`.
//!
//! Checks:
//!   - Every non-entry block holds no `Alloca` (allocas live in the entry block
//!     only — `emitAllocaHoisted` enforces this at astgen time).
//!   - Every block ends with exactly one terminator (Br/CondBr/Switch/Ret/
//!     Unreachable) as its last instruction; empty blocks are rejected.
//!   - Every `JirRef` operand points into `insts` (1..len) and was defined in an
//!     earlier block (or earlier in the same block) — the def-before-use
//!     invariant the codegen's instruction cache relies on. `NO_JIR_REF` is
//!     allowed only where the slot is optional (e.g. `Ret`'s `a`).
//!   - Every `JirBlockRef` (Br/CondBr/Switch targets) points within `blocks`.
//!   - Every extra-pool slice (Call/CondBr/Switch/StructLit/ArrayLit) fits.
//!   - Binary-op / comparison operands have structurally-agreeing types (when a
//!     [`TypePool`] is supplied; opaque aggregate/pointer types bypass to avoid
//!     false positives).
//!
//! The optional `resolver` maps a `TypeIdx` through generic-call resolution to
//! its canonical concrete form (the C++ passes a function pointer + ctx; here a
//! borrowed closure carries its own context).

use jam_core::diag::{Diagnostic, Severity, SrcLoc};
use jam_core::index::{StringIdx, TypeIdx};
use jam_syntax::ast_flat::{StringPool, TypeKind, TypePool};

use crate::jir::{JirBlockRef, JirExtraIdx, JirFunction, JirRef, JirTag, NO_JIR_BLOCK, NO_JIR_REF};

/// Maps a `TypeIdx` through generic-call resolution / aliasing to its canonical
/// form. Returning `TypeIdx::NONE` means "no better resolution".
pub type JirVerifyResolver<'r> = &'r dyn Fn(TypeIdx) -> TypeIdx;

struct Verifier<'a> {
    jfn: &'a JirFunction,
    types: Option<&'a TypePool>,
    strings: Option<&'a StringPool>,
    resolver: Option<JirVerifyResolver<'a>>,
    diags: Vec<Diagnostic>,
    /// For def-before-use: each instruction is marked defined as the walk steps
    /// past it. Slot 0 (the sentinel) is always usable as `NO_JIR_REF`.
    defined: Vec<bool>,
}

impl<'a> Verifier<'a> {
    fn new(
        jfn: &'a JirFunction,
        types: Option<&'a TypePool>,
        strings: Option<&'a StringPool>,
        resolver: Option<JirVerifyResolver<'a>>,
    ) -> Verifier<'a> {
        let mut defined = vec![false; jfn.insts.len()];
        defined[0] = true;
        Verifier {
            jfn,
            types,
            strings,
            resolver,
            diags: Vec::new(),
            defined,
        }
    }

    /// Walk a `TypeIdx` through the optional resolver.
    fn resolve_ty(&self, ty: TypeIdx) -> TypeIdx {
        match self.resolver {
            Some(f) if ty != TypeIdx::NONE => {
                let r = f(ty);
                if r == TypeIdx::NONE { ty } else { r }
            }
            _ => ty,
        }
    }

    fn err(&mut self, r: JirRef, msg: String) {
        let (line, tag) = if (r as usize) < self.jfn.insts.len() {
            let in_ = &self.jfn.insts[r as usize];
            (in_.src_line, in_.tag.name())
        } else {
            (0, "?")
        };
        let message = format!(
            "jir-verify: fn `{}` ref #{} ({}): {}",
            self.jfn.name, r, tag, msg
        );
        self.diags.push(Diagnostic::new(
            Severity::Error,
            SrcLoc::new("", line),
            message,
        ));
    }

    fn block_err(&mut self, b: JirBlockRef, msg: &str) {
        let name = if (b as usize) < self.jfn.blocks.len() {
            self.jfn.blocks[b as usize].name.as_str()
        } else {
            "?"
        };
        let message = format!(
            "jir-verify: fn `{}` block #{} ({}): {}",
            self.jfn.name, b, name, msg
        );
        self.diags.push(Diagnostic::new(
            Severity::Error,
            SrcLoc::new("", 0),
            message,
        ));
    }

    /// Check that `r` points within `insts` and was defined in an earlier block
    /// (or earlier in this block). `NO_JIR_REF` is OK iff `optional`.
    fn check_ref(&mut self, r: JirRef, optional: bool, site: JirRef, what: &str) {
        if r == NO_JIR_REF {
            if !optional {
                self.err(site, format!("required operand `{what}` is kNoJirRef"));
            }
            return;
        }
        if r as usize >= self.jfn.insts.len() {
            let max = self.jfn.insts.len() - 1;
            self.err(
                site,
                format!("operand `{what}` ref {r} out of bounds (max {max})"),
            );
            return;
        }
        if !self.defined[r as usize] {
            self.err(
                site,
                format!("operand `{what}` ref {r} used before its defining block"),
            );
        }
    }

    fn check_block_ref(&mut self, b: JirBlockRef, site: JirRef, what: &str) {
        if b == NO_JIR_BLOCK {
            self.err(site, format!("block ref `{what}` is null"));
            return;
        }
        if b as usize >= self.jfn.blocks.len() {
            let max = self.jfn.blocks.len() - 1;
            self.err(
                site,
                format!("block ref `{what}` {b} out of bounds (max {max})"),
            );
        }
    }

    fn check_extra_slice(&mut self, start: JirExtraIdx, len: usize, site: JirRef, what: &str) {
        if start as usize + len > self.jfn.extra.len() {
            let end = start as usize + len;
            let pool = self.jfn.extra.len();
            self.err(
                site,
                format!(
                    "extra slice `{what}` overflows: needs [{start}..{end}) but pool size is {pool}"
                ),
            );
        }
    }

    /// Type-consistency: binary-op / comparison operands should have agreeing
    /// types. Compared structurally (by `TypeKey`) because the same logical type
    /// can be interned at multiple indices; opaque aggregate/pointer kinds
    /// bypass to avoid false positives. No-op without a `TypePool`.
    fn ref_types_match(&self, a: JirRef, b: JirRef) -> bool {
        let types = match self.types {
            Some(t) => t,
            None => return true,
        };
        if a as usize >= self.jfn.insts.len() || b as usize >= self.jfn.insts.len() {
            return true;
        }
        let ta = self.resolve_ty(self.jfn.insts[a as usize].ty);
        let tb = self.resolve_ty(self.jfn.insts[b as usize].ty);
        if ta == TypeIdx::NONE || tb == TypeIdx::NONE {
            return true;
        }
        if ta == tb {
            return true;
        }
        let ka = types.get(ta);
        let kb = types.get(tb);
        if ka == kb {
            return true;
        }
        let is_opaque = |k: TypeKind| {
            matches!(
                k,
                TypeKind::Named
                    | TypeKind::Struct
                    | TypeKind::Enum
                    | TypeKind::Slice
                    | TypeKind::Array
                    | TypeKind::PtrSingle
                    | TypeKind::PtrMany
            )
        };
        if is_opaque(ka.kind) || is_opaque(kb.kind) {
            return true;
        }
        false
    }

