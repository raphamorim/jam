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

    fn check_inst(&mut self, r: JirRef) {
        let inst = self.jfn.insts[r as usize];
        match inst.tag {
            JirTag::Invalid => {
                self.err(r, "Invalid tag in live instruction stream".to_string());
            }
            // Constants / placeholders / param access — no operand refs.
            JirTag::Poison
            | JirTag::Int
            | JirTag::Float
            | JirTag::Bool
            | JirTag::Str
            | JirTag::Param
            | JirTag::SretArg
            | JirTag::Unreachable
            | JirTag::Alloca => {}
            // Single-operand instructions (`a` is a ref).
            JirTag::Load
            | JirTag::Deref
            | JirTag::AddrOf
            | JirTag::FNeg
            | JirTag::BitNot
            | JirTag::LogNot
            | JirTag::ZExt
            | JirTag::SExt
            | JirTag::Trunc
            | JirTag::SIToFP
            | JirTag::UIToFP
            | JirTag::FPToSI
            | JirTag::FPToUI
            | JirTag::FPExt
            | JirTag::FPTrunc
            | JirTag::BitCast
            | JirTag::PtrToInt
            | JirTag::IntToPtr
            | JirTag::FieldAccess
            | JirTag::ExtractValue
            | JirTag::FieldAddr
            | JirTag::EnumPayload => {
                self.check_ref(inst.a, false, r, "a");
            }
            // `a` is a StringIdx, not a ref — codegen fails loudly on a missing
            // symbol.
            JirTag::FnRef => {}
            JirTag::DropBinding => {
                self.check_ref(inst.a, false, r, "a");
                // `b` is a StringIdx for the drop-fn's LLVM symbol. An empty
                // symbol means astgen's lookupDropFn returned a default string.
                if let Some(strings) = self.strings {
                    let sid = StringIdx::new(inst.b);
                    if sid.is_none() || strings.get(sid).is_empty() {
                        self.err(
                            r,
                            "DropBinding with empty / unresolved drop symbol".to_string(),
                        );
                    }
                }
            }
            JirTag::Store => {
                self.check_ref(inst.a, false, r, "a");
                self.check_ref(inst.b, false, r, "b");
            }
            // Same-type binary ops + comparisons: lhs and rhs must agree.
            JirTag::Add
            | JirTag::Sub
            | JirTag::Mul
            | JirTag::SDiv
            | JirTag::UDiv
            | JirTag::SRem
            | JirTag::URem
            | JirTag::FAdd
            | JirTag::FSub
            | JirTag::FMul
            | JirTag::FDiv
            | JirTag::FRem
            | JirTag::ICmpEq
            | JirTag::ICmpNe
            | JirTag::ICmpSlt
            | JirTag::ICmpSle
            | JirTag::ICmpSgt
            | JirTag::ICmpSge
            | JirTag::ICmpUlt
            | JirTag::ICmpUle
            | JirTag::ICmpUgt
            | JirTag::ICmpUge
            | JirTag::FCmpOeq
            | JirTag::FCmpOne
            | JirTag::FCmpOlt
            | JirTag::FCmpOle
            | JirTag::FCmpOgt
            | JirTag::FCmpOge
            | JirTag::BitAnd
            | JirTag::BitOr
            | JirTag::BitXor => {
                self.check_ref(inst.a, false, r, "a");
                self.check_ref(inst.b, false, r, "b");
                if !self.ref_types_match(inst.a, inst.b) {
                    let aty = self.jfn.insts[inst.a as usize].ty.raw();
                    let bty = self.jfn.insts[inst.b as usize].ty.raw();
                    self.err(
                        r,
                        format!(
                            "operand type mismatch for {} (a.ty={} b.ty={})",
                            inst.tag.name(),
                            aty,
                            bty
                        ),
                    );
                }
            }
            // Shifts: amount may be a narrower int — no match check.
            JirTag::Shl | JirTag::AShr | JirTag::LShr => {
                self.check_ref(inst.a, false, r, "a");
                self.check_ref(inst.b, false, r, "b");
            }
            // Index ops: base + (integer) index of possibly different type.
            JirTag::Index | JirTag::IndexAddr => {
                self.check_ref(inst.a, false, r, "a");
                self.check_ref(inst.b, false, r, "b");
            }
            JirTag::Ret => {
                self.check_ref(inst.a, true, r, "a");
            }
            JirTag::Br => {
                self.check_block_ref(inst.a, r, "target");
            }
            JirTag::CondBr => {
                self.check_ref(inst.a, false, r, "cond");
                self.check_extra_slice(inst.b, 2, r, "branches");
                if inst.b as usize + 2 <= self.jfn.extra.len() {
                    let then_b = self.jfn.extra[inst.b as usize];
                    let else_b = self.jfn.extra[inst.b as usize + 1];
                    self.check_block_ref(then_b, r, "then");
                    self.check_block_ref(else_b, r, "else");
                }
            }
            JirTag::Switch => {
                self.check_ref(inst.a, false, r, "scrut");
                if inst.b as usize + 2 > self.jfn.extra.len() {
                    self.err(r, "Switch extra header out of bounds".to_string());
                    return;
                }
                let default_b = self.jfn.extra[inst.b as usize];
                let case_count = self.jfn.extra[inst.b as usize + 1];
                self.check_block_ref(default_b, r, "default");
                self.check_extra_slice(inst.b, 2 + case_count as usize * 4, r, "switch");
                for i in 0..case_count {
                    let idx = inst.b as usize + 2 + i as usize * 4 + 3;
                    if idx < self.jfn.extra.len() {
                        let case_b = self.jfn.extra[idx];
                        self.check_block_ref(case_b, r, "case-block");
                    }
                }
            }
            JirTag::Call => {
                // `a` is StringIdx (callee), `b` is the extra offset of
                // [argCount, arg0_ref, ...].
                if inst.b as usize >= self.jfn.extra.len() {
                    self.err(r, "Call extra index out of bounds".to_string());
                    return;
                }
                let arg_count = self.jfn.extra[inst.b as usize];
                self.check_extra_slice(inst.b, 1 + arg_count as usize, r, "call-args");
                // The C++ reads `extra[b + 1 + i]` unconditionally, relying on
                // the `check_extra_slice` above to have fired for a malformed
                // (overflowing) slice — but it still performs the OOB read,
                // which is UB. Rust can't, so the per-element bounds guard here
                // (and in CallIndirect / Switch / aggregate-lits below) avoids
                // the panic. For well-formed input the slice always fits, so
                // behaviour is identical; for malformed input the overflow
                // diagnostic has already been emitted.
                for i in 0..arg_count {
                    let idx = inst.b as usize + 1 + i as usize;
                    if idx < self.jfn.extra.len() {
                        let ar = self.jfn.extra[idx];
                        self.check_ref(ar, false, r, "call-arg");
                    }
                }
            }
            JirTag::CallIndirect => {
                self.check_ref(inst.a, false, r, "callee");
                if inst.b as usize >= self.jfn.extra.len() {
                    self.err(r, "CallIndirect extra index out of bounds".to_string());
                    return;
                }
                let arg_count = self.jfn.extra[inst.b as usize];
                self.check_extra_slice(inst.b, 1 + arg_count as usize, r, "call-args");
                for i in 0..arg_count {
                    let idx = inst.b as usize + 1 + i as usize;
                    if idx < self.jfn.extra.len() {
                        let ar = self.jfn.extra[idx];
                        self.check_ref(ar, false, r, "call-arg");
                    }
                }
            }
            JirTag::StructLit | JirTag::ArrayLit => {
                if inst.b as usize >= self.jfn.extra.len() {
                    self.err(r, "aggregate-lit extra index out of bounds".to_string());
                    return;
                }
                let count = self.jfn.extra[inst.b as usize];
                self.check_extra_slice(inst.b, 1 + count as usize, r, "agg-fields");
                for i in 0..count {
                    let idx = inst.b as usize + 1 + i as usize;
                    if idx < self.jfn.extra.len() {
                        let fr = self.jfn.extra[idx];
                        self.check_ref(fr, false, r, "agg-field");
                    }
                }
            }
            JirTag::MakeSlice => {
                self.check_ref(inst.a, false, r, "slice-ptr");
                self.check_ref(inst.b, false, r, "slice-len");
            }
            JirTag::MemSet => {
                // `a` = dest ptr (ref); `b` = byte length (raw constant).
                self.check_ref(inst.a, false, r, "memset-dst");
            }
        }
    }

    fn run(&mut self) {
        let jfn = self.jfn;

        // Pass 1: allocas live exclusively in the entry block (block 1).
        for b in 1..jfn.blocks.len() {
            if b == 1 {
                continue;
            }
            for &r in &jfn.blocks[b].insts {
                if (r as usize) < jfn.insts.len() && jfn.insts[r as usize].tag == JirTag::Alloca {
                    self.block_err(
                        b as JirBlockRef,
                        "Alloca outside entry block — use emitAllocaHoisted",
                    );
                }
            }
        }

        // Pass 2: walk blocks in order, marking each instruction defined as we
        // step over it; check operands against already-defined slots and verify
        // terminator placement.
        for b in 1..jfn.blocks.len() {
            let insts = jfn.blocks[b].insts.clone();
            if insts.is_empty() {
                self.block_err(b as JirBlockRef, "empty block (no terminator)");
                continue;
            }
            for (i, &r) in insts.iter().enumerate() {
                if r == NO_JIR_REF || r as usize >= jfn.insts.len() {
                    self.block_err(
                        b as JirBlockRef,
                        &format!("instruction list contains invalid ref {r}"),
                    );
                    continue;
                }
                self.check_inst(r);
                self.defined[r as usize] = true;

                let is_last = i + 1 == insts.len();
                let term = jfn.insts[r as usize].tag.is_terminator();
                let tag = jfn.insts[r as usize].tag.name();
                if term && !is_last {
                    self.block_err(
                        b as JirBlockRef,
                        &format!("terminator {tag} is not last in block"),
                    );
                }
                if !term && is_last {
                    self.block_err(
                        b as JirBlockRef,
                        &format!("block ends with non-terminator {tag}"),
                    );
                }
            }
        }
    }
}

/// Verify a finished [`JirFunction`]. Returns the (possibly empty) list of
/// structural diagnostics. Each diagnostic's `loc.line` carries the offending
/// instruction's `src_line` (file left empty for the caller to stamp).
pub fn verify_jir_function(
    jfn: &JirFunction,
    types: Option<&TypePool>,
    strings: Option<&StringPool>,
    resolver: Option<JirVerifyResolver>,
) -> Vec<Diagnostic> {
    let mut v = Verifier::new(jfn, types, strings, resolver);
    v.run();
    v.diags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jir::{JirInst, JirTag};

    /// Build a minimal well-formed function: entry block with `ret void`.
    fn ret_void_fn() -> JirFunction {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        let ret = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.push(ret);
        f
    }

    #[test]
    fn well_formed_function_has_no_diagnostics() {
        let f = ret_void_fn();
        let d = verify_jir_function(&f, None, None, None);
        assert!(d.is_empty(), "unexpected: {:?}", d);
    }

    #[test]
    fn empty_block_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        f.push_block("entry"); // left empty
        let d = verify_jir_function(&f, None, None, None);
        assert_eq!(d.len(), 1);
        assert!(d[0].message.contains("empty block (no terminator)"));
        assert!(d[0].message.contains("fn `f`"));
    }

    #[test]
    fn block_without_terminator_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        // A non-terminator as the last (and only) instruction.
        let b = f.push_inst(JirInst {
            tag: JirTag::Bool,
            a: 1,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.push(b);
        let d = verify_jir_function(&f, None, None, None);
        assert!(
            d.iter()
                .any(|x| x.message.contains("block ends with non-terminator Bool"))
        );
    }

    #[test]
    fn terminator_not_last_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        let r1 = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        let r2 = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.extend([r1, r2]);
        let d = verify_jir_function(&f, None, None, None);
        assert!(
            d.iter()
                .any(|x| x.message.contains("terminator Ret is not last in block"))
        );
    }

    #[test]
    fn out_of_bounds_operand_ref_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        // Load from a ref far past the end of `insts`.
        let load = f.push_inst(JirInst {
            tag: JirTag::Load,
            a: 999,
            ..Default::default()
        });
        let ret = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.extend([load, ret]);
        let d = verify_jir_function(&f, None, None, None);
        assert!(d.iter().any(|x| x.message.contains("out of bounds")));
    }

    #[test]
    fn use_before_def_across_blocks_is_rejected() {
        // Entry uses a value defined only in a later block.
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        let later = f.push_block("later");
        // value defined in `later`
        let val = f.push_inst(JirInst {
            tag: JirTag::Bool,
            a: 1,
            ty: TypeIdx::NONE,
            ..Default::default()
        });
        // entry: Load uses `val` (defined later), then Br to later
        let load = f.push_inst(JirInst {
            tag: JirTag::Load,
            a: val,
            ..Default::default()
        });
        let br = f.push_inst(JirInst {
            tag: JirTag::Br,
            a: later,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.extend([load, br]);
        // later: define val, then ret
        let ret = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(later).insts.extend([val, ret]);
        let d = verify_jir_function(&f, None, None, None);
        assert!(
            d.iter()
                .any(|x| x.message.contains("used before its defining block"))
        );
    }

    #[test]
    fn alloca_outside_entry_block_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        let other = f.push_block("other");
        let br = f.push_inst(JirInst {
            tag: JirTag::Br,
            a: other,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.push(br);
        let alloca = f.push_inst(JirInst {
            tag: JirTag::Alloca,
            ..Default::default()
        });
        let ret = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(other).insts.extend([alloca, ret]);
        let d = verify_jir_function(&f, None, None, None);
        assert!(
            d.iter()
                .any(|x| x.message.contains("Alloca outside entry block"))
        );
    }

    #[test]
    fn br_to_missing_block_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        let br = f.push_inst(JirInst {
            tag: JirTag::Br,
            a: 42,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.push(br);
        let d = verify_jir_function(&f, None, None, None);
        assert!(d.iter().any(
            |x| x.message.contains("block ref `target`") && x.message.contains("out of bounds")
        ));
    }

    #[test]
    fn call_arg_slice_overflow_is_rejected() {
        let mut f = JirFunction::new();
        f.name = "f".to_string();
        let entry = f.push_block("entry");
        // extra = [argCount=3] but no actual args -> slice overflow.
        let ex = f.push_extra(&[3]);
        let call = f.push_inst(JirInst {
            tag: JirTag::Call,
            a: 1,
            b: ex,
            ..Default::default()
        });
        let ret = f.push_inst(JirInst {
            tag: JirTag::Ret,
            ..Default::default()
        });
        f.get_block_mut(entry).insts.extend([call, ret]);
        let d = verify_jir_function(&f, None, None, None);
        assert!(
            d.iter()
                .any(|x| x.message.contains("call-args") && x.message.contains("overflows"))
        );
    }
}
