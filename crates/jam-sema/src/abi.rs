/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! ABI classification — how a parameter is passed and how a return value is
//! communicated at the LLVM-IR level.
//!
//! [`is_by_ref`] is the load-bearing verdict: it decides whether a type's
//! runtime form lives in memory (referred to by pointer in SSA) or rides in
//! registers. Every load/store/field-access and the param/return classification
//! pivot on it, so caller and callee MUST consult it identically.
//!
//! The `GenericCall` / `ArrayExpr` / type-alias resolution paths route through
//! the context's deferred hooks (which return `NONE` until the substitution
//! engine lands); for non-generic, non-aliased types — the common case — they
//! are never the deciding path.

use jam_core::index::{StringIdx, TypeIdx};
use jam_core::param_mode::ParamMode;
use jam_llvm::Type;
use jam_syntax::ast_flat::TypeKind;

use crate::codegen_context::CodegenContext;

/// How a single parameter is passed at the LLVM-IR level.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ParamAbiKind {
    /// A single LLVM argument of the parameter's natural representation.
    #[default]
    ByValue,
    /// A single `ptr align A` argument to caller-owned storage.
    ByPointer,
}

/// The ABI of one parameter.
#[derive(Copy, Clone, Debug, Default)]
pub struct ParamAbi<'ctx> {
    pub kind: ParamAbiKind,
    /// `ByValue`: the parameter's LLVM type. `None` for `ByPointer`.
    pub llvm_type: Option<Type<'ctx>>,
    /// `ByPointer`: pointee alignment in bytes. 0 for `ByValue`.
    pub pointer_align: u32,
}

/// How a function's return value is communicated.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ReturnAbiKind {
    /// Returned as a single LLVM value.
    #[default]
    Direct,
    /// Caller passes a leading `ptr sret(%T) align A`; callee returns `void`.
    Indirect,
}

/// The ABI of a function's return.
#[derive(Copy, Clone, Debug, Default)]
pub struct ReturnAbi<'ctx> {
    pub kind: ReturnAbiKind,
    /// `Direct`: the LLVM return type. `None` for `Indirect`.
    pub direct_type: Option<Type<'ctx>>,
    /// `Indirect`: pointee alignment in bytes. 0 for `Direct`.
    pub sret_align: u32,
}

/// Is this type carried by-reference at the codegen level? Aggregates
/// (arrays / structs / unions / payloaded enums) are; scalars, pointers,
/// slices (`{ptr,len}` register pair), fn-pointers, and unit-only enums are not.
pub fn is_by_ref(ty: TypeIdx, ctx: &CodegenContext) -> bool {
    if ty.is_none() {
        return false;
    }
    // Resolve a bare imported-module type to its qualified registry name so the
    // struct/union/enum registry checks below hit (no-op for the entry module).
    let ty = ctx.requalify_type(ty, &ctx.current_body_module());
    let k = ctx.type_pool.get(ty);
    match k.kind {
        // Source-only / comptime-only: never lowered to runtime memory.
        TypeKind::Invalid
        | TypeKind::Void
        | TypeKind::NoReturn
        | TypeKind::Type
        | TypeKind::Module => false,
        // Generic-call / deferred-array: resolve and recurse on the
        // instantiated shape.
        TypeKind::GenericCall => {
            let resolved = ctx.resolve_generic_call(ty);
            !resolved.is_none() && resolved != ty && is_by_ref(resolved, ctx)
        }
        TypeKind::ArrayExpr => {
            let resolved = ctx.resolve_array_expr(ty);
            !resolved.is_none() && resolved != ty && is_by_ref(resolved, ctx)
        }
        // Scalars + pointer-shaped values: byval.
        TypeKind::Bool
        | TypeKind::Int
        | TypeKind::Float
        | TypeKind::PtrSingle
        | TypeKind::PtrMany
        | TypeKind::Fn => false,
        // Slice is a `{ptr, len}` aggregate but treated as a register pair.
        TypeKind::Slice => false,
        // Aggregates: byref.
        TypeKind::Array | TypeKind::Struct | TypeKind::Union => true,
        // Unit-only enums lower to a bare i8 tag (byval); payloaded enums are
        // `{tag, payload}` aggregates (byref).
        TypeKind::Enum => ctx.enum_has_payload(ty).unwrap_or(false),
        // `Named` is a deferred reference into the registries.
        TypeKind::Named => {
            if ctx.is_struct_registered(ty) || ctx.is_union_registered(ty) {
                return true;
            }
            if let Some(has_payload) = ctx.enum_has_payload(ty) {
                return has_payload;
            }
            let name =
                String::from_utf8_lossy(&ctx.string_pool.get(StringIdx::new(k.a))).into_owned();
            let alias = ctx.lookup_type_alias(&name);
            if !alias.is_none() && alias != ty {
                return is_by_ref(alias, ctx);
            }
            // Unresolved Named with no registry hit — treat as byval (the
            // existing failure surface).
            false
        }
    }
}

/// One coerced 64-bit word of a small aggregate crossing the C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CWord {
    Int,
    Fp,
}

/// C-ABI class of a type crossing an indirect C call. `Direct` covers
/// scalars, pointers, slices and unit enums, whose natural LLVM types
/// already land in the right registers. The rest is per-arch:
///
///   AArch64 (AAPCS64): HFAs of <= 4 same-width floats go in v0..v3 as
///   their natural struct type; other aggregates <= 16 bytes coerce to
///   one or two GP words; bigger ones pass as a pointer to a caller
///   copy and return through sret (x8).
///
///   x86-64 (SysV): no HFA rule — aggregates <= 16 bytes classify per
///   eightbyte (SSE when every leaf overlapping the word is a float,
///   INTEGER otherwise), coercing to i64/double words; bigger ones go
///   on the stack (byval) and return through sret (rdi).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CAbiClass {
    Direct,
    /// AArch64 only: homogeneous float aggregate, natural struct value.
    Hfa,
    /// <= 16 bytes, coerced to one or two 64-bit words.
    Coerce { w0: CWord, w1: Option<CWord> },
    /// > 16 bytes. Args: caller copy — a plain pointer on aarch64,
    /// byval on x86-64. Returns: sret on both.
    Memory { byval: bool },
}

pub fn classify_c_abi(ty: TypeIdx, ctx: &CodegenContext) -> Result<CAbiClass, String> {
    if !is_by_ref(ty, ctx) {
        return Ok(CAbiClass::Direct);
    }
    match ctx.target_arch() {
        crate::target::Arch::X86_64 => {
            let size = ctx.type_size(ty)?;
            if size <= 16 {
                let w = sysv_words(ty, ctx)?;
                let w0 = w[0].unwrap_or(CWord::Int);
                let w1 = if size > 8 { Some(w[1].unwrap_or(CWord::Int)) } else { None };
                return Ok(CAbiClass::Coerce { w0, w1 });
            }
            Ok(CAbiClass::Memory { byval: true })
        }
        // AArch64 rules for everything else — the only other arch today.
        _ => {
            if let Some((_bits, leaves)) = c_abi_float_leaves(ty, ctx) {
                if (1..=4).contains(&leaves) {
                    return Ok(CAbiClass::Hfa);
                }
            }
            let size = ctx.type_size(ty)?;
            if size <= 16 {
                let w1 = if size > 8 { Some(CWord::Int) } else { None };
                return Ok(CAbiClass::Coerce { w0: CWord::Int, w1 });
            }
            Ok(CAbiClass::Memory { byval: false })
        }
    }
}

/// Walk an aggregate's leaf fields; `Some((bits, count))` when every leaf is
/// a float of one width — the homogeneous-float-aggregate shape AAPCS64
/// carries in FP registers. `None` for any other composition (unions
/// included: no dominant member to classify by).
fn c_abi_float_leaves(ty: TypeIdx, ctx: &CodegenContext) -> Option<(u32, u32)> {
    let k = ctx.type_pool.get(ty);
    match k.kind {
        TypeKind::Float => Some((k.a, 1)),
        TypeKind::Array => {
            let (bits, n) = c_abi_float_leaves(TypeIdx::new(k.a), ctx)?;
            Some((bits, n * k.b))
        }
        TypeKind::Struct | TypeKind::Named => {
            let fields = ctx.struct_fields(ty)?;
            let mut bits = 0u32;
            let mut count = 0u32;
            for (_, fty) in fields {
                let (fbits, fcount) = c_abi_float_leaves(fty, ctx)?;
                if bits == 0 {
                    bits = fbits;
                } else if bits != fbits {
                    return None; // mixed widths — not an HFA
                }
                count += fcount;
            }
            if count == 0 { None } else { Some((bits, count)) }
        }
        _ => None,
    }
}

/// SysV eightbyte classes of an aggregate <= 16 bytes: walk every leaf
/// with its byte offset (natural layout — align each field, elements at
/// their aligned stride) and mark the 8-byte words it spans. A word is
/// SSE (`Fp`) only when floats are all it ever sees; INTEGER wins any mix.
fn sysv_words(ty: TypeIdx, ctx: &CodegenContext) -> Result<[Option<CWord>; 2], String> {
    let mut w = [None, None];
    sysv_walk(ty, ctx, 0, &mut w)?;
    Ok(w)
}

fn sysv_mark(w: &mut [Option<CWord>; 2], base: u64, size: u64, cls: CWord) {
    if size == 0 {
        return;
    }
    for word in (base / 8)..=((base + size - 1) / 8) {
        if (word as usize) < 2 {
            let slot = &mut w[word as usize];
            *slot = Some(match (*slot, cls) {
                (Some(CWord::Int), _) | (_, CWord::Int) => CWord::Int,
                _ => CWord::Fp,
            });
        }
    }
}

fn sysv_walk(
    ty: TypeIdx,
    ctx: &CodegenContext,
    base: u64,
    w: &mut [Option<CWord>; 2],
) -> Result<(), String> {
    let ty = ctx.requalify_type(ty, &ctx.current_body_module());
    let k = ctx.type_pool.get(ty);
    let align_up = |off: u64, a: u64| -> u64 {
        if a == 0 { off } else { off.div_ceil(a) * a }
    };
    match k.kind {
        TypeKind::Float => {
            sysv_mark(w, base, (k.a as u64) / 8, CWord::Fp);
            Ok(())
        }
        TypeKind::Array => {
            let elem = TypeIdx::new(k.a);
            let stride = align_up(ctx.type_size(elem)?, ctx.type_align(elem)?);
            for i in 0..(k.b as u64) {
                sysv_walk(elem, ctx, base + i * stride, w)?;
            }
            Ok(())
        }
        TypeKind::Struct | TypeKind::Named if ctx.struct_fields(ty).is_some() => {
            let mut off = base;
            for (_, fty) in ctx.struct_fields(ty).unwrap() {
                off = align_up(off, ctx.type_align(fty)?);
                sysv_walk(fty, ctx, off, w)?;
                off += ctx.type_size(fty)?;
            }
            Ok(())
        }
        TypeKind::Union | TypeKind::Named if ctx.union_fields(ty).is_some() => {
            for (_, fty) in ctx.union_fields(ty).unwrap() {
                sysv_walk(fty, ctx, base, w)?;
            }
            Ok(())
        }
        TypeKind::Slice => {
            sysv_mark(w, base, 16, CWord::Int);
            Ok(())
        }
        // Ints, bools, pointers, fn pointers, enums (tagged payloads are
        // jam-internal, never a real C type) — and any unresolved leaf —
        // conservatively INTEGER over the type's span.
        _ => {
            sysv_mark(w, base, ctx.type_size(ty)?, CWord::Int);
            Ok(())
        }
    }
}

/// Classify a parameter `(mode, type)` pair. `mut` always carries a pointer to
/// caller-owned storage regardless of the type's classification.
pub fn classify_param<'ctx>(
    mode: ParamMode,
    ty: TypeIdx,
    ctx: &CodegenContext<'ctx>,
) -> Result<ParamAbi<'ctx>, String> {
    if mode == ParamMode::Mut || is_by_ref(ty, ctx) {
        Ok(ParamAbi {
            kind: ParamAbiKind::ByPointer,
            llvm_type: None,
            pointer_align: ctx.type_align(ty)? as u32,
        })
    } else {
        Ok(ParamAbi {
            kind: ParamAbiKind::ByValue,
            llvm_type: Some(ctx.get_llvm_type(ty)?),
            pointer_align: 0,
        })
    }
}

/// Classify a return type. `void`/`NONE` returns `Direct void`; byref aggregates
/// return `Indirect` (sret); everything else `Direct`.
pub fn classify_return<'ctx>(
    ty: TypeIdx,
    ctx: &CodegenContext<'ctx>,
) -> Result<ReturnAbi<'ctx>, String> {
    if ty.is_none() {
        return Ok(ReturnAbi {
            kind: ReturnAbiKind::Direct,
            direct_type: Some(ctx.context().void_type()),
            sret_align: 0,
        });
    }
    if is_by_ref(ty, ctx) {
        return Ok(ReturnAbi {
            kind: ReturnAbiKind::Indirect,
            direct_type: None,
            sret_align: ctx.type_align(ty)? as u32,
        });
    }
    Ok(ReturnAbi {
        kind: ReturnAbiKind::Direct,
        direct_type: Some(ctx.get_llvm_type(ty)?),
        sret_align: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_context::EnumVariantInfo;
    use jam_llvm::Context;
    use jam_syntax::ast_flat::builtin;

    #[test]
    fn scalars_pointers_slices_are_byval() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        assert!(!is_by_ref(builtin::I32, &cg));
        assert!(!is_by_ref(builtin::F64, &cg));
        assert!(!is_by_ref(builtin::BOOL, &cg));
        let p = cg.type_pool.intern_ptr_single(builtin::I32);
        assert!(!is_by_ref(p, &cg));
        let s = cg.type_pool.intern_slice(builtin::U8);
        assert!(!is_by_ref(s, &cg));
    }

    #[test]
    fn arrays_and_structs_are_byref() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let a = cg.type_pool.intern_array(builtin::I32, 4);
        assert!(is_by_ref(a, &cg));

        let nid = cg.string_pool.intern(b"Point");
        let sty = cg.type_pool.intern_struct(nid);
        let llvm = ctx.struct_type(&[ctx.i32_type(), ctx.i32_type()], false);
        cg.register_struct("Point", llvm, vec![("x".into(), builtin::I32)]);
        assert!(is_by_ref(sty, &cg));
        // Named referencing the same struct also classifies byref.
        let named = cg.type_pool.intern_named(nid);
        assert!(is_by_ref(named, &cg));
    }

    #[test]
    fn unit_enum_byval_payload_enum_byref() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        // unit-only enum -> byval
        let uid = cg.string_pool.intern(b"Color");
        let unamed = cg.type_pool.intern_named(uid);
        cg.register_enum(
            "Color",
            vec![EnumVariantInfo {
                name: "Red".into(),
                payload_types: vec![],
                discriminant: 0,
            }],
        );
        assert!(!is_by_ref(unamed, &cg));

        // payloaded enum -> byref
        let pid = cg.string_pool.intern(b"Opt");
        let pnamed = cg.type_pool.intern_named(pid);
        cg.register_enum(
            "Opt",
            vec![EnumVariantInfo {
                name: "Some".into(),
                payload_types: vec![builtin::I32],
                discriminant: 1,
            }],
        );
        let payload_llvm = ctx.struct_type(&[ctx.i8_type(), ctx.i32_type()], false);
        cg.set_enum_llvm_type("Opt", payload_llvm, 4, 4, true);
        assert!(is_by_ref(pnamed, &cg));
    }

    #[test]
    fn classify_param_byvalue_bypointer() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        // let i32 -> ByValue with the i32 LLVM type
        let p = classify_param(ParamMode::Let, builtin::I32, &cg).unwrap();
        assert_eq!(p.kind, ParamAbiKind::ByValue);
        assert!(p.llvm_type.unwrap().is_integer());
        // mut i32 -> ByPointer, align 4
        let m = classify_param(ParamMode::Mut, builtin::I32, &cg).unwrap();
        assert_eq!(m.kind, ParamAbiKind::ByPointer);
        assert_eq!(m.pointer_align, 4);
        // let <struct> -> ByPointer (byref)
        let nid = cg.string_pool.intern(b"S");
        let sty = cg.type_pool.intern_struct(nid);
        let llvm = ctx.struct_type(&[ctx.i64_type()], false);
        cg.register_struct("S", llvm, vec![("a".into(), builtin::I64)]);
        let s = classify_param(ParamMode::Let, sty, &cg).unwrap();
        assert_eq!(s.kind, ParamAbiKind::ByPointer);
        assert_eq!(s.pointer_align, 8);
    }

    #[test]
    fn classify_return_direct_void_indirect() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        // void
        let v = classify_return(TypeIdx::NONE, &cg).unwrap();
        assert_eq!(v.kind, ReturnAbiKind::Direct);
        assert!(v.direct_type.unwrap().is_void());
        // i32 direct
        let d = classify_return(builtin::I32, &cg).unwrap();
        assert_eq!(d.kind, ReturnAbiKind::Direct);
        assert!(d.direct_type.unwrap().is_integer());
        // struct -> indirect sret
        let nid = cg.string_pool.intern(b"S");
        let sty = cg.type_pool.intern_struct(nid);
        let llvm = ctx.struct_type(&[ctx.i64_type()], false);
        cg.register_struct("S", llvm, vec![("a".into(), builtin::I64)]);
        let s = classify_return(sty, &cg).unwrap();
        assert_eq!(s.kind, ReturnAbiKind::Indirect);
        assert_eq!(s.sret_align, 8);
    }
}
