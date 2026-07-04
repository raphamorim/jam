/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! ABI classification — how a parameter is passed and how a return value is
//! communicated at the LLVM-IR level. Ported from `src/abi.{h,cpp}`.
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

