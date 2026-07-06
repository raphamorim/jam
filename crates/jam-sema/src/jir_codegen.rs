/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! JIR codegen — walks a fully-typed [`JirFunction`] and emits LLVM IR. By
//! design this stage is mechanical: no type inference, no peer resolution. Each
//! `JirInst` maps to a few LLVM instructions; the `JirRef -> LLVM Value` map
//! carries the dataflow. Ported from `src/jir_codegen.{h,cpp}`.
//!
//! Two-step API (so forward references between functions resolve before any
//! body is lowered): [`jir_declare_prototype`] emits the LLVM signature, then
//! `jir_define_body` (next increment) lowers the instructions. The ABI
//! classifier here is the single source of truth for the signature — call sites
//! and arg lowering consult the same [`crate::abi`] routines, so caller and
//! callee can't disagree on by-value/by-pointer or direct/sret.
//!
//! ## Scope of this increment
//!
//! Prototype emission only: the function signature, linkage, calling
//! convention, and ABI attributes (sret / zeroext / noreturn). The 47-tag
//! `JirInst` → LLVM lowering (`jir_define_body` + `emit_inst`) — which needs a
//! broad jam_llvm Builder surface — lands next, validatable via `jir_verify`
//! on hand-built `JirFunction`s, then end-to-end against `--emit-jir`/`--emit-ir`.

use std::collections::HashMap;

use jam_core::index::{StringIdx, TypeIdx};
use jam_core::param_mode::ParamMode;
use jam_llvm::{
    BasicBlock, CallConv, Function, IntPredicate, Linkage, RealPredicate, Value, append_to_used,
};
use jam_syntax::ast_flat::{TypeKind, builtin};

use crate::abi::{
    ParamAbi, ParamAbiKind, ReturnAbi, ReturnAbiKind, classify_param, classify_return, is_by_ref,
};
use crate::codegen_context::CodegenContext;
use crate::jir::{JirBlockRef, JirFunction, JirInst, JirRef, JirTag, NO_JIR_REF};

/// Classify a JIR function's return. `extern`/`tfn` preserve the user-written
/// type verbatim (no sret); a test returns void; `kNoType` is void.
fn jir_classify_return<'ctx>(
    jfn: &JirFunction,
    ctx: &CodegenContext<'ctx>,
) -> Result<ReturnAbi<'ctx>, String> {
    if jfn.is_extern || jfn.is_test {
        let direct_type = if jfn.is_test || jfn.return_type.is_none() {
            Some(ctx.context().void_type())
        } else {
            Some(ctx.get_llvm_type(jfn.return_type)?)
        };
        return Ok(ReturnAbi {
            kind: ReturnAbiKind::Direct,
            direct_type,
            sret_align: 0,
        });
    }
    if jfn.return_type.is_none() {
        return Ok(ReturnAbi {
            kind: ReturnAbiKind::Direct,
            direct_type: Some(ctx.context().void_type()),
            sret_align: 0,
        });
    }
    classify_return(jfn.return_type, ctx)
}

/// Does the LLVM signature have a leading `ptr sret(%T)` argument?
pub fn jir_return_is_sret(jfn: &JirFunction, ctx: &CodegenContext) -> bool {
    matches!(
        jir_classify_return(jfn, ctx).map(|r| r.kind),
        Ok(ReturnAbiKind::Indirect)
    )
}

/// Classify param `i`. `extern` preserves the user-written type verbatim (the
/// FFI boundary is what the user wrote, e.g. `*const T` for an out-ptr).
fn jir_classify_param<'ctx>(
    jfn: &JirFunction,
    i: usize,
    ctx: &CodegenContext<'ctx>,
) -> Result<ParamAbi<'ctx>, String> {
    let t = jfn.param_types[i];
    if jfn.is_extern {
        return Ok(ParamAbi {
            kind: ParamAbiKind::ByValue,
            llvm_type: Some(ctx.get_llvm_type(t)?),
            pointer_align: 0,
        });
    }
    let mode = jfn.param_modes.get(i).copied().unwrap_or(ParamMode::Let);
    classify_param(mode, t, ctx)
}

