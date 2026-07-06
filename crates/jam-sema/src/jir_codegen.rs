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

/// Emit the LLVM function declaration (signature + linkage + ABI attributes).
/// Run before any body so cross-function forward references resolve.
pub fn jir_declare_prototype<'ctx>(
    jfn: &JirFunction,
    ctx: &CodegenContext<'ctx>,
) -> Result<Function<'ctx>, String> {
    let rabi = jir_classify_return(jfn, ctx)?;
    let sret = rabi.kind == ReturnAbiKind::Indirect;

    let mut arg_types = Vec::with_capacity(jfn.param_types.len() + sret as usize);
    if sret {
        // Leading `ptr` carries the caller-owned return slot (sret attr below).
        arg_types.push(ctx.context().pointer_type(0));
    }
    for i in 0..jfn.param_types.len() {
        let pabi = jir_classify_param(jfn, i, ctx)?;
        match pabi.kind {
            ParamAbiKind::ByPointer => arg_types.push(ctx.context().pointer_type(0)),
            ParamAbiKind::ByValue => {
                arg_types.push(pabi.llvm_type.ok_or("ByValue param missing LLVM type")?)
            }
        }
    }

    let ret_type = if sret {
        ctx.context().void_type()
    } else {
        rabi.direct_type.ok_or("Direct return missing LLVM type")?
    };
    let ft = ret_type.fn_type(&arg_types, jfn.is_var_args);
    let f = ctx.module().add_function(&jfn.name, ft);
    f.apply_default_attrs(jfn.is_extern);
    if jfn.return_type == builtin::NORETURN {
        f.set_no_return();
    }
    if sret {
        f.add_param_attr_sret(0, ctx.get_llvm_type(jfn.return_type)?, rabi.sret_align);
    }

    let external_linkage = jfn.is_extern || jfn.is_export || jfn.name == "main";
    if external_linkage {
        f.set_linkage(Linkage::External);
        // `export` symbols exist for C callers the optimizer can't see — pin
        // them in llvm.used so the internalize/global-DCE pass spares them.
        if jfn.is_export {
            append_to_used(ctx.module(), f);
        }
        f.set_call_conv(CallConv::C);
        // C ABI requires bool args/returns to be zero-extended to the register
        // width. Internal-linkage callers use our own ABI, so we skip zext.
        let arg_offset = if sret { 1u32 } else { 0 };
        for i in 0..jfn.param_types.len() {
            if jfn.param_types[i] == builtin::BOOL {
                f.add_param_attr_zeroext(i as u32 + arg_offset);
            }
        }
        if !sret && jfn.return_type == builtin::BOOL {
            f.add_ret_attr_zeroext();
        }
    } else {
        f.set_linkage(Linkage::Internal);
    }
    Ok(f)
}

// ---- body lowering ----

/// Per-function lowering state: `JirRef -> LLVM Value` (dataflow) and
/// `JirBlockRef -> LLVM BasicBlock` (terminators).
struct JirCodegenCtx<'a, 'ctx> {
    jfn: &'a JirFunction,
    ctx: &'a CodegenContext<'ctx>,
    value_map: HashMap<JirRef, Value<'ctx>>,
    block_map: HashMap<JirBlockRef, BasicBlock<'ctx>>,
}

