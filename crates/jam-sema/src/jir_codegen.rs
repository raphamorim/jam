/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! JIR codegen — walks a fully-typed [`JirFunction`] and emits LLVM IR. Purely
//! mechanical: no inference here, each `JirInst` maps to a few LLVM
//! instructions. [`jir_declare_prototype`] emits signatures first (so forward
//! references resolve), then [`jir_define_body`] lowers each body; both consult
//! the same [`crate::abi`] classifier, so caller and callee can't disagree on
//! by-value/by-pointer or direct/sret.

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

/// Look up the cached LLVM value for `r`, or emit it (recursing for
/// subexpressions) and cache. The caching is load-bearing: a JirRef referenced
/// across out-of-order block walks must lower to exactly one LLVM value, or
/// reads split from writes.
fn emit_inst<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    r: JirRef,
) -> Result<Option<Value<'ctx>>, String> {
    if r == NO_JIR_REF {
        return Ok(None);
    }
    if let Some(&v) = lctx.value_map.get(&r) {
        return Ok(Some(v));
    }
    let v = emit_inst_impl(lctx, r)?;
    if let Some(val) = v {
        lctx.value_map.insert(r, val);
    }
    Ok(v)
}

/// Emit both operands of a binary instruction (each must produce a value).
fn binop_operands<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
) -> Result<(Value<'ctx>, Value<'ctx>), String> {
    let a = emit_inst(lctx, inst.a)?.ok_or("binop lhs produced no value")?;
    let b = emit_inst(lctx, inst.b)?.ok_or("binop rhs produced no value")?;
    Ok((a, b))
}

fn emit_inst_impl<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    r: JirRef,
) -> Result<Option<Value<'ctx>>, String> {
    let inst = *lctx.jfn.get_inst(r);
    match inst.tag {
        JirTag::Poison => {
            let ty = if inst.ty.is_none() {
                lctx.ctx.context().void_type()
            } else {
                lctx.ctx.get_llvm_type(inst.ty)?
            };
            Ok(Some(ty.undef()))
        }
        JirTag::Int => {
            let val = (inst.a as u64) | ((inst.b as u64) << 32);
            let is_neg = inst.flags & 1 != 0;
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let materialized = if is_neg {
                (val as i64).wrapping_neg() as u64
            } else {
                val
            };
            Ok(Some(ty.const_int(materialized, is_neg)))
        }
        JirTag::Float => {
            let bits = (inst.a as u64) | ((inst.b as u64) << 32);
            let mut d = f64::from_bits(bits);
            if inst.flags & 1 != 0 {
                d = -d;
            }
            Ok(Some(lctx.ctx.get_llvm_type(inst.ty)?.const_real(d)))
        }
        JirTag::Bool => Ok(Some(
            lctx.ctx
                .context()
                .i1_type()
                .const_int(if inst.a != 0 { 1 } else { 0 }, false),
        )),

        // Integer arithmetic.
        JirTag::Add => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().add(a, b, "add")))
        }
        JirTag::Sub => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().sub(a, b, "sub")))
        }
        JirTag::Mul => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().mul(a, b, "mul")))
        }
        JirTag::SDiv => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().sdiv(a, b, "sdiv")))
        }
        JirTag::UDiv => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().udiv(a, b, "udiv")))
        }
        JirTag::SRem => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().srem(a, b, "srem")))
        }
        JirTag::URem => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().urem(a, b, "urem")))
        }

        // Float arithmetic.
        JirTag::FAdd => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().fadd(a, b, "fadd")))
        }
        JirTag::FSub => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().fsub(a, b, "fsub")))
        }
        JirTag::FMul => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().fmul(a, b, "fmul")))
        }
        JirTag::FDiv => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().fdiv(a, b, "fdiv")))
        }
        JirTag::FRem => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().frem(a, b, "frem")))
        }
        JirTag::FNeg => {
            let a = emit_inst(lctx, inst.a)?.ok_or("FNeg operand")?;
            Ok(Some(lctx.ctx.builder().fneg(a, "fneg")))
        }

        // Integer comparison.
        JirTag::ICmpEq => icmp(lctx, &inst, IntPredicate::Eq, "eq"),
        JirTag::ICmpNe => icmp(lctx, &inst, IntPredicate::Ne, "ne"),
        JirTag::ICmpSlt => icmp(lctx, &inst, IntPredicate::Slt, "slt"),
        JirTag::ICmpSle => icmp(lctx, &inst, IntPredicate::Sle, "sle"),
        JirTag::ICmpSgt => icmp(lctx, &inst, IntPredicate::Sgt, "sgt"),
        JirTag::ICmpSge => icmp(lctx, &inst, IntPredicate::Sge, "sge"),
        JirTag::ICmpUlt => icmp(lctx, &inst, IntPredicate::Ult, "ult"),
        JirTag::ICmpUle => icmp(lctx, &inst, IntPredicate::Ule, "ule"),
        JirTag::ICmpUgt => icmp(lctx, &inst, IntPredicate::Ugt, "ugt"),
        JirTag::ICmpUge => icmp(lctx, &inst, IntPredicate::Uge, "uge"),

        // Float comparison (ordered; note `FCmpOne` maps to LLVM `une`).
        JirTag::FCmpOeq => fcmp(lctx, &inst, RealPredicate::Oeq, "oeq"),
        JirTag::FCmpOne => fcmp(lctx, &inst, RealPredicate::Une, "one"),
        JirTag::FCmpOlt => fcmp(lctx, &inst, RealPredicate::Olt, "olt"),
        JirTag::FCmpOle => fcmp(lctx, &inst, RealPredicate::Ole, "ole"),
        JirTag::FCmpOgt => fcmp(lctx, &inst, RealPredicate::Ogt, "ogt"),
        JirTag::FCmpOge => fcmp(lctx, &inst, RealPredicate::Oge, "oge"),

        // Bitwise / shift.
        JirTag::BitAnd => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().and(a, b, "and")))
        }
        JirTag::BitOr => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().or(a, b, "or")))
        }
        JirTag::BitXor => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().xor(a, b, "xor")))
        }
        JirTag::Shl => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().shl(a, b, "shl")))
        }
        JirTag::AShr => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().ashr(a, b, "ashr")))
        }
        JirTag::LShr => {
            let (a, b) = binop_operands(lctx, &inst)?;
            Ok(Some(lctx.ctx.builder().lshr(a, b, "lshr")))
        }
        JirTag::BitNot => {
            // LLVM has no NOT; xor with all-ones of the operand's type.
            let v = emit_inst(lctx, inst.a)?.ok_or("BitNot operand")?;
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let ones = ty.const_int(!0u64, true);
            Ok(Some(lctx.ctx.builder().xor(v, ones, "not")))
        }
        JirTag::LogNot => {
            // Boolean inversion: xor with i1 1 (operand is already i1).
            let v = emit_inst(lctx, inst.a)?.ok_or("LogNot operand")?;
            let one = lctx.ctx.context().i1_type().const_int(1, false);
            Ok(Some(lctx.ctx.builder().xor(v, one, "lnot")))
        }

        // Type conversions.
        JirTag::ZExt => cast(lctx, &inst, "zext", |b, v, t| b.zext(v, t, "zext")),
        JirTag::SExt => cast(lctx, &inst, "sext", |b, v, t| b.sext(v, t, "sext")),
        JirTag::Trunc => cast(lctx, &inst, "trunc", |b, v, t| b.trunc(v, t, "trunc")),
        JirTag::SIToFP => cast(lctx, &inst, "si2fp", |b, v, t| b.si_to_fp(v, t, "si2fp")),
        JirTag::UIToFP => cast(lctx, &inst, "ui2fp", |b, v, t| b.ui_to_fp(v, t, "ui2fp")),
        JirTag::FPToSI => cast(lctx, &inst, "fp2si", |b, v, t| b.fp_to_si(v, t, "fp2si")),
        JirTag::FPToUI => cast(lctx, &inst, "fp2ui", |b, v, t| b.fp_to_ui(v, t, "fp2ui")),
        JirTag::FPExt | JirTag::FPTrunc => {
            cast(lctx, &inst, "fpcast", |b, v, t| b.fp_cast(v, t, "fpcast"))
        }
        JirTag::BitCast => cast(lctx, &inst, "bitcast", |b, v, t| b.bitcast(v, t, "bitcast")),
        JirTag::PtrToInt => cast(lctx, &inst, "p2i", |b, v, t| b.ptr_to_int(v, t, "p2i")),
        JirTag::IntToPtr => cast(lctx, &inst, "i2p", |b, v, t| b.int_to_ptr(v, t, "i2p")),

        // Control flow.
        JirTag::Br => {
            let bb = *lctx
                .block_map
                .get(&inst.a)
                .ok_or("Br: missing target block")?;
            lctx.ctx.builder().br(bb);
            Ok(None)
        }
        JirTag::CondBr => {
            let cond = emit_inst(lctx, inst.a)?.ok_or("CondBr cond")?;
            let then_b = lctx.jfn.get_extra(inst.b);
            let else_b = lctx.jfn.get_extra(inst.b + 1);
            let then_bb = *lctx.block_map.get(&then_b).ok_or("CondBr then block")?;
            let else_bb = *lctx.block_map.get(&else_b).ok_or("CondBr else block")?;
            lctx.ctx.builder().cond_br(cond, then_bb, else_bb);
            Ok(None)
        }
        JirTag::Switch => emit_switch(lctx, &inst),
        JirTag::Ret => emit_ret(lctx, &inst),
        JirTag::Unreachable => {
            lctx.ctx.builder().unreachable();
            Ok(None)
        }

        // ---- function parameters ----
        JirTag::Param => {
            // When the function uses sret, source param `i` is LLVM arg `i+1`.
            let f = lctx
                .ctx
                .module()
                .get_function(&lctx.jfn.name)
                .ok_or("Param: fn missing")?;
            let arg_offset = if jir_return_is_sret(lctx.jfn, lctx.ctx) {
                1
            } else {
                0
            };
            Ok(Some(f.param(inst.a + arg_offset)))
        }
        JirTag::SretArg => {
            let f = lctx
                .ctx
                .module()
                .get_function(&lctx.jfn.name)
                .ok_or("SretArg: fn missing")?;
            Ok(Some(f.param(0)))
        }

        // ---- storage ----
        JirTag::Alloca => {
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let align = lctx.ctx.alloca_align(inst.ty)?;
            Ok(Some(lctx.ctx.builder().alloca(ty, align, &format!("v{r}"))))
        }
        JirTag::Load => {
            let ptr = emit_inst(lctx, inst.a)?.ok_or("Load ptr")?;
            // Byref aggregate: the value IS the storage pointer (no SSA load).
            if is_by_ref(inst.ty, lctx.ctx) {
                return Ok(Some(ptr));
            }
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            Ok(Some(lctx.ctx.builder().load(ty, ptr, "load")))
        }
        JirTag::Store => {
            let ptr = emit_inst(lctx, inst.a)?.ok_or("Store ptr")?;
            let val = emit_inst(lctx, inst.b)?.ok_or("Store val")?;
            let val_ty = lctx.jfn.get_inst(inst.b).ty;
            if is_by_ref(val_ty, lctx.ctx) {
                // `val` is a pointer to a byref aggregate — memcpy, don't store.
                let size = lctx.ctx.type_size(val_ty)?;
                let align = lctx.ctx.alloca_align(val_ty)?;
                lctx.ctx.builder().memcpy(ptr, align, val, align, size);
            } else {
                lctx.ctx.builder().store(val, ptr);
            }
            Ok(None)
        }
        JirTag::AddrOf => emit_inst(lctx, inst.a),
        JirTag::Deref => {
            let ptr = emit_inst(lctx, inst.a)?.ok_or("Deref ptr")?;
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            Ok(Some(lctx.ctx.builder().load(ty, ptr, "deref")))
        }

        // ---- aggregates ----
        JirTag::FieldAccess | JirTag::ExtractValue => {
            let base = emit_inst(lctx, inst.a)?.ok_or("FieldAccess base")?;
            let base_ty = lctx.jfn.get_inst(inst.a).ty;
            if is_by_ref(base_ty, lctx.ctx) {
                let struct_ty = lctx.ctx.get_llvm_type(base_ty)?;
                let fp = lctx.ctx.builder().struct_gep(struct_ty, base, inst.b, "fa");
                if is_by_ref(inst.ty, lctx.ctx) {
                    return Ok(Some(fp));
                }
                let field_ty = lctx.ctx.get_llvm_type(inst.ty)?;
                return Ok(Some(lctx.ctx.builder().load(field_ty, fp, "field")));
            }
            Ok(Some(
                lctx.ctx.builder().extract_value(base, inst.b, "field"),
            ))
        }
        JirTag::StructLit => {
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let extra = inst.b;
            let count = lctx.jfn.get_extra(extra);
            if is_by_ref(inst.ty, lctx.ctx) {
                let align = lctx.ctx.type_align(inst.ty)?;
                let slot = lctx.ctx.builder().alloca(ty, align, "lit");
                for i in 0..count {
                    let fr = lctx.jfn.get_extra(extra + 1 + i);
                    let fv = emit_inst(lctx, fr)?.ok_or("StructLit field")?;
                    let fp = lctx.ctx.builder().struct_gep(ty, slot, i, "lit.f");
                    let ft = lctx.jfn.get_inst(fr).ty;
                    if is_by_ref(ft, lctx.ctx) {
                        let (fsz, fal) = (lctx.ctx.type_size(ft)?, lctx.ctx.type_align(ft)?);
                        lctx.ctx.builder().memcpy(fp, fal, fv, fal, fsz);
                    } else {
                        lctx.ctx.builder().store(fv, fp);
                    }
                }
                return Ok(Some(slot));
            }
            let mut agg = ty.undef();
            for i in 0..count {
                let fr = lctx.jfn.get_extra(extra + 1 + i);
                let fv = emit_inst(lctx, fr)?.ok_or("StructLit field")?;
                agg = lctx.ctx.builder().insert_value(agg, fv, i, "field");
            }
            Ok(Some(agg))
        }
        JirTag::ArrayLit => {
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let extra = inst.b;
            let count = lctx.jfn.get_extra(extra);
            if is_by_ref(inst.ty, lctx.ctx) {
                let align = lctx.ctx.type_align(inst.ty)?;
                let slot = lctx.ctx.builder().alloca(ty, align, "alit");
                let i64ty = lctx.ctx.context().i64_type();
                for i in 0..count {
                    let er = lctx.jfn.get_extra(extra + 1 + i);
                    let ev = emit_inst(lctx, er)?.ok_or("ArrayLit elem")?;
                    let idx_val = i64ty.const_int(i as u64, false);
                    let ep = lctx.ctx.builder().array_gep(ty, slot, idx_val, "alit.e");
                    let et = lctx.jfn.get_inst(er).ty;
                    if is_by_ref(et, lctx.ctx) {
                        let (esz, eal) = (lctx.ctx.type_size(et)?, lctx.ctx.type_align(et)?);
                        lctx.ctx.builder().memcpy(ep, eal, ev, eal, esz);
                    } else {
                        lctx.ctx.builder().store(ev, ep);
                    }
                }
                return Ok(Some(slot));
            }
            let mut agg = ty.undef();
            for i in 0..count {
                let er = lctx.jfn.get_extra(extra + 1 + i);
                let ev = emit_inst(lctx, er)?.ok_or("ArrayLit elem")?;
                agg = lctx.ctx.builder().insert_value(agg, ev, i, "elem");
            }
            Ok(Some(agg))
        }
        JirTag::Index => {
            let base_ty = lctx.jfn.get_inst(inst.a).ty;
            let basek = lctx.ctx.type_pool.get(base_ty).kind;
            let mut idx_val = emit_inst(lctx, inst.b)?.ok_or("Index idx")?;
            let i64ty = lctx.ctx.context().i64_type();
            if idx_val.type_of() != i64ty {
                idx_val = lctx
                    .ctx
                    .builder()
                    .int_cast(idx_val, i64ty, false, "idx.cast");
            }
            let elem_llvm = lctx.ctx.get_llvm_type(inst.ty)?;
            let elem_byref = is_by_ref(inst.ty, lctx.ctx);
            match basek {
                TypeKind::Slice => {
                    let base = emit_inst(lctx, inst.a)?.ok_or("Index base")?;
                    let ptr = lctx.ctx.builder().extract_value(base, 0, "slice.ptr");
                    let gep = lctx
                        .ctx
                        .builder()
                        .ptr_gep(elem_llvm, ptr, idx_val, "idx.gep");
                    if elem_byref {
                        return Ok(Some(gep));
                    }
                    Ok(Some(lctx.ctx.builder().load(elem_llvm, gep, "idx")))
                }
                TypeKind::PtrMany => {
                    let base = emit_inst(lctx, inst.a)?.ok_or("Index base")?;
                    let gep = lctx
                        .ctx
                        .builder()
                        .ptr_gep(elem_llvm, base, idx_val, "idx.gep");
                    if elem_byref {
                        return Ok(Some(gep));
                    }
                    Ok(Some(lctx.ctx.builder().load(elem_llvm, gep, "idx")))
                }
                _ => {
                    // Array: byref base is already a pointer; a by-value
                    // aggregate base must be spilled to a temp alloca first.
                    let base_llvm = lctx.ctx.get_llvm_type(base_ty)?;
                    let base = emit_inst(lctx, inst.a)?.ok_or("Index base")?;
                    let storage = if !is_by_ref(base_ty, lctx.ctx) {
                        let align = lctx.ctx.type_align(base_ty)?;
                        let st = lctx.ctx.builder().alloca(base_llvm, align, "arr.tmp");
                        lctx.ctx.builder().store(base, st);
                        st
                    } else {
                        base
                    };
                    let gep = lctx
                        .ctx
                        .builder()
                        .array_gep(base_llvm, storage, idx_val, "idx.gep");
                    if elem_byref {
                        return Ok(Some(gep));
                    }
                    Ok(Some(lctx.ctx.builder().load(elem_llvm, gep, "idx")))
                }
            }
        }
        JirTag::FieldAddr => {
            let base_ptr = emit_inst(lctx, inst.a)?.ok_or("FieldAddr base")?;
            let base_inst = *lctx.jfn.get_inst(inst.a);
            let alloca_like = base_inst.tag == JirTag::Alloca
                || (base_inst.tag == JirTag::Param && base_inst.flags & 1 != 0);
            let pointee_ty = if alloca_like {
                base_inst.ty
            } else {
                let k = lctx.ctx.type_pool.get(base_inst.ty);
                if k.kind != TypeKind::PtrSingle && k.kind != TypeKind::PtrMany {
                    return Err("FieldAddr base must be a pointer".into());
                }
                TypeIdx::new(k.a)
            };
            let struct_ty = lctx.ctx.get_llvm_type(pointee_ty)?;
            Ok(Some(
                lctx.ctx
                    .builder()
                    .struct_gep(struct_ty, base_ptr, inst.b, "fieldp"),
            ))
        }
        JirTag::IndexAddr => {
            let base_ptr = emit_inst(lctx, inst.a)?.ok_or("IndexAddr base")?;
            let base_inst = *lctx.jfn.get_inst(inst.a);
            let alloca_like = base_inst.tag == JirTag::Alloca
                || (base_inst.tag == JirTag::Param && base_inst.flags & 1 != 0);
            let pointee_ty = if alloca_like {
                base_inst.ty
            } else {
                let k = lctx.ctx.type_pool.get(base_inst.ty);
                if k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany {
                    TypeIdx::new(k.a)
                } else {
                    base_inst.ty
                }
            };
            let pk = lctx.ctx.type_pool.get(pointee_ty).kind;
            let mut idx_val = emit_inst(lctx, inst.b)?.ok_or("IndexAddr idx")?;
            let i64ty = lctx.ctx.context().i64_type();
            if idx_val.type_of() != i64ty {
                idx_val = lctx
                    .ctx
                    .builder()
                    .int_cast(idx_val, i64ty, false, "idx.cast");
            }
            let elem_ty = TypeIdx::new(lctx.ctx.type_pool.get(inst.ty).a);
            let elem_llvm = lctx.ctx.get_llvm_type(elem_ty)?;
            if pk == TypeKind::Array {
                let arr_llvm = lctx.ctx.get_llvm_type(pointee_ty)?;
                return Ok(Some(
                    lctx.ctx
                        .builder()
                        .array_gep(arr_llvm, base_ptr, idx_val, "idx.addr"),
                ));
            }
            Ok(Some(
                lctx.ctx
                    .builder()
                    .ptr_gep(elem_llvm, base_ptr, idx_val, "idx.addr"),
            ))
        }
        JirTag::EnumPayload => {
            let enum_ptr = emit_inst(lctx, inst.a)?.ok_or("EnumPayload base")?;
            let base_ty = lctx.jfn.get_inst(inst.a).ty;
            let enum_ty = lctx.ctx.get_llvm_type(base_ty)?;
            let payload_area = lctx
                .ctx
                .builder()
                .struct_gep(enum_ty, enum_ptr, 1, "enum.payload");
            let field_ptr = if inst.b != 0 {
                let off = lctx
                    .ctx
                    .context()
                    .i64_type()
                    .const_int(inst.b as u64, false);
                let i8 = lctx.ctx.context().i8_type();
                lctx.ctx
                    .builder()
                    .ptr_gep(i8, payload_area, off, "enum.payload.off")
            } else {
                payload_area
            };
            if is_by_ref(inst.ty, lctx.ctx) {
                return Ok(Some(field_ptr));
            }
            let field_ty = lctx.ctx.get_llvm_type(inst.ty)?;
            Ok(Some(lctx.ctx.builder().load(
                field_ty,
                field_ptr,
                "enum.payload.val",
            )))
        }

        // ---- slices / strings / fills ----
        JirTag::MakeSlice => {
            let ptr_v = emit_inst(lctx, inst.a)?.ok_or("MakeSlice ptr")?;
            let len_v = emit_inst(lctx, inst.b)?.ok_or("MakeSlice len")?;
            let slice_ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let mut slice = slice_ty.undef();
            slice = lctx
                .ctx
                .builder()
                .insert_value(slice, ptr_v, 0, "slice.ptr");
            slice = lctx
                .ctx
                .builder()
                .insert_value(slice, len_v, 1, "slice.len");
            Ok(Some(slice))
        }
        JirTag::MemSet => {
            let ptr = emit_inst(lctx, inst.a)?.ok_or("MemSet dst")?;
            let byte_val = lctx
                .ctx
                .context()
                .i8_type()
                .const_int(inst.flags as u64, false);
            lctx.ctx.builder().memset(ptr, byte_val, inst.b as u64, 1);
            Ok(None)
        }
        JirTag::Str => {
            let bytes = lctx.ctx.string_pool.get(StringIdx::new(inst.a)).to_vec();
            let str_const = lctx.ctx.context().const_string(&bytes, true);
            let arr_ty = lctx
                .ctx
                .context()
                .i8_type()
                .array_type(bytes.len() as u64 + 1);
            let str_global = lctx.ctx.module().add_global(arr_ty, "str");
            str_global.set_global_constant(true);
            str_global.set_initializer(str_const);
            let i8ptr = lctx.ctx.context().pointer_type(0);
            let str_ptr = lctx.ctx.builder().bitcast(str_global, i8ptr, "str_ptr");
            // Pointer decay: a pointer-typed Str yields the bare address.
            let strk = lctx.ctx.type_pool.get(inst.ty).kind;
            if strk == TypeKind::PtrMany || strk == TypeKind::PtrSingle {
                return Ok(Some(str_ptr));
            }
            let slice_ty = lctx.ctx.get_llvm_type(inst.ty)?;
            let mut slice = slice_ty.undef();
            slice = lctx
                .ctx
                .builder()
                .insert_value(slice, str_ptr, 0, "slice_ptr");
            let len = lctx
                .ctx
                .context()
                .i64_type()
                .const_int(bytes.len() as u64, false);
            slice = lctx.ctx.builder().insert_value(slice, len, 1, "slice_len");
            Ok(Some(slice))
        }

        // ---- function references / calls / drops ----
        JirTag::FnRef => {
            let name = String::from_utf8_lossy(&lctx.ctx.string_pool.get(StringIdx::new(inst.a)))
                .into_owned();
            let f = lctx
                .ctx
                .module()
                .get_function(&name)
                .ok_or_else(|| format!("unknown fn-ref `{name}`"))?;
            let fn_val = f.as_value();
            if lctx.ctx.type_pool.get(inst.ty).kind == TypeKind::Fn {
                return Ok(Some(fn_val)); // typed fn-pointer: already a ptr value
            }
            let ty = lctx.ctx.get_llvm_type(inst.ty)?;
            Ok(Some(lctx.ctx.builder().ptr_to_int(fn_val, ty, "fnref.u64")))
        }
        JirTag::DropBinding => {
            let binding_ptr = emit_inst(lctx, inst.a)?.ok_or("DropBinding ptr")?;
            let symbol = String::from_utf8_lossy(&lctx.ctx.string_pool.get(StringIdx::new(inst.b)))
                .into_owned();
            let f = lctx
                .ctx
                .module()
                .get_function(&symbol)
                .ok_or_else(|| format!("drop callee `{symbol}` not declared"))?;
            lctx.ctx.builder().call(f, &[binding_ptr], "");
            Ok(None)
        }
        JirTag::Call => {
            let symbol = String::from_utf8_lossy(&lctx.ctx.string_pool.get(StringIdx::new(inst.a)))
                .into_owned();
            let f = lctx
                .ctx
                .module()
                .get_function(&symbol)
                .ok_or_else(|| format!("unknown callee `{symbol}`"))?;
            let extra = inst.b;
            let arg_count = lctx.jfn.get_extra(extra);
            let callee_uses_sret = f.uses_sret();
            let mut args = Vec::with_capacity(arg_count as usize);
            for i in 0..arg_count {
                let ar = lctx.jfn.get_extra(extra + 1 + i);
                args.push(emit_inst(lctx, ar)?.ok_or("call arg")?);
            }
            if callee_uses_sret {
                // LLVM call returns void; the result is in args[0] (the sret
                // slot) — hand that pointer back for byref consumers.
                lctx.ctx.builder().call(f, &args, "");
                return Ok(Some(args[0]));
            }
            let name = if inst.ty.is_none() { "" } else { "call" };
            Ok(Some(lctx.ctx.builder().call(f, &args, name)))
        }
        JirTag::CallIndirect => {
            let callee_val = emit_inst(lctx, inst.a)?.ok_or("CallIndirect callee")?;
            let callee_ty = lctx.jfn.get_inst(inst.a).ty;
            let k = lctx.ctx.type_pool.get(callee_ty);
            if k.kind != TypeKind::Fn {
                return Err("CallIndirect callee is not of Fn type".into());
            }
            let ret_ty = TypeIdx::new(k.a);
            let param_tys: Vec<TypeIdx> = lctx.ctx.type_pool.fn_params_at(k.b).to_vec();
            let mut llvm_params = Vec::with_capacity(param_tys.len());
            for &pt in &param_tys {
                llvm_params.push(lctx.ctx.get_llvm_type(pt)?);
            }
            let llvm_ret = if ret_ty.is_none() {
                lctx.ctx.context().void_type()
            } else {
                lctx.ctx.get_llvm_type(ret_ty)?
            };
            let llvm_fn_ty = llvm_ret.fn_type(&llvm_params, false);
            let extra = inst.b;
            let arg_count = lctx.jfn.get_extra(extra);
            let mut args = Vec::with_capacity(arg_count as usize);
            for i in 0..arg_count {
                let ar = lctx.jfn.get_extra(extra + 1 + i);
                let mut av = emit_inst(lctx, ar)?.ok_or("CallIndirect arg")?;
                // The indirect signature passes aggregates by value, but a
                // byref aggregate is a pointer in our SSA model — load it.
                if (i as usize) < param_tys.len() && is_by_ref(param_tys[i as usize], lctx.ctx) {
                    let agg_ty = lctx.ctx.get_llvm_type(param_tys[i as usize])?;
                    av = lctx.ctx.builder().load(agg_ty, av, "arg.byval");
                }
                args.push(av);
            }
            let name = if inst.ty.is_none() {
                ""
            } else {
                "call.indirect"
            };
            let result = lctx
                .ctx
                .builder()
                .indirect_call(llvm_fn_ty, callee_val, &args, name);
            // A byref aggregate return arrives as an SSA value; spill to a slot
            // and hand back the pointer to match the byref model.
            if !inst.ty.is_none() && is_by_ref(inst.ty, lctx.ctx) {
                let agg_ty = lctx.ctx.get_llvm_type(inst.ty)?;
                let align = lctx.ctx.type_align(inst.ty)?;
                let slot = lctx.ctx.builder().alloca(agg_ty, align, "ret.byval");
                lctx.ctx.builder().store(result, slot);
                return Ok(Some(slot));
            }
            Ok(Some(result))
        }

        JirTag::Invalid => Err("jir_codegen: Invalid tag in instruction stream".into()),
    }
}

fn icmp<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
    pred: IntPredicate,
    name: &str,
) -> Result<Option<Value<'ctx>>, String> {
    let (a, b) = binop_operands(lctx, inst)?;
    Ok(Some(lctx.ctx.builder().icmp(pred, a, b, name)))
}

fn fcmp<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
    pred: RealPredicate,
    name: &str,
) -> Result<Option<Value<'ctx>>, String> {
    let (a, b) = binop_operands(lctx, inst)?;
    Ok(Some(lctx.ctx.builder().fcmp(pred, a, b, name)))
}

fn cast<'ctx, F>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
    what: &str,
    build: F,
) -> Result<Option<Value<'ctx>>, String>
where
    F: FnOnce(&jam_llvm::Builder<'ctx>, Value<'ctx>, jam_llvm::Type<'ctx>) -> Value<'ctx>,
{
    let v = emit_inst(lctx, inst.a)?.ok_or_else(|| format!("{what} operand"))?;
    let ty = lctx.ctx.get_llvm_type(inst.ty)?;
    Ok(Some(build(lctx.ctx.builder(), v, ty)))
}

fn emit_switch<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
) -> Result<Option<Value<'ctx>>, String> {
    let scrut = emit_inst(lctx, inst.a)?.ok_or("Switch scrut")?;
    let scrut_ty = lctx.jfn.get_inst(inst.a).ty;
    let scrut_llvm = lctx.ctx.get_llvm_type(scrut_ty)?;
    let extra = inst.b;
    let default_b = lctx.jfn.get_extra(extra);
    let case_count = lctx.jfn.get_extra(extra + 1);
    let default_bb = *lctx
        .block_map
        .get(&default_b)
        .ok_or("Switch default block")?;

    let read_case = |i: u32| -> Result<(Value<'ctx>, BasicBlock<'ctx>), String> {
        let base = extra + 2 + i * 4;
        let lo = lctx.jfn.get_extra(base) as u64;
        let hi = lctx.jfn.get_extra(base + 1) as u64;
        let signed = lctx.jfn.get_extra(base + 2) != 0;
        let case_b = lctx.jfn.get_extra(base + 3);
        let bits = lo | (hi << 32);
        let case_val = scrut_llvm.const_int(bits, signed);
        let case_bb = *lctx.block_map.get(&case_b).ok_or("Switch case block")?;
        Ok((case_val, case_bb))
    };

    // 1 case + default collapses to icmp + cond_br.
    if case_count == 1 {
        let (case_val, case_bb) = read_case(0)?;
        let cmp = lctx
            .ctx
            .builder()
            .icmp(IntPredicate::Eq, scrut, case_val, "match.eq");
        lctx.ctx.builder().cond_br(cmp, case_bb, default_bb);
        return Ok(None);
    }
    let sw = lctx.ctx.builder().switch(scrut, default_bb, case_count);
    for i in 0..case_count {
        let (case_val, case_bb) = read_case(i)?;
        sw.add_case(case_val, case_bb);
    }
    Ok(None)
}

fn emit_ret<'ctx>(
    lctx: &mut JirCodegenCtx<'_, 'ctx>,
    inst: &JirInst,
) -> Result<Option<Value<'ctx>>, String> {
    if jir_return_is_sret(lctx.jfn, lctx.ctx) {
        let f = lctx
            .ctx
            .module()
            .get_function(&lctx.jfn.name)
            .ok_or("Ret: function missing")?;
        let sret_slot = f.param(0);
        if inst.a != NO_JIR_REF {
            let vty = lctx.jfn.get_inst(inst.a).ty;
            if is_by_ref(vty, lctx.ctx) {
                let src = emit_inst(lctx, inst.a)?.ok_or("Ret src")?;
                let size = lctx.ctx.type_size(vty)?;
                let align = lctx.ctx.type_align(vty)?;
                lctx.ctx
                    .builder()
                    .memcpy(sret_slot, align, src, align, size);
            } else {
                let v = emit_inst(lctx, inst.a)?.ok_or("Ret value")?;
                lctx.ctx.builder().store(v, sret_slot);
            }
        }
        lctx.ctx.builder().ret_void();
        return Ok(None);
    }
    if inst.a == NO_JIR_REF {
        lctx.ctx.builder().ret_void();
    } else {
        let v = emit_inst(lctx, inst.a)?.ok_or("Ret value")?;
        lctx.ctx.builder().ret(v);
    }
    Ok(None)
}

/// Mark blocks reachable from entry (block 1) by walking terminator successors.
/// Unreachable blocks (orphaned by constant-folded branches) are skipped in
/// codegen — frontend DCE that keeps their instructions (incl. extern calls)
/// out of the LLVM module.
fn compute_reachable_blocks(jfn: &JirFunction) -> Vec<bool> {
    let mut reachable = vec![false; jfn.blocks.len()];
    if jfn.blocks.len() <= 1 {
        return reachable;
    }
    reachable[1] = true;
    let mut stack = vec![1u32];
    while let Some(b) = stack.pop() {
        let blk = jfn.get_block(b);
        if blk.insts.is_empty() {
            continue;
        }
        let last = *jfn.get_inst(*blk.insts.last().unwrap());
        let mut visit = |t: u32| {
            if t >= 1 && (t as usize) < jfn.blocks.len() && !reachable[t as usize] {
                reachable[t as usize] = true;
                stack.push(t);
            }
        };
        match last.tag {
            JirTag::Br => visit(last.a),
            JirTag::CondBr => {
                let ex = last.b;
                if ex as usize + 2 <= jfn.extra.len() {
                    visit(jfn.get_extra(ex));
                    visit(jfn.get_extra(ex + 1));
                }
            }
            JirTag::Switch => {
                let ex = last.b;
                if ex as usize + 2 <= jfn.extra.len() {
                    visit(jfn.get_extra(ex)); // default
                    let case_count = jfn.get_extra(ex + 1);
                    for i in 0..case_count {
                        let slot = ex + 2 + i * 4 + 3;
                        if (slot as usize) < jfn.extra.len() {
                            visit(jfn.get_extra(slot));
                        }
                    }
                }
            }
            _ => {} // Ret / Unreachable: no successors
        }
    }
    reachable
}

/// Lower a function body to LLVM IR. The prototype must already exist
/// ([`jir_declare_prototype`]). `extern` functions have no body.
pub fn jir_define_body<'ctx>(jfn: &JirFunction, ctx: &CodegenContext<'ctx>) -> Result<(), String> {
    if jfn.is_extern {
        return Ok(());
    }
    let f = ctx
        .module()
        .get_function(&jfn.name)
        .ok_or_else(|| format!("jir_define_body: prototype missing for `{}`", jfn.name))?;

    let mut lctx = JirCodegenCtx {
        jfn,
        ctx,
        value_map: HashMap::new(),
        block_map: HashMap::new(),
    };

    let reachable = compute_reachable_blocks(jfn);

    // Create LLVM blocks first so terminators resolve forward references; skip
    // the sentinel (0) and dead blocks.
    for (b, &live) in reachable.iter().enumerate().skip(1) {
        if !live {
            continue;
        }
        let bb = f.append_basic_block(&jfn.get_block(b as JirBlockRef).name);
        lctx.block_map.insert(b as JirBlockRef, bb);
    }

    // Emit instructions block-by-block.
    for (b, &live) in reachable.iter().enumerate().skip(1) {
        if !live {
            continue;
        }
        let bb = lctx.block_map[&(b as JirBlockRef)];
        ctx.builder().position_at_end(bb);
        let insts = jfn.get_block(b as JirBlockRef).insts.clone();
        for r in insts {
            let v = emit_inst(&mut lctx, r)?;
            if let Some(val) = v {
                lctx.value_map.insert(r, val);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jir::{JirFunction, JirInst};
    use jam_core::index::TypeIdx;
    use jam_llvm::Context;

    #[test]
    fn declare_simple_function_prototype() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "add".into();
        jfn.param_types = vec![builtin::I32, builtin::I32];
        jfn.param_modes = vec![ParamMode::Let, ParamMode::Let];
        jfn.return_type = builtin::I32;
        let f = jir_declare_prototype(&jfn, &cg).unwrap();
        assert_eq!(f.count_params(), 2);
        assert!(f.return_type().is_integer());
        // NB: a bodyless *internal*-linkage function does NOT verify yet — LLVM
        // requires internal functions to have a definition; the body is added
        // by jir_define_body. (An `extern` decl with external linkage would.)
    }

    #[test]
    fn mut_param_lowers_to_pointer() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "bump".into();
        jfn.param_types = vec![builtin::I32];
        jfn.param_modes = vec![ParamMode::Mut];
        jfn.return_type = TypeIdx::NONE; // void
        let f = jir_declare_prototype(&jfn, &cg).unwrap();
        assert_eq!(f.count_params(), 1);
        assert!(f.param(0).type_of().is_pointer()); // mut -> ByPointer
        assert!(f.return_type().is_void());
    }

    #[test]
    fn extern_declaration_verifies() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "putchar".into();
        jfn.param_types = vec![builtin::I32];
        jfn.return_type = builtin::I32;
        jfn.is_extern = true;
        let f = jir_declare_prototype(&jfn, &cg).unwrap();
        // extern -> external linkage; a bodyless external declaration verifies.
        assert!(f.verify());
        assert_eq!(f.count_params(), 1);
    }

    #[test]
    fn lower_arithmetic_body_emits_real_ir() {
        // fn five() i32 { return 2 + 3 }
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "five".into();
        jfn.return_type = builtin::I32;
        let entry = jfn.push_block("entry");
        let i2 = jfn.push_inst(JirInst {
            tag: JirTag::Int,
            a: 2,
            ty: builtin::I32,
            ..Default::default()
        });
        let i3 = jfn.push_inst(JirInst {
            tag: JirTag::Int,
            a: 3,
            ty: builtin::I32,
            ..Default::default()
        });
        let add = jfn.push_inst(JirInst {
            tag: JirTag::Add,
            a: i2,
            b: i3,
            ty: builtin::I32,
            ..Default::default()
        });
        let ret = jfn.push_inst(JirInst {
            tag: JirTag::Ret,
            a: add,
            ..Default::default()
        });
        jfn.get_block_mut(entry).insts.extend([i2, i3, add, ret]);

        jir_declare_prototype(&jfn, &cg).unwrap();
        jir_define_body(&jfn, &cg).unwrap();
        let f = cg.module().get_function("five").unwrap();
        // An internal-linkage function WITH a body now verifies.
        assert!(f.verify());
    }

    #[test]
    fn lower_local_variable_body() {
        // fn f() i32 { var x: i32; x = 7; return x }  (Alloca, Store, Load, Ret)
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "f".into();
        jfn.return_type = builtin::I32;
        let entry = jfn.push_block("entry");
        let slot = jfn.push_inst(JirInst {
            tag: JirTag::Alloca,
            ty: builtin::I32,
            ..Default::default()
        });
        let seven = jfn.push_inst(JirInst {
            tag: JirTag::Int,
            a: 7,
            ty: builtin::I32,
            ..Default::default()
        });
        let store = jfn.push_inst(JirInst {
            tag: JirTag::Store,
            a: slot,
            b: seven,
            ..Default::default()
        });
        let load = jfn.push_inst(JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: builtin::I32,
            ..Default::default()
        });
        let ret = jfn.push_inst(JirInst {
            tag: JirTag::Ret,
            a: load,
            ..Default::default()
        });
        jfn.get_block_mut(entry)
            .insts
            .extend([slot, seven, store, load, ret]);

        jir_declare_prototype(&jfn, &cg).unwrap();
        jir_define_body(&jfn, &cg).unwrap();
        assert!(cg.module().get_function("f").unwrap().verify());
    }

    #[test]
    fn lower_comparison_body() {
        // fn lt() bool { return 2 < 3 }
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let mut jfn = JirFunction::new();
        jfn.name = "lt".into();
        jfn.return_type = builtin::BOOL;
        let entry = jfn.push_block("entry");
        let i2 = jfn.push_inst(JirInst {
            tag: JirTag::Int,
            a: 2,
            ty: builtin::I32,
            ..Default::default()
        });
        let i3 = jfn.push_inst(JirInst {
            tag: JirTag::Int,
            a: 3,
            ty: builtin::I32,
            ..Default::default()
        });
        let lt = jfn.push_inst(JirInst {
            tag: JirTag::ICmpSlt,
            a: i2,
            b: i3,
            ty: builtin::BOOL,
            ..Default::default()
        });
        let ret = jfn.push_inst(JirInst {
            tag: JirTag::Ret,
            a: lt,
            ..Default::default()
        });
        jfn.get_block_mut(entry).insts.extend([i2, i3, lt, ret]);

        jir_declare_prototype(&jfn, &cg).unwrap();
        jir_define_body(&jfn, &cg).unwrap();
        let f = cg.module().get_function("lt").unwrap();
        assert!(f.verify());
    }

    #[test]
    fn struct_return_uses_sret() {
        let ctx = Context::new();
        let cg = CodegenContext::new(&ctx, "m");
        let nid = cg.string_pool.intern(b"Pair");
        let pair = cg.type_pool.intern_struct(nid);
        let named = ctx.named_struct("Pair");
        cg.register_struct(
            "Pair",
            named,
            vec![("a".into(), builtin::I64), ("b".into(), builtin::I64)],
        );
        let mut jfn = JirFunction::new();
        jfn.name = "makePair".into();
        jfn.return_type = pair;
        let f = jir_declare_prototype(&jfn, &cg).unwrap();
        // sret: a leading ptr param + a void return.
        assert_eq!(f.count_params(), 1);
        assert!(f.param(0).type_of().is_pointer());
        assert!(f.return_type().is_void());
    }
}
