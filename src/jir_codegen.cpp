/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "jir_codegen.h"

#include "ast.h"
#include "codegen.h"
#include "jam_llvm.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// Per-function lowering state: maps JirRef → LLVM Value so dataflow
// references resolve as instructions are emitted in block order.
// Also maps JirBlockRef → LLVM BasicBlock for terminators.
struct JirCodegenCtx {
	const JirFunction &jfn;
	JamCodegenContext &ctx;
	std::unordered_map<JirRef, JamValueRef> valueMap;
	std::unordered_map<JirBlockRef, JamBasicBlockRef> blockMap;
};

// Helper: integer compare via the existing JamLLVMBuildICmp wrapper.
static JamValueRef buildICmp(JirCodegenCtx &lctx, JamIntPredicate p,
                             JamValueRef lhs, JamValueRef rhs,
                             const char *name) {
	return JamLLVMBuildICmp(lctx.ctx.getBuilder(), p, lhs, rhs, name);
}

static JamValueRef buildFCmp(JirCodegenCtx &lctx, JamFloatPredicate p,
                             JamValueRef lhs, JamValueRef rhs,
                             const char *name) {
	return JamLLVMBuildFCmp(lctx.ctx.getBuilder(), p, lhs, rhs, name);
}

static JamValueRef emitInstImpl(JirCodegenCtx &lctx, JirRef r);

// Public entry: look up the cached LLVM value for `r`, or emit it
// (recursing for any unrelated subexpressions) and cache the result.
// All recursive uses of `emitInst` go through this so a JirRef is
// codegen'd exactly once even when it's referenced by multiple
// instructions across blocks. Without caching here, walking the
// block list out-of-order (e.g. the arm body before the matchbind
// block where the binding's alloca lives) causes the dataflow
// references to lower to different LLVM values, splitting reads from
// writes — see the match-binding bug fix.
static JamValueRef emitInst(JirCodegenCtx &lctx, JirRef r) {
	if (r == kNoJirRef) return nullptr;
	auto cached = lctx.valueMap.find(r);
	if (cached != lctx.valueMap.end()) return cached->second;
	JamValueRef v = emitInstImpl(lctx, r);
	if (v != nullptr) lctx.valueMap[r] = v;
	return v;
}

static JamValueRef emitInstImpl(JirCodegenCtx &lctx, JirRef r) {
	const JirInst &inst = lctx.jfn.getInst(r);

	switch (inst.tag) {
	case JirTag::Int: {
		uint64_t val = static_cast<uint64_t>(inst.a) |
		               (static_cast<uint64_t>(inst.b) << 32);
		bool isNeg = (inst.flags & 1) != 0;
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		uint64_t materialized =
		    isNeg ? static_cast<uint64_t>(-static_cast<int64_t>(val)) : val;
		return JamLLVMConstInt(ty, materialized, isNeg);
	}
	case JirTag::Float: {
		uint64_t bits = static_cast<uint64_t>(inst.a) |
		                (static_cast<uint64_t>(inst.b) << 32);
		double d;
		std::memcpy(&d, &bits, sizeof(d));
		if (inst.flags & 1) d = -d;
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMConstReal(ty, d);
	}
	case JirTag::Bool:
		return JamLLVMConstInt(lctx.ctx.getInt1Type(),
		                      inst.a != 0 ? 1 : 0, false);
	case JirTag::Str: {
		StringIdx sid = static_cast<StringIdx>(inst.a);
		const std::string &val = lctx.ctx.getStringPool().get(sid);
		JamValueRef strConst = JamLLVMConstString(
		    lctx.ctx.getContext(), val.c_str(),
		    static_cast<unsigned>(val.length()), true);
		JamTypeRef arrTy =
		    JamLLVMArrayType(lctx.ctx.getInt8Type(),
		                     static_cast<uint64_t>(val.length() + 1));
		JamValueRef strGlobal =
		    JamLLVMAddGlobal(lctx.ctx.getModule(), arrTy, "str");
		JamLLVMSetGlobalConstant(strGlobal, true);
		JamLLVMSetInitializer(strGlobal, strConst);

		JamTypeRef sliceTy = lctx.ctx.getLLVMType(inst.ty);
		JamTypeRef i8PtrTy = JamLLVMPointerType(lctx.ctx.getInt8Type(), 0);
		JamValueRef strPtr = JamLLVMBuildBitCast(
		    lctx.ctx.getBuilder(), strGlobal, i8PtrTy, "str_ptr");
		JamValueRef slice = JamLLVMGetUndef(sliceTy);
		slice = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), slice, strPtr,
		                                0, "slice_ptr");
		slice = JamLLVMBuildInsertValue(
		    lctx.ctx.getBuilder(), slice,
		    JamLLVMConstInt(lctx.ctx.getInt64Type(),
		                    static_cast<uint64_t>(val.length()), false),
		    1, "slice_len");
		return slice;
	}
	case JirTag::Param: {
		// flags & 1 means the param is a `mut` / `move` and the LLVM
		// signature passes a pointer to the pointee type. We hand
		// back the pointer directly so the local map can use it as
		// the alloca-equivalent for `self`-style field access.
		JamFunctionRef f = JamLLVMGetFunction(
		    lctx.ctx.getModule(), lctx.jfn.name.c_str());
		return JamLLVMGetParam(f, inst.a);
	}
	case JirTag::Alloca: {
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		uint64_t align = lctx.ctx.typeAlign(inst.ty);
		std::string nm = "v" + std::to_string(r);
		return JamLLVMBuildAlloca(lctx.ctx.getBuilder(), ty, align,
		                          nm.c_str());
	}
	case JirTag::Load: {
		JamValueRef ptr = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildLoad(lctx.ctx.getBuilder(), ty, ptr, "load");
	}
	case JirTag::Store: {
		JamValueRef ptr = emitInst(lctx, inst.a);
		JamValueRef val = emitInst(lctx, inst.b);
		JamLLVMBuildStore(lctx.ctx.getBuilder(), val, ptr);
		return nullptr;
	}
	// === Integer arithmetic ===
	case JirTag::Add: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildAdd(lctx.ctx.getBuilder(), a, b, "add");
	}
	case JirTag::Sub: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildSub(lctx.ctx.getBuilder(), a, b, "sub");
	}
	case JirTag::Mul: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildMul(lctx.ctx.getBuilder(), a, b, "mul");
	}
	case JirTag::SDiv: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildSDiv(lctx.ctx.getBuilder(), a, b, "sdiv");
	}
	case JirTag::UDiv: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildUDiv(lctx.ctx.getBuilder(), a, b, "udiv");
	}
	case JirTag::SRem: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildSRem(lctx.ctx.getBuilder(), a, b, "srem");
	}
	case JirTag::URem: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildURem(lctx.ctx.getBuilder(), a, b, "urem");
	}
	// === Float arithmetic ===
	case JirTag::FAdd: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildFAdd(lctx.ctx.getBuilder(), a, b, "fadd");
	}
	case JirTag::FSub: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildFSub(lctx.ctx.getBuilder(), a, b, "fsub");
	}
	case JirTag::FMul: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildFMul(lctx.ctx.getBuilder(), a, b, "fmul");
	}
	case JirTag::FDiv: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildFDiv(lctx.ctx.getBuilder(), a, b, "fdiv");
	}
	case JirTag::FRem: {
		JamValueRef a = emitInst(lctx, inst.a);
		JamValueRef b = emitInst(lctx, inst.b);
		return JamLLVMBuildFRem(lctx.ctx.getBuilder(), a, b, "frem");
	}
	case JirTag::FNeg: {
		JamValueRef a = emitInst(lctx, inst.a);
		return JamLLVMBuildFNeg(lctx.ctx.getBuilder(), a, "fneg");
	}
	// === Integer comparison ===
	case JirTag::ICmpEq:
		return buildICmp(lctx, JAM_ICMP_EQ,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "eq");
	case JirTag::ICmpNe:
		return buildICmp(lctx, JAM_ICMP_NE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ne");
	case JirTag::ICmpSlt:
		return buildICmp(lctx, JAM_ICMP_SLT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "slt");
	case JirTag::ICmpSle:
		return buildICmp(lctx, JAM_ICMP_SLE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "sle");
	case JirTag::ICmpSgt:
		return buildICmp(lctx, JAM_ICMP_SGT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "sgt");
	case JirTag::ICmpSge:
		return buildICmp(lctx, JAM_ICMP_SGE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "sge");
	case JirTag::ICmpUlt:
		return buildICmp(lctx, JAM_ICMP_ULT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ult");
	case JirTag::ICmpUle:
		return buildICmp(lctx, JAM_ICMP_ULE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ule");
	case JirTag::ICmpUgt:
		return buildICmp(lctx, JAM_ICMP_UGT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ugt");
	case JirTag::ICmpUge:
		return buildICmp(lctx, JAM_ICMP_UGE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "uge");
	// === Float comparison ===
	case JirTag::FCmpOeq:
		return buildFCmp(lctx, JAM_FCMP_OEQ,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "oeq");
	case JirTag::FCmpOne:
		return buildFCmp(lctx, JAM_FCMP_UNE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "one");
	case JirTag::FCmpOlt:
		return buildFCmp(lctx, JAM_FCMP_OLT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "olt");
	case JirTag::FCmpOle:
		return buildFCmp(lctx, JAM_FCMP_OLE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ole");
	case JirTag::FCmpOgt:
		return buildFCmp(lctx, JAM_FCMP_OGT,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "ogt");
	case JirTag::FCmpOge:
		return buildFCmp(lctx, JAM_FCMP_OGE,
		                 emitInst(lctx, inst.a), emitInst(lctx, inst.b), "oge");
	// === Bitwise / shift ===
	case JirTag::BitAnd:
		return JamLLVMBuildAnd(lctx.ctx.getBuilder(),
		                       emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "and");
	case JirTag::BitOr:
		return JamLLVMBuildOr(lctx.ctx.getBuilder(),
		                      emitInst(lctx, inst.a),
		                      emitInst(lctx, inst.b), "or");
	case JirTag::BitXor:
		return JamLLVMBuildXor(lctx.ctx.getBuilder(),
		                       emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "xor");
	case JirTag::Shl:
		return JamLLVMBuildShl(lctx.ctx.getBuilder(),
		                       emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "shl");
	case JirTag::AShr:
		return JamLLVMBuildAShr(lctx.ctx.getBuilder(),
		                        emitInst(lctx, inst.a),
		                        emitInst(lctx, inst.b), "ashr");
	case JirTag::LShr:
		return JamLLVMBuildLShr(lctx.ctx.getBuilder(),
		                        emitInst(lctx, inst.a),
		                        emitInst(lctx, inst.b), "lshr");
	case JirTag::BitNot: {
		// LLVM has no NOT op; emit XOR with all-ones of the operand's type.
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		JamValueRef ones = JamLLVMConstInt(ty, ~static_cast<uint64_t>(0),
		                                   true);
		return JamLLVMBuildXor(lctx.ctx.getBuilder(), v, ones, "not");
	}
	case JirTag::LogNot: {
		// Boolean inversion: XOR with i1 1. Operand is already i1.
		JamValueRef v = emitInst(lctx, inst.a);
		JamValueRef one =
		    JamLLVMConstInt(lctx.ctx.getInt1Type(), 1, false);
		return JamLLVMBuildXor(lctx.ctx.getBuilder(), v, one, "lnot");
	}
	// === Type conversions ===
	case JirTag::ZExt: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntCast(lctx.ctx.getBuilder(), v, ty, false,
		                           "zext");
	}
	case JirTag::SExt: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntCast(lctx.ctx.getBuilder(), v, ty, true,
		                           "sext");
	}
	case JirTag::Trunc: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntCast(lctx.ctx.getBuilder(), v, ty, false,
		                           "trunc");
	}
	case JirTag::SIToFP: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildSIToFP(lctx.ctx.getBuilder(), v, ty, "si2fp");
	}
	case JirTag::UIToFP: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildUIToFP(lctx.ctx.getBuilder(), v, ty, "ui2fp");
	}
	case JirTag::FPToSI: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildFPToSI(lctx.ctx.getBuilder(), v, ty, "fp2si");
	}
	case JirTag::FPToUI: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildFPToUI(lctx.ctx.getBuilder(), v, ty, "fp2ui");
	}
	case JirTag::FPExt:
	case JirTag::FPTrunc: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildFPCast(lctx.ctx.getBuilder(), v, ty, "fpcast");
	}
	case JirTag::BitCast: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildBitCast(lctx.ctx.getBuilder(), v, ty, "bitcast");
	}
	// === Aggregates ===
	case JirTag::StructLit: {
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		uint32_t count = lctx.jfn.getExtra(extra);
		JamValueRef agg = JamLLVMGetUndef(ty);
		for (uint32_t i = 0; i < count; i++) {
			JirRef fr =
			    static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			JamValueRef fv = emitInst(lctx, fr);
			agg = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), agg, fv,
			                              i, "field");
		}
		return agg;
	}
	case JirTag::FieldAccess:
	case JirTag::ExtractValue: {
		JamValueRef base = emitInst(lctx, inst.a);
		return JamLLVMBuildExtractValue(lctx.ctx.getBuilder(), base, inst.b,
		                                "field");
	}
	case JirTag::ArrayLit: {
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		uint32_t count = lctx.jfn.getExtra(extra);
		JamValueRef agg = JamLLVMGetUndef(ty);
		for (uint32_t i = 0; i < count; i++) {
			JirRef er =
			    static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			JamValueRef ev = emitInst(lctx, er);
			agg = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), agg, ev,
			                              i, "elem");
		}
		return agg;
	}
	case JirTag::Index: {
		// Index reads. If the JIR ref already maps to a pointer
		// (the base was loaded from an alloca), GEP+Load. Otherwise
		// the base is an SSA aggregate value, so spill it to a temp
		// alloca first, GEP, then load.
		const JirInst &baseInst = lctx.jfn.getInst(inst.a);
		const TypeKey &basek = lctx.ctx.getTypePool().get(baseInst.ty);
		JamValueRef idxVal = emitInst(lctx, inst.b);
		JamTypeRef i64 = lctx.ctx.getInt64Type();
		// Coerce index to i64 (the GEP wrappers expect a sized integer).
		JamTypeRef idxLLVMTy = JamLLVMTypeOf(idxVal);
		if (idxLLVMTy != i64) {
			idxVal = JamLLVMBuildIntCast(lctx.ctx.getBuilder(), idxVal,
			                             i64, false, "idx.cast");
		}

		JamTypeRef baseLLVMTy = lctx.ctx.getLLVMType(baseInst.ty);
		JamTypeRef elemLLVMTy = lctx.ctx.getLLVMType(inst.ty);

		if (basek.kind == TypeKind::Slice) {
			// SSA slice value: extract the pointer field (0), GEP by elem.
			JamValueRef base = emitInst(lctx, inst.a);
			JamValueRef ptr = JamLLVMBuildExtractValue(
			    lctx.ctx.getBuilder(), base, 0, "slice.ptr");
			JamValueRef gep = JamLLVMBuildPtrGEP(
			    lctx.ctx.getBuilder(), elemLLVMTy, ptr, idxVal, "idx.gep");
			return JamLLVMBuildLoad(lctx.ctx.getBuilder(), elemLLVMTy, gep,
			                        "idx");
		}
		if (basek.kind == TypeKind::PtrMany) {
			JamValueRef base = emitInst(lctx, inst.a);
			JamValueRef gep = JamLLVMBuildPtrGEP(
			    lctx.ctx.getBuilder(), elemLLVMTy, base, idxVal, "idx.gep");
			return JamLLVMBuildLoad(lctx.ctx.getBuilder(), elemLLVMTy, gep,
			                        "idx");
		}
		// Array case: spill the SSA value to a temp alloca and GEP.
		JamValueRef base = emitInst(lctx, inst.a);
		uint64_t align = lctx.ctx.typeAlign(baseInst.ty);
		JamValueRef tmp = JamLLVMBuildAlloca(lctx.ctx.getBuilder(),
		                                     baseLLVMTy, align, "arr.tmp");
		JamLLVMBuildStore(lctx.ctx.getBuilder(), base, tmp);
		JamValueRef gep = JamLLVMBuildArrayGEP(
		    lctx.ctx.getBuilder(), baseLLVMTy, tmp, idxVal, "idx.gep");
		return JamLLVMBuildLoad(lctx.ctx.getBuilder(), elemLLVMTy, gep,
		                        "idx");
	}
	case JirTag::AddrOf: {
		// AstGen plants the alloca's own JirRef into `a` when taking
		// the address of a Variable, so the cached LLVM value IS the
		// pointer we want.
		return emitInst(lctx, inst.a);
	}
	case JirTag::FieldAddr: {
		// `a` references a pointer-producing instruction (Alloca,
		// AddrOf, FieldAddr, IndexAddr, or a by-pointer Param). Its
		// `ty` tells us the pointee struct type — for Alloca and
		// flagged Param, `ty` IS the pointee; for the rest, `ty` is
		// PtrSingle(pointee).
		JamValueRef basePtr = emitInst(lctx, inst.a);
		const JirInst &baseInst = lctx.jfn.getInst(inst.a);
		bool baseIsAllocaLike =
		    baseInst.tag == JirTag::Alloca ||
		    (baseInst.tag == JirTag::Param && (baseInst.flags & 1));
		TypeIdx pointeeTy;
		if (baseIsAllocaLike) {
			pointeeTy = baseInst.ty;
		} else {
			const TypeKey &k =
			    lctx.ctx.getTypePool().get(baseInst.ty);
			if (k.kind != TypeKind::PtrSingle &&
			    k.kind != TypeKind::PtrMany) {
				throw std::runtime_error(
				    "jirCodegen: FieldAddr base must be a pointer");
			}
			pointeeTy = static_cast<TypeIdx>(k.a);
		}
		JamTypeRef structTy = lctx.ctx.getLLVMType(pointeeTy);
		return JamLLVMBuildStructGEP(lctx.ctx.getBuilder(), structTy,
		                             basePtr, inst.b, "fieldp");
	}
	case JirTag::IndexAddr: {
		JamValueRef basePtr = emitInst(lctx, inst.a);
		const JirInst &baseInst = lctx.jfn.getInst(inst.a);
		TypeIdx pointeeTy;
		bool baseIsAllocaLike =
		    baseInst.tag == JirTag::Alloca ||
		    (baseInst.tag == JirTag::Param && (baseInst.flags & 1));
		if (baseIsAllocaLike) {
			pointeeTy = baseInst.ty;
		} else {
			const TypeKey &k =
			    lctx.ctx.getTypePool().get(baseInst.ty);
			pointeeTy = (k.kind == TypeKind::PtrSingle ||
			             k.kind == TypeKind::PtrMany)
			                 ? static_cast<TypeIdx>(k.a)
			                 : baseInst.ty;
		}
		const TypeKey &pk = lctx.ctx.getTypePool().get(pointeeTy);
		JamValueRef idxVal = emitInst(lctx, inst.b);
		JamTypeRef i64 = lctx.ctx.getInt64Type();
		if (JamLLVMTypeOf(idxVal) != i64) {
			idxVal = JamLLVMBuildIntCast(lctx.ctx.getBuilder(), idxVal,
			                             i64, false, "idx.cast");
		}
		const TypeKey &resKey = lctx.ctx.getTypePool().get(inst.ty);
		TypeIdx elemTy = static_cast<TypeIdx>(resKey.a);
		JamTypeRef elemLLVM = lctx.ctx.getLLVMType(elemTy);
		if (pk.kind == TypeKind::Array) {
			JamTypeRef arrLLVM = lctx.ctx.getLLVMType(pointeeTy);
			return JamLLVMBuildArrayGEP(lctx.ctx.getBuilder(), arrLLVM,
			                            basePtr, idxVal, "idx.addr");
		}
		// Slice / PtrMany / PtrSingle to a single elem: treat as a
		// many-item pointer and PtrGEP by element-sized stride.
		return JamLLVMBuildPtrGEP(lctx.ctx.getBuilder(), elemLLVM,
		                          basePtr, idxVal, "idx.addr");
	}
	case JirTag::Deref: {
		JamValueRef ptr = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildLoad(lctx.ctx.getBuilder(), ty, ptr, "deref");
	}
	case JirTag::DropBinding: {
		// `a` is the binding's alloca JirRef (already a pointer at
		// LLVM); `b` is the StringIdx of the drop fn's LLVM symbol.
		// Mechanically emit `call void <symbol>(ptr <alloca>)`.
		JamValueRef bindingPtr = emitInst(lctx, inst.a);
		StringIdx symId = static_cast<StringIdx>(inst.b);
		const std::string &symbol =
		    lctx.ctx.getStringPool().get(symId);
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), symbol.c_str());
		if (!f) {
			throw std::runtime_error("jirCodegen: drop callee `" +
			                         symbol + "` not declared");
		}
		JamValueRef args[1] = {bindingPtr};
		JamLLVMBuildCall(lctx.ctx.getBuilder(), f, args, 1, "");
		return nullptr;
	}
	case JirTag::EnumPayload: {
		// `a` is a pointer to a tagged enum struct {tag, payloadDriver,
		// [extra...]}. `b` is the byte offset *within the payload
		// area* to load from (0 for the first payload field; layout-
		// aware for subsequent ones).
		JamValueRef enumPtr = emitInst(lctx, inst.a);
		const JirInst &basePtrInst = lctx.jfn.getInst(inst.a);
		JamTypeRef enumTy = lctx.ctx.getLLVMType(basePtrInst.ty);
		JamValueRef payloadAreaPtr = JamLLVMBuildStructGEP(
		    lctx.ctx.getBuilder(), enumTy, enumPtr, 1, "enum.payload");
		JamValueRef fieldPtr = payloadAreaPtr;
		if (inst.b != 0) {
			JamValueRef off = JamLLVMConstInt(
			    lctx.ctx.getInt64Type(),
			    static_cast<uint64_t>(inst.b), false);
			fieldPtr = JamLLVMBuildPtrGEP(
			    lctx.ctx.getBuilder(),
			    lctx.ctx.getInt8Type(), payloadAreaPtr, off,
			    "enum.payload.off");
		}
		JamTypeRef fieldTy = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildLoad(lctx.ctx.getBuilder(), fieldTy, fieldPtr,
		                        "enum.payload.val");
	}
	// === Function call ===
	case JirTag::Call: {
		// `inst.a` is the StringIdx of the LLVM symbol name —
		// astgen has already done the mangling (test functions get
		// `__test_`; `fn drop(self: mut T)` becomes `__drop_T`;
		// instantiated cloned methods keep their qualified
		// `Vec__i32.push` form). Codegen does a single lookup.
		StringIdx calleeId = static_cast<StringIdx>(inst.a);
		const std::string &symbol =
		    lctx.ctx.getStringPool().get(calleeId);
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), symbol.c_str());
		if (!f) {
			throw std::runtime_error("jirCodegen: unknown callee `" +
			                         symbol + "`");
		}
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		uint32_t argCount = lctx.jfn.getExtra(extra);
		std::vector<JamValueRef> args;
		args.reserve(argCount);
		for (uint32_t i = 0; i < argCount; i++) {
			JirRef ar =
			    static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			args.push_back(emitInst(lctx, ar));
		}
		const char *resultName = (inst.ty == kNoType) ? "" : "call";
		return JamLLVMBuildCall(lctx.ctx.getBuilder(), f, args.data(),
		                        argCount, resultName);
	}
	// === Control ===
	case JirTag::Br: {
		JirBlockRef target = static_cast<JirBlockRef>(inst.a);
		JamLLVMBuildBr(lctx.ctx.getBuilder(), lctx.blockMap.at(target));
		return nullptr;
	}
	case JirTag::CondBr: {
		JamValueRef cond = emitInst(lctx, inst.a);
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		JirBlockRef thenB =
		    static_cast<JirBlockRef>(lctx.jfn.getExtra(extra));
		JirBlockRef elseB =
		    static_cast<JirBlockRef>(lctx.jfn.getExtra(extra + 1));
		JamLLVMBuildCondBr(lctx.ctx.getBuilder(), cond,
		                  lctx.blockMap.at(thenB),
		                  lctx.blockMap.at(elseB));
		return nullptr;
	}
	case JirTag::Ret: {
		if (inst.a == kNoJirRef) {
			JamLLVMBuildRetVoid(lctx.ctx.getBuilder());
		} else {
			JamValueRef v = emitInst(lctx, inst.a);
			JamLLVMBuildRet(lctx.ctx.getBuilder(), v);
		}
		return nullptr;
	}
	case JirTag::Unreachable:
		JamLLVMBuildUnreachable(lctx.ctx.getBuilder());
		return nullptr;
	default:
		throw std::runtime_error(
		    "jirCodegen: unsupported JIR tag (tag = " +
		    std::to_string(static_cast<int>(inst.tag)) + ")");
	}
}

}  // namespace

void jirDeclarePrototype(const JirFunction &jfn, JamCodegenContext &ctx) {
	// Direct-return ABI: large aggregate returns still go through the
	// legacy declarePrototype path. JIR-declared functions stay
	// by-value for returns; sret is a future extension.
	JamTypeRef retType = (jfn.returnType == kNoType)
	                         ? ctx.getVoidType()
	                         : ctx.getLLVMType(jfn.returnType);
	std::vector<JamTypeRef> argTypes;
	argTypes.reserve(jfn.paramTypes.size());
	for (size_t i = 0; i < jfn.paramTypes.size(); i++) {
		TypeIdx t = jfn.paramTypes[i];
		// Choose ABI per-mode. Today's LLVM backend lowers Mut and
		// Move parameters by pointer; Let/Const pass by value. A
		// non-LLVM backend can read `paramModes` and pick its own
		// strategy (e.g. register-passing small Move structs).
		ParamMode mode = i < jfn.paramModes.size()
		                     ? jfn.paramModes[i]
		                     : ParamMode::Let;
		bool byPtr = mode == ParamMode::Mut || mode == ParamMode::Move;
		if (byPtr) {
			JamTypeRef pointee = ctx.getLLVMType(t);
			argTypes.push_back(JamLLVMPointerType(pointee, 0));
		} else {
			argTypes.push_back(ctx.getLLVMType(t));
		}
	}
	JamTypeRef ft = JamLLVMFunctionType(
	    retType, argTypes.data(), static_cast<unsigned>(argTypes.size()),
	    jfn.isVarArgs);
	JamFunctionRef f =
	    JamLLVMAddFunction(ctx.getModule(), jfn.name.c_str(), ft);
	JamLLVMApplyDefaultFnAttrs(f, jfn.isExtern);
	if (jfn.isExtern || jfn.isExport || jfn.name == "main") {
		JamLLVMSetLinkage(reinterpret_cast<JamValueRef>(f),
		                  JAM_LINKAGE_EXTERNAL);
		JamLLVMSetFunctionCallConv(f, JAM_CALLCONV_C);
	} else {
		JamLLVMSetLinkage(reinterpret_cast<JamValueRef>(f),
		                  JAM_LINKAGE_INTERNAL);
	}
}

void jirDefineBody(const JirFunction &jfn, JamCodegenContext &ctx) {
	if (jfn.isExtern) return;

	JamFunctionRef f = JamLLVMGetFunction(ctx.getModule(), jfn.name.c_str());
	if (!f) {
		throw std::runtime_error("jirDefineBody: prototype missing for `" +
		                         jfn.name + "`");
	}

	JirCodegenCtx lctx{jfn, ctx, {}, {}};

	// Create LLVM blocks first so terminators can resolve forward
	// references. Skip the sentinel block at index 0.
	for (JirBlockRef b = 1; b < jfn.blocks.size(); b++) {
		JamBasicBlockRef bb =
		    JamLLVMAppendBasicBlock(f, jfn.getBlock(b).name.c_str());
		lctx.blockMap[b] = bb;
	}

	// Emit instructions block-by-block.
	for (JirBlockRef b = 1; b < jfn.blocks.size(); b++) {
		JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), lctx.blockMap[b]);
		for (JirRef r : jfn.getBlock(b).insts) {
			JamValueRef v = emitInst(lctx, r);
			if (v != nullptr) lctx.valueMap[r] = v;
		}
	}
}
