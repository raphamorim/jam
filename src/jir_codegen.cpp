/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "jir_codegen.h"

#include "abi.h"
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

// Forward declarations of the shared ABI classifier helpers used by
// both prototype emission (`jirDeclarePrototype`) and per-instruction
// lowering (`Param`, `Ret`, `Call`). Definitions are below — kept near
// `jirDeclarePrototype` so the prototype + caller-side ABI policy
// lives in one block.
jam::abi::ReturnABI jirClassifyReturn(const JirFunction &jfn,
                                      const JamCodegenContext &ctx);
bool jirReturnIsSret(const JirFunction &jfn, const JamCodegenContext &ctx);
jam::abi::ParamABI jirClassifyParam(const JirFunction &jfn, size_t i,
                                    const JamCodegenContext &ctx);

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
	case JirTag::Poison: {
		// Error recovery: astgen pushed a diagnostic and synthesized
		// this typed placeholder so the rest of the function could
		// still be walked. Codegen lowers it to LLVM `undef`, but in
		// practice we should never reach here — the driver checks
		// `Diagnostics::hasErrors()` after astgen and bails before
		// running jir_codegen.
		JamTypeRef ty = (inst.ty == kNoType) ? lctx.ctx.getVoidType()
		                                     : lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMGetUndef(ty);
	}
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
		return JamLLVMConstInt(lctx.ctx.getInt1Type(), inst.a != 0 ? 1 : 0,
		                       false);
	case JirTag::Str: {
		StringIdx sid = static_cast<StringIdx>(inst.a);
		const std::string &val = lctx.ctx.getStringPool().get(sid);
		JamValueRef strConst =
		    JamLLVMConstString(lctx.ctx.getContext(), val.c_str(),
		                       static_cast<unsigned>(val.length()), true);
		JamTypeRef arrTy = JamLLVMArrayType(
		    lctx.ctx.getInt8Type(), static_cast<uint64_t>(val.length() + 1));
		JamValueRef strGlobal =
		    JamLLVMAddGlobal(lctx.ctx.getModule(), arrTy, "str");
		JamLLVMSetGlobalConstant(strGlobal, true);
		JamLLVMSetInitializer(strGlobal, strConst);

		JamTypeRef sliceTy = lctx.ctx.getLLVMType(inst.ty);
		JamTypeRef i8PtrTy = JamLLVMPointerType(lctx.ctx.getInt8Type(), 0);
		JamValueRef strPtr = JamLLVMBuildBitCast(lctx.ctx.getBuilder(),
		                                         strGlobal, i8PtrTy, "str_ptr");
		JamValueRef slice = JamLLVMGetUndef(sliceTy);
		slice = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), slice, strPtr, 0,
		                                "slice_ptr");
		slice = JamLLVMBuildInsertValue(
		    lctx.ctx.getBuilder(), slice,
		    JamLLVMConstInt(lctx.ctx.getInt64Type(),
		                    static_cast<uint64_t>(val.length()), false),
		    1, "slice_len");
		return slice;
	}
	case JirTag::Param: {
		// flags & 1 means the param is ByPointer (mut / move always;
		// let / const for aggregates > kByValueMaxBytes). We hand back
		// the LLVM pointer directly so the local map can use it as
		// the alloca-equivalent for field access / self-style reads.
		// When the function uses sret, the source-level param at JIR
		// index `i` lives at LLVM index `i + 1` (the sret slot owns
		// LLVM index 0).
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), lctx.jfn.name.c_str());
		unsigned argOffset = jirReturnIsSret(lctx.jfn, lctx.ctx) ? 1u : 0u;
		return JamLLVMGetParam(f, inst.a + argOffset);
	}
	case JirTag::Alloca: {
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		uint64_t align = lctx.ctx.typeAlign(inst.ty);
		std::string nm = "v" + std::to_string(r);
		return JamLLVMBuildAlloca(lctx.ctx.getBuilder(), ty, align, nm.c_str());
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
		return buildICmp(lctx, JAM_ICMP_EQ, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "eq");
	case JirTag::ICmpNe:
		return buildICmp(lctx, JAM_ICMP_NE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ne");
	case JirTag::ICmpSlt:
		return buildICmp(lctx, JAM_ICMP_SLT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "slt");
	case JirTag::ICmpSle:
		return buildICmp(lctx, JAM_ICMP_SLE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "sle");
	case JirTag::ICmpSgt:
		return buildICmp(lctx, JAM_ICMP_SGT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "sgt");
	case JirTag::ICmpSge:
		return buildICmp(lctx, JAM_ICMP_SGE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "sge");
	case JirTag::ICmpUlt:
		return buildICmp(lctx, JAM_ICMP_ULT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ult");
	case JirTag::ICmpUle:
		return buildICmp(lctx, JAM_ICMP_ULE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ule");
	case JirTag::ICmpUgt:
		return buildICmp(lctx, JAM_ICMP_UGT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ugt");
	case JirTag::ICmpUge:
		return buildICmp(lctx, JAM_ICMP_UGE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "uge");
	// === Float comparison ===
	case JirTag::FCmpOeq:
		return buildFCmp(lctx, JAM_FCMP_OEQ, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "oeq");
	case JirTag::FCmpOne:
		return buildFCmp(lctx, JAM_FCMP_UNE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "one");
	case JirTag::FCmpOlt:
		return buildFCmp(lctx, JAM_FCMP_OLT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "olt");
	case JirTag::FCmpOle:
		return buildFCmp(lctx, JAM_FCMP_OLE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ole");
	case JirTag::FCmpOgt:
		return buildFCmp(lctx, JAM_FCMP_OGT, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "ogt");
	case JirTag::FCmpOge:
		return buildFCmp(lctx, JAM_FCMP_OGE, emitInst(lctx, inst.a),
		                 emitInst(lctx, inst.b), "oge");
	// === Bitwise / shift ===
	case JirTag::BitAnd:
		return JamLLVMBuildAnd(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "and");
	case JirTag::BitOr:
		return JamLLVMBuildOr(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                      emitInst(lctx, inst.b), "or");
	case JirTag::BitXor:
		return JamLLVMBuildXor(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "xor");
	case JirTag::Shl:
		return JamLLVMBuildShl(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                       emitInst(lctx, inst.b), "shl");
	case JirTag::AShr:
		return JamLLVMBuildAShr(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                        emitInst(lctx, inst.b), "ashr");
	case JirTag::LShr:
		return JamLLVMBuildLShr(lctx.ctx.getBuilder(), emitInst(lctx, inst.a),
		                        emitInst(lctx, inst.b), "lshr");
	case JirTag::BitNot: {
		// LLVM has no NOT op; emit XOR with all-ones of the operand's type.
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		JamValueRef ones = JamLLVMConstInt(ty, ~static_cast<uint64_t>(0), true);
		return JamLLVMBuildXor(lctx.ctx.getBuilder(), v, ones, "not");
	}
	case JirTag::LogNot: {
		// Boolean inversion: XOR with i1 1. Operand is already i1.
		JamValueRef v = emitInst(lctx, inst.a);
		JamValueRef one = JamLLVMConstInt(lctx.ctx.getInt1Type(), 1, false);
		return JamLLVMBuildXor(lctx.ctx.getBuilder(), v, one, "lnot");
	}
	// === Type conversions ===
	case JirTag::ZExt: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntCast(lctx.ctx.getBuilder(), v, ty, false, "zext");
	}
	case JirTag::SExt: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntCast(lctx.ctx.getBuilder(), v, ty, true, "sext");
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
	case JirTag::PtrToInt: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildPtrToInt(lctx.ctx.getBuilder(), v, ty, "p2i");
	}
	case JirTag::IntToPtr: {
		JamValueRef v = emitInst(lctx, inst.a);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildIntToPtr(lctx.ctx.getBuilder(), v, ty, "i2p");
	}
	case JirTag::FnRef: {
		// `inst.a` carries the StringIdx of the LLVM symbol name. We
		// resolve to the LLVM Function (already a ptr-typed Value),
		// then lower to ptrtoint to match the JIR's u64 result type.
		// This mirrors Rust's `my_fn as u64` lowering.
		StringIdx nameId = static_cast<StringIdx>(inst.a);
		const std::string &name = lctx.ctx.getStringPool().get(nameId);
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), name.c_str());
		if (!f) {
			throw std::runtime_error("jirCodegen: unknown fn-ref `" + name +
			                         "`");
		}
		JamValueRef fnVal = JamLLVMFunctionAsValue(f);
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		return JamLLVMBuildPtrToInt(lctx.ctx.getBuilder(), fnVal, ty,
		                            "fnref.u64");
	}
	// === Aggregates ===
	case JirTag::StructLit: {
		JamTypeRef ty = lctx.ctx.getLLVMType(inst.ty);
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		uint32_t count = lctx.jfn.getExtra(extra);
		JamValueRef agg = JamLLVMGetUndef(ty);
		for (uint32_t i = 0; i < count; i++) {
			JirRef fr = static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			JamValueRef fv = emitInst(lctx, fr);
			agg = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), agg, fv, i,
			                              "field");
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
			JirRef er = static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			JamValueRef ev = emitInst(lctx, er);
			agg = JamLLVMBuildInsertValue(lctx.ctx.getBuilder(), agg, ev, i,
			                              "elem");
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
			idxVal = JamLLVMBuildIntCast(lctx.ctx.getBuilder(), idxVal, i64,
			                             false, "idx.cast");
		}

		JamTypeRef baseLLVMTy = lctx.ctx.getLLVMType(baseInst.ty);
		JamTypeRef elemLLVMTy = lctx.ctx.getLLVMType(inst.ty);

		if (basek.kind == TypeKind::Slice) {
			// SSA slice value: extract the pointer field (0), GEP by elem.
			JamValueRef base = emitInst(lctx, inst.a);
			JamValueRef ptr = JamLLVMBuildExtractValue(lctx.ctx.getBuilder(),
			                                           base, 0, "slice.ptr");
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
		JamValueRef tmp = JamLLVMBuildAlloca(lctx.ctx.getBuilder(), baseLLVMTy,
		                                     align, "arr.tmp");
		JamLLVMBuildStore(lctx.ctx.getBuilder(), base, tmp);
		JamValueRef gep = JamLLVMBuildArrayGEP(
		    lctx.ctx.getBuilder(), baseLLVMTy, tmp, idxVal, "idx.gep");
		return JamLLVMBuildLoad(lctx.ctx.getBuilder(), elemLLVMTy, gep, "idx");
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
			const TypeKey &k = lctx.ctx.getTypePool().get(baseInst.ty);
			if (k.kind != TypeKind::PtrSingle && k.kind != TypeKind::PtrMany) {
				throw std::runtime_error(
				    "jirCodegen: FieldAddr base must be a pointer");
			}
			pointeeTy = static_cast<TypeIdx>(k.a);
		}
		JamTypeRef structTy = lctx.ctx.getLLVMType(pointeeTy);
		return JamLLVMBuildStructGEP(lctx.ctx.getBuilder(), structTy, basePtr,
		                             inst.b, "fieldp");
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
			const TypeKey &k = lctx.ctx.getTypePool().get(baseInst.ty);
			pointeeTy =
			    (k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany)
			        ? static_cast<TypeIdx>(k.a)
			        : baseInst.ty;
		}
		const TypeKey &pk = lctx.ctx.getTypePool().get(pointeeTy);
		JamValueRef idxVal = emitInst(lctx, inst.b);
		JamTypeRef i64 = lctx.ctx.getInt64Type();
		if (JamLLVMTypeOf(idxVal) != i64) {
			idxVal = JamLLVMBuildIntCast(lctx.ctx.getBuilder(), idxVal, i64,
			                             false, "idx.cast");
		}
		const TypeKey &resKey = lctx.ctx.getTypePool().get(inst.ty);
		TypeIdx elemTy = static_cast<TypeIdx>(resKey.a);
		JamTypeRef elemLLVM = lctx.ctx.getLLVMType(elemTy);
		if (pk.kind == TypeKind::Array) {
			JamTypeRef arrLLVM = lctx.ctx.getLLVMType(pointeeTy);
			return JamLLVMBuildArrayGEP(lctx.ctx.getBuilder(), arrLLVM, basePtr,
			                            idxVal, "idx.addr");
		}
		// Slice / PtrMany / PtrSingle to a single elem: treat as a
		// many-item pointer and PtrGEP by element-sized stride.
		return JamLLVMBuildPtrGEP(lctx.ctx.getBuilder(), elemLLVM, basePtr,
		                          idxVal, "idx.addr");
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
		const std::string &symbol = lctx.ctx.getStringPool().get(symId);
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), symbol.c_str());
		if (!f) {
			throw std::runtime_error("jirCodegen: drop callee `" + symbol +
			                         "` not declared");
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
			    lctx.ctx.getInt64Type(), static_cast<uint64_t>(inst.b), false);
			fieldPtr = JamLLVMBuildPtrGEP(
			    lctx.ctx.getBuilder(), lctx.ctx.getInt8Type(), payloadAreaPtr,
			    off, "enum.payload.off");
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
		const std::string &symbol = lctx.ctx.getStringPool().get(calleeId);
		JamFunctionRef f =
		    JamLLVMGetFunction(lctx.ctx.getModule(), symbol.c_str());
		if (!f) {
			throw std::runtime_error("jirCodegen: unknown callee `" + symbol +
			                         "`");
		}
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		uint32_t argCount = lctx.jfn.getExtra(extra);
		bool calleeUsesSret = JamLLVMFunctionUsesSret(f);
		std::vector<JamValueRef> args;
		args.reserve(argCount + (calleeUsesSret ? 1u : 0u));
		JamValueRef sretSlot = nullptr;
		if (calleeUsesSret) {
			// Allocate caller-owned storage for the result, prepend
			// its pointer as the hidden first argument. The slot type
			// comes from the callee's sret attribute, which carries
			// the pointee type. Result is loaded back after the call.
			JamTypeRef pointee = JamLLVMFunctionSretPointeeType(f);
			uint64_t align = lctx.ctx.typeAlign(inst.ty);
			sretSlot = JamLLVMBuildAlloca(lctx.ctx.getBuilder(), pointee, align,
			                              "sret.slot");
			args.push_back(sretSlot);
		}
		for (uint32_t i = 0; i < argCount; i++) {
			JirRef ar = static_cast<JirRef>(lctx.jfn.getExtra(extra + 1 + i));
			args.push_back(emitInst(lctx, ar));
		}
		if (calleeUsesSret) {
			// The LLVM call itself returns void; the value lives in
			// the sret slot. Load it so the JirRef → LLVM value map
			// holds the materialized return value.
			JamLLVMBuildCall(lctx.ctx.getBuilder(), f, args.data(),
			                 static_cast<unsigned>(args.size()), "");
			JamTypeRef retLlvmTy = lctx.ctx.getLLVMType(inst.ty);
			return JamLLVMBuildLoad(lctx.ctx.getBuilder(), retLlvmTy, sretSlot,
			                        "sret.val");
		}
		const char *resultName = (inst.ty == kNoType) ? "" : "call";
		return JamLLVMBuildCall(lctx.ctx.getBuilder(), f, args.data(),
		                        static_cast<unsigned>(args.size()), resultName);
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
		JirBlockRef thenB = static_cast<JirBlockRef>(lctx.jfn.getExtra(extra));
		JirBlockRef elseB =
		    static_cast<JirBlockRef>(lctx.jfn.getExtra(extra + 1));
		JamLLVMBuildCondBr(lctx.ctx.getBuilder(), cond, lctx.blockMap.at(thenB),
		                   lctx.blockMap.at(elseB));
		return nullptr;
	}
	case JirTag::Switch: {
		// Multi-way branch over an integer scrutinee. Layout per
		// jir.h: extra[0] = default block, extra[1] = caseCount,
		// extra[2 + i*4 + 0..3] = {lo32, hi32, signed, block} for
		// each case. The case constants must match the scrut's LLVM
		// type — we read it from the scrut JirInst's `ty`.
		//
		// Specialization, mirroring rustc_codegen_ssa/mir/block.rs::
		// codegen_switchint_terminator: collapse Switch back to a
		// cond_br when the shape is trivial.
		//   * 1 case + default → ICmpEq + CondBr (always valid: the
		//     default is the cond_br's `else` whether it's a live
		//     fall-through block or an unreachable one).
		// Rust also collapses 2-case bool shapes — that requires
		// knowing the default block is unreachable (i.e. the match is
		// exhaustive on {0, 1}). astgen doesn't track exhaustiveness
		// yet, so we skip it and let LLVM's middle-end fold the
		// pattern in -O1+.
		JamValueRef scrut = emitInst(lctx, inst.a);
		const JirInst &scrutInst = lctx.jfn.getInst(inst.a);
		JamTypeRef scrutLlvmTy = lctx.ctx.getLLVMType(scrutInst.ty);
		JirExtraIdx extra = static_cast<JirExtraIdx>(inst.b);
		JirBlockRef defaultB =
		    static_cast<JirBlockRef>(lctx.jfn.getExtra(extra));
		uint32_t caseCount = lctx.jfn.getExtra(extra + 1);

		auto readCase = [&](uint32_t i) {
			JirExtraIdx base = extra + 2 + i * 4;
			uint64_t lo = lctx.jfn.getExtra(base + 0);
			uint64_t hi = lctx.jfn.getExtra(base + 1);
			bool isSigned = lctx.jfn.getExtra(base + 2) != 0;
			JirBlockRef caseB =
			    static_cast<JirBlockRef>(lctx.jfn.getExtra(base + 3));
			uint64_t bits = lo | (hi << 32);
			JamValueRef caseVal = JamLLVMConstInt(scrutLlvmTy, bits, isSigned);
			return std::tuple{caseVal, caseB, bits};
		};

		if (caseCount == 1) {
			auto [caseVal, caseB, _bits] = readCase(0);
			JamValueRef cmp =
			    buildICmp(lctx, JAM_ICMP_EQ, scrut, caseVal, "match.eq");
			JamLLVMBuildCondBr(lctx.ctx.getBuilder(), cmp,
			                   lctx.blockMap.at(caseB),
			                   lctx.blockMap.at(defaultB));
			return nullptr;
		}

		JamValueRef sw =
		    JamLLVMBuildSwitch(lctx.ctx.getBuilder(), scrut,
		                       lctx.blockMap.at(defaultB), caseCount);
		for (uint32_t i = 0; i < caseCount; i++) {
			auto [caseVal, caseB, _bits] = readCase(i);
			JamLLVMAddCase(sw, caseVal, lctx.blockMap.at(caseB));
		}
		return nullptr;
	}
	case JirTag::Ret: {
		// sret return: store the value through the leading hidden
		// `ptr sret(%T)` arg and emit `ret void`. The caller already
		// owns the slot, so we don't allocate anything here.
		if (jirReturnIsSret(lctx.jfn, lctx.ctx)) {
			JamFunctionRef f =
			    JamLLVMGetFunction(lctx.ctx.getModule(), lctx.jfn.name.c_str());
			JamValueRef sretSlot = JamLLVMGetParam(f, 0);
			if (inst.a != kNoJirRef) {
				JamValueRef v = emitInst(lctx, inst.a);
				JamLLVMBuildStore(lctx.ctx.getBuilder(), v, sretSlot);
			}
			JamLLVMBuildRetVoid(lctx.ctx.getBuilder());
			return nullptr;
		}
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
		throw std::runtime_error("jirCodegen: unsupported JIR tag (tag = " +
		                         std::to_string(static_cast<int>(inst.tag)) +
		                         ")");
	}
}

// Single ABI source-of-truth used by both prototype emission and
// call/body lowering. extern and test functions skip the classifier
// for returns (extern follows the C ABI literally, tests are always
// nullary-void). Classification is a pure function of
// (mode, type, fn-kind), so call sites and definitions never disagree.
jam::abi::ReturnABI jirClassifyReturn(const JirFunction &jfn,
                                      const JamCodegenContext &ctx) {
	if (jfn.isExtern || jfn.isTest) {
		return jam::abi::ReturnABI{jam::abi::ReturnABI::Kind::Direct,
		                           (jfn.isTest || jfn.returnType == kNoType)
		                               ? ctx.getVoidType()
		                               : ctx.getLLVMType(jfn.returnType),
		                           0};
	}
	if (jfn.returnType == kNoType) {
		return jam::abi::ReturnABI{jam::abi::ReturnABI::Kind::Direct,
		                           ctx.getVoidType(), 0};
	}
	return jam::abi::classifyReturn(jfn.returnType, ctx);
}

// Does the LLVM signature have a leading `ptr sret(%T)` argument?
bool jirReturnIsSret(const JirFunction &jfn, const JamCodegenContext &ctx) {
	return jirClassifyReturn(jfn, ctx).kind ==
	       jam::abi::ReturnABI::Kind::Indirect;
}

// Param classification used by both prototype and call site. extern
// preserves the user-written type verbatim (the user already wrote
// what they want at the FFI boundary, e.g. `*const T` for an out-ptr).
jam::abi::ParamABI jirClassifyParam(const JirFunction &jfn, size_t i,
                                    const JamCodegenContext &ctx) {
	TypeIdx t = jfn.paramTypes[i];
	if (jfn.isExtern) {
		return jam::abi::ParamABI{jam::abi::ParamABI::Kind::ByValue,
		                          ctx.getLLVMType(t), 0};
	}
	ParamMode mode =
	    i < jfn.paramModes.size() ? jfn.paramModes[i] : ParamMode::Let;
	return jam::abi::classifyParam(mode, t, ctx);
}

}  // namespace

void jirDeclarePrototype(const JirFunction &jfn, JamCodegenContext &ctx) {
	jam::abi::ReturnABI rabi = jirClassifyReturn(jfn, ctx);
	bool sret = rabi.kind == jam::abi::ReturnABI::Kind::Indirect;

	std::vector<JamTypeRef> argTypes;
	argTypes.reserve(jfn.paramTypes.size() + (sret ? 1u : 0u));
	if (sret) {
		// Leading `ptr` carries the caller-owned return slot. Attribute
		// (sret + align) applied after AddFunction below.
		argTypes.push_back(
		    JamLLVMPointerType(ctx.getLLVMType(jfn.returnType), 0));
	}
	for (size_t i = 0; i < jfn.paramTypes.size(); i++) {
		jam::abi::ParamABI pabi = jirClassifyParam(jfn, i, ctx);
		if (pabi.kind == jam::abi::ParamABI::Kind::ByPointer) {
			argTypes.push_back(
			    JamLLVMPointerType(ctx.getLLVMType(jfn.paramTypes[i]), 0));
		} else {
			argTypes.push_back(pabi.llvmType);
		}
	}

	JamTypeRef retType = sret ? ctx.getVoidType() : rabi.directType;
	JamTypeRef ft = JamLLVMFunctionType(retType, argTypes.data(),
	                                    static_cast<unsigned>(argTypes.size()),
	                                    jfn.isVarArgs);
	JamFunctionRef f =
	    JamLLVMAddFunction(ctx.getModule(), jfn.name.c_str(), ft);
	JamLLVMApplyDefaultFnAttrs(f, jfn.isExtern);
	if (jfn.returnType == BuiltinType::NoReturn) {
		JamLLVMSetFunctionNoReturn(f);
	}
	if (sret) {
		JamLLVMAddParamAttrSret(f, 0, ctx.getLLVMType(jfn.returnType),
		                        rabi.sretAlign);
	}

	bool externalLinkage = jfn.isExtern || jfn.isExport || jfn.name == "main";
	if (externalLinkage) {
		JamLLVMSetLinkage(reinterpret_cast<JamValueRef>(f),
		                  JAM_LINKAGE_EXTERNAL);
		JamLLVMSetFunctionCallConv(f, JAM_CALLCONV_C);
		// C ABI requires bool args / returns to be zero-extended to
		// the underlying register width. Internal-linkage callers use
		// our own ABI, so we skip zext there.
		const unsigned argOffset = sret ? 1u : 0u;
		for (size_t i = 0; i < jfn.paramTypes.size(); i++) {
			if (jfn.paramTypes[i] == BuiltinType::Bool) {
				JamLLVMAddParamAttrZeroExt(f, static_cast<unsigned>(i) +
				                                  argOffset);
			}
		}
		if (!sret && jfn.returnType == BuiltinType::Bool) {
			JamLLVMAddRetAttrZeroExt(f);
		}
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
