/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "jir_verify.h"

#include "ast_flat.h"  // for TypePool / TypeKey / TypeKind

#include <string>
#include <vector>

namespace {

bool isTerminator(JirTag t) {
	return t == JirTag::Br || t == JirTag::CondBr || t == JirTag::Switch ||
	       t == JirTag::Ret || t == JirTag::Unreachable;
}

const char *tagName(JirTag t) {
	switch (t) {
	case JirTag::Invalid:
		return "Invalid";
	case JirTag::Int:
		return "Int";
	case JirTag::Float:
		return "Float";
	case JirTag::Bool:
		return "Bool";
	case JirTag::Str:
		return "Str";
	case JirTag::Poison:
		return "Poison";
	case JirTag::Alloca:
		return "Alloca";
	case JirTag::Load:
		return "Load";
	case JirTag::Store:
		return "Store";
	case JirTag::Add:
		return "Add";
	case JirTag::Sub:
		return "Sub";
	case JirTag::Mul:
		return "Mul";
	case JirTag::SDiv:
		return "SDiv";
	case JirTag::UDiv:
		return "UDiv";
	case JirTag::SRem:
		return "SRem";
	case JirTag::URem:
		return "URem";
	case JirTag::FAdd:
		return "FAdd";
	case JirTag::FSub:
		return "FSub";
	case JirTag::FMul:
		return "FMul";
	case JirTag::FDiv:
		return "FDiv";
	case JirTag::FRem:
		return "FRem";
	case JirTag::FNeg:
		return "FNeg";
	case JirTag::ICmpEq:
		return "ICmpEq";
	case JirTag::ICmpNe:
		return "ICmpNe";
	case JirTag::ICmpSlt:
		return "ICmpSlt";
	case JirTag::ICmpSle:
		return "ICmpSle";
	case JirTag::ICmpSgt:
		return "ICmpSgt";
	case JirTag::ICmpSge:
		return "ICmpSge";
	case JirTag::ICmpUlt:
		return "ICmpUlt";
	case JirTag::ICmpUle:
		return "ICmpUle";
	case JirTag::ICmpUgt:
		return "ICmpUgt";
	case JirTag::ICmpUge:
		return "ICmpUge";
	case JirTag::FCmpOeq:
		return "FCmpOeq";
	case JirTag::FCmpOne:
		return "FCmpOne";
	case JirTag::FCmpOlt:
		return "FCmpOlt";
	case JirTag::FCmpOle:
		return "FCmpOle";
	case JirTag::FCmpOgt:
		return "FCmpOgt";
	case JirTag::FCmpOge:
		return "FCmpOge";
	case JirTag::BitAnd:
		return "BitAnd";
	case JirTag::BitOr:
		return "BitOr";
	case JirTag::BitXor:
		return "BitXor";
	case JirTag::BitNot:
		return "BitNot";
	case JirTag::Shl:
		return "Shl";
	case JirTag::AShr:
		return "AShr";
	case JirTag::LShr:
		return "LShr";
	case JirTag::LogNot:
		return "LogNot";
	case JirTag::ZExt:
		return "ZExt";
	case JirTag::SExt:
		return "SExt";
	case JirTag::Trunc:
		return "Trunc";
	case JirTag::SIToFP:
		return "SIToFP";
	case JirTag::UIToFP:
		return "UIToFP";
	case JirTag::FPToSI:
		return "FPToSI";
	case JirTag::FPToUI:
		return "FPToUI";
	case JirTag::FPExt:
		return "FPExt";
	case JirTag::FPTrunc:
		return "FPTrunc";
	case JirTag::BitCast:
		return "BitCast";
	case JirTag::PtrToInt:
		return "PtrToInt";
	case JirTag::IntToPtr:
		return "IntToPtr";
	case JirTag::FnRef:
		return "FnRef";
	case JirTag::Br:
		return "Br";
	case JirTag::CondBr:
		return "CondBr";
	case JirTag::Switch:
		return "Switch";
	case JirTag::Ret:
		return "Ret";
	case JirTag::Unreachable:
		return "Unreachable";
	case JirTag::Call:
		return "Call";
	case JirTag::CallIndirect:
		return "CallIndirect";
	case JirTag::Param:
		return "Param";
	case JirTag::SretArg:
		return "SretArg";
	case JirTag::StructLit:
		return "StructLit";
	case JirTag::FieldAccess:
		return "FieldAccess";
	case JirTag::ExtractValue:
		return "ExtractValue";
	case JirTag::ArrayLit:
		return "ArrayLit";
	case JirTag::MakeSlice:
		return "MakeSlice";
	case JirTag::MemSet:
		return "MemSet";
	case JirTag::Index:
		return "Index";
	case JirTag::FieldAddr:
		return "FieldAddr";
	case JirTag::IndexAddr:
		return "IndexAddr";
	case JirTag::AddrOf:
		return "AddrOf";
	case JirTag::Deref:
		return "Deref";
	case JirTag::EnumPayload:
		return "EnumPayload";
	case JirTag::DropBinding:
		return "DropBinding";
	}
	return "?";
}

struct Verifier {
	const JirFunction &jfn;
	const TypePool *types;
	const StringPool *strings;
	JirVerifyResolver resolver;
	void *resolverCtx;
	std::vector<jam::Diagnostic> diags;
	// For def-before-use: as we walk blocks in order, mark each
	// instruction as "defined" the moment we step past it.
	std::vector<bool> defined;

	Verifier(const JirFunction &f, const TypePool *tp, const StringPool *sp,
	         JirVerifyResolver r, void *rctx)
	    : jfn(f), types(tp), strings(sp), resolver(r), resolverCtx(rctx),
	      defined(f.insts.size(), false) {
		defined[0] = true;  // sentinel always usable as kNoJirRef
	}

	// Walk a TypeIdx through the optional resolver. Used when the
	// raw TypeIdx is a `GenericCall` or alias that maps to a more
	// concrete type — comparing the resolved forms catches
	// mismatches that comparing the raw indices would miss.
	TypeIdx resolveTy(TypeIdx ty) const {
		if (resolver == nullptr || ty == kNoType) return ty;
		TypeIdx r = resolver(resolverCtx, ty);
		return (r == kNoType) ? ty : r;
	}

	void err(JirRef r, const std::string &msg) {
		uint32_t line = (r < jfn.insts.size()) ? jfn.insts[r].srcLine : 0;
		std::string tag =
		    (r < jfn.insts.size()) ? tagName(jfn.insts[r].tag) : "?";
		jam::Diagnostic d;
		d.loc.line = static_cast<int>(line);
		d.severity = jam::Diagnostic::Severity::Error;
		d.message = "jir-verify: fn `" + jfn.name + "` ref #" +
		            std::to_string(r) + " (" + tag + "): " + msg;
		diags.push_back(std::move(d));
	}

	void blockErr(JirBlockRef b, const std::string &msg) {
		std::string name = (b < jfn.blocks.size()) ? jfn.blocks[b].name : "?";
		jam::Diagnostic d;
		d.severity = jam::Diagnostic::Severity::Error;
		d.message = "jir-verify: fn `" + jfn.name + "` block #" +
		            std::to_string(b) + " (" + name + "): " + msg;
		diags.push_back(std::move(d));
	}

	// Check that JirRef `r` (0-based interpretation: kNoJirRef OK iff
	// optional) points within `insts` and was defined in an earlier
	// block (or earlier in this block).
	void checkRef(JirRef r, bool optional, JirRef siteRef, const char *what) {
		if (r == kNoJirRef) {
			if (!optional) {
				err(siteRef, std::string("required operand `") + what +
				                 "` is kNoJirRef");
			}
			return;
		}
		if (r >= jfn.insts.size()) {
			err(siteRef, std::string("operand `") + what + "` ref " +
			                 std::to_string(r) + " out of bounds (max " +
			                 std::to_string(jfn.insts.size() - 1) + ")");
			return;
		}
		if (!defined[r]) {
			err(siteRef, std::string("operand `") + what + "` ref " +
			                 std::to_string(r) +
			                 " used before its defining block");
		}
	}

	void checkBlockRef(JirBlockRef b, JirRef siteRef, const char *what) {
		if (b == kNoJirBlock) {
			err(siteRef, std::string("block ref `") + what + "` is null");
			return;
		}
		if (b >= jfn.blocks.size()) {
			err(siteRef, std::string("block ref `") + what + "` " +
			                 std::to_string(b) + " out of bounds (max " +
			                 std::to_string(jfn.blocks.size() - 1) + ")");
		}
	}

	void checkExtraSlice(JirExtraIdx start, std::size_t len, JirRef siteRef,
	                     const char *what) {
		if (start + len > jfn.extra.size()) {
			err(siteRef,
			    std::string("extra slice `") + what + "` overflows: needs [" +
			        std::to_string(start) + ".." + std::to_string(start + len) +
			        ") but pool size is " + std::to_string(jfn.extra.size()));
		}
	}

	// Type-consistency: refs that participate in a binary op or
	// comparison should have agreeing operand types. We compare
	// TypeKeys structurally rather than by TypeIdx — the same
	// logical type can be interned at multiple indices in edge
	// cases (e.g. substitutions building the same `Named` from
	// different paths) and a strict TypeIdx == TypeIdx check
	// generates false positives. If no TypePool is available
	// (verifier callers can skip the typecheck phase), the check
	// is a no-op.
	// Walk through one level of name-aliasing — a `Named` TypeIdx is
	// often just a synonym for a struct/enum that's already
	// registered. We don't chase `Alias` chains here (the codegen
	// `resolveGenericCall` does that elsewhere); the goal is to
	// compare types loosely enough that legitimate JIR shapes don't
	// trigger false positives, while still flagging real Int-vs-Float
	// or Int8-vs-Int32 mistakes.
	bool refTypesMatch(JirRef a, JirRef b) {
		if (types == nullptr) return true;
		if (a >= jfn.insts.size() || b >= jfn.insts.size()) return true;
		TypeIdx ta = resolveTy(jfn.insts[a].ty);
		TypeIdx tb = resolveTy(jfn.insts[b].ty);
		if (ta == kNoType || tb == kNoType) return true;
		if (ta == tb) return true;
		const TypeKey &ka = types->get(ta);
		const TypeKey &kb = types->get(tb);
		if (ka == kb) return true;
		// Even after resolution, some TypeIdxes (named struct/enum
		// aliases) might be distinct slots for the same logical type
		// — the resolver only covers GenericCall. Conservative bypass
		// on Named/Struct/Enum/Slice/Array/Ptr* avoids false
		// positives without losing the primitive Int / Float coverage.
		auto isOpaque = [](TypeKind k) {
			return k == TypeKind::Named || k == TypeKind::Struct ||
			       k == TypeKind::Enum || k == TypeKind::Slice ||
			       k == TypeKind::Array || k == TypeKind::PtrSingle ||
			       k == TypeKind::PtrMany;
		};
		if (isOpaque(ka.kind) || isOpaque(kb.kind)) return true;
		return false;
	}

	void checkInst(JirRef r) {
		const JirInst &inst = jfn.insts[r];
		switch (inst.tag) {
		case JirTag::Invalid:
			err(r, "Invalid tag in live instruction stream");
			return;
		// Constants and the Poison placeholder — no operand refs.
		case JirTag::Poison:
		case JirTag::Int:
		case JirTag::Float:
		case JirTag::Bool:
		case JirTag::Str:
		case JirTag::Param:
		case JirTag::SretArg:
		case JirTag::Unreachable:
			return;
		case JirTag::Alloca:
			// `ty` is the pointee; no JirRef operands.
			return;
		case JirTag::Load:
		case JirTag::Deref:
		case JirTag::AddrOf:
		case JirTag::FNeg:
		case JirTag::BitNot:
		case JirTag::LogNot:
		case JirTag::ZExt:
		case JirTag::SExt:
		case JirTag::Trunc:
		case JirTag::SIToFP:
		case JirTag::UIToFP:
		case JirTag::FPToSI:
		case JirTag::FPToUI:
		case JirTag::FPExt:
		case JirTag::FPTrunc:
		case JirTag::BitCast:
		case JirTag::PtrToInt:
		case JirTag::IntToPtr:
		case JirTag::FieldAccess:
		case JirTag::ExtractValue:
		case JirTag::FieldAddr:
		case JirTag::EnumPayload:
			checkRef(inst.a, false, r, "a");
			return;
		case JirTag::FnRef:
			// `a` is a StringIdx, not a JirRef — nothing to check
			// against `insts.size()`. Codegen will fail loudly if the
			// referenced function symbol isn't in the module.
			return;
		case JirTag::DropBinding: {
			checkRef(inst.a, false, r, "a");
			// The `b` field is a StringIdx pointing at the LLVM
			// symbol of the drop fn. We don't have direct access to
			// the LLVM module here, but we can sanity-check that
			// the StringIdx is a non-empty string — a zero / empty
			// symbol is a sure sign astgen's lookupDropFn returned
			// a default-constructed empty string.
			if (strings != nullptr) {
				StringIdx sid = static_cast<StringIdx>(inst.b);
				if (sid == kNoString || strings->get(sid).empty()) {
					err(r, "DropBinding with empty / unresolved drop symbol");
				}
			}
			return;
		}
		case JirTag::Store:
			checkRef(inst.a, false, r, "a");
			checkRef(inst.b, false, r, "b");
			return;
		// Same-type binary ops: lhs and rhs must agree. Comparisons
		// produce i1 from same-typed operands.
		case JirTag::Add:
		case JirTag::Sub:
		case JirTag::Mul:
		case JirTag::SDiv:
		case JirTag::UDiv:
		case JirTag::SRem:
		case JirTag::URem:
		case JirTag::FAdd:
		case JirTag::FSub:
		case JirTag::FMul:
		case JirTag::FDiv:
		case JirTag::FRem:
		case JirTag::ICmpEq:
		case JirTag::ICmpNe:
		case JirTag::ICmpSlt:
		case JirTag::ICmpSle:
		case JirTag::ICmpSgt:
		case JirTag::ICmpSge:
		case JirTag::ICmpUlt:
		case JirTag::ICmpUle:
		case JirTag::ICmpUgt:
		case JirTag::ICmpUge:
		case JirTag::FCmpOeq:
		case JirTag::FCmpOne:
		case JirTag::FCmpOlt:
		case JirTag::FCmpOle:
		case JirTag::FCmpOgt:
		case JirTag::FCmpOge:
		case JirTag::BitAnd:
		case JirTag::BitOr:
		case JirTag::BitXor:
			checkRef(inst.a, false, r, "a");
			checkRef(inst.b, false, r, "b");
			if (!refTypesMatch(inst.a, inst.b)) {
				err(r, std::string("operand type mismatch for ") +
				           tagName(inst.tag) +
				           " (a.ty=" + std::to_string(jfn.insts[inst.a].ty) +
				           " b.ty=" + std::to_string(jfn.insts[inst.b].ty) +
				           ")");
			}
			return;
		// Shifts: LHS is the value being shifted (any int width),
		// RHS is the shift amount (often a narrower int). LLVM
		// matches them via zext/trunc in the backend; we don't
		// require equal types here.
		case JirTag::Shl:
		case JirTag::AShr:
		case JirTag::LShr:
			checkRef(inst.a, false, r, "a");
			checkRef(inst.b, false, r, "b");
			return;
		// Index ops: base + index. Index can be a different
		// (integer) type than the base — no match check.
		case JirTag::Index:
		case JirTag::IndexAddr:
			checkRef(inst.a, false, r, "a");
			checkRef(inst.b, false, r, "b");
			return;
		case JirTag::Ret:
			checkRef(inst.a, /*optional=*/true, r, "a");
			return;
		case JirTag::Br:
			checkBlockRef(inst.a, r, "target");
			return;
		case JirTag::CondBr: {
			checkRef(inst.a, false, r, "cond");
			checkExtraSlice(inst.b, 2, r, "branches");
			if (inst.b + 2 <= jfn.extra.size()) {
				JirBlockRef thenB = static_cast<JirBlockRef>(jfn.extra[inst.b]);
				JirBlockRef elseB =
				    static_cast<JirBlockRef>(jfn.extra[inst.b + 1]);
				checkBlockRef(thenB, r, "then");
				checkBlockRef(elseB, r, "else");
			}
			return;
		}
		case JirTag::Switch: {
			checkRef(inst.a, false, r, "scrut");
			if (inst.b + 2 > jfn.extra.size()) {
				err(r, "Switch extra header out of bounds");
				return;
			}
			JirBlockRef defaultB = static_cast<JirBlockRef>(jfn.extra[inst.b]);
			uint32_t caseCount = jfn.extra[inst.b + 1];
			checkBlockRef(defaultB, r, "default");
			checkExtraSlice(inst.b, 2 + caseCount * 4, r, "switch");
			for (uint32_t i = 0; i < caseCount; i++) {
				JirBlockRef caseB =
				    static_cast<JirBlockRef>(jfn.extra[inst.b + 2 + i * 4 + 3]);
				checkBlockRef(caseB, r, "case-block");
			}
			return;
		}
		case JirTag::Call: {
			// `a` is StringIdx (callee name), `b` is the extra-pool
			// offset of the [argCount, arg0_ref, …] slice. Index 0
			// is a valid pool position when the function's first
			// extra-allocation lands at offset zero.
			if (inst.b >= jfn.extra.size()) {
				err(r, "Call extra index out of bounds");
				return;
			}
			uint32_t argCount = jfn.extra[inst.b];
			checkExtraSlice(inst.b, 1 + argCount, r, "call-args");
			for (uint32_t i = 0; i < argCount; i++) {
				JirRef ar = static_cast<JirRef>(jfn.extra[inst.b + 1 + i]);
				checkRef(ar, false, r, "call-arg");
			}
			return;
		}
		case JirTag::CallIndirect: {
			// `a` is a JirRef of a Fn-typed value; `b` mirrors Call's
			// extra-pool layout [argCount, arg0_ref, ...].
			checkRef(inst.a, false, r, "callee");
			if (inst.b >= jfn.extra.size()) {
				err(r, "CallIndirect extra index out of bounds");
				return;
			}
			uint32_t argCount = jfn.extra[inst.b];
			checkExtraSlice(inst.b, 1 + argCount, r, "call-args");
			for (uint32_t i = 0; i < argCount; i++) {
				JirRef ar = static_cast<JirRef>(jfn.extra[inst.b + 1 + i]);
				checkRef(ar, false, r, "call-arg");
			}
			return;
		}
		case JirTag::StructLit:
		case JirTag::ArrayLit: {
			if (inst.b >= jfn.extra.size()) {
				err(r, "aggregate-lit extra index out of bounds");
				return;
			}
			uint32_t count = jfn.extra[inst.b];
			checkExtraSlice(inst.b, 1 + count, r, "agg-fields");
			for (uint32_t i = 0; i < count; i++) {
				JirRef fr = static_cast<JirRef>(jfn.extra[inst.b + 1 + i]);
				checkRef(fr, false, r, "agg-field");
			}
			return;
		}
		case JirTag::MakeSlice: {
			// a = data ptr, b = length — both refs.
			checkRef(inst.a, false, r, "slice-ptr");
			checkRef(inst.b, false, r, "slice-len");
			return;
		}
		case JirTag::MemSet: {
			// a = dest ptr ref; b = byte length (raw constant, not a ref);
			// flags = fill byte. Only `a` is a ref.
			checkRef(inst.a, false, r, "memset-dst");
			return;
		}
		}
	}
};

}  // namespace

std::vector<jam::Diagnostic> verifyJirFunction(const JirFunction &jfn,
                                               const TypePool *types,
                                               const StringPool *strings,
                                               JirVerifyResolver resolver,
                                               void *resolverCtx) {
	Verifier v(jfn, types, strings, resolver, resolverCtx);

	// Walk blocks in declaration order, mirroring jir_codegen's pass.
	// Mark each instruction defined as we step over it; references
	// must hit already-defined slots (the cache-invariant the codegen
	// relies on for SSA correctness).
	//
	// Allocas live exclusively in the entry block (block 1). The
	// `emitAllocaHoisted` helper enforces this at astgen time —
	// stack slots are conceptually function-scoped, and putting them
	// elsewhere would invite the def-before-use bug the cache fix in
	// jir_codegen mitigates. We enforce the invariant here too so a
	// future codepath that bypasses the helper fails loudly.
	for (JirBlockRef b = 1; b < jfn.blocks.size(); b++) {
		if (b == 1) continue;
		for (JirRef r : jfn.blocks[b].insts) {
			if (r < jfn.insts.size() && jfn.insts[r].tag == JirTag::Alloca) {
				v.blockErr(b, "Alloca outside entry block — use "
				              "emitAllocaHoisted");
			}
		}
	}
	for (JirBlockRef b = 1; b < jfn.blocks.size(); b++) {
		const JirBlock &block = jfn.blocks[b];
		if (block.insts.empty()) {
			v.blockErr(b, "empty block (no terminator)");
			continue;
		}
		for (size_t i = 0; i < block.insts.size(); i++) {
			JirRef r = block.insts[i];
			if (r == kNoJirRef || r >= jfn.insts.size()) {
				v.blockErr(b, "instruction list contains invalid ref " +
				                  std::to_string(r));
				continue;
			}
			v.checkInst(r);
			v.defined[r] = true;
			// Terminator must be the last instruction in the block —
			// anything emitted after it is dead and indicates a bug
			// in astgen's emit ordering.
			bool isLast = (i + 1 == block.insts.size());
			bool term = isTerminator(jfn.insts[r].tag);
			if (term && !isLast) {
				v.blockErr(b, "terminator " +
				                  std::string(tagName(jfn.insts[r].tag)) +
				                  " is not last in block");
			}
			if (!term && isLast) {
				v.blockErr(b, "block ends with non-terminator " +
				                  std::string(tagName(jfn.insts[r].tag)));
			}
		}
	}

	return std::move(v.diags);
}
