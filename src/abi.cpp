/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "abi.h"
#include "ast.h"
#include "codegen.h"

namespace jam {
namespace abi {

bool isByRef(TypeIdx ty, const JamCodegenContext &ctx) {
	if (ty == kNoType) return false;
	const TypeKey &k = ctx.getTypePool().get(ty);
	switch (k.kind) {
	// Source-only / comptime-only — never lowered to runtime memory,
	// so no byref classification makes sense. Caller has bigger
	// problems if they reach here at codegen time.
	case TypeKind::Invalid:
	case TypeKind::Void:
	case TypeKind::NoReturn:
	case TypeKind::Type:
	case TypeKind::Module:
		return false;
	// `c.Vec(i32)` and similar generic-call type expressions live in
	// the TypePool as GenericCall keys; their *instantiated* struct
	// TypeIdx is what carries runtime shape. Resolve and recurse so
	// `var v: c.Vec(i32)` classifies as byref like the underlying
	// `Vec__i32` does.
	case TypeKind::GenericCall: {
		TypeIdx resolved = ctx.resolveGenericCall(ty);
		return resolved != kNoType && resolved != ty && isByRef(resolved, ctx);
	}
	// Deferred-length arrays classify exactly like their resolved
	// Array form (byref) — resolve so the recursion also validates
	// the length expression on first ABI consultation.
	case TypeKind::ArrayExpr: {
		TypeIdx resolved = ctx.resolveArrayExpr(ty);
		return resolved != kNoType && resolved != ty && isByRef(resolved, ctx);
	}
	// Scalars and pointer-shaped values: byval. The whole value
	// rides in registers; SSA representation is the value itself.
	case TypeKind::Bool:
	case TypeKind::Int:
	case TypeKind::Float:
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
	case TypeKind::Fn:
		return false;
	// Slice is a `{ ptr, len }` aggregate but we treat it as a
	// packed scalar pair LLVM tolerates in registers — same as the
	// current ABI path. Flipping this to byref would force a memory
	// detour on every slice-typed local, which is a much larger
	// blast radius than the codegen perf bug this work is fixing.
	// Revisit if slice ergonomics warrant it later.
	case TypeKind::Slice:
		return false;
	// Aggregates: byref. Every user-defined aggregate lives in memory
	// at the JIR level and is referred to by pointer in SSA. Small
	// structs still promote to registers via LLVM's mem2reg + SROA at
	// -O1+; we don't need to second-guess the optimizer.
	case TypeKind::Array:
	case TypeKind::Struct:
	case TypeKind::Union:
		return true;
	// Unit-only enums (no payload variants) lower to a bare i8 tag
	// and ride through SSA like any other scalar — keep them byval.
	// Payloaded enums are {tag, payload} aggregates and follow the
	// strict-byref rule.
	case TypeKind::Enum: {
		const auto *einfo = ctx.lookupEnum(ty);
		return einfo != nullptr && einfo->hasPayloadVariant;
	}
	// `Named` is a deferred reference into the struct / enum / union
	// registries. Resolve once and recurse on the canonical TypeIdx
	// so type aliases and `Self` substitutions classify the same as
	// what they point at.
	case TypeKind::Named: {
		const std::string &name =
		    ctx.getStringPool().get(static_cast<StringIdx>(k.a));
		if (ctx.lookupStruct(ty) || ctx.lookupUnion(ty)) { return true; }
		// Enum is byref only when it carries a payload — unit-only
		// enums lower to a bare i8 tag and stay byval like scalars.
		if (const auto *einfo = ctx.lookupEnum(ty)) {
			return einfo->hasPayloadVariant;
		}
		TypeIdx alias = ctx.lookupTypeAlias(name);
		if (alias != kNoType && alias != ty) return isByRef(alias, ctx);
		// Unresolved Named with no struct/enum/union registry hit —
		// shouldn't reach codegen, but treat as byval to keep the
		// existing failure surface unchanged.
		return false;
	}
	}
	return false;
}

// Any byref aggregate (Struct / Array / Union / payloaded Enum)
// crosses the call boundary by pointer; everything else rides in
// registers as its LLVM-natural value. `mut` is the one exception —
// it always carries a pointer to caller-owned storage regardless of
// the underlying type's classification.
//
// Slice ({ptr, len}) and Fn (a single ptr) are explicit byval cases
// today: they're already register-shaped at LLVM-IR level. Adding a
// new byval aggregate that doesn't fit naturally in registers should
// flip its type-kind to byref in `isByRef`, not patch this code.
ParamABI classifyParam(ParamMode mode, TypeIdx ty,
                       const JamCodegenContext &ctx) {
	if (mode == ParamMode::Mut || isByRef(ty, ctx)) {
		return ParamABI{ParamABI::Kind::ByPointer, nullptr,
		                static_cast<uint32_t>(ctx.typeAlign(ty))};
	}
	return ParamABI{ParamABI::Kind::ByValue, ctx.getLLVMType(ty), 0};
}

ReturnABI classifyReturn(TypeIdx ty, const JamCodegenContext &ctx) {
	if (ty == kNoType) {
		return ReturnABI{ReturnABI::Kind::Direct, ctx.getVoidType(), 0};
	}
	if (isByRef(ty, ctx)) {
		return ReturnABI{ReturnABI::Kind::Indirect, nullptr,
		                 static_cast<uint32_t>(ctx.typeAlign(ty))};
	}
	return ReturnABI{ReturnABI::Kind::Direct, ctx.getLLVMType(ty), 0};
}

}  // namespace abi
}  // namespace jam
