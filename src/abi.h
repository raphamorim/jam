/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef ABI_H
#define ABI_H

#include "ast_flat.h"
#include "jam_llvm.h"
#include <cstdint>

class JamCodegenContext;
// `ParamMode` is defined in ast.h. We forward-declare it (the explicit
// underlying type makes the forward declaration legal as of C++11) so
// abi.h's signatures don't pull in the full AST header.
enum class ParamMode : uint8_t;

namespace jam {
namespace abi {

// How a single function parameter is passed at the LLVM IR level.
//
//   ByValue   — the parameter is a single LLVM argument of its natural
//               LLVM representation (a scalar like `i32`, a pointer
//               like `ptr`, or an aggregate like `%MyStruct`). The
//               callee receives the value directly; the caller copies
//               it across the call boundary.
//
//   ByPointer — the parameter is a single `ptr align A` argument. The
//               callee reads/writes the pointee through it. The caller
//               passes the address of caller-owned (or sret-style
//               freshly allocated) storage.
//
// See docs/ABI.md §3 for the full mapping from (mode, type) to ABI kind.
struct ParamABI {
	enum class Kind { ByValue, ByPointer };
	Kind kind;
	JamTypeRef llvmType;    // ByValue: the parameter's LLVM type.
	uint32_t pointerAlign;  // ByPointer: pointee alignment in bytes.
};

// How a function's return value is communicated to the caller.
//
//   Direct   — return as a single LLVM value (scalar or small aggregate).
//   Indirect — caller passes a leading `ptr sret(%T) align A` argument;
//              the callee writes the result through it and returns
//              `void`. Used for return types whose size exceeds the
//              by-value threshold.
struct ReturnABI {
	enum class Kind { Direct, Indirect };
	Kind kind;
	JamTypeRef directType;  // Direct: the LLVM return type.
	uint32_t sretAlign;     // Indirect: pointee alignment in bytes.
};

// Is this type carried by-reference at the codegen level?
//
//   Byref: arrays / structs / non-packed unions / payloaded enums —
//   anything whose natural runtime form lives in memory and whose
//   JIR-level value is a pointer to that storage. This is *codegen*
//   shape, distinct from the source-level ABI / param-mode
//   classification — even a small 2-field struct returns true here.
//
//   Not byref: scalars (Int / Float / Bool), Pointer types (PtrSingle
//   / PtrMany / Slice — Slice is a 2-field aggregate but is treated
//   as a packed { ptr, len } pair that LLVM passes in registers), Fn
//   pointers, unit-only enums (lower to a bare i8 tag).
//
// Single source of truth: codegen reroutings (StructLit / ArrayLit /
// Store / Load / FieldAccess) and `classifyParam` / `classifyReturn`
// all dispatch on this.
bool isByRef(TypeIdx ty, const JamCodegenContext &ctx);

// Classify a parameter (mode, type) pair. Pure function of its inputs;
// safe to call any number of times.
//
//   mut                       -> ByPointer (always — caller storage)
//   isByRef(T)                -> ByPointer (byref aggregate)
//   anything else             -> ByValue (rides in registers)
ParamABI classifyParam(ParamMode mode, TypeIdx ty,
                       const JamCodegenContext &ctx);

// Classify a return type.
//
//   void / kNoType            -> Direct void
//   isByRef(T)                -> Indirect (sret)
//   anything else             -> Direct (rides in return regs)
ReturnABI classifyReturn(TypeIdx ty, const JamCodegenContext &ctx);

}  // namespace abi
}  // namespace jam

#endif  // ABI_H
