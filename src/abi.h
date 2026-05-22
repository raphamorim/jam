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

// Classify a parameter (mode, type) pair. Pure function of its inputs;
// safe to call any number of times. See docs/ABI.md §4.
//
//   mut                    -> always ByPointer
//   let / move, scalar T   -> ByValue
//   let / move, aggregate
//     size <= kByValueMaxBytes -> ByValue (LLVM handles register packing)
//     size  > kByValueMaxBytes -> ByPointer
ParamABI classifyParam(ParamMode mode, TypeIdx ty,
                       const JamCodegenContext &ctx);

// Classify a return type. See docs/ABI.md §4.
//
//   scalar T                       -> Direct
//   aggregate with size <= 16 B    -> Direct (LLVM packs into return regs)
//   aggregate with size  > 16 B    -> Indirect (sret)
ReturnABI classifyReturn(TypeIdx ty, const JamCodegenContext &ctx);

// Threshold above which an owned aggregate is passed/returned by pointer
// rather than by value. Matches the System V AMD64 ABI's two-eightbyte
// MEMORY classification and Rust's observed behavior at -C opt-level=0.
constexpr uint64_t kByValueMaxBytes = 16;

}  // namespace abi
}  // namespace jam

#endif  // ABI_H
