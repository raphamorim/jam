/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef INIT_ANALYSIS_H
#define INIT_ANALYSIS_H

#include "ast_flat.h"
#include "drop_registry.h"
#include "token.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class FunctionAST;

namespace jam {
namespace init_analysis {

// Per-binding initialization state, encoded as a 2-bit lattice. Merging two
// states is a bitwise OR — the same shape Clang's UninitializedValues
// analysis uses (see clang/lib/Analysis/UninitializedValues.cpp).
//
//   Unknown     binding has not been touched on this path
//   Init        binding holds a valid value; reads OK
//   Uninit      binding has been moved out of; reads must produce an error
//   MaybeInit   merge of Init + Uninit at a control-flow join; reads must
//               produce an error
//
// The OR-merge means MaybeInit dominates everything except (transient)
// Unknown. See docs/MVS.md §5 for the full semantics.
enum class InitState : uint8_t {
	Unknown = 0b00,
	Init = 0b01,
	Uninit = 0b10,
	MaybeInit = 0b11,
};

inline InitState mergeState(InitState a, InitState b) {
	return static_cast<InitState>(static_cast<uint8_t>(a) |
	                              static_cast<uint8_t>(b));
}

// One reportable problem found during analysis. `line` is taken from the
// AST node's `mainToken` lookup against the parser's token stream. If a
// caller does not supply tokens, line will be 0.
struct Diagnostic {
	std::string message;
	int line;
	std::string varName;
};

// Map of function name -> FunctionAST* used by the analyzer to look up
// callee parameter modes for callsite propagation. The map is
// borrowed; the analyzer never owns the FunctionAST pointers. May be
// null, in which case mode propagation is conservatively skipped at
// every call site (caller's bindings are unchanged).
using FunctionRegistry = std::unordered_map<std::string, const FunctionAST *>;

// Run the definite-init analysis on a function body. Returns an empty
// vector on success; on failure, the vector contains every detected
// uninit-read with location info.
//
// Scope:
//   - Tracks per-binding init state for locals declared with
//     `var name: T = ...`.
//   - Parameter entry state: every parameter starts Init (the caller is
//     required to pass an initialized binding).
//   - Each call propagates caller-side state per the callee's parameter
//     modes: `move` arg ⇒ caller's base binding becomes Uninit.
//
// Control-flow handling:
//   - Straight-line statement sequence: states thread through.
//   - if/else: states merge at the join (Init + Uninit = MaybeInit).
//   - return: terminates flow on its path; the alternative path's
//     state survives the merge unchanged.
//   - while/for/match: conservative pass — body is analyzed once against
//     the pre-loop state, and the post state merges body output with
//     pre-loop. For-loops over a range additionally assume the body runs
//     at least once. May replace with fixed-point iteration later.
std::vector<Diagnostic> analyze(const FunctionAST &fn, const NodeStore &nodes,
                                const StringPool &strings,
                                const std::vector<Token> &tokens,
                                const FunctionRegistry *registry = nullptr,
                                const drops::DropRegistry *drops = nullptr,
                                const TypePool *types = nullptr);

}  // namespace init_analysis
}  // namespace jam

#endif  // INIT_ANALYSIS_H
