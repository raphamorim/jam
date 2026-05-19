/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JIR_VERIFY_H
#define JIR_VERIFY_H

#include "diagnostics.h"
#include "jir.h"

#include <string>
#include <vector>

// Structural verification of a finished JirFunction. Catches drift
// that the codegen would otherwise hit as a silent miscompile or a
// late LLVM-verifier failure. Run after astgen finishes a function;
// returns an empty vector when the function is well-formed,
// otherwise a list of human-readable diagnostics (with source-line
// hints where available).
//
// Checks:
//   - Every reachable block ends with exactly one terminator
//     (Br, CondBr, Switch, Ret, Unreachable). Empty blocks are
//     rejected — astgen should have either emitted a terminator
//     or left no block at all.
//   - Every JirRef inside an instruction points into the function's
//     `insts` array (1..size-1). kNoJirRef is allowed where the
//     instruction's spec marks the slot optional (e.g. Ret's `a`).
//   - Every JirBlockRef in CondBr / Br / Switch / for-loop step
//     blocks points within `blocks` (1..size-1).
//   - Every JirExtraIdx in Call / CondBr / Switch / StructLit /
//     ArrayLit points within `extra` and the slice it references
//     fits.
//   - Def-before-use across blocks: when block B references a
//     JirRef defined in block C, C must precede B in walk order.
//     The codegen's instruction-cache invariant depends on this,
//     and an inlining or block-reordering pass would silently
//     break it without a check here.
// The verifier needs access to:
//   - the `TypePool` for type-consistency checks
//   - the codegen context's string pool (for resolving `DropBinding`
//     symbols back to readable diagnostics)
//   - an optional resolver callback that maps a `TypeIdx` through
//     generic-call resolution and type aliases to its canonical
//     concrete form. Without it, the typecheck bails out on
//     `Named`/`GenericCall` operands; with it, we can compare
//     resolved types and catch more bugs.
// Passed as opaque pointers so this header doesn't drag the rest of
// codegen.h. Each is independently nullable.
class TypePool;
class StringPool;
using JirVerifyResolver = TypeIdx (*)(void *ctx, TypeIdx ty);

// Each diagnostic carries the offending JIR instruction's `srcLine`
// in `loc.line` (file is left empty so the caller can stamp the
// module's currentFile() before forwarding into the global
// Diagnostics channel). The `message` text already contains
// fn-name + ref + tag context so the user can pin which
// instruction tripped the check.
std::vector<jam::Diagnostic> verifyJirFunction(
    const JirFunction &jfn, const TypePool *types = nullptr,
    const StringPool *strings = nullptr,
    JirVerifyResolver resolver = nullptr, void *resolverCtx = nullptr);

#endif  // JIR_VERIFY_H
