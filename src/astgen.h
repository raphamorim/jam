/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef ASTGEN_H
#define ASTGEN_H

#include "ast.h"
#include "jir.h"

class JamCodegenContext;

// Thrown when astgen can't finish analyzing a declaration. The
// diagnostic was already pushed onto `JamCodegenContext::diagnostics()`
// before the throw; catch sites (per-decl in main.cpp / generic
// instantiation in codegen.cpp) recover by abandoning the current
// decl so siblings still produce diagnostics. Mirrors Zig's
// `error.AnalysisFail` (`src/AstGen.zig:60`).
class AstGenAnalysisFail {};

// AstGen — recursive, eager lowering of a FunctionAST into a typed
// JirFunction. Mirrors Zig's `AstGen` (lib/std/zig/AstGen.zig) except
// the output is typed from the start (Jam doesn't have comptime-as-
// runtime-values, so the ZIR/AIR split isn't needed).
//
// Each AST node visited:
//   * pushes zero or more `JirInst`s into the function's instruction
//     array;
//   * threads its source line into every emitted instruction so
//     diagnostics carry `file:line:` precision;
//   * resolves types eagerly — literals lower at their natural width
//     or at an `expectedType` hint passed from the parent.
//
// Generic instantiation, peer-type resolution, divergence analysis,
// and constant folding all happen here. The downstream `jirCodegen`
// is a mechanical AIR-to-LLVM walk.
JirFunction astgenFunction(const FunctionAST &fn, JamCodegenContext &ctx);

// Populate only a JirFunction's *signature* — name, return type,
// per-param TypeIdx and by-pointer flag, extern/test/pub bookkeeping
// — without walking the body. Used by generic instantiation's
// declare-prototypes-first pass so a method body can reference its
// siblings before any body has been lowered.
JirFunction astgenMetadata(const FunctionAST &fn, JamCodegenContext &ctx);

// Append the body of `fn` to a pre-populated metadata JirFunction.
// Used by generic instantiation's Pass 2 to avoid redoing the
// metadata work already done in Pass 1. The caller owns `jfn`.
void astgenBodyInto(JirFunction &jfn, const FunctionAST &fn,
                    JamCodegenContext &ctx);

#endif  // ASTGEN_H
