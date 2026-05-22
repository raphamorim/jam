/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JIR_CODEGEN_H
#define JIR_CODEGEN_H

#include "jir.h"

class JamCodegenContext;

// JIR codegen — walks a fully-typed JirFunction and emits LLVM IR
// instructions. By design this stage is mechanical: there is no type
// inference, no peer resolution, no divergence analysis. Each JirInst
// maps to a small handful of LLVM IR instructions and the JirRef ->
// LLVM Value mapping carries the dataflow.
//
// Two-step API: prototype emission runs first so forward references
// between functions resolve before any body is lowered. The classifier
// inside `jirDeclarePrototype` is the single source of truth for the
// LLVM signature — call sites in `jirDefineBody` and arg lowering in
// astgen both consult the same `jam::abi::classify*` routines, so the
// caller and callee can never disagree on by-value vs by-pointer or
// direct vs sret returns.
void jirDeclarePrototype(const JirFunction &jfn, JamCodegenContext &ctx);
void jirDefineBody(const JirFunction &jfn, JamCodegenContext &ctx);

#endif  // JIR_CODEGEN_H
