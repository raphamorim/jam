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
// maps to a small handful of LLVM IR instructions and the JirRef →
// LLVM Value mapping carries the dataflow.
//
// Two-step API matching the existing FunctionAST::declarePrototype /
// defineBody split: the prototype emission happens before body
// emission so forward references between functions resolve.
void jirDeclarePrototype(const JirFunction &jfn, JamCodegenContext &ctx);
void jirDefineBody(const JirFunction &jfn, JamCodegenContext &ctx);

#endif  // JIR_CODEGEN_H
