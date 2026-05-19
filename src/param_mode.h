/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_PARAM_MODE_H
#define JAM_PARAM_MODE_H

#include <cstdint>

// Parameter passing mode — Let (read-only, the default), Mut
// (read-write borrow), Move (consume ownership). See docs/MVS.md §2
// for the language-level semantics. Modes are static-only; the
// per-mode ABI decision (e.g. mut/move passed by pointer in the
// current LLVM backend) lives in the codegen, not here.
//
// Extracted into its own header so JIR can carry per-param modes
// without dragging in the rest of ast.h.
enum class ParamMode : uint8_t {
	Let = 0,
	Mut,
	Move,
};

#endif  // JAM_PARAM_MODE_H
