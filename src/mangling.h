/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_MANGLING_H
#define JAM_MANGLING_H

#include "ast.h"
#include "ast_flat.h"

#include <string>

// Translate a FunctionAST into the LLVM-level symbol the linker sees.
// The rules:
//   - `tfn t()` → `__test_t` (the harness in main.cpp calls these by
//     prefixed name).
//   - `fn drop(self: mut T)` → `__drop_T` (the legacy mangling, used
//     so callers don't have to spell out the full struct name every
//     time they `T.drop(&x)`).
//   - Cloned instantiated methods carry their qualified name already
//     (`Vec__i32.push`) — we keep it.
//   - Everything else → bare `fn.Name`.
//
// Centralised so every site that picks a function's LLVM symbol
// (`jirDeclarePrototype`, `astgen::emitCall`, generic instantiation in
// `codegen.cpp`) reads the same rules — duplicating the mangling table
// per call site is exactly how "unknown callee" miscompiles creep in
// when one site gains a new case and the others don't.
inline std::string mangledFunctionName(const FunctionAST &fn,
                                       const TypePool &types,
                                       const StringPool &strings) {
	if (fn.isTest) return "__test_" + fn.Name;
	if (fn.Name == "drop" && fn.Args.size() == 1) {
		const Param &p = fn.Args[0];
		if (p.Name == "self" && p.Mode == ParamMode::Mut) {
			const TypeKey &k = types.get(p.Type);
			if (k.kind == TypeKind::Struct || k.kind == TypeKind::Named) {
				StringIdx ni = static_cast<StringIdx>(k.a);
				if (ni != kNoString) { return "__drop_" + strings.get(ni); }
			}
		}
	}
	return fn.Name;
}

#endif  // JAM_MANGLING_H
