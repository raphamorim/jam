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
// Modeled on Zig's `Decl.getFullyQualifiedName` (Module.zig:713) —
// dot-separated, walking up the namespace chain. LLVM accepts `.` in
// symbol names, so no further escaping is needed.
//
// The rules:
//   - `tfn t()` -> `__test_t` (the harness in main.cpp calls these by
//     prefixed name).
//   - Cloned instantiated methods carry their qualified name already
//     (`Vec__i32.push`) — we keep it.
//   - Free-function `cfn drop(self: mut T)` -> `T.drop`. The receiver
//     type qualifies it so two top-level drops for different types
//     don't collide on the bare name `drop`. Matches what an
//     equivalent in-struct `cfn drop` would mangle to.
//   - Otherwise: `[modulePath.][parentStruct.]Name`
//       free fn in entry module       -> `name`
//       free fn in module `m`         -> `m.name`
//       method on `T` in entry module -> `T.name`
//       method on `T` in module `m`   -> `m.T.name`
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
	// Free-function drop: qualify by receiver type so `cfn drop(self:
	// mut A)` and `cfn drop(self: mut B)` get distinct symbols. Only
	// kicks in when parentStruct is empty — in-struct `cfn drop` goes
	// through the standard FQN path below.
	if (fn.parentStruct.empty() && fn.Name == "drop" && fn.Args.size() == 1) {
		const Param &p = fn.Args[0];
		if (p.Name == "self" && p.Mode == ParamMode::Mut) {
			const TypeKey &k = types.get(p.Type);
			if (k.kind == TypeKind::Struct || k.kind == TypeKind::Named) {
				StringIdx ni = static_cast<StringIdx>(k.a);
				if (ni != kNoString) { return strings.get(ni) + ".drop"; }
			}
		}
	}
	std::string out;
	if (!fn.modulePath.empty()) {
		out += fn.modulePath;
		out += '.';
	}
	if (!fn.parentStruct.empty()) {
		out += fn.parentStruct;
		out += '.';
	}
	out += fn.Name;
	return out;
}

#endif  // JAM_MANGLING_H
