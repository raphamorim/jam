/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "drop_registry.h"
#include "ast.h"
#include "ast_flat.h"

namespace jam {
namespace drops {

// Inspect a candidate function and, if it has the drop-fn shape
// (`cfn drop(self: mut <Struct>)`), add it to the registry under
// the struct's name. Used by both the top-level-function scan and
// the struct-method scan below.
//
// `cfn` (not plain `fn`) is required: only methods opted into the
// compiler-synthesized-call set get auto-fired at scope exit. A
// plain `fn drop(self)` is treated as an ordinary method the user
// invokes explicitly — no implicit destructor call. This matters
// because a user-callable `fn drop` plus an auto-fire would
// double-drop; the `cfn` opt-in is the signal that the user is
// handing control of the call to the compiler.
static void considerDropCandidate(const FunctionAST *fn, const TypePool &types,
                                  const StringPool &strings,
                                  DropRegistry &registry) {
	if (!fn->isCfn) return;
	if (fn->Name != "drop") return;
	if (fn->Args.size() != 1) return;
	const Param &p = fn->Args[0];
	if (p.Name != "self") return;
	if (p.Mode != ParamMode::Mut) return;

	// Resolve the parameter's TypeIdx to a struct name. Both Struct and
	// Named TypeKinds carry the struct's name in their `a` slot as a
	// StringIdx; either form means "drop fn for that struct".
	const TypeKey &key = types.get(p.Type);
	if (key.kind != TypeKind::Struct && key.kind != TypeKind::Named) { return; }
	StringIdx nameIdx = static_cast<StringIdx>(key.a);
	if (nameIdx == kNoString) return;
	const std::string &structName = strings.get(nameIdx);
	registry[structName] = fn;
}

void addDropCandidates(DropRegistry &registry, const ModuleAST &module,
                       const TypePool &types, const StringPool &strings) {
	// Top-level `fn drop(self: mut T)` declarations.
	for (const auto &fn : module.Functions) {
		considerDropCandidate(fn.get(), types, strings, registry);
	}
	// Methods declared inside struct bodies. The validation in main.cpp
	// already enforces that the self-param's type matches the enclosing
	// struct, but we still re-derive the struct name from the param here
	// so the registry stays self-consistent and the in-struct form is a
	// pure synonym of the free-function form.
	for (const auto &s : module.Structs) {
		for (const auto &m : s->Methods) {
			considerDropCandidate(m.get(), types, strings, registry);
		}
	}
}

DropRegistry buildDropRegistry(const ModuleAST &module, const TypePool &types,
                               const StringPool &strings) {
	DropRegistry registry;
	addDropCandidates(registry, module, types, strings);
	return registry;
}

// Clone counterpart of considerDropCandidate: `cfn clone(self: T) T`
// (let-mode self — cloning borrows the original). The self param's
// type names the struct the clone belongs to.
static void considerCloneCandidate(const FunctionAST *fn,
                                   const TypePool &types,
                                   const StringPool &strings,
                                   CloneRegistry &registry) {
	if (!fn->isCfn) return;
	if (fn->Name != "clone") return;
	if (fn->Args.size() != 1) return;
	const Param &p = fn->Args[0];
	if (p.Name != "self") return;
	if (p.Mode != ParamMode::Let) return;

	const TypeKey &key = types.get(p.Type);
	if (key.kind != TypeKind::Struct && key.kind != TypeKind::Named) { return; }
	StringIdx nameIdx = static_cast<StringIdx>(key.a);
	if (nameIdx == kNoString) return;
	const std::string &structName = strings.get(nameIdx);
	registry[structName] = fn;
}

void addCloneCandidates(CloneRegistry &registry, const ModuleAST &module,
                        const TypePool &types, const StringPool &strings) {
	for (const auto &fn : module.Functions) {
		considerCloneCandidate(fn.get(), types, strings, registry);
	}
	for (const auto &s : module.Structs) {
		for (const auto &m : s->Methods) {
			considerCloneCandidate(m.get(), types, strings, registry);
		}
	}
}

}  // namespace drops
}  // namespace jam
