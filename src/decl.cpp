/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "decl.h"

#include <algorithm>

namespace jam {

DeclTable::DeclTable() {
	// Slot 0 = sentinel. kNoDecl reads as Invalid with no fields set.
	decls_.emplace_back();
}

DeclIndex DeclTable::create(DeclKind kind, std::string name) {
	DeclIndex idx = static_cast<DeclIndex>(decls_.size());
	Decl d;
	d.kind = kind;
	d.name = name;
	decls_.push_back(std::move(d));
	// First registration wins. Later duplicate-name decls (e.g. an
	// imported pub fn with the same bare name as a main-module fn) are
	// reachable only via qualified-name aliasing, which lives in the
	// codegen context's separate alias / handle tables.
	byName_.emplace(std::move(name), idx);
	return idx;
}

Decl &DeclTable::get(DeclIndex idx) { return decls_[idx]; }
const Decl &DeclTable::get(DeclIndex idx) const { return decls_[idx]; }

DeclIndex DeclTable::findByName(const std::string &name) const {
	auto it = byName_.find(name);
	if (it == byName_.end()) return kNoDecl;
	return it->second;
}

DeclIndex DeclTable::findByNameAndKind(const std::string &name,
                                       DeclKind kind) const {
	// Linear scan over all decls. byName_ only tracks the first
	// registration per name; for kind-aware lookup we need to walk
	// every entry. The number of decls is bounded by source size so
	// this stays O(decls) per call — acceptable for the per-struct /
	// per-call invocation pattern.
	for (std::size_t i = 1; i < decls_.size(); ++i) {
		if (decls_[i].kind == kind && decls_[i].name == name) {
			return static_cast<DeclIndex>(i);
		}
	}
	return kNoDecl;
}

void DeclTable::declareDependency(DeclIndex from, DeclIndex to) {
	if (from == kNoDecl || to == kNoDecl || from == to) return;
	auto &f = decls_[from];
	if (std::find(f.dependencies.begin(), f.dependencies.end(), to) ==
	    f.dependencies.end()) {
		f.dependencies.push_back(to);
	}
	auto &t = decls_[to];
	if (std::find(t.dependants.begin(), t.dependants.end(), from) ==
	    t.dependants.end()) {
		t.dependants.push_back(from);
	}
}

}  // namespace jam
