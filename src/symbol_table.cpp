/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "symbol_table.h"
#include <iostream>

void SymbolTable::registerModule(const std::string &modulePath,
                                 ModuleAST *module) {
	if (!module) return;

	auto &symbols = moduleSymbols[modulePath];

	auto registerPub = [&](const std::string &name, FunctionAST *func) {
		ExportedSymbol sym;
		sym.name = name;
		sym.modulePath = modulePath;
		sym.function = func;  // nullptr for non-function exports
		sym.isPub = true;
		symbols[name] = sym;
	};

	for (auto &func : module->Functions) {
		if (func->isPub) registerPub(func->Name, func.get());
	}
	// Top-level type / value decls are exportable when marked `pub`.
	// Downstream consumers only check for existence (codegen finds the
	// concrete struct/enum/union/const via its own per-module registry),
	// so the function pointer stays null.
	for (auto &s : module->Structs) {
		if (s->isPub) registerPub(s->Name, nullptr);
	}
	for (auto &e : module->Enums) {
		if (e->isPub) registerPub(e->Name, nullptr);
	}
	for (auto &u : module->Unions) {
		if (u->isPub) registerPub(u->Name, nullptr);
	}
	for (auto &c : module->Consts) {
		if (c->isPub) registerPub(c->Name, nullptr);
	}
	// `pub const X = import(...)` re-exports. Destructuring imports
	// (`const { X } = import("std");`) check existence through this
	// table; without surfacing re-exports here, every module-typed
	// destructure would be rejected as "not exported".
	for (auto &i : module->Imports) {
		if (i->isPub) registerPub(i->Name, nullptr);
	}
}

void SymbolTable::registerBuiltinSymbol(const std::string &modulePath,
                                        const std::string &name) {
	ExportedSymbol sym;
	sym.name = name;
	sym.modulePath = modulePath;
	sym.function = nullptr;
	sym.isPub = true;
	moduleSymbols[modulePath][name] = sym;
}

const ExportedSymbol *
SymbolTable::lookupByModule(const std::string &modulePath,
                            const std::string &name) const {
	auto modIt = moduleSymbols.find(modulePath);
	if (modIt == moduleSymbols.end()) { return nullptr; }

	auto symIt = modIt->second.find(name);
	if (symIt == modIt->second.end()) { return nullptr; }

	return &symIt->second;
}

const ExportedSymbol *SymbolTable::lookupByName(const std::string &name) const {
	// Search all modules for this symbol
	for (const auto &modPair : moduleSymbols) {
		auto symIt = modPair.second.find(name);
		if (symIt != modPair.second.end()) { return &symIt->second; }
	}
	return nullptr;
}

void SymbolTable::registerBinding(const std::string &localName,
                                  const std::string &modulePath,
                                  const std::string &symbolName) {
	bindings[localName] = {modulePath, symbolName};
}

const ExportedSymbol *
SymbolTable::lookupBinding(const std::string &localName) const {
	auto it = bindings.find(localName);
	if (it == bindings.end()) { return nullptr; }

	return lookupByModule(it->second.first, it->second.second);
}

bool SymbolTable::hasSymbol(const std::string &modulePath,
                            const std::string &name) const {
	return lookupByModule(modulePath, name) != nullptr;
}

std::vector<const ExportedSymbol *>
SymbolTable::getModuleSymbols(const std::string &modulePath) const {
	std::vector<const ExportedSymbol *> result;

	auto modIt = moduleSymbols.find(modulePath);
	if (modIt != moduleSymbols.end()) {
		for (const auto &symPair : modIt->second) {
			result.push_back(&symPair.second);
		}
	}

	return result;
}
