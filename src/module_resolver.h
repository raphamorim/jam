/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef MODULE_RESOLVER_H
#define MODULE_RESOLVER_H

#include "ast.h"
#include "ast_flat.h"
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Override the standard-library lookup root.
void setStdPathOverride(const std::string &path);

class ModuleResolver {
  public:
	ModuleResolver(const std::string &baseDir, TypePool &typePool,
	               StringPool &stringPool, NodeStore &nodeStore);

	std::string resolve(const std::string &importPath) const;

	// Canonical cache key for an import spelling: resolve it, then map
	// the file to its stable identity (entry-relative, `std/<name>` for
	// standard-library files, or the absolute path for out-of-tree
	// files). Distinct spellings of one file agree on the key; returns
	// the spelling unchanged when resolution fails (the load will then
	// report the error) or for the `test` builtin.
	std::string canonicalKey(const std::string &importPath) const;

	ModuleAST *getOrLoadModule(const std::string &importPath);

	bool isLoaded(const std::string &importPath) const;

	const std::unordered_map<std::string, std::unique_ptr<ModuleAST>> &
	getLoadedModules() const;

	const std::string &getBaseDir() const { return baseDir; }

	void
	setSharedAnonRegistries(std::vector<std::unique_ptr<StructDeclAST>> *as,
	                        std::vector<std::unique_ptr<EnumDeclAST>> *ae) {
		sharedAnonStructs_ = as;
		sharedAnonEnums_ = ae;
	}

  private:
	std::string baseDir;
	// fs::canonical(baseDir), computed once in the ctor — moduleIdentity
	// runs per import edge, and realpath walks every path component.
	std::filesystem::path baseAbs_;
	bool baseAbsOk_ = false;
	// resolve() results per import spelling: up to five exists/canonical
	// probes each, and the same spelling resolves repeatedly
	// (canonicalKey, then getOrLoadModule, then chain walks).
	mutable std::unordered_map<std::string, std::string> resolveCache_;
	TypePool *typePool;
	StringPool *stringPool;
	NodeStore *nodeStore;
	std::vector<std::unique_ptr<StructDeclAST>> *sharedAnonStructs_ = nullptr;
	std::vector<std::unique_ptr<EnumDeclAST>> *sharedAnonEnums_ = nullptr;
	std::unordered_map<std::string, std::unique_ptr<ModuleAST>> loadedModules;

	std::string resolveUncached(const std::string &importPath) const;

	// Canonical identity for a resolved module file, used as the cache
	// key and `modulePath` prefix: entry-relative for in-tree files,
	// `std/<name>` for standard-library files, the absolute path for
	// out-of-tree files. Empty only when canonicalization fails.
	std::string moduleIdentity(const std::string &resolvedFile) const;

	// Load (or return the cached) module for `key`, reading from
	// `resolvedPath` — the file the caller already resolved. Never
	// re-resolves: a nested import loads the file its own module's
	// resolver found, not whatever the entry dir happens to shadow it
	// with.
	ModuleAST *loadModuleAt(const std::string &key,
	                        const std::string &resolvedPath);

	std::string readFile(const std::string &path) const;

	std::unique_ptr<ModuleAST> parseSource(const std::string &source) const;
};

#endif  // MODULE_RESOLVER_H
