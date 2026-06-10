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
	TypePool *typePool;
	StringPool *stringPool;
	NodeStore *nodeStore;
	std::vector<std::unique_ptr<StructDeclAST>> *sharedAnonStructs_ = nullptr;
	std::vector<std::unique_ptr<EnumDeclAST>> *sharedAnonEnums_ = nullptr;
	std::unordered_map<std::string, std::unique_ptr<ModuleAST>> loadedModules;

	// Canonical entry-relative identity for a resolved module file, used
	// as the cache key and `modulePath` prefix. Empty when the file lies
	// outside the base dir (e.g. std-library modules).
	std::string moduleIdentity(const std::string &resolvedFile) const;

	std::string readFile(const std::string &path) const;

	std::unique_ptr<ModuleAST> parseSource(const std::string &source) const;
};

#endif  // MODULE_RESOLVER_H
