/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "module_resolver.h"
#include "lexer.h"
#include "parser.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

// Maximum filesystem path length we accept for the executable lookup.
#if defined(__APPLE__)
constexpr size_t kMaxPathBytes = 1024;  // macOS PATH_MAX
#elif defined(__linux__)
constexpr size_t kMaxPathBytes = 4096;  // Linux PATH_MAX
#endif

std::optional<std::string> g_stdPathOverride;

// Locate the running jam executable's filesystem path, with symlinks
// resolved. Returns std::nullopt on platforms we don't handle so
// callers fall back to other lookups.
std::optional<std::string> getExecutablePath() {
#if defined(__APPLE__)
	char buf[kMaxPathBytes];
	uint32_t size = sizeof(buf);
	if (_NSGetExecutablePath(buf, &size) != 0) return std::nullopt;
	char real[kMaxPathBytes];
	if (realpath(buf, real) != nullptr) return std::string(real);
	return std::string(buf);
#elif defined(__linux__)
	char buf[kMaxPathBytes];
	ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (len <= 0) return std::nullopt;
	buf[len] = '\0';
	return std::string(buf);
#else
	return std::nullopt;
#endif
}

// Standard-library root once per process. Order:
//   1. `--std-path <path>` CLI flag (via setStdPathOverride).
//   2. `JAM_STD_PATH` env var — used as-is when non-empty.
//   3. Walk up from the running binary's directory, picking the first
//      ancestor that holds a `lib/jam/std/` subtree. Covers both the
//      FHS install layout (`$PREFIX/bin/jam` + `$PREFIX/lib/jam/std`)
//      and a relocatable tarball (`<dir>/jam` + `<dir>/lib/jam/std`)
//      with one rule.
const std::optional<std::string> &stdRoot() {
	static const std::optional<std::string> root =
	    []() -> std::optional<std::string> {
		if (g_stdPathOverride) return g_stdPathOverride;
		if (const char *env = std::getenv("JAM_STD_PATH")) {
			if (env[0] != '\0') return std::string(env);
		}
		auto exe = getExecutablePath();
		if (!exe) return std::nullopt;
		fs::path cur = fs::path(*exe);
		while (cur.has_parent_path() && cur.parent_path() != cur) {
			cur = cur.parent_path();
			fs::path candidate = cur / "lib" / "jam" / "std";
			std::error_code ec;
			if (fs::is_directory(candidate, ec)) {
				return fs::canonical(candidate, ec).string();
			}
		}
		return std::nullopt;
	}();
	return root;
}

}  // namespace

void setStdPathOverride(const std::string &path) {
	if (path.empty()) g_stdPathOverride.reset();
	else g_stdPathOverride = path;
}

ModuleResolver::ModuleResolver(const std::string &baseDir, TypePool &typePool_,
                               StringPool &stringPool_, NodeStore &nodeStore_)
    : baseDir(baseDir), typePool(&typePool_), stringPool(&stringPool_),
      nodeStore(&nodeStore_) {}

std::string ModuleResolver::resolve(const std::string &importPath) const {
	// `test` stays a compiler-builtin namespace (provides `assert`).
	// `std` used to short-circuit too, but now resolves to a real
	// `std/std.jam` file that re-exports the standard-library modules.
	if (importPath == "test") { return importPath; }
	std::string path = importPath;

	if (path.substr(0, 2) == "./") { path = path.substr(2); }

	fs::path base(baseDir);
	fs::path directPath = base / (path + ".jam");
	if (fs::exists(directPath) && fs::is_regular_file(directPath)) {
		return fs::canonical(directPath).string();
	}

	fs::path indexPath = base / path / "mod.jam";
	if (fs::exists(indexPath) && fs::is_regular_file(indexPath)) {
		return fs::canonical(indexPath).string();
	}

	// Standard-library lookup. Accept both `import("collections")` and
	// `import("std/collections")` spellings by stripping a leading
	// `std/` so the bare module name resolves under the std root.
	std::string stdPath = path;
	if (stdPath.rfind("std/", 0) == 0) { stdPath = stdPath.substr(4); }

	if (const auto &root = stdRoot(); root) {
		fs::path fileCandidate = fs::path(*root) / (stdPath + ".jam");
		if (fs::exists(fileCandidate) && fs::is_regular_file(fileCandidate)) {
			return fs::canonical(fileCandidate).string();
		}
		fs::path indexCandidate = fs::path(*root) / stdPath / "mod.jam";
		if (fs::exists(indexCandidate) && fs::is_regular_file(indexCandidate)) {
			return fs::canonical(indexCandidate).string();
		}
	}

	// In-tree dev fallback: `<CWD>/std/<path>.jam`. Lets a fresh build
	// of jam.out run unit tests without first installing the std lib.
	fs::path devPath = fs::path("std") / (stdPath + ".jam");
	if (fs::exists(devPath) && fs::is_regular_file(devPath)) {
		return fs::canonical(devPath).string();
	}

	return "";  // Not found
}

std::string ModuleResolver::readFile(const std::string &path) const {
	std::ifstream file(path);
	if (!file.is_open()) { return ""; }
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

std::unique_ptr<ModuleAST>
ModuleResolver::parseSource(const std::string &source) const {
	Lexer lexer(source);
	std::vector<Token> tokens = lexer.scanTokens();
	Parser parser(tokens, lexer.sourceBuffer(), *typePool, *stringPool,
	              *nodeStore);
	parser.sharedAnonStructs = sharedAnonStructs_;
	parser.sharedAnonEnums = sharedAnonEnums_;
	return parser.parse();
}

ModuleAST *ModuleResolver::getOrLoadModule(const std::string &importPath) {
	auto it = loadedModules.find(importPath);
	if (it != loadedModules.end()) { return it->second.get(); }

	if (currentlyLoading.count(importPath) > 0) {
		std::cerr << "Error: Circular import detected for module: "
		          << importPath << std::endl;
		return nullptr;
	}

	std::string resolvedPath = resolve(importPath);
	if (resolvedPath.empty()) {
		std::cerr << "Error: Cannot resolve import path: " << importPath
		          << std::endl;
		return nullptr;
	}

	if (resolvedPath == "test") {
		auto builtinModule = std::make_unique<ModuleAST>();
		loadedModules[importPath] = std::move(builtinModule);
		return loadedModules[importPath].get();
	}

	currentlyLoading.insert(importPath);
	std::string source = readFile(resolvedPath);
	if (source.empty()) {
		std::cerr << "Error: Cannot read module file: " << resolvedPath
		          << std::endl;
		currentlyLoading.erase(importPath);
		return nullptr;
	}

	auto module = parseSource(source);
	if (!module) {
		std::cerr << "Error: Failed to parse module: " << resolvedPath
		          << std::endl;
		currentlyLoading.erase(importPath);
		return nullptr;
	}

	// Recursively load both regular imports (`const x = import(...)`)
	// and destructuring imports (`const { X } = import(...)`). The
	// nested resolver runs in the source module's directory so relative
	// paths work; we then forward to the top-level `getOrLoadModule` so
	// every resolved module ends up in the shared `loadedModules` map
	// and gets its `pub` symbols registered by main.cpp.
	auto loadNested = [&](const std::string &importPath) {
		if (importPath == "test") return;
		fs::path modulePath(resolvedPath);
		std::string moduleDir = modulePath.parent_path().string();
		ModuleResolver nestedResolver(moduleDir, *typePool, *stringPool,
		                              *nodeStore);
		std::string nestedResolved = nestedResolver.resolve(importPath);
		if (!nestedResolved.empty() && nestedResolved != "test") {
			getOrLoadModule(importPath);
		}
	};
	for (const auto &import : module->Imports) { loadNested(import->Path); }
	for (const auto &destImport : module->DestructuringImports) {
		loadNested(destImport->Path);
	}

	// Stamp the import path on every function/method so the mangler
	// can build Zig-style FQNs (`timer.Timer.read32`). Without this,
	// two modules that both define `pub fn helper()` or `pub const
	// Counter = struct { pub fn init() }` would emit the same LLVM
	// symbol and the linker would silently merge them.
	//
	// Extern fns are skipped — they reference libc / external C
	// symbols by bare name (`malloc`, `free`, `printf`) and the
	// linker has to find those exactly. Same for export — the user
	// asked for that exact symbol to be visible to C callers.
	for (auto &fn : module->Functions) {
		if (fn->isExtern || fn->isExport) continue;
		fn->modulePath = importPath;
	}
	for (auto &s : module->Structs) {
		for (auto &m : s->Methods) {
			if (m->isExtern || m->isExport) continue;
			m->modulePath = importPath;
		}
	}

	currentlyLoading.erase(importPath);
	loadedModules[importPath] = std::move(module);
	return loadedModules[importPath].get();
}

bool ModuleResolver::isLoaded(const std::string &importPath) const {
	return loadedModules.find(importPath) != loadedModules.end();
}

const std::unordered_map<std::string, std::unique_ptr<ModuleAST>> &
ModuleResolver::getLoadedModules() const {
	return loadedModules;
}
