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

std::string
ModuleResolver::moduleIdentity(const std::string &resolvedFile) const {
	// Map a resolved (canonical) module file to the stable identity used
	// as both the `loadedModules` cache key and the `modulePath`
	// mangling prefix: the path relative to the entry base dir, with the
	// `.jam` extension stripped and forward slashes. It is the
	// project-root-relative path that names a module regardless of which
	// relative spelling reached it, so `import("lib/b")` and
	// `import("./b")` from `lib/a` agree on the identity `lib/b`.
	//
	// Returns "" when the file sits outside the base dir — std-library
	// modules resolve under their own root, and the caller keeps the
	// original import spelling, which `resolve` already maps through the
	// std root.
	std::error_code ec;
	fs::path baseAbs = fs::canonical(baseDir, ec);
	if (ec) return "";
	fs::path rel = fs::relative(resolvedFile, baseAbs, ec);
	if (ec || rel.empty()) return "";
	std::string id = rel.generic_string();
	if (id.rfind("..", 0) == 0) return "";  // escapes the base dir
	static const std::string ext = ".jam";
	if (id.size() > ext.size() &&
	    id.compare(id.size() - ext.size(), ext.size(), ext) == 0) {
		id.resize(id.size() - ext.size());
	}
	// A `<dir>/mod.jam` index file shares its directory's identity, so
	// `import("foo")` and `import("foo/mod")` name one module — matching
	// `resolve`, which maps a bare `import("foo")` to `foo/mod.jam`.
	static const std::string modSuffix = "/mod";
	if (id.size() > modSuffix.size() &&
	    id.compare(id.size() - modSuffix.size(), modSuffix.size(),
	               modSuffix) == 0) {
		id.resize(id.size() - modSuffix.size());
	}
	return id;
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

	std::string source = readFile(resolvedPath);
	if (source.empty()) {
		std::cerr << "Error: Cannot read module file: " << resolvedPath
		          << std::endl;
		return nullptr;
	}

	auto module = parseSource(source);
	if (!module) {
		std::cerr << "Error: Failed to parse module: " << resolvedPath
		          << std::endl;
		return nullptr;
	}

	// Register the parsed module in the cache BEFORE recursing into its
	// imports: the entry is published as soon as the module exists, with
	// no cycle check. A cyclic import (`bus.jam` imports `dma.jam`
	// imports `bus.jam`) then hits the cache and returns this same
	// partially-initialised ModuleAST instead of erroring. The
	// post-parse passes below (loadNested + module-path stamping) mutate
	// the module in place — by the time codegen / semantic analysis
	// touches a cyclic-import target, it's complete.
	ModuleAST *modPtr = module.get();
	loadedModules[importPath] = std::move(module);

	// Recursively load both regular imports (`const x = import(...)`)
	// and destructuring imports (`const { X } = import(...)`).
	//
	// Each nested import is resolved against THIS module's directory, so
	// a relative `./b` in `lib/a.jam` means `lib/b.jam` — not a same-
	// named file beside the entry module. We then rewrite the import's
	// path in place to its canonical entry-relative identity (see
	// `moduleIdentity`) and recurse on that. Together these resolve each
	// import against the importing file's directory and key the cache by
	// the resolved identity, so the same file reached via different
	// spellings is one module.
	//
	// The rewrite is load-bearing: without it `lib/b.jam` imported as
	// both `lib/b` (from the entry) and `./b` (from `lib/a`) would key
	// the cache under two strings, load twice, and register its `pub`
	// types twice — colliding in the global by-name type registry that
	// `main.cpp` builds. Collapsing to one identity dedupes the module
	// and keeps `modulePath` (the mangling prefix) stable.
	auto loadNested = [&](std::string &importPath) {
		if (importPath == "test") return;
		fs::path moduleDir = fs::path(resolvedPath).parent_path();
		ModuleResolver nestedResolver(moduleDir.string(), *typePool,
		                              *stringPool, *nodeStore);
		std::string nestedResolved = nestedResolver.resolve(importPath);
		if (nestedResolved.empty() || nestedResolved == "test") return;
		std::string id = moduleIdentity(nestedResolved);
		if (!id.empty()) importPath = id;
		getOrLoadModule(importPath);
	};
	for (auto &import : modPtr->Imports) { loadNested(import->Path); }
	for (auto &destImport : modPtr->DestructuringImports) {
		loadNested(destImport->Path);
	}

	// Stamp the import path on every function/method so the mangler
	// can build dotted fully-qualified names (`timer.Timer.read32`).
	// Without this,
	// two modules that both define `pub fn helper()` or `pub const
	// Counter = struct { pub fn init() }` would emit the same LLVM
	// symbol and the linker would silently merge them.
	//
	// Extern fns are skipped — they reference libc / external C
	// symbols by bare name (`malloc`, `free`, `printf`) and the
	// linker has to find those exactly. Same for export — the user
	// asked for that exact symbol to be visible to C callers.
	for (auto &fn : modPtr->Functions) {
		if (fn->isExtern || fn->isExport) continue;
		fn->modulePath = importPath;
	}
	for (auto &s : modPtr->Structs) {
		for (auto &m : s->Methods) {
			if (m->isExtern || m->isExport) continue;
			m->modulePath = importPath;
		}
	}

	return modPtr;
}

bool ModuleResolver::isLoaded(const std::string &importPath) const {
	return loadedModules.find(importPath) != loadedModules.end();
}

const std::unordered_map<std::string, std::unique_ptr<ModuleAST>> &
ModuleResolver::getLoadedModules() const {
	return loadedModules;
}
