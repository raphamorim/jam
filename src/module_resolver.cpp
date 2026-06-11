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
      nodeStore(&nodeStore_) {
	std::error_code ec;
	baseAbs_ = fs::canonical(baseDir, ec);
	baseAbsOk_ = !ec;
}

std::string ModuleResolver::resolve(const std::string &importPath) const {
	// Memoized per spelling: each uncached resolve runs up to five
	// exists/canonical probes, and the same spelling resolves several
	// times per import edge (canonicalKey, getOrLoadModule, chain
	// walks). The filesystem doesn't change mid-compile.
	auto cached = resolveCache_.find(importPath);
	if (cached != resolveCache_.end()) return cached->second;
	std::string r = resolveUncached(importPath);
	resolveCache_.emplace(importPath, r);
	return r;
}

std::string
ModuleResolver::resolveUncached(const std::string &importPath) const {
	// `test` stays a compiler-builtin namespace (provides `assert`).
	// `std` used to short-circuit too, but now resolves to a real
	// `std/std.jam` file that re-exports the standard-library modules.
	if (importPath == "test") { return importPath; }
	std::string path = importPath;

	// An explicitly file-relative spelling (`./x`, `../x`) names a file
	// near the importer and nothing else — it must not fall through to
	// the standard-library lookups below.
	bool explicitRelative =
	    path.rfind("./", 0) == 0 || path.rfind("../", 0) == 0;
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

	if (explicitRelative) { return ""; }  // never std for ./ or ../

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

namespace {

// Strip the `.jam` extension and collapse a `<dir>/mod` index file to
// `<dir>` — but only when no sibling `<dir>.jam` exists under `root`.
// `resolve` prefers the direct file, so when both exist they are two
// distinct modules and must keep distinct identities.
std::string finishIdentity(std::string id, const fs::path &rootAbs) {
	static const std::string ext = ".jam";
	if (id.size() > ext.size() &&
	    id.compare(id.size() - ext.size(), ext.size(), ext) == 0) {
		id.resize(id.size() - ext.size());
	}
	static const std::string modSuffix = "/mod";
	if (id.size() > modSuffix.size() &&
	    id.compare(id.size() - modSuffix.size(), modSuffix.size(), modSuffix) ==
	        0) {
		std::string collapsed = id.substr(0, id.size() - modSuffix.size());
		std::error_code ec;
		if (!fs::exists(rootAbs / (collapsed + ext), ec)) {
			id = std::move(collapsed);
		}
	}
	return id;
}

// Identity of `resolvedFile` relative to `rootAbs`, or "" when the file
// is not under that root.
std::string identityUnder(const std::string &resolvedFile,
                          const fs::path &rootAbs) {
	std::error_code ec;
	fs::path rel = fs::relative(resolvedFile, rootAbs, ec);
	if (ec || rel.empty()) return "";
	std::string id = rel.generic_string();
	if (id.rfind("..", 0) == 0 || id == ".") return "";
	return finishIdentity(std::move(id), rootAbs);
}

}  // namespace

std::string
ModuleResolver::moduleIdentity(const std::string &resolvedFile) const {
	// Map a resolved (canonical) module file to the stable identity used
	// as both the `loadedModules` cache key and the `modulePath`
	// mangling prefix, so the same file reached via different spellings
	// is one module. Three tiers:
	//   1. Under the entry base dir: the entry-relative path, `.jam`
	//      stripped (`lib/b`). `import("lib/b")` and `import("./b")`
	//      from `lib/a` agree on `lib/b`.
	//   2. Under the standard-library root (installed or the in-tree
	//      dev `std/` fallback): `std/<name>`. `import("fs")` and
	//      `import("std/fs")` agree on `std/fs` — without this the std
	//      file loads twice and registers its types under two
	//      identities.
	//   3. Anywhere else: the canonical absolute path, `.jam` stripped.
	//      Out-of-tree imports still converge per file. (The reference
	//      compiler rejects these outright; jam permits them.)
	// The roots are loop-invariant: the base dir canonicalizes once in
	// the ctor, and the std roots once per process (the installed root
	// is a static; the dev fallback hangs off the process CWD). Without
	// this, every import edge re-walked all three with realpath.
	if (baseAbsOk_) {
		std::string id = identityUnder(resolvedFile, baseAbs_);
		if (!id.empty()) return id;
	}
	static const std::pair<bool, fs::path> stdAbsOnce = [] {
		std::error_code e;
		if (const auto &root = stdRoot(); root) {
			fs::path p = fs::canonical(*root, e);
			if (!e) return std::make_pair(true, p);
		}
		return std::make_pair(false, fs::path());
	}();
	if (stdAbsOnce.first) {
		std::string id = identityUnder(resolvedFile, stdAbsOnce.second);
		if (!id.empty()) return "std/" + id;
	}
	// Dev fallback root (`<CWD>/std`), mirroring resolve()'s last
	// lookup tier.
	static const std::pair<bool, fs::path> devAbsOnce = [] {
		std::error_code e;
		fs::path p = fs::canonical("std", e);
		return std::make_pair(!e, e ? fs::path() : p);
	}();
	if (devAbsOnce.first) {
		std::string id = identityUnder(resolvedFile, devAbsOnce.second);
		if (!id.empty()) return "std/" + id;
	}
	fs::path abs = fs::path(resolvedFile);
	std::string id = abs.generic_string();
	static const std::string ext = ".jam";
	if (id.size() > ext.size() &&
	    id.compare(id.size() - ext.size(), ext.size(), ext) == 0) {
		id.resize(id.size() - ext.size());
	}
	return id;
}

std::string ModuleResolver::canonicalKey(const std::string &importPath) const {
	if (importPath == "test") return importPath;
	std::string resolved = resolve(importPath);
	if (resolved.empty() || resolved == "test") return importPath;
	std::string id = moduleIdentity(resolved);
	return id.empty() ? importPath : id;
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

	// Key the cache by the file's canonical identity so every spelling
	// of the same file shares one module.
	std::string key = moduleIdentity(resolvedPath);
	if (key.empty()) key = importPath;
	return loadModuleAt(key, resolvedPath);
}

ModuleAST *ModuleResolver::loadModuleAt(const std::string &key,
                                        const std::string &resolvedPath) {
	auto it = loadedModules.find(key);
	if (it != loadedModules.end()) { return it->second.get(); }

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
	loadedModules[key] = std::move(module);

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
	// One nested resolver per MODULE (not per import edge): its ctor
	// canonicalizes the module dir, and its resolve cache serves every
	// edge this module declares.
	ModuleResolver nestedResolver(fs::path(resolvedPath).parent_path().string(),
	                              *typePool, *stringPool, *nodeStore);
	auto loadNested = [&](std::string &importPath) {
		if (importPath == "test") return;
		std::string nestedResolved = nestedResolver.resolve(importPath);
		if (nestedResolved.empty() || nestedResolved == "test") return;
		std::string id = moduleIdentity(nestedResolved);
		if (!id.empty()) importPath = id;
		// Load the file the nested resolver actually found. Re-resolving
		// the spelling from the entry dir could bind a DIFFERENT file —
		// e.g. a user `fmt.jam` beside the entry silently hijacking the
		// std library's internal `import("fmt")`.
		loadModuleAt(importPath, nestedResolved);
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
		fn->modulePath = key;
	}
	for (auto &s : modPtr->Structs) {
		for (auto &m : s->Methods) {
			if (m->isExtern || m->isExport) continue;
			m->modulePath = key;
		}
	}

	// Stamp the module path on every type declaration too, so a struct/
	// enum/union gets a qualified identity (`lib/b.Thing`) and two
	// modules that each define `pub const Thing = struct {…}` no longer
	// collide in the global by-name type registries. The entry module
	// (parsed directly in main.cpp, never via this resolver) keeps an
	// empty modulePath, so its types stay bare-named.
	for (auto &s : modPtr->Structs) { s->modulePath = key; }
	for (auto &e : modPtr->Enums) { e->modulePath = key; }
	for (auto &u : modPtr->Unions) { u->modulePath = key; }
	for (auto &c : modPtr->Consts) { c->modulePath = key; }

	return modPtr;
}

bool ModuleResolver::isLoaded(const std::string &importPath) const {
	return loadedModules.find(importPath) != loadedModules.end();
}

const std::unordered_map<std::string, std::unique_ptr<ModuleAST>> &
ModuleResolver::getLoadedModules() const {
	return loadedModules;
}
