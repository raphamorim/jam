/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Module resolution — resolve `import("…")` spellings to `.jam` files, load +
//! parse the transitive closure into the SHARED pools, and stamp each loaded
//! module's `module_path` (the mangling prefix). Ported from
//! `src/module_resolver.{h,cpp}`.
//!
//! The C++ recurses through nested resolvers; here [`ModuleResolver::load_all`]
//! drives a flat work-list (avoiding the self-borrow that recursion-into-the-
//! cache would create in Rust) and keys the cache by a file's canonical
//! *identity* so distinct spellings of one file converge to a single module.
//!
//! Path-resolution tiers mirror the C++ `resolveUncached`: a sibling `.jam` /
//! `dir/mod.jam` under the importing module's directory, then (unless the
//! spelling was explicitly `./` / `../`) the standard library — `JAM_STD_PATH`,
//! the installed exe-relative `lib/jam/std` root, or the in-tree `<CWD>/std/`
//! dev fallback, with a leading `std/` stripped.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use jam_core::diag::Diagnostics;
use jam_syntax::ast::ModuleAST;
use jam_syntax::ast_flat::{NodeStore, StringPool, TypePool};
use jam_syntax::lexer::Lexer;
use jam_syntax::parser::Parser;

/// Loads `.jam` modules reachable from an entry module into shared pools.
pub struct ModuleResolver {
    /// Canonical entry base directory (identities are computed relative to it).
    base_abs: Option<PathBuf>,
    loaded_keys: HashSet<String>,
    /// `(identity, module)` in load order — the C++ `getLoadedModules()`.
    pub loaded: Vec<(String, ModuleAST)>,
    /// `(importer identity, import spelling) -> resolved identity`. Lets the
    /// driver map a RELATIVE re-export (`pub const fmt = import("fmt")` in
    /// std/std.jam) to its real identity (`std/fmt`) for namespace registration,
    /// instead of the raw spelling which never matches a loaded key.
    pub import_identities: std::collections::HashMap<(String, String), String>,
}

impl ModuleResolver {
    /// A resolver rooted at the entry file's directory.
    pub fn new(base_dir: &str) -> Self {
        ModuleResolver {
            base_abs: std::fs::canonicalize(base_dir).ok(),
            loaded_keys: HashSet::new(),
            loaded: Vec::new(),
            import_identities: std::collections::HashMap::new(),
        }
    }

    /// Standard-library root, computed once per process (the C++ `stdRoot`,
    /// module_resolver.cpp:59-89). Order:
    ///   1. `JAM_STD_PATH` env var (the `--std-path` CLI flag sets it before
    ///      resolution starts) — used as-is when non-empty.
    ///   2. Walk up from the running binary's (symlink-resolved) path, picking
    ///      the first ancestor that holds a `lib/jam/std/` subtree. Covers both
    ///      the FHS install layout (`$PREFIX/bin/jam` + `$PREFIX/lib/jam/std`)
    ///      and a relocatable tarball (`<dir>/jam` + `<dir>/lib/jam/std`) with
    ///      one rule.
    /// (The `<CWD>/std` dev fallback lives in `resolve_uncached`.)
    fn std_root() -> Option<PathBuf> {
        static ROOT: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            if let Ok(p) = std::env::var("JAM_STD_PATH")
                && !p.is_empty()
            {
                return Some(PathBuf::from(p));
            }
            let exe = std::env::current_exe().ok()?;
            let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
            let mut cur = exe.as_path();
            while let Some(parent) = cur.parent() {
                let candidate = parent.join("lib").join("jam").join("std");
                if candidate.is_dir() {
                    return std::fs::canonicalize(&candidate).ok().or(Some(candidate));
                }
                cur = parent;
            }
            None
        })
        .clone()
    }

    /// Resolve `spelling` (imported from `from_dir`) to a canonical `.jam` path.
    /// `None` for the `test` builtin, unresolvable spellings, and any error.
    fn resolve_uncached(&self, spelling: &str, from_dir: &Path) -> Option<PathBuf> {
        if spelling == "test" {
            return None;
        }
        let explicit_relative = spelling.starts_with("./") || spelling.starts_with("../");
        let path = spelling.strip_prefix("./").unwrap_or(spelling);

        let direct = from_dir.join(format!("{path}.jam"));
        if direct.is_file() {
            return std::fs::canonicalize(&direct).ok();
        }
        let index = from_dir.join(path).join("mod.jam");
        if index.is_file() {
            return std::fs::canonicalize(&index).ok();
        }
        if explicit_relative {
            return None;
        }

        // Standard library: strip a leading `std/`, then probe the env root and
        // the in-tree dev fallback.
        let std_path = path.strip_prefix("std/").unwrap_or(path);
        if let Some(root) = Self::std_root() {
            let fc = root.join(format!("{std_path}.jam"));
            if fc.is_file() {
                return std::fs::canonicalize(&fc).ok();
            }
            let ic = root.join(std_path).join("mod.jam");
            if ic.is_file() {
                return std::fs::canonicalize(&ic).ok();
            }
        }
        let dev = Path::new("std").join(format!("{std_path}.jam"));
        if dev.is_file() {
            return std::fs::canonicalize(&dev).ok();
        }
        let dev_idx = Path::new("std").join(std_path).join("mod.jam");
        if dev_idx.is_file() {
            return std::fs::canonicalize(&dev_idx).ok();
        }
        None
    }

    /// Stable identity of a resolved file: entry-relative, `std/<name>`, or the
    /// absolute path (`.jam` stripped). Doubles as the cache key and the
    /// `module_path` mangling prefix.
    fn module_identity(&self, resolved: &Path) -> String {
        if let Some(base) = &self.base_abs
            && let Some(id) = identity_under(resolved, base)
        {
            return id;
        }
        if let Some(root) = Self::std_root()
            && let Ok(root_abs) = std::fs::canonicalize(&root)
            && let Some(id) = identity_under(resolved, &root_abs)
        {
            return format!("std/{id}");
        }
        if let Ok(dev_abs) = std::fs::canonicalize("std")
            && let Some(id) = identity_under(resolved, &dev_abs)
        {
            return format!("std/{id}");
        }
        let s = resolved.to_string_lossy();
        s.strip_suffix(".jam").unwrap_or(&s).to_string()
    }

