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

    /// Load the transitive closure of `entry`'s imports into the shared pools,
    /// stamping each module's `module_path`. Idempotent per canonical identity.
    pub fn load_all(
        &mut self,
        entry: &ModuleAST,
        tp: &mut TypePool,
        sp: &mut StringPool,
        ns: &mut NodeStore,
    ) {
        let entry_dir = self.base_abs.clone().unwrap_or_else(|| PathBuf::from("."));
        // Anon-struct NAMES are numbered globally across all parses (shared-pool
        // numbering, matching the C++ shared anon registry): the entry's anon
        // structs are 0..E, then each loaded module continues from there.
        let mut anon_base = entry.anon_structs.len() as u32;
        // The C++ resolver is DFS PRE-ORDER (`for import : Imports { loadNested }`,
        // recursing immediately, imports before destructuring). This LIFO work-
        // list reproduces that order by pushing in REVERSE (so pops come out
        // forward) — load/parse order fixes the string-pool intern order.
        // Work tuple: (spelling, importer's dir, importer's identity). The
        // importer identity lets us record (importer, spelling) -> resolved id.
        let mut work: Vec<(String, PathBuf, String)> = Vec::new();
        for di in entry.destructuring_imports.iter().rev() {
            work.push((di.path.clone(), entry_dir.clone(), String::new()));
        }
        for imp in entry.imports.iter().rev() {
            work.push((imp.path.clone(), entry_dir.clone(), String::new()));
        }

        while let Some((spelling, from_dir, importer)) = work.pop() {
            if spelling == "test" {
                continue;
            }
            let resolved = match self.resolve_uncached(&spelling, &from_dir) {
                Some(r) => r,
                None => continue,
            };
            let key = self.module_identity(&resolved);
            // Record the importer's spelling -> resolved identity (even when the
            // target is already loaded — the importer still references it).
            if !key.is_empty() {
                self.import_identities
                    .insert((importer, spelling.clone()), key.clone());
            }
            if key.is_empty() || self.loaded_keys.contains(&key) {
                continue;
            }
            self.loaded_keys.insert(key.clone());

            let source = match std::fs::read(&resolved) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut lexer = Lexer::new(source);
            if lexer.scan_tokens().is_err() {
                continue;
            }
            let tokens = lexer.tokens().to_vec();
            let src_vec = lexer.source().to_vec();
            let mut diags = Diagnostics::new();
            let path_str = resolved.to_string_lossy().into_owned();
            let mut module = {
                let mut p = Parser::new(tokens, src_vec, tp, sp, ns, &mut diags, &path_str);
                p.set_anon_base(anon_base);
                match p.parse() {
                    Ok(m) => m,
                    Err(_) => continue,
                }
            };
            anon_base += module.anon_structs.len() as u32;
            stamp_module_path(&mut module, &key);

            // Nested imports resolve against THIS module's directory.
            let mod_dir = resolved
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            for di in module.destructuring_imports.iter().rev() {
                work.push((di.path.clone(), mod_dir.clone(), key.clone()));
            }
            for imp in module.imports.iter().rev() {
                work.push((imp.path.clone(), mod_dir.clone(), key.clone()));
            }
            self.loaded.push((key, module));
        }
    }

    /// Validate the ENTRY module's imports the way the C++ driver does
    /// (`main.cpp:1190-1232` + `getOrLoadModule`/`canonicalKey`): every
    /// non-`test` import must resolve to a loadable module, and every
    /// destructuring-import name must be EXPORTED (`pub`) from its module.
    /// Returns the exact multi-line stderr text the C++ prints (without a
    /// trailing newline) on the first failure, mirroring its exit-1 abort.
    ///
    /// `entry_dir` is the entry file's directory; `entry_file` is the entry
    /// file's display path (for the destructuring `<file>: error:` prefix).
    pub fn validate_entry_imports(
        &self,
        entry: &ModuleAST,
        entry_file: &str,
    ) -> Result<(), String> {
        let entry_dir = self.base_abs.clone().unwrap_or_else(|| PathBuf::from("."));
        let entry_dir = entry_dir.as_path();
        // Resolve an import base spelling to its identity key, or `None` if it
        // cannot be resolved (the C++ `canonicalKey` + `getOrLoadModule` miss).
        let resolve_base = |spelling: &str| -> Option<String> {
            if spelling == "test" {
                return Some("test".to_string());
            }
            let resolved = self.resolve_uncached(spelling, entry_dir)?;
            let key = self.module_identity(&resolved);
            Some(if key.is_empty() {
                spelling.to_string()
            } else {
                key
            })
        };
        // Walk a base + `.seg` re-export chain to the final module key.
        // `Err(failing_path)` reproduces the C++ `resolved.first` on a miss.
        let resolve_chain = |base: &str, chain: &[String]| -> Result<String, String> {
            // For an empty chain (the common case), a failed base resolution is
            // the only failure mode. `getOrLoadModule` already printed the
            // "Cannot resolve import path" line in C++; we synthesize both.
            let mut cur_path = self
                .canonical_spelling(base, entry_dir)
                .unwrap_or_else(|| base.to_string());
            let mut cur_loaded = resolve_base(base).filter(|k| self.is_loaded(k));
            if cur_loaded.is_none() {
                return Err(cur_path);
            }
            for seg in chain {
                let key = cur_loaded.take().unwrap();
                let Some(re) = self
                    .find_module(&key)
                    .and_then(|m| m.imports.iter().find(|i| i.is_pub && i.name == *seg))
                else {
                    return Err(format!("{cur_path}.{seg}"));
                };
                cur_path = match resolve_base(&re.path).filter(|k| self.is_loaded(k)) {
                    Some(k) => k,
                    None => return Err(re.path.clone()),
                };
                cur_loaded = Some(cur_path.clone());
            }
            Ok(cur_loaded.unwrap_or(cur_path))
        };

        for imp in &entry.imports {
            if imp.path == "test" {
                continue;
            }
            if let Err(failing) = resolve_chain(&imp.path, &imp.chain) {
                return Err(format!(
                    "Error: Cannot resolve import path: {failing}\nError: Failed to load module: {failing}"
                ));
            }
        }
        for di in &entry.destructuring_imports {
            if di.path == "test" {
                // The `test` builtin only exports `assert`.
                for name in &di.names {
                    if name != "assert" {
                        return Err(format!(
                            "{entry_file}: error: symbol `{name}` is not exported from module `test`"
                        ));
                    }
                }
                continue;
            }
            let key = match resolve_chain(&di.path, &di.chain) {
                Ok(k) => k,
                Err(failing) => {
                    return Err(format!(
                        "Error: Cannot resolve import path: {failing}\nError: Failed to load module: {failing}"
                    ));
                }
            };
            let module = self.find_module(&key);
            for name in &di.names {
                let exported = module.is_some_and(|m| module_exports(m, name));
                if !exported {
                    return Err(format!(
                        "{entry_file}: error: symbol `{name}` is not exported from module `{key}`"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Canonical-key spelling of an import (the C++ `canonicalKey`): the
    /// module identity, or the original spelling when unresolvable.
    fn canonical_spelling(&self, spelling: &str, from_dir: &Path) -> Option<String> {
        let resolved = self.resolve_uncached(spelling, from_dir)?;
        let key = self.module_identity(&resolved);
        Some(if key.is_empty() {
            spelling.to_string()
        } else {
            key
        })
    }

    fn is_loaded(&self, key: &str) -> bool {
        key == "test" || self.loaded.iter().any(|p| p.0 == key)
    }

    fn find_module(&self, key: &str) -> Option<&ModuleAST> {
        self.loaded.iter().find(|p| p.0 == key).map(|p| &p.1)
    }
}

/// Whether `name` is an EXPORTED symbol of `module`. Mirrors the C++
/// `SymbolTable::registerModule` (symbol_table.cpp:11-52): ONLY `pub`
/// functions, `pub` types (struct/enum/union), `pub` consts, and `pub`
/// import re-exports are surfaced; everything else is module-private.
fn module_exports(module: &ModuleAST, name: &str) -> bool {
    module.functions.iter().any(|f| f.name == name && f.is_pub)
        || module.structs.iter().any(|s| s.name == name && s.is_pub)
        || module.enums.iter().any(|e| e.name == name && e.is_pub)
        || module.unions.iter().any(|u| u.name == name && u.is_pub)
        || module.consts.iter().any(|c| c.name == name && c.is_pub)
        || module.imports.iter().any(|i| i.name == name && i.is_pub)
}

/// Identity of `resolved` relative to `root`, or `None` when it isn't under it.
/// Strips `.jam` and collapses a `<dir>/mod` index to `<dir>` (unless a sibling
/// `<dir>.jam` exists).
fn identity_under(resolved: &Path, root: &Path) -> Option<String> {
    let rel = resolved.strip_prefix(root).ok()?;
    let mut id = rel.to_string_lossy().replace('\\', "/");
    if id.is_empty() || id == "." || id.starts_with("..") {
        return None;
    }
    if let Some(stripped) = id.strip_suffix(".jam") {
        id = stripped.to_string();
    }
    if let Some(collapsed) = id.strip_suffix("/mod")
        && !root.join(format!("{collapsed}.jam")).exists()
    {
        id = collapsed.to_string();
    }
    Some(id)
}

/// Stamp `module_path = key` on a loaded module's decls (the C++ post-parse
/// pass). Functions / methods skip `extern` (libc bare names) and `export`
/// (user-requested exact symbols); types and consts are stamped unconditionally.
fn stamp_module_path(module: &mut ModuleAST, key: &str) {
    for f in &mut module.functions {
        if !f.is_extern && !f.is_export {
            f.module_path = key.to_string();
        }
    }
    for s in &mut module.structs {
        for m in &mut s.methods {
            if !m.is_extern && !m.is_export {
                m.module_path = key.to_string();
            }
        }
        s.module_path = key.to_string();
    }
    for e in &mut module.enums {
        e.module_path = key.to_string();
    }
    for u in &mut module.unions {
        u.module_path = key.to_string();
    }
    for c in &mut module.consts {
        c.module_path = key.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &[u8], tp: &mut TypePool, sp: &mut StringPool, ns: &mut NodeStore) -> ModuleAST {
        let mut lexer = Lexer::new(src.to_vec());
        lexer.scan_tokens().expect("lex");
        let tokens = lexer.tokens().to_vec();
        let source = lexer.source().to_vec();
        let mut diags = Diagnostics::new();
        let mut p = Parser::new(tokens, source, tp, sp, ns, &mut diags, "entry.jam");
        let m = p.parse().expect("parse");
        assert!(!diags.has_errors());
        m
    }

    #[test]
    fn loads_sibling_import_and_stamps_module_path() {
        // A temp dir with a leaf module the entry imports by sibling name.
        let dir = std::env::temp_dir().join(format!("jam_res_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("leaf.jam"), b"pub fn leafFn() u32 { return 7; }").unwrap();

        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let entry = parse(
            b"const l = import(\"leaf\"); fn useIt() u32 { return 0; }",
            &mut tp,
            &mut sp,
            &mut ns,
        );

        let mut r = ModuleResolver::new(dir.to_str().unwrap());
        r.load_all(&entry, &mut tp, &mut sp, &mut ns);

        assert_eq!(r.loaded.len(), 1, "the sibling `leaf` module should load");
        assert_eq!(r.loaded[0].0, "leaf", "identity is the entry-relative name");
        // Its function carries the stamped module_path (the mangling prefix).
        assert_eq!(r.loaded[0].1.functions[0].name, "leafFn");
        assert_eq!(r.loaded[0].1.functions[0].module_path, "leaf");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_builtin_and_unresolvable_are_skipped() {
        let dir = std::env::temp_dir().join(format!("jam_res_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let entry = parse(
            b"const t = import(\"test\"); const x = import(\"does_not_exist\"); fn f() u32 { return 0; }",
            &mut tp,
            &mut sp,
            &mut ns,
        );
        let mut r = ModuleResolver::new(dir.to_str().unwrap());
        r.load_all(&entry, &mut tp, &mut sp, &mut ns);
        assert!(
            r.loaded.is_empty(),
            "`test` is a builtin and the other is unresolvable"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
