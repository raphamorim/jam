/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Exported-symbol registry — tracks the `pub` functions/types/consts/imports a
//! module exposes, resolvable by module path, by bare name (destructuring
//! imports), or through a registered binding.
//!
//! The table borrows the parsed modules for its lifetime. Only `pub` functions
//! carry the borrow — type / const / import re-exports store `None` (downstream
//! finds the concrete decl through its own per-module registry; the table only
//! answers existence).
//!
//! The inner maps are `BTreeMap`, not hash maps: `lookup_by_name` returns the
//! first match across modules, so a stable (path-sorted) scan order keeps the
//! result deterministic.

use std::collections::BTreeMap;

use jam_syntax::ast::{FunctionAST, ModuleAST};

/// A symbol exported from a module.
#[derive(Clone, Debug)]
pub struct ExportedSymbol<'a> {
    pub name: String,
    pub module_path: String,
    /// The function AST for a `pub fn`; `None` for type/const/import exports.
    pub function: Option<&'a FunctionAST>,
    pub is_pub: bool,
}

/// Registry of exported symbols, keyed by module path then symbol name, plus a
/// side table of destructuring-import bindings (local name -> (module, symbol)).
#[derive(Default)]
pub struct SymbolTable<'a> {
    module_symbols: BTreeMap<String, BTreeMap<String, ExportedSymbol<'a>>>,
    bindings: BTreeMap<String, (String, String)>,
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> SymbolTable<'a> {
        SymbolTable::default()
    }

    /// Register every `pub` declaration a module exposes. `pub fn`s carry their
    /// AST; `pub` struct/enum/union/const/import re-exports register with a
    /// `None` function (existence-only).
    pub fn register_module(&mut self, module_path: &str, module: &'a ModuleAST) {
        let mp = module_path.to_string();
        let symbols = self.module_symbols.entry(mp.clone()).or_default();

        for f in &module.functions {
            if f.is_pub {
                symbols.insert(
                    f.name.clone(),
                    ExportedSymbol {
                        name: f.name.clone(),
                        module_path: mp.clone(),
                        function: Some(f),
                        is_pub: true,
                    },
                );
            }
        }
        let mut put_null = |name: &str| {
            symbols.insert(
                name.to_string(),
                ExportedSymbol {
                    name: name.to_string(),
                    module_path: mp.clone(),
                    function: None,
                    is_pub: true,
                },
            );
        };
        for s in &module.structs {
            if s.is_pub {
                put_null(&s.name);
            }
        }
        for e in &module.enums {
            if e.is_pub {
                put_null(&e.name);
            }
        }
        for u in &module.unions {
            if u.is_pub {
                put_null(&u.name);
            }
        }
        for c in &module.consts {
            if c.is_pub {
                put_null(&c.name);
            }
        }
        // `pub const X = import(...)` re-exports: destructuring imports check
        // existence through this table, so re-exports must surface here.
        for i in &module.imports {
            if i.is_pub {
                put_null(&i.name);
            }
        }
    }

    /// Register a builtin symbol that has no backing AST function.
    pub fn register_builtin_symbol(&mut self, module_path: &str, name: &str) {
        self.module_symbols
            .entry(module_path.to_string())
            .or_default()
            .insert(
                name.to_string(),
                ExportedSymbol {
                    name: name.to_string(),
                    module_path: module_path.to_string(),
                    function: None,
                    is_pub: true,
                },
            );
    }

    /// Look up a symbol by its module path and name.
    pub fn lookup_by_module(&self, module_path: &str, name: &str) -> Option<&ExportedSymbol<'a>> {
        self.module_symbols.get(module_path)?.get(name)
    }

    /// Look up a symbol by bare name, returning the first match in path-sorted
    /// module order (deterministic).
    pub fn lookup_by_name(&self, name: &str) -> Option<&ExportedSymbol<'a>> {
        for syms in self.module_symbols.values() {
            if let Some(s) = syms.get(name) {
                return Some(s);
            }
        }
        None
    }

    /// Register a destructuring-import binding: `local_name -> (module, symbol)`.
    pub fn register_binding(&mut self, local_name: &str, module_path: &str, symbol_name: &str) {
        self.bindings.insert(
            local_name.to_string(),
            (module_path.to_string(), symbol_name.to_string()),
        );
    }

    /// Resolve a binding's local name to the bound symbol.
    pub fn lookup_binding(&self, local_name: &str) -> Option<&ExportedSymbol<'a>> {
        let (mp, sym) = self.bindings.get(local_name)?;
        self.lookup_by_module(mp, sym)
    }

    /// Does a module export this symbol?
    pub fn has_symbol(&self, module_path: &str, name: &str) -> bool {
        self.lookup_by_module(module_path, name).is_some()
    }

    /// All symbols a module exports (name-sorted).
    pub fn module_symbols(&self, module_path: &str) -> Vec<&ExportedSymbol<'a>> {
        match self.module_symbols.get(module_path) {
            Some(syms) => syms.values().collect(),
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jam_core::index::TypeIdx;
    use jam_syntax::ast::{FunctionAST, ModuleAST, StructDeclAST};

    fn pub_fn(name: &str) -> FunctionAST {
        let mut f = FunctionAST::new(name, Vec::new(), TypeIdx::NONE, Vec::new());
        f.is_pub = true;
        f
    }

    fn priv_fn(name: &str) -> FunctionAST {
        FunctionAST::new(name, Vec::new(), TypeIdx::NONE, Vec::new())
    }

    fn module_with(funcs: Vec<FunctionAST>, structs: Vec<StructDeclAST>) -> ModuleAST {
        let mut m = ModuleAST::new();
        m.functions = funcs;
        m.structs = structs;
        m
    }

    #[test]
    fn registers_pub_fns_and_skips_private() {
        let m = module_with(vec![pub_fn("add"), priv_fn("helper")], Vec::new());
        let mut t = SymbolTable::new();
        t.register_module("math", &m);
        assert!(t.has_symbol("math", "add"));
        assert!(!t.has_symbol("math", "helper"));
        let add = t.lookup_by_module("math", "add").unwrap();
        assert_eq!(add.module_path, "math");
        assert!(add.function.is_some());
        assert_eq!(add.function.unwrap().name, "add");
    }

    #[test]
    fn pub_struct_registers_with_null_function() {
        let mut s = StructDeclAST::new("Point", Vec::new());
        s.is_pub = true;
        let m = module_with(Vec::new(), vec![s]);
        let mut t = SymbolTable::new();
        t.register_module("geo", &m);
        let sym = t.lookup_by_module("geo", "Point").unwrap();
        assert!(sym.function.is_none());
        assert!(sym.is_pub);
    }

    #[test]
    fn lookup_by_name_is_deterministic_first_in_path_order() {
        // Same symbol name in two modules: BTreeMap path order means the
        // alphabetically-first module ("aaa") wins, deterministically.
        let ma = module_with(vec![pub_fn("dup")], Vec::new());
        let mz = module_with(vec![pub_fn("dup")], Vec::new());
        let mut t = SymbolTable::new();
        t.register_module("zzz", &mz);
        t.register_module("aaa", &ma);
        let found = t.lookup_by_name("dup").unwrap();
        assert_eq!(found.module_path, "aaa");
        assert!(t.lookup_by_name("missing").is_none());
    }

    #[test]
    fn bindings_resolve_through_module_lookup() {
        let m = module_with(vec![pub_fn("println")], Vec::new());
        let mut t = SymbolTable::new();
        t.register_module("std", &m);
        t.register_binding("print", "std", "println");
        let b = t.lookup_binding("print").unwrap();
        assert_eq!(b.name, "println");
        assert!(t.lookup_binding("nope").is_none());
    }

    #[test]
    fn builtin_symbol_and_module_listing() {
        let mut t = SymbolTable::new();
        t.register_builtin_symbol("builtin", "assert");
        assert!(t.has_symbol("builtin", "assert"));
        let m = module_with(vec![pub_fn("a"), pub_fn("b")], Vec::new());
        t.register_module("m", &m);
        let names: Vec<&str> = t
            .module_symbols("m")
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]); // name-sorted
        assert!(t.module_symbols("absent").is_empty());
    }
}
