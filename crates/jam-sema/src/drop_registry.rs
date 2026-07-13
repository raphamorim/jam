/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Drop / clone registries — scan a module's functions for the
//! compiler-synthesized destructor / clone shapes and map each to its struct.
//!
//! A type `T` is *drop-bearing* when the program declares
//! `cfn drop(self: mut T)` (top-level or in-struct). `cfn` (not plain `fn`) is
//! required: only methods opted into the compiler-synthesized-call set fire at
//! scope exit — a plain `fn drop` is a user-invoked method, and auto-firing it
//! too would double-drop. The clone counterpart is `cfn clone(self: T) T`
//! (let-mode self — cloning borrows the original).
//!
//! The registries are `BTreeMap` so the codegen's scope-exit walk emits drop
//! calls in a deterministic order.
//! Registries borrow the module (`&'a FunctionAST`), which outlives them.

use std::collections::BTreeMap;

use jam_core::index::StringIdx;
use jam_core::param_mode::ParamMode;
use jam_syntax::ast::{FunctionAST, ModuleAST};
use jam_syntax::ast_flat::{StringPool, TypeKind, TypePool};

/// Struct name -> its user-defined `cfn drop`.
pub type DropRegistry<'a> = BTreeMap<String, &'a FunctionAST>;
/// Struct name -> its user-defined `cfn clone`.
pub type CloneRegistry<'a> = BTreeMap<String, &'a FunctionAST>;

/// Resolve a `self`-param type to the struct name it names (Struct/Named both
/// carry the name `StringIdx` in their `a` slot). Returns `None` for any other
/// kind or the empty-string sentinel.
fn receiver_struct_name(
    ty: jam_core::index::TypeIdx,
    types: &TypePool,
    strings: &StringPool,
) -> Option<String> {
    let key = types.get(ty);
    if key.kind != TypeKind::Struct && key.kind != TypeKind::Named {
        return None;
    }
    let ni = StringIdx::new(key.a);
    if ni.is_none() {
        return None;
    }
    Some(String::from_utf8_lossy(&strings.get(ni)).into_owned())
}

/// If `fn_ast` has the `cfn drop(self: mut <Struct>)` shape, register it under
/// the struct's name.
fn consider_drop_candidate<'a>(
    fn_ast: &'a FunctionAST,
    types: &TypePool,
    strings: &StringPool,
    registry: &mut DropRegistry<'a>,
) {
    if !fn_ast.is_cfn || fn_ast.name != "drop" || fn_ast.args.len() != 1 {
        return;
    }
    let p = &fn_ast.args[0];
    if p.name != "self" || p.mode != ParamMode::Mut {
        return;
    }
    if let Some(name) = receiver_struct_name(p.ty, types, strings) {
        registry.insert(name, fn_ast);
    }
}

/// If `fn_ast` has the `cfn clone(self: <Struct>) <Struct>` shape (let-mode
/// self), register it under the struct's name.
fn consider_clone_candidate<'a>(
    fn_ast: &'a FunctionAST,
    types: &TypePool,
    strings: &StringPool,
    registry: &mut CloneRegistry<'a>,
) {
    if !fn_ast.is_cfn || fn_ast.name != "clone" || fn_ast.args.len() != 1 {
        return;
    }
    let p = &fn_ast.args[0];
    if p.name != "self" || p.mode != ParamMode::Let {
        return;
    }
    if let Some(name) = receiver_struct_name(p.ty, types, strings) {
        registry.insert(name, fn_ast);
    }
}

/// Fold `module`'s drop fns (top-level + in-struct methods) into `registry`.
/// Used to merge imported modules' drops so a drop site in one module fires the
/// destructor of a type defined in another.
pub fn add_drop_candidates<'a>(
    registry: &mut DropRegistry<'a>,
    module: &'a ModuleAST,
    types: &TypePool,
    strings: &StringPool,
) {
    for f in &module.functions {
        consider_drop_candidate(f, types, strings, registry);
    }
    for s in &module.structs {
        for m in &s.methods {
            consider_drop_candidate(m, types, strings, registry);
        }
    }
}

/// Build a fresh [`DropRegistry`] from a single module.
pub fn build_drop_registry<'a>(
    module: &'a ModuleAST,
    types: &TypePool,
    strings: &StringPool,
) -> DropRegistry<'a> {
    let mut registry = DropRegistry::new();
    add_drop_candidates(&mut registry, module, types, strings);
    registry
}

/// Fold `module`'s clone fns (top-level + in-struct methods) into `registry`.
pub fn add_clone_candidates<'a>(
    registry: &mut CloneRegistry<'a>,
    module: &'a ModuleAST,
    types: &TypePool,
    strings: &StringPool,
) {
    for f in &module.functions {
        consider_clone_candidate(f, types, strings, registry);
    }
    for s in &module.structs {
        for m in &s.methods {
            consider_clone_candidate(m, types, strings, registry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jam_core::index::TypeIdx;
    use jam_syntax::ast::{FunctionAST, ModuleAST, Param, StructDeclAST};
    use jam_syntax::ast_flat::{StringPool, TypePool};

    /// `cfn <name>(self: <mode> <recv_ty>)` factory.
    fn cfn_self(name: &str, mode: ParamMode, recv_ty: TypeIdx) -> FunctionAST {
        let mut f = FunctionAST::new(name, Vec::new(), TypeIdx::NONE, Vec::new());
        f.is_cfn = true;
        let mut p = Param::new("self", recv_ty);
        p.mode = mode;
        f.args = vec![p];
        f
    }

    #[test]
    fn registers_top_level_cfn_drop() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ty = tp.intern_named(sp.intern(b"Counter"));
        let mut m = ModuleAST::new();
        m.functions = vec![cfn_self("drop", ParamMode::Mut, ty)];
        let reg = build_drop_registry(&m, &tp, &sp);
        assert!(reg.contains_key("Counter"));
        assert_eq!(reg["Counter"].name, "drop");
    }

    #[test]
    fn plain_fn_drop_is_ignored() {
        // Not `cfn` -> not an auto-fired destructor.
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ty = tp.intern_named(sp.intern(b"Counter"));
        let mut f = cfn_self("drop", ParamMode::Mut, ty);
        f.is_cfn = false;
        let mut m = ModuleAST::new();
        m.functions = vec![f];
        assert!(build_drop_registry(&m, &tp, &sp).is_empty());
    }

    #[test]
    fn wrong_self_mode_is_ignored() {
        // drop needs `mut self`; let-mode self is not a destructor.
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ty = tp.intern_named(sp.intern(b"Counter"));
        let mut m = ModuleAST::new();
        m.functions = vec![cfn_self("drop", ParamMode::Let, ty)];
        assert!(build_drop_registry(&m, &tp, &sp).is_empty());
    }

    #[test]
    fn in_struct_drop_method_registers() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ty = tp.intern_named(sp.intern(b"Buf"));
        let mut s = StructDeclAST::new("Buf", Vec::new());
        s.methods = vec![cfn_self("drop", ParamMode::Mut, ty)];
        let mut m = ModuleAST::new();
        m.structs = vec![s];
        let reg = build_drop_registry(&m, &tp, &sp);
        assert!(reg.contains_key("Buf"));
    }

    #[test]
    fn clone_registry_needs_let_self() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ty = tp.intern_named(sp.intern(b"Vec"));
        let mut m = ModuleAST::new();
        m.functions = vec![
            cfn_self("clone", ParamMode::Let, ty), // valid
            cfn_self("clone", ParamMode::Mut, ty), // wrong mode -> ignored (overwrites? no)
        ];
        let mut reg = CloneRegistry::new();
        add_clone_candidates(&mut reg, &m, &tp, &sp);
        // The let-mode one registered; the mut-mode one was rejected. Since the
        // valid one comes first and the invalid is skipped, Vec maps to clone.
        assert!(reg.contains_key("Vec"));
        assert_eq!(reg["Vec"].args[0].mode, ParamMode::Let);
    }

    #[test]
    fn add_drop_candidates_merges_modules() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let ta = tp.intern_named(sp.intern(b"A"));
        let tb = tp.intern_named(sp.intern(b"B"));
        let mut ma = ModuleAST::new();
        ma.functions = vec![cfn_self("drop", ParamMode::Mut, ta)];
        let mut mb = ModuleAST::new();
        mb.functions = vec![cfn_self("drop", ParamMode::Mut, tb)];
        let mut reg = DropRegistry::new();
        add_drop_candidates(&mut reg, &ma, &tp, &sp);
        add_drop_candidates(&mut reg, &mb, &tp, &sp);
        assert_eq!(reg.len(), 2);
        assert!(reg.contains_key("A") && reg.contains_key("B"));
    }
}
