/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The declaration table — the per-compilation registry of every named
//! top-level binding (functions, types, consts) from the main module and every
//! imported module's `pub` items. Ported from `src/decl.{h,cpp}`.
//!
//! `Decl`s are held by index (`DeclIndex`) so references stay stable across the
//! vector's reallocation. Index 0 is the sentinel (`DeclIndex::NONE`).
//!
//! ## Lifetime
//!
//! A `Decl` borrows AST nodes (`&'a FunctionAST`, …) and embeds a
//! [`FnSignature`] whose `ParamAbi`/`ReturnAbi` borrow the LLVM context. The
//! C++ keeps these as two independent borrows; here a single lifetime `'a`
//! covers both, since the driver's `Context` and `ModuleAST` both outlive the
//! table (the two borrows unify to the shorter at the call site).
//!
//! The mutation pattern matters for the analyzer port: the C++ `analyzeStruct`
//! holds a `Decl&` across a recursive re-fetch that can reallocate `decls_`.
//! The Rust analyzer must read by index, recurse, then write back by index —
//! never hold a `&mut Decl` across a call that can `create` another decl.

use jam_core::index::{DeclIndex, TypeIdx};
use jam_syntax::ast::{ConstDeclAST, EnumDeclAST, FunctionAST, StructDeclAST, UnionDeclAST};
use std::collections::HashMap;

use crate::abi::{ParamAbi, ReturnAbi};

/// Cached ABI signature of a non-generic function (set by the analyzer). Generic
/// functions skip this cache. `computed` distinguishes "analyzer ran" from the
/// default-constructed empty state.
#[derive(Clone, Debug, Default)]
pub struct FnSignature<'a> {
    pub params: Vec<ParamAbi<'a>>,
    pub return_abi: ReturnAbi<'a>,
    pub computed: bool,
}

/// What sort of source-level binding produced this `Decl`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DeclKind {
    #[default]
    Invalid,
    /// `fn name(...) ret { ... }` + extern + tfn.
    Function,
    /// `const Name = struct { ... };`
    Struct,
    /// `const Name = enum { ... };`
    Enum,
    /// `const Name = union { ... };`
    Union,
    /// `const NAME[: T]? = expr;` (value, not type alias).
    Const,
    /// `const NAME = SomeType` / `GenericCall`, evaluated to a type.
    TypeAlias,
}

/// Where this `Decl` sits in the analysis lifecycle. `InProgress` is the
/// cycle-detector key (re-entering it means a cycle).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DeclAnalysis {
    #[default]
    Unreferenced,
    InProgress,
    Complete,
    AnalysisFailure,
    DependencyFailure,
}

/// A struct body's lifecycle, distinct from its `Decl`'s value. Field
/// materialisation is demand-driven (`get_llvm_type` / `@sizeOf` / field access).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum StructStatus {
    #[default]
    None,
    FieldTypesWip,
    HaveFieldTypes,
    LayoutWip,
    HaveLayout,
}

/// Enum body lifecycle (`None -> BodyWip -> HaveBody`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum EnumStatus {
    #[default]
    None,
    BodyWip,
    HaveBody,
}

/// Union body lifecycle (`None -> BodyWip -> HaveBody`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum UnionStatus {
    #[default]
    None,
    BodyWip,
    HaveBody,
}

/// The value a fully-analyzed `Decl` resolves to. Jam's generic args are types
/// today, so no comptime-int/pointer variants are needed yet.
#[derive(Clone, Debug, Default)]
pub enum DeclValue<'a> {
    #[default]
    None,
    /// A type value (struct, enum, alias result, builtin).
    Type(TypeIdx),
    /// A function reference.
    Function(&'a FunctionAST),
}

/// One source-level declaration. Owned by [`DeclTable`]; held by index so it's
/// stable across reallocation. Exactly one `*_ast` is `Some`, selected by `kind`.
#[derive(Clone, Debug, Default)]
pub struct Decl<'a> {
    pub name: String,
    pub kind: DeclKind,
    pub analysis: DeclAnalysis,
    pub struct_status: StructStatus,
    pub enum_status: EnumStatus,
    pub union_status: UnionStatus,

    pub fn_ast: Option<&'a FunctionAST>,
    pub struct_ast: Option<&'a StructDeclAST>,
    pub enum_ast: Option<&'a EnumDeclAST>,
    pub union_ast: Option<&'a UnionDeclAST>,
    pub const_ast: Option<&'a ConstDeclAST>,

    /// Populated when `analysis == Complete`.
    pub value: DeclValue<'a>,
    /// Cached ABI signature for non-generic functions; check `computed`.
    pub signature: FnSignature<'a>,

    pub line: i32,
    pub file: String,

    /// Symmetric dependency graph: who I depend on / who depends on me.
    pub dependencies: Vec<DeclIndex>,
    pub dependants: Vec<DeclIndex>,
}

/// Central registry, one per compilation. Index 0 is a sentinel `Invalid` decl
/// so `DeclIndex::NONE` reads as "not found".
#[derive(Default)]
pub struct DeclTable<'a> {
    decls: Vec<Decl<'a>>,
    by_name: HashMap<String, DeclIndex>,
}

impl<'a> DeclTable<'a> {
    pub fn new() -> DeclTable<'a> {
        // Slot 0 = sentinel.
        DeclTable {
            decls: vec![Decl::default()],
            by_name: HashMap::new(),
        }
    }

    /// Create a new `Decl`. The caller fills the `*_ast` field matching `kind`
    /// after the call. Returns the new index.
    pub fn create(&mut self, kind: DeclKind, name: impl Into<String>) -> DeclIndex {
        let name = name.into();
        let idx = DeclIndex::new(self.decls.len() as u32);
        self.decls.push(Decl {
            kind,
            name: name.clone(),
            ..Default::default()
        });
        // First registration wins (later duplicate-name decls are reached only
        // via qualified-name aliasing elsewhere).
        self.by_name.entry(name).or_insert(idx);
        idx
    }

    pub fn get(&self, idx: DeclIndex) -> &Decl<'a> {
        &self.decls[idx.index()]
    }
    pub fn get_mut(&mut self, idx: DeclIndex) -> &mut Decl<'a> {
        &mut self.decls[idx.index()]
    }

    /// Find a decl by source-level name, or `NONE`.
    pub fn find_by_name(&self, name: &str) -> DeclIndex {
        self.by_name.get(name).copied().unwrap_or(DeclIndex::NONE)
    }

    /// Kind-filtered lookup (a fn and a type may share a name). Linear scan —
    /// `by_name` only tracks the first registration per name.
    pub fn find_by_name_and_kind(&self, name: &str, kind: DeclKind) -> DeclIndex {
        for i in 1..self.decls.len() {
            if self.decls[i].kind == kind && self.decls[i].name == name {
                return DeclIndex::new(i as u32);
            }
        }
        DeclIndex::NONE
    }

    /// Wire a symmetric `from -> to` dependency edge. Idempotent; self/sentinel
    /// edges are ignored.
    pub fn declare_dependency(&mut self, from: DeclIndex, to: DeclIndex) {
        if from.is_none() || to.is_none() || from == to {
            return;
        }
        {
            let f = &mut self.decls[from.index()];
            if !f.dependencies.contains(&to) {
                f.dependencies.push(to);
            }
        }
        {
            let t = &mut self.decls[to.index()];
            if !t.dependants.contains(&from) {
                t.dependants.push(from);
            }
        }
    }

    /// Number of registered decls, excluding the sentinel.
    pub fn len(&self) -> usize {
        self.decls.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All decls including the sentinel at index 0 (callers skip it).
    pub fn all(&self) -> &[Decl<'a>] {
        &self.decls
    }
}

