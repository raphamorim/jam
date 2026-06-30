/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The declaration tree — ported 1:1 from `src/ast.h`.
//!
//! This is the parser's output: a [`ModuleAST`] of declarations. Function
//! bodies are sequences of [`NodeIdx`] into the shared flat `NodeStore`. The
//! C++ used `unique_ptr<FunctionAST>` for pointer stability; the Rust port owns
//! the values directly (`Vec<FunctionAST>`), and the raw-pointer-to-`FunctionId`
//! arena refactor lands when decl/modules need stable cross-references.
//!
//! The C++ `mutable mangledNameCache` is intentionally omitted here — it is a
//! codegen-time memo, not parser data, and will be a `OnceCell` when mangling
//! is ported.

use jam_core::ParamMode;
use jam_core::index::{NodeIdx, TypeIdx};

use crate::ast_flat::builtin;

/// One function parameter. `mode` defaults to `Let` when unannotated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: TypeIdx,
    pub mode: ParamMode,
    /// Declared with `comp` (e.g. `fn f(comp n: u32)`): the value must be known
    /// at the call site and is bound into the substitution map at
    /// instantiation, not lowered as an LLVM parameter.
    pub is_comp: bool,
}

impl Param {
    pub fn new(name: impl Into<String>, ty: TypeIdx) -> Param {
        Param {
            name: name.into(),
            ty,
            mode: ParamMode::Let,
            is_comp: false,
        }
    }
}

/// Function declaration. The body is flat-AST node indices owned by the shared
/// `NodeStore`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionAST {
    pub name: String,
    pub args: Vec<Param>,
    /// `kNoType` if void / unspecified.
    pub return_type: TypeIdx,
    pub body: Vec<NodeIdx>,
    pub is_extern: bool,
    pub is_export: bool,
    pub is_pub: bool,
    pub is_test: bool,
    pub is_var_args: bool,
    /// Declared with `cfn` inside a struct body — opts a method into the
    /// compiler-synthesized-call set (drop / at / default).
    pub is_cfn: bool,
    /// Top-level `cfn` — a compile-time function whose body runs at compile
    /// time, emitting runtime code into the caller via `@emit*` intrinsics.
    pub is_comp_time_fn: bool,
    /// Enclosing struct name for a method (empty for free fns / qualified clones).
    pub parent_struct: String,
    /// Owning module path (empty for the entry module / qualified clones).
    pub module_path: String,
}

impl FunctionAST {
    pub fn new(
        name: impl Into<String>,
        args: Vec<Param>,
        return_type: TypeIdx,
        body: Vec<NodeIdx>,
    ) -> FunctionAST {
        FunctionAST {
            name: name.into(),
            args,
            return_type,
            body,
            is_extern: false,
            is_export: false,
            is_pub: false,
            is_test: false,
            is_var_args: false,
            is_cfn: false,
            is_comp_time_fn: false,
            parent_struct: String::new(),
            module_path: String::new(),
        }
    }

    /// A function is generic iff its return type is the meta-type `type`, it is
    /// a compile-time fn, or any parameter is `type`-typed or `comp`. Generic
    /// functions are instantiated per call site, not lowered at decl time.
    pub fn is_generic(&self) -> bool {
        if self.return_type == builtin::TYPE {
            return true;
        }
        if self.is_comp_time_fn {
            return true;
        }
        self.args.iter().any(|p| p.ty == builtin::TYPE || p.is_comp)
    }
}

/// Top-level struct: `const Vec3 = struct { x: f32, y: f32 };`. Methods declared
/// in the body live in `methods`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDeclAST {
    pub name: String,
    pub fields: Vec<(String, TypeIdx)>,
    pub methods: Vec<FunctionAST>,
    pub is_pub: bool,
    /// Owning module identity (empty for the entry module); stamped post-parse.
    pub module_path: String,
}

impl StructDeclAST {
    pub fn new(name: impl Into<String>, fields: Vec<(String, TypeIdx)>) -> StructDeclAST {
        StructDeclAST {
            name: name.into(),
            fields,
            methods: Vec::new(),
            is_pub: false,
            module_path: String::new(),
        }
    }
}

/// One enum variant. Unit variants have an empty `payload_types`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumVariantAST {
    pub name: String,
    pub payload_types: Vec<TypeIdx>,
    /// Resolved discriminant value, set during parse.
    pub discriminant: u32,
}

/// Top-level enum: `const Direction = enum { Up, Down };` (variants may carry
/// positional payloads, e.g. `Jp(u16)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumDeclAST {
    pub name: String,
    pub variants: Vec<EnumVariantAST>,
    pub is_pub: bool,
    pub module_path: String,
}

impl EnumDeclAST {
    pub fn new(name: impl Into<String>, variants: Vec<EnumVariantAST>) -> EnumDeclAST {
        EnumDeclAST {
            name: name.into(),
            variants,
            is_pub: false,
            module_path: String::new(),
        }
    }
}

/// Top-level union: `const FloatBits = union { i: u32, f: f32 };` (untagged,
/// C-style).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionDeclAST {
    pub name: String,
    pub fields: Vec<(String, TypeIdx)>,
    pub is_pub: bool,
    pub module_path: String,
}

impl UnionDeclAST {
    pub fn new(name: impl Into<String>, fields: Vec<(String, TypeIdx)>) -> UnionDeclAST {
        UnionDeclAST {
            name: name.into(),
            fields,
            is_pub: false,
            module_path: String::new(),
        }
    }
}

