/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Flat, tag-dispatched AST — ported 1:1 from `src/ast_flat.h`.
//!
//! [`AstNode`] is a 16-byte struct; indices replace pointers; variadic payloads
//! spill into [`NodeStore`]'s `extra` pool. The flat layout keeps nodes packed
//! so codegen walks a cache-friendly array via `match` on [`AstTag`].
//!
//! Conventions (load-bearing):
//!   * `NodeIdx(0)` is the null/absent node; slot 0 is reserved with
//!     `AstTag::Invalid` so a zero `AstNode` is the sentinel.
//!   * `lhs`/`rhs` are **raw `u32`** generic slots whose meaning depends on the
//!     tag (a `NodeIdx`, `StringIdx`, `TypeIdx`, `ExtraIdx`, or half of a
//!     value). The per-tag encodings are documented on [`AstTag`] and must match
//!     the C++ exactly — the parser writes them and astgen/codegen decode them.
//!   * `StringIdx(0)` = empty string, `TypeIdx(0)` = invalid (slot-0 sentinels).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};

/// Node tags (one byte; matched on in codegen). Per-tag payload encodings are
/// documented inline — the parser/astgen/codegen share these as a contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AstTag {
    Invalid = 0,

    // Literals
    NumberLit, // lhs = lo32(val), rhs = hi32(val); flags bit 0 = isNeg
    BoolLit,   // lhs = 0|1
    StringLit, // lhs = StringIdx

    // Lvalues / refs
    Variable,     // lhs = StringIdx (name)
    MemberAccess, // lhs = NodeIdx (object), rhs = StringIdx (member)
    Index,        // lhs = NodeIdx (object), rhs = NodeIdx (index)
    Slice,        // lhs = NodeIdx (base), rhs = ExtraIdx -> [start, end]
    Deref,        // lhs = NodeIdx (operand)
    AddressOf,    // lhs = NodeIdx (operand)

    // Operators
    UnaryOp,  // lhs = NodeIdx (operand); op kind in `op`
    BinaryOp, // lhs = NodeIdx, rhs = NodeIdx; op in `op`

    // Calls: lhs = StringIdx (callee FQN), rhs = ExtraIdx -> [argCount, arg0, ...]
    Call,

    // Statements
    Return,    // lhs = NodeIdx (operand) or kNoNode
    Assign,    // lhs = NodeIdx (target), rhs = NodeIdx (value)
    VarDecl,   // lhs = ExtraIdx -> [name, type, init]; rhs = flags (bit0 = isConst)
    IfNode,    // lhs = NodeIdx (cond); rhs = ExtraIdx -> [thenCount, elseCount, then.., else..]
    WhileNode, // lhs = NodeIdx (cond); rhs = ExtraIdx -> [bodyCount, body..]
    ForNode,   // lhs = ExtraIdx -> [var, start, end, bodyCount, body..]
    Break,
    Continue,

    // Module-level
    ImportLit,      // lhs = StringIdx (module path)
    StructLit,      // lhs = TypeIdx; rhs = ExtraIdx -> [fieldCount, name0, expr0, ...]
    ArrayLit,       // lhs = TypeIdx (elem); rhs = ExtraIdx -> [count, elem0, ...]
    ArrayRepeat,    // lhs = TypeIdx; rhs = ExtraIdx -> [valueNode, countNode]
    StructExpr,     // lhs = u32 (index into ModuleAST::AnonStructs)
    EnumExpr,       // lhs = u32 (index into ModuleAST::AnonEnums)
    MatchNode,      // lhs = NodeIdx (scrutinee); rhs = ExtraIdx -> [armCount, ...]
    AsCast,         // lhs = NodeIdx (operand), rhs = TypeIdx (target)
    AtCall,         // lhs = StringIdx (name); rhs = TypeIdx | ExtraIdx (flags bit0)
    TypeMethodCall, // lhs = TypeIdx (receiver); rhs = ExtraIdx -> [methodNameId, argCount, ...]

    // Pattern atoms (internal to MatchNode arms)
    PatLit,         // lhs = lo32, rhs = hi32, flags bit0 = isNeg
    PatRange,       // lhs = lo32 low, rhs = lo32 high
    PatWildcard,    // no payload
    PatOr,          // lhs = ExtraIdx -> [count, sub0, ...]
    PatEnumVariant, // lhs = StringIdx (enum), rhs = StringIdx (variant)

    Count, // sentinel for table sizing
}

/// Binary operator kinds (stored in `AstNode.op` for `BinaryOp`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BinOp {
    Invalid = 0,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    LogAnd,
    LogOr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operator kinds (stored in `AstNode.op` for `UnaryOp`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum UnaryOp {
    Invalid = 0,
    Neg,
    LogNot,
    BitNot,
}

/// 16 bytes: tag + op + flags + main_token + two raw `u32` data slots.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AstNode {
    pub tag: AstTag,
    /// `BinOp`/`UnaryOp` encoded as `u8`.
    pub op: u8,
    /// Tag-specific bit field (e.g. `NumberLit` isNeg in bit 0).
    pub flags: u16,
    /// Token index for error reporting.
    pub main_token: u32,
    pub lhs: u32,
    pub rhs: u32,
}

impl AstNode {
    pub const INVALID: AstNode = AstNode {
        tag: AstTag::Invalid,
        op: 0,
        flags: 0,
        main_token: 0,
        lhs: 0,
        rhs: 0,
    };
}

