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

/// Owns the flat node array, the `extra` `u32` pool for variadic payloads, and
/// the parallel per-node source-line table.
pub struct NodeStore {
    nodes: Vec<AstNode>,
    extra: Vec<u32>,
    lines: Vec<u32>,
}

impl Default for NodeStore {
    fn default() -> Self {
        NodeStore::new()
    }
}

impl NodeStore {
    pub fn new() -> NodeStore {
        // Reserve slot 0 as the null sentinel so kNoNode is a no-op.
        NodeStore {
            nodes: vec![AstNode::INVALID],
            extra: Vec::new(),
            lines: vec![0],
        }
    }

    pub fn add_node(&mut self, n: AstNode) -> NodeIdx {
        self.add_node_at(n, 0)
    }

    /// Add a node, recording its source line.
    pub fn add_node_at(&mut self, n: AstNode, line: u32) -> NodeIdx {
        let id = NodeIdx::from_usize(self.nodes.len());
        self.nodes.push(n);
        self.lines.push(line);
        id
    }

    pub fn get_line(&self, id: NodeIdx) -> u32 {
        self.lines.get(id.index()).copied().unwrap_or(0)
    }

    pub fn get(&self, id: NodeIdx) -> &AstNode {
        &self.nodes[id.index()]
    }
    pub fn get_mut(&mut self, id: NodeIdx) -> &mut AstNode {
        &mut self.nodes[id.index()]
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append a single `u32` to the extra pool; return its index.
    pub fn push_extra(&mut self, v: u32) -> ExtraIdx {
        let i = ExtraIdx::from_usize(self.extra.len());
        self.extra.push(v);
        i
    }

    /// Append a span of `u32`s; return the index of the first.
    pub fn push_extra_span(&mut self, data: &[u32]) -> ExtraIdx {
        let start = ExtraIdx::from_usize(self.extra.len());
        self.extra.extend_from_slice(data);
        start
    }

    /// Reserve `len` zeroed `u32`s; fill later via [`NodeStore::set_extra`].
    pub fn reserve_extra(&mut self, len: usize) -> ExtraIdx {
        let start = ExtraIdx::from_usize(self.extra.len());
        self.extra.resize(self.extra.len() + len, 0);
        start
    }

    pub fn get_extra(&self, i: ExtraIdx) -> u32 {
        self.extra[i.index()]
    }
    pub fn set_extra(&mut self, i: ExtraIdx, v: u32) {
        self.extra[i.index()] = v;
    }
    pub fn extra(&self) -> &[u32] {
        &self.extra
    }
}

/// Interned identifiers + string literals. The C++ used a `deque<string>` to
/// keep `string_view` keys stable; Rust owns the bytes in a `Vec` and keys the
/// map by the owned bytes, so no deque is needed. Stores `Vec<u8>` because
/// decoded string-literal values can be non-UTF-8 (`\xFF`).
/// Interior mutability (RefCell) so `intern` is `&self` — this mirrors the C++
/// `stringPool` being a `mutable` field, letting `&self` layout queries
/// (get_llvm_type/type_size) instantiate generics on-demand. `get` returns owned
/// bytes (every caller copies them anyway via `into_owned()`/`to_vec()`), so no
/// borrow is ever held across an `intern` — there is no double-borrow panic risk.
pub struct StringPool {
    strings: RefCell<Vec<Vec<u8>>>,
    idx: RefCell<HashMap<Vec<u8>, u32>>,
}

impl Default for StringPool {
    fn default() -> Self {
        StringPool::new()
    }
}

impl StringPool {
    pub fn new() -> StringPool {
        // Slot 0 reserved for kNoString (the empty string).
        let mut idx = HashMap::new();
        idx.insert(Vec::new(), 0u32);
        StringPool {
            strings: RefCell::new(vec![Vec::new()]),
            idx: RefCell::new(idx),
        }
    }

    pub fn intern(&self, s: &[u8]) -> StringIdx {
        if let Some(&id) = self.idx.borrow().get(s) {
            return StringIdx::new(id);
        }
        let mut strings = self.strings.borrow_mut();
        let id = strings.len() as u32;
        strings.push(s.to_vec());
        self.idx.borrow_mut().insert(s.to_vec(), id);
        StringIdx::new(id)
    }

    pub fn intern_str(&self, s: &str) -> StringIdx {
        self.intern(s.as_bytes())
    }

    /// Resolve a `StringIdx` to its (owned) bytes. Out-of-range indices return
    /// the empty slice — the AST dumper reads `lhs`/`rhs` of some nodes (e.g.
    /// `PatEnumVariant` with bindings, whose slots actually hold an ExtraIdx)
    /// as if they were string indices; the C++ oracle's `std::deque` yields a
    /// zeroed (empty) string for those reads, and we reproduce that exactly.
    pub fn get(&self, id: StringIdx) -> Vec<u8> {
        self.strings
            .borrow()
            .get(id.index())
            .cloned()
            .unwrap_or_default()
    }
    pub fn len(&self) -> usize {
        self.strings.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.strings.borrow().is_empty()
    }
}

/// Type kinds. See `src/ast_flat.h` for the per-kind `a`/`b` slot meanings.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TypeKind {
    Invalid = 0,
    Void,
    NoReturn,
    Bool,
    Int,         // a = bits, b = isSigned
    Float,       // a = bits
    PtrSingle,   // a = elem
    PtrMany,     // a = elem
    Slice,       // a = elem
    Array,       // a = elem, b = len
    ArrayExpr,   // a = elem, b = len-expr NodeIdx
    Struct,      // a = StringIdx (name)
    Enum,        // a = StringIdx (name)
    Union,       // a = StringIdx (name)
    Named,       // a = StringIdx (name)
    Type,        // the meta-type (singleton)
    GenericCall, // a = StringIdx (callee), b = genericArgs index
    Module,      // a = StringIdx (resolved path)
    Fn,          // a = return TypeIdx, b = fnParams index
}

/// Interned type key. `a`/`b` are raw `u32` slots whose meaning depends on
/// `kind`. Unused slots are always 0 (the constructors guarantee it), so the
/// derived `PartialEq`/`Hash` is exactly the C++ kind-aware `operator==`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeKey {
    pub kind: TypeKind,
    pub a: u32,
    pub b: u32,
}

impl TypeKey {
    pub const fn new(kind: TypeKind, a: u32, b: u32) -> TypeKey {
        TypeKey { kind, a, b }
    }
}

/// Built-in type indices — pre-interned at fixed slots by [`TypePool::new`].
pub mod builtin {
    use jam_core::index::TypeIdx;
    pub const VOID: TypeIdx = TypeIdx::new(1);
    pub const BOOL: TypeIdx = TypeIdx::new(2);
    pub const U8: TypeIdx = TypeIdx::new(3);
    pub const I8: TypeIdx = TypeIdx::new(4);
    pub const U16: TypeIdx = TypeIdx::new(5);
    pub const I16: TypeIdx = TypeIdx::new(6);
    pub const U32: TypeIdx = TypeIdx::new(7);
    pub const I32: TypeIdx = TypeIdx::new(8);
    pub const U64: TypeIdx = TypeIdx::new(9);
    pub const I64: TypeIdx = TypeIdx::new(10);
    pub const F32: TypeIdx = TypeIdx::new(11);
    pub const F64: TypeIdx = TypeIdx::new(12);
    pub const TYPE: TypeIdx = TypeIdx::new(13);
    pub const NORETURN: TypeIdx = TypeIdx::new(14);
    pub const U1: TypeIdx = BOOL;
}

/// Type interning. Built-ins occupy fixed pre-interned slots (see [`builtin`]).
/// Generic-call arg lists and fn-param lists are interned in ordered side
/// tables (`BTreeMap`, deterministic) so identical lists share an index — and
/// thus identical `GenericCall`/`Fn` keys share a `TypeIdx`.
/// Interior mutability (RefCell) so `intern*` are `&self` — mirrors the C++
/// `typePool` being a `mutable` field, so `&self` layout queries can instantiate
/// generics on-demand. `get` returns `TypeKey` by Copy and `generic_args_at`/
/// `fn_params_at` return owned `Vec`s, so no borrow is held across an `intern`.
pub struct TypePool {
    keys: RefCell<Vec<TypeKey>>,
    idx: RefCell<HashMap<TypeKey, u32>>,
    generic_args: RefCell<Vec<Vec<TypeIdx>>>,
    generic_args_idx: RefCell<BTreeMap<Vec<TypeIdx>, u32>>,
    fn_params: RefCell<Vec<Vec<TypeIdx>>>,
    fn_params_idx: RefCell<BTreeMap<Vec<TypeIdx>, u32>>,
}

impl Default for TypePool {
    fn default() -> Self {
        TypePool::new()
    }
}

impl TypePool {
    pub fn new() -> TypePool {
        let p = TypePool {
            keys: RefCell::new(Vec::new()),
            idx: RefCell::new(HashMap::new()),
            generic_args: RefCell::new(Vec::new()),
            generic_args_idx: RefCell::new(BTreeMap::new()),
            fn_params: RefCell::new(Vec::new()),
            fn_params_idx: RefCell::new(BTreeMap::new()),
        };
        // Slot 0 = invalid sentinel.
        p.keys
            .borrow_mut()
            .push(TypeKey::new(TypeKind::Invalid, 0, 0));
        // Pre-intern built-ins in the order the `builtin` constants expect.
        p.push_key(TypeKey::new(TypeKind::Void, 0, 0)); // 1
        p.push_key(TypeKey::new(TypeKind::Bool, 0, 0)); // 2
        p.push_key(TypeKey::new(TypeKind::Int, 8, 0)); // 3 u8
        p.push_key(TypeKey::new(TypeKind::Int, 8, 1)); // 4 i8
        p.push_key(TypeKey::new(TypeKind::Int, 16, 0)); // 5 u16
        p.push_key(TypeKey::new(TypeKind::Int, 16, 1)); // 6 i16
        p.push_key(TypeKey::new(TypeKind::Int, 32, 0)); // 7 u32
        p.push_key(TypeKey::new(TypeKind::Int, 32, 1)); // 8 i32
        p.push_key(TypeKey::new(TypeKind::Int, 64, 0)); // 9 u64
        p.push_key(TypeKey::new(TypeKind::Int, 64, 1)); // 10 i64
        p.push_key(TypeKey::new(TypeKind::Float, 32, 0)); // 11 f32
        p.push_key(TypeKey::new(TypeKind::Float, 64, 0)); // 12 f64
        p.push_key(TypeKey::new(TypeKind::Type, 0, 0)); // 13
        p.push_key(TypeKey::new(TypeKind::NoReturn, 0, 0)); // 14
        p
    }

    fn push_key(&self, k: TypeKey) -> TypeIdx {
        let mut keys = self.keys.borrow_mut();
        let id = keys.len() as u32;
        keys.push(k);
        self.idx.borrow_mut().insert(k, id);
        TypeIdx::new(id)
    }

    pub fn intern(&self, k: TypeKey) -> TypeIdx {
        let existing = self.idx.borrow().get(&k).copied();
        if let Some(id) = existing {
            return TypeIdx::new(id);
        }
        self.push_key(k)
    }

    /// Returned by value (Copy) — interning can grow `keys`, so a reference
    /// could dangle (the C++ documents this exact footgun).
    pub fn get(&self, i: TypeIdx) -> TypeKey {
        self.keys.borrow()[i.index()]
    }
    pub fn len(&self) -> usize {
        self.keys.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.borrow().is_empty()
    }

    pub fn intern_int(&self, bits: u16, is_signed: bool) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Int, bits as u32, is_signed as u32))
    }
    pub fn intern_float(&self, bits: u16) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Float, bits as u32, 0))
    }
    pub fn intern_ptr_single(&self, elem: TypeIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::PtrSingle, elem.raw(), 0))
    }
    pub fn intern_ptr_many(&self, elem: TypeIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::PtrMany, elem.raw(), 0))
    }
    pub fn intern_slice(&self, elem: TypeIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Slice, elem.raw(), 0))
    }
    pub fn intern_array(&self, elem: TypeIdx, len: u32) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Array, elem.raw(), len))
    }
    pub fn intern_array_expr(&self, elem: TypeIdx, len_expr: NodeIdx) -> TypeIdx {
        self.intern(TypeKey::new(
            TypeKind::ArrayExpr,
            elem.raw(),
            len_expr.raw(),
        ))
    }
    pub fn intern_struct(&self, name: StringIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Struct, name.raw(), 0))
    }
    pub fn intern_enum(&self, name: StringIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Enum, name.raw(), 0))
    }
    pub fn intern_union(&self, name: StringIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Union, name.raw(), 0))
    }
    pub fn intern_named(&self, name: StringIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Named, name.raw(), 0))
    }
    pub fn intern_module(&self, path: StringIdx) -> TypeIdx {
        self.intern(TypeKey::new(TypeKind::Module, path.raw(), 0))
    }

    pub fn intern_generic_call(&self, name: StringIdx, args: Vec<TypeIdx>) -> TypeIdx {
        let existing = self.generic_args_idx.borrow().get(&args).copied();
        let args_idx = match existing {
            Some(i) => i,
            None => {
                let mut ga = self.generic_args.borrow_mut();
                let i = ga.len() as u32;
                ga.push(args.clone());
                self.generic_args_idx.borrow_mut().insert(args, i);
                i
            }
        };
        self.intern(TypeKey::new(TypeKind::GenericCall, name.raw(), args_idx))
    }

    pub fn generic_args_at(&self, idx: u32) -> Vec<TypeIdx> {
        self.generic_args.borrow()[idx as usize].clone()
    }

    pub fn intern_fn(&self, return_ty: TypeIdx, param_tys: Vec<TypeIdx>) -> TypeIdx {
        let existing = self.fn_params_idx.borrow().get(&param_tys).copied();
        let params_idx = match existing {
            Some(i) => i,
            None => {
                let mut fp = self.fn_params.borrow_mut();
                let i = fp.len() as u32;
                fp.push(param_tys.clone());
                self.fn_params_idx.borrow_mut().insert(param_tys, i);
                i
            }
        };
        self.intern(TypeKey::new(TypeKind::Fn, return_ty.raw(), params_idx))
    }

    pub fn fn_params_at(&self, idx: u32) -> Vec<TypeIdx> {
        self.fn_params.borrow()[idx as usize].clone()
    }
}

