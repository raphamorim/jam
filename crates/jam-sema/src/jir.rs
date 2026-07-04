/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! JIR (Jam IR) — a single typed flat intermediate representation produced by
//! AstGen from the parsed AST and consumed by codegen to emit LLVM IR. The IR
//! is typed from the start: Jam doesn't have comptime-as-values, so a separate
//! untyped lowering pass isn't needed.
//!
//! Pipeline (target):
//!   Source -> Tokens -> AST -> AstGen -> JIR -> Codegen -> LLVM IR
//!
//! Each [`JirInst`] carries a typed result (or `kNoType`/[`TypeIdx::NONE`] for
//! control-flow instructions) plus a source line for diagnostics. Variable-
//! width payloads (call arg lists, struct field values, switch cases, etc.)
//! live in a per-function `extra` pool, just like the AST's `NodeStore`.
//!
//! Ported faithfully from `src/jir.h`. `JirRef` / `JirBlockRef` index newtypes
//! live in [`jam_core::index`]; the per-instruction `a` / `b` slots stay raw
//! `u32` because they are polymorphic — depending on the tag they hold a
//! `JirRef`, a field index, a `StringIdx`, a `JirExtraIdx`, a block ref, or a
//! raw constant (the per-tag contracts below spell out which).

use jam_core::index::TypeIdx;
use jam_core::param_mode::ParamMode;

/// Ref into a function's instruction list. Index `0` is reserved as a sentinel
/// "no value" / invalid ref so a zero-initialized field reads as null.
pub type JirRef = u32;
pub const NO_JIR_REF: JirRef = 0;

/// Ref into a function's block list. Same shape as [`JirRef`].
pub type JirBlockRef = u32;
pub const NO_JIR_BLOCK: JirBlockRef = 0;

/// Ref into a function's extra-data pool.
pub type JirExtraIdx = u32;

/// JIR opcode. The discriminant order mirrors `src/jir.h` exactly.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum JirTag {
    #[default]
    Invalid = 0,

    // Constants
    // Int:   `a` = lo32, `b` = hi32 of u64 value; `ty` = int type.
    // Float: `a` = lo32, `b` = hi32 of f32/f64 bit pattern; `ty` = f32/f64.
    // Bool:  `a` = 0 or 1; `ty` = i1.
    // Str:   `a` = StringIdx; `ty` = []u8.
    Int,
    Float,
    Bool,
    Str,

    // MakeSlice: build a `{ptr,len}` slice value. `a` = ptr ref, `b` = len ref
    // (u64); `ty` = the slice type []T. Backs the `ptr[start..end]` expression.
    MakeSlice,

    // Storage
    // Alloca: `ty` = pointee type. Result type is implicitly ptr-to-`ty`.
    // Load:   `a` = ptr ref; `ty` = loaded type.
    // Store:  `a` = ptr ref, `b` = value ref. No result.
    Alloca,
    Load,
    Store,

    // Integer arithmetic. Binary form: `a` = lhs, `b` = rhs, `ty` = result.
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,

    // Float arithmetic
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    FNeg, // unary: `a` = operand

    // Integer comparison. Binary form; `ty` is always i1.
    ICmpEq,
    ICmpNe,
    ICmpSlt,
    ICmpSle,
    ICmpSgt,
    ICmpSge,
    ICmpUlt,
    ICmpUle,
    ICmpUgt,
    ICmpUge,

    // Float comparison. Ordered predicates (NaN inputs => false). `ty` = i1.
    FCmpOeq,
    FCmpOne,
    FCmpOlt,
    FCmpOle,
    FCmpOgt,
    FCmpOge,

    // Bitwise / shift
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    AShr,
    LShr,

    // Logical (short-circuit handled by control flow)
    LogNot, // unary

    // Type conversions. All take `a` = operand, `ty` = destination type.
    ZExt,
    SExt,
    Trunc,
    SIToFP,
    UIToFP,
    FPToSI,
    FPToUI,
    FPExt,
    FPTrunc,
    BitCast,
    // Pointer <-> integer conversions (raw-address round-trips and surfacing
    // function addresses as integers, paired with `FnRef`).
    PtrToInt,
    IntToPtr,

    // Control flow
    // Br:          `a` = JirBlockRef target. No result.
    // CondBr:      `a` = cond ref; `b` = ExtraIdx -> [thenBlock, elseBlock].
    // Switch:      `a` = scrut ref; `b` = ExtraIdx ->
    //                  [defaultBlock, caseCount,
    //                   case0_lo, case0_hi, case0_signed, case0_block, ...]
    // Ret:         `a` = value ref (or NO_JIR_REF for void).
    // Unreachable: no operands.
    Br,
    CondBr,
    Switch,
    Ret,
    Unreachable,

    // Function call
    // Call: `a` = StringIdx (callee qualified name);
    //       `b` = ExtraIdx -> [argCount, arg0, arg1, ...]; `ty` = return type.
    Call,
    // CallIndirect: `a` = JirRef of a fn-typed value (the function pointer);
    //               `b` = ExtraIdx -> [argCount, arg0, ...]; `ty` = return.
    CallIndirect,

    // Function reference (item-as-value). `a` = StringIdx (LLVM symbol name);
    // `ty` = u64.
    FnRef,

    // Function parameter access
    // Param:   `a` = parameter index; `ty` = param type.
    Param,
    // SretArg: evaluates to the function's hidden sret pointer (param 0 when
    // `jirReturnIsSret` holds). `ty` is the sret pointee (the return type).
    SretArg,

    // Aggregates
    // StructLit:    `b` = ExtraIdx -> [fieldCount, field0_val, ...]; `ty` = struct.
    // FieldAccess:  `a` = base ref, `b` = field index; `ty` = field type.
    // ExtractValue: `a` = aggregate ref, `b` = field index; `ty` = field type.
    // ArrayLit:     `b` = ExtraIdx -> [count, elem0, ...]; `ty` = array type.
    StructLit,
    FieldAccess,
    ExtractValue,
    ArrayLit,

    // MemSet: `a` = dest ptr ref, `b` = length in BYTES (raw constant, not a
    // JirRef), `flags` = fill byte (0..255). Backs `[fill; N]` byte fills.
    MemSet,
    Index,
    // FieldAddr: `a` = base pointer ref; `b` = field index; `ty` = ptr-to-field.
    FieldAddr,
    // IndexAddr: `a` = base pointer ref; `b` = index value ref; `ty` = ptr-to-elem.
    IndexAddr,

    // Address-of / dereference
    AddrOf, // `a` = lvalue ref
    Deref,  // `a` = ptr ref

    // Pattern binding payload extraction.
    EnumPayload, // `a` = enum value ref, `b` = field index

    // Drop: explicit destructor call for a tracked binding. `a` = the binding's
    // alloca JirRef; `b` = StringIdx of the LLVM symbol to call. ty is void.
    DropBinding,

    // Error recovery: a typed placeholder (lowers to LLVM `undef`). Programs
    // that produce any Poison must not reach codegen.
    Poison,
}

impl JirTag {
    /// Stable opcode name (matches the `--emit-jir` dump contract and the C++
    /// `jir_verify` / `main.cpp` tag-name tables).
    pub fn name(self) -> &'static str {
        match self {
            JirTag::Invalid => "Invalid",
            JirTag::Int => "Int",
            JirTag::Float => "Float",
            JirTag::Bool => "Bool",
            JirTag::Str => "Str",
            JirTag::MakeSlice => "MakeSlice",
            JirTag::Alloca => "Alloca",
            JirTag::Load => "Load",
            JirTag::Store => "Store",
            JirTag::Add => "Add",
            JirTag::Sub => "Sub",
            JirTag::Mul => "Mul",
            JirTag::SDiv => "SDiv",
            JirTag::UDiv => "UDiv",
            JirTag::SRem => "SRem",
            JirTag::URem => "URem",
            JirTag::FAdd => "FAdd",
            JirTag::FSub => "FSub",
            JirTag::FMul => "FMul",
            JirTag::FDiv => "FDiv",
            JirTag::FRem => "FRem",
            JirTag::FNeg => "FNeg",
            JirTag::ICmpEq => "ICmpEq",
            JirTag::ICmpNe => "ICmpNe",
            JirTag::ICmpSlt => "ICmpSlt",
            JirTag::ICmpSle => "ICmpSle",
            JirTag::ICmpSgt => "ICmpSgt",
            JirTag::ICmpSge => "ICmpSge",
            JirTag::ICmpUlt => "ICmpUlt",
            JirTag::ICmpUle => "ICmpUle",
            JirTag::ICmpUgt => "ICmpUgt",
            JirTag::ICmpUge => "ICmpUge",
            JirTag::FCmpOeq => "FCmpOeq",
            JirTag::FCmpOne => "FCmpOne",
            JirTag::FCmpOlt => "FCmpOlt",
            JirTag::FCmpOle => "FCmpOle",
            JirTag::FCmpOgt => "FCmpOgt",
            JirTag::FCmpOge => "FCmpOge",
            JirTag::BitAnd => "BitAnd",
            JirTag::BitOr => "BitOr",
            JirTag::BitXor => "BitXor",
            JirTag::BitNot => "BitNot",
            JirTag::Shl => "Shl",
            JirTag::AShr => "AShr",
            JirTag::LShr => "LShr",
            JirTag::LogNot => "LogNot",
            JirTag::ZExt => "ZExt",
            JirTag::SExt => "SExt",
            JirTag::Trunc => "Trunc",
            JirTag::SIToFP => "SIToFP",
            JirTag::UIToFP => "UIToFP",
            JirTag::FPToSI => "FPToSI",
            JirTag::FPToUI => "FPToUI",
            JirTag::FPExt => "FPExt",
            JirTag::FPTrunc => "FPTrunc",
            JirTag::BitCast => "BitCast",
            JirTag::PtrToInt => "PtrToInt",
            JirTag::IntToPtr => "IntToPtr",
            JirTag::Br => "Br",
            JirTag::CondBr => "CondBr",
            JirTag::Switch => "Switch",
            JirTag::Ret => "Ret",
            JirTag::Unreachable => "Unreachable",
            JirTag::Call => "Call",
            JirTag::CallIndirect => "CallIndirect",
            JirTag::FnRef => "FnRef",
            JirTag::Param => "Param",
            JirTag::SretArg => "SretArg",
            JirTag::StructLit => "StructLit",
            JirTag::FieldAccess => "FieldAccess",
            JirTag::ExtractValue => "ExtractValue",
            JirTag::ArrayLit => "ArrayLit",
            JirTag::MemSet => "MemSet",
            JirTag::Index => "Index",
            JirTag::FieldAddr => "FieldAddr",
            JirTag::IndexAddr => "IndexAddr",
            JirTag::AddrOf => "AddrOf",
            JirTag::Deref => "Deref",
            JirTag::EnumPayload => "EnumPayload",
            JirTag::DropBinding => "DropBinding",
            JirTag::Poison => "Poison",
        }
    }

    /// True for the block-terminating opcodes.
    pub fn is_terminator(self) -> bool {
        matches!(
            self,
            JirTag::Br | JirTag::CondBr | JirTag::Switch | JirTag::Ret | JirTag::Unreachable
        )
    }
}

