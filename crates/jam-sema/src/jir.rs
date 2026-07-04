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

/// A single JIR instruction. Trivially copyable so dense instruction arrays
/// stay cache-friendly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JirInst {
    pub tag: JirTag,
    /// Tag-specific flags.
    pub flags: u16,
    /// For diagnostics: `file:line:`.
    pub src_line: u32,
    pub a: JirRef,
    pub b: JirRef,
    /// Result type (`TypeIdx::NONE` for void / control).
    pub ty: TypeIdx,
}

impl Default for JirInst {
    fn default() -> Self {
        JirInst {
            tag: JirTag::Invalid,
            flags: 0,
            src_line: 0,
            a: NO_JIR_REF,
            b: NO_JIR_REF,
            ty: TypeIdx::NONE,
        }
    }
}

/// A basic block within a [`JirFunction`]: a list of [`JirRef`] instruction
/// indices into the function's `insts` array.
#[derive(Clone, Debug, Default)]
pub struct JirBlock {
    /// Diagnostic label.
    pub name: String,
    pub insts: Vec<JirRef>,
}

/// A single function lowered to JIR. Self-contained: its `insts` array holds
/// every instruction in the body (across all blocks), and the blocks reference
/// instructions by index. Variable-width payloads live in `extra`.
#[derive(Clone, Debug)]
pub struct JirFunction {
    pub name: String,
    pub insts: Vec<JirInst>,
    pub extra: Vec<u32>,
    pub blocks: Vec<JirBlock>,
    pub param_types: Vec<TypeIdx>,
    /// Parallel to `param_types`: the source-level parameter mode. The backend
    /// decides the ABI from this rather than baking the by-pointer decision in.
    pub param_modes: Vec<ParamMode>,
    pub return_type: TypeIdx,
    /// Owning module's identity (`FunctionAST::module_path`). The JIR-consuming
    /// passes push this as the body module so bare body-level type references
    /// resolve to their owning module.
    pub module_path: String,
    pub is_extern: bool,
    pub is_export: bool,
    pub is_pub: bool,
    pub is_test: bool,
    pub is_var_args: bool,
}

impl Default for JirFunction {
    fn default() -> Self {
        JirFunction::new()
    }
}

impl JirFunction {
    /// A fresh function with the slot-0 sentinels installed: `insts[0]` is the
    /// invalid instruction and `blocks[0]` is `<sentinel>`, so `NO_JIR_REF` /
    /// `NO_JIR_BLOCK` (both `0`) read as null.
    pub fn new() -> Self {
        JirFunction {
            name: String::new(),
            insts: vec![JirInst::default()],
            extra: Vec::new(),
            blocks: vec![JirBlock {
                name: "<sentinel>".to_string(),
                insts: Vec::new(),
            }],
            param_types: Vec::new(),
            param_modes: Vec::new(),
            return_type: TypeIdx::NONE,
            module_path: String::new(),
            is_extern: false,
            is_export: false,
            is_pub: false,
            is_test: false,
            is_var_args: false,
        }
    }

    /// Append an instruction; returns its ref (1-based, 0 reserved).
    pub fn push_inst(&mut self, inst: JirInst) -> JirRef {
        let r = self.insts.len() as JirRef;
        self.insts.push(inst);
        r
    }

    /// Append a basic block; returns its ref.
    pub fn push_block(&mut self, label: impl Into<String>) -> JirBlockRef {
        let r = self.blocks.len() as JirBlockRef;
        self.blocks.push(JirBlock {
            name: label.into(),
            insts: Vec::new(),
        });
        r
    }

    /// Append `data` to the extra pool; returns the start index.
    pub fn push_extra(&mut self, data: &[u32]) -> JirExtraIdx {
        let start = self.extra.len() as JirExtraIdx;
        self.extra.extend_from_slice(data);
        start
    }

    /// Reserve `len` zero-filled u32 slots; returns the start index.
    pub fn reserve_extra(&mut self, len: usize) -> JirExtraIdx {
        let start = self.extra.len() as JirExtraIdx;
        self.extra.resize(self.extra.len() + len, 0);
        start
    }

    pub fn get_inst(&self, r: JirRef) -> &JirInst {
        &self.insts[r as usize]
    }
    pub fn get_inst_mut(&mut self, r: JirRef) -> &mut JirInst {
        &mut self.insts[r as usize]
    }
    pub fn get_block(&self, r: JirBlockRef) -> &JirBlock {
        &self.blocks[r as usize]
    }
    pub fn get_block_mut(&mut self, r: JirBlockRef) -> &mut JirBlock {
        &mut self.blocks[r as usize]
    }
    pub fn get_extra(&self, i: JirExtraIdx) -> u32 {
        self.extra[i as usize]
    }
    pub fn set_extra(&mut self, i: JirExtraIdx, v: u32) {
        self.extra[i as usize] = v;
    }
}

/// A whole-program JIR module: a collection of [`JirFunction`]s.
#[derive(Clone, Debug, Default)]
pub struct JirModule {
    pub functions: Vec<JirFunction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The data-structure invariants mirror the C++ `test_jir_skeleton`
    // coverage: slot-0 sentinels, 1-based push refs, and extra-pool growth.
    // (End-to-end JIR validation waits for the full AstGen path — there is no
    // intermediate JIR-dump oracle.)

    #[test]
    fn sentinels_occupy_slot_zero() {
        let f = JirFunction::new();
        assert_eq!(f.insts.len(), 1);
        assert_eq!(f.insts[0].tag, JirTag::Invalid);
        assert_eq!(f.blocks.len(), 1);
        assert_eq!(f.blocks[0].name, "<sentinel>");
        assert_eq!(NO_JIR_REF, 0);
        assert_eq!(NO_JIR_BLOCK, 0);
    }

    #[test]
    fn push_inst_returns_one_based_refs() {
        let mut f = JirFunction::new();
        let r1 = f.push_inst(JirInst {
            tag: JirTag::Bool,
            a: 1,
            ty: TypeIdx::NONE,
            ..Default::default()
        });
        let r2 = f.push_inst(JirInst {
            tag: JirTag::Ret,
            a: r1,
            ..Default::default()
        });
        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_ne!(r1, NO_JIR_REF);
        assert_eq!(f.get_inst(r1).tag, JirTag::Bool);
        assert_eq!(f.get_inst(r2).a, r1);
    }

    #[test]
    fn push_block_returns_one_based_refs() {
        let mut f = JirFunction::new();
        let entry = f.push_block("entry");
        let exit = f.push_block("exit");
        assert_eq!(entry, 1);
        assert_eq!(exit, 2);
        assert_eq!(f.get_block(entry).name, "entry");
    }

    #[test]
    fn extra_pool_push_and_reserve() {
        let mut f = JirFunction::new();
        let start = f.push_extra(&[3, 10, 20, 30]);
        assert_eq!(start, 0);
        assert_eq!(f.get_extra(start), 3);
        assert_eq!(f.get_extra(start + 3), 30);
        let r = f.reserve_extra(2);
        assert_eq!(r, 4);
        assert_eq!(f.get_extra(r), 0);
        f.set_extra(r, 99);
        assert_eq!(f.get_extra(r), 99);
        assert_eq!(f.extra.len(), 6);
    }

    #[test]
    fn jir_tag_discriminants_match_cpp_order() {
        // A handful of anchor points pinned against `src/jir.h` ordering.
        assert_eq!(JirTag::Invalid as u8, 0);
        assert_eq!(JirTag::Int as u8, 1);
        assert_eq!(JirTag::MakeSlice as u8, 5);
        assert_eq!(JirTag::Alloca as u8, 6);
        assert_eq!(JirTag::Add as u8, 9);
        assert_eq!(JirTag::IntToPtr as u8, 57);
        assert_eq!(JirTag::Call as u8, 63);
        assert_eq!(JirTag::DropBinding as u8, 79);
        assert_eq!(JirTag::Poison as u8, 80);
    }

    #[test]
    fn jir_inst_default_is_invalid_null() {
        let i = JirInst::default();
        assert_eq!(i.tag, JirTag::Invalid);
        assert_eq!(i.a, NO_JIR_REF);
        assert_eq!(i.b, NO_JIR_REF);
        assert_eq!(i.ty, TypeIdx::NONE);
    }
}
