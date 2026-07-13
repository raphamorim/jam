/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam_llvm` — hand-rolled Rust bindings to LLVM (22.1) for the Jam compiler.
//!
//! Layered:
//!
//!   * [`raw`] — the only `unsafe extern "C"` surface: declarations of the
//!     LLVM-C API functions we use, transcribed from the LLVM 22 headers. No
//!     `inkwell`, no `llvm-sys`; the link is wired by `build.rs`.
//!   * the safe wrappers ([`Context`], [`Module`], [`Builder`], [`Type`],
//!     [`Value`], [`Function`], [`BasicBlock`], [`TargetMachine`]) — RAII
//!     owners and `Copy` handles.
//!   * binary128 float support lives in the standalone [`float128`] crate —
//!     re-exported here for API stability — since it is pure Rust and needed
//!     by `jam_syntax` too.
//!
//! Handles ([`Type`]/[`Value`]/`…`) borrow `'ctx` from the [`Context`] that owns
//! the underlying LLVM objects, so they cannot outlive it.

mod raw;

mod ll;
mod target;

pub use float128::{ParsedFloat, parse_decimal_float, quad_to_target_as_double};
pub use ll::{BasicBlock, Builder, Context, Function, Module, Type, Value};
pub use target::{
    Lto, OptLevel, Strip, TargetMachine, append_to_used, default_target_triple, host_cpu_features,
    host_cpu_name, init_all_targets, init_native_asm_parser, init_native_asm_printer,
    init_native_target,
};

/// LLVM calling conventions jam emits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CallConv {
    C,
    Fast,
    Cold,
    /// Microsoft x64 calling convention (Windows MSVC).
    Win64,
}

impl CallConv {
    pub(crate) fn to_raw(self) -> std::os::raw::c_uint {
        match self {
            // Values from LLVMCallConv (C=0, Fast=8, Cold=9, Win64=79).
            CallConv::C => 0,
            CallConv::Fast => 8,
            CallConv::Cold => 9,
            CallConv::Win64 => 79,
        }
    }
}

/// Linkage kinds jam sets on globals/functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Linkage {
    External,
    Internal,
    Private,
}

impl Linkage {
    pub(crate) fn to_raw(self) -> raw::LLVMLinkage {
        match self {
            Linkage::External => raw::LLVMLinkage::External,
            Linkage::Internal => raw::LLVMLinkage::Internal,
            Linkage::Private => raw::LLVMLinkage::Private,
        }
    }
}

/// Integer comparison predicates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntPredicate {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

impl IntPredicate {
    pub(crate) fn to_raw(self) -> raw::LLVMIntPredicate {
        use raw::LLVMIntPredicate as P;
        match self {
            IntPredicate::Eq => P::EQ,
            IntPredicate::Ne => P::NE,
            IntPredicate::Ugt => P::UGT,
            IntPredicate::Uge => P::UGE,
            IntPredicate::Ult => P::ULT,
            IntPredicate::Ule => P::ULE,
            IntPredicate::Sgt => P::SGT,
            IntPredicate::Sge => P::SGE,
            IntPredicate::Slt => P::SLT,
            IntPredicate::Sle => P::SLE,
        }
    }
}

/// Float comparison predicates. `O` = ordered (NaN inputs -> false); `U` =
/// unordered (NaN inputs -> true). Jam emits the ordered variants by default.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RealPredicate {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Uno,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    True,
}

impl RealPredicate {
    pub(crate) fn to_raw(self) -> raw::LLVMRealPredicate {
        use raw::LLVMRealPredicate as P;
        match self {
            RealPredicate::False => P::PredicateFalse,
            RealPredicate::Oeq => P::OEQ,
            RealPredicate::Ogt => P::OGT,
            RealPredicate::Oge => P::OGE,
            RealPredicate::Olt => P::OLT,
            RealPredicate::Ole => P::OLE,
            RealPredicate::One => P::ONE,
            RealPredicate::Ord => P::ORD,
            RealPredicate::Uno => P::UNO,
            RealPredicate::Ueq => P::UEQ,
            RealPredicate::Ugt => P::UGT,
            RealPredicate::Uge => P::UGE,
            RealPredicate::Ult => P::ULT,
            RealPredicate::Ule => P::ULE,
            RealPredicate::Une => P::UNE,
            RealPredicate::True => P::PredicateTrue,
        }
    }
}
