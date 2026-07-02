/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Hand-rolled `extern "C"` bindings to the LLVM-C API (LLVM 22.1).
//!
//! This is the *only* unsafe FFI surface in the crate. Every declaration here
//! is transcribed verbatim from the system headers under
//! `$(llvm-config --includedir)/llvm-c/` — `Core.h`, `Analysis.h`, `Target.h`,
//! `TargetMachine.h`, `Transforms/PassBuilder.h`, `BitWriter.h`, `Error.h`.
//!
//! FFI signature mistakes are *silent* (the dynamic linker resolves the symbol
//! regardless of the declared prototype; a wrong arg count/type corrupts the
//! stack at call time, not at link time). So these signatures are
//! intentionally exact, and the safe wrappers in the rest of the crate are the
//! only thing the compiler should ever call. Do not add a declaration here
//! without checking it against the header it claims to come from.
#![allow(non_snake_case, non_camel_case_types, dead_code)]
#![allow(clippy::upper_case_acronyms)]

use std::os::raw::{c_char, c_int, c_uint, c_ulonglong};

pub type LLVMBool = c_int;
pub type size_t = usize;

// Opaque object types — we only ever hold pointers to these.
pub enum LLVMOpaqueContext {}
pub enum LLVMOpaqueModule {}
pub enum LLVMOpaqueType {}
pub enum LLVMOpaqueValue {}
pub enum LLVMOpaqueBasicBlock {}
pub enum LLVMOpaqueBuilder {}
pub enum LLVMOpaqueAttributeRef {}
pub enum LLVMOpaqueMetadata {}
pub enum LLVMTarget {}
pub enum LLVMOpaqueTargetMachine {}
pub enum LLVMOpaqueTargetData {}
pub enum LLVMOpaquePassBuilderOptions {}
pub enum LLVMOpaqueError {}

pub type LLVMContextRef = *mut LLVMOpaqueContext;
pub type LLVMModuleRef = *mut LLVMOpaqueModule;
pub type LLVMTypeRef = *mut LLVMOpaqueType;
pub type LLVMValueRef = *mut LLVMOpaqueValue;
pub type LLVMBasicBlockRef = *mut LLVMOpaqueBasicBlock;
pub type LLVMBuilderRef = *mut LLVMOpaqueBuilder;
pub type LLVMAttributeRef = *mut LLVMOpaqueAttributeRef;
pub type LLVMMetadataRef = *mut LLVMOpaqueMetadata;
pub type LLVMTargetRef = *mut LLVMTarget;
pub type LLVMTargetMachineRef = *mut LLVMOpaqueTargetMachine;
pub type LLVMTargetDataRef = *mut LLVMOpaqueTargetData;
pub type LLVMPassBuilderOptionsRef = *mut LLVMOpaquePassBuilderOptions;
pub type LLVMErrorRef = *mut LLVMOpaqueError;
pub type LLVMAttributeIndex = c_uint;

// ---- enums (values verified against the LLVM 22 headers) -------------------

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LLVMTypeKind {
    Void = 0,
    Half = 1,
    Float = 2,
    Double = 3,
    X86_FP80 = 4,
    FP128 = 5,
    PPC_FP128 = 6,
    Label = 7,
    Integer = 8,
    Function = 9,
    Struct = 10,
    Array = 11,
    Pointer = 12,
    Vector = 13,
    Metadata = 14,
    Token = 16,
    ScalableVector = 17,
    BFloat = 18,
    X86_AMX = 19,
    TargetExt = 20,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMLinkage {
    External = 0,
    AvailableExternally = 1,
    LinkOnceAny = 2,
    LinkOnceODR = 3,
    LinkOnceODRAutoHide = 4,
    WeakAny = 5,
    WeakODR = 6,
    Appending = 7,
    Internal = 8,
    Private = 9,
    DLLImport = 10,
    DLLExport = 11,
    ExternalWeak = 12,
    Ghost = 13,
    Common = 14,
    LinkerPrivate = 15,
    LinkerPrivateWeak = 16,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMUnnamedAddr {
    No = 0,
    Local = 1,
    Global = 2,
}

// ICMP predicate values (32..=41) match Jam's JamIntPredicate exactly.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMIntPredicate {
    EQ = 32,
    NE = 33,
    UGT = 34,
    UGE = 35,
    ULT = 36,
    ULE = 37,
    SGT = 38,
    SGE = 39,
    SLT = 40,
    SLE = 41,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMRealPredicate {
    PredicateFalse = 0,
    OEQ = 1,
    OGT = 2,
    OGE = 3,
    OLT = 4,
    OLE = 5,
    ONE = 6,
    ORD = 7,
    UNO = 8,
    UEQ = 9,
    UGT = 10,
    UGE = 11,
    ULT = 12,
    ULE = 13,
    UNE = 14,
    PredicateTrue = 15,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMVerifierFailureAction {
    AbortProcess = 0,
    PrintMessage = 1,
    ReturnStatus = 2,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMCodeGenOptLevel {
    None = 0,
    Less = 1,
    Default = 2,
    Aggressive = 3,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMRelocMode {
    Default = 0,
    Static = 1,
    PIC = 2,
    DynamicNoPic = 3,
    ROPI = 4,
    RWPI = 5,
    ROPI_RWPI = 6,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMCodeModel {
    Default = 0,
    JITDefault = 1,
    Tiny = 2,
    Small = 3,
    Kernel = 4,
    Medium = 5,
    Large = 6,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMCodeGenFileType {
    AssemblyFile = 0,
    ObjectFile = 1,
}

// The C enum stops at AppendUnique — there is no `Max`/`Min` here (those exist
// only on the C++ `Module::ModFlagBehavior`). PIC/PIE level flags therefore use
// `Override`; for jam's single, never-IR-linked module the behavior value has
// no codegen effect.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum LLVMModuleFlagBehavior {
    Error = 0,
    Warning = 1,
    Require = 2,
    Override = 3,
    Append = 4,
    AppendUnique = 5,
}

unsafe extern "C" {
    // ---- per-target initialization (real exported symbols; the
    // `LLVMInitializeNative*`/`All*` convenience fns are `static inline` in the
    // header and are NOT linkable, so we call the per-target entry points). ----
    pub fn LLVMInitializeAArch64TargetInfo();
    pub fn LLVMInitializeAArch64Target();
    pub fn LLVMInitializeAArch64TargetMC();
    pub fn LLVMInitializeAArch64AsmParser();
    pub fn LLVMInitializeAArch64AsmPrinter();
    pub fn LLVMInitializeX86TargetInfo();
    pub fn LLVMInitializeX86Target();
    pub fn LLVMInitializeX86TargetMC();
    pub fn LLVMInitializeX86AsmParser();
    pub fn LLVMInitializeX86AsmPrinter();

