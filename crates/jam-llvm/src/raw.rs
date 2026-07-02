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

    // ---- context / module / builder ----
    pub fn LLVMContextCreate() -> LLVMContextRef;
    pub fn LLVMContextDispose(C: LLVMContextRef);
    pub fn LLVMContextSetDiscardValueNames(C: LLVMContextRef, Discard: LLVMBool);
    pub fn LLVMModuleCreateWithNameInContext(
        ModuleID: *const c_char,
        C: LLVMContextRef,
    ) -> LLVMModuleRef;
    pub fn LLVMDisposeModule(M: LLVMModuleRef);
    pub fn LLVMSetTarget(M: LLVMModuleRef, Triple: *const c_char);
    pub fn LLVMSetModuleDataLayout(M: LLVMModuleRef, DL: LLVMTargetDataRef);
    pub fn LLVMGetModuleContext(M: LLVMModuleRef) -> LLVMContextRef;
    pub fn LLVMGetNamedFunction(M: LLVMModuleRef, Name: *const c_char) -> LLVMValueRef;
    pub fn LLVMPrintModuleToString(M: LLVMModuleRef) -> *mut c_char;
    pub fn LLVMDisposeMessage(Message: *mut c_char);
    pub fn LLVMAddModuleFlag(
        M: LLVMModuleRef,
        Behavior: LLVMModuleFlagBehavior,
        Key: *const c_char,
        KeyLen: size_t,
        Val: LLVMMetadataRef,
    );
    pub fn LLVMValueAsMetadata(Val: LLVMValueRef) -> LLVMMetadataRef;
    pub fn LLVMCreateBuilderInContext(C: LLVMContextRef) -> LLVMBuilderRef;
    pub fn LLVMDisposeBuilder(Builder: LLVMBuilderRef);
    pub fn LLVMPositionBuilderAtEnd(Builder: LLVMBuilderRef, Block: LLVMBasicBlockRef);
    pub fn LLVMPositionBuilder(
        Builder: LLVMBuilderRef,
        Block: LLVMBasicBlockRef,
        Instr: LLVMValueRef,
    );
    pub fn LLVMPositionBuilderBefore(Builder: LLVMBuilderRef, Instr: LLVMValueRef);
    pub fn LLVMGetInsertBlock(Builder: LLVMBuilderRef) -> LLVMBasicBlockRef;

    // ---- types ----
    pub fn LLVMInt1TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt8TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt16TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt32TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMInt64TypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMFloatTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMDoubleTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMVoidTypeInContext(C: LLVMContextRef) -> LLVMTypeRef;
    pub fn LLVMPointerTypeInContext(C: LLVMContextRef, AddressSpace: c_uint) -> LLVMTypeRef;
    pub fn LLVMStructTypeInContext(
        C: LLVMContextRef,
        ElementTypes: *mut LLVMTypeRef,
        ElementCount: c_uint,
        Packed: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMStructCreateNamed(C: LLVMContextRef, Name: *const c_char) -> LLVMTypeRef;
    pub fn LLVMStructSetBody(
        StructTy: LLVMTypeRef,
        ElementTypes: *mut LLVMTypeRef,
        ElementCount: c_uint,
        Packed: LLVMBool,
    );
    pub fn LLVMFunctionType(
        ReturnType: LLVMTypeRef,
        ParamTypes: *mut LLVMTypeRef,
        ParamCount: c_uint,
        IsVarArg: LLVMBool,
    ) -> LLVMTypeRef;
    pub fn LLVMArrayType2(ElementType: LLVMTypeRef, ElementCount: u64) -> LLVMTypeRef;
    pub fn LLVMGetTypeKind(Ty: LLVMTypeRef) -> LLVMTypeKind;
    pub fn LLVMGetIntTypeWidth(IntegerTy: LLVMTypeRef) -> c_uint;
    pub fn LLVMGetElementType(Ty: LLVMTypeRef) -> LLVMTypeRef;
    pub fn LLVMGetTypeContext(Ty: LLVMTypeRef) -> LLVMContextRef;
    pub fn LLVMIsFunctionVarArg(FunctionTy: LLVMTypeRef) -> LLVMBool;
    pub fn LLVMGlobalGetValueType(Global: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGetReturnType(FunctionTy: LLVMTypeRef) -> LLVMTypeRef;
    pub fn LLVMTypeOf(Val: LLVMValueRef) -> LLVMTypeRef;
    pub fn LLVMGetAllocatedType(Alloca: LLVMValueRef) -> LLVMTypeRef;

    // ---- constants / globals ----
    pub fn LLVMConstInt(IntTy: LLVMTypeRef, N: c_ulonglong, SignExtend: LLVMBool) -> LLVMValueRef;
    pub fn LLVMConstReal(RealTy: LLVMTypeRef, N: f64) -> LLVMValueRef;
    pub fn LLVMConstNull(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMConstStringInContext(
        C: LLVMContextRef,
        Str: *const c_char,
        Length: c_uint,
        DontNullTerminate: LLVMBool,
    ) -> LLVMValueRef;
    pub fn LLVMConstArray2(
        ElementTy: LLVMTypeRef,
        ConstantVals: *mut LLVMValueRef,
        Length: u64,
    ) -> LLVMValueRef;
    pub fn LLVMGetUndef(Ty: LLVMTypeRef) -> LLVMValueRef;
    pub fn LLVMAddGlobal(M: LLVMModuleRef, Ty: LLVMTypeRef, Name: *const c_char) -> LLVMValueRef;
    pub fn LLVMSetInitializer(GlobalVar: LLVMValueRef, ConstantVal: LLVMValueRef);
    pub fn LLVMSetGlobalConstant(GlobalVar: LLVMValueRef, IsConstant: LLVMBool);
    pub fn LLVMSetSection(Global: LLVMValueRef, Section: *const c_char);
    pub fn LLVMSetLinkage(Global: LLVMValueRef, Linkage: LLVMLinkage);
    pub fn LLVMSetUnnamedAddress(Global: LLVMValueRef, UnnamedAddr: LLVMUnnamedAddr);
    pub fn LLVMGetNamedGlobal(M: LLVMModuleRef, Name: *const c_char) -> LLVMValueRef;
    pub fn LLVMGetInitializer(GlobalVar: LLVMValueRef) -> LLVMValueRef;

    // ---- functions / attributes ----
    pub fn LLVMAddFunction(
        M: LLVMModuleRef,
        Name: *const c_char,
        FunctionTy: LLVMTypeRef,
    ) -> LLVMValueRef;
    pub fn LLVMSetFunctionCallConv(Fn: LLVMValueRef, CC: c_uint);
    pub fn LLVMCountParams(Fn: LLVMValueRef) -> c_uint;
    pub fn LLVMGetParam(Fn: LLVMValueRef, Index: c_uint) -> LLVMValueRef;
    pub fn LLVMSetValueName2(Val: LLVMValueRef, Name: *const c_char, NameLen: size_t);
    pub fn LLVMGetValueName2(Val: LLVMValueRef, Length: *mut size_t) -> *const c_char;
    pub fn LLVMGetEnumAttributeKindForName(Name: *const c_char, SLen: size_t) -> c_uint;
    pub fn LLVMCreateEnumAttribute(C: LLVMContextRef, KindID: c_uint, Val: u64)
    -> LLVMAttributeRef;
    pub fn LLVMCreateTypeAttribute(
        C: LLVMContextRef,
        KindID: c_uint,
        type_ref: LLVMTypeRef,
    ) -> LLVMAttributeRef;
    pub fn LLVMCreateStringAttribute(
        C: LLVMContextRef,
        K: *const c_char,
        KLength: c_uint,
        V: *const c_char,
        VLength: c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMAddAttributeAtIndex(F: LLVMValueRef, Idx: LLVMAttributeIndex, A: LLVMAttributeRef);
    pub fn LLVMGetEnumAttributeAtIndex(
        F: LLVMValueRef,
        Idx: LLVMAttributeIndex,
        KindID: c_uint,
    ) -> LLVMAttributeRef;
    pub fn LLVMGetTypeAttributeValue(A: LLVMAttributeRef) -> LLVMTypeRef;

    // ---- basic blocks / iteration / misc ----
    pub fn LLVMCreateBasicBlockInContext(
        C: LLVMContextRef,
        Name: *const c_char,
    ) -> LLVMBasicBlockRef;
    pub fn LLVMAppendBasicBlockInContext(
        C: LLVMContextRef,
        Fn: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMBasicBlockRef;
    pub fn LLVMGetBasicBlockParent(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetBasicBlockTerminator(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetEntryBasicBlock(Fn: LLVMValueRef) -> LLVMBasicBlockRef;
    pub fn LLVMGetFirstInstruction(BB: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMGetFirstFunction(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetNextFunction(Fn: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetFirstGlobal(M: LLVMModuleRef) -> LLVMValueRef;
    pub fn LLVMGetNextGlobal(GlobalVar: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMGetNumOperands(Val: LLVMValueRef) -> c_int;
    pub fn LLVMGetOperand(Val: LLVMValueRef, Index: c_uint) -> LLVMValueRef;
    pub fn LLVMSetAlignment(V: LLVMValueRef, Bytes: c_uint);
    pub fn LLVMIsDeclaration(Global: LLVMValueRef) -> LLVMBool;
    pub fn LLVMDeleteGlobal(GlobalVar: LLVMValueRef);

    // ---- builder: arithmetic / bitwise (lhs, rhs, name) ----
    pub fn LLVMBuildAdd(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSub(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMul(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildUDiv(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSDiv(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildURem(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSRem(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAnd(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildOr(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildXor(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildShl(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildLShr(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildAShr(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFAdd(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFSub(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFMul(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFDiv(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFRem(
        B: LLVMBuilderRef,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFNeg(B: LLVMBuilderRef, V: LLVMValueRef, Name: *const c_char) -> LLVMValueRef;

    pub fn LLVMBuildNot(B: LLVMBuilderRef, V: LLVMValueRef, Name: *const c_char) -> LLVMValueRef;

    pub fn LLVMBuildTrunc(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;

    pub fn LLVMBuildICmp(
        B: LLVMBuilderRef,
        Op: LLVMIntPredicate,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFCmp(
        B: LLVMBuilderRef,
        Op: LLVMRealPredicate,
        LHS: LLVMValueRef,
        RHS: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;

    // ---- builder: casts (val, destTy, name) ----
    pub fn LLVMBuildZExt(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSExt(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildBitCast(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildPtrToInt(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIntToPtr(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSIToFP(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildUIToFP(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPCast(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPToSI(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildFPToUI(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildIntCast2(
        B: LLVMBuilderRef,
        Val: LLVMValueRef,
        DestTy: LLVMTypeRef,
        IsSigned: LLVMBool,
        Name: *const c_char,
    ) -> LLVMValueRef;

    // ---- builder: memory ----
    pub fn LLVMBuildAlloca(B: LLVMBuilderRef, Ty: LLVMTypeRef, Name: *const c_char)
    -> LLVMValueRef;
    pub fn LLVMBuildLoad2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        PointerVal: LLVMValueRef,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildStore(B: LLVMBuilderRef, Val: LLVMValueRef, Ptr: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildMemCpy(
        B: LLVMBuilderRef,
        Dst: LLVMValueRef,
        DstAlign: c_uint,
        Src: LLVMValueRef,
        SrcAlign: c_uint,
        Size: LLVMValueRef,
    ) -> LLVMValueRef;
    pub fn LLVMBuildMemSet(
        B: LLVMBuilderRef,
        Ptr: LLVMValueRef,
        Val: LLVMValueRef,
        Len: LLVMValueRef,
        Align: c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMBuildInBoundsGEP2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Pointer: LLVMValueRef,
        Indices: *mut LLVMValueRef,
        NumIndices: c_uint,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildStructGEP2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Pointer: LLVMValueRef,
        Idx: c_uint,
        Name: *const c_char,
    ) -> LLVMValueRef;

    // ---- builder: control flow / calls / aggregates ----
    pub fn LLVMBuildBr(B: LLVMBuilderRef, Dest: LLVMBasicBlockRef) -> LLVMValueRef;
    pub fn LLVMBuildCondBr(
        B: LLVMBuilderRef,
        If: LLVMValueRef,
        Then: LLVMBasicBlockRef,
        Else: LLVMBasicBlockRef,
    ) -> LLVMValueRef;
    pub fn LLVMBuildSwitch(
        B: LLVMBuilderRef,
        V: LLVMValueRef,
        Else: LLVMBasicBlockRef,
        NumCases: c_uint,
    ) -> LLVMValueRef;
    pub fn LLVMAddCase(Switch: LLVMValueRef, OnVal: LLVMValueRef, Dest: LLVMBasicBlockRef);
    pub fn LLVMBuildRet(B: LLVMBuilderRef, V: LLVMValueRef) -> LLVMValueRef;
    pub fn LLVMBuildRetVoid(B: LLVMBuilderRef) -> LLVMValueRef;
    pub fn LLVMBuildUnreachable(B: LLVMBuilderRef) -> LLVMValueRef;
    pub fn LLVMBuildCall2(
        B: LLVMBuilderRef,
        Ty: LLVMTypeRef,
        Fn: LLVMValueRef,
        Args: *mut LLVMValueRef,
        NumArgs: c_uint,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildPhi(B: LLVMBuilderRef, Ty: LLVMTypeRef, Name: *const c_char) -> LLVMValueRef;
    pub fn LLVMAddIncoming(
        PhiNode: LLVMValueRef,
        IncomingValues: *mut LLVMValueRef,
        IncomingBlocks: *mut LLVMBasicBlockRef,
        Count: c_uint,
    );
    pub fn LLVMBuildGlobalStringPtr(
        B: LLVMBuilderRef,
        Str: *const c_char,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildInsertValue(
        B: LLVMBuilderRef,
        AggVal: LLVMValueRef,
        EltVal: LLVMValueRef,
        Index: c_uint,
        Name: *const c_char,
    ) -> LLVMValueRef;
    pub fn LLVMBuildExtractValue(
        B: LLVMBuilderRef,
        AggVal: LLVMValueRef,
        Index: c_uint,
        Name: *const c_char,
    ) -> LLVMValueRef;

    // ---- verifier (Analysis.h) ----
    pub fn LLVMVerifyFunction(Fn: LLVMValueRef, Action: LLVMVerifierFailureAction) -> LLVMBool;

    // ---- target machine (TargetMachine.h) ----
    pub fn LLVMGetTargetFromTriple(
        Triple: *const c_char,
        T: *mut LLVMTargetRef,
        ErrorMessage: *mut *mut c_char,
    ) -> LLVMBool;
    pub fn LLVMCreateTargetMachine(
        T: LLVMTargetRef,
        Triple: *const c_char,
        CPU: *const c_char,
        Features: *const c_char,
        Level: LLVMCodeGenOptLevel,
        Reloc: LLVMRelocMode,
        CodeModel: LLVMCodeModel,
    ) -> LLVMTargetMachineRef;
    pub fn LLVMDisposeTargetMachine(T: LLVMTargetMachineRef);
    pub fn LLVMCreateTargetDataLayout(T: LLVMTargetMachineRef) -> LLVMTargetDataRef;
    pub fn LLVMTargetMachineEmitToFile(
        T: LLVMTargetMachineRef,
        M: LLVMModuleRef,
        Filename: *const c_char,
        codegen: LLVMCodeGenFileType,
        ErrorMessage: *mut *mut c_char,
    ) -> LLVMBool;
    pub fn LLVMGetDefaultTargetTriple() -> *mut c_char;
    pub fn LLVMGetHostCPUName() -> *mut c_char;
    pub fn LLVMGetHostCPUFeatures() -> *mut c_char;

    // ---- new-PM pass pipeline (Transforms/PassBuilder.h) ----
    pub fn LLVMRunPasses(
        M: LLVMModuleRef,
        Passes: *const c_char,
        TM: LLVMTargetMachineRef,
        Options: LLVMPassBuilderOptionsRef,
    ) -> LLVMErrorRef;
    pub fn LLVMCreatePassBuilderOptions() -> LLVMPassBuilderOptionsRef;
    pub fn LLVMDisposePassBuilderOptions(Options: LLVMPassBuilderOptionsRef);
    pub fn LLVMPassBuilderOptionsSetLoopUnrolling(
        Options: LLVMPassBuilderOptionsRef,
        LoopUnrolling: LLVMBool,
    );
    pub fn LLVMPassBuilderOptionsSetLoopVectorization(
        Options: LLVMPassBuilderOptionsRef,
        LoopVectorization: LLVMBool,
    );
    pub fn LLVMPassBuilderOptionsSetSLPVectorization(
        Options: LLVMPassBuilderOptionsRef,
        SLPVectorization: LLVMBool,
    );
    pub fn LLVMPassBuilderOptionsSetLoopInterleaving(
        Options: LLVMPassBuilderOptionsRef,
        LoopInterleaving: LLVMBool,
    );
    pub fn LLVMPassBuilderOptionsSetMergeFunctions(
        Options: LLVMPassBuilderOptionsRef,
        MergeFunctions: LLVMBool,
    );

    // ---- bitcode (BitWriter.h) ----
    pub fn LLVMWriteBitcodeToFile(M: LLVMModuleRef, Path: *const c_char) -> c_int;

    // ---- error handling (Error.h) ----
    pub fn LLVMGetErrorMessage(Err: LLVMErrorRef) -> *mut c_char;
    pub fn LLVMDisposeErrorMessage(ErrMsg: *mut c_char);
}

// jam C++ shim — compiled by build.rs from `shim/jam_shim.cpp`, NOT part of
// libLLVM. Sets `TargetOptions` fields (FunctionSections/DataSections) that the
// LLVM-C API does not expose, and runs the new-PM optimization pipeline copied
// verbatim from the C++ facade (so the pass pipeline is provably identical to
// the oracle's).
unsafe extern "C" {
    pub fn jam_set_target_machine_sections(
        TM: LLVMTargetMachineRef,
        FunctionSections: c_int,
        DataSections: c_int,
    );

    /// Run the module optimization pipeline (PassBuilder + the pre-pipeline
    /// internalize/globaldce + size attrs), exactly as the C++ facade does.
    /// `optLevel` matches the `OptLevel` enum discriminants
    /// (None=0,Less=1,Default=2,Aggressive=3,Size=4,Small=5); `lto` matches the
    /// `Lto` enum (Off=0,Thin=1,Fat=2).
    pub fn jam_shim_optimize(
        M: LLVMModuleRef,
        TM: LLVMTargetMachineRef,
        optLevel: c_int,
        isDebug: c_int,
        lto: c_int,
    );
}
