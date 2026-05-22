/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_LLVM_H
#define JAM_LLVM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define JAM_EXTERN_C extern "C"
#else
#define JAM_EXTERN_C
#endif

// Opaque pointer types for LLVM objects
typedef struct JamLLVMContext *JamContextRef;
typedef struct JamLLVMModule *JamModuleRef;
typedef struct JamLLVMBuilder *JamBuilderRef;
typedef struct JamLLVMType *JamTypeRef;
typedef struct JamLLVMValue *JamValueRef;
typedef struct JamLLVMBasicBlock *JamBasicBlockRef;
typedef struct JamLLVMFunction *JamFunctionRef;
typedef struct JamLLVMTargetMachine *JamTargetMachineRef;

// Calling conventions
typedef enum {
	JAM_CALLCONV_C = 0,
	JAM_CALLCONV_FAST = 8,
	JAM_CALLCONV_COLD = 9,
} JamCallingConv;

// Linkage types
typedef enum {
	JAM_LINKAGE_EXTERNAL = 0,
	JAM_LINKAGE_INTERNAL = 1,
	JAM_LINKAGE_PRIVATE = 2,
} JamLinkage;

// Integer comparison predicates
typedef enum {
	JAM_ICMP_EQ = 32,   // equal
	JAM_ICMP_NE = 33,   // not equal
	JAM_ICMP_UGT = 34,  // unsigned greater than
	JAM_ICMP_UGE = 35,  // unsigned greater or equal
	JAM_ICMP_ULT = 36,  // unsigned less than
	JAM_ICMP_ULE = 37,  // unsigned less or equal
	JAM_ICMP_SGT = 38,  // signed greater than
	JAM_ICMP_SGE = 39,  // signed greater or equal
	JAM_ICMP_SLT = 40,  // signed less than
	JAM_ICMP_SLE = 41,  // signed less or equal
} JamIntPredicate;

// Float comparison predicates. `O` = ordered (NaN inputs -> false);
// `U` = unordered (NaN inputs -> true). Jam emits the ordered
// variants by default so NaN never satisfies a comparison.
typedef enum {
	JAM_FCMP_FALSE = 50,  // always false
	JAM_FCMP_OEQ = 51,
	JAM_FCMP_OGT = 52,
	JAM_FCMP_OGE = 53,
	JAM_FCMP_OLT = 54,
	JAM_FCMP_OLE = 55,
	JAM_FCMP_ONE = 56,
	JAM_FCMP_ORD = 57,
	JAM_FCMP_UNO = 58,
	JAM_FCMP_UEQ = 59,
	JAM_FCMP_UGT = 60,
	JAM_FCMP_UGE = 61,
	JAM_FCMP_ULT = 62,
	JAM_FCMP_ULE = 63,
	JAM_FCMP_UNE = 64,
	JAM_FCMP_TRUE = 65,  // always true
} JamFloatPredicate;

JAM_EXTERN_C void JamLLVMInitializeNativeTarget(void);
JAM_EXTERN_C void JamLLVMInitializeNativeAsmPrinter(void);
JAM_EXTERN_C void JamLLVMInitializeNativeAsmParser(void);
JAM_EXTERN_C void JamLLVMInitializeAllTargets(void);

JAM_EXTERN_C JamContextRef JamLLVMCreateContext(void);
JAM_EXTERN_C void JamLLVMDisposeContext(JamContextRef ctx);

JAM_EXTERN_C JamModuleRef JamLLVMCreateModule(const char *name,
                                              JamContextRef ctx);
JAM_EXTERN_C void JamLLVMDisposeModule(JamModuleRef mod);
JAM_EXTERN_C void JamLLVMSetTargetTriple(JamModuleRef mod, const char *triple);
JAM_EXTERN_C void JamLLVMSetDataLayout(JamModuleRef mod,
                                       JamTargetMachineRef tm);
JAM_EXTERN_C JamFunctionRef JamLLVMGetFunction(JamModuleRef mod,
                                               const char *name);
JAM_EXTERN_C char *JamLLVMPrintModuleToString(JamModuleRef mod);
JAM_EXTERN_C void JamLLVMDisposeMessage(char *msg);

JAM_EXTERN_C JamBuilderRef JamLLVMCreateBuilder(JamContextRef ctx);
JAM_EXTERN_C void JamLLVMDisposeBuilder(JamBuilderRef builder);
JAM_EXTERN_C void JamLLVMPositionBuilderAtEnd(JamBuilderRef builder,
                                              JamBasicBlockRef block);
JAM_EXTERN_C JamBasicBlockRef JamLLVMGetInsertBlock(JamBuilderRef builder);

JAM_EXTERN_C JamTypeRef JamLLVMInt1Type(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMInt8Type(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMInt16Type(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMInt32Type(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMInt64Type(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMFloatType(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMDoubleType(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMVoidType(JamContextRef ctx);
JAM_EXTERN_C JamTypeRef JamLLVMPointerType(JamTypeRef elementType,
                                           unsigned addressSpace);
JAM_EXTERN_C JamTypeRef JamLLVMStructType(JamContextRef ctx,
                                          JamTypeRef *elementTypes,
                                          unsigned elementCount, bool packed);
JAM_EXTERN_C JamTypeRef JamLLVMStructCreateNamed(JamContextRef ctx,
                                                 const char *name);
JAM_EXTERN_C void JamLLVMStructSetBody(JamTypeRef structType,
                                       JamTypeRef *elementTypes,
                                       unsigned elementCount, bool packed);
JAM_EXTERN_C JamTypeRef JamLLVMFunctionType(JamTypeRef returnType,
                                            JamTypeRef *paramTypes,
                                            unsigned paramCount, bool isVarArg);
JAM_EXTERN_C JamTypeRef JamLLVMArrayType(JamTypeRef elementType,
                                         unsigned elementCount);
JAM_EXTERN_C bool JamLLVMTypeIsVoid(JamTypeRef type);
JAM_EXTERN_C bool JamLLVMTypeIsStruct(JamTypeRef type);
JAM_EXTERN_C bool JamLLVMTypeIsInteger(JamTypeRef type);
JAM_EXTERN_C bool JamLLVMTypeIsFloat(JamTypeRef type);
JAM_EXTERN_C bool JamLLVMTypeIsPointer(JamTypeRef type);
JAM_EXTERN_C bool JamLLVMTypeIsArray(JamTypeRef type);
JAM_EXTERN_C unsigned JamLLVMGetIntTypeWidth(JamTypeRef type);

JAM_EXTERN_C JamValueRef JamLLVMConstInt(JamTypeRef type, uint64_t val,
                                         bool signExtend);
JAM_EXTERN_C JamValueRef JamLLVMConstReal(JamTypeRef type, double val);
JAM_EXTERN_C JamValueRef JamLLVMConstNull(JamTypeRef type);
JAM_EXTERN_C JamValueRef JamLLVMConstString(JamContextRef ctx, const char *str,
                                            unsigned length,
                                            bool nullTerminate);
JAM_EXTERN_C JamValueRef JamLLVMConstStringInContext(JamContextRef ctx,
                                                     const char *str,
                                                     unsigned length,
                                                     bool dontNullTerminate);
JAM_EXTERN_C JamValueRef JamLLVMGetUndef(JamTypeRef type);

JAM_EXTERN_C JamValueRef JamLLVMAddGlobalString(JamModuleRef mod,
                                                const char *str,
                                                const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildGlobalStringPtr(JamBuilderRef builder,
                                                     const char *str,
                                                     const char *name);
JAM_EXTERN_C JamValueRef JamLLVMAddGlobal(JamModuleRef mod, JamTypeRef type,
                                          const char *name);
JAM_EXTERN_C void JamLLVMSetGlobalConstant(JamValueRef global, bool isConstant);
JAM_EXTERN_C void JamLLVMSetInitializer(JamValueRef global,
                                        JamValueRef constantVal);

JAM_EXTERN_C JamFunctionRef JamLLVMAddFunction(JamModuleRef mod,
                                               const char *name,
                                               JamTypeRef funcType);
JAM_EXTERN_C void JamLLVMSetFunctionCallConv(JamFunctionRef func,
                                             JamCallingConv cc);
JAM_EXTERN_C void JamLLVMSetLinkage(JamValueRef global, JamLinkage linkage);
JAM_EXTERN_C unsigned JamLLVMCountParams(JamFunctionRef func);
JAM_EXTERN_C JamValueRef JamLLVMGetParam(JamFunctionRef func, unsigned index);
JAM_EXTERN_C bool JamLLVMFunctionIsVarArg(JamFunctionRef func);
// Attach the LLVM `zeroext` attribute to a function parameter / return slot.
// Used to satisfy the C ABI when an i1 (Jam bool) crosses the extern/export
// boundary — C's _Bool is conventionally zero-extended to int.
JAM_EXTERN_C void JamLLVMAddParamAttrZeroExt(JamFunctionRef func,
                                             unsigned argIdx);
JAM_EXTERN_C void JamLLVMAddRetAttrZeroExt(JamFunctionRef func);

JAM_EXTERN_C void JamLLVMSetFunctionNoReturn(JamFunctionRef func);

JAM_EXTERN_C void JamLLVMApplyDefaultFnAttrs(JamFunctionRef func,
                                             bool isExtern);

// mark a function parameter as the sret (struct-return) slot.
// Equivalent to LLVM `sret(<type>) align <a> noalias`. The argument
// must be `ptr`-typed; the pointee type and alignment are passed
// explicitly. Used by codegen for functions returning aggregates whose
// size exceeds the by-value threshold.
// Reflective queries on a function's parameter ABI. Used at call sites
// to detect that the LLVM signature has a leading `ptr sret(%T)` arg —
// the call needs to alloca a slot and prepend it, then load the result
// back. Returns nullptr-wrapped type if the function has no sret arg.
JAM_EXTERN_C bool JamLLVMFunctionUsesSret(JamFunctionRef func);
JAM_EXTERN_C JamTypeRef JamLLVMFunctionSretPointeeType(JamFunctionRef func);

JAM_EXTERN_C void JamLLVMAddParamAttrSret(JamFunctionRef func, unsigned argIdx,
                                          JamTypeRef pointeeType,
                                          unsigned align);
JAM_EXTERN_C void JamLLVMSetValueName(JamValueRef val, const char *name);
JAM_EXTERN_C JamTypeRef JamLLVMGetReturnType(JamFunctionRef func);
JAM_EXTERN_C bool JamLLVMVerifyFunction(JamFunctionRef func);

JAM_EXTERN_C JamBasicBlockRef JamLLVMCreateBasicBlock(JamContextRef ctx,
                                                      const char *name);
JAM_EXTERN_C JamBasicBlockRef JamLLVMAppendBasicBlock(JamFunctionRef func,
                                                      const char *name);
JAM_EXTERN_C JamFunctionRef JamLLVMGetBasicBlockParent(JamBasicBlockRef block);
JAM_EXTERN_C JamValueRef JamLLVMGetBasicBlockTerminator(JamBasicBlockRef block);

// LLVM `switch` instruction: structural dispatch on an integer
// scrutinee. `defaultBlock` is the else target reached when no case
// matches. `numCasesHint` is a size hint (LLVM resizes if more cases
// are added). Returns the SwitchInst as a JamValueRef so callers can
// hand it back to JamLLVMAddCase.
JAM_EXTERN_C JamValueRef JamLLVMBuildSwitch(JamBuilderRef builder,
                                            JamValueRef scrut,
                                            JamBasicBlockRef defaultBlock,
                                            unsigned numCasesHint);
JAM_EXTERN_C void JamLLVMAddCase(JamValueRef switchInst, JamValueRef caseVal,
                                 JamBasicBlockRef caseBlock);

// Position the builder immediately before the block's terminator
// instruction. Used to insert zext/sext for peer-type promotion of
// match-arm values after the arm has already emitted its `br merge`.
JAM_EXTERN_C void
JamLLVMPositionBuilderBeforeTerminator(JamBuilderRef builder,
                                       JamBasicBlockRef block);

JAM_EXTERN_C JamValueRef JamLLVMBuildZExt(JamBuilderRef builder,
                                          JamValueRef val, JamTypeRef destType,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildSExt(JamBuilderRef builder,
                                          JamValueRef val, JamTypeRef destType,
                                          const char *name);

// Stack alloca with an explicit alignment in bytes. Pass 0 to fall back to
// LLVM's data-layout-derived inference, but prefer passing the type's real
// alignment (via JamCodegenContext::typeAlign). LLVM's getPrefTypeAlign
// over-aligns aggregates relative to what other LLVM-based compilers
// produce on the same target — explicitly setting the alignment from
// our own type model keeps layout consistent with C/C++ peer code.
JAM_EXTERN_C JamValueRef JamLLVMBuildAlloca(JamBuilderRef builder,
                                            JamTypeRef type,
                                            uint64_t alignBytes,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildLoad(JamBuilderRef builder,
                                          JamTypeRef type, JamValueRef ptr,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildStore(JamBuilderRef builder,
                                           JamValueRef val, JamValueRef ptr);
// In-bounds GEP for indexing into a fixed-size array: gep [N x T], ptr, 0, idx.
// `arrayType` must be the array aggregate type that `ptr` points to.
JAM_EXTERN_C JamValueRef JamLLVMBuildArrayGEP(JamBuilderRef builder,
                                              JamTypeRef arrayType,
                                              JamValueRef ptr, JamValueRef idx,
                                              const char *name);
// Struct field GEP: returns a pointer to field `fieldIdx` of the struct
// pointed to by `ptr`. `structType` must be the struct type `ptr` points to.
JAM_EXTERN_C JamValueRef JamLLVMBuildStructGEP(JamBuilderRef builder,
                                               JamTypeRef structType,
                                               JamValueRef ptr,
                                               unsigned fieldIdx,
                                               const char *name);
// Pointer GEP: single-index `gep T, ptr, idx` for stepping by element-sized
// strides through a many-item pointer (no leading 0). `elemType` is the
// pointee type (T).
JAM_EXTERN_C JamValueRef JamLLVMBuildPtrGEP(JamBuilderRef builder,
                                            JamTypeRef elemType,
                                            JamValueRef ptr, JamValueRef idx,
                                            const char *name);
// Returns the element type of an array type (e.g. [200 x i8] -> i8).
JAM_EXTERN_C JamTypeRef JamLLVMGetArrayElementType(JamTypeRef arrayType);

JAM_EXTERN_C JamValueRef JamLLVMBuildAdd(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildSub(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildMul(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildURem(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildSRem(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildUDiv(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildSDiv(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildAnd(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildOr(JamBuilderRef builder, JamValueRef lhs,
                                        JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildXor(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildShl(JamBuilderRef builder, JamValueRef lhs,
                                         JamValueRef rhs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildLShr(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildAShr(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);

JAM_EXTERN_C JamValueRef JamLLVMBuildICmp(JamBuilderRef builder,
                                          JamIntPredicate pred, JamValueRef lhs,
                                          JamValueRef rhs, const char *name);

JAM_EXTERN_C JamValueRef JamLLVMBuildBr(JamBuilderRef builder,
                                        JamBasicBlockRef dest);
JAM_EXTERN_C JamValueRef JamLLVMBuildCondBr(JamBuilderRef builder,
                                            JamValueRef cond,
                                            JamBasicBlockRef thenBlock,
                                            JamBasicBlockRef elseBlock);
JAM_EXTERN_C JamValueRef JamLLVMBuildRet(JamBuilderRef builder,
                                         JamValueRef val);
JAM_EXTERN_C JamValueRef JamLLVMBuildRetVoid(JamBuilderRef builder);
JAM_EXTERN_C JamValueRef JamLLVMBuildUnreachable(JamBuilderRef builder);
JAM_EXTERN_C JamValueRef JamLLVMBuildCall(JamBuilderRef builder,
                                          JamFunctionRef func,
                                          JamValueRef *args, unsigned numArgs,
                                          const char *name);
// Indirect call through a function-typed value. `funcType` is the LLVM
// FunctionType built from the Jam-level Fn TypeIdx (return + params);
// `callee` is the ptr-typed Value holding the code address. Used by
// JirTag::CallIndirect lowering.
JAM_EXTERN_C JamValueRef JamLLVMBuildIndirectCall(
    JamBuilderRef builder, JamTypeRef funcType, JamValueRef callee,
    JamValueRef *args, unsigned numArgs, const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildPhi(JamBuilderRef builder, JamTypeRef type,
                                         const char *name);
JAM_EXTERN_C void JamLLVMAddIncoming(JamValueRef phi, JamValueRef *values,
                                     JamBasicBlockRef *blocks, unsigned count);

JAM_EXTERN_C JamValueRef JamLLVMBuildBitCast(JamBuilderRef builder,
                                             JamValueRef val,
                                             JamTypeRef destType,
                                             const char *name);
// Raw-address conversions used to lower Jam's `ptr as int` and
// `int as ptr` casts (plus the FnRef path which surfaces a function's
// address as a u64). Both lower to a single LLVM instruction.
JAM_EXTERN_C JamValueRef JamLLVMBuildPtrToInt(JamBuilderRef builder,
                                              JamValueRef val,
                                              JamTypeRef destType,
                                              const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildIntToPtr(JamBuilderRef builder,
                                              JamValueRef val,
                                              JamTypeRef destType,
                                              const char *name);
// View a function as a generic Value pointer — needed because
// llvm::Function inherits from llvm::Value, and the rest of the
// codegen plumbing speaks in JamValueRef. Used by the FnRef lowering
// to feed the function into JamLLVMBuildPtrToInt.
JAM_EXTERN_C JamValueRef JamLLVMFunctionAsValue(JamFunctionRef func);
JAM_EXTERN_C JamValueRef JamLLVMBuildIntCast(JamBuilderRef builder,
                                             JamValueRef val,
                                             JamTypeRef destType, bool isSigned,
                                             const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildSIToFP(JamBuilderRef builder,
                                            JamValueRef val,
                                            JamTypeRef destType,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildUIToFP(JamBuilderRef builder,
                                            JamValueRef val,
                                            JamTypeRef destType,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFPCast(JamBuilderRef builder,
                                            JamValueRef val,
                                            JamTypeRef destType,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFPToSI(JamBuilderRef builder,
                                            JamValueRef val,
                                            JamTypeRef destType,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFPToUI(JamBuilderRef builder,
                                            JamValueRef val,
                                            JamTypeRef destType,
                                            const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFAdd(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFSub(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFMul(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFDiv(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFRem(JamBuilderRef builder,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFNeg(JamBuilderRef builder,
                                          JamValueRef operand,
                                          const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildFCmp(JamBuilderRef builder,
                                          JamFloatPredicate pred,
                                          JamValueRef lhs, JamValueRef rhs,
                                          const char *name);

JAM_EXTERN_C JamValueRef JamLLVMBuildInsertValue(JamBuilderRef builder,
                                                 JamValueRef agg,
                                                 JamValueRef val,
                                                 unsigned index,
                                                 const char *name);
JAM_EXTERN_C JamValueRef JamLLVMBuildExtractValue(JamBuilderRef builder,
                                                  JamValueRef agg,
                                                  unsigned index,
                                                  const char *name);

JAM_EXTERN_C JamTypeRef JamLLVMTypeOf(JamValueRef val);
JAM_EXTERN_C JamTypeRef JamLLVMGetAllocatedType(JamValueRef alloca);

JAM_EXTERN_C char *JamLLVMGetDefaultTargetTriple(void);
JAM_EXTERN_C char *JamLLVMGetHostCPUName(void);
JAM_EXTERN_C char *JamLLVMGetHostCPUFeatures(void);

// Optimization levels selected at TargetMachine creation. Default for jam is
// `None` — debug builds compile 5-10× faster than -O2 because LLVM's machine
// codegen + the full new-PM module pipeline are the dominant cost. The IR-
// level pipeline (inlining, GVN, SROA, vectorization, MergeFunctions,
// globaldce, …) is driven by these values inside JamLLVMEmitObjectFile.
//
// SIZE/SMALL still run codegen at Aggressive (instruction selection / regalloc
// have no "size" tier) — size optimization happens at the IR-pipeline level
// and via function attributes.
typedef enum {
	JAM_OPT_NONE = 0,        // -O0
	JAM_OPT_LESS = 1,        // -O1
	JAM_OPT_DEFAULT = 2,     // -O2  (LLVM default)
	JAM_OPT_AGGRESSIVE = 3,  // -O3
	JAM_OPT_SIZE = 4,        // -Os  (moderate size; optsize attr)
	JAM_OPT_SMALL = 5,       // -Oz  (minsize + optsize attrs)
} JamOptLevel;

// Link-time optimization mode. When enabled, jam emits LLVM bitcode (.bc)
// instead of an object file, and clang/lld re-runs the optimization pipeline
// across the bitcode plus any LTO-compatible static libraries at link time.
//
// Useful even though jam already compiles to a single combined module:
// link-time LTO lets the optimizer see across into libc++/libc/static
// archives that were themselves built with LTO bitcode, enabling
// cross-boundary inlining and DCE.
typedef enum {
	JAM_LTO_OFF = 0,   // emit a regular object file (default)
	JAM_LTO_THIN = 1,  // ThinLTO bitcode — fast, parallel link
	JAM_LTO_FAT = 2,   // full LTO bitcode — slowest link, most opt
} JamLTO;

// Symbol/debug-info stripping mode applied at link time. Pure linker-flag
// plumbing — emit-time IR doesn't carry debug info today, so these flags
// remove the tiny tables (unwind info, dyld stubs, local symbols) that ld
// emits by default. Matches rustc's `-C strip={none,debuginfo,symbols}`.
typedef enum {
	JAM_STRIP_NONE = 0,       // keep all debug info & symbols (default)
	JAM_STRIP_DEBUGINFO = 1,  // strip debug info only
	JAM_STRIP_SYMBOLS = 2,    // strip debug info + local symbols
} JamStrip;

JAM_EXTERN_C JamTargetMachineRef JamLLVMCreateTargetMachine(
    const char *triple, const char *cpu, const char *features,
    bool isRelocationPIC, JamOptLevel optLevel, JamLTO lto);
JAM_EXTERN_C void JamLLVMDisposeTargetMachine(JamTargetMachineRef tm);

// Emits an object file at `filename` when the TargetMachine's LTO mode is
// `JAM_LTO_OFF`. When LTO is on, emits LLVM bitcode at the same path — the
// caller is responsible for choosing a `.bc`-shaped path and passing
// `-flto={thin,full}` plus the bitcode file to the system linker.
JAM_EXTERN_C bool JamLLVMEmitObjectFile(JamModuleRef mod,
                                        JamTargetMachineRef tm,
                                        const char *filename,
                                        char **errorMessage);

#endif  // JAM_LLVM_H
