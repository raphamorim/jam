/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "jam_llvm.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <cstring>

#define WRAP_CONTEXT(ctx) reinterpret_cast<JamContextRef>(ctx)
#define UNWRAP_CONTEXT(ctx) reinterpret_cast<llvm::LLVMContext *>(ctx)

#define WRAP_MODULE(mod) reinterpret_cast<JamModuleRef>(mod)
#define UNWRAP_MODULE(mod) reinterpret_cast<llvm::Module *>(mod)

#define WRAP_BUILDER(builder) reinterpret_cast<JamBuilderRef>(builder)
#define UNWRAP_BUILDER(builder) reinterpret_cast<llvm::IRBuilder<> *>(builder)

#define WRAP_TYPE(type) reinterpret_cast<JamTypeRef>(type)
#define UNWRAP_TYPE(type) reinterpret_cast<llvm::Type *>(type)

#define WRAP_VALUE(val) reinterpret_cast<JamValueRef>(val)
#define UNWRAP_VALUE(val) reinterpret_cast<llvm::Value *>(val)

#define WRAP_BLOCK(block) reinterpret_cast<JamBasicBlockRef>(block)
#define UNWRAP_BLOCK(block) reinterpret_cast<llvm::BasicBlock *>(block)

#define WRAP_FUNCTION(func) reinterpret_cast<JamFunctionRef>(func)
#define UNWRAP_FUNCTION(func) reinterpret_cast<llvm::Function *>(func)

namespace {
// Internal wrapper that pairs the TargetMachine with the JamOptLevel and LTO
// mode the user requested. Both are captured at TargetMachine creation so the
// emit step can build the correct new-PM pipeline (O0 / Oz / O1-O3, optionally
// LTO pre-link) without changing the public C API. UNWRAP_TARGET_MACHINE still
// yields the raw TargetMachine*, so existing callsites keep working.
struct JamTargetMachineImpl {
	llvm::TargetMachine *tm;
	JamOptLevel optLevel;
	JamLTO lto;
};
}  // namespace
#define WRAP_TARGET_MACHINE(impl) reinterpret_cast<JamTargetMachineRef>(impl)
#define UNWRAP_TARGET_MACHINE_IMPL(tm)                                         \
	reinterpret_cast<JamTargetMachineImpl *>(tm)
#define UNWRAP_TARGET_MACHINE(tm) (UNWRAP_TARGET_MACHINE_IMPL(tm)->tm)

void JamLLVMInitializeNativeTarget(void) { llvm::InitializeNativeTarget(); }

void JamLLVMInitializeNativeAsmPrinter(void) {
	llvm::InitializeNativeTargetAsmPrinter();
}

void JamLLVMInitializeNativeAsmParser(void) {
	llvm::InitializeNativeTargetAsmParser();
}

void JamLLVMInitializeAllTargets(void) {
	llvm::InitializeAllTargetInfos();
	llvm::InitializeAllTargets();
	llvm::InitializeAllTargetMCs();
	llvm::InitializeAllAsmParsers();
	llvm::InitializeAllAsmPrinters();
}

JamContextRef JamLLVMCreateContext(void) {
	auto *ctx = new llvm::LLVMContext();
	// Drop SSA value names at construction time. Most production LLVM
	// frontends emit IR with auto-numbered temporaries (`%0`, `%1`, …)
	// instead of the source-named values our codegen passes through.
	// Names cost LLVM memory (string storage on every Value) and don't
	// affect codegen at
	// all — they only show up in printed IR. Discarding them here makes
	// `--emit-ir` output match what production compilers print without
	// having to plumb empty strings through every CreateAlloca/CreateGEP
	// call site.
	ctx->setDiscardValueNames(true);
	return WRAP_CONTEXT(ctx);
}

void JamLLVMDisposeContext(JamContextRef ctx) { delete UNWRAP_CONTEXT(ctx); }

JamModuleRef JamLLVMCreateModule(const char *name, JamContextRef ctx) {
	return WRAP_MODULE(new llvm::Module(name, *UNWRAP_CONTEXT(ctx)));
}

void JamLLVMDisposeModule(JamModuleRef mod) { delete UNWRAP_MODULE(mod); }

void JamLLVMSetTargetTriple(JamModuleRef mod, const char *triple) {
	UNWRAP_MODULE(mod)->setTargetTriple(triple);
}

void JamLLVMSetDataLayout(JamModuleRef mod, JamTargetMachineRef tm) {
	llvm::Module *m = UNWRAP_MODULE(mod);
	llvm::TargetMachine *targetMachine = UNWRAP_TARGET_MACHINE(tm);
	m->setDataLayout(targetMachine->createDataLayout());

	// it does stamp the module-level PIC/PIE flags so the emitted ELF object's
	// e_flags advertise PIC. Without this, even when the TargetMachine uses
	// Reloc::PIC_ to emit PC-relative relocations, the object is marked
	// "not PIC" and a system linker building a PIE binary will refuse it
	// (R_X86_64_32 against .rodata can not be used when making a PIE
	// object).
	const auto relocModel = targetMachine->getRelocationModel();
	if (relocModel == llvm::Reloc::PIC_) {
		m->setPICLevel(llvm::PICLevel::BigPIC);
		m->setPIELevel(llvm::PIELevel::Large);
	}
}

JamFunctionRef JamLLVMGetFunction(JamModuleRef mod, const char *name) {
	return WRAP_FUNCTION(UNWRAP_MODULE(mod)->getFunction(name));
}

char *JamLLVMPrintModuleToString(JamModuleRef mod) {
	std::string output;
	llvm::raw_string_ostream stream(output);
	UNWRAP_MODULE(mod)->print(stream, nullptr);
	return strdup(output.c_str());
}

void JamLLVMDisposeMessage(char *msg) { free(msg); }

JamBuilderRef JamLLVMCreateBuilder(JamContextRef ctx) {
	return WRAP_BUILDER(new llvm::IRBuilder<>(*UNWRAP_CONTEXT(ctx)));
}

void JamLLVMDisposeBuilder(JamBuilderRef builder) {
	delete UNWRAP_BUILDER(builder);
}

void JamLLVMPositionBuilderAtEnd(JamBuilderRef builder,
                                 JamBasicBlockRef block) {
	UNWRAP_BUILDER(builder)->SetInsertPoint(UNWRAP_BLOCK(block));
}

JamBasicBlockRef JamLLVMGetInsertBlock(JamBuilderRef builder) {
	return WRAP_BLOCK(UNWRAP_BUILDER(builder)->GetInsertBlock());
}

JamTypeRef JamLLVMInt1Type(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getInt1Ty(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMInt8Type(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getInt8Ty(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMInt16Type(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getInt16Ty(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMInt32Type(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getInt32Ty(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMInt64Type(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getInt64Ty(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMFloatType(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getFloatTy(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMDoubleType(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getDoubleTy(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMVoidType(JamContextRef ctx) {
	return WRAP_TYPE(llvm::Type::getVoidTy(*UNWRAP_CONTEXT(ctx)));
}

JamTypeRef JamLLVMPointerType(JamTypeRef elementType, unsigned addressSpace) {
	return WRAP_TYPE(
	    llvm::PointerType::get(UNWRAP_TYPE(elementType), addressSpace));
}

JamTypeRef JamLLVMStructType(JamContextRef ctx, JamTypeRef *elementTypes,
                             unsigned elementCount, bool packed) {
	std::vector<llvm::Type *> types;
	for (unsigned i = 0; i < elementCount; i++) {
		types.push_back(UNWRAP_TYPE(elementTypes[i]));
	}
	return WRAP_TYPE(
	    llvm::StructType::get(*UNWRAP_CONTEXT(ctx), types, packed));
}

JamTypeRef JamLLVMStructCreateNamed(JamContextRef ctx, const char *name) {
	return WRAP_TYPE(llvm::StructType::create(*UNWRAP_CONTEXT(ctx), name));
}

void JamLLVMStructSetBody(JamTypeRef structType, JamTypeRef *elementTypes,
                          unsigned elementCount, bool packed) {
	std::vector<llvm::Type *> types;
	for (unsigned i = 0; i < elementCount; i++) {
		types.push_back(UNWRAP_TYPE(elementTypes[i]));
	}
	llvm::cast<llvm::StructType>(UNWRAP_TYPE(structType))
	    ->setBody(types, packed);
}

JamTypeRef JamLLVMFunctionType(JamTypeRef returnType, JamTypeRef *paramTypes,
                               unsigned paramCount, bool isVarArg) {
	std::vector<llvm::Type *> params;
	for (unsigned i = 0; i < paramCount; i++) {
		params.push_back(UNWRAP_TYPE(paramTypes[i]));
	}
	return WRAP_TYPE(
	    llvm::FunctionType::get(UNWRAP_TYPE(returnType), params, isVarArg));
}

bool JamLLVMTypeIsVoid(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isVoidTy();
}

bool JamLLVMTypeIsStruct(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isStructTy();
}

bool JamLLVMTypeIsInteger(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isIntegerTy();
}

bool JamLLVMTypeIsFloat(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isFloatingPointTy();
}

bool JamLLVMTypeIsPointer(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isPointerTy();
}

bool JamLLVMTypeIsArray(JamTypeRef type) {
	return UNWRAP_TYPE(type)->isArrayTy();
}

JamTypeRef JamLLVMArrayType(JamTypeRef elementType, unsigned elementCount) {
	return WRAP_TYPE(
	    llvm::ArrayType::get(UNWRAP_TYPE(elementType), elementCount));
}

JamTypeRef JamLLVMGetArrayElementType(JamTypeRef arrayType) {
	auto *aT = llvm::cast<llvm::ArrayType>(UNWRAP_TYPE(arrayType));
	return WRAP_TYPE(aT->getElementType());
}

unsigned JamLLVMGetIntTypeWidth(JamTypeRef type) {
	return UNWRAP_TYPE(type)->getIntegerBitWidth();
}

JamValueRef JamLLVMConstInt(JamTypeRef type, uint64_t val, bool signExtend) {
	return WRAP_VALUE(
	    llvm::ConstantInt::get(UNWRAP_TYPE(type), val, signExtend));
}

JamValueRef JamLLVMConstReal(JamTypeRef type, double val) {
	return WRAP_VALUE(llvm::ConstantFP::get(UNWRAP_TYPE(type), val));
}

JamValueRef JamLLVMConstNull(JamTypeRef type) {
	return WRAP_VALUE(llvm::Constant::getNullValue(UNWRAP_TYPE(type)));
}

JamValueRef JamLLVMConstString(JamContextRef ctx, const char *str,
                               unsigned length, bool nullTerminate) {
	return WRAP_VALUE(llvm::ConstantDataArray::getString(
	    *UNWRAP_CONTEXT(ctx), llvm::StringRef(str, length), nullTerminate));
}

JamValueRef JamLLVMConstStringInContext(JamContextRef ctx, const char *str,
                                        unsigned length,
                                        bool dontNullTerminate) {
	return WRAP_VALUE(llvm::ConstantDataArray::getString(
	    *UNWRAP_CONTEXT(ctx), llvm::StringRef(str, length),
	    !dontNullTerminate));
}

JamValueRef JamLLVMGetUndef(JamTypeRef type) {
	return WRAP_VALUE(llvm::UndefValue::get(UNWRAP_TYPE(type)));
}

JamValueRef JamLLVMAddGlobalString(JamModuleRef mod, const char *str,
                                   const char *name) {
	llvm::Module *module = UNWRAP_MODULE(mod);
	llvm::Constant *strConstant =
	    llvm::ConstantDataArray::getString(module->getContext(), str, true);
	llvm::GlobalVariable *global = new llvm::GlobalVariable(
	    *module, strConstant->getType(),
	    true,  // isConstant
	    llvm::GlobalValue::PrivateLinkage, strConstant, name);
	return WRAP_VALUE(global);
}

JamValueRef JamLLVMBuildGlobalStringPtr(JamBuilderRef builder, const char *str,
                                        const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateGlobalString(str, name));
}

JamValueRef JamLLVMAddGlobal(JamModuleRef mod, JamTypeRef type,
                             const char *name) {
	llvm::Module *module = UNWRAP_MODULE(mod);
	return WRAP_VALUE(
	    new llvm::GlobalVariable(*module, UNWRAP_TYPE(type),
	                             false,  // isConstant (set separately)
	                             llvm::GlobalValue::PrivateLinkage,
	                             nullptr,  // initializer (set separately)
	                             name));
}

void JamLLVMSetGlobalConstant(JamValueRef global, bool isConstant) {
	llvm::cast<llvm::GlobalVariable>(UNWRAP_VALUE(global))
	    ->setConstant(isConstant);
}

void JamLLVMSetInitializer(JamValueRef global, JamValueRef constantVal) {
	llvm::cast<llvm::GlobalVariable>(UNWRAP_VALUE(global))
	    ->setInitializer(llvm::cast<llvm::Constant>(UNWRAP_VALUE(constantVal)));
}

JamFunctionRef JamLLVMAddFunction(JamModuleRef mod, const char *name,
                                  JamTypeRef funcType) {
	return WRAP_FUNCTION(llvm::Function::Create(
	    llvm::cast<llvm::FunctionType>(UNWRAP_TYPE(funcType)),
	    llvm::Function::ExternalLinkage, name, UNWRAP_MODULE(mod)));
}

void JamLLVMSetFunctionCallConv(JamFunctionRef func, JamCallingConv cc) {
	llvm::CallingConv::ID llvmCC;
	switch (cc) {
	case JAM_CALLCONV_C:
		llvmCC = llvm::CallingConv::C;
		break;
	case JAM_CALLCONV_FAST:
		llvmCC = llvm::CallingConv::Fast;
		break;
	case JAM_CALLCONV_COLD:
		llvmCC = llvm::CallingConv::Cold;
		break;
	default:
		llvmCC = llvm::CallingConv::C;
		break;
	}
	UNWRAP_FUNCTION(func)->setCallingConv(llvmCC);
}

void JamLLVMSetLinkage(JamValueRef global, JamLinkage linkage) {
	llvm::GlobalValue::LinkageTypes llvmLinkage;
	switch (linkage) {
	case JAM_LINKAGE_EXTERNAL:
		llvmLinkage = llvm::GlobalValue::ExternalLinkage;
		break;
	case JAM_LINKAGE_INTERNAL:
		llvmLinkage = llvm::GlobalValue::InternalLinkage;
		break;
	case JAM_LINKAGE_PRIVATE:
		llvmLinkage = llvm::GlobalValue::PrivateLinkage;
		break;
	default:
		llvmLinkage = llvm::GlobalValue::ExternalLinkage;
		break;
	}
	llvm::cast<llvm::GlobalValue>(UNWRAP_VALUE(global))
	    ->setLinkage(llvmLinkage);
}

unsigned JamLLVMCountParams(JamFunctionRef func) {
	return UNWRAP_FUNCTION(func)->arg_size();
}

JamValueRef JamLLVMGetParam(JamFunctionRef func, unsigned index) {
	return WRAP_VALUE(UNWRAP_FUNCTION(func)->getArg(index));
}

bool JamLLVMFunctionIsVarArg(JamFunctionRef func) {
	return UNWRAP_FUNCTION(func)->getFunctionType()->isVarArg();
}

void JamLLVMAddParamAttrZeroExt(JamFunctionRef func, unsigned argIdx) {
	UNWRAP_FUNCTION(func)->addParamAttr(argIdx, llvm::Attribute::ZExt);
}

void JamLLVMAddRetAttrZeroExt(JamFunctionRef func) {
	UNWRAP_FUNCTION(func)->addRetAttr(llvm::Attribute::ZExt);
}

// Lazily-cached host CPU + feature strings. Computed once on first call and
// reused for every function — querying the host on every AddFunction would
// add up across a large module.
static const std::string &cachedHostCPU() {
	static const std::string s = llvm::sys::getHostCPUName().str();
	return s;
}

static const std::string &cachedHostFeatures() {
	static const std::string s = []() {
		std::string out;
		auto feats = llvm::sys::getHostCPUFeatures();
		for (auto &f : feats) {
			if (!out.empty()) out += ',';
			out += (f.second ? '+' : '-');
			out += f.first().str();
		}
		return out;
	}();
	return s;
}

void JamLLVMApplyDefaultFnAttrs(JamFunctionRef func, bool isExtern) {
	llvm::Function *F = UNWRAP_FUNCTION(func);
	if (!isExtern) {
		F->addFnAttr(llvm::Attribute::NoUnwind);
		F->addFnAttr(llvm::Attribute::getWithUWTableKind(
		    F->getContext(), llvm::UWTableKind::Sync));
	}
	F->addFnAttr("frame-pointer", "all");
	const std::string &cpu = cachedHostCPU();
	if (!cpu.empty()) F->addFnAttr("target-cpu", cpu);
	const std::string &feats = cachedHostFeatures();
	if (!feats.empty()) F->addFnAttr("target-features", feats);
}

void JamLLVMSetFunctionNoReturn(JamFunctionRef func) {
	UNWRAP_FUNCTION(func)->addFnAttr(llvm::Attribute::NoReturn);
}

void JamLLVMAddParamAttrSret(JamFunctionRef func, unsigned argIdx,
                             JamTypeRef pointeeType, unsigned align) {
	llvm::Function *F = UNWRAP_FUNCTION(func);
	llvm::LLVMContext &ctx = F->getContext();
	llvm::Type *ty = UNWRAP_TYPE(pointeeType);
	F->addParamAttr(argIdx, llvm::Attribute::getWithStructRetType(ctx, ty));
	F->addParamAttr(argIdx, llvm::Attribute::NoAlias);
	F->addParamAttr(argIdx,
	                llvm::Attribute::getWithAlignment(ctx, llvm::Align(align)));
}

bool JamLLVMFunctionUsesSret(JamFunctionRef func) {
	llvm::Function *F = UNWRAP_FUNCTION(func);
	return F->arg_size() > 0 &&
	       F->getArg(0)->hasAttribute(llvm::Attribute::StructRet);
}

JamTypeRef JamLLVMFunctionSretPointeeType(JamFunctionRef func) {
	llvm::Function *F = UNWRAP_FUNCTION(func);
	if (F->arg_size() == 0) return WRAP_TYPE((llvm::Type *)nullptr);
	llvm::Attribute attr =
	    F->getArg(0)->getAttribute(llvm::Attribute::StructRet);
	if (!attr.isValid()) return WRAP_TYPE((llvm::Type *)nullptr);
	return WRAP_TYPE(attr.getValueAsType());
}

void JamLLVMSetValueName(JamValueRef val, const char *name) {
	UNWRAP_VALUE(val)->setName(name);
}

JamTypeRef JamLLVMGetReturnType(JamFunctionRef func) {
	return WRAP_TYPE(UNWRAP_FUNCTION(func)->getReturnType());
}

bool JamLLVMVerifyFunction(JamFunctionRef func) {
	return !llvm::verifyFunction(*UNWRAP_FUNCTION(func), &llvm::errs());
}

JamBasicBlockRef JamLLVMCreateBasicBlock(JamContextRef ctx, const char *name) {
	return WRAP_BLOCK(llvm::BasicBlock::Create(*UNWRAP_CONTEXT(ctx), name));
}

JamBasicBlockRef JamLLVMAppendBasicBlock(JamFunctionRef func,
                                         const char *name) {
	llvm::Function *function = UNWRAP_FUNCTION(func);
	return WRAP_BLOCK(
	    llvm::BasicBlock::Create(function->getContext(), name, function));
}

JamFunctionRef JamLLVMGetBasicBlockParent(JamBasicBlockRef block) {
	return WRAP_FUNCTION(UNWRAP_BLOCK(block)->getParent());
}

JamValueRef JamLLVMGetBasicBlockTerminator(JamBasicBlockRef block) {
	return WRAP_VALUE(UNWRAP_BLOCK(block)->getTerminator());
}

JamValueRef JamLLVMBuildAlloca(JamBuilderRef builder, JamTypeRef type,
                               uint64_t alignBytes, const char *name) {
	// Every alloca lives in the function's entry block, not at the
	// builder's current insertion point. Without this, an alloca emitted
	// inside a loop body (e.g. an sret slot for a struct-returning call,
	// an `argtmp` for a by-pointer rvalue, a per-arm match-binding slot)
	// gets a fresh stack allocation every iteration. Because LLVM
	// allocas only release at function return, that quietly grows the
	// stack until SP crosses the guard page and the process dies with
	// EXC_BAD_ACCESS.
	auto *b = UNWRAP_BUILDER(builder);
	auto *curBlock = b->GetInsertBlock();
	llvm::AllocaInst *inst;
	if (curBlock == nullptr) {
		// No active block (shouldn't happen during normal codegen) —
		// emit at the builder's current point and let LLVM verify.
		inst = b->CreateAlloca(UNWRAP_TYPE(type), nullptr, name);
	} else {
		llvm::BasicBlock &entry = curBlock->getParent()->getEntryBlock();
		if (&entry == curBlock) {
			// Already in the entry block — emit at current point. This
			// is the common case at the top of defineBody when laying
			// out parameter and locals; no save/restore needed.
			inst = b->CreateAlloca(UNWRAP_TYPE(type), nullptr, name);
		} else {
			llvm::IRBuilderBase::InsertPoint savedIP = b->saveIP();
			if (entry.empty()) {
				b->SetInsertPoint(&entry);
			} else {
				// Position before the entry block's first instruction
				// so subsequent allocas naturally cluster at the top.
				b->SetInsertPoint(&entry, entry.begin());
			}
			inst = b->CreateAlloca(UNWRAP_TYPE(type), nullptr, name);
			b->restoreIP(savedIP);
		}
	}
	if (alignBytes != 0) { inst->setAlignment(llvm::Align(alignBytes)); }
	return WRAP_VALUE(inst);
}

JamValueRef JamLLVMBuildLoad(JamBuilderRef builder, JamTypeRef type,
                             JamValueRef ptr, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateLoad(
	    UNWRAP_TYPE(type), UNWRAP_VALUE(ptr), name));
}

JamValueRef JamLLVMBuildStore(JamBuilderRef builder, JamValueRef val,
                              JamValueRef ptr) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateStore(UNWRAP_VALUE(val),
	                                                       UNWRAP_VALUE(ptr)));
}

JamValueRef JamLLVMBuildArrayGEP(JamBuilderRef builder, JamTypeRef arrayType,
                                 JamValueRef ptr, JamValueRef idx,
                                 const char *name) {
	auto *b = UNWRAP_BUILDER(builder);
	llvm::Value *zero =
	    llvm::ConstantInt::get(llvm::Type::getInt32Ty(b->getContext()), 0);
	llvm::Value *indices[2] = {zero, UNWRAP_VALUE(idx)};
	return WRAP_VALUE(
	    b->CreateInBoundsGEP(UNWRAP_TYPE(arrayType), UNWRAP_VALUE(ptr),
	                         llvm::ArrayRef<llvm::Value *>(indices, 2), name));
}

JamValueRef JamLLVMBuildStructGEP(JamBuilderRef builder, JamTypeRef structType,
                                  JamValueRef ptr, unsigned fieldIdx,
                                  const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateStructGEP(
	    UNWRAP_TYPE(structType), UNWRAP_VALUE(ptr), fieldIdx, name));
}

JamValueRef JamLLVMBuildPtrGEP(JamBuilderRef builder, JamTypeRef elemType,
                               JamValueRef ptr, JamValueRef idx,
                               const char *name) {
	auto *b = UNWRAP_BUILDER(builder);
	llvm::Value *indices[1] = {UNWRAP_VALUE(idx)};
	return WRAP_VALUE(
	    b->CreateInBoundsGEP(UNWRAP_TYPE(elemType), UNWRAP_VALUE(ptr),
	                         llvm::ArrayRef<llvm::Value *>(indices, 1), name));
}

JamValueRef JamLLVMBuildAdd(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateAdd(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildSub(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSub(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildMul(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateMul(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildURem(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateURem(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildSRem(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSRem(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildUDiv(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateUDiv(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildSDiv(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSDiv(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildAnd(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateAnd(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildOr(JamBuilderRef builder, JamValueRef lhs,
                           JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateOr(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildXor(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateXor(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildShl(JamBuilderRef builder, JamValueRef lhs,
                            JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateShl(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildLShr(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateLShr(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildAShr(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateAShr(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildICmp(JamBuilderRef builder, JamIntPredicate pred,
                             JamValueRef lhs, JamValueRef rhs,
                             const char *name) {
	llvm::CmpInst::Predicate llvmPred;
	switch (pred) {
	case JAM_ICMP_EQ:
		llvmPred = llvm::CmpInst::ICMP_EQ;
		break;
	case JAM_ICMP_NE:
		llvmPred = llvm::CmpInst::ICMP_NE;
		break;
	case JAM_ICMP_UGT:
		llvmPred = llvm::CmpInst::ICMP_UGT;
		break;
	case JAM_ICMP_UGE:
		llvmPred = llvm::CmpInst::ICMP_UGE;
		break;
	case JAM_ICMP_ULT:
		llvmPred = llvm::CmpInst::ICMP_ULT;
		break;
	case JAM_ICMP_ULE:
		llvmPred = llvm::CmpInst::ICMP_ULE;
		break;
	case JAM_ICMP_SGT:
		llvmPred = llvm::CmpInst::ICMP_SGT;
		break;
	case JAM_ICMP_SGE:
		llvmPred = llvm::CmpInst::ICMP_SGE;
		break;
	case JAM_ICMP_SLT:
		llvmPred = llvm::CmpInst::ICMP_SLT;
		break;
	case JAM_ICMP_SLE:
		llvmPred = llvm::CmpInst::ICMP_SLE;
		break;
	default:
		llvmPred = llvm::CmpInst::ICMP_EQ;
		break;
	}
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateICmp(
	    llvmPred, UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildBr(JamBuilderRef builder, JamBasicBlockRef dest) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateBr(UNWRAP_BLOCK(dest)));
}

JamValueRef JamLLVMBuildCondBr(JamBuilderRef builder, JamValueRef cond,
                               JamBasicBlockRef thenBlock,
                               JamBasicBlockRef elseBlock) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateCondBr(
	    UNWRAP_VALUE(cond), UNWRAP_BLOCK(thenBlock), UNWRAP_BLOCK(elseBlock)));
}

JamValueRef JamLLVMBuildSwitch(JamBuilderRef builder, JamValueRef scrut,
                               JamBasicBlockRef defaultBlock,
                               unsigned numCasesHint) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSwitch(
	    UNWRAP_VALUE(scrut), UNWRAP_BLOCK(defaultBlock), numCasesHint));
}

void JamLLVMAddCase(JamValueRef switchInst, JamValueRef caseVal,
                    JamBasicBlockRef caseBlock) {
	llvm::cast<llvm::SwitchInst>(UNWRAP_VALUE(switchInst))
	    ->addCase(llvm::cast<llvm::ConstantInt>(UNWRAP_VALUE(caseVal)),
	              UNWRAP_BLOCK(caseBlock));
}

// Position the builder immediately before `block`'s terminator. Used
// by codegenMatch's value-promotion pass: each arm has already
// emitted drops + a `br merge`, and we need to insert zext/sext/fpext
// (for peer-type resolution) before the branch without losing it.
void JamLLVMPositionBuilderBeforeTerminator(JamBuilderRef builder,
                                            JamBasicBlockRef block) {
	llvm::BasicBlock *bb = UNWRAP_BLOCK(block);
	llvm::Instruction *term = bb->getTerminator();
	if (term) {
		UNWRAP_BUILDER(builder)->SetInsertPoint(term);
	} else {
		UNWRAP_BUILDER(builder)->SetInsertPoint(bb);
	}
}

JamValueRef JamLLVMBuildZExt(JamBuilderRef builder, JamValueRef val,
                             JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateZExt(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildSExt(JamBuilderRef builder, JamValueRef val,
                             JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSExt(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildRet(JamBuilderRef builder, JamValueRef val) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateRet(UNWRAP_VALUE(val)));
}

JamValueRef JamLLVMBuildRetVoid(JamBuilderRef builder) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateRetVoid());
}

JamValueRef JamLLVMBuildUnreachable(JamBuilderRef builder) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateUnreachable());
}

JamValueRef JamLLVMBuildCall(JamBuilderRef builder, JamFunctionRef func,
                             JamValueRef *args, unsigned numArgs,
                             const char *name) {
	std::vector<llvm::Value *> argValues;
	for (unsigned i = 0; i < numArgs; i++) {
		argValues.push_back(UNWRAP_VALUE(args[i]));
	}
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateCall(UNWRAP_FUNCTION(func),
	                                                      argValues, name));
}

JamValueRef JamLLVMBuildPhi(JamBuilderRef builder, JamTypeRef type,
                            const char *name) {
	return WRAP_VALUE(
	    UNWRAP_BUILDER(builder)->CreatePHI(UNWRAP_TYPE(type), 2, name));
}

void JamLLVMAddIncoming(JamValueRef phi, JamValueRef *values,
                        JamBasicBlockRef *blocks, unsigned count) {
	llvm::PHINode *phiNode = llvm::cast<llvm::PHINode>(UNWRAP_VALUE(phi));
	for (unsigned i = 0; i < count; i++) {
		phiNode->addIncoming(UNWRAP_VALUE(values[i]), UNWRAP_BLOCK(blocks[i]));
	}
}

JamValueRef JamLLVMBuildBitCast(JamBuilderRef builder, JamValueRef val,
                                JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateBitCast(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildIntCast(JamBuilderRef builder, JamValueRef val,
                                JamTypeRef destType, bool isSigned,
                                const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateIntCast(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), isSigned, name));
}

JamValueRef JamLLVMBuildPtrToInt(JamBuilderRef builder, JamValueRef val,
                                  JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreatePtrToInt(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildIntToPtr(JamBuilderRef builder, JamValueRef val,
                                  JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateIntToPtr(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMFunctionAsValue(JamFunctionRef func) {
	return WRAP_VALUE(static_cast<llvm::Value *>(UNWRAP_FUNCTION(func)));
}

JamValueRef JamLLVMBuildSIToFP(JamBuilderRef builder, JamValueRef val,
                               JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateSIToFP(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildUIToFP(JamBuilderRef builder, JamValueRef val,
                               JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateUIToFP(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildFPCast(JamBuilderRef builder, JamValueRef val,
                               JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFPCast(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildFPToSI(JamBuilderRef builder, JamValueRef val,
                               JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFPToSI(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildFPToUI(JamBuilderRef builder, JamValueRef val,
                               JamTypeRef destType, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFPToUI(
	    UNWRAP_VALUE(val), UNWRAP_TYPE(destType), name));
}

JamValueRef JamLLVMBuildFAdd(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFAdd(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildFSub(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFSub(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildFMul(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFMul(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildFDiv(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFDiv(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildFRem(JamBuilderRef builder, JamValueRef lhs,
                             JamValueRef rhs, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFRem(
	    UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildFNeg(JamBuilderRef builder, JamValueRef operand,
                             const char *name) {
	return WRAP_VALUE(
	    UNWRAP_BUILDER(builder)->CreateFNeg(UNWRAP_VALUE(operand), name));
}

JamValueRef JamLLVMBuildFCmp(JamBuilderRef builder, JamFloatPredicate pred,
                             JamValueRef lhs, JamValueRef rhs,
                             const char *name) {
	llvm::CmpInst::Predicate llvmPred;
	switch (pred) {
	case JAM_FCMP_FALSE:
		llvmPred = llvm::CmpInst::FCMP_FALSE;
		break;
	case JAM_FCMP_OEQ:
		llvmPred = llvm::CmpInst::FCMP_OEQ;
		break;
	case JAM_FCMP_OGT:
		llvmPred = llvm::CmpInst::FCMP_OGT;
		break;
	case JAM_FCMP_OGE:
		llvmPred = llvm::CmpInst::FCMP_OGE;
		break;
	case JAM_FCMP_OLT:
		llvmPred = llvm::CmpInst::FCMP_OLT;
		break;
	case JAM_FCMP_OLE:
		llvmPred = llvm::CmpInst::FCMP_OLE;
		break;
	case JAM_FCMP_ONE:
		llvmPred = llvm::CmpInst::FCMP_ONE;
		break;
	case JAM_FCMP_ORD:
		llvmPred = llvm::CmpInst::FCMP_ORD;
		break;
	case JAM_FCMP_UNO:
		llvmPred = llvm::CmpInst::FCMP_UNO;
		break;
	case JAM_FCMP_UEQ:
		llvmPred = llvm::CmpInst::FCMP_UEQ;
		break;
	case JAM_FCMP_UGT:
		llvmPred = llvm::CmpInst::FCMP_UGT;
		break;
	case JAM_FCMP_UGE:
		llvmPred = llvm::CmpInst::FCMP_UGE;
		break;
	case JAM_FCMP_ULT:
		llvmPred = llvm::CmpInst::FCMP_ULT;
		break;
	case JAM_FCMP_ULE:
		llvmPred = llvm::CmpInst::FCMP_ULE;
		break;
	case JAM_FCMP_UNE:
		llvmPred = llvm::CmpInst::FCMP_UNE;
		break;
	case JAM_FCMP_TRUE:
		llvmPred = llvm::CmpInst::FCMP_TRUE;
		break;
	default:
		llvmPred = llvm::CmpInst::FCMP_OEQ;
		break;
	}
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateFCmp(
	    llvmPred, UNWRAP_VALUE(lhs), UNWRAP_VALUE(rhs), name));
}

JamValueRef JamLLVMBuildInsertValue(JamBuilderRef builder, JamValueRef agg,
                                    JamValueRef val, unsigned index,
                                    const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateInsertValue(
	    UNWRAP_VALUE(agg), UNWRAP_VALUE(val), index, name));
}

JamValueRef JamLLVMBuildExtractValue(JamBuilderRef builder, JamValueRef agg,
                                     unsigned index, const char *name) {
	return WRAP_VALUE(UNWRAP_BUILDER(builder)->CreateExtractValue(
	    UNWRAP_VALUE(agg), index, name));
}

JamTypeRef JamLLVMTypeOf(JamValueRef val) {
	return WRAP_TYPE(UNWRAP_VALUE(val)->getType());
}

JamTypeRef JamLLVMGetAllocatedType(JamValueRef alloca) {
	return WRAP_TYPE(
	    llvm::cast<llvm::AllocaInst>(UNWRAP_VALUE(alloca))->getAllocatedType());
}

char *JamLLVMGetDefaultTargetTriple(void) {
	return strdup(llvm::sys::getDefaultTargetTriple().c_str());
}

char *JamLLVMGetHostCPUName(void) {
	return strdup(llvm::sys::getHostCPUName().str().c_str());
}

char *JamLLVMGetHostCPUFeatures(void) {
	std::string features;
	auto hostFeatures = llvm::sys::getHostCPUFeatures();
	for (auto &f : hostFeatures) {
		if (!features.empty()) features += ",";
		features += (f.second ? "+" : "-");
		features += f.first().str();
	}
	return strdup(features.c_str());
}

JamTargetMachineRef
JamLLVMCreateTargetMachine(const char *triple, const char *cpu,
                           const char *features, bool isRelocationPIC,
                           JamOptLevel optLevel, JamLTO lto) {
	std::string error;
	const llvm::Target *target =
	    llvm::TargetRegistry::lookupTarget(triple, error);
	if (!target) { return nullptr; }

	llvm::TargetOptions opt;
	// Emit each function/global into its own section so the linker can drop
	// the unreferenced ones at link time (-Wl,--gc-sections on ELF,
	// -Wl,-dead_strip on Mach-O). Debug builds skip this — the extra section
	// table entries cost compile time we don't want to pay there.
	if (optLevel != JAM_OPT_NONE) {
		opt.FunctionSections = true;
		opt.DataSections = true;
	}
	auto rm = isRelocationPIC ? llvm::Reloc::PIC_ : llvm::Reloc::Static;

	llvm::CodeGenOptLevel cgOpt;
	switch (optLevel) {
	case JAM_OPT_NONE:
		cgOpt = llvm::CodeGenOptLevel::None;
		break;
	case JAM_OPT_LESS:
		cgOpt = llvm::CodeGenOptLevel::Less;
		break;
	case JAM_OPT_DEFAULT:
		cgOpt = llvm::CodeGenOptLevel::Default;
		break;
	case JAM_OPT_AGGRESSIVE:
		cgOpt = llvm::CodeGenOptLevel::Aggressive;
		break;
	// Codegen-level has no "size" tier — size optimization happens in
	// the IR pipeline and via function attributes, while codegen stays
	// at Aggressive for the actual machine emission.
	case JAM_OPT_SIZE:
		cgOpt = llvm::CodeGenOptLevel::Aggressive;
		break;
	case JAM_OPT_SMALL:
		cgOpt = llvm::CodeGenOptLevel::Aggressive;
		break;
	default:
		cgOpt = llvm::CodeGenOptLevel::None;
		break;
	}

	llvm::TargetMachine *tm = target->createTargetMachine(
	    triple, cpu ? cpu : "generic", features ? features : "", opt, rm,
	    std::nullopt, cgOpt);
	if (!tm) { return nullptr; }

	auto *impl = new JamTargetMachineImpl{tm, optLevel, lto};
	return WRAP_TARGET_MACHINE(impl);
}

void JamLLVMDisposeTargetMachine(JamTargetMachineRef tm) {
	JamTargetMachineImpl *impl = UNWRAP_TARGET_MACHINE_IMPL(tm);
	delete impl->tm;
	delete impl;
}

bool JamLLVMEmitObjectFile(JamModuleRef mod, JamTargetMachineRef tm,
                           const char *filename, char **errorMessage) {
	std::error_code ec;
	llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);

	if (ec) {
		if (errorMessage) { *errorMessage = strdup(ec.message().c_str()); }
		return false;
	}

	llvm::Module *M = UNWRAP_MODULE(mod);
	JamTargetMachineImpl *impl = UNWRAP_TARGET_MACHINE_IMPL(tm);
	llvm::TargetMachine *targetMachine = impl->tm;
	const JamOptLevel optLevel = impl->optLevel;
	const JamLTO lto = impl->lto;
	const bool isDebug = (optLevel == JAM_OPT_NONE);

	// Size-optimized modes: stamp size-favoring attrs on every Jam-defined
	// function so the inliner and the rest of the pipeline favor code size.
	// Mirrors rustc's behavior: `s` adds optsize, `z` adds both minsize and
	// optsize. Declarations are skipped — those are extern fns whose body
	// lives in another translation unit; we don't get to dictate their
	// inlining policy.
	if (optLevel == JAM_OPT_SIZE || optLevel == JAM_OPT_SMALL) {
		for (llvm::Function &F : *M) {
			if (F.isDeclaration()) continue;
			F.addFnAttr(llvm::Attribute::OptimizeForSize);
			if (optLevel == JAM_OPT_SMALL) {
				F.addFnAttr(llvm::Attribute::MinSize);
			}
		}
	}

	// New-PM module-level optimization pipeline. Without this we'd only run
	// LLVM's codegen passes (instruction selection, register allocation),
	// leaving every IR-level pass — inlining, GVN, mem2reg, SROA, loop opts,
	// vectorization, MergeFunctions, globaldce — disabled. That made
	// `--release` little better than `-O0` for real programs. The
	// configuration below builds a full new-PM optimization pipeline so
	// release builds actually optimize.
	llvm::PipelineTuningOptions pto;
	pto.LoopUnrolling = !isDebug;
	pto.SLPVectorization = !isDebug;
	pto.LoopVectorization = !isDebug;
	pto.LoopInterleaving = !isDebug;
	pto.MergeFunctions = !isDebug;

	llvm::PassInstrumentationCallbacks pic;
	llvm::StandardInstrumentations si(M->getContext(),
	                                  /*DebugLogging=*/false);
	si.registerCallbacks(pic);

	llvm::PassBuilder pb(targetMachine, pto, std::nullopt, &pic);

	llvm::LoopAnalysisManager lam;
	llvm::FunctionAnalysisManager fam;
	llvm::CGSCCAnalysisManager cam;
	llvm::ModuleAnalysisManager mam;

	fam.registerPass([&] { return pb.buildDefaultAAPipeline(); });
	fam.registerPass([&] {
		return llvm::TargetLibraryAnalysis(
		    llvm::TargetLibraryInfoImpl(llvm::Triple(M->getTargetTriple())));
	});

	pb.registerModuleAnalyses(mam);
	pb.registerCGSCCAnalyses(cam);
	pb.registerFunctionAnalyses(fam);
	pb.registerLoopAnalyses(lam);
	pb.crossRegisterProxies(lam, fam, cam, mam);

	llvm::OptimizationLevel level;
	switch (optLevel) {
	case JAM_OPT_NONE:
		level = llvm::OptimizationLevel::O0;
		break;
	case JAM_OPT_LESS:
		level = llvm::OptimizationLevel::O1;
		break;
	case JAM_OPT_DEFAULT:
		level = llvm::OptimizationLevel::O2;
		break;
	case JAM_OPT_AGGRESSIVE:
		level = llvm::OptimizationLevel::O3;
		break;
	case JAM_OPT_SIZE:
		level = llvm::OptimizationLevel::Os;
		break;
	case JAM_OPT_SMALL:
		level = llvm::OptimizationLevel::Oz;
		break;
	default:
		level = llvm::OptimizationLevel::O0;
		break;
	}

	// LTO mode swaps in the LTO pre-link pipeline. The actual cross-module
	// optimization happens at link time inside lld/ld's LTO plugin once it
	// sees this module's bitcode plus any other LTO inputs.
	llvm::ModulePassManager mpm;
	if (level == llvm::OptimizationLevel::O0) {
		mpm = pb.buildO0DefaultPipeline(level);
	} else if (lto != JAM_LTO_OFF) {
		mpm = pb.buildLTOPreLinkDefaultPipeline(level);
	} else {
		mpm = pb.buildPerModuleDefaultPipeline(level);
	}

	mpm.run(*M, mam);

	// LTO modes emit bitcode; the linker will run the rest of the pipeline.
	// Otherwise lower IR → MIR → object via the legacy codegen-PM (LLVM has
	// not migrated codegen to the new PM yet).
	if (lto != JAM_LTO_OFF) {
		llvm::WriteBitcodeToFile(*M, dest);
		dest.close();
		return true;
	}

	llvm::legacy::PassManager pass;
	if (targetMachine->addPassesToEmitFile(pass, dest, nullptr,
	                                       llvm::CodeGenFileType::ObjectFile)) {
		if (errorMessage) {
			*errorMessage = strdup("Target machine cannot emit object file");
		}
		return false;
	}

	pass.run(*M);
	dest.close();
	return true;
}
