/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "ast.h"

#include "abi.h"
#include "codegen.h"
#include "jam_llvm.h"
#include "mangling.h"

#include <string>
#include <vector>

// Emits the LLVM prototype for a `FunctionAST` — signature + linkage +
// ABI attributes, no body. Run before any caller is lowered so forward
// references between functions resolve in a single pass.
//
// The corresponding body is lowered through the AstGen → JIR → LLVM
// pipeline (`astgenFunction` + `jirDefineBody`). Splitting the two
// phases is what lets the file walk emit declarations top-down before
// any body needs a callee that's still further down the file.
//
// `drop` methods on top-level structs get the `__drop_<TypeName>`
// mangling here (shared with the JIR Call path via `mangledFunctionName`)
// so a struct's drop fn doesn't collide with other types' drop fns at
// the LLVM symbol level.
JamFunctionRef FunctionAST::declarePrototype(JamCodegenContext &ctx) {
	std::string funcName =
	    mangledFunctionName(*this, ctx.getTypePool(), ctx.getStringPool());

	// return ABI: large aggregates returned by Jam-defined fns
	// (not extern, not test) are sret — caller passes a leading
	// `ptr sret(%T) align A` arg, callee stores into it and returns
	// void. Small returns stay direct.
	jam::abi::ReturnABI rabi =
	    (isTest || isExtern)
	        ? jam::abi::ReturnABI{jam::abi::ReturnABI::Kind::Direct,
	                              (isTest || ReturnType == kNoType)
	                                  ? ctx.getVoidType()
	                                  : ctx.getLLVMType(ReturnType),
	                              0}
	        : jam::abi::classifyReturn(ReturnType, ctx);

	std::vector<JamTypeRef> ArgTypes;
	if (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
		// sret slot leads the LLVM arg list.
		ArgTypes.push_back(JamLLVMPointerType(ctx.getLLVMType(ReturnType), 0));
	}
	if (!isTest) {
		for (const auto &arg : Args) {
			if (isExtern) {
				// extern fns follow the C ABI literally. The user
				// already wrote the parameter types as they want them
				// to appear at the FFI boundary (e.g. `*const T` for a
				// pointer arg, `u32` for a scalar). LLVM's backend
				// handles `byval` for large aggregates per the
				// platform's MEMORY classification; we do NOT
				// re-classify with mode-aware rules.
				ArgTypes.push_back(ctx.getLLVMType(arg.Type));
				continue;
			}
			// mode-aware ABI. classifyParam decides per-(mode, type)
			// whether the parameter is passed by value (the type's natural
			// LLVM representation) or by pointer.
			jam::abi::ParamABI pabi =
			    jam::abi::classifyParam(arg.Mode, arg.Type, ctx);
			if (pabi.kind == jam::abi::ParamABI::Kind::ByPointer) {
				ArgTypes.push_back(
				    JamLLVMPointerType(ctx.getLLVMType(arg.Type), 0));
			} else {
				ArgTypes.push_back(pabi.llvmType);
			}
		}
	}

	JamTypeRef RetType = (rabi.kind == jam::abi::ReturnABI::Kind::Indirect)
	                         ? ctx.getVoidType()
	                         : rabi.directType;

	JamTypeRef FT = JamLLVMFunctionType(RetType, ArgTypes.data(),
	                                    ArgTypes.size(), isVarArgs);

	JamFunctionRef F =
	    JamLLVMAddFunction(ctx.getModule(), funcName.c_str(), FT);
	JamLLVMApplyDefaultFnAttrs(F, isExtern);
	if (ReturnType == BuiltinType::NoReturn) {
		JamLLVMSetFunctionNoReturn(F);
	}

	// Apply sret attributes to the leading parameter when applicable.
	if (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
		JamLLVMAddParamAttrSret(F, 0, ctx.getLLVMType(ReturnType),
		                        rabi.sretAlign);
	}

	if (isExtern || isExport || Name == "main") {
		JamLLVMSetLinkage((JamValueRef)F, JAM_LINKAGE_EXTERNAL);
	} else {
		JamLLVMSetLinkage((JamValueRef)F, JAM_LINKAGE_INTERNAL);
	}

	// when the function uses sret, the user's parameter at source
	// index `i` lives at LLVM index `i + 1` (the sret slot is index 0).
	const unsigned argOffset =
	    (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) ? 1u : 0u;

	if (isExtern || isExport || Name == "main") {
		JamLLVMSetFunctionCallConv(F, JAM_CALLCONV_C);
		for (unsigned i = 0; i < Args.size(); i++) {
			if (Args[i].Type == BuiltinType::Bool) {
				JamLLVMAddParamAttrZeroExt(F, i + argOffset);
			}
		}
		if (ReturnType == BuiltinType::Bool) { JamLLVMAddRetAttrZeroExt(F); }
	}

	for (unsigned i = 0; i < Args.size(); i++) {
		JamValueRef param = JamLLVMGetParam(F, i + argOffset);
		JamLLVMSetValueName(param, Args[i].Name.c_str());
	}

	return F;
}
