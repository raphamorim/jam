/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 *
 * The single piece of C++ in jam_llvm. It exists for two reasons the LLVM-C
 * API does not cover:
 *
 *   1. a handful of codegen knobs live on llvm::TargetOptions, which the
 *      LLVM-C API does not expose (there is no LLVMTargetOptionsRef); and
 *   2. the optimization pipeline is reproduced VERBATIM from the C++ facade
 *      (src/jam_llvm.cpp) so that the new-PM PassBuilder configuration — the
 *      tuning options, the analysis-manager wiring, the OptimizationLevel
 *      switch, the pre-pipeline internalize+globaldce, and the pipeline
 *      selection — is provably byte-identical to the C++ oracle's. Running the
 *      same pipeline on the same LLVM gives byte-identical optimized IR.
 *
 * This mirrors rustc's `rustllvm` shim: bind the C API directly for everything,
 * and reach into C++ only for the gaps. Keep it tiny.
 */

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <optional>

// Mirror the optimization-level discriminants of the Rust `OptLevel` enum
// (crates/jam-llvm/src/target.rs): None,Less,Default,Aggressive,Size,Small ->
// 0,1,2,3,4,5. The C++ facade's JamOptLevel enum has the same ordering, so the
// switch below matches src/jam_llvm.cpp exactly.
enum {
	JAM_OPT_NONE = 0,
	JAM_OPT_LESS = 1,
	JAM_OPT_DEFAULT = 2,
	JAM_OPT_AGGRESSIVE = 3,
	JAM_OPT_SIZE = 4,
	JAM_OPT_SMALL = 5,
};

// Mirror the Rust `Lto` enum: Off,Thin,Fat -> 0,1,2. Only "off vs not-off"
// matters to the pipeline selection below.
enum {
	JAM_LTO_OFF = 0,
};

// Enable per-function / per-data sections on an already-created target machine
// so the linker can dead-strip at function granularity (-ffunction-sections /
// -fdata-sections). `LLVMTargetMachineRef` is LLVM's simple-conversion wrapper
// around `TargetMachine*`, so this reinterpret_cast is exactly what the C API's
// own `unwrap()` does. `Options` is a public, mutable member read by the
// AsmPrinter at emit time, so setting it post-creation takes effect.
extern "C" void jam_set_target_machine_sections(LLVMTargetMachineRef TM,
                                                int FunctionSections,
                                                int DataSections) {
	auto *tm = reinterpret_cast<llvm::TargetMachine *>(TM);
	tm->Options.FunctionSections = FunctionSections != 0;
	tm->Options.DataSections = DataSections != 0;
}

// Run the new-PM module optimization pipeline against `M`, configured exactly
// like the C++ facade `JamLLVMEmitObjectFile` (src/jam_llvm.cpp lines
// ~1190-1300). Copied VERBATIM from there so the pass pipeline is provably
// identical; the only edits are the reinterpret_cast unwraps (the C++ used the
// facade's UNWRAP macros) and the JamOptLevel/JamLTO constants reproduced
// above. Reaching the C++ API requires the C++ classes, which live in the
// libLLVM the crate already links.
extern "C" void jam_shim_optimize(LLVMModuleRef M, LLVMTargetMachineRef TM,
                                  int optLevel, int isDebug, int lto) {
	auto *m = reinterpret_cast<llvm::Module *>(M);
	auto *targetMachine = reinterpret_cast<llvm::TargetMachine *>(TM);

	// Size-optimized modes: stamp size-favoring attrs on every Jam-defined
	// function so the inliner and the rest of the pipeline favor code size.
	// Mirrors rustc's behavior: `s` adds optsize, `z` adds both minsize and
	// optsize. Declarations are skipped — those are extern fns whose body
	// lives in another translation unit; we don't get to dictate their
	// inlining policy.
	if (optLevel == JAM_OPT_SIZE || optLevel == JAM_OPT_SMALL) {
		for (llvm::Function &F : *m) {
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
	llvm::StandardInstrumentations si(m->getContext(),
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
		    llvm::TargetLibraryInfoImpl(llvm::Triple(m->getTargetTriple())));
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

	// Internalize + strip dead code BEFORE the optimization pipeline.
	// Every loaded module's every function is emitted eagerly, so a
	// typical program carries unused std (and project) functions that
	// the O2/O3 pipeline would otherwise fully optimize — and ISel /
	// register allocation would lower — only for the linker's
	// dead-strip to discard the result. `main` is preserved by the
	// predicate; `export` fns sit in llvm.used (see jirDeclarePrototype
	// via JamLLVMAppendToUsed), which InternalizePass always respects;
	// declarations are untouched. Skipped under LTO, where the link-time
	// pipeline owns whole-program internalization.
	if (level != llvm::OptimizationLevel::O0 && lto == JAM_LTO_OFF) {
		llvm::ModulePassManager pre;
		pre.addPass(llvm::InternalizePass([](const llvm::GlobalValue &gv) {
			return gv.getName() == "main";
		}));
		pre.addPass(llvm::GlobalDCEPass());
		pre.run(*m, mam);
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

	mpm.run(*m, mam);
}
