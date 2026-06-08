/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef TARGET_H
#define TARGET_H

#include "llvm/TargetParser/Triple.h"
#include <string>

namespace jam {

enum class Arch { X86_64, AArch64, ARM, RISCV64, Unknown };
enum class OS { Linux, MacOS, Windows, FreeBSD, Unknown };

enum class ABI {
	None,   // No specific ABI (e.g., macOS)
	GNU,    // GNU C library
	MSVC,   // Microsoft Visual C++
	MinGW,  // Minimalist GNU for Windows
	Musl,   // Musl C library
	Unknown
};

// target triplet is basically a pattern in gnu build system
// it's : name of the CPU family/model then the vendor and the OS name
struct Target {
	Arch arch;
	OS os;
	ABI abi;

	Target() : arch(Arch::Unknown), os(OS::Unknown), abi(ABI::Unknown) {}
	Target(Arch a, OS o, ABI ab) : arch(a), os(o), abi(ab) {}

	// Create target from LLVM triple
	static Target fromLLVMTriple(const llvm::Triple &triple);

	// Create target from host system
	static Target getHostTarget();

	std::string toLLVMTriple() const;
	std::string getName() const;

	bool requiresLibC() const;
	bool requiresPIC() const;
	bool requiresPIE() const;
	bool canDynamicLink() const;
	bool usesCabi() const;

	const char *getLibCName() const;
	int getPointerSize() const;
	int getPointerAlignment() const;

	// Calling convention
	enum class CallingConvention {
		C,         // Standard C calling convention
		Fast,      // Fast calling convention
		Cold,      // Cold calling convention
		Preserve,  // Preserve all registers
	};

	CallingConvention getDefaultCC() const;
};

}

#endif
