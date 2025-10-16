/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef TARGET_H
#define TARGET_H

#include <string>
#include "llvm/TargetParser/Triple.h"

namespace jam {

// CPU Architectures
enum class Arch {
    X86_64,
    AArch64,
    ARM,
    RISCV64,
    Unknown
};

// Operating Systems
enum class OS {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
    Unknown
};

// ABIs (Application Binary Interface)
enum class ABI {
    None,      // No specific ABI (e.g., macOS)
    GNU,       // GNU C library
    MSVC,      // Microsoft Visual C++
    MinGW,     // Minimalist GNU for Windows
    Musl,      // Musl C library
    Unknown
};

// Target triple information
struct Target {
    Arch arch;
    OS os;
    ABI abi;
    
    Target() : arch(Arch::Unknown), os(OS::Unknown), abi(ABI::Unknown) {}
    Target(Arch a, OS o, ABI ab) : arch(a), os(o), abi(ab) {}
    
    // Create target from LLVM triple
    static Target fromLLVMTriple(const llvm::Triple& triple);
    
    // Create target from host system
    static Target getHostTarget();
    
    // Convert to LLVM triple string
    std::string toLLVMTriple() const;
    
    // Get human-readable name
    std::string getName() const;
    
    // Target characteristics
    bool requiresLibC() const;
    bool requiresPIC() const;
    bool requiresPIE() const;
    bool canDynamicLink() const;
    bool usesCabi() const;  // Whether this target uses C ABI
    
    // ABI-specific queries
    const char* getLibCName() const;
    int getPointerSize() const;  // in bytes
    int getPointerAlignment() const;  // in bytes
    
    // Calling convention
    enum class CallingConvention {
        C,        // Standard C calling convention
        Fast,     // Fast calling convention
        Cold,     // Cold calling convention
        Preserve, // Preserve all registers
    };
    
    CallingConvention getDefaultCC() const;
};

// Common target configurations
namespace targets {
    inline Target x86_64_linux_gnu()   { return Target(Arch::X86_64, OS::Linux, ABI::GNU); }
    inline Target x86_64_linux_musl()  { return Target(Arch::X86_64, OS::Linux, ABI::Musl); }
    inline Target x86_64_macos()       { return Target(Arch::X86_64, OS::MacOS, ABI::None); }
    inline Target x86_64_windows_gnu() { return Target(Arch::X86_64, OS::Windows, ABI::MinGW); }
    inline Target x86_64_windows_msvc(){ return Target(Arch::X86_64, OS::Windows, ABI::MSVC); }
    inline Target aarch64_linux_gnu()  { return Target(Arch::AArch64, OS::Linux, ABI::GNU); }
    inline Target aarch64_macos()      { return Target(Arch::AArch64, OS::MacOS, ABI::None); }
}

} // namespace jam

#endif // TARGET_H
