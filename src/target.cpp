/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "target.h"
#include "llvm/TargetParser/Host.h"
#include <sstream>

namespace jam {

Target Target::fromLLVMTriple(const llvm::Triple& triple) {
    Target target;
    
    // Parse architecture
    switch (triple.getArch()) {
        case llvm::Triple::x86_64:
            target.arch = Arch::X86_64;
            break;
        case llvm::Triple::aarch64:
        case llvm::Triple::aarch64_be:
            target.arch = Arch::AArch64;
            break;
        case llvm::Triple::arm:
        case llvm::Triple::armeb:
        case llvm::Triple::thumb:
        case llvm::Triple::thumbeb:
            target.arch = Arch::ARM;
            break;
        case llvm::Triple::riscv64:
            target.arch = Arch::RISCV64;
            break;
        default:
            target.arch = Arch::Unknown;
    }
    
    // Parse OS
    switch (triple.getOS()) {
        case llvm::Triple::Linux:
            target.os = OS::Linux;
            break;
        case llvm::Triple::Darwin:
        case llvm::Triple::MacOSX:
            target.os = OS::MacOS;
            break;
        case llvm::Triple::Win32:
            target.os = OS::Windows;
            break;
        case llvm::Triple::FreeBSD:
            target.os = OS::FreeBSD;
            break;
        default:
            target.os = OS::Unknown;
    }
    
    // Parse ABI
    switch (triple.getEnvironment()) {
        case llvm::Triple::GNU:
        case llvm::Triple::GNUEABI:
        case llvm::Triple::GNUEABIHF:
        case llvm::Triple::GNUX32:
            target.abi = ABI::GNU;
            break;
        case llvm::Triple::MSVC:
            target.abi = ABI::MSVC;
            break;
        case llvm::Triple::Musl:
        case llvm::Triple::MuslEABI:
        case llvm::Triple::MuslEABIHF:
            target.abi = ABI::Musl;
            break;
        default:
            if (target.os == OS::Windows && triple.getEnvironment() == llvm::Triple::UnknownEnvironment) {
                target.abi = ABI::MinGW;
            } else if (target.os == OS::MacOS) {
                target.abi = ABI::None;
            } else {
                target.abi = ABI::Unknown;
            }
    }
    
    return target;
}

Target Target::getHostTarget() {
    std::string tripleStr = llvm::sys::getDefaultTargetTriple();
    llvm::Triple triple(tripleStr);
    return fromLLVMTriple(triple);
}

std::string Target::toLLVMTriple() const {
    std::ostringstream oss;
    
    // Architecture
    switch (arch) {
        case Arch::X86_64:   oss << "x86_64"; break;
        case Arch::AArch64:  oss << "aarch64"; break;
        case Arch::ARM:      oss << "arm"; break;
        case Arch::RISCV64:  oss << "riscv64"; break;
        default:             oss << "unknown"; break;
    }
    
    oss << "-";
    
    // Vendor (usually unknown for our purposes)
    oss << "unknown-";
    
    // OS
    switch (os) {
        case OS::Linux:    oss << "linux"; break;
        case OS::MacOS:    oss << "darwin"; break;
        case OS::Windows:  oss << "windows"; break;
        case OS::FreeBSD:  oss << "freebsd"; break;
        default:           oss << "unknown"; break;
    }
    
    // ABI (environment)
    switch (abi) {
        case ABI::GNU:     oss << "-gnu"; break;
        case ABI::MSVC:    oss << "-msvc"; break;
        case ABI::MinGW:   oss << "-gnu"; break;
        case ABI::Musl:    oss << "-musl"; break;
        case ABI::None:    break; // No suffix
        default:           break;
    }
    
    return oss.str();
}

std::string Target::getName() const {
    std::ostringstream oss;
    
    switch (arch) {
        case Arch::X86_64:   oss << "x86_64"; break;
        case Arch::AArch64:  oss << "aarch64"; break;
        case Arch::ARM:      oss << "arm"; break;
        case Arch::RISCV64:  oss << "riscv64"; break;
        default:             oss << "unknown"; break;
    }
    
    oss << "-";
    
    switch (os) {
        case OS::Linux:    oss << "linux"; break;
        case OS::MacOS:    oss << "macos"; break;
        case OS::Windows:  oss << "windows"; break;
        case OS::FreeBSD:  oss << "freebsd"; break;
        default:           oss << "unknown"; break;
    }
    
    if (abi != ABI::None && abi != ABI::Unknown) {
        oss << "-";
        switch (abi) {
            case ABI::GNU:     oss << "gnu"; break;
            case ABI::MSVC:    oss << "msvc"; break;
            case ABI::MinGW:   oss << "mingw"; break;
            case ABI::Musl:    oss << "musl"; break;
            default:           break;
        }
    }
    
    return oss.str();
}

bool Target::requiresLibC() const {
    // Darwin (macOS) always links libSystem which contains libc
    // FreeBSD and some others require libc for stable syscall interface
    return os == OS::MacOS || os == OS::FreeBSD;
}

bool Target::requiresPIC() const {
    // Position Independent Code is required on:
    // - Windows (DLLs)
    // - When using GNU libc on most platforms
    return os == OS::Windows || abi == ABI::GNU;
}

bool Target::requiresPIE() const {
    // Position Independent Executable is required on:
    // - macOS/Darwin
    // - OpenBSD (not in our enum yet but can be added)
    return os == OS::MacOS;
}

bool Target::canDynamicLink() const {
    // Most targets support dynamic linking
    // Freestanding and some embedded targets don't
    return true;
}

bool Target::usesCabi() const {
    // All our targets use C ABI for external function calls
    // This could be false for pure Jam-to-Jam calls in the future
    return true;
}

const char* Target::getLibCName() const {
    switch (os) {
        case OS::Windows:
            return "mingw";
        case OS::MacOS:
            return "darwin";
        default:
            break;
    }
    
    switch (abi) {
        case ABI::GNU:
        case ABI::MinGW:
            return "glibc";
        case ABI::Musl:
            return "musl";
        case ABI::MSVC:
            return "msvcrt";
        default:
            return "unknown";
    }
}

int Target::getPointerSize() const {
    switch (arch) {
        case Arch::X86_64:
        case Arch::AArch64:
        case Arch::RISCV64:
            return 8;  // 64-bit
        case Arch::ARM:
            return 4;  // 32-bit
        default:
            return 8;  // Default to 64-bit
    }
}

int Target::getPointerAlignment() const {
    // Pointer alignment is typically the same as pointer size
    return getPointerSize();
}

Target::CallingConvention Target::getDefaultCC() const {
    // Most platforms use C calling convention by default
    // We could optimize with Fast for internal functions later
    return CallingConvention::C;
}

} // namespace jam
