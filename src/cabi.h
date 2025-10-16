/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef CABI_H
#define CABI_H

#include "target.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include <string>
#include <vector>

namespace jam {

// C ABI helper for extern "C" function calling
class CAbi {
public:
    explicit CAbi(const Target& target) : target_(target) {}
    
    // Get the appropriate LLVM calling convention for the target
    llvm::CallingConv::ID getCallingConvention() const;
    
    // Apply C ABI attributes to a function
    void applyFunctionAttributes(llvm::Function* func) const;
    
    // Check if a type is compatible with C ABI
    bool isCAbiCompatible(llvm::Type* type) const;
    
    // Transform a Jam type to its C ABI equivalent
    llvm::Type* toCAbiType(llvm::Type* jamType, llvm::LLVMContext& context) const;
    
    // Get the mangled name for an extern function
    std::string getExternName(const std::string& jamName) const;
    
    // Apply proper linkage for extern functions
    void setExternLinkage(llvm::Function* func) const;
    
    // Create an extern function declaration
    llvm::Function* createExternFunction(
        llvm::Module* module,
        const std::string& name,
        llvm::FunctionType* funcType
    ) const;
    
private:
    Target target_;
    
    // Platform-specific name mangling
    std::string platformMangle(const std::string& name) const;
};

// Annotation for Jam functions to specify calling convention
enum class CallConv {
    Jam,      // Default Jam calling convention (may be optimized)
    C,        // C calling convention (for interop)
    Fast,     // Optimized for speed (LLVM Fast CC)
    Cold,     // Optimized for code size (LLVM Cold CC)
};

// Function attributes for C interop
struct FunctionAbi {
    CallConv callconv = CallConv::Jam;
    bool isExtern = false;        // Is this an external C function?
    bool isExport = false;        // Should this be exported for C?
    bool isInline = false;        // Inline hint
    bool noReturn = false;        // Function never returns
    std::string externName;       // Custom external name
    
    // Apply to LLVM function
    void applyToFunction(llvm::Function* func, const CAbi& cabi) const;
};

} // namespace jam

#endif // CABI_H
