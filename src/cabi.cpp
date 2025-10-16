/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "cabi.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Attributes.h"

namespace jam {

llvm::CallingConv::ID CAbi::getCallingConvention() const {
    // Most platforms use C calling convention
    // We could specialize this per platform if needed
    switch (target_.os) {
        case OS::Windows:
            // Windows x64 uses Microsoft x64 calling convention
            if (target_.abi == ABI::MSVC) {
                return llvm::CallingConv::Win64;
            }
            // MinGW uses System V AMD64 ABI even on Windows
            return llvm::CallingConv::C;
        default:
            return llvm::CallingConv::C;
    }
}

void CAbi::applyFunctionAttributes(llvm::Function* func) const {
    // Set the calling convention
    func->setCallingConv(getCallingConvention());
    
    // On some platforms, we need to ensure proper name visibility
    if (target_.os == OS::Windows && target_.abi == ABI::MSVC) {
        func->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    }
}

bool CAbi::isCAbiCompatible(llvm::Type* type) const {
    if (!type) return false;
    
    // Primitive types are always C ABI compatible
    if (type->isIntegerTy() || type->isFloatingPointTy() || type->isPointerTy()) {
        return true;
    }
    
    // Void is compatible
    if (type->isVoidTy()) {
        return true;
    }
    
    // Structs can be C ABI compatible if they're POD-like
    if (type->isStructTy()) {
        llvm::StructType* structTy = llvm::cast<llvm::StructType>(type);
        // For now, we consider all structs potentially compatible
        // In the future, we might check layout, padding, etc.
        return true;
    }
    
    // Arrays are C ABI compatible
    if (type->isArrayTy()) {
        llvm::ArrayType* arrayTy = llvm::cast<llvm::ArrayType>(type);
        return isCAbiCompatible(arrayTy->getElementType());
    }
    
    // Other types (functions, vectors, etc.) may need special handling
    return false;
}

llvm::Type* CAbi::toCAbiType(llvm::Type* jamType, llvm::LLVMContext& context) const {
    if (!jamType) return nullptr;
    
    // Most Jam types map directly to C types
    // In the future, we might need to transform:
    // - Jam slices -> pointer + length (passed separately or as struct)
    // - Jam strings -> C strings (null-terminated)
    // - Other Jam-specific types
    
    // For now, pass through as-is
    return jamType;
}

std::string CAbi::getExternName(const std::string& jamName) const {
    return platformMangle(jamName);
}

std::string CAbi::platformMangle(const std::string& name) const {
    // Most platforms use the name as-is for C functions
    // Windows MSVC might add decorations, but we'll keep it simple
    
    switch (target_.os) {
        case OS::Windows:
            if (target_.abi == ABI::MSVC) {
                // MSVC may add decorations for __stdcall, __fastcall, etc.
                // For __cdecl (C calling convention), no decoration
                return name;
            }
            return name;
        
        case OS::MacOS:
            // macOS traditionally adds an underscore prefix
            // But modern macOS doesn't require it for C functions
            return name;
        
        default:
            // Linux and most Unix systems use the name as-is
            return name;
    }
}

void CAbi::setExternLinkage(llvm::Function* func) const {
    // External functions should have external linkage
    func->setLinkage(llvm::GlobalValue::ExternalLinkage);
    
    // Set visibility to default (public)
    func->setVisibility(llvm::GlobalValue::DefaultVisibility);
}

llvm::Function* CAbi::createExternFunction(
    llvm::Module* module,
    const std::string& name,
    llvm::FunctionType* funcType
) const {
    // Create the function declaration
    llvm::Function* func = llvm::Function::Create(
        funcType,
        llvm::GlobalValue::ExternalLinkage,
        getExternName(name),
        module
    );
    
    // Apply C ABI attributes
    applyFunctionAttributes(func);
    
    return func;
}

void FunctionAbi::applyToFunction(llvm::Function* func, const CAbi& cabi) const {
    // Set linkage
    if (isExtern) {
        cabi.setExternLinkage(func);
    } else if (isExport) {
        func->setLinkage(llvm::GlobalValue::ExternalLinkage);
        func->setVisibility(llvm::GlobalValue::DefaultVisibility);
    } else {
        // Internal linkage for non-exported functions
        func->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
    
    // Set calling convention
    switch (callconv) {
        case CallConv::C:
            func->setCallingConv(cabi.getCallingConvention());
            break;
        case CallConv::Fast:
            func->setCallingConv(llvm::CallingConv::Fast);
            break;
        case CallConv::Cold:
            func->setCallingConv(llvm::CallingConv::Cold);
            break;
        case CallConv::Jam:
        default:
            // Use default/optimized calling convention
            func->setCallingConv(llvm::CallingConv::C);
            break;
    }
    
    // Set attributes
    if (isInline) {
        func->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    
    if (noReturn) {
        func->addFnAttr(llvm::Attribute::NoReturn);
    }
}

} // namespace jam
