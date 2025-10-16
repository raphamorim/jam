/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "codegen.h"
#include <stdexcept>
#include "llvm/IR/DerivedTypes.h"

llvm::Type* getTypeFromString(const std::string& typeStr, llvm::LLVMContext& context) {
    if (typeStr == "u8" || typeStr == "i8") {
        return llvm::Type::getInt8Ty(context);
    } else if (typeStr == "u16" || typeStr == "i16") {
        return llvm::Type::getInt16Ty(context);
    } else if (typeStr == "u32" || typeStr == "i32") {
        return llvm::Type::getInt32Ty(context);
    } else if (typeStr == "bool") {
        return llvm::Type::getInt1Ty(context);
    } else if (typeStr == "str") {
        // String slice: struct { ptr: *u8, len: usize }
        llvm::Type* i8PtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0);
        llvm::Type* usizeType = llvm::Type::getInt64Ty(context); // usize as i64
        return llvm::StructType::get(context, {i8PtrType, usizeType});
    } else if (typeStr.substr(0, 2) == "[]") {
        // Slice type: []T -> struct { ptr: *T, len: usize }
        std::string elementType = typeStr.substr(2); // Remove "[]"
        llvm::Type* elemType = getTypeFromString(elementType, context);
        llvm::Type* elemPtrType = llvm::PointerType::get(elemType, 0);
        llvm::Type* usizeType = llvm::Type::getInt64Ty(context); // usize as i64
        return llvm::StructType::get(context, {elemPtrType, usizeType});
    }
    throw std::runtime_error("Unknown type: " + typeStr);
}
