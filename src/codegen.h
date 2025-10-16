/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef CODEGEN_H
#define CODEGEN_H

#include <string>
#include "llvm/IR/Type.h"
#include "llvm/IR/LLVMContext.h"

// Helper function to get LLVM type from type string
llvm::Type* getTypeFromString(const std::string& typeStr, llvm::LLVMContext& context);

#endif // CODEGEN_H
