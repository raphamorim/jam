/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "ast.h"
#include "codegen.h"
#include <stdexcept>
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/GlobalVariable.h"

// Global variables to track loop context for break/continue
llvm::BasicBlock* CurrentLoopContinue = nullptr;
llvm::BasicBlock* CurrentLoopBreak = nullptr;

llvm::Value* NumberExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    // Choose appropriate type based on value range
    llvm::Type* IntType;
    if (Val >= 0 && Val <= 255) {
        IntType = llvm::Type::getInt8Ty(TheModule->getContext());
    } else if (Val >= -128 && Val < 0) {
        IntType = llvm::Type::getInt8Ty(TheModule->getContext());
    } else if (Val >= 0 && Val <= 65535) {
        IntType = llvm::Type::getInt16Ty(TheModule->getContext());
    } else if (Val >= -32768 && Val < 0) {
        IntType = llvm::Type::getInt16Ty(TheModule->getContext());
    } else if (Val >= 0 && Val <= 4294967295ULL) {
        IntType = llvm::Type::getInt32Ty(TheModule->getContext());
    } else if (Val >= -2147483648LL && Val < 0) {
        IntType = llvm::Type::getInt32Ty(TheModule->getContext());
    } else {
        IntType = llvm::Type::getInt64Ty(TheModule->getContext());
    }
    return llvm::ConstantInt::get(IntType, Val, true);
}

llvm::Value* BooleanExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    return llvm::ConstantInt::get(llvm::Type::getInt1Ty(TheModule->getContext()), Val ? 1 : 0);
}

llvm::Value* StringLiteralExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    // Create a global string constant (null-terminated for C compatibility)
    llvm::Constant* StrConstant = llvm::ConstantDataArray::getString(TheModule->getContext(), Val, true);
    llvm::GlobalVariable* StrGlobal = new llvm::GlobalVariable(
        *TheModule,
        StrConstant->getType(),
        true, // isConstant
        llvm::GlobalValue::PrivateLinkage,
        StrConstant,
        "str"
    );
    
    // Create a string slice struct { ptr: *u8, len: usize }
    llvm::Type* i8PtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
    llvm::Type* usizeType = llvm::Type::getInt64Ty(TheModule->getContext());
    llvm::Type* sliceType = llvm::StructType::get(TheModule->getContext(), {i8PtrType, usizeType});
    
    // Get pointer to the string data
    llvm::Value* StrPtr = Builder.CreateBitCast(StrGlobal, i8PtrType);
    
    // Create the slice struct
    llvm::Value* SliceStruct = llvm::UndefValue::get(sliceType);
    SliceStruct = Builder.CreateInsertValue(SliceStruct, StrPtr, 0); // ptr
    SliceStruct = Builder.CreateInsertValue(SliceStruct, llvm::ConstantInt::get(usizeType, Val.length()), 1); // len
    
    return SliceStruct;
}

llvm::Value* VariableExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Value* V = NamedValues[Name];
    if (!V)
        throw std::runtime_error("Unknown variable name: " + Name);
    
    // Get the type from the allocated value (for newer LLVM versions)
    llvm::AllocaInst* Alloca = llvm::cast<llvm::AllocaInst>(V);
    llvm::Type* LoadType = Alloca->getAllocatedType();
    return Builder.CreateLoad(LoadType, V, Name.c_str());
}

llvm::Value* BinaryExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Value* L = LHS->codegen(Builder, TheModule, NamedValues);
    llvm::Value* R = RHS->codegen(Builder, TheModule, NamedValues);

    if (!L || !R)
        return nullptr;

    if (Op == "+")
        return Builder.CreateAdd(L, R, "addtmp");
    else if (Op == "==")
        return Builder.CreateICmpEQ(L, R, "cmptmp");
    else if (Op == "!=")
        return Builder.CreateICmpNE(L, R, "cmptmp");
    else if (Op == "<")
        return Builder.CreateICmpULT(L, R, "cmptmp");
    else if (Op == "<=")
        return Builder.CreateICmpULE(L, R, "cmptmp");
    else if (Op == ">")
        return Builder.CreateICmpUGT(L, R, "cmptmp");
    else if (Op == ">=")
        return Builder.CreateICmpUGE(L, R, "cmptmp");

    throw std::runtime_error("Invalid binary operator: " + Op);
}

llvm::Value* CallExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    // Handle built-in print functions
    if (Callee == "print" || Callee == "println" || Callee == "printf") {
        return generatePrintCall(Builder, TheModule, NamedValues);
    }
    
    llvm::Function* CalleeF = TheModule->getFunction(Callee);
    if (!CalleeF)
        throw std::runtime_error("Unknown function referenced: " + Callee);

    if (CalleeF->arg_size() != Args.size())
        throw std::runtime_error("Incorrect number of arguments passed");

    std::vector<llvm::Value*> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->codegen(Builder, TheModule, NamedValues));
        if (!ArgsV.back())
            return nullptr;
    }

    return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}

llvm::Value* CallExprAST::generatePrintCall(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    // Declare printf function if not already declared
    llvm::Function* printfFunc = TheModule->getFunction("printf");
    if (!printfFunc) {
        llvm::FunctionType* printfType = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(TheModule->getContext()),
            llvm::PointerType::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0),
            true // varargs
        );
        printfFunc = llvm::Function::Create(
            printfType,
            llvm::Function::ExternalLinkage,
            "printf",
            TheModule
        );
    }
    
    // Declare puts function for simple println
    llvm::Function* putsFunc = TheModule->getFunction("puts");
    if (!putsFunc) {
        llvm::FunctionType* putsType = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(TheModule->getContext()),
            llvm::PointerType::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0),
            false // not varargs
        );
        putsFunc = llvm::Function::Create(
            putsType,
            llvm::Function::ExternalLinkage,
            "puts",
            TheModule
        );
    }
    
    llvm::Value* result = nullptr;
    
    if (Callee == "println" && Args.size() == 1) {
        // Simple println with one string argument - use puts
        llvm::Value* arg = Args[0]->codegen(Builder, TheModule, NamedValues);
        if (!arg) return nullptr;
        
        // If it's a string slice, extract the pointer
        if (arg->getType()->isStructTy()) {
            arg = Builder.CreateExtractValue(arg, 0, "str_ptr");
        }
        
        result = Builder.CreateCall(putsFunc, {arg}, "puts_call");
    } else if (Callee == "print" && Args.size() == 1) {
        // Simple print with one string argument - use printf without newline
        llvm::Value* arg = Args[0]->codegen(Builder, TheModule, NamedValues);
        if (!arg) return nullptr;
        
        // If it's a string slice, extract the pointer
        if (arg->getType()->isStructTy()) {
            arg = Builder.CreateExtractValue(arg, 0, "str_ptr");
        }
        
        // For print, we just print the string directly without format string
        // Create format string "%s" for printf
        llvm::Constant* formatStr = llvm::ConstantDataArray::getString(TheModule->getContext(), "%s", true);
        llvm::GlobalVariable* formatGlobal = new llvm::GlobalVariable(
            *TheModule,
            formatStr->getType(),
            true,
            llvm::GlobalValue::PrivateLinkage,
            formatStr,
            "print_fmt"
        );
        llvm::Value* formatPtr = Builder.CreateBitCast(formatGlobal, llvm::PointerType::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0));
        
        result = Builder.CreateCall(printfFunc, {formatPtr, arg}, "printf_call");
    } else {
        // For now, just handle simple cases
        throw std::runtime_error("Complex print formatting not yet implemented");
    }
    
    // Return the result (printf/puts return int)
    return result;
}

llvm::Value* ReturnExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Value* RetVal = this->RetVal->codegen(Builder, TheModule, NamedValues);
    if (!RetVal)
        return nullptr;

    Builder.CreateRet(RetVal);
    return RetVal;
}

llvm::Value* VarDeclAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Type* VarType = getTypeFromString(Type, TheModule->getContext());
    llvm::AllocaInst* Alloca = Builder.CreateAlloca(VarType, nullptr, Name);
    
    if (Init) {
        llvm::Value* InitVal = Init->codegen(Builder, TheModule, NamedValues);
        if (!InitVal)
            return nullptr;
        Builder.CreateStore(InitVal, Alloca);
    } else {
        // Initialize with zero/null value
        llvm::Value* ZeroVal = llvm::Constant::getNullValue(VarType);
        Builder.CreateStore(ZeroVal, Alloca);
    }

    NamedValues[Name] = Alloca;
    return Alloca;
}

llvm::Value* IfExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Value* CondV = Condition->codegen(Builder, TheModule, NamedValues);
    if (!CondV)
        return nullptr;

    // Convert condition to a bool by comparing non-equal to 0
    CondV = Builder.CreateICmpNE(CondV, llvm::ConstantInt::get(CondV->getType(), 0), "ifcond");

    llvm::Function* TheFunction = Builder.GetInsertBlock()->getParent();

    // Create blocks for the then and else cases. Insert the 'then' block at the end of the function.
    llvm::BasicBlock* ThenBB = llvm::BasicBlock::Create(TheModule->getContext(), "then", TheFunction);
    llvm::BasicBlock* ElseBB = llvm::BasicBlock::Create(TheModule->getContext(), "else", TheFunction);
    llvm::BasicBlock* MergeBB = llvm::BasicBlock::Create(TheModule->getContext(), "ifcont", TheFunction);

    Builder.CreateCondBr(CondV, ThenBB, ElseBB);

    // Emit then value.
    Builder.SetInsertPoint(ThenBB);
    llvm::Value* ThenV = nullptr;
    for (auto& Expr : ThenBody) {
        ThenV = Expr->codegen(Builder, TheModule, NamedValues);
    }
    // Only create branch if the block doesn't already have a terminator (like return)
    if (!Builder.GetInsertBlock()->getTerminator()) {
        Builder.CreateBr(MergeBB);
    }
    // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
    ThenBB = Builder.GetInsertBlock();

    // Emit else block.
    Builder.SetInsertPoint(ElseBB);
    llvm::Value* ElseV = nullptr;
    for (auto& Expr : ElseBody) {
        ElseV = Expr->codegen(Builder, TheModule, NamedValues);
    }
    // Only create branch if the block doesn't already have a terminator (like return)
    if (!Builder.GetInsertBlock()->getTerminator()) {
        Builder.CreateBr(MergeBB);
    }
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = Builder.GetInsertBlock();

    // Emit merge block.
    Builder.SetInsertPoint(MergeBB);

    // For now, if statements don't return values, so we return a dummy value
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
}

llvm::Value* WhileExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Function* TheFunction = Builder.GetInsertBlock()->getParent();
    
    // Create blocks for the loop
    llvm::BasicBlock* CondBB = llvm::BasicBlock::Create(TheModule->getContext(), "whilecond", TheFunction);
    llvm::BasicBlock* LoopBB = llvm::BasicBlock::Create(TheModule->getContext(), "whileloop", TheFunction);
    llvm::BasicBlock* AfterBB = llvm::BasicBlock::Create(TheModule->getContext(), "afterloop", TheFunction);
    
    // Save previous loop context
    llvm::BasicBlock* PrevContinue = CurrentLoopContinue;
    llvm::BasicBlock* PrevBreak = CurrentLoopBreak;
    CurrentLoopContinue = CondBB;
    CurrentLoopBreak = AfterBB;
    
    // Jump to condition block
    Builder.CreateBr(CondBB);
    
    // Emit condition block
    Builder.SetInsertPoint(CondBB);
    llvm::Value* CondV = Condition->codegen(Builder, TheModule, NamedValues);
    if (!CondV) {
        // Restore previous loop context
        CurrentLoopContinue = PrevContinue;
        CurrentLoopBreak = PrevBreak;
        return nullptr;
    }
    
    // Convert condition to a bool by comparing non-equal to 0
    CondV = Builder.CreateICmpNE(CondV, llvm::ConstantInt::get(CondV->getType(), 0), "whilecond");
    Builder.CreateCondBr(CondV, LoopBB, AfterBB);
    
    // Emit loop body
    Builder.SetInsertPoint(LoopBB);
    for (auto& Expr : Body) {
        Expr->codegen(Builder, TheModule, NamedValues);
    }
    
    // Only create branch if the block doesn't already have a terminator
    if (!Builder.GetInsertBlock()->getTerminator()) {
        Builder.CreateBr(CondBB);
    }
    
    // Emit after block
    Builder.SetInsertPoint(AfterBB);
    
    // Restore previous loop context
    CurrentLoopContinue = PrevContinue;
    CurrentLoopBreak = PrevBreak;
    
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
}

llvm::Value* ForExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    llvm::Function* TheFunction = Builder.GetInsertBlock()->getParent();
    
    // Compute start and end values first
    llvm::Value* StartVal = Start->codegen(Builder, TheModule, NamedValues);
    llvm::Value* EndVal = End->codegen(Builder, TheModule, NamedValues);
    if (!StartVal || !EndVal)
        return nullptr;
    
    // Use the type of the start value for the loop variable
    llvm::Type* VarType = StartVal->getType();
    
    // Convert end value to match start value type if needed
    if (EndVal->getType() != VarType) {
        if (VarType->isIntegerTy() && EndVal->getType()->isIntegerTy()) {
            EndVal = Builder.CreateIntCast(EndVal, VarType, true, "endcast");
        } else {
            throw std::runtime_error("Type mismatch in for loop range");
        }
    }
    
    // Create an alloca for the loop variable
    llvm::AllocaInst* Alloca = Builder.CreateAlloca(VarType, nullptr, VarName);
    
    // Store the start value
    Builder.CreateStore(StartVal, Alloca);
    
    // Save the old variable binding (if any)
    llvm::Value* OldVal = NamedValues[VarName];
    NamedValues[VarName] = Alloca;
    
    // Create blocks for the loop
    llvm::BasicBlock* CondBB = llvm::BasicBlock::Create(TheModule->getContext(), "forcond", TheFunction);
    llvm::BasicBlock* LoopBB = llvm::BasicBlock::Create(TheModule->getContext(), "forloop", TheFunction);
    llvm::BasicBlock* IncrBB = llvm::BasicBlock::Create(TheModule->getContext(), "forincr", TheFunction);
    llvm::BasicBlock* AfterBB = llvm::BasicBlock::Create(TheModule->getContext(), "afterloop", TheFunction);
    
    // Save previous loop context - continue should go to increment, break to after
    llvm::BasicBlock* PrevContinue = CurrentLoopContinue;
    llvm::BasicBlock* PrevBreak = CurrentLoopBreak;
    CurrentLoopContinue = IncrBB;  // Continue goes to increment block
    CurrentLoopBreak = AfterBB;
    
    // Jump to condition block
    Builder.CreateBr(CondBB);
    
    // Emit condition block
    Builder.SetInsertPoint(CondBB);
    llvm::Value* CurVar = Builder.CreateLoad(VarType, Alloca, VarName.c_str());
    llvm::Value* CondV = Builder.CreateICmpSLT(CurVar, EndVal, "forcond");
    Builder.CreateCondBr(CondV, LoopBB, AfterBB);
    
    // Emit loop body
    Builder.SetInsertPoint(LoopBB);
    for (auto& Expr : Body) {
        Expr->codegen(Builder, TheModule, NamedValues);
    }
    
    // Only create branch if the block doesn't already have a terminator
    if (!Builder.GetInsertBlock()->getTerminator()) {
        Builder.CreateBr(IncrBB);
    }
    
    // Emit increment block
    Builder.SetInsertPoint(IncrBB);
    llvm::Value* CurVarForIncrement = Builder.CreateLoad(VarType, Alloca, VarName.c_str());
    llvm::Value* StepVal = llvm::ConstantInt::get(VarType, 1);
    llvm::Value* NextVar = Builder.CreateAdd(CurVarForIncrement, StepVal, "nextvar");
    Builder.CreateStore(NextVar, Alloca);
    Builder.CreateBr(CondBB);
    
    // Emit after block
    Builder.SetInsertPoint(AfterBB);
    
    // Restore the old variable binding
    if (OldVal)
        NamedValues[VarName] = OldVal;
    else
        NamedValues.erase(VarName);
    
    // Restore previous loop context
    CurrentLoopContinue = PrevContinue;
    CurrentLoopBreak = PrevBreak;
    
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
}

llvm::Value* BreakExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    if (!CurrentLoopBreak) {
        throw std::runtime_error("break statement not inside a loop");
    }
    
    Builder.CreateBr(CurrentLoopBreak);
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
}

llvm::Value* ContinueExprAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    if (!CurrentLoopContinue) {
        throw std::runtime_error("continue statement not inside a loop");
    }
    
    Builder.CreateBr(CurrentLoopContinue);
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0);
}

llvm::Function* FunctionAST::codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) {
    // Create function prototype
    std::vector<llvm::Type*> ArgTypes;
    for (const auto& arg : Args) {
        ArgTypes.push_back(getTypeFromString(arg.second, TheModule->getContext()));
    }

    llvm::Type* RetType = ReturnType.empty() ? 
        llvm::Type::getVoidTy(TheModule->getContext()) : 
        getTypeFromString(ReturnType, TheModule->getContext());

    llvm::FunctionType* FT = llvm::FunctionType::get(
        RetType,        // Return type
        ArgTypes,       // Arg types
        false          // Not vararg
    );

    // Create the function with appropriate linkage
    // main() is always exported, as are extern and export functions
    llvm::Function::LinkageTypes linkage;
    if (isExtern || isExport || Name == "main") {
        linkage = llvm::Function::ExternalLinkage;
    } else {
        linkage = llvm::Function::InternalLinkage;
    }

    llvm::Function* F = llvm::Function::Create(
        FT,
        linkage,
        Name,
        TheModule
    );

    // Set calling convention to C for extern/export functions and main
    if (isExtern || isExport || Name == "main") {
        F->setCallingConv(llvm::CallingConv::C);
    }

    // Set names for all arguments
    unsigned ArgIdx = 0;
    for (auto& Arg : F->args())
        Arg.setName(Args[ArgIdx++].first);

    // Extern functions don't have a body
    if (isExtern) {
        return F;
    }

    // Create a new basic block to start insertion into
    llvm::BasicBlock* BB = llvm::BasicBlock::Create(TheModule->getContext(), "entry", F);
    Builder.SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map
    NamedValues.clear();
    unsigned Idx = 0;
    for (auto& Arg : F->args()) {
        // Create an alloca for this variable using the correct type
        llvm::Type* ArgType = getTypeFromString(Args[Idx].second, TheModule->getContext());
        llvm::AllocaInst* Alloca = Builder.CreateAlloca(ArgType, nullptr, Arg.getName());

        // Store the initial value into the alloca
        Builder.CreateStore(&Arg, Alloca);

        // Add arguments to variable symbol table
        NamedValues[std::string(Arg.getName())] = Alloca;
        Idx++;
    }

    // Generate code for each expression in the function body
    for (auto& Expr : Body) {
        Expr->codegen(Builder, TheModule, NamedValues);
    }

    // Add implicit return for void functions if not already present
    if (ReturnType.empty() && !Builder.GetInsertBlock()->getTerminator()) {
        Builder.CreateRetVoid();
    }

    // Validate the generated code, checking for consistency
    llvm::verifyFunction(*F);

    return F;
}
