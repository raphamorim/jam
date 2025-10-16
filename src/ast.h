/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef AST_H
#define AST_H

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

// Forward declarations
class ExprAST;
class FunctionAST;

// AST node base class
class ExprAST {
public:
    virtual ~ExprAST() = default;
    virtual llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) = 0;
};

// Number literal
class NumberExprAST : public ExprAST {
    int64_t Val;
public:
    NumberExprAST(int64_t Val) : Val(Val) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Boolean literal
class BooleanExprAST : public ExprAST {
    bool Val;
public:
    BooleanExprAST(bool Val) : Val(Val) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// String literal
class StringLiteralExprAST : public ExprAST {
    std::string Val;
public:
    StringLiteralExprAST(std::string Val) : Val(std::move(Val)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Variable reference
class VariableExprAST : public ExprAST {
    std::string Name;
public:
    VariableExprAST(std::string Name) : Name(std::move(Name)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Binary operation
class BinaryExprAST : public ExprAST {
    std::string Op;
    std::unique_ptr<ExprAST> LHS, RHS;
public:
    BinaryExprAST(std::string Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS)
        : Op(std::move(Op)), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Function call
class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
public:
    CallExprAST(std::string Callee, std::vector<std::unique_ptr<ExprAST>> Args)
        : Callee(std::move(Callee)), Args(std::move(Args)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
    
private:
    llvm::Value* generatePrintCall(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues);
};

// Return statement
class ReturnExprAST : public ExprAST {
    std::unique_ptr<ExprAST> RetVal;
public:
    ReturnExprAST(std::unique_ptr<ExprAST> RetVal) : RetVal(std::move(RetVal)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Variable declaration
class VarDeclAST : public ExprAST {
    std::string Name;
    std::string Type;
    bool IsConst;
    std::unique_ptr<ExprAST> Init;
public:
    VarDeclAST(std::string Name, std::string Type, bool IsConst, std::unique_ptr<ExprAST> Init)
        : Name(std::move(Name)), Type(std::move(Type)), IsConst(IsConst), Init(std::move(Init)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// If statement
class IfExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Condition;
    std::vector<std::unique_ptr<ExprAST>> ThenBody;
    std::vector<std::unique_ptr<ExprAST>> ElseBody;
public:
    IfExprAST(std::unique_ptr<ExprAST> Condition, 
              std::vector<std::unique_ptr<ExprAST>> ThenBody,
              std::vector<std::unique_ptr<ExprAST>> ElseBody)
        : Condition(std::move(Condition)), ThenBody(std::move(ThenBody)), ElseBody(std::move(ElseBody)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// While loop
class WhileExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Condition;
    std::vector<std::unique_ptr<ExprAST>> Body;
public:
    WhileExprAST(std::unique_ptr<ExprAST> Condition, std::vector<std::unique_ptr<ExprAST>> Body)
        : Condition(std::move(Condition)), Body(std::move(Body)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// For loop
class ForExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Start;
    std::unique_ptr<ExprAST> End;
    std::vector<std::unique_ptr<ExprAST>> Body;
public:
    ForExprAST(std::string VarName, std::unique_ptr<ExprAST> Start, std::unique_ptr<ExprAST> End, std::vector<std::unique_ptr<ExprAST>> Body)
        : VarName(std::move(VarName)), Start(std::move(Start)), End(std::move(End)), Body(std::move(Body)) {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Break statement
class BreakExprAST : public ExprAST {
public:
    BreakExprAST() {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Continue statement
class ContinueExprAST : public ExprAST {
public:
    ContinueExprAST() {}
    llvm::Value* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues) override;
};

// Function declaration
class FunctionAST {
public:
    std::string Name;
    std::vector<std::pair<std::string, std::string>> Args; // (name, type)
    std::string ReturnType;
    std::vector<std::unique_ptr<ExprAST>> Body;
    bool isExtern;  // extern function (no body)
    bool isExport;  // export function (visible to C)

    FunctionAST(std::string Name, std::vector<std::pair<std::string, std::string>> Args,
                std::string ReturnType, std::vector<std::unique_ptr<ExprAST>> Body,
                bool isExtern = false, bool isExport = false)
        : Name(std::move(Name)), Args(std::move(Args)), ReturnType(std::move(ReturnType)), 
          Body(std::move(Body)), isExtern(isExtern), isExport(isExport) {}

    llvm::Function* codegen(llvm::IRBuilder<>& Builder, llvm::Module* TheModule, std::map<std::string, llvm::Value*>& NamedValues);
};

// Global variables to track loop context for break/continue
extern llvm::BasicBlock* CurrentLoopContinue;
extern llvm::BasicBlock* CurrentLoopBreak;

#endif // AST_H
