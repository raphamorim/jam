#include "test_framework.h"
#include "../../src/main.cpp"

class TypeSystemTests {
public:
    static void registerTests(TestFramework& framework) {
        framework.addTest("Type System - getTypeFromString u8", testGetTypeU8);
        framework.addTest("Type System - getTypeFromString u16", testGetTypeU16);
        framework.addTest("Type System - getTypeFromString u32", testGetTypeU32);
        framework.addTest("Type System - getTypeFromString i8", testGetTypeI8);
        framework.addTest("Type System - getTypeFromString i16", testGetTypeI16);
        framework.addTest("Type System - getTypeFromString i32", testGetTypeI32);
        framework.addTest("Type System - Invalid Type", testInvalidType);
        framework.addTest("Number AST - u8 Range", testNumberASTU8);
        framework.addTest("Number AST - u16 Range", testNumberASTU16);
        framework.addTest("Number AST - u32 Range", testNumberASTU32);
        framework.addTest("Number AST - i8 Range", testNumberASTI8);
        framework.addTest("Number AST - i16 Range", testNumberASTI16);
        framework.addTest("Number AST - i32 Range", testNumberASTI32);
        framework.addTest("Number AST - Large Values", testNumberASTLarge);
    }

private:
    static void testGetTypeU8() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u8", context);
        ASSERT_TRUE(type->isIntegerTy(8));
    }
    
    static void testGetTypeU16() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u16", context);
        ASSERT_TRUE(type->isIntegerTy(16));
    }
    
    static void testGetTypeU32() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u32", context);
        ASSERT_TRUE(type->isIntegerTy(32));
    }
    
    static void testGetTypeI8() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("i8", context);
        ASSERT_TRUE(type->isIntegerTy(8));
    }
    
    static void testGetTypeI16() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("i16", context);
        ASSERT_TRUE(type->isIntegerTy(16));
    }
    
    static void testGetTypeI32() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("i32", context);
        ASSERT_TRUE(type->isIntegerTy(32));
    }
    
    static void testInvalidType() {
        llvm::LLVMContext context;
        ASSERT_THROWS(getTypeFromString("invalid", context));
    }
    
    static void testNumberASTU8() {
        // Test u8 range (0-255)
        NumberExprAST num255(255);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto value = num255.codegen(builder, &module, namedValues);
        ASSERT_TRUE(value != nullptr);
        ASSERT_TRUE(value->getType()->isIntegerTy(8));
    }
    
    static void testNumberASTU16() {
        // Test u16 range (256-65535)
        NumberExprAST num65535(65535);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto value = num65535.codegen(builder, &module, namedValues);
        ASSERT_TRUE(value != nullptr);
        ASSERT_TRUE(value->getType()->isIntegerTy(16));
    }
    
    static void testNumberASTU32() {
        // Test u32 range (65536-4294967295)
        NumberExprAST num4billion(4000000000ULL);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto value = num4billion.codegen(builder, &module, namedValues);
        ASSERT_TRUE(value != nullptr);
        ASSERT_TRUE(value->getType()->isIntegerTy(32));
    }
    
    static void testNumberASTLarge() {
        // Test very large numbers
        NumberExprAST numMax(4294967295ULL);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto value = numMax.codegen(builder, &module, namedValues);
        ASSERT_TRUE(value != nullptr);
        ASSERT_TRUE(value->getType()->isIntegerTy(32));
    }
    
    static void testNumberASTI8() {
        // Test i8 range (-128 to 127)
        NumberExprAST numNeg42(-42);
        NumberExprAST numPos42(42);
        NumberExprAST numMin(-128);
        NumberExprAST numMax(127);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto valueNeg = numNeg42.codegen(builder, &module, namedValues);
        auto valuePos = numPos42.codegen(builder, &module, namedValues);
        auto valueMin = numMin.codegen(builder, &module, namedValues);
        auto valueMax = numMax.codegen(builder, &module, namedValues);
        
        ASSERT_TRUE(valueNeg != nullptr && valueNeg->getType()->isIntegerTy(8));
        ASSERT_TRUE(valuePos != nullptr && valuePos->getType()->isIntegerTy(8));
        ASSERT_TRUE(valueMin != nullptr && valueMin->getType()->isIntegerTy(8));
        ASSERT_TRUE(valueMax != nullptr && valueMax->getType()->isIntegerTy(8));
    }
    
    static void testNumberASTI16() {
        // Test i16 range (-32768 to 32767)
        NumberExprAST numNeg1000(-1000);
        NumberExprAST numPos1000(1000);
        NumberExprAST numMin(-32768);
        NumberExprAST numMax(32767);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto valueNeg = numNeg1000.codegen(builder, &module, namedValues);
        auto valuePos = numPos1000.codegen(builder, &module, namedValues);
        auto valueMin = numMin.codegen(builder, &module, namedValues);
        auto valueMax = numMax.codegen(builder, &module, namedValues);
        
        ASSERT_TRUE(valueNeg != nullptr && valueNeg->getType()->isIntegerTy(16));
        ASSERT_TRUE(valuePos != nullptr && valuePos->getType()->isIntegerTy(16));
        ASSERT_TRUE(valueMin != nullptr && valueMin->getType()->isIntegerTy(16));
        ASSERT_TRUE(valueMax != nullptr && valueMax->getType()->isIntegerTy(16));
    }
    
    static void testNumberASTI32() {
        // Test i32 range (-2147483648 to 2147483647)
        NumberExprAST numNeg100k(-100000);
        NumberExprAST numPos100k(100000);
        NumberExprAST numMin(-2147483648LL);
        NumberExprAST numMax(2147483647);
        
        llvm::LLVMContext context;
        llvm::Module module("test", context);
        llvm::IRBuilder<> builder(context);
        std::map<std::string, llvm::Value*> namedValues;
        
        auto valueNeg = numNeg100k.codegen(builder, &module, namedValues);
        auto valuePos = numPos100k.codegen(builder, &module, namedValues);
        auto valueMin = numMin.codegen(builder, &module, namedValues);
        auto valueMax = numMax.codegen(builder, &module, namedValues);
        
        ASSERT_TRUE(valueNeg != nullptr && valueNeg->getType()->isIntegerTy(32));
        ASSERT_TRUE(valuePos != nullptr && valuePos->getType()->isIntegerTy(32));
        ASSERT_TRUE(valueMin != nullptr && valueMin->getType()->isIntegerTy(32));
        ASSERT_TRUE(valueMax != nullptr && valueMax->getType()->isIntegerTy(32));
    }
};