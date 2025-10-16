#include "test_framework.h"
#include "../../src/main.cpp"
#include <sstream>

class IntegrationTests {
public:
    static void registerTests(TestFramework& framework) {
        framework.addTest("Integration - Simple u8 Function", testSimpleU8Function);
        framework.addTest("Integration - u16 Function", testU16Function);
        framework.addTest("Integration - u32 Function", testU32Function);
        framework.addTest("Integration - Simple i8 Function", testSimpleI8Function);
        framework.addTest("Integration - i16 Function", testI16Function);
        framework.addTest("Integration - i32 Function", testI32Function);
        framework.addTest("Integration - Mixed Types", testMixedTypes);
        framework.addTest("Integration - Mixed Signed Types", testMixedSignedTypes);
        framework.addTest("Integration - Complex Function", testComplexFunction);
        framework.addTest("Integration - Multiple Functions", testMultipleFunctions);
        framework.addTest("Integration - Boundary Values", testBoundaryValues);
        framework.addTest("Integration - Signed Boundary Values", testSignedBoundaryValues);
    }

private:
    static std::string compileToIR(const std::string& source) {
        // Initialize LLVM
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
        
        // Create LLVM context and module
        llvm::LLVMContext context;
        auto module = std::make_unique<llvm::Module>("test_module", context);
        llvm::IRBuilder<> builder(context);
        
        // Tokenize and parse
        Lexer lexer(source);
        auto tokens = lexer.scanTokens();
        Parser parser(tokens);
        auto functions = parser.parse();
        
        // Generate code
        std::map<std::string, llvm::Value*> namedValues;
        for (auto& function : functions) {
            function->codegen(builder, module.get(), namedValues);
        }
        
        // Convert to string
        std::string output;
        llvm::raw_string_ostream stream(output);
        module->print(stream, nullptr);
        return output;
    }
    
    static void testSimpleU8Function() {
        std::string source = "fn test() -> u8 { return 42; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i8 @test()");
        ASSERT_CONTAINS(ir, "ret i8");
        ASSERT_CONTAINS(ir, "42");
    }
    
    static void testU16Function() {
        std::string source = "fn test() -> u16 { return 30000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i16 @test()");
        ASSERT_CONTAINS(ir, "ret i16");
        ASSERT_CONTAINS(ir, "30000");
    }
    
    static void testU32Function() {
        std::string source = "fn test() -> u32 { return 1000000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test()");
        ASSERT_CONTAINS(ir, "ret i32");
        ASSERT_CONTAINS(ir, "1000000");
    }
    
    static void testSimpleI8Function() {
        std::string source = "fn test() -> i8 { return -42; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i8 @test()");
        ASSERT_CONTAINS(ir, "ret i8 -42");
    }
    
    static void testI16Function() {
        std::string source = "fn test() -> i16 { return -1000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i16 @test()");
        ASSERT_CONTAINS(ir, "ret i16 -1000");
    }
    
    static void testI32Function() {
        std::string source = "fn test() -> i32 { return -100000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test()");
        ASSERT_CONTAINS(ir, "ret i32 -100000");
    }
    
    static void testMixedTypes() {
        std::string source = "fn test(a: u8, b: u16) -> u32 { return a; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test(i8 %a, i16 %b)");
        ASSERT_CONTAINS(ir, "alloca i8");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testMixedSignedTypes() {
        std::string source = "fn test(a: i8, b: i16, c: i32) -> i32 { return c; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test(i8 %a, i16 %b, i32 %c)");
        ASSERT_CONTAINS(ir, "alloca i8");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "alloca i32");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testComplexFunction() {
        std::string source = R"(
            fn add(a: u16, b: u16) -> u16 {
                const sum: u16 = a + b;
                return sum;
            }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i16 @add(i16 %a, i16 %b)");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "add i16");
        ASSERT_CONTAINS(ir, "ret i16");
    }
    
    static void testMultipleFunctions() {
        std::string source = R"(
            fn test_u8() -> u8 { return 255; }
            fn test_u16() -> u16 { return 65535; }
            fn test_u32() -> u32 { return 4294967295; }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i8 @test_u8()");
        ASSERT_CONTAINS(ir, "define i16 @test_u16()");
        ASSERT_CONTAINS(ir, "define i32 @test_u32()");
    }
    
    static void testBoundaryValues() {
        std::string source = R"(
            fn test_boundaries() -> u32 {
                const u8_max: u8 = 255;
                const u16_max: u16 = 65535;
                const u32_max: u32 = 4294967295;
                return u32_max;
            }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test_boundaries()");
        ASSERT_CONTAINS(ir, "store i8 -1"); // 255 as signed i8
        ASSERT_CONTAINS(ir, "store i16 -1"); // 65535 as signed i16
        ASSERT_CONTAINS(ir, "store i32 -1"); // 4294967295 as signed i32
    }
    
    static void testSignedBoundaryValues() {
        std::string source = R"(
            fn test_signed_boundaries() -> i32 {
                const i8_min: i8 = -128;
                const i8_max: i8 = 127;
                const i16_min: i16 = -32768;
                const i16_max: i16 = 32767;
                const i32_min: i32 = -2147483648;
                const i32_max: i32 = 2147483647;
                return i32_max;
            }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test_signed_boundaries()");
        ASSERT_CONTAINS(ir, "store i8 -128");
        ASSERT_CONTAINS(ir, "store i8 127");
        ASSERT_CONTAINS(ir, "store i16 -32768");
        ASSERT_CONTAINS(ir, "store i16 32767");
        ASSERT_CONTAINS(ir, "store i32 -2147483648");
        ASSERT_CONTAINS(ir, "store i32 2147483647");
    }
};