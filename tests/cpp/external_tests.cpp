#include "test_framework.h"
#include <cstdlib>
#include <fstream>
#include <sstream>

// Forward declaration
class PrintFunctionTests {
public:
    static void registerAllTests(TestFramework& framework);
};

class LoopTests {
public:
    static void registerAllTests(TestFramework& framework);
};

class CompilerExternalTests {
public:
    static void registerAllTests(TestFramework& framework) {
        framework.addTest("External - Build Compiler", testBuildCompiler);
        framework.addTest("External - u8 Function", testU8Function);
        framework.addTest("External - u16 Function", testU16Function);
        framework.addTest("External - u32 Function", testU32Function);
        framework.addTest("External - i8 Function", testI8Function);
        framework.addTest("External - i16 Function", testI16Function);
        framework.addTest("External - i32 Function", testI32Function);
        framework.addTest("External - Mixed Types", testMixedTypes);
        framework.addTest("External - Mixed Signed Types", testMixedSignedTypes);
        framework.addTest("External - Boundary Values", testBoundaryValues);
        framework.addTest("External - Signed Boundary Values", testSignedBoundaryValues);
        framework.addTest("External - Complex Function", testComplexFunction);
        framework.addTest("External - Multiple Functions", testMultipleFunctions);
    }

private:
    static std::string runCommand(const std::string& command, bool throwOnError = true) {
        std::string result;
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to run command: " + command);
        }
        
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        
        int exitCode = pclose(pipe);
        if (throwOnError && exitCode != 0) {
            throw std::runtime_error("Command failed with exit code " + std::to_string(exitCode) + ": " + command);
        }
        
        return result;
    }
    
    static void writeTestFile(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create test file: " + filename);
        }
        file << content;
        file.close();
    }
    
    static std::string compileTestFile(const std::string& filename) {
        // Get current working directory and navigate to project root
        std::string pwd = runCommand("pwd", true);
        pwd.erase(pwd.find_last_not_of(" \n\r\t") + 1); // trim whitespace
        std::string projectRoot = pwd.substr(0, pwd.find("/tests/cpp"));
        std::string command = "cd " + projectRoot + " && ./build/jam " + filename + " 2>&1";
        return runCommand(command, false); // Don't throw on compiler errors
    }
    
    static void testBuildCompiler() {
        // Ensure the compiler is built using dynamic project root
        std::string pwd = runCommand("pwd", true);
        pwd.erase(pwd.find_last_not_of(" \n\r\t") + 1); // trim whitespace
        std::string projectRoot = pwd.substr(0, pwd.find("/tests/cpp"));
        std::string output = runCommand("cd " + projectRoot + " && ./build.sh > /dev/null 2>&1 && echo 'SUCCESS'");
        ASSERT_CONTAINS(output, "SUCCESS");
    }
    
    static void testU8Function() {
        std::string testCode = R"(
            fn test_u8() -> u8 {
                const a: u8 = 100;
                const b: u8 = 155;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_u8.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_u8.jam");
        
        ASSERT_CONTAINS(ir, "define i8 @test_u8()");
        ASSERT_CONTAINS(ir, "store i8 100");
        ASSERT_CONTAINS(ir, "store i8 -101"); // 155 as signed i8
        ASSERT_CONTAINS(ir, "add i8");
        ASSERT_CONTAINS(ir, "ret i8");
    }
    
    static void testU16Function() {
        std::string testCode = R"(
            fn test_u16() -> u16 {
                const a: u16 = 30000;
                const b: u16 = 35535;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_u16.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_u16.jam");
        
        ASSERT_CONTAINS(ir, "define i16 @test_u16()");
        ASSERT_CONTAINS(ir, "store i16 30000");
        ASSERT_CONTAINS(ir, "add i16");
        ASSERT_CONTAINS(ir, "ret i16");
    }
    
    static void testU32Function() {
        std::string testCode = R"(
            fn test_u32() -> u32 {
                const a: u32 = 1000000;
                const b: u32 = 2000000;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_u32.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_u32.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @test_u32()");
        ASSERT_CONTAINS(ir, "store i32 1000000");
        ASSERT_CONTAINS(ir, "store i32 2000000");
        ASSERT_CONTAINS(ir, "add i32");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testI8Function() {
        std::string testCode = R"(
            fn test_i8() -> i8 {
                const a: i8 = -42;
                const b: i8 = 42;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_i8.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_i8.jam");
        
        ASSERT_CONTAINS(ir, "define i8 @test_i8()");
        ASSERT_CONTAINS(ir, "store i8 -42");
        ASSERT_CONTAINS(ir, "store i8 42");
        ASSERT_CONTAINS(ir, "add i8");
        ASSERT_CONTAINS(ir, "ret i8");
    }
    
    static void testI16Function() {
        std::string testCode = R"(
            fn test_i16() -> i16 {
                const a: i16 = -1000;
                const b: i16 = 2000;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_i16.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_i16.jam");
        
        ASSERT_CONTAINS(ir, "define i16 @test_i16()");
        ASSERT_CONTAINS(ir, "store i16 -1000");
        ASSERT_CONTAINS(ir, "store i16 2000");
        ASSERT_CONTAINS(ir, "add i16");
        ASSERT_CONTAINS(ir, "ret i16");
    }
    
    static void testI32Function() {
        std::string testCode = R"(
            fn test_i32() -> i32 {
                const a: i32 = -100000;
                const b: i32 = 200000;
                return a + b;
            }
        )";
        
        writeTestFile("/tmp/test_i32.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_i32.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @test_i32()");
        ASSERT_CONTAINS(ir, "store i32 -100000");
        ASSERT_CONTAINS(ir, "store i32 200000");
        ASSERT_CONTAINS(ir, "add i32");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testMixedTypes() {
        std::string testCode = R"(
            fn mixed_types(a: u8, b: u16, c: u32) -> u32 {
                return c;
            }
        )";
        
        writeTestFile("/tmp/test_mixed.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_mixed.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @mixed_types(i8 %a, i16 %b, i32 %c)");
        ASSERT_CONTAINS(ir, "alloca i8");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "alloca i32");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testMixedSignedTypes() {
        std::string testCode = R"(
            fn mixed_signed_types(a: i8, b: i16, c: i32) -> i32 {
                return c;
            }
        )";
        
        writeTestFile("/tmp/test_mixed_signed.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_mixed_signed.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @mixed_signed_types(i8 %a, i16 %b, i32 %c)");
        ASSERT_CONTAINS(ir, "alloca i8");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "alloca i32");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    static void testBoundaryValues() {
        std::string testCode = R"(
            fn boundary_test() -> u32 {
                const u8_max: u8 = 255;
                const u16_max: u16 = 65535;
                const u32_max: u32 = 4294967295;
                return u32_max;
            }
        )";
        
        writeTestFile("/tmp/test_boundary.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_boundary.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @boundary_test()");
        ASSERT_CONTAINS(ir, "store i8 -1");  // 255 as signed i8
        ASSERT_CONTAINS(ir, "store i16 -1"); // 65535 as signed i16
        ASSERT_CONTAINS(ir, "store i32 -1"); // 4294967295 as signed i32
    }
    
    static void testSignedBoundaryValues() {
        std::string testCode = R"(
            fn signed_boundary_test() -> i32 {
                const i8_min: i8 = -128;
                const i8_max: i8 = 127;
                const i16_min: i16 = -32768;
                const i16_max: i16 = 32767;
                const i32_min: i32 = -2147483648;
                const i32_max: i32 = 2147483647;
                return i32_max;
            }
        )";
        
        writeTestFile("/tmp/test_signed_boundary.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_signed_boundary.jam");
        
        ASSERT_CONTAINS(ir, "define i32 @signed_boundary_test()");
        ASSERT_CONTAINS(ir, "store i8 -128");
        ASSERT_CONTAINS(ir, "store i8 127");
        ASSERT_CONTAINS(ir, "store i16 -32768");
        ASSERT_CONTAINS(ir, "store i16 32767");
        ASSERT_CONTAINS(ir, "store i32 -2147483648");
        ASSERT_CONTAINS(ir, "store i32 2147483647");
    }
    
    static void testComplexFunction() {
        std::string testCode = R"(
            fn complex_add(x: u16, y: u16) -> u16 {
                const temp: u16 = x + y;
                return temp;
            }
        )";
        
        writeTestFile("/tmp/test_complex.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_complex.jam");
        
        ASSERT_CONTAINS(ir, "define i16 @complex_add(i16 %x, i16 %y)");
        ASSERT_CONTAINS(ir, "add i16");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "ret i16");
    }
    
    static void testMultipleFunctions() {
        std::string testCode = R"(
            fn func_u8() -> u8 { return 255; }
            fn func_u16() -> u16 { return 65535; }
            fn func_u32() -> u32 { return 4294967295; }
        )";
        
        writeTestFile("/tmp/test_multiple.jam", testCode);
        std::string ir = compileTestFile("/tmp/test_multiple.jam");
        
        ASSERT_CONTAINS(ir, "define i8 @func_u8()");
        ASSERT_CONTAINS(ir, "define i16 @func_u16()");
        ASSERT_CONTAINS(ir, "define i32 @func_u32()");
        ASSERT_CONTAINS(ir, "ret i8");
        ASSERT_CONTAINS(ir, "ret i16");
        ASSERT_CONTAINS(ir, "ret i32");
    }
};

int main() {
    TestFramework framework;
    
    // Register all tests
    CompilerExternalTests::registerAllTests(framework);
    PrintFunctionTests::registerAllTests(framework);
    LoopTests::registerAllTests(framework);
    
    // Run all tests
    framework.runAll();
    
    return framework.allPassed() ? 0 : 1;
}