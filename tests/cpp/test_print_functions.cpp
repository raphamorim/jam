#include "test_framework.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>

class PrintFunctionTests {
public:
    static void registerAllTests(TestFramework& framework);

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

    // Helper function to compile and run a jam file, capturing output
    static std::string compileAndRun(const std::string& jamCode) {
        // Write jam code to temporary file
        std::string tempFile = "/tmp/test_print.jam";
        writeTestFile(tempFile, jamCode);

        // Get current working directory and navigate to project root
        std::string pwd = runCommand("pwd", true);
        pwd.erase(pwd.find_last_not_of(" \n\r\t") + 1); // trim whitespace
        std::string projectRoot = pwd.substr(0, pwd.find("/tests/cpp"));

        // Compile the jam file
        std::string compileCmd = "cd " + projectRoot + " && ./jam.out " + tempFile + " > /dev/null 2>&1";
        int compileResult = system(compileCmd.c_str());
        
        if (compileResult != 0) {
            throw std::runtime_error("Compilation failed");
        }

        // Run the executable and capture output
        std::string runCmd = "cd " + projectRoot + " && ./output 2>&1";
        std::string output = runCommand(runCmd, false);
        
        // Remove trailing newline if present
        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }
        
        return output;
    }

    static void testPrintlnBasic() {
        std::string jamCode = R"(
fn main() -> u32 {
    println("Hello, world!");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Hello, world!");
    }

    static void testPrintBasic() {
        std::string jamCode = R"(
fn main() -> u32 {
    print("Hello");
    print(" ");
    print("world!");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Hello world!");
    }

    static void testPrintAndPrintln() {
        std::string jamCode = R"(
fn main() -> u32 {
    print("Line 1: ");
    println("First line");
    print("Line 2: ");
    println("Second line");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Line 1: First line\nLine 2: Second line");
    }

    static void testPrintlnWithVariable() {
        std::string jamCode = R"(
fn main() -> u32 {
    const message: str = "Hello from variable";
    println(message);
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Hello from variable");
    }

    static void testPrintWithVariable() {
        std::string jamCode = R"(
fn main() -> u32 {
    const greeting: str = "Hello";
    const name: str = "World";
    print(greeting);
    print(" ");
    println(name);
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Hello World");
    }

    static void testMultiplePrintlnStatements() {
        std::string jamCode = R"(
fn main() -> u32 {
    println("First");
    println("Second");
    println("Third");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "First\nSecond\nThird");
    }

    static void testPrintlnInIfStatement() {
        std::string jamCode = R"(
fn main() -> u32 {
    const condition: bool = true;
    if (condition) {
        println("Condition is true");
    } else {
        println("Condition is false");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Condition is true");
    }

    static void testPrintlnInElseStatement() {
        std::string jamCode = R"(
fn main() -> u32 {
    const condition: bool = false;
    if (condition) {
        println("Condition is true");
    } else {
        println("Condition is false");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Condition is false");
    }

    static void testEmptyString() {
        std::string jamCode = R"(
fn main() -> u32 {
    println("");
    print("");
    println("After empty");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "\nAfter empty");
    }

    static void testSpecialCharacters() {
        std::string jamCode = R"(
fn main() -> u32 {
    println("Special: !@#$%^&*");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Special: !@#$%^&*");
    }
};

// Implementation of registerAllTests
void PrintFunctionTests::registerAllTests(TestFramework& framework) {
    framework.addTest("Print - Basic println", testPrintlnBasic);
    framework.addTest("Print - Basic print", testPrintBasic);
    framework.addTest("Print - Print and println combined", testPrintAndPrintln);
    framework.addTest("Print - println with variable", testPrintlnWithVariable);
    framework.addTest("Print - print with variable", testPrintWithVariable);
    framework.addTest("Print - Multiple println statements", testMultiplePrintlnStatements);
    framework.addTest("Print - println in if statement", testPrintlnInIfStatement);
    framework.addTest("Print - println in else statement", testPrintlnInElseStatement);
    framework.addTest("Print - Empty string", testEmptyString);
    framework.addTest("Print - Special characters", testSpecialCharacters);
}