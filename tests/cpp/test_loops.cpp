#include "test_framework.h"
#include <fstream>
#include <sstream>
#include <cstdio>

class LoopTests {
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

    static std::string compileAndRun(const std::string& jamCode) {
        std::string tempFile = "/tmp/test_loop.jam";
        writeTestFile(tempFile, jamCode);

        std::string pwd = runCommand("pwd", true);
        pwd.erase(pwd.find_last_not_of(" \n\r\t") + 1);
        std::string projectRoot = pwd.substr(0, pwd.find("/tests/cpp"));

        std::string compileCmd = "cd " + projectRoot + " && ./jam.out " + tempFile + " > /dev/null 2>&1";
        int compileResult = system(compileCmd.c_str());
        
        if (compileResult != 0) {
            throw std::runtime_error("Compilation failed");
        }

        std::string runCmd = "cd " + projectRoot + " && ./output 2>&1";
        std::string output = runCommand(runCmd, false);
        
        if (!output.empty() && output.back() == '\n') {
            output.pop_back();
        }
        
        return output;
    }

    static void testBasicForLoop() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:3 {
        println("For loop");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "For loop\nFor loop\nFor loop");
    }

    static void testForLoopWithBreak() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:10 {
        if (i == 3) {
            break;
        }
        println("Before break");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Before break\nBefore break\nBefore break");
    }

    static void testForLoopWithContinue() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:5 {
        if (i == 2) {
            continue;
        }
        println("Not skipped");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Not skipped\nNot skipped\nNot skipped\nNot skipped");
    }

    static void testForLoopWithBreakAndContinue() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:10 {
        if (i == 2) {
            continue;
        }
        if (i == 5) {
            break;
        }
        println("Iteration");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Iteration\nIteration\nIteration\nIteration");
    }

    static void testWhileLoopWithBreak() {
        std::string jamCode = R"(
fn main() -> u32 {
    const condition: bool = true;
    while (condition) {
        println("While loop");
        break;
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "While loop");
    }

    static void testWhileLoopWithFalseCondition() {
        std::string jamCode = R"(
fn main() -> u32 {
    const condition: bool = false;
    while (condition) {
        println("Should not print");
    }
    println("After while");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "After while");
    }

    static void testNestedForLoops() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:2 {
        for j in 0:2 {
            println("Nested");
        }
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Nested\nNested\nNested\nNested");
    }

    static void testForLoopWithDifferentTypes() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 1:4 {
        println("Type test");
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Type test\nType test\nType test");
    }

    static void testEmptyForLoop() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 5:5 {
        println("Should not print");
    }
    println("After empty loop");
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "After empty loop");
    }

    static void testForLoopVariableAccess() {
        std::string jamCode = R"(
fn main() -> u32 {
    for i in 0:3 {
        if (i == 1) {
            println("Found one");
        }
    }
    return 0;
}
)";
        
        std::string output = compileAndRun(jamCode);
        ASSERT_EQ(output, "Found one");
    }
};

void LoopTests::registerAllTests(TestFramework& framework) {
    framework.addTest("Loop - Basic for loop", testBasicForLoop);
    framework.addTest("Loop - For loop with break", testForLoopWithBreak);
    framework.addTest("Loop - For loop with continue", testForLoopWithContinue);
    framework.addTest("Loop - For loop with break and continue", testForLoopWithBreakAndContinue);
    framework.addTest("Loop - While loop with break", testWhileLoopWithBreak);
    framework.addTest("Loop - While loop with false condition", testWhileLoopWithFalseCondition);
    framework.addTest("Loop - Nested for loops", testNestedForLoops);
    framework.addTest("Loop - For loop with different types", testForLoopWithDifferentTypes);
    framework.addTest("Loop - Empty for loop", testEmptyForLoop);
    framework.addTest("Loop - For loop variable access", testForLoopVariableAccess);
}