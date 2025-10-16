#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <sstream>

// Simple test framework for Jam compiler
class TestFramework {
private:
    struct TestCase {
        std::string name;
        std::function<void()> test;
        bool passed = false;
        std::string error;
    };
    
    std::vector<TestCase> tests;
    int passed = 0;
    int failed = 0;

public:
    void addTest(const std::string& name, std::function<void()> test) {
        tests.push_back({name, test});
    }
    
    void runAll() {
        std::cout << "Running C++ Unit Tests for Jam Compiler\n";
        std::cout << "=======================================\n\n";
        
        for (auto& test : tests) {
            std::cout << test.name << "... ";
            try {
                test.test();
                test.passed = true;
                std::cout << "PASS\n";
                passed++;
            } catch (const std::exception& e) {
                test.passed = false;
                test.error = e.what();
                std::cout << "FAIL\n";
                std::cout << "   Error: " << e.what() << "\n";
                failed++;
            }
        }
        
        std::cout << "\nTest Results\n";
        std::cout << "============\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Total:  " << (passed + failed) << "\n";
        
        if (failed == 0) {
            std::cout << "\nAll C++ tests passed!\n";
        } else {
            std::cout << "\nSome C++ tests failed.\n";
        }
    }
    
    bool allPassed() const {
        return failed == 0;
    }
};

// Test assertion macros
#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        throw std::runtime_error("Assertion failed: " #condition " should be false"); \
    }

#define ASSERT_EQ(expected, actual) \
    if ((expected) != (actual)) { \
        std::ostringstream oss; \
        oss << "Assertion failed: expected " << (expected) << " but got " << (actual); \
        throw std::runtime_error(oss.str()); \
    }

#define ASSERT_NE(expected, actual) \
    if ((expected) == (actual)) { \
        std::ostringstream oss; \
        oss << "Assertion failed: expected " << (expected) << " to not equal " << (actual); \
        throw std::runtime_error(oss.str()); \
    }

#define ASSERT_CONTAINS(haystack, needle) \
    if ((haystack).find(needle) == std::string::npos) { \
        std::ostringstream oss; \
        oss << "Assertion failed: '" << (haystack) << "' does not contain '" << (needle) << "'"; \
        throw std::runtime_error(oss.str()); \
    }

#define ASSERT_THROWS(expression) \
    { \
        bool threw = false; \
        try { \
            expression; \
        } catch (...) { \
            threw = true; \
        } \
        if (!threw) { \
            throw std::runtime_error("Assertion failed: " #expression " should have thrown an exception"); \
        } \
    }