#include "test_framework.h"
#include "test_lexer.cpp"
#include "test_parser.cpp"
#include "test_types.cpp"
#include "test_integration.cpp"

int main() {
    TestFramework framework;
    
    // Register all test suites
    LexerTests::registerTests(framework);
    ParserTests::registerTests(framework);
    TypeSystemTests::registerTests(framework);
    IntegrationTests::registerTests(framework);
    
    // Run all tests
    framework.runAll();
    
    return framework.allPassed() ? 0 : 1;
}