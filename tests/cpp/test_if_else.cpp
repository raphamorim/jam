#include "test_framework.h"
#include "../../src/main.cpp"  // Include the main source to test internal classes

class IfElseTests {
public:
    static void registerTests(TestFramework& framework) {
        framework.addTest("IfElse - Basic If Keywords", testIfKeywords);
        framework.addTest("IfElse - Comparison Operators", testComparisonOperators);
        framework.addTest("IfElse - If Statement Parsing", testIfStatementParsing);
        framework.addTest("IfElse - If-Else Statement Parsing", testIfElseStatementParsing);
        framework.addTest("IfElse - Nested Expressions", testNestedExpressions);
        framework.addTest("IfElse - Complex Conditions", testComplexConditions);
    }

private:
    static void testIfKeywords() {
        Lexer lexer("if else");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(3, tokens.size()); // if, else, EOF
        ASSERT_EQ(TOK_IF, tokens[0].type);
        ASSERT_EQ("if", tokens[0].lexeme);
        ASSERT_EQ(TOK_ELSE, tokens[1].type);
        ASSERT_EQ("else", tokens[1].lexeme);
        ASSERT_EQ(TOK_EOF, tokens[2].type);
    }

    static void testComparisonOperators() {
        Lexer lexer("== != < <= > >=");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(7, tokens.size()); // 6 operators + EOF
        ASSERT_EQ(TOK_EQUAL_EQUAL, tokens[0].type);
        ASSERT_EQ("==", tokens[0].lexeme);
        ASSERT_EQ(TOK_NOT_EQUAL, tokens[1].type);
        ASSERT_EQ("!=", tokens[1].lexeme);
        ASSERT_EQ(TOK_LESS, tokens[2].type);
        ASSERT_EQ("<", tokens[2].lexeme);
        ASSERT_EQ(TOK_LESS_EQUAL, tokens[3].type);
        ASSERT_EQ("<=", tokens[3].lexeme);
        ASSERT_EQ(TOK_GREATER, tokens[4].type);
        ASSERT_EQ(">", tokens[4].lexeme);
        ASSERT_EQ(TOK_GREATER_EQUAL, tokens[5].type);
        ASSERT_EQ(">=", tokens[5].lexeme);
    }

    static void testIfStatementParsing() {
        std::string code = R"(
            fn test() -> u8 {
                if (1 == 1) {
                    return 42;
                }
                return 0;
            }
        )";
        
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(std::move(tokens));
        
        // Should parse without throwing an exception
        auto functions = parser.parse();
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }

    static void testIfElseStatementParsing() {
        std::string code = R"(
            fn test() -> u8 {
                if (1 == 1) {
                    return 42;
                } else {
                    return 0;
                }
            }
        )";
        
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(std::move(tokens));
        
        // Should parse without throwing an exception
        auto functions = parser.parse();
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }

    static void testNestedExpressions() {
        std::string code = R"(
            fn test() -> u8 {
                const a: u8 = 10;
                const b: u8 = 20;
                if ((a + b) == 30) {
                    return 1;
                } else {
                    return 0;
                }
            }
        )";
        
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(std::move(tokens));
        
        // Should parse without throwing an exception
        auto functions = parser.parse();
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }

    static void testComplexConditions() {
        std::string code = R"(
            fn test() -> u8 {
                const x: u8 = 15;
                const y: u8 = 10;
                if (x > y) {
                    if (x >= 15) {
                        return 1;
                    } else {
                        return 2;
                    }
                } else {
                    return 0;
                }
            }
        )";
        
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(std::move(tokens));
        
        // Should parse without throwing an exception
        auto functions = parser.parse();
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }
};