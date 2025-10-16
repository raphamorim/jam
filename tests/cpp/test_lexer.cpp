#include "test_framework.h"
#include "../../src/main.cpp"  // Include the main source to test internal classes

class LexerTests {
public:
    static void registerTests(TestFramework& framework) {
        framework.addTest("Lexer - Basic Keywords", testBasicKeywords);
        framework.addTest("Lexer - Type Tokens", testTypeTokens);
        framework.addTest("Lexer - Signed Type Tokens", testSignedTypeTokens);
        framework.addTest("Lexer - Numbers", testNumbers);
        framework.addTest("Lexer - Negative Numbers", testNegativeNumbers);
        framework.addTest("Lexer - Identifiers", testIdentifiers);
        framework.addTest("Lexer - Operators", testOperators);
        framework.addTest("Lexer - Complex Expression", testComplexExpression);
        framework.addTest("Lexer - Comments", testComments);
        framework.addTest("Lexer - Whitespace Handling", testWhitespace);
    }

private:
    static void testBasicKeywords() {
        Lexer lexer("fn return const var");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(5, tokens.size()); // 4 keywords + EOF
        ASSERT_EQ(TOK_FN, tokens[0].type);
        ASSERT_EQ("fn", tokens[0].lexeme);
        ASSERT_EQ(TOK_RETURN, tokens[1].type);
        ASSERT_EQ("return", tokens[1].lexeme);
        ASSERT_EQ(TOK_CONST, tokens[2].type);
        ASSERT_EQ("const", tokens[2].lexeme);
        ASSERT_EQ(TOK_VAR, tokens[3].type);
        ASSERT_EQ("var", tokens[3].lexeme);
        ASSERT_EQ(TOK_EOF, tokens[4].type);
    }
    
    static void testTypeTokens() {
        Lexer lexer("u8 u16 u32");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(4, tokens.size()); // 3 types + EOF
        ASSERT_EQ(TOK_TYPE, tokens[0].type);
        ASSERT_EQ("u8", tokens[0].lexeme);
        ASSERT_EQ(TOK_TYPE, tokens[1].type);
        ASSERT_EQ("u16", tokens[1].lexeme);
        ASSERT_EQ(TOK_TYPE, tokens[2].type);
        ASSERT_EQ("u32", tokens[2].lexeme);
    }
    
    static void testSignedTypeTokens() {
        Lexer lexer("i8 i16 i32");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(4, tokens.size()); // 3 types + EOF
        ASSERT_EQ(TOK_TYPE, tokens[0].type);
        ASSERT_EQ("i8", tokens[0].lexeme);
        ASSERT_EQ(TOK_TYPE, tokens[1].type);
        ASSERT_EQ("i16", tokens[1].lexeme);
        ASSERT_EQ(TOK_TYPE, tokens[2].type);
        ASSERT_EQ("i32", tokens[2].lexeme);
    }
    
    static void testNumbers() {
        Lexer lexer("0 42 255 65535 4294967295");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(6, tokens.size()); // 5 numbers + EOF
        ASSERT_EQ(TOK_NUMBER, tokens[0].type);
        ASSERT_EQ("0", tokens[0].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[1].type);
        ASSERT_EQ("42", tokens[1].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[2].type);
        ASSERT_EQ("255", tokens[2].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[3].type);
        ASSERT_EQ("65535", tokens[3].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[4].type);
        ASSERT_EQ("4294967295", tokens[4].lexeme);
    }
    
    static void testNegativeNumbers() {
        Lexer lexer("-42 -128 -32768 -2147483648");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(5, tokens.size()); // 4 negative numbers + EOF
        ASSERT_EQ(TOK_NUMBER, tokens[0].type);
        ASSERT_EQ("-42", tokens[0].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[1].type);
        ASSERT_EQ("-128", tokens[1].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[2].type);
        ASSERT_EQ("-32768", tokens[2].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[3].type);
        ASSERT_EQ("-2147483648", tokens[3].lexeme);
    }
    
    static void testIdentifiers() {
        Lexer lexer("variable_name function_name test123");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(4, tokens.size()); // 3 identifiers + EOF
        ASSERT_EQ(TOK_IDENTIFIER, tokens[0].type);
        ASSERT_EQ("variable_name", tokens[0].lexeme);
        ASSERT_EQ(TOK_IDENTIFIER, tokens[1].type);
        ASSERT_EQ("function_name", tokens[1].lexeme);
        ASSERT_EQ(TOK_IDENTIFIER, tokens[2].type);
        ASSERT_EQ("test123", tokens[2].lexeme);
    }
    
    static void testOperators() {
        Lexer lexer("+ = : -> ( ) { } , ;");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(11, tokens.size()); // 10 operators + EOF
        ASSERT_EQ(TOK_PLUS, tokens[0].type);
        ASSERT_EQ(TOK_EQUAL, tokens[1].type);
        ASSERT_EQ(TOK_COLON, tokens[2].type);
        ASSERT_EQ(TOK_ARROW, tokens[3].type);
        ASSERT_EQ(TOK_OPEN_PAREN, tokens[4].type);
        ASSERT_EQ(TOK_CLOSE_PAREN, tokens[5].type);
        ASSERT_EQ(TOK_OPEN_BRACE, tokens[6].type);
        ASSERT_EQ(TOK_CLOSE_BRACE, tokens[7].type);
        ASSERT_EQ(TOK_COMMA, tokens[8].type);
        ASSERT_EQ(TOK_SEMI, tokens[9].type);
    }
    
    static void testComplexExpression() {
        Lexer lexer("fn test(a: u8, b: u16) -> u32 { const result: u32 = a + b; return result; }");
        auto tokens = lexer.scanTokens();
        
        // Verify we have the right number of tokens
        ASSERT_TRUE(tokens.size() > 20);
        
        // Check key tokens are present
        ASSERT_EQ(TOK_FN, tokens[0].type);
        ASSERT_EQ(TOK_IDENTIFIER, tokens[1].type);
        ASSERT_EQ("test", tokens[1].lexeme);
        
        // Find and verify type tokens
        bool foundU8 = false, foundU16 = false, foundU32 = false;
        for (const auto& token : tokens) {
            if (token.type == TOK_TYPE) {
                if (token.lexeme == "u8") foundU8 = true;
                if (token.lexeme == "u16") foundU16 = true;
                if (token.lexeme == "u32") foundU32 = true;
            }
        }
        ASSERT_TRUE(foundU8);
        ASSERT_TRUE(foundU16);
        ASSERT_TRUE(foundU32);
    }
    
    static void testComments() {
        Lexer lexer("fn test() { // This is a comment\n return 42; }");
        auto tokens = lexer.scanTokens();
        
        // Comments should be ignored
        bool foundComment = false;
        for (const auto& token : tokens) {
            if (token.lexeme.find("comment") != std::string::npos) {
                foundComment = true;
            }
        }
        ASSERT_FALSE(foundComment);
        
        // But other tokens should still be there
        ASSERT_EQ(TOK_FN, tokens[0].type);
        ASSERT_EQ(TOK_RETURN, tokens[5].type);
        ASSERT_EQ(TOK_NUMBER, tokens[6].type);
        ASSERT_EQ("42", tokens[6].lexeme);
    }
    
    static void testWhitespace() {
        Lexer lexer("   fn   test   (   )   {   }   ");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(7, tokens.size()); // 6 tokens + EOF
        ASSERT_EQ(TOK_FN, tokens[0].type);
        ASSERT_EQ(TOK_IDENTIFIER, tokens[1].type);
        ASSERT_EQ("test", tokens[1].lexeme);
        ASSERT_EQ(TOK_OPEN_PAREN, tokens[2].type);
        ASSERT_EQ(TOK_CLOSE_PAREN, tokens[3].type);
        ASSERT_EQ(TOK_OPEN_BRACE, tokens[4].type);
        ASSERT_EQ(TOK_CLOSE_BRACE, tokens[5].type);
    }
};