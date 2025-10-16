#include "test_framework.h"

// Include the main source file only once
#include "../../src/main.cpp"

class CompilerTests {
public:
    static void registerAllTests(TestFramework& framework) {
        // Lexer Tests
        framework.addTest("Lexer - Basic Keywords", testLexerKeywords);
        framework.addTest("Lexer - Type Tokens", testLexerTypes);
        framework.addTest("Lexer - Numbers", testLexerNumbers);
        framework.addTest("Lexer - Identifiers", testLexerIdentifiers);
        
        // Parser Tests
        framework.addTest("Parser - Simple Function", testParserSimpleFunction);
        framework.addTest("Parser - Function Parameters", testParserFunctionParams);
        framework.addTest("Parser - Variable Declaration", testParserVarDecl);
        
        // Type System Tests
        framework.addTest("Type System - u8", testTypeSystemU8);
        framework.addTest("Type System - u16", testTypeSystemU16);
        framework.addTest("Type System - u32", testTypeSystemU32);
        
        // Integration Tests
        framework.addTest("Integration - u8 Function", testIntegrationU8);
        framework.addTest("Integration - u16 Function", testIntegrationU16);
        framework.addTest("Integration - u32 Function", testIntegrationU32);
        framework.addTest("Integration - Mixed Types", testIntegrationMixed);
        
        // If/Else Tests
        framework.addTest("IfElse - If Keywords", testIfKeywords);
        framework.addTest("IfElse - Comparison Operators", testComparisonOperators);
        framework.addTest("IfElse - If Statement", testIfStatement);
        framework.addTest("IfElse - If-Else Statement", testIfElseStatement);
        framework.addTest("IfElse - If Statement Integration", testIfIntegration);
        
        // Bool Tests
        framework.addTest("Bool - Bool Keywords", testBoolKeywords);
        framework.addTest("Bool - Bool Type", testBoolType);
        framework.addTest("Bool - Bool Parsing", testBoolParsing);
        framework.addTest("Bool - Bool Integration", testBoolIntegration);
    }

private:
    // Lexer Tests
    static void testLexerKeywords() {
        Lexer lexer("fn return const var");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(5, tokens.size()); // 4 keywords + EOF
        ASSERT_EQ(TOK_FN, tokens[0].type);
        ASSERT_EQ("fn", tokens[0].lexeme);
        ASSERT_EQ(TOK_RETURN, tokens[1].type);
        ASSERT_EQ(TOK_CONST, tokens[2].type);
        ASSERT_EQ(TOK_VAR, tokens[3].type);
    }
    
    static void testLexerTypes() {
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
    
    static void testLexerNumbers() {
        Lexer lexer("42 255 65535 4294967295");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(5, tokens.size()); // 4 numbers + EOF
        ASSERT_EQ(TOK_NUMBER, tokens[0].type);
        ASSERT_EQ("42", tokens[0].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[1].type);
        ASSERT_EQ("255", tokens[1].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[2].type);
        ASSERT_EQ("65535", tokens[2].lexeme);
        ASSERT_EQ(TOK_NUMBER, tokens[3].type);
        ASSERT_EQ("4294967295", tokens[3].lexeme);
    }
    
    static void testLexerIdentifiers() {
        Lexer lexer("variable_name test123");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(3, tokens.size()); // 2 identifiers + EOF
        ASSERT_EQ(TOK_IDENTIFIER, tokens[0].type);
        ASSERT_EQ("variable_name", tokens[0].lexeme);
        ASSERT_EQ(TOK_IDENTIFIER, tokens[1].type);
        ASSERT_EQ("test123", tokens[1].lexeme);
    }
    
    // Parser Tests
    static std::vector<std::unique_ptr<FunctionAST>> parseCode(const std::string& code) {
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(tokens);
        return parser.parse();
    }
    
    static void testParserSimpleFunction() {
        auto functions = parseCode("fn test() -> u8 { return 42; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
        ASSERT_EQ("u8", functions[0]->ReturnType);
        ASSERT_EQ(0, functions[0]->Args.size());
        ASSERT_EQ(1, functions[0]->Body.size());
    }
    
    static void testParserFunctionParams() {
        auto functions = parseCode("fn add(a: u8, b: u16) -> u32 { return a; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("add", functions[0]->Name);
        ASSERT_EQ("u32", functions[0]->ReturnType);
        ASSERT_EQ(2, functions[0]->Args.size());
        
        ASSERT_EQ("a", functions[0]->Args[0].first);
        ASSERT_EQ("u8", functions[0]->Args[0].second);
        ASSERT_EQ("b", functions[0]->Args[1].first);
        ASSERT_EQ("u16", functions[0]->Args[1].second);
    }
    
    static void testParserVarDecl() {
        auto functions = parseCode("fn test() -> u8 { const x: u8 = 42; return x; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(2, functions[0]->Body.size()); // const declaration + return
    }
    
    // Type System Tests
    static void testTypeSystemU8() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u8", context);
        ASSERT_TRUE(type->isIntegerTy(8));
    }
    
    static void testTypeSystemU16() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u16", context);
        ASSERT_TRUE(type->isIntegerTy(16));
    }
    
    static void testTypeSystemU32() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("u32", context);
        ASSERT_TRUE(type->isIntegerTy(32));
    }
    
    // Integration Tests
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
    
    static void testIntegrationU8() {
        std::string source = "fn test() -> u8 { return 42; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i8 @test()");
        ASSERT_CONTAINS(ir, "ret i8");
        ASSERT_CONTAINS(ir, "42");
    }
    
    static void testIntegrationU16() {
        std::string source = "fn test() -> u16 { return 30000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i16 @test()");
        ASSERT_CONTAINS(ir, "ret i16");
        ASSERT_CONTAINS(ir, "30000");
    }
    
    static void testIntegrationU32() {
        std::string source = "fn test() -> u32 { return 1000000; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test()");
        ASSERT_CONTAINS(ir, "ret i32");
        ASSERT_CONTAINS(ir, "1000000");
    }
    
    static void testIntegrationMixed() {
        std::string source = "fn test(a: u8, b: u16) -> u32 { return a; }";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i32 @test(i8 %a, i16 %b)");
        ASSERT_CONTAINS(ir, "alloca i8");
        ASSERT_CONTAINS(ir, "alloca i16");
        ASSERT_CONTAINS(ir, "ret i32");
    }
    
    // If/Else Tests
    static void testIfKeywords() {
        Lexer lexer("if else");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(3, tokens.size()); // if, else, EOF
        ASSERT_EQ(TOK_IF, tokens[0].type);
        ASSERT_EQ("if", tokens[0].lexeme);
        ASSERT_EQ(TOK_ELSE, tokens[1].type);
        ASSERT_EQ("else", tokens[1].lexeme);
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

    static void testIfStatement() {
        std::string code = R"(
            fn test() -> u8 {
                if (1 == 1) {
                    return 42;
                }
                return 0;
            }
        )";
        
        auto functions = parseCode(code);
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }

    static void testIfElseStatement() {
        std::string code = R"(
            fn test() -> u8 {
                if (1 == 1) {
                    return 42;
                } else {
                    return 0;
                }
            }
        )";
        
        auto functions = parseCode(code);
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
    }

    static void testIfIntegration() {
        std::string source = R"(
            fn test() -> u8 {
                if (1 == 1) {
                    return 42;
                } else {
                    return 0;
                }
            }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i8 @test()");
        ASSERT_CONTAINS(ir, "icmp eq");
        ASSERT_CONTAINS(ir, "br i1");
        ASSERT_CONTAINS(ir, "ret i8 42");
        ASSERT_CONTAINS(ir, "ret i8 0");
    }
    
    // Bool Tests
    static void testBoolKeywords() {
        Lexer lexer("bool true false");
        auto tokens = lexer.scanTokens();
        
        ASSERT_EQ(4, tokens.size()); // bool, true, false, EOF
        ASSERT_EQ(TOK_TYPE, tokens[0].type);
        ASSERT_EQ("bool", tokens[0].lexeme);
        ASSERT_EQ(TOK_TRUE, tokens[1].type);
        ASSERT_EQ("true", tokens[1].lexeme);
        ASSERT_EQ(TOK_FALSE, tokens[2].type);
        ASSERT_EQ("false", tokens[2].lexeme);
    }

    static void testBoolType() {
        llvm::LLVMContext context;
        auto type = getTypeFromString("bool", context);
        ASSERT_TRUE(type->isIntegerTy(1));
    }

    static void testBoolParsing() {
        std::string code = R"(
            fn test() -> bool {
                const flag: bool = true;
                return flag;
            }
        )";
        
        auto functions = parseCode(code);
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
        ASSERT_EQ("bool", functions[0]->ReturnType);
    }

    static void testBoolIntegration() {
        std::string source = R"(
            fn test() -> bool {
                const flag: bool = true;
                if (flag == true) {
                    return true;
                } else {
                    return false;
                }
            }
        )";
        std::string ir = compileToIR(source);
        
        ASSERT_CONTAINS(ir, "define i1 @test()");
        ASSERT_CONTAINS(ir, "store i1 true");
        ASSERT_CONTAINS(ir, "ret i1 true");
        ASSERT_CONTAINS(ir, "ret i1 false");
    }
};

int main() {
    TestFramework framework;
    
    // Register all tests
    CompilerTests::registerAllTests(framework);
    
    // Run all tests
    framework.runAll();
    
    return framework.allPassed() ? 0 : 1;
}