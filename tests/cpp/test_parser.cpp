#include "test_framework.h"
#include "../../src/main.cpp"

class ParserTests {
public:
    static void registerTests(TestFramework& framework) {
        framework.addTest("Parser - Simple Function", testSimpleFunction);
        framework.addTest("Parser - Function with Parameters", testFunctionWithParameters);
        framework.addTest("Parser - Variable Declaration", testVariableDeclaration);
        framework.addTest("Parser - Const Declaration", testConstDeclaration);
        framework.addTest("Parser - Return Statement", testReturnStatement);
        framework.addTest("Parser - Binary Expression", testBinaryExpression);
        framework.addTest("Parser - Function Call", testFunctionCall);
        framework.addTest("Parser - Type Annotations", testTypeAnnotations);
    }

private:
    static std::vector<std::unique_ptr<FunctionAST>> parseCode(const std::string& code) {
        Lexer lexer(code);
        auto tokens = lexer.scanTokens();
        Parser parser(tokens);
        return parser.parse();
    }
    
    static void testSimpleFunction() {
        auto functions = parseCode("fn test() -> u8 { return 42; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("test", functions[0]->Name);
        ASSERT_EQ("u8", functions[0]->ReturnType);
        ASSERT_EQ(0, functions[0]->Args.size());
        ASSERT_EQ(1, functions[0]->Body.size());
    }
    
    static void testFunctionWithParameters() {
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
    
    static void testVariableDeclaration() {
        auto functions = parseCode("fn test() -> u8 { var x: u8 = 42; return x; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(2, functions[0]->Body.size()); // var declaration + return
    }
    
    static void testConstDeclaration() {
        auto functions = parseCode("fn test() -> u8 { const x: u8 = 42; return x; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(2, functions[0]->Body.size()); // const declaration + return
    }
    
    static void testReturnStatement() {
        auto functions = parseCode("fn test() -> u8 { return 255; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(1, functions[0]->Body.size());
    }
    
    static void testBinaryExpression() {
        auto functions = parseCode("fn test() -> u8 { const result: u8 = 10 + 20; return result; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(2, functions[0]->Body.size()); // const declaration + return
    }
    
    static void testFunctionCall() {
        auto functions = parseCode("fn test() -> u8 { return other(42); }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ(1, functions[0]->Body.size());
    }
    
    static void testTypeAnnotations() {
        auto functions = parseCode("fn test() -> u32 { const a: u8 = 255; const b: u16 = 65535; const c: u32 = 4294967295; return c; }");
        
        ASSERT_EQ(1, functions.size());
        ASSERT_EQ("u32", functions[0]->ReturnType);
        ASSERT_EQ(4, functions[0]->Body.size()); // 3 const declarations + return
    }
};