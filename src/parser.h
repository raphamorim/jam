/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef PARSER_H
#define PARSER_H

#include "token.h"
#include "ast.h"
#include <vector>
#include <memory>

class Parser {
private:
    std::vector<Token> tokens;
    int current = 0;

    Token peek() const;
    Token previous() const;
    bool isAtEnd() const;
    Token advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    void consume(TokenType type, const std::string& message);
    
    std::unique_ptr<ExprAST> parsePrimary();
    std::string parseType();
    std::unique_ptr<ExprAST> parseExpression();
    std::unique_ptr<ExprAST> parseComparison();
    std::unique_ptr<ExprAST> parseAddition();
    std::unique_ptr<FunctionAST> parseFunction();

public:
    explicit Parser(std::vector<Token> tokens);
    std::vector<std::unique_ptr<FunctionAST>> parse();
};

#endif // PARSER_H
