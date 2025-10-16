/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef LEXER_H
#define LEXER_H

#include "token.h"
#include <vector>
#include <string>

class Lexer {
private:
    std::string source;
    std::vector<Token> tokens;
    int current = 0;
    int line = 1;

    bool isAtEnd() const;
    char advance();
    char peek() const;
    char peekNext() const;
    bool match(char expected);
    void skipWhitespace();
    bool isDigit(char c) const;
    bool isAlpha(char c) const;
    bool isAlphaNumeric(char c) const;
    void addToken(TokenType type);
    void addToken(TokenType type, const std::string& lexeme);
    void identifier();
    void number();
    void negativeNumber();
    void stringLiteral();

public:
    explicit Lexer(std::string source);
    std::vector<Token> scanTokens();
};

#endif // LEXER_H
