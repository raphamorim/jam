/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef TOKEN_H
#define TOKEN_H

#include <string>

// Token types
enum TokenType {
    TOK_EOF = 0,
    TOK_FN,
    TOK_IDENTIFIER,
    TOK_COLON,
    TOK_ARROW,
    TOK_OPEN_BRACE,
    TOK_CLOSE_BRACE,
    TOK_OPEN_PAREN,
    TOK_CLOSE_PAREN,
    TOK_COMMA,
    TOK_RETURN,
    TOK_PLUS,
    TOK_MINUS,
    TOK_SEMI,
    TOK_NUMBER,
    TOK_CONST,
    TOK_VAR,
    TOK_EQUAL,
    TOK_TYPE,
    TOK_IF,
    TOK_ELSE,
    TOK_EQUAL_EQUAL,
    TOK_NOT_EQUAL,
    TOK_LESS,
    TOK_LESS_EQUAL,
    TOK_GREATER,
    TOK_GREATER_EQUAL,
    TOK_TRUE,
    TOK_FALSE,
    TOK_OPEN_BRACKET,
    TOK_CLOSE_BRACKET,
    TOK_STRING_LITERAL,
    TOK_WHILE,
    TOK_FOR,
    TOK_BREAK,
    TOK_CONTINUE,
    TOK_IN,
    TOK_EXTERN,    // extern keyword
    TOK_EXPORT,    // export keyword
};

// Token structure
struct Token {
    TokenType type;
    std::string lexeme;
    int line;

    Token(TokenType type, std::string lexeme, int line)
        : type(type), lexeme(std::move(lexeme)), line(line) {}
};

#endif // TOKEN_H
