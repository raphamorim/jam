/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "lexer.h"
#include <stdexcept>
#include <iostream>

Lexer::Lexer(std::string source) : source(std::move(source)) {}

bool Lexer::isAtEnd() const {
    return current >= source.length();
}

char Lexer::advance() {
    return source[current++];
}

char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return source[current];
}

char Lexer::peekNext() const {
    if (current + 1 >= source.length()) return '\0';
    return source[current + 1];
}

bool Lexer::match(char expected) {
    if (isAtEnd() || source[current] != expected) return false;
    current++;
    return true;
}

void Lexer::skipWhitespace() {
    while (true) {
        char c = peek();
        switch (c) {
            case ' ':
            case '\r':
            case '\t':
                advance();
                break;
            case '\n':
                line++;
                advance();
                break;
            case '/':
                if (peekNext() == '/') {
                    // Comment until end of line
                    while (peek() != '\n' && !isAtEnd()) advance();
                } else {
                    return;
                }
                break;
            default:
                return;
        }
    }
}

bool Lexer::isDigit(char c) const {
    return c >= '0' && c <= '9';
}

bool Lexer::isAlpha(char c) const {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool Lexer::isAlphaNumeric(char c) const {
    return isAlpha(c) || isDigit(c);
}

void Lexer::addToken(TokenType type) {
    addToken(type, "");
}

void Lexer::addToken(TokenType type, const std::string& lexeme) {
    tokens.emplace_back(type, lexeme, line);
}

void Lexer::identifier() {
    int start = current - 1; // Start position (we already consumed the first character)
    while (isAlphaNumeric(peek())) advance();

    std::string text = source.substr(start, current - start);

    // Check for keywords
    if (text == "fn") {
        addToken(TOK_FN, text);
    } else if (text == "return") {
        addToken(TOK_RETURN, text);
    } else if (text == "const") {
        addToken(TOK_CONST, text);
    } else if (text == "var") {
        addToken(TOK_VAR, text);
    } else if (text == "if") {
        addToken(TOK_IF, text);
    } else if (text == "else") {
        addToken(TOK_ELSE, text);
    } else if (text == "while") {
        addToken(TOK_WHILE, text);
    } else if (text == "for") {
        addToken(TOK_FOR, text);
    } else if (text == "break") {
        addToken(TOK_BREAK, text);
    } else if (text == "continue") {
        addToken(TOK_CONTINUE, text);
    } else if (text == "in") {
        addToken(TOK_IN, text);
    } else if (text == "true") {
        addToken(TOK_TRUE, text);
    } else if (text == "false") {
        addToken(TOK_FALSE, text);
    } else if (text == "extern") {
        addToken(TOK_EXTERN, text);
    } else if (text == "export") {
        addToken(TOK_EXPORT, text);
    } else if (text == "print" || text == "println" || text == "printf") {
        addToken(TOK_IDENTIFIER, text); // Treat as regular identifiers for now
    } else if (text == "u8" || text == "u16" || text == "u32" || text == "i8" || text == "i16" || text == "i32" || text == "bool" || text == "str") {
        addToken(TOK_TYPE, text);
    } else {
        addToken(TOK_IDENTIFIER, text);
    }
}

void Lexer::number() {
    int start = current - 1; // Start position (we already consumed the first digit)
    while (isDigit(peek())) advance();

    std::string num = source.substr(start, current - start);
    addToken(TOK_NUMBER, num);
}

void Lexer::negativeNumber() {
    int start = current - 1; // Start position (we already consumed the minus)
    while (isDigit(peek())) advance();

    std::string num = source.substr(start, current - start);
    addToken(TOK_NUMBER, num);
}

void Lexer::stringLiteral() {
    int start = current; // Start after the opening quote
    
    while (peek() != '"' && !isAtEnd()) {
        if (peek() == '\n') line++;
        advance();
    }

    if (isAtEnd()) {
        throw std::runtime_error("Unterminated string at line " + std::to_string(line));
    }

    // The closing "
    advance();

    // Trim the surrounding quotes
    std::string value = source.substr(start, current - start - 1);
    addToken(TOK_STRING_LITERAL, value);
}

std::vector<Token> Lexer::scanTokens() {
    while (!isAtEnd()) {
        skipWhitespace();
        if (isAtEnd()) break;

        char c = advance();

        switch (c) {
            case '(': addToken(TOK_OPEN_PAREN, "("); break;
            case ')': addToken(TOK_CLOSE_PAREN, ")"); break;
            case '{': addToken(TOK_OPEN_BRACE, "{"); break;
            case '}': addToken(TOK_CLOSE_BRACE, "}"); break;
            case '[': addToken(TOK_OPEN_BRACKET, "["); break;
            case ']': addToken(TOK_CLOSE_BRACKET, "]"); break;
            case ',': addToken(TOK_COMMA, ","); break;
            case ';': addToken(TOK_SEMI, ";"); break;
            case ':': addToken(TOK_COLON, ":"); break;
            case '+': addToken(TOK_PLUS, "+"); break;
            case '"': stringLiteral(); break;
            
            case '=':
                if (match('=')) {
                    addToken(TOK_EQUAL_EQUAL, "==");
                } else {
                    addToken(TOK_EQUAL, "=");
                }
                break;
            
            case '!':
                if (match('=')) {
                    addToken(TOK_NOT_EQUAL, "!=");
                } else {
                    std::cerr << "Unexpected character at line " << line << ": " << c << std::endl;
                }
                break;
            
            case '<':
                if (match('=')) {
                    addToken(TOK_LESS_EQUAL, "<=");
                } else {
                    addToken(TOK_LESS, "<");
                }
                break;
            
            case '>':
                if (match('=')) {
                    addToken(TOK_GREATER_EQUAL, ">=");
                } else {
                    addToken(TOK_GREATER, ">");
                }
                break;

            case '-':
                if (match('>')) {
                    addToken(TOK_ARROW, "->");
                } else if (isDigit(peek())) {
                    // Handle negative number
                    negativeNumber();
                } else {
                    addToken(TOK_MINUS, "-");
                }
                break;

            default:
                if (isDigit(c)) {
                    number();
                } else if (isAlpha(c)) {
                    identifier();
                } else {
                    std::cerr << "Unexpected character at line " << line << ": " << c << std::endl;
                }
                break;
        }
    }

    tokens.emplace_back(TOK_EOF, "", line);
    return tokens;
}
