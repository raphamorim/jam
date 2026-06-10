/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef TOKEN_H
#define TOKEN_H

#include <cstdint>
#include <string>
#include <string_view>

enum TokenType {
	TOK_EOF = 0,
	TOK_FN,
	TOK_CFN,
	TOK_IDENTIFIER,
	TOK_COLON,
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
	TOK_MUT,
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
	TOK_LOOP,
	TOK_FOR,
	TOK_BREAK,
	TOK_CONTINUE,
	TOK_IN,
	TOK_EXTERN,
	TOK_EXPORT,
	TOK_PUB,
	TOK_IMPORT,
	TOK_DOT,
	TOK_AND,
	TOK_OR,
	TOK_NOT,
	TOK_TFN,
	TOK_STRUCT,
	TOK_UNION,
	TOK_ENUM,
	TOK_STAR,     // * (pointer prefix; multiplication)
	TOK_SLASH,    // / (division)
	TOK_PERCENT,  // % (modulo)
	TOK_AMP,      // & (bitwise AND; address-of)
	TOK_PIPE,     // | (bitwise OR)
	TOK_CARET,    // ^ (bitwise XOR)
	TOK_TILDE,    // ~ (bitwise NOT)
	TOK_LSHIFT,   // << (left shift)
	TOK_RSHIFT,   // >> (right shift)
	TOK_MOVE,
	TOK_ELLIPSIS,  // ... (variadic marker in extern fn parameters)
	TOK_MATCH,
	TOK_DOTDOT,     // .. (exclusive range / slice)
	TOK_DOTDOT_EQ,  // ..= (inclusive range in match patterns)
	TOK_AS,
	TOK_AT,  // @
	TOK_COMP,
	TOK_INLINE,
};

// Token structure.
//
// Storage model: `byteOffset` + `length` mark the token's raw span in
// the original source buffer; `text(source)` returns a string_view over
// that span and is the right call for every token kind EXCEPT
// `TOK_STRING_LITERAL`.
//
// `lexeme` is populated only for `TOK_STRING_LITERAL` and carries the
// *decoded* value — escape sequences like `\n` / `\u{2603}` are
// resolved to the bytes the runtime expects, which the raw source
// doesn't contain. Use `lexeme` for string literals, `text(source)`
// for everything else.
struct Token {
	TokenType type;
	std::string lexeme;
	int line;
	uint32_t byteOffset = 0;
	uint32_t length = 0;

	Token(TokenType type, std::string lexeme, int line, uint32_t byteOffset = 0,
	      uint32_t length = 0)
	    : type(type), lexeme(std::move(lexeme)), line(line),
	      byteOffset(byteOffset), length(length) {}

	// Raw source span. Caller passes the same `source` the lexer ran
	// over. The returned view is valid as long as `source` is. For
	// `TOK_STRING_LITERAL` this returns the raw `"..."`-bracketed bytes
	// from the source — use `lexeme` for the decoded value.
	std::string_view text(const std::string &source) const {
		return std::string_view(source).substr(byteOffset, length);
	}
};

#endif  // TOKEN_H
