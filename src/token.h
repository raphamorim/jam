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

// Token types
enum TokenType {
	TOK_EOF = 0,
	TOK_FN,
	// `cfn` declares a method whose calls are synthesized by the
	// compiler instead of being typed by the user. The shape on disk
	// is identical to a regular function (same args / body / return);
	// the difference is opt-in for the compiler hooks (`drop`, `at`,
	// `default`). A `fn drop(self)` without `cfn` is just a method.
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
	TOK_LOOP,  // `loop { ... }` — infinite loop, syntactically noreturn unless
	           // body breaks
	TOK_FOR,
	TOK_BREAK,
	TOK_CONTINUE,
	TOK_IN,
	TOK_EXTERN,     // extern keyword (import C function)
	TOK_EXPORT,     // export keyword (C ABI export)
	TOK_PUB,        // pub keyword (visible across Jam modules)
	TOK_IMPORT,     // import keyword
	TOK_DOT,        // . for member access
	TOK_AND,        // && (logical AND, short-circuit)
	TOK_OR,         // || (logical OR, short-circuit)
	TOK_NOT,        // ! (logical NOT)
	TOK_TFN,        // tfn keyword (test function)
	TOK_STRUCT,     // struct keyword
	TOK_UNION,      // union keyword (untagged; FFI-shaped)
	TOK_ENUM,       // enum keyword (tagged sum type; payload-less)
	TOK_STAR,       // * (pointer prefix; multiplication)
	TOK_SLASH,      // / (division)
	TOK_PERCENT,    // % (modulo)
	TOK_AMP,        // & (bitwise AND; address-of)
	TOK_PIPE,       // | (bitwise OR)
	TOK_CARET,      // ^ (bitwise XOR)
	TOK_TILDE,      // ~ (bitwise NOT)
	TOK_LSHIFT,     // << (left shift)
	TOK_RSHIFT,     // >> (right shift)
	TOK_MOVE,       // move keyword (parameter mode: consume ownership)
	TOK_ELLIPSIS,   // ... (variadic marker in extern fn parameters)
	TOK_MATCH,      // match keyword
	TOK_DOTDOT,     // .. (exclusive range / slice)
	TOK_DOTDOT_EQ,  // ..= (inclusive range in match patterns)
	TOK_AS,         // as keyword (explicit type cast)
	TOK_AT,         // @ — prefix for comptime-function invocations:
	                //   `@sizeOf(T)`, `@alignOf(T)`, user-defined cfns
	TOK_COMP,       // comp keyword — comp param marker (`fn f(comp n: u32)
	                //   ...`); also wraps an expression that must fold at
	                //   compile time (`comp const X = 1 + 1;`). Source-
	                //   level keyword for the broader "compile-time
	                //   evaluation" mechanism implemented in src/comptime.h.
	TOK_INLINE,     // inline keyword — prefix for `inline while` (loop
	                //   unrolling at compile time, mirrors Zig's
	                //   `inline while`). Required by std.fmt's format-
	                //   string parser.
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
