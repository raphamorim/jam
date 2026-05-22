/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "lexer.h"
#include <iostream>
#include <stdexcept>

Lexer::Lexer(std::string source) : source(std::move(source)) {}

bool Lexer::isAtEnd() const { return current >= source.length(); }

char Lexer::advance() { return source[current++]; }

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
			// `//` starts a line comment; bare `/` is the divide operator
			// (lexed as TOK_SLASH at the main scan-loop level below).
			if (peekNext() == '/') {
				// Comment until end of line
				while (peek() != '\n' && !isAtEnd()) advance();
				break;
			}
			return;
		default:
			return;
		}
	}
}

bool Lexer::isDigit(char c) const { return c >= '0' && c <= '9'; }

bool Lexer::isAlpha(char c) const {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool Lexer::isAlphaNumeric(char c) const { return isAlpha(c) || isDigit(c); }

void Lexer::addToken(TokenType type) { addToken(type, ""); }

void Lexer::addToken(TokenType type, const std::string &lexeme) {
	// `tokenStart` is captured at the top of each scan-loop iteration
	// (before the first `advance()`); `length` is the span between
	// then and `current`. Together they let callers slice the source
	// for the token's raw bytes via `Token::text(source)` — no per-
	// token string copy. The `lexeme` argument is only passed by the
	// string-literal path, which needs to carry the escape-decoded
	// value (a `\n` in the raw source becomes a single LF byte that
	// the source slice can't reproduce).
	uint32_t length =
	    static_cast<uint32_t>(static_cast<uint32_t>(current) - tokenStart);
	tokens.emplace_back(type, lexeme, line, tokenStart, length);
}

void Lexer::identifier() {
	int start = current -
	            1;  // Start position (we already consumed the first character)
	while (isAlphaNumeric(peek())) advance();

	// `text` is a view into the source — keyword classification only
	// reads it, so no copy is needed. The emitted Token's `lexeme`
	// stays empty; downstream callers reconstruct the string via
	// `token.text(source)`.
	std::string_view text(source.data() + start,
	                      static_cast<std::size_t>(current - start));

	// Check for keywords
	if (text == "fn") {
		addToken(TOK_FN);
	} else if (text == "cfn") {
		addToken(TOK_CFN);
	} else if (text == "return") {
		addToken(TOK_RETURN);
	} else if (text == "const") {
		addToken(TOK_CONST);
	} else if (text == "var") {
		addToken(TOK_VAR);
	} else if (text == "mut") {
		addToken(TOK_MUT);
	} else if (text == "if") {
		addToken(TOK_IF);
	} else if (text == "else") {
		addToken(TOK_ELSE);
	} else if (text == "match") {
		addToken(TOK_MATCH);
	} else if (text == "while") {
		addToken(TOK_WHILE);
	} else if (text == "loop") {
		addToken(TOK_LOOP);
	} else if (text == "for") {
		addToken(TOK_FOR);
	} else if (text == "break") {
		addToken(TOK_BREAK);
	} else if (text == "continue") {
		addToken(TOK_CONTINUE);
	} else if (text == "in") {
		addToken(TOK_IN);
	} else if (text == "true") {
		addToken(TOK_TRUE);
	} else if (text == "false") {
		addToken(TOK_FALSE);
	} else if (text == "extern") {
		addToken(TOK_EXTERN);
	} else if (text == "export") {
		addToken(TOK_EXPORT);
	} else if (text == "pub") {
		addToken(TOK_PUB);
	} else if (text == "import") {
		addToken(TOK_IMPORT);
	} else if (text == "tfn") {
		addToken(TOK_TFN);
	} else if (text == "struct") {
		addToken(TOK_STRUCT);
	} else if (text == "union") {
		addToken(TOK_UNION);
	} else if (text == "enum") {
		addToken(TOK_ENUM);
	} else if (text == "as") {
		addToken(TOK_AS);
	} else if (text == "move") {
		addToken(TOK_MOVE);
	} else if (text == "comp") {
		addToken(TOK_COMP);
	} else if (text == "inline") {
		addToken(TOK_INLINE);
	} else if (text == "u1" || text == "u8" || text == "u16" || text == "u32" ||
	           text == "u64" || text == "i8" || text == "i16" ||
	           text == "i32" || text == "i64" || text == "f32" ||
	           text == "f64" || text == "bool" || text == "str" ||
	           text == "type" || text == "noreturn") {
		// `type` is the meta-type — values of this type are
		// themselves types, used at compile time only. Lexed as TOK_TYPE
		// alongside the scalar built-ins so the parser sees it in a type
		// position uniformly.
		addToken(TOK_TYPE);
	} else {
		addToken(TOK_IDENTIFIER);
	}
}

// Permissive number-literal scanner. The rule: collect everything that
// could plausibly belong to a numeric literal (digits, hex digits,
// underscores, periods, exponent letters and their sign) and produce
// one token. Validation is deferred to `parseNumberLiteral` in
// number_literal.cpp.
//
// Greedy character set:
//   .          → entering a float (or `..=` separator — see below)
//   _          → digit separator
//   0-9, a-z, A-Z → digits or future suffix letters
//   e, E, p, P → exponent markers
//   + / -      → exponent sign immediately after p/P/e/E
//
// Escape hatch:
//   `..` (followed by a non-digit) ends the number — required so that
//   `0..=9` lexes as `0`, `..=`, `9` rather than swallowing the `.` into
//   the number. The check is "if we see `.`, peek the next char; if
//   it's `.` or `=`, stop here."

static bool isAlphaDigit(char c) {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
	       (c >= 'A' && c <= 'Z');
}

void Lexer::number() { scanNumberBody(); }

void Lexer::negativeNumber() {
	advance();  // consume the first digit after the leading `-`
	scanNumberBody();
}

void Lexer::scanNumberBody() {
	// State machine: int → int_period → float → float_exp. We track
	// only enough state to know whether `+`/`-` is valid (it is only
	// after a p/P/e/E exponent letter).
	enum NumState { S_INT, S_INT_PERIOD, S_FLOAT, S_FLOAT_EXP };
	NumState state = S_INT;

	while (true) {
		char c = peek();
		bool stop = false;

		switch (state) {
		case S_INT:
			if (c == '.') {
				// Distinguish `1.5` (float) from `0..=9` (range op):
				// if the next char is `.` or `=`, this `.` does NOT
				// belong to the number.
				char n = peekNext();
				if (n == '.' || n == '=') {
					stop = true;
					break;
				}
				advance();
				state = S_INT_PERIOD;
			} else if (c == '_' || isAlphaDigit(c)) {
				if (c == 'e' || c == 'E' || c == 'p' || c == 'P') {
					advance();
					state = S_FLOAT_EXP;
				} else {
					advance();
				}
			} else {
				stop = true;
			}
			break;
		case S_INT_PERIOD:
			if (c == '_' || isAlphaDigit(c)) {
				if (c == 'e' || c == 'E' || c == 'p' || c == 'P') {
					advance();
					state = S_FLOAT_EXP;
				} else {
					advance();
					state = S_FLOAT;
				}
			} else {
				// `1.` followed by something non-numeric: rewind the
				// `.` so it can be tokenized separately as a member
				// access or other operator.
				current--;
				stop = true;
			}
			break;
		case S_FLOAT:
			if (c == '_' || isAlphaDigit(c)) {
				if (c == 'e' || c == 'E' || c == 'p' || c == 'P') {
					advance();
					state = S_FLOAT_EXP;
				} else {
					advance();
				}
			} else {
				stop = true;
			}
			break;
		case S_FLOAT_EXP:
			if (c == '+' || c == '-') {
				advance();
				state = S_FLOAT;
			} else {
				// The sign was optional; bounce to .float and try again.
				state = S_FLOAT;
			}
			break;
		}

		if (stop) break;
	}

	addToken(TOK_NUMBER);
}

char Lexer::parseHexByte() {
	// Parse two hex digits
	char value = 0;
	for (int i = 0; i < 2; i++) {
		char c = peek();
		if (c >= '0' && c <= '9') {
			value = value * 16 + (c - '0');
		} else if (c >= 'a' && c <= 'f') {
			value = value * 16 + (c - 'a' + 10);
		} else if (c >= 'A' && c <= 'F') {
			value = value * 16 + (c - 'A' + 10);
		} else {
			throw std::runtime_error("Expected hex digit at line " +
			                         std::to_string(line));
		}
		advance();
	}
	return value;
}

std::string Lexer::parseUnicodeEscape() {
	// Parse \u{HHHHH} - Unicode codepoint (1-6 hex digits)
	if (peek() != '{') {
		throw std::runtime_error("Expected '{' after \\u at line " +
		                         std::to_string(line));
	}
	advance();  // consume '{'

	uint32_t codepoint = 0;
	int digitCount = 0;

	while (peek() != '}' && !isAtEnd()) {
		char c = peek();
		if (c >= '0' && c <= '9') {
			codepoint = codepoint * 16 + (c - '0');
		} else if (c >= 'a' && c <= 'f') {
			codepoint = codepoint * 16 + (c - 'a' + 10);
		} else if (c >= 'A' && c <= 'F') {
			codepoint = codepoint * 16 + (c - 'A' + 10);
		} else {
			throw std::runtime_error(
			    "Expected hex digit in unicode escape at line " +
			    std::to_string(line));
		}
		advance();
		digitCount++;

		if (digitCount > 6) {
			throw std::runtime_error(
			    "Unicode escape too long (max 6 hex digits) at line " +
			    std::to_string(line));
		}
	}

	if (isAtEnd() || peek() != '}') {
		throw std::runtime_error(
		    "Expected '}' to close unicode escape at line " +
		    std::to_string(line));
	}
	advance();  // consume '}'

	if (digitCount == 0) {
		throw std::runtime_error("Empty unicode escape at line " +
		                         std::to_string(line));
	}

	if (codepoint > 0x10FFFF) {
		throw std::runtime_error(
		    "Unicode codepoint out of range (max 0x10FFFF) at line " +
		    std::to_string(line));
	}

	// Encode codepoint as UTF-8
	std::string result;
	if (codepoint <= 0x7F) {
		result += static_cast<char>(codepoint);
	} else if (codepoint <= 0x7FF) {
		result += static_cast<char>(0xC0 | (codepoint >> 6));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	} else if (codepoint <= 0xFFFF) {
		result += static_cast<char>(0xE0 | (codepoint >> 12));
		result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	} else {
		result += static_cast<char>(0xF0 | (codepoint >> 18));
		result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
		result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (codepoint & 0x3F));
	}
	return result;
}

void Lexer::stringLiteral() {
	std::string value;

	while (peek() != '"' && !isAtEnd()) {
		if (peek() == '\n') {
			throw std::runtime_error("Unterminated string at line " +
			                         std::to_string(line));
		}

		if (peek() == '\\') {
			advance();  // consume backslash
			if (isAtEnd()) {
				throw std::runtime_error(
				    "Unterminated escape sequence at line " +
				    std::to_string(line));
			}

			char escaped = advance();
			switch (escaped) {
			case 'n':
				value += '\n';
				break;
			case 'r':
				value += '\r';
				break;
			case 't':
				value += '\t';
				break;
			case '\\':
				value += '\\';
				break;
			case '"':
				value += '"';
				break;
			case '\'':
				value += '\'';
				break;
			case '0':
				value += '\0';
				break;
			case 'x':
				value += parseHexByte();
				break;
			case 'u':
				value += parseUnicodeEscape();
				break;
			default:
				throw std::runtime_error("Invalid escape sequence '\\" +
				                         std::string(1, escaped) +
				                         "' at line " + std::to_string(line));
			}
		} else {
			value += advance();
		}
	}

	if (isAtEnd()) {
		throw std::runtime_error("Unterminated string at line " +
		                         std::to_string(line));
	}

	// The closing "
	advance();

	addToken(TOK_STRING_LITERAL, value);
}

std::vector<Token> Lexer::scanTokens() {
	while (!isAtEnd()) {
		skipWhitespace();
		if (isAtEnd()) break;

		// Snapshot the byte offset of the token we're about to scan.
		// Every `addToken` call below reads this via member state, so
		// each emitted Token carries its start offset, used downstream
		// for source-range and column resolution.
		tokenStart = static_cast<uint32_t>(current);
		char c = advance();

		switch (c) {
		case '(':
			addToken(TOK_OPEN_PAREN, "(");
			break;
		case ')':
			addToken(TOK_CLOSE_PAREN, ")");
			break;
		case '{':
			addToken(TOK_OPEN_BRACE, "{");
			break;
		case '}':
			addToken(TOK_CLOSE_BRACE, "}");
			break;
		case '[':
			addToken(TOK_OPEN_BRACKET, "[");
			break;
		case ']':
			addToken(TOK_CLOSE_BRACKET, "]");
			break;
		case ',':
			addToken(TOK_COMMA, ",");
			break;
		case ';':
			addToken(TOK_SEMI, ";");
			break;
		case ':':
			addToken(TOK_COLON, ":");
			break;
		case '+':
			addToken(TOK_PLUS, "+");
			break;
		case '.':
			if (peek() == '.' && peekNext() == '.') {
				advance();
				advance();
				addToken(TOK_ELLIPSIS, "...");
			} else if (peek() == '.' && peekNext() == '=') {
				advance();
				advance();
				addToken(TOK_DOTDOT_EQ, "..=");
			} else {
				addToken(TOK_DOT, ".");
			}
			break;
		case '"':
			stringLiteral();
			break;

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
				addToken(TOK_NOT, "!");
			}
			break;

		// `@` is the prefix for comptime-function invocations. The
		// identifier and arg list are lexed normally; the parser
		// recognizes the `@`+IDENTIFIER sequence as an AtCall.
		case '@':
			addToken(TOK_AT, "@");
			break;

		case '<':
			if (match('=')) {
				addToken(TOK_LESS_EQUAL, "<=");
			} else if (match('<')) {
				addToken(TOK_LSHIFT, "<<");
			} else {
				addToken(TOK_LESS, "<");
			}
			break;

		case '>':
			if (match('=')) {
				addToken(TOK_GREATER_EQUAL, ">=");
			} else if (match('>')) {
				addToken(TOK_RSHIFT, ">>");
			} else {
				addToken(TOK_GREATER, ">");
			}
			break;

		case '*':
			addToken(TOK_STAR, "*");
			break;
		case '/':
			// Bare `/` (not `//`, which skipWhitespace consumed as a
			// comment): the division operator.
			addToken(TOK_SLASH, "/");
			break;
		case '%':
			addToken(TOK_PERCENT, "%");
			break;
		case '&':
			if (match('&')) {
				addToken(TOK_AND, "&&");
			} else {
				addToken(TOK_AMP, "&");
			}
			break;
		case '|':
			if (match('|')) {
				addToken(TOK_OR, "||");
			} else {
				addToken(TOK_PIPE, "|");
			}
			break;
		case '^':
			addToken(TOK_CARET, "^");
			break;
		case '~':
			addToken(TOK_TILDE, "~");
			break;

		case '-':
			if (isDigit(peek())) {
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
				std::cerr << "Unexpected character at line " << line << ": "
				          << c << std::endl;
			}
			break;
		}
	}

	tokens.emplace_back(TOK_EOF, "", line);
	return tokens;
}
