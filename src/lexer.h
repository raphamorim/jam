/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef LEXER_H
#define LEXER_H

#include "token.h"
#include <string>
#include <vector>

class Lexer {
  private:
	std::string source;
	std::vector<Token> tokens;
	int current = 0;
	int line = 1;
	// Byte offset where the current token began — captured at the top
	// of each scan-loop iteration and recorded onto every emitted
	// Token. The persistent per-token info is just the start offset;
	// line and column are recomputed from this when needed.
	uint32_t tokenStart = 0;

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
	void addToken(TokenType type, const std::string &lexeme);
	void identifier();
	void number();
	void negativeNumber();
	void scanNumberBody();
	void stringLiteral();
	char parseHexByte();
	std::string parseUnicodeEscape();

  public:
	explicit Lexer(std::string source);
	std::vector<Token> scanTokens();

	// Const view of the source buffer the lexer holds. Used by the
	// parser to resolve `Token::text(source)` calls — every emitted
	// Token's `byteOffset`/`length` index into THIS buffer, so the
	// parser must reference the lexer's owned copy (or another
	// identical copy, but reusing the lexer's avoids a duplicate).
	const std::string &sourceBuffer() const { return source; }
};

#endif  // LEXER_H
