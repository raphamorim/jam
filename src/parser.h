/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "ast_flat.h"
#include "diagnostics.h"
#include "token.h"
#include <memory>
#include <unordered_set>
#include <vector>

// Thrown by the parser after pushing a diagnostic — the parser
// doesn't try to recover (would require synchronisation points the
// grammar doesn't define), so a single bad token ends the parse and
// main.cpp emits whatever is queued in the unified Diagnostics
// channel.
class ParserAbort {};

class Parser {
  private:
	std::vector<Token> tokens;
	int current = 0;
	TypePool *typePool;
	StringPool *stringPool;
	NodeStore *nodes;

	Token peek() const;
	Token previous() const;
	bool isAtEnd() const;
	Token advance();
	bool check(TokenType type) const;
	bool match(TokenType type);
	void consume(TokenType type, const std::string &message);

	NodeIdx emit(AstNode n);

	// Decode a numeric literal lexeme into (magnitude, isNeg, isFloat).
	// Member rather than free function so it can route over-large /
	// malformed literals through `parseError` with the token's line.
	uint64_t parseNumLexeme(const std::string &s, bool &isNegOut,
	                         bool &isFloatOut) const;

	NodeIdx parsePrimary();
	NodeIdx parseUnary();
	TypeIdx parseType();
	NodeIdx parseExpression();
	NodeIdx parseLogicalOr();
	NodeIdx parseLogicalAnd();
	NodeIdx parseComparison();
	NodeIdx parseBitwise();
	NodeIdx parseShift();
	NodeIdx parseAddition();
	NodeIdx parseMultiplication();
	NodeIdx parseStructLiteral();
	NodeIdx parseStructExpression();
	NodeIdx parseEnumExpression();
	NodeIdx parseMatch();
	NodeIdx parsePattern();      // OrPattern
	NodeIdx parsePatternAtom();  // single-atom pattern
	std::unique_ptr<FunctionAST> parseFunction();
	std::unique_ptr<ImportDeclAST> parseImportDecl();
	std::unique_ptr<DestructuringImportDeclAST> parseDestructuringImport();
	std::unique_ptr<StructDeclAST> parseStructDecl();
	void parseStructBody(std::vector<std::pair<std::string, TypeIdx>> &fields,
	                     std::vector<std::unique_ptr<FunctionAST>> &methods);
	std::vector<std::unique_ptr<StructDeclAST>> anonStructs;
	std::vector<std::unique_ptr<EnumDeclAST>> anonEnums;
	std::vector<std::string> structContextStack;
	std::unordered_set<std::string> importHandles;
	std::unique_ptr<UnionDeclAST> parseUnionDecl();
	std::unique_ptr<EnumDeclAST> parseEnumDecl();
	std::unique_ptr<ConstDeclAST> parseConstDecl();
	std::string qualifiedName(NodeIdx chainRoot) const;
	bool isQualifiedNameChain(NodeIdx chainRoot) const;

  public:
	Parser(std::vector<Token> tokens, TypePool &typePool,
	       StringPool &stringPool, NodeStore &nodes,
	       jam::Diagnostics *diagnostics = nullptr,
	       std::string filename = "");
	std::unique_ptr<ModuleAST> parse();
	std::vector<std::unique_ptr<StructDeclAST>> *sharedAnonStructs = nullptr;
	std::vector<std::unique_ptr<EnumDeclAST>> *sharedAnonEnums = nullptr;

  private:
	jam::Diagnostics *diagnostics_ = nullptr;
	std::string filename_;

	// Build a SrcLoc anchored at the current token (or the most
	// recently consumed token if `current` ran past the end). All
	// parser error helpers funnel through here.
	jam::SrcLoc currentLoc() const;
	[[noreturn]] void parseError(std::string message) const;
};

#endif  // PARSER_H
