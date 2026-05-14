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
#include "token.h"
#include <memory>
#include <unordered_set>
#include <vector>

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
	       StringPool &stringPool, NodeStore &nodes);
	std::unique_ptr<ModuleAST> parse();
	std::vector<std::unique_ptr<StructDeclAST>> *sharedAnonStructs = nullptr;
	std::vector<std::unique_ptr<EnumDeclAST>> *sharedAnonEnums = nullptr;
};

#endif  // PARSER_H
