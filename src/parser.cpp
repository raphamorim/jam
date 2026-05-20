/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "parser.h"
#include "number_literal.h"
#include <cstring>

// Validate a number lexeme via the dedicated validator
// (number_literal.cpp). Strips a leading `-` sign before delegating —
// negation is the caller's responsibility, since it depends on context
// (negative literal vs. unary minus on a literal).
//
// Returns the 64-bit magnitude. For float literals the bit pattern of
// the parsed `double` is returned via `bit_cast`, and `isFloatOut` is
// set so the caller can mark the AST node with the float flag.
uint64_t Parser::parseNumLexeme(std::string_view s, bool &isNegOut,
                                bool &isFloatOut) const {
	bool neg = !s.empty() && s[0] == '-';
	std::string_view abs = neg ? s.substr(1) : s;
	isNegOut = neg;
	isFloatOut = false;

	NumberResult r = parseNumberLiteral(abs);
	switch (r.kind) {
	case NumberResultKind::Int:
		return r.intValue;
	case NumberResultKind::BigInt:
		parseError("integer literal `" + std::string(abs) +
		           "` exceeds u64 range");
	case NumberResultKind::Float: {
		isFloatOut = true;
		// pack the double's bit pattern into u64. memcpy keeps it
		// strictly aliasing-safe; the codegen reinterprets back to
		// double via `bit_cast` in numberLitConst.
		uint64_t bits = 0;
		double v = r.floatValue;
		static_assert(sizeof(v) == sizeof(bits), "double must be 8 bytes");
		std::memcpy(&bits, &v, sizeof(bits));
		return bits;
	}
	case NumberResultKind::Failure:
		parseError(std::string("invalid numeric literal `") + std::string(abs) +
		           "`: " + numberErrorMessage(r.failure.kind));
	}
	parseError("unreachable number-literal classification");
}

Parser::Parser(std::vector<Token> tokens, const std::string &source,
               TypePool &typePool_, StringPool &stringPool_, NodeStore &nodes_,
               jam::Diagnostics *diagnostics, std::string filename)
    : tokens(std::move(tokens)), source_(source), typePool(&typePool_),
      stringPool(&stringPool_), nodes(&nodes_), diagnostics_(diagnostics),
      filename_(std::move(filename)) {}

jam::SrcLoc Parser::currentLoc() const {
	int line = 0;
	if (!tokens.empty()) {
		int idx =
		    current < static_cast<int>(tokens.size()) ? current : current - 1;
		if (idx < 0) idx = 0;
		line = tokens[idx].line;
	}
	return jam::SrcLoc{filename_, line};
}

void Parser::parseError(std::string message) const {
	if (diagnostics_) { diagnostics_->error(currentLoc(), std::move(message)); }
	throw ParserAbort{};
}

Token Parser::peek() const { return tokens[current]; }

Token Parser::previous() const { return tokens[current - 1]; }

bool Parser::isAtEnd() const { return peek().type == TOK_EOF; }

Token Parser::advance() {
	if (!isAtEnd()) current++;
	return previous();
}

bool Parser::check(TokenType type) const {
	if (isAtEnd()) return false;
	return peek().type == type;
}

bool Parser::match(TokenType type) {
	if (check(type)) {
		advance();
		return true;
	}
	return false;
}

void Parser::consume(TokenType type, const std::string &message) {
	if (check(type)) {
		advance();
		return;
	}
	parseError(message);
}

NodeIdx Parser::emit(AstNode n) {
	// Most call sites pass mainToken=0; default to "the most recently
	// consumed token" so error messages report meaningful line numbers
	// without forcing every emit site to thread the token index. Sites
	// that need a specific token (e.g. a construct's start keyword
	// rather than its closing punctuation) can set mainToken explicitly.
	if (n.mainToken == 0 && current > 0) {
		n.mainToken = static_cast<uint32_t>(current - 1);
	}
	// Record source line for codegen-time diagnostics. The mainToken's
	// line lets `file:line:` prefixes appear on errors thrown deep in
	// codegen even though the AST itself is positionless.
	int line = 0;
	if (n.mainToken < tokens.size()) { line = tokens[n.mainToken].line; }
	return nodes->addNodeAt(n, line);
}

// Walk a chain of MemberAccess nodes back to its root Variable and produce
// the dotted qualified name. Used to resolve `std.fmt.println`-style call
// targets at parse time so the codegen can switch on the full name.
std::string Parser::qualifiedName(NodeIdx chainRoot) const {
	const AstNode &n = nodes->get(chainRoot);
	if (n.tag == AstTag::Variable) {
		return stringPool->get(static_cast<StringIdx>(n.lhs));
	}
	if (n.tag == AstTag::MemberAccess) {
		std::string base = qualifiedName(static_cast<NodeIdx>(n.lhs));
		const std::string &member =
		    stringPool->get(static_cast<StringIdx>(n.rhs));
		return base + "." + member;
	}
	parseError("Invalid member access chain");
}

bool Parser::isQualifiedNameChain(NodeIdx chainRoot) const {
	const AstNode &n = nodes->get(chainRoot);
	if (n.tag == AstTag::Variable) return true;
	if (n.tag == AstTag::MemberAccess) {
		return isQualifiedNameChain(static_cast<NodeIdx>(n.lhs));
	}
	return false;
}

// Walk the chain root (leftmost Variable in a Variable.member.member...
// chain) and return its source-level name. Caller guarantees the chain
// is a qualified-name shape.
static std::string chainRootName(const NodeStore &ns,
                                  const StringPool &pool, NodeIdx chainRoot) {
	const AstNode &n = ns.get(chainRoot);
	if (n.tag == AstTag::Variable) {
		return pool.get(static_cast<StringIdx>(n.lhs));
	}
	if (n.tag == AstTag::MemberAccess) {
		return chainRootName(ns, pool, static_cast<NodeIdx>(n.lhs));
	}
	return std::string();
}

// Count the number of dots (== MemberAccess hops) in a qualified-name
// chain. `foo` → 0, `foo.bar` → 1, `foo.bar.baz` → 2, etc.
static int chainDotCount(const NodeStore &ns, NodeIdx chainRoot) {
	const AstNode &n = ns.get(chainRoot);
	if (n.tag != AstTag::MemberAccess) return 0;
	return 1 + chainDotCount(ns, static_cast<NodeIdx>(n.lhs));
}

NodeIdx Parser::parsePrimary() {
	// `match (…) { … }` is also valid in expression position so it can
	// produce a value. The same call works for both statement and
	// expression forms; the codegen builds a phi over arm values.
	if (check(TOK_MATCH)) { return parseMatch(); }

	// `@name(arg)` — comptime intrinsic invocation. Resolved to a
	// constant at codegen time; LLVM never sees a call. Stage 1 only
	// supports compiler-supplied intrinsics that take a single TYPE
	// argument (sizeOf, alignOf); the type is parsed via parseType()
	// and stored as a TypeIdx in the rhs slot. User-defined cfn bodies
	// + arbitrary value args arrive in Stage 2 with CTFE.
	if (match(TOK_AT)) {
		consume(TOK_IDENTIFIER, "Expected intrinsic name after '@'");
		StringIdx nameId = stringPool->intern(previous().text(source_));
		consume(TOK_OPEN_PAREN, "Expected '(' after '@name'");
		TypeIdx tyArg = parseType();
		consume(TOK_CLOSE_PAREN, "Expected ')' after '@' intrinsic argument");
		return emit(AstNode{AstTag::AtCall, 0, 0, 0, nameId,
		                    static_cast<uint32_t>(tyArg)});
	}

	if (match(TOK_NUMBER)) {
		// `parseNumLexeme` returns the magnitude; the sign is recorded in
		// the node's flags bit 0 (negative) and bit 1 (isFloat: when
		// set the magnitude is the bit pattern of a `double`).
		bool isNegative = false;
		bool isFloat = false;
		uint64_t mag =
		    parseNumLexeme(previous().text(source_), isNegative, isFloat);
		uint16_t flags = 0;
		if (isNegative) flags |= 1;
		if (isFloat) flags |= 2;
		AstNode n{AstTag::NumberLit,
		          0,
		          flags,
		          0,
		          static_cast<uint32_t>(mag & 0xFFFFFFFFu),
		          static_cast<uint32_t>(mag >> 32)};
		return emit(n);
	}
	if (match(TOK_TRUE)) {
		return emit(AstNode{AstTag::BoolLit, 0, 0, 0, 1, 0});
	}
	if (match(TOK_FALSE)) {
		return emit(AstNode{AstTag::BoolLit, 0, 0, 0, 0, 0});
	}
	if (match(TOK_STRING_LITERAL)) {
		// String literals carry decoded bytes (escape sequences
		// resolved); `text(source_)` would return the raw `"..."`-
		// bracketed source including escapes — wrong for callers
		// that want the runtime string value.
		StringIdx s = stringPool->intern(previous().lexeme);
		return emit(AstNode{AstTag::StringLit, 0, 0, 0, s, 0});
	}
	if (match(TOK_IMPORT)) {
		consume(TOK_OPEN_PAREN, "Expected '(' after 'import'");
		consume(TOK_STRING_LITERAL, "Expected string literal for import path");
		StringIdx path = stringPool->intern(previous().lexeme);
		consume(TOK_CLOSE_PAREN, "Expected ')' after import path");
		return emit(AstNode{AstTag::ImportLit, 0, 0, 0, path, 0});
	}
	if (match(TOK_OPEN_PAREN)) {
		NodeIdx expr = parseLogicalOr();
		consume(TOK_CLOSE_PAREN, "Expected ')' after expression");
		return expr;
	}
	// Bare `{ field: val }` is no longer a valid expression. Struct
	// literals always carry their type: `Foo { field: val }` or
	// `Vec(i32) { field: val }` (handled when an identifier precedes
	// the brace, in the TOK_IDENTIFIER branch below). A naked brace
	// in expression position is a parse error.
	if (check(TOK_OPEN_BRACE)) {
		parseError("Struct literals must name their type: write "
		           "`TypeName { ... }` (or `Self { ... }` inside a "
		           "struct body)");
	}
	if (match(TOK_OPEN_BRACKET)) {
		// Array literal `[a, b, c]`, array repeat `[expr; N]`, or empty
		// `[]` (well-typed only against a slice or zero-length array
		// target — codegen rejects others).
		if (match(TOK_CLOSE_BRACKET)) {
			ExtraIdx extra = nodes->reserveExtra(1);
			nodes->setExtra(extra, 0);
			return emit(AstNode{AstTag::ArrayLit, 0, 0, 0, kNoType, extra});
		}
		NodeIdx first = parseLogicalOr();
		if (match(TOK_SEMI)) {
			NodeIdx count = parseLogicalOr();
			consume(TOK_CLOSE_BRACKET, "Expected `]` after array repeat count");
			ExtraIdx extra = nodes->reserveExtra(2);
			nodes->setExtra(extra, static_cast<uint32_t>(first));
			nodes->setExtra(extra + 1, static_cast<uint32_t>(count));
			return emit(AstNode{AstTag::ArrayRepeat, 0, 0, 0, kNoType, extra});
		}
		std::vector<NodeIdx> elems;
		elems.push_back(first);
		while (match(TOK_COMMA)) {
			if (check(TOK_CLOSE_BRACKET)) break;  // trailing comma
			elems.push_back(parseLogicalOr());
		}
		consume(TOK_CLOSE_BRACKET, "Expected `]` or `,` in array literal");
		ExtraIdx extra = nodes->reserveExtra(1 + elems.size());
		nodes->setExtra(extra, static_cast<uint32_t>(elems.size()));
		for (size_t i = 0; i < elems.size(); i++) {
			nodes->setExtra(extra + 1 + i, elems[i]);
		}
		return emit(AstNode{AstTag::ArrayLit, 0, 0, 0, kNoType, extra});
	}
	if (match(TOK_STRUCT)) { return parseStructExpression(); }
	// Same shape for tagged sum types: `enum { Variant1, Variant2(T), ... }`
	// in expression position. Enables generic enums like
	// `Option(T)`/`Result(T, E)`; variant payload types may reference
	// the generic params and get substituted at instantiation time.
	if (match(TOK_ENUM)) { return parseEnumExpression(); }
	// Builtin type names (`i32`, `u8`, `bool`, ...) appear in
	// expression position only in const-decl RHS used as a type-alias
	// argument: `const ListI32 = Vec(i32);`. We let the lexer's
	// TOK_TYPE flow through as a synthetic Identifier so the regular
	// Call-expression machinery handles `Vec(i32)` uniformly. Type-
	// alias resolution at registerConsts in main.cpp recognises the
	// shape (Call to a generic-yielding-type fn whose args resolve as
	// types) and binds the alias. Anywhere else this Identifier hits
	// astgen as a value, it falls through to the "unknown variable"
	// diagnostic — same outcome as today's grammar would have given
	// for `const X = i32;`.
	if (match(TOK_TYPE) || match(TOK_IDENTIFIER)) {
		std::string name(previous().text(source_));
		StringIdx nameId = stringPool->intern(name);

		// Typed struct literal: `Name { field: val, ... }`. Only
		// admitted in expression position when struct literals are
		// allowed (see `allowStructLit_`); inside `if`/`while`/`for`/
		// `match` heads the same shape stays as a Variable read
		// followed by the head's `{` block opener.
		if (allowStructLit_ && check(TOK_OPEN_BRACE)) {
			advance();  // consume '{'
			NodeIdx lit = parseStructLiteral();
			// Resolve `Self` against the current struct context so
			// `return Self { ... }` works inside a method body. For
			// any other identifier we intern the Named TypeIdx as-is
			// — codegen does the struct/enum lookup at use time.
			StringIdx typeNameId = nameId;
			if (name == "Self") {
				if (structContextStack.empty()) {
					parseError("`Self` is only valid inside a struct body");
				}
				typeNameId = stringPool->intern(structContextStack.back());
			}
			AstNode &litNode = nodes->getMut(lit);
			litNode.lhs = static_cast<uint32_t>(
			    typePool->internNamed(typeNameId));
			return lit;
		}

		NodeIdx expr = emit(AstNode{AstTag::Variable, 0, 0, 0, nameId, 0});
		bool chainStarted = false;

		// `.field`, `.*`, `[i]`, and `(args)`
		while (check(TOK_DOT) || check(TOK_OPEN_BRACKET) ||
		       check(TOK_OPEN_PAREN)) {
			if (match(TOK_DOT)) {
				chainStarted = true;
				if (match(TOK_STAR)) {
					expr = emit(AstNode{AstTag::Deref, 0, 0, 0,
					                    static_cast<uint32_t>(expr), 0});
				} else {
					consume(TOK_IDENTIFIER, "Expected member name after '.'");
					StringIdx mem =
					    stringPool->intern(previous().text(source_));
					expr = emit(AstNode{AstTag::MemberAccess, 0, 0, 0,
					                    static_cast<uint32_t>(expr), mem});
				}
				continue;
			}
			if (match(TOK_OPEN_BRACKET)) {
				chainStarted = true;
				NodeIdx idx = parseLogicalOr();
				consume(TOK_CLOSE_BRACKET, "Expected ']' after index");
				expr = emit(AstNode{AstTag::Index, 0, 0, 0,
				                    static_cast<uint32_t>(expr),
				                    static_cast<uint32_t>(idx)});
				continue;
			}

			// TOK_OPEN_PAREN: typecall on a generic — either bare
			// `Foo(T).method(args)` (chainStarted=false) or namespace-
			// qualified `handle.Foo(T).method(args)` when `handle` was
			// bound by `import(...)`. Also the generic-struct-literal
			// shape `Foo(T) { field: val }`.
			bool isNamespacedTypecall =
			    chainStarted && importHandles.count(name) > 0;
			if (!chainStarted || isNamespacedTypecall) {
				int peekIdx = current + 1;  // first token after (
				int depth = 1;
				while (peekIdx < (int)tokens.size() && depth > 0) {
					TokenType tt = tokens[peekIdx].type;
					if (tt == TOK_OPEN_PAREN) depth++;
					else if (tt == TOK_CLOSE_PAREN) depth--;
					if (depth == 0) break;
					peekIdx++;
				}
				// Generic struct literal: `Foo(T) { ... }`. The matching
				// `)` is immediately followed by `{`. Parse the parens
				// as type arguments, intern as a GenericCall TypeIdx,
				// then hand off to parseStructLiteral so the body uses
				// the same name:value layout as a non-generic literal.
				if (allowStructLit_ && depth == 0 &&
				    peekIdx + 1 < (int)tokens.size() &&
				    tokens[peekIdx].type == TOK_CLOSE_PAREN &&
				    tokens[peekIdx + 1].type == TOK_OPEN_BRACE) {
					advance();  // consume (
					std::vector<TypeIdx> typeArgs;
					if (!check(TOK_CLOSE_PAREN)) {
						do {
							typeArgs.push_back(parseType());
						} while (match(TOK_COMMA));
					}
					consume(TOK_CLOSE_PAREN,
					        "Expected ')' after type arguments");
					consume(TOK_OPEN_BRACE,
					        "Expected '{' after generic type arguments");
					std::string receiverName =
					    isNamespacedTypecall ? qualifiedName(expr) : name;
					TypeIdx genericTy = typePool->internGenericCall(
					    stringPool->intern(receiverName),
					    std::move(typeArgs));
					NodeIdx lit = parseStructLiteral();
					AstNode &litNode = nodes->getMut(lit);
					litNode.lhs = static_cast<uint32_t>(genericTy);
					return lit;
				}
				if (depth == 0 && peekIdx + 3 < (int)tokens.size() &&
				    tokens[peekIdx].type == TOK_CLOSE_PAREN &&
				    tokens[peekIdx + 1].type == TOK_DOT &&
				    tokens[peekIdx + 2].type == TOK_IDENTIFIER &&
				    tokens[peekIdx + 3].type == TOK_OPEN_PAREN) {
					advance();  // consume (
					std::vector<TypeIdx> typeArgs;
					if (!check(TOK_CLOSE_PAREN)) {
						do {
							typeArgs.push_back(parseType());
						} while (match(TOK_COMMA));
					}
					consume(TOK_CLOSE_PAREN,
					        "Expected ')' after type arguments");
					consume(TOK_DOT, "Expected '.' after generic call");
					consume(TOK_IDENTIFIER, "Expected method name after '.'");
					StringIdx methodName =
					    stringPool->intern(previous().text(source_));
					consume(TOK_OPEN_PAREN, "Expected '(' after method name");

					std::vector<NodeIdx> methodArgs;
					if (!check(TOK_CLOSE_PAREN)) {
						do {
							methodArgs.push_back(parseComparison());
						} while (match(TOK_COMMA));
					}
					consume(TOK_CLOSE_PAREN,
					        "Expected ')' after method arguments");

					// For `handle.Vec(T)...`, the receiver type's name
					// is the dotted chain so it matches the qualified
					// alias main.cpp registered under handle.Vec.
					std::string receiverName =
					    isNamespacedTypecall ? qualifiedName(expr) : name;
					TypeIdx receiverTy = typePool->internGenericCall(
					    stringPool->intern(receiverName), std::move(typeArgs));

					ExtraIdx extra = nodes->reserveExtra(2 + methodArgs.size());
					nodes->setExtra(extra, static_cast<uint32_t>(methodName));
					nodes->setExtra(extra + 1,
					                static_cast<uint32_t>(methodArgs.size()));
					for (size_t i = 0; i < methodArgs.size(); i++) {
						nodes->setExtra(extra + 2 + i,
						                static_cast<uint32_t>(methodArgs[i]));
					}
					expr =
					    emit(AstNode{AstTag::TypeMethodCall, 0, 0, 0,
					                 static_cast<uint32_t>(receiverTy), extra});
					chainStarted = true;
					continue;
				}
			}

			advance();  // consume (
			std::vector<NodeIdx> args;
			if (!check(TOK_CLOSE_PAREN)) {
				do {
					args.push_back(parseComparison());
				} while (match(TOK_COMMA));
			}
			consume(TOK_CLOSE_PAREN, "Expected ')' after function arguments");

			// extra layout (shared by both Call encodings):
			//   [argCount, arg0, arg1, ...]
			ExtraIdx extra = nodes->reserveExtra(1 + args.size());
			nodes->setExtra(extra, static_cast<uint32_t>(args.size()));
			for (size_t i = 0; i < args.size(); i++) {
				nodes->setExtra(extra + 1 + i, args[i]);
			}

			if (isQualifiedNameChain(expr)) {
				// Qualified-name chains can be one of:
				//   1. `mod.fn(...)` / `mod.submod.fn(...)` — module-
				//      handle prefixed call; root is in `importHandles`.
				//   2. `Type.method(...)` — static dispatch on a type
				//      name; single dot, root is a struct/enum/alias.
				//   3. `local.field.method(...)` — instance dispatch
				//      with a multi-segment receiver path; root is a
				//      local variable in the caller's scope.
				//
				// Cases 1 and 2 stay as direct calls (lhs = qualified
				// StringIdx); astgen resolves them via the function
				// registry or the Type.method handler. Case 3 only
				// works through the indirect-call path (flags & 1):
				// astgen needs the MemberAccess chain so it can lower
				// `local.field` as a value-producing expression and
				// dispatch the suffix as a method on that value.
				//
				// We can't tell at parse time whether the root is a
				// local — the parser has no scope info. But we know
				// (a) what's in `importHandles`, and (b) the dot
				// count. A multi-dot chain whose root is NOT an
				// import handle is case 3 by elimination: case 1
				// would need an import handle, case 2 is single-dot.
				int dotCount = chainDotCount(*nodes, expr);
				std::string root = chainRootName(*nodes, *stringPool, expr);
				bool rootIsImportHandle = importHandles.count(root) > 0;
				bool isMultiDotLocalChain =
				    dotCount >= 2 && !rootIsImportHandle;
				if (isMultiDotLocalChain) {
					expr = emit(AstNode{AstTag::Call, 0, 1, 0,
					                    static_cast<uint32_t>(expr), extra});
				} else {
					std::string callee;
					const AstNode &en = nodes->get(expr);
					if (en.tag == AstTag::MemberAccess) {
						callee = qualifiedName(expr);
					} else {
						callee = name;
					}
					StringIdx calleeId = stringPool->intern(callee);
					expr = emit(AstNode{AstTag::Call, 0, 0, 0, calleeId, extra});
				}
			} else {
				expr = emit(AstNode{AstTag::Call, 0, 1, 0,
				                    static_cast<uint32_t>(expr), extra});
			}
			chainStarted = true;
		}

		// Qualified-name struct literal: `lib.Private { field: val }`.
		// After the chain loop finishes, if the expression is a
		// MemberAccess chain rooted at a Variable (no calls/indices
		// in between) and the next token is `{`, the chain names a
		// type. Build the Named TypeIdx from the dotted form and
		// parse the body. Same shape as the simple `Name { ... }`
		// path above, but the type is the multi-segment qualified
		// name so codegen looks it up via the import-handle alias.
		// Mirrors Rust's `path::Type { ... }`.
		if (allowStructLit_ && check(TOK_OPEN_BRACE) &&
		    isQualifiedNameChain(expr)) {
			const AstNode &exprNode = nodes->get(expr);
			if (exprNode.tag == AstTag::MemberAccess) {
				advance();  // consume '{'
				std::string qname = qualifiedName(expr);
				TypeIdx namedTy =
				    typePool->internNamed(stringPool->intern(qname));
				NodeIdx lit = parseStructLiteral();
				AstNode &litNode = nodes->getMut(lit);
				litNode.lhs = static_cast<uint32_t>(namedTy);
				return lit;
			}
		}

		return expr;
	}

	parseError("Expected primary expression");
}

// Type grammar:
//
//   *const T     — single-item pointer, read-only pointee
//   *mut T       — single-item pointer, writable pointee
//   *const[] T   — many-item pointer, read-only pointee
//   *mut[] T     — many-item pointer, writable pointee
//   []T          — slice (mutability follows the binding)
//   [N]T         — fixed-size array (mutability follows the binding;
//                  `N` is part of the type)
TypeIdx Parser::parseType() {
	if (match(TOK_STAR)) {
		bool ptrConst = false;
		if (match(TOK_CONST)) {
			ptrConst = true;
		} else if (!match(TOK_MUT)) {
			parseError("Expected `const` or `mut` after `*` (e.g. `*const T`, "
			           "`*mut T`)");
		}
		// Optional `[]` promotes to many-item form. We need to commit to
		// "many" only when the next two tokens are exactly `[ ]`; a `[`
		// followed by a number is the start of a `[N]T` element type
		// (single-item pointer to a fixed array).
		bool isMany = false;
		if (check(TOK_OPEN_BRACKET)) {
			int saved = current;
			advance();  // consume `[`
			if (match(TOK_CLOSE_BRACKET)) {
				isMany = true;
			} else {
				current = saved;  // rewind: `[` belongs to element type
			}
		}
		TypeIdx elem = parseType();
		(void)ptrConst;  // mutability not enforced yet
		return isMany ? typePool->internPtrMany(elem)
		              : typePool->internPtrSingle(elem);
	}
	if (match(TOK_OPEN_BRACKET)) {
		// `[]T` — slice. No tag.
		if (match(TOK_CLOSE_BRACKET)) {
			return typePool->internSlice(parseType());
		}
		// `[N]T` — fixed-size array. No tag.
		consume(TOK_NUMBER, "Expected size or `]` after `[`");
		uint32_t len = static_cast<uint32_t>(
		    std::stoul(std::string(previous().text(source_))));
		consume(TOK_CLOSE_BRACKET, "Expected `]` after array size");
		return typePool->internArray(parseType(), len);
	}
	if (match(TOK_TYPE)) {
		std::string_view s = previous().text(source_);
		if (s == "u8") return BuiltinType::U8;
		if (s == "i8") return BuiltinType::I8;
		if (s == "u16") return BuiltinType::U16;
		if (s == "i16") return BuiltinType::I16;
		if (s == "u32") return BuiltinType::U32;
		if (s == "i32") return BuiltinType::I32;
		if (s == "u64") return BuiltinType::U64;
		if (s == "i64") return BuiltinType::I64;
		if (s == "f32") return BuiltinType::F32;
		if (s == "f64") return BuiltinType::F64;
		if (s == "bool" || s == "u1") return BuiltinType::Bool;
		if (s == "str") return typePool->internSlice(BuiltinType::U8);
		if (s == "type") return BuiltinType::Type;
		if (s == "noreturn") return BuiltinType::NoReturn;
		parseError("Unknown base type: " + std::string(s));
	}
	if (match(TOK_IDENTIFIER)) {
		std::string_view firstIdent = previous().text(source_);
		if (firstIdent == "Self") {
			if (structContextStack.empty()) {
				parseError("`Self` is only valid inside a struct body");
			}
			return typePool->internNamed(
			    stringPool->intern(structContextStack.back()));
		}
		// Optional module qualifier: `Handle.TypeName`. main.cpp
		// registers each main-module import's pub items under
		// `<handle>.<name>` so the existing struct/enum/union/type-alias
		// and generic-fn lookups resolve the qualified form transparently.
		std::string ident(firstIdent);
		if (match(TOK_DOT)) {
			consume(TOK_IDENTIFIER, "Expected type name after `.`");
			ident = std::string(firstIdent) + "." +
			        std::string(previous().text(source_));
		}
		if (check(TOK_OPEN_PAREN)) {
			advance();  // consume `(`
			std::vector<TypeIdx> args;
			if (!check(TOK_CLOSE_PAREN)) {
				do { args.push_back(parseType()); } while (match(TOK_COMMA));
			}
			consume(TOK_CLOSE_PAREN,
			        "Expected ')' after generic type arguments");
			return typePool->internGenericCall(stringPool->intern(ident),
			                                   std::move(args));
		}
		return typePool->internNamed(stringPool->intern(ident));
	}
	parseError("Expected type");
}

NodeIdx Parser::parseStructLiteral() {
	// Caller has consumed '{'. Layout in extra:
	//   [fieldCount, name0, expr0, name1, expr1, ...]
	std::vector<std::pair<StringIdx, NodeIdx>> fields;
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		consume(TOK_IDENTIFIER, "Expected field name in struct literal");
		StringIdx fieldName = stringPool->intern(previous().text(source_));
		consume(TOK_COLON, "Expected ':' after field name");
		NodeIdx value = parseLogicalOr();
		fields.emplace_back(fieldName, value);
		if (!match(TOK_COMMA)) break;
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' to close struct literal");

	ExtraIdx extra = nodes->reserveExtra(1 + fields.size() * 2);
	nodes->setExtra(extra, static_cast<uint32_t>(fields.size()));
	for (size_t i = 0; i < fields.size(); i++) {
		nodes->setExtra(extra + 1 + i * 2, fields[i].first);
		nodes->setExtra(extra + 2 + i * 2, fields[i].second);
	}
	// d.lhs holds the struct TypeIdx; codegen fills it from the use site
	// (var-decl target type or enclosing struct field type).
	return emit(AstNode{AstTag::StructLit, 0, 0, 0, kNoType, extra});
}

NodeIdx Parser::parsePatternAtom() {
	if (check(TOK_IDENTIFIER) && peek().text(source_) != "_") {
		int peekIdx = current + 1;
		if (peekIdx < (int)tokens.size() &&
		    tokens[peekIdx].type == TOK_OPEN_PAREN) {
			int depth = 1;
			int scan = peekIdx + 1;
			while (scan < (int)tokens.size() && depth > 0) {
				if (tokens[scan].type == TOK_OPEN_PAREN) depth++;
				else if (tokens[scan].type == TOK_CLOSE_PAREN) depth--;
				if (depth == 0) break;
				scan++;
			}
			if (depth == 0 && scan + 2 < (int)tokens.size() &&
			    tokens[scan].type == TOK_CLOSE_PAREN &&
			    tokens[scan + 1].type == TOK_DOT &&
			    tokens[scan + 2].type == TOK_IDENTIFIER) {
				std::string typeName(peek().text(source_));
				advance();  // IDENT
				advance();  // `(`
				std::vector<TypeIdx> typeArgs;
				if (!check(TOK_CLOSE_PAREN)) {
					do {
						typeArgs.push_back(parseType());
					} while (match(TOK_COMMA));
				}
				consume(TOK_CLOSE_PAREN,
				        "Expected ')' after generic type arguments");
				consume(TOK_DOT, "Expected '.' after generic type");
				consume(TOK_IDENTIFIER, "Expected variant name after `.`");
				StringIdx variantNameId =
				    stringPool->intern(previous().text(source_));
				TypeIdx receiverTy = typePool->internGenericCall(
				    stringPool->intern(typeName), std::move(typeArgs));

				if (match(TOK_OPEN_PAREN)) {
					std::vector<StringIdx> bindings;
					if (!check(TOK_CLOSE_PAREN)) {
						do {
							consume(TOK_IDENTIFIER,
							        "Expected binding name in variant payload");
							bindings.push_back(
							    stringPool->intern(previous().text(source_)));
						} while (match(TOK_COMMA));
					}
					consume(TOK_CLOSE_PAREN,
					        "Expected `)` to close payload bindings");
					// flags = 3 (bit 0 = bindings, bit 1 = TypeIdx recv)
					// extra: [typeIdx, variantNameId, count, b0, ...]
					ExtraIdx extra = nodes->reserveExtra(3 + bindings.size());
					nodes->setExtra(extra, static_cast<uint32_t>(receiverTy));
					nodes->setExtra(extra + 1, variantNameId);
					nodes->setExtra(extra + 2,
					                static_cast<uint32_t>(bindings.size()));
					for (size_t i = 0; i < bindings.size(); i++) {
						nodes->setExtra(extra + 3 + i, bindings[i]);
					}
					return emit(
					    AstNode{AstTag::PatEnumVariant, 0, 3u, 0, extra, 0});
				}
				// No bindings: flags = 2 (TypeIdx recv, no bindings).
				return emit(AstNode{AstTag::PatEnumVariant, 0, 2u, 0,
				                    static_cast<uint32_t>(receiverTy),
				                    variantNameId});
			}
		}
	}

	if (check(TOK_IDENTIFIER) && peek().text(source_) != "_") {
		int saved = current;
		advance();  // consume enum name
		if (match(TOK_DOT)) {
			consume(TOK_IDENTIFIER, "Expected variant name after `.`");
			StringIdx enumNameId =
			    stringPool->intern(tokens[saved].text(source_));
			StringIdx variantNameId =
			    stringPool->intern(previous().text(source_));

			// Optional payload binding: `(name1, name2, ...)`. Empty
			// list `()` is permitted and equivalent to no parens.
			if (match(TOK_OPEN_PAREN)) {
				std::vector<StringIdx> bindings;
				if (!check(TOK_CLOSE_PAREN)) {
					do {
						consume(TOK_IDENTIFIER,
						        "Expected binding name in variant payload");
						bindings.push_back(
						    stringPool->intern(previous().text(source_)));
					} while (match(TOK_COMMA));
				}
				consume(TOK_CLOSE_PAREN,
				        "Expected `)` to close payload bindings");
				ExtraIdx extra = nodes->reserveExtra(3 + bindings.size());
				nodes->setExtra(extra, enumNameId);
				nodes->setExtra(extra + 1, variantNameId);
				nodes->setExtra(extra + 2,
				                static_cast<uint32_t>(bindings.size()));
				for (size_t i = 0; i < bindings.size(); i++) {
					nodes->setExtra(extra + 3 + i, bindings[i]);
				}
				return emit(
				    AstNode{AstTag::PatEnumVariant, 0, 1u, 0, extra, 0});
			}
			// No bindings: lhs = enumNameId, rhs = variantNameId,
			// flags = 0.
			return emit(AstNode{AstTag::PatEnumVariant, 0, 0, 0, enumNameId,
			                    variantNameId});
		}
		current = saved;
		parseError("Bare identifier patterns are not yet supported "
		           "(use `EnumName.Variant`, an integer literal, or `_`)");
	}
	if (match(TOK_NUMBER)) {
		bool isNegative = false;
		bool isFloat = false;
		uint64_t lo =
		    parseNumLexeme(previous().text(source_), isNegative, isFloat);
		if (isFloat) {
			parseError("Float literals are not allowed in `match` patterns "
			           "(use an integer literal or a `..=` range)");
		}
		// Inclusive range `lo..=hi`?
		if (match(TOK_DOTDOT_EQ)) {
			consume(TOK_NUMBER, "Expected upper bound after `..=`");
			bool hiNeg = false;
			bool hiFloat = false;
			uint64_t hi =
			    parseNumLexeme(previous().text(source_), hiNeg, hiFloat);
			if (hiFloat) {
				parseError(
				    "Float literals are not allowed in `match` patterns");
			}
			// range bounds fit in u32; truncate gracefully.
			return emit(AstNode{AstTag::PatRange, 0, 0, 0,
			                    static_cast<uint32_t>(lo & 0xFFFFFFFFu),
			                    static_cast<uint32_t>(hi & 0xFFFFFFFFu)});
		}
		uint16_t flags = isNegative ? 1u : 0u;
		return emit(AstNode{AstTag::PatLit, 0, flags, 0,
		                    static_cast<uint32_t>(lo & 0xFFFFFFFFu),
		                    static_cast<uint32_t>(lo >> 32)});
	}
	// Char literal — TOK_STRING_LITERAL is currently the only string-y
	// token; we treat single-quote chars as TOK_NUMBER via the
	// lexer in a future patch. For now, only TOK_NUMBER is accepted.
	if (match(TOK_STRING_LITERAL)) {
		parseError("Char literals in patterns are not yet supported");
	}
	// Wildcard `_` is lexed as TOK_IDENTIFIER; recognize it here.
	if (check(TOK_IDENTIFIER) && peek().text(source_) == "_") {
		advance();
		return emit(AstNode{AstTag::PatWildcard, 0, 0, 0, 0, 0});
	}
	parseError("Expected pattern (integer literal, range, or `_`)");
}

NodeIdx Parser::parsePattern() {
	NodeIdx first = parsePatternAtom();
	if (!check(TOK_PIPE)) { return first; }
	std::vector<NodeIdx> alternatives;
	alternatives.push_back(first);
	while (match(TOK_PIPE)) { alternatives.push_back(parsePatternAtom()); }
	ExtraIdx extra = nodes->reserveExtra(1 + alternatives.size());
	nodes->setExtra(extra, static_cast<uint32_t>(alternatives.size()));
	for (size_t i = 0; i < alternatives.size(); i++) {
		nodes->setExtra(extra + 1 + i, alternatives[i]);
	}
	return emit(AstNode{AstTag::PatOr, 0, 0, 0, extra, 0});
}

NodeIdx Parser::parseMatch() {
	consume(TOK_MATCH, "Expected `match`");
	consume(TOK_OPEN_PAREN, "Expected `(` after `match`");
	NodeIdx scrutinee = parseLogicalOr();
	consume(TOK_CLOSE_PAREN, "Expected `)` after match scrutinee");
	consume(TOK_OPEN_BRACE, "Expected `{` to begin match body");

	struct Arm {
		NodeIdx pat;
		std::vector<NodeIdx> body;
	};
	std::vector<Arm> arms;

	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		Arm arm;
		arm.pat = parsePattern();
		consume(TOK_OPEN_BRACE, "Expected `{` to begin arm body");
		while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
			arm.body.push_back(parseExpression());
		}
		consume(TOK_CLOSE_BRACE, "Expected `}` to close arm body");
		arms.push_back(std::move(arm));
	}
	consume(TOK_CLOSE_BRACE, "Expected `}` to close match body");

	// Compute total extra size and pack the arms.
	size_t total = 1;  // armCount
	for (const Arm &a : arms) {
		total += 2 + a.body.size();  // patIdx + bodyCount + body...
	}

	ExtraIdx extra = nodes->reserveExtra(total);
	uint32_t pos = 0;
	nodes->setExtra(extra + pos++, static_cast<uint32_t>(arms.size()));
	for (const Arm &a : arms) {
		nodes->setExtra(extra + pos++, a.pat);
		nodes->setExtra(extra + pos++, static_cast<uint32_t>(a.body.size()));
		for (NodeIdx s : a.body) { nodes->setExtra(extra + pos++, s); }
	}

	return emit(AstNode{AstTag::MatchNode, 0, 0, 0,
	                    static_cast<uint32_t>(scrutinee), extra});
}

NodeIdx Parser::parseExpression() {
	if (match(TOK_RETURN)) {
		if (match(TOK_SEMI)) {
			return emit(AstNode{AstTag::Return, 0, 0, 0, kNoNode, 0});
		}
		NodeIdx expr = parseLogicalOr();
		consume(TOK_SEMI, "Expected ';' after return statement");
		return emit(
		    AstNode{AstTag::Return, 0, 0, 0, static_cast<uint32_t>(expr), 0});
	}
	if (match(TOK_CONST) || match(TOK_VAR)) {
		bool isConst = previous().type == TOK_CONST;
		consume(TOK_IDENTIFIER, "Expected variable name");
		StringIdx name = stringPool->intern(previous().text(source_));

		// kNoType signals "infer from init"; astgenVarDecl lowers the
		// init first in that case and takes its type. Previously this
		// defaulted to `u8`, which silently allocated a 1-byte slot
		// for any unannotated binding (including slices, bools, and
		// f64) and stored wider values into it.
		TypeIdx type = kNoType;
		if (match(TOK_COLON)) { type = parseType(); }

		consume(TOK_EQUAL,
		        "Expected '=' (every variable must be initialized at "
		        "declaration)");
		NodeIdx init = parseLogicalOr();
		consume(TOK_SEMI, "Expected ';' after variable declaration");

		// extra layout: [name StringIdx, type TypeIdx, init NodeIdx]
		ExtraIdx extra = nodes->reserveExtra(3);
		nodes->setExtra(extra, name);
		nodes->setExtra(extra + 1, type);
		nodes->setExtra(extra + 2, init);
		return emit(
		    AstNode{AstTag::VarDecl, 0, 0, 0, extra, isConst ? 1u : 0u});
	}
	if (check(TOK_MATCH)) { return parseMatch(); }
	if (match(TOK_IF)) {
		consume(TOK_OPEN_PAREN, "Expected '(' after 'if'");
		NodeIdx cond = parseLogicalOr();
		consume(TOK_CLOSE_PAREN, "Expected ')' after if condition");

		consume(TOK_OPEN_BRACE, "Expected '{' after if condition");
		std::vector<NodeIdx> thenBody;
		while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
			thenBody.push_back(parseExpression());
		}
		consume(TOK_CLOSE_BRACE, "Expected '}' after if body");

		std::vector<NodeIdx> elseBody;
		if (match(TOK_ELSE)) {
			if (check(TOK_IF)) {
				elseBody.push_back(parseExpression());
			} else {
				consume(TOK_OPEN_BRACE, "Expected '{' or 'if' after 'else'");
				while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
					elseBody.push_back(parseExpression());
				}
				consume(TOK_CLOSE_BRACE, "Expected '}' after else body");
			}
		}

		// extra layout: [thenCount, elseCount, then..., else...]
		ExtraIdx extra =
		    nodes->reserveExtra(2 + thenBody.size() + elseBody.size());
		nodes->setExtra(extra, static_cast<uint32_t>(thenBody.size()));
		nodes->setExtra(extra + 1, static_cast<uint32_t>(elseBody.size()));
		for (size_t i = 0; i < thenBody.size(); i++) {
			nodes->setExtra(extra + 2 + i, thenBody[i]);
		}
		for (size_t i = 0; i < elseBody.size(); i++) {
			nodes->setExtra(extra + 2 + thenBody.size() + i, elseBody[i]);
		}
		return emit(AstNode{AstTag::IfNode, 0, 0, 0,
		                    static_cast<uint32_t>(cond), extra});
	}
	if (match(TOK_WHILE)) {
		consume(TOK_OPEN_PAREN, "Expected '(' after 'while'");
		NodeIdx cond = parseLogicalOr();
		consume(TOK_CLOSE_PAREN, "Expected ')' after while condition");

		consume(TOK_OPEN_BRACE, "Expected '{' after while condition");
		std::vector<NodeIdx> body;
		while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
			body.push_back(parseExpression());
		}
		consume(TOK_CLOSE_BRACE, "Expected '}' after while body");

		ExtraIdx extra = nodes->reserveExtra(1 + body.size());
		nodes->setExtra(extra, static_cast<uint32_t>(body.size()));
		for (size_t i = 0; i < body.size(); i++) {
			nodes->setExtra(extra + 1 + i, body[i]);
		}
		return emit(AstNode{AstTag::WhileNode, 0, 0, 0,
		                    static_cast<uint32_t>(cond), extra});
	}
	if (match(TOK_LOOP)) {
		// `loop { body }` desugars to `while (true) { body }`. Reusing
		// WhileNode means the existing codegen, drop-tracking, and
		// stmtDiverges all "just work" — `stmtDiverges` already
		// recognizes `while (true)` with no reachable break as
		// diverging. The dedicated keyword exists for readability and
		// to express "this never falls through" at the source level.
		consume(TOK_OPEN_BRACE, "Expected '{' after 'loop'");
		std::vector<NodeIdx> body;
		while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
			body.push_back(parseExpression());
		}
		consume(TOK_CLOSE_BRACE, "Expected '}' after loop body");

		NodeIdx cond = emit(AstNode{AstTag::BoolLit, 0, 0, 0, 1, 0});
		ExtraIdx extra = nodes->reserveExtra(1 + body.size());
		nodes->setExtra(extra, static_cast<uint32_t>(body.size()));
		for (size_t i = 0; i < body.size(); i++) {
			nodes->setExtra(extra + 1 + i, body[i]);
		}
		return emit(AstNode{AstTag::WhileNode, 0, 0, 0,
		                    static_cast<uint32_t>(cond), extra});
	}
	if (match(TOK_FOR)) {
		consume(TOK_IDENTIFIER, "Expected variable name after 'for'");
		StringIdx varName = stringPool->intern(previous().text(source_));

		consume(TOK_IN, "Expected 'in' after for variable");
		// Disable typed struct literals while parsing the range so a
		// `for i in 0:n { body }` doesn't greedily swallow `n { body }`
		// as a literal. Restored before the body opener.
		bool prevAllow = allowStructLit_;
		allowStructLit_ = false;
		NodeIdx start = parseComparison();
		consume(TOK_COLON, "Expected ':' in for range");
		NodeIdx end = parseComparison();
		allowStructLit_ = prevAllow;

		consume(TOK_OPEN_BRACE, "Expected '{' after for range");
		std::vector<NodeIdx> body;
		while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
			body.push_back(parseExpression());
		}
		consume(TOK_CLOSE_BRACE, "Expected '}' after for body");

		// extra layout: [varName, start, end, bodyCount, body...]
		ExtraIdx extra = nodes->reserveExtra(4 + body.size());
		nodes->setExtra(extra, varName);
		nodes->setExtra(extra + 1, start);
		nodes->setExtra(extra + 2, end);
		nodes->setExtra(extra + 3, static_cast<uint32_t>(body.size()));
		for (size_t i = 0; i < body.size(); i++) {
			nodes->setExtra(extra + 4 + i, body[i]);
		}
		return emit(AstNode{AstTag::ForNode, 0, 0, 0, extra, 0});
	}
	if (match(TOK_BREAK)) {
		consume(TOK_SEMI, "Expected ';' after break");
		return emit(AstNode{AstTag::Break, 0, 0, 0, 0, 0});
	}
	if (match(TOK_CONTINUE)) {
		consume(TOK_SEMI, "Expected ';' after continue");
		return emit(AstNode{AstTag::Continue, 0, 0, 0, 0, 0});
	}
	if (check(TOK_IDENTIFIER)) {
		NodeIdx expr = parseComparison();

		if (match(TOK_EQUAL)) {
			NodeIdx value = parseLogicalOr();
			consume(TOK_SEMI, "Expected ';' after assignment");
			return emit(AstNode{AstTag::Assign, 0, 0, 0,
			                    static_cast<uint32_t>(expr),
			                    static_cast<uint32_t>(value)});
		}

		// Function-call statement requires a trailing semicolon.
		if (nodes->get(expr).tag == AstTag::Call) {
			consume(TOK_SEMI, "Expected ';' after function call");
		}
		return expr;
	}

	return parseLogicalOr();
}

NodeIdx Parser::parseLogicalOr() {
	NodeIdx lhs = parseLogicalAnd();
	while (match(TOK_OR)) {
		NodeIdx rhs = parseLogicalAnd();
		lhs = emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(BinOp::LogOr),
		                   0, 0, static_cast<uint32_t>(lhs),
		                   static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

NodeIdx Parser::parseLogicalAnd() {
	NodeIdx lhs = parseComparison();
	while (match(TOK_AND)) {
		NodeIdx rhs = parseComparison();
		lhs = emit(
		    AstNode{AstTag::BinaryOp, static_cast<uint8_t>(BinOp::LogAnd), 0, 0,
		            static_cast<uint32_t>(lhs), static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

static BinOp comparisonOp(TokenType t) {
	switch (t) {
	case TOK_EQUAL_EQUAL:
		return BinOp::Eq;
	case TOK_NOT_EQUAL:
		return BinOp::Ne;
	case TOK_LESS:
		return BinOp::Lt;
	case TOK_LESS_EQUAL:
		return BinOp::Le;
	case TOK_GREATER:
		return BinOp::Gt;
	case TOK_GREATER_EQUAL:
		return BinOp::Ge;
	default:
		return BinOp::Invalid;
	}
}

NodeIdx Parser::parseComparison() {
	NodeIdx lhs = parseBitwise();
	if (check(TOK_EQUAL_EQUAL) || check(TOK_NOT_EQUAL) || check(TOK_LESS) ||
	    check(TOK_LESS_EQUAL) || check(TOK_GREATER) ||
	    check(TOK_GREATER_EQUAL)) {
		Token op = advance();
		NodeIdx rhs = parseBitwise();
		BinOp k = comparisonOp(op.type);
		return emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(k), 0, 0,
		                    static_cast<uint32_t>(lhs),
		                    static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

NodeIdx Parser::parseBitwise() {
	NodeIdx lhs = parseShift();
	while (true) {
		BinOp k = BinOp::Invalid;
		if (match(TOK_AMP)) k = BinOp::BitAnd;
		else if (match(TOK_PIPE)) k = BinOp::BitOr;
		else if (match(TOK_CARET)) k = BinOp::BitXor;
		else break;
		NodeIdx rhs = parseShift();
		lhs = emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(k), 0, 0,
		                   static_cast<uint32_t>(lhs),
		                   static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

NodeIdx Parser::parseShift() {
	NodeIdx lhs = parseAddition();
	while (true) {
		BinOp k = BinOp::Invalid;
		if (match(TOK_LSHIFT)) k = BinOp::Shl;
		else if (match(TOK_RSHIFT)) k = BinOp::Shr;
		else break;
		NodeIdx rhs = parseAddition();
		lhs = emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(k), 0, 0,
		                   static_cast<uint32_t>(lhs),
		                   static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

NodeIdx Parser::parseAddition() {
	NodeIdx lhs = parseMultiplication();
	while (true) {
		BinOp k = BinOp::Invalid;
		if (match(TOK_PLUS)) k = BinOp::Add;
		else if (match(TOK_MINUS)) k = BinOp::Sub;
		else break;
		NodeIdx rhs = parseMultiplication();
		lhs = emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(k), 0, 0,
		                   static_cast<uint32_t>(lhs),
		                   static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

NodeIdx Parser::parseMultiplication() {
	NodeIdx lhs = parseUnary();
	while (true) {
		BinOp k = BinOp::Invalid;
		if (match(TOK_STAR)) k = BinOp::Mul;
		else if (match(TOK_SLASH)) k = BinOp::Div;
		else if (match(TOK_PERCENT)) k = BinOp::Mod;
		else break;
		NodeIdx rhs = parseUnary();
		lhs = emit(AstNode{AstTag::BinaryOp, static_cast<uint8_t>(k), 0, 0,
		                   static_cast<uint32_t>(lhs),
		                   static_cast<uint32_t>(rhs)});
	}
	return lhs;
}

// Parse a postfix `as Type` chain on top of any unary expression. The
// cast binds tightly — `5 + (x as u32)` not `(5 + x) as u32` — placing
// the cast just above primary expressions in the precedence table.
static NodeIdx parseAsChain(Parser *self, NodeIdx expr, NodeStore &nodes,
                            bool (Parser::*matchTok)(TokenType),
                            TypeIdx (Parser::*parseType)());
// Forward declaration removed; we just inline the postfix loop below.

NodeIdx Parser::parseUnary() {
	auto wrapAs = [&](NodeIdx e) {
		while (match(TOK_AS)) {
			TypeIdx ty = parseType();
			e = emit(
			    AstNode{AstTag::AsCast, 0, 0, 0, static_cast<uint32_t>(e), ty});
		}
		return e;
	};

	if (match(TOK_NOT)) {
		NodeIdx operand = parseUnary();
		return wrapAs(
		    emit(AstNode{AstTag::UnaryOp, static_cast<uint8_t>(UnaryOp::LogNot),
		                 0, 0, static_cast<uint32_t>(operand), 0}));
	}
	if (match(TOK_TILDE)) {
		NodeIdx operand = parseUnary();
		return wrapAs(
		    emit(AstNode{AstTag::UnaryOp, static_cast<uint8_t>(UnaryOp::BitNot),
		                 0, 0, static_cast<uint32_t>(operand), 0}));
	}
	if (match(TOK_AMP)) {
		NodeIdx operand = parseUnary();
		return wrapAs(emit(AstNode{AstTag::AddressOf, 0, 0, 0,
		                           static_cast<uint32_t>(operand), 0}));
	}
	if (match(TOK_MINUS)) {
		NodeIdx operand = parseUnary();
		// fold a leading "-" into the literal's sign so the natural type
		// is signed. Otherwise "-16 as i64" parses as "Neg(AsCast(16,
		// i64))" where the inner 16 is a u8.
		auto innerLit = [&](NodeIdx idx) -> NodeIdx {
			NodeIdx cur = idx;
			while (true) {
				const AstNode &n = nodes->get(cur);
				if (n.tag == AstTag::NumberLit) { return cur; }
				if (n.tag == AstTag::AsCast) {
					cur = static_cast<NodeIdx>(n.lhs);
					continue;
				}
				return kNoNode;
			}
		};
		NodeIdx litIdx = innerLit(operand);
		if (litIdx != kNoNode) {
			AstNode lit = nodes->get(litIdx);
			lit.flags = static_cast<uint16_t>(lit.flags ^ 1u);
			NodeIdx newLit = emit(lit);
			// if the original operand was a cast, rebuild the cast chain
			// around the sign-flipped literal so "-16 as i64" becomes
			// "AsCast(NumberLit(-16), i64)".
			if (operand != litIdx) {
				NodeIdx cur = operand;
				std::vector<TypeIdx> casts;
				while (cur != litIdx) {
					const AstNode &n = nodes->get(cur);
					casts.push_back(static_cast<TypeIdx>(n.rhs));
					cur = static_cast<NodeIdx>(n.lhs);
				}
				for (auto it = casts.rbegin(); it != casts.rend(); ++it) {
					newLit = emit(AstNode{AstTag::AsCast, 0, 0, 0,
					                      static_cast<uint32_t>(newLit),
					                      static_cast<uint32_t>(*it)});
				}
			}
			return wrapAs(newLit);
		}
		return wrapAs(
		    emit(AstNode{AstTag::UnaryOp, static_cast<uint8_t>(UnaryOp::Neg), 0,
		                 0, static_cast<uint32_t>(operand), 0}));
	}
	return wrapAs(parsePrimary());
}

std::unique_ptr<FunctionAST> Parser::parseFunction() {
	bool isExtern = false;
	bool isExport = false;
	bool isPub = false;
	bool isTest = false;

	// Modifiers can combine: `pub extern fn ...` exports an FFI
	// symbol that other Jam modules can call. Loop so any order works.
	while (true) {
		if (!isExtern && match(TOK_EXTERN)) isExtern = true;
		else if (!isExport && match(TOK_EXPORT)) isExport = true;
		else if (!isPub && match(TOK_PUB)) isPub = true;
		else if (!isTest && match(TOK_TFN)) isTest = true;
		else break;
	}

	if (!isTest) { consume(TOK_FN, "Expected 'fn' keyword"); }
	consume(TOK_IDENTIFIER, "Expected function name");
	std::string name(previous().text(source_));

	consume(TOK_OPEN_PAREN, "Expected '(' after function name");

	std::vector<Param> args;
	bool isVarArgs = false;
	if (!check(TOK_CLOSE_PAREN)) {
		do {
			if (match(TOK_ELLIPSIS)) {
				if (!isExtern) {
					parseError(
					    "`...` is only allowed in extern fn declarations");
				}
				isVarArgs = true;
				break;
			}
			consume(TOK_IDENTIFIER, "Expected parameter name");
			std::string paramName(previous().text(source_));

			consume(TOK_COLON, "Expected ':' after parameter name");

			// Optional mode keyword between `:` and the type.
			//   x: u32             — Let (default, read-only)
			//   x: mut u32         — Mut (exclusive read-write)
			//   x: move List       — Move (consume ownership)
			//
			// `*mut T` pointer types start with `*`, not `mut`, so there is
			// no conflict with the existing pointer syntax. See MVS.md §2.
			ParamMode mode = ParamMode::Let;
			if (match(TOK_MUT)) {
				mode = ParamMode::Mut;
			} else if (match(TOK_MOVE)) {
				mode = ParamMode::Move;
			}

			TypeIdx paramType = parseType();
			args.push_back(Param{std::move(paramName), paramType, mode});
		} while (match(TOK_COMMA));
	}

	consume(TOK_CLOSE_PAREN, "Expected ')' after parameters");

	TypeIdx returnType = kNoType;
	if (check(TOK_TYPE) || check(TOK_OPEN_BRACKET) || check(TOK_CONST) ||
	    check(TOK_IDENTIFIER) || check(TOK_STAR)) {
		returnType = parseType();
	}

	if (isExtern) {
		consume(TOK_SEMI, "Expected ';' after extern function declaration");
		return std::make_unique<FunctionAST>(name, std::move(args), returnType,
		                                     std::vector<NodeIdx>{}, true,
		                                     isExport, isPub, false, isVarArgs);
	}

	consume(TOK_OPEN_BRACE, "Expected '{' before function body");

	std::vector<NodeIdx> body;
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		body.push_back(parseExpression());
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' after function body");

	return std::make_unique<FunctionAST>(name, std::move(args), returnType,
	                                     std::move(body), false, isExport,
	                                     isPub, isTest, false);
}

void Parser::parseStructBody(
    std::vector<std::pair<std::string, TypeIdx>> &fields,
    std::vector<std::unique_ptr<FunctionAST>> &methods) {
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		// Method: `fn name(self: ..., ...) ReturnType { body }`. Methods
		// can appear in any order relative to fields. parseFunction
		// consumes the `fn` keyword itself.
		if (check(TOK_FN)) {
			methods.push_back(parseFunction());
			match(TOK_COMMA);  // optional trailing comma after a method
			continue;
		}
		// Field: `name: Type`. Comma separates from the next member.
		consume(TOK_IDENTIFIER, "Expected field name or 'fn'");
		std::string fieldName(previous().text(source_));
		consume(TOK_COLON, "Expected ':' after field name");
		TypeIdx fieldType = parseType();
		fields.emplace_back(std::move(fieldName), fieldType);
		if (!check(TOK_CLOSE_BRACE)) {
			consume(TOK_COMMA, "Expected ',' or '}' after struct field");
		}
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' to close struct definition");
}

std::unique_ptr<StructDeclAST> Parser::parseStructDecl() {
	consume(TOK_CONST, "Expected 'const' for struct declaration");
	consume(TOK_IDENTIFIER, "Expected struct name");
	std::string name(previous().text(source_));
	consume(TOK_EQUAL, "Expected '=' after struct name");
	consume(TOK_STRUCT, "Expected 'struct' keyword");
	consume(TOK_OPEN_BRACE, "Expected '{' after 'struct'");

	std::vector<std::pair<std::string, TypeIdx>> fields;
	std::vector<std::unique_ptr<FunctionAST>> methods;
	structContextStack.push_back(name);
	parseStructBody(fields, methods);
	structContextStack.pop_back();
	consume(TOK_SEMI, "Expected ';' after struct declaration");

	return std::make_unique<StructDeclAST>(name, std::move(fields),
	                                       std::move(methods));
}

// Parse `enum { Variant1, Variant2(T1, T2), ... }` in expression
// position. Caller has already consumed the `enum` keyword. Variants
// get sequential discriminants starting at 0; the body's payload types
// may reference generic params and will be substituted at each
// instantiation. Methods on enum expressions are not supported in v1.
NodeIdx Parser::parseEnumExpression() {
	consume(TOK_OPEN_BRACE, "Expected '{' after 'enum'");
	auto &enumsRef = sharedAnonEnums ? *sharedAnonEnums : anonEnums;
	uint32_t idx = static_cast<uint32_t>(enumsRef.size());
	std::string name = "__anon_enum_" + std::to_string(idx);

	std::vector<EnumVariantAST> variants;
	uint32_t nextDiscrim = 0;
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		consume(TOK_IDENTIFIER, "Expected enum variant name");
		EnumVariantAST v;
		v.Name = previous().text(source_);
		// Optional positional payload list.
		if (match(TOK_OPEN_PAREN)) {
			if (!check(TOK_CLOSE_PAREN)) {
				do {
					v.PayloadTypes.push_back(parseType());
				} while (match(TOK_COMMA));
			}
			consume(TOK_CLOSE_PAREN,
			        "Expected `)` to close variant payload list");
		}
		v.Discriminant = nextDiscrim++;
		variants.push_back(std::move(v));
		if (!match(TOK_COMMA)) break;
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' to close enum body");

	enumsRef.push_back(
	    std::make_unique<EnumDeclAST>(std::move(name), std::move(variants)));

	return emit(AstNode{AstTag::EnumExpr, 0, 0, 0, idx, 0});
}

NodeIdx Parser::parseStructExpression() {
	consume(TOK_OPEN_BRACE, "Expected '{' after 'struct'");
	auto &structsRef = sharedAnonStructs ? *sharedAnonStructs : anonStructs;
	uint32_t idx = static_cast<uint32_t>(structsRef.size());
	std::string name = "__anon_struct_" + std::to_string(idx);

	std::vector<std::pair<std::string, TypeIdx>> fields;
	std::vector<std::unique_ptr<FunctionAST>> methods;
	structContextStack.push_back(name);
	parseStructBody(fields, methods);
	structContextStack.pop_back();

	structsRef.push_back(std::make_unique<StructDeclAST>(
	    std::move(name), std::move(fields), std::move(methods)));

	return emit(AstNode{AstTag::StructExpr, 0, 0, 0, idx, 0});
}

// Parse `const Name = enum { Variant1, Variant2(T1, T2), ... };`.
// Variants get sequential discriminant values starting from zero in
// declaration order. Unit variants (no payload) and tagged variants
// (positional payload types) coexist — the parser accepts both.
std::unique_ptr<EnumDeclAST> Parser::parseEnumDecl() {
	consume(TOK_CONST, "Expected 'const' for enum declaration");
	consume(TOK_IDENTIFIER, "Expected enum name");
	std::string name(previous().text(source_));
	consume(TOK_EQUAL, "Expected '=' after enum name");
	consume(TOK_ENUM, "Expected 'enum' keyword");
	consume(TOK_OPEN_BRACE, "Expected '{' after 'enum'");

	std::vector<EnumVariantAST> variants;
	uint32_t nextDiscrim = 0;
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		consume(TOK_IDENTIFIER, "Expected enum variant name");
		EnumVariantAST v;
		v.Name = previous().text(source_);
		// Optional payload: `Variant(T1, T2, ...)`. Unit variants omit
		// the parenthesized list.
		if (match(TOK_OPEN_PAREN)) {
			if (!check(TOK_CLOSE_PAREN)) {
				do {
					v.PayloadTypes.push_back(parseType());
				} while (match(TOK_COMMA));
			}
			consume(TOK_CLOSE_PAREN,
			        "Expected `)` to close variant payload list");
		}
		if (match(TOK_EQUAL)) {
			consume(TOK_NUMBER,
			        "Expected integer literal after `=` in enum variant");
			bool neg = false;
			uint64_t mag = 0;
			try {
				NumberResult r = parseNumberLiteral(previous().text(source_));
				if (r.kind != NumberResultKind::Int) {
					parseError(
					    "Enum discriminant must be a non-negative integer");
				}
				mag = r.intValue;
			} catch (const std::exception &e) {
				parseError(std::string("Invalid enum discriminant: ") +
				           e.what());
			}
			(void)neg;
			if (mag > 255) {
				parseError("Enum discriminant " + std::to_string(mag) +
				           " is out of range; M2 enums are u8-tagged");
			}
			v.Discriminant = static_cast<uint32_t>(mag);
			nextDiscrim = v.Discriminant + 1;
		} else {
			v.Discriminant = nextDiscrim++;
		}
		variants.push_back(std::move(v));
		if (!match(TOK_COMMA)) break;
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' to close enum definition");
	consume(TOK_SEMI, "Expected ';' after enum declaration");

	if (variants.empty()) {
		parseError("Enum `" + name + "` must declare at least one variant");
	}
	if (variants.size() > 256) {
		parseError("Enum `" + name +
		           "` has more than 256 variants; M2 enums are u8-tagged");
	}
	return std::make_unique<EnumDeclAST>(name, std::move(variants));
}

std::unique_ptr<UnionDeclAST> Parser::parseUnionDecl() {
	consume(TOK_CONST, "Expected 'const' for union declaration");
	consume(TOK_IDENTIFIER, "Expected union name");
	std::string name(previous().text(source_));
	consume(TOK_EQUAL, "Expected '=' after union name");
	consume(TOK_UNION, "Expected 'union' keyword");
	consume(TOK_OPEN_BRACE, "Expected '{' after 'union'");

	std::vector<std::pair<std::string, TypeIdx>> fields;
	while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
		consume(TOK_IDENTIFIER, "Expected union field name");
		std::string fieldName(previous().text(source_));
		consume(TOK_COLON, "Expected ':' after field name");
		TypeIdx fieldType = parseType();
		fields.emplace_back(std::move(fieldName), fieldType);
		if (!match(TOK_COMMA)) break;
	}
	consume(TOK_CLOSE_BRACE, "Expected '}' to close union definition");
	consume(TOK_SEMI, "Expected ';' after union declaration");

	return std::make_unique<UnionDeclAST>(name, std::move(fields));
}

std::unique_ptr<ImportDeclAST> Parser::parseImportDecl() {
	consume(TOK_CONST, "Expected 'const' for import declaration");
	consume(TOK_IDENTIFIER, "Expected identifier for import name");
	std::string name(previous().text(source_));

	consume(TOK_EQUAL, "Expected '=' after import name");
	consume(TOK_IMPORT, "Expected 'import' keyword");
	consume(TOK_OPEN_PAREN, "Expected '(' after 'import'");
	consume(TOK_STRING_LITERAL, "Expected string literal for import path");
	// Decoded value (escapes resolved); raw `text(source_)` would
	// include the surrounding `"..."` quotes which the resolver
	// doesn't expect.
	std::string path = previous().lexeme;
	consume(TOK_CLOSE_PAREN, "Expected ')' after import path");
	consume(TOK_SEMI, "Expected ';' after import declaration");

	importHandles.insert(name);
	return std::make_unique<ImportDeclAST>(name, path);
}

std::unique_ptr<DestructuringImportDeclAST> Parser::parseDestructuringImport() {
	consume(TOK_CONST, "Expected 'const' for destructuring import");
	consume(TOK_OPEN_BRACE, "Expected '{' for destructuring import");

	std::vector<std::string> names;
	do {
		consume(TOK_IDENTIFIER, "Expected identifier in destructuring import");
		names.emplace_back(previous().text(source_));
	} while (match(TOK_COMMA));

	consume(TOK_CLOSE_BRACE, "Expected '}' after destructuring names");
	consume(TOK_EQUAL, "Expected '=' after destructuring");
	consume(TOK_IMPORT, "Expected 'import' keyword");
	consume(TOK_OPEN_PAREN, "Expected '(' after 'import'");
	consume(TOK_STRING_LITERAL, "Expected string literal for import path");
	std::string path = previous().lexeme;
	consume(TOK_CLOSE_PAREN, "Expected ')' after import path");
	consume(TOK_SEMI, "Expected ';' after import declaration");

	return std::make_unique<DestructuringImportDeclAST>(std::move(names), path);
}

std::unique_ptr<ConstDeclAST> Parser::parseConstDecl() {
	consume(TOK_CONST, "Expected 'const' for module-scope constant");
	consume(TOK_IDENTIFIER, "Expected identifier for constant name");
	std::string name(previous().text(source_));

	TypeIdx declared = kNoType;
	if (match(TOK_COLON)) { declared = parseType(); }

	consume(TOK_EQUAL, "Expected '=' in module-scope const declaration");
	// Parse RHS as an expression — no speculation. Type-alias intent
	// (`const ListI32 = Vec(i32);`) is recognised at registration
	// time in main.cpp::resolveExprAsType by pattern-matching the
	// resulting AST (`Variable` for type-named bindings, direct
	// `Call` of a generic-fn-returning-`type`). The pattern match
	// covers the common cases; richer forms (if/else types, member
	// access on types, chained calls, 0-arg type-returning fns)
	// are deferred until comptime evaluation lands.
	NodeIdx init = parseLogicalOr();
	consume(TOK_SEMI, "Expected ';' after module-scope const declaration");

	return std::make_unique<ConstDeclAST>(std::move(name), declared, init);
}

std::unique_ptr<ModuleAST> Parser::parse() {
	auto module = std::make_unique<ModuleAST>();

	while (!isAtEnd()) {
		// `pub` at module scope marks the next decl as exported. When
		// the next token is `const` we absorb `pub` here and tag the
		// resulting decl AST. When it's `fn`/`extern`/`tfn`/`export`,
		// we leave it untouched so parseFunction can consume it itself.
		bool isPub = false;
		if (check(TOK_PUB)) {
			int pubSaved = current;
			advance();
			bool nextIsConst = check(TOK_CONST);
			current = pubSaved;
			if (nextIsConst) {
				advance();  // consume `pub`; const branch below handles the
				            // rest.
				isPub = true;
				// Reject `pub const { ... } = import(...)` upfront so the
				// later destructuring-import path doesn't silently drop pub.
				int peek = current + 1;
				if (peek < static_cast<int>(tokens.size()) &&
				    tokens[peek].type == TOK_OPEN_BRACE) {
					parseError("`pub` is not allowed on destructuring imports");
				}
			}
		}

		if (check(TOK_CONST)) {
			int saved = current;
			advance();

			if (check(TOK_OPEN_BRACE)) {
				current = saved;
				module->DestructuringImports.push_back(
				    parseDestructuringImport());
				continue;
			}

			if (check(TOK_IDENTIFIER)) {
				advance();
				// `const NAME : T = …` — typed module const.
				if (check(TOK_COLON)) {
					current = saved;
					auto c = parseConstDecl();
					c->isPub = isPub;
					module->Consts.push_back(std::move(c));
					continue;
				}
				if (check(TOK_EQUAL)) {
					advance();
					if (check(TOK_IMPORT)) {
						if (isPub) {
							parseError("`pub` is not allowed on imports");
						}
						current = saved;
						module->Imports.push_back(parseImportDecl());
						continue;
					}
					if (check(TOK_STRUCT)) {
						current = saved;
						auto s = parseStructDecl();
						s->isPub = isPub;
						module->Structs.push_back(std::move(s));
						continue;
					}
					if (check(TOK_UNION)) {
						current = saved;
						auto u = parseUnionDecl();
						u->isPub = isPub;
						module->Unions.push_back(std::move(u));
						continue;
					}
					if (check(TOK_ENUM)) {
						current = saved;
						auto e = parseEnumDecl();
						e->isPub = isPub;
						module->Enums.push_back(std::move(e));
						continue;
					}
					// `const NAME = expr;` — untyped module const,
					// type inferred from the initializer.
					current = saved;
					auto c = parseConstDecl();
					c->isPub = isPub;
					module->Consts.push_back(std::move(c));
					continue;
				}
			}

			current = saved;
		}

		// `pub fn …` falls through here; parseFunction handles `pub`
		// itself (its modifier-loop accepts `pub`/`extern`/`export`/`tfn`).
		module->Functions.push_back(parseFunction());
	}

	if (!sharedAnonStructs) { module->AnonStructs = std::move(anonStructs); }
	if (!sharedAnonEnums) { module->AnonEnums = std::move(anonEnums); }

	return module;
}
