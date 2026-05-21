/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_COMPTIME_H
#define JAM_COMPTIME_H

#include "ast_flat.h"
#include "diagnostics.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace jam {

// A value known at compile time. Carries one of the primitive kinds the
// comp evaluator can produce. Aggregate is for tuple / struct values
// constructed at compile time (`.{1, 2, 3}` literal). Type is a value
// of the meta-type (`@TypeOf(x)`, generic-parameter args).
//
// The payload is split: trivially-copyable scalars live in a union; non-
// trivial members (strings, vectors) live alongside. None is the failure
// / "not foldable" state — every operation that can fail returns it
// rather than throwing.
//
// Source-level keyword for the feature is `comp` (e.g. `comp const N =
// 42;`, `comp if (cond) ...`); the compiler-internal naming uses the
// older "comptime" term to refer to the broader compile-time-evaluation
// concept.
struct ComptimeValue {
	enum class Kind : uint8_t {
		None,
		Int,
		Float,
		Bool,
		Str,
		Type,
		Aggregate,
	};

	Kind kind = Kind::None;

	// Scalar payload (active iff kind is Int / Float / Bool).
	union {
		struct {
			uint64_t bits;
			uint16_t width;
			bool isSigned;
		} intVal;
		struct {
			double value;
			uint16_t width;
		} floatVal;
		bool boolVal;
		uint8_t _pad;  // ensure trivially-default-constructible union
	};

	// Non-trivial payload (active per matching kind).
	StringIdx strVal = 0;
	TypeIdx typeVal = kNoType;
	std::vector<ComptimeValue> aggFields;

	ComptimeValue() : kind(Kind::None), _pad(0) {}

	bool isNone() const { return kind == Kind::None; }
	bool isInt() const { return kind == Kind::Int; }
	bool isBool() const { return kind == Kind::Bool; }
	bool isStr() const { return kind == Kind::Str; }
	bool isType() const { return kind == Kind::Type; }

	// Construction helpers — clearer than direct field stomping.
	static ComptimeValue makeNone();
	static ComptimeValue makeInt(uint64_t bits, uint16_t width, bool isSigned);
	static ComptimeValue makeFloat(double v, uint16_t width);
	static ComptimeValue makeBool(bool b);
	static ComptimeValue makeStr(StringIdx s);
	static ComptimeValue makeType(TypeIdx t);
	static ComptimeValue makeAggregate(std::vector<ComptimeValue> fields);

	// Convert an Int payload to int64 / uint64 for arithmetic. Both
	// helpers truncate to the value's width to keep the bit pattern
	// faithful — e.g. `i8(-1)` reads back as 0xFF.
	int64_t asI64() const;
	uint64_t asU64() const;
};

// A lexically-scoped map of name → ComptimeValue. The evaluator reads
// from it when it encounters a Variable node. Bindings can be pushed
// and popped to model nested scopes during inline-for unrolling, where
// the loop variable lives in a transient frame.
class ComptimeScope {
  public:
	ComptimeScope() = default;
	explicit ComptimeScope(const ComptimeScope *parent) : parent_(parent) {}

	void bind(const std::string &name, ComptimeValue value);

	// Returns nullptr if `name` isn't bound here OR in any ancestor.
	const ComptimeValue *lookup(const std::string &name) const;

  private:
	const ComptimeScope *parent_ = nullptr;
	std::unordered_map<std::string, ComptimeValue> bindings_;
};

// Folds AST expression nodes to compile-time values. Failure modes
// (depends on runtime value, unsupported operator, type mismatch) all
// surface as ComptimeValue::None — the evaluator never throws. Callers
// that *require* a fold (e.g. `comp expr`, `inline for` cond) call
// `evalRequired` which pushes a diagnostic on failure.
//
// The evaluator is stateless across calls: it captures references to
// the node/string/type pools at construction and reads bindings from a
// caller-supplied scope per `eval` invocation. Constructing one per
// call is cheap.
class ComptimeEvaluator {
  public:
	ComptimeEvaluator(const NodeStore &nodes, const StringPool &strings,
	                   const TypePool &types);

	// Try to fold `expr` to a value. Returns None on any failure.
	ComptimeValue eval(NodeIdx expr, const ComptimeScope &scope) const;

	// Same but pushes a diagnostic + returns None when the expression
	// can't be folded. Used by `comp expr` (which the user has
	// explicitly marked as requiring fold) and by control-flow primitives
	// where a non-fold is a hard error.
	ComptimeValue evalRequired(NodeIdx expr, const ComptimeScope &scope,
	                            Diagnostics &diags, SrcLoc loc) const;

  private:
	const NodeStore &nodes_;
	const StringPool &strings_;
	const TypePool &types_;

	// Per-tag handlers. Each returns None on failure; callers compose.
	ComptimeValue evalNumberLit(const AstNode &n) const;
	ComptimeValue evalBoolLit(const AstNode &n) const;
	ComptimeValue evalStringLit(const AstNode &n) const;
	ComptimeValue evalVariable(const AstNode &n,
	                            const ComptimeScope &scope) const;
	ComptimeValue evalUnaryOp(const AstNode &n,
	                           const ComptimeScope &scope) const;
	ComptimeValue evalBinaryOp(const AstNode &n,
	                            const ComptimeScope &scope) const;
	ComptimeValue evalIndex(const AstNode &n,
	                         const ComptimeScope &scope) const;
};

}  // namespace jam

#endif  // JAM_COMPTIME_H
