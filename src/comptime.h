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
// from it when it encounters a Variable node, and the executor mutates
// it on var-decl / assignment statements.
//
// Mutation model:
//   * bind(name, value) — declare a new local in THIS scope (var decl).
//     Always succeeds; shadows any same-named binding in a parent
//     scope.
//   * set(name, value) — walk the parent chain, mutate the scope where
//     `name` was first bound. Returns false if not found anywhere.
//     Used by assignment statements.
//   * lookup(name) — read; walks the parent chain.
//
// `parent_` is mutable (non-const) so set() can write through to outer
// scopes. The contract is that callers pass scopes that they own.
class ComptimeScope {
  public:
	ComptimeScope() = default;
	explicit ComptimeScope(ComptimeScope *parent) : parent_(parent) {}

	void bind(const std::string &name, ComptimeValue value);

	// Walk up looking for `name`. If found, replace its value and
	// return true. If not found, return false (caller decides whether
	// to surface a diagnostic or implicitly bind in current scope).
	bool set(const std::string &name, ComptimeValue value);

	// Returns nullptr if `name` isn't bound here OR in any ancestor.
	const ComptimeValue *lookup(const std::string &name) const;

  private:
	ComptimeScope *parent_ = nullptr;
	std::unordered_map<std::string, ComptimeValue> bindings_;
};

// Outcome of executing a statement (or a block). The evaluator threads
// this through nested control-flow so a `return` inside `while { if { ... } }`
// propagates correctly without throwing.
enum class ExecResult : uint8_t {
	Continue,      // statement executed normally, keep going
	Returned,      // a `return` was hit; caller propagates upward
	IterationCap,  // a while loop or recursive call exceeded the iteration cap
	Error,         // an unrecoverable failure (diagnostic already pushed)
};

// Sink for `@`-emit intrinsics inside a cfn body. The evaluator hits an
// AtCall node while interpreting cfn body code; it evaluates each arg
// to a ComptimeValue and forwards to `handleAtCall`. The astgen-side
// implementation has access to the caller's gctx and uses it to emit
// JIR instructions that lower the cfn's compile-time work into runtime
// code at the call site.
//
// Returning Error from handleAtCall aborts cfn execution and propagates
// up through the evaluator's ExecResult.
class CompEmitter {
  public:
	virtual ~CompEmitter() = default;
	virtual ExecResult handleAtCall(const std::string &name,
	                                  const std::vector<ComptimeValue> &args,
	                                  Diagnostics &diags, SrcLoc loc) = 0;
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

	// Execute a statement node. The scope is mutated for var-decls and
	// assignments; control-flow statements (if/while) recurse. The
	// returned ExecResult tells the caller whether to continue, stop
	// (return), or bail (iteration cap / error). `iterCounter` is
	// incremented on every loop iteration and recursive cfn call; the
	// caller seeds it at 0 and checks it never exceeds `iterCap`. If
	// the cap is hit a diagnostic is pushed and IterationCap is
	// returned all the way out.
	//
	// `outReturnValue` receives the value of a `return EXPR;` if one
	// fires; otherwise it's left None.
	static constexpr uint32_t kDefaultIterCap = 10'000;
	ExecResult execStmt(NodeIdx stmt, ComptimeScope &scope,
	                     uint32_t &iterCounter, uint32_t iterCap,
	                     ComptimeValue &outReturnValue, Diagnostics &diags,
	                     SrcLoc loc) const;

	// Execute a sequence of statements in order. Returns the first
	// non-Continue result. Wraps a list of NodeIdx from an Extra slice
	// like a fn body or if/while body.
	ExecResult execBlock(const NodeIdx *stmts, std::size_t count,
	                      ComptimeScope &scope, uint32_t &iterCounter,
	                      uint32_t iterCap, ComptimeValue &outReturnValue,
	                      Diagnostics &diags, SrcLoc loc) const;

	// Install / clear the AtCall emitter + the diagnostic context.
	// Active for the duration of a cfn body's execution; cleared
	// right after. When emitter is set, AtCall nodes (expr-arg multi-
	// form) are forwarded to the emitter. The diags+loc are used by
	// the emitter and by `evalAtCall` to push error diagnostics.
	void setCallContext(CompEmitter *e, Diagnostics *d, SrcLoc loc) const {
		emitter_ = e;
		diags_ = d;
		loc_ = std::move(loc);
	}
	void clearCallContext() const {
		emitter_ = nullptr;
		diags_ = nullptr;
		loc_ = {};
	}

  private:
	const NodeStore &nodes_;
	const StringPool &strings_;
	const TypePool &types_;
	mutable CompEmitter *emitter_ = nullptr;
	mutable Diagnostics *diags_ = nullptr;
	mutable SrcLoc loc_;

	// Eval an `AtCall` node from inside a cfn body. Type-arg single-
	// form (`@sizeOf(T)`) returns None here (caller handles via the
	// regular AtCall lowering). Expr-arg multi-form (`@emit*(...)`)
	// is forwarded to the installed emitter; returns None as well
	// (the @-emit family produces side effects, not values).
	ComptimeValue evalAtCall(const AstNode &n,
	                          const ComptimeScope &scope) const;

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
	ComptimeValue evalMemberAccess(const AstNode &n,
	                                const ComptimeScope &scope) const;
};

}  // namespace jam

#endif  // JAM_COMPTIME_H
