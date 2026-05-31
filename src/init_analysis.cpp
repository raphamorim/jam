/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

// Definite-initialization analysis for Jam.
//
// A tree-walking dataflow over the function body produces, at each program
// point, a map from binding name to lattice state {Unknown, Init, Uninit,
// MaybeInit}. The lattice is the same one Clang's UninitializedValues pass
// uses (clang/lib/Analysis/UninitializedValues.cpp): two bits per binding,
// merge by bitwise OR.
//
// Jam programs are structured (no goto, no break-out-of-multiple-levels),
// so a recursive analyzer over the AST suffices — there is no separate
// CFG-construction pass. Branches fork and merge in lockstep with the
// `if` / `match` AST shape; loops merge their body output back with the
// pre-loop state.
//
// Scope (this file):
//   - Tracks state for locals declared `var name: T = ...`.
//   - Function parameters enter as Init (modes are applied).
//   - Detects reads of Uninit / MaybeInit Variable nodes.
//   - if/else: precise state merge.
//   - while/for/match: single-pass merge of body output with pre-state
//     (sound for most patterns; refine to fixed-point if needed).
//
// Out of scope (deferred):
//   - Field-level init tracking (struct.field, array[i] partial init).
//   - Mode-aware parameter entry states.
//   - `move` / `undefined` callsite mode propagation.
//   - Drop synthesis driven by the live-init set.
//
// See docs/MVS.md §5 (Definite Initialization) and §8.3.

#include "init_analysis.h"
#include "ast.h"
#include <algorithm>
#include <utility>

namespace jam {
namespace init_analysis {

namespace {

using NameMap = std::unordered_map<std::string, InitState>;

// Result of analyzing a statement or expression.
//
//   state       — the binding states at the statement's exit
//   terminated  — true ⇒ control flow does not reach the next statement
//                 (the statement was a return/break/continue or contained
//                 such an exit on every path)
struct Result {
	NameMap state;
	bool terminated = false;
};

class Analyzer {
  public:
	Analyzer(const NodeStore &nodes, const StringPool &strings,
	         const std::vector<Token> &tokens, const FunctionRegistry *registry,
	         const drops::DropRegistry *drops, const TypePool *types)
	    : nodes_(nodes), strings_(strings), tokens_(tokens),
	      registry_(registry), drops_(drops), types_(types) {}

	std::vector<Diagnostic> run(const FunctionAST &fn);

  private:
	Result analyze(NodeIdx idx, NameMap state);
	Result analyzeVarDecl(NodeIdx idx, NameMap state);
	Result analyzeAssign(NodeIdx idx, NameMap state);
	Result analyzeAssignTarget(NodeIdx idx, NameMap state);
	Result analyzeIf(NodeIdx idx, NameMap state);
	Result analyzeWhile(NodeIdx idx, NameMap state);
	Result analyzeFor(NodeIdx idx, NameMap state);
	Result analyzeMatch(NodeIdx idx, NameMap state);
	Result analyzeReturn(NodeIdx idx, NameMap state);
	Result analyzeCall(NodeIdx idx, NameMap state);
	Result analyzeStructLit(NodeIdx idx, NameMap state);

	void checkVariableRead(NodeIdx idx, const NameMap &state);
	void checkDropBearingLocalsInit(const NameMap &state, NodeIdx anchor);
	void checkScopeEscape(NodeIdx exprIdx);
	void emitError(std::string message, NodeIdx anchor, std::string varName);
	int lineOf(NodeIdx idx) const;

	// Walk an argument expression (`x`, `&x`, `&p.x`, `&arr[i]`, `p.*`)
	// down to the leftmost Variable AST node, and return that node's name
	// StringIdx. Returns kNoString if the expression's base isn't a
	// trackable binding (e.g. a literal, a complex computed value).
	StringIdx findBasePathBinding(NodeIdx argIdx) const;

	// One step in a borrow path: a field projection, an array indexing,
	// or a dereference. Used by the exclusivity check to compare two
	// arg expressions for overlapping access.
	struct PathStep {
		enum Kind : uint8_t { Field, Index, Deref };
		Kind kind;
		bool indexIsConst;    // valid when kind == Index
		uint32_t fieldName;   // StringIdx; valid when kind == Field
		uint64_t constIndex;  // valid when kind == Index && indexIsConst
	};
	struct BorrowPath {
		StringIdx base = kNoString;   // base binding name, or kNoString if
		                              // the arg isn't a simple lvalue chain
		std::vector<PathStep> steps;  // root -> leaf order after extraction
	};

	BorrowPath extractPath(NodeIdx argIdx) const;
	static bool pathsOverlap(const BorrowPath &a, const BorrowPath &b);

	static NameMap mergeMaps(const NameMap &a, const NameMap &b);

	const NodeStore &nodes_;
	const StringPool &strings_;
	const std::vector<Token> &tokens_;
	const FunctionRegistry *registry_;
	const drops::DropRegistry *drops_;
	const TypePool *types_;
	std::vector<Diagnostic> diagnostics_;

	// Pointer to the current function's parameter list. Set in run(),
	// Cleared on exit. Lifetime bounded by the run() activation that set it.
	const std::vector<Param> *args_ = nullptr;

	// Static type per binding name. Populated as parameter list and
	// VarDecls are walked. Used by the drop-bearing check on `move` args.
	// Reset per function in run().
	std::unordered_map<std::string, TypeIdx> varTypes_;

	// Look up the drop function for `bindingName` if its type is a struct
	// declared with `fn drop(self: mut StructName)`. Returns nullptr when
	// no drop is registered, when the binding's type is unknown, or when
	// the drop or type registry is unavailable.
	const FunctionAST *lookupDropFor(const std::string &bindingName) const;
};

// Entry point.

std::vector<Diagnostic> Analyzer::run(const FunctionAST &fn) {
	args_ = &fn.Args;
	varTypes_.clear();

	NameMap state;
	// parameter entry state depends on the declared mode.
	//   Let / Mut / Move -> Init (caller's binding is valid)
	// `move` does not change anything for the callee's view of its own
	// parameter — the moved-from-ness applies to the *caller's* binding
	// after the call.
	for (const Param &p : fn.Args) {
		state[p.Name] = InitState::Init;
		varTypes_[p.Name] = p.Type;
	}

	Result r{std::move(state), false};
	for (NodeIdx stmt : fn.Body) {
		if (r.terminated) {
			// Statements after an unconditional return/break/continue are
			// unreachable. Don't analyze them — they would emit spurious
			// errors against state we lost track of.
			break;
		}
		r = analyze(stmt, std::move(r.state));
	}

	// If control reaches the end of the body without an
	// explicit return, every drop-bearing local must have been
	// initialized. (Functions that always return on every path produce
	// r.terminated == true and skip this check; the per-return checks
	// in analyzeReturn cover them.)
	if (!r.terminated) { checkDropBearingLocalsInit(r.state, kNoNode); }

	args_ = nullptr;
	return std::move(diagnostics_);
}

// Tag dispatch.

Result Analyzer::analyze(NodeIdx idx, NameMap state) {
	if (idx == kNoNode) return Result{std::move(state), false};
	const AstNode &n = nodes_.get(idx);

	switch (n.tag) {
	// Control flow.
	case AstTag::VarDecl:
		return analyzeVarDecl(idx, std::move(state));
	case AstTag::Assign:
		return analyzeAssign(idx, std::move(state));
	case AstTag::IfNode:
		return analyzeIf(idx, std::move(state));
	case AstTag::WhileNode:
		return analyzeWhile(idx, std::move(state));
	case AstTag::ForNode:
		return analyzeFor(idx, std::move(state));
	case AstTag::MatchNode:
		return analyzeMatch(idx, std::move(state));
	case AstTag::Return:
		return analyzeReturn(idx, std::move(state));
	case AstTag::Break:
	case AstTag::Continue:
		// Both terminate the current straight-line path. The enclosing
		// loop's caller is responsible for reasoning about loop exits.
		return Result{std::move(state), true};

	// Expressions that may read variables.
	case AstTag::Variable:
		checkVariableRead(idx, state);
		return Result{std::move(state), false};
	case AstTag::Call:
		return analyzeCall(idx, std::move(state));
	case AstTag::TypeMethodCall: {
		// Receiver is a TypeIdx — no binding state involved. Walk the
		// method's value-arg list (stored after [methodName, argCount]
		// in the extra payload) so any variable reads inside the args
		// are checked.
		ExtraIdx extra = n.rhs;
		uint32_t argCount = nodes_.getExtra(extra + 1);
		Result r{std::move(state), false};
		for (uint32_t i = 0; i < argCount; i++) {
			NodeIdx argIdx =
			    static_cast<NodeIdx>(nodes_.getExtra(extra + 2 + i));
			r = analyze(argIdx, std::move(r.state));
			if (r.terminated) return r;
		}
		return r;
	}
	case AstTag::BinaryOp: {
		auto r = analyze(n.lhs, std::move(state));
		if (r.terminated) return r;
		return analyze(n.rhs, std::move(r.state));
	}
	case AstTag::UnaryOp:
		return analyze(n.lhs, std::move(state));
	case AstTag::MemberAccess:
		// Walk the base. We treat `s.f` as reading `s` (the binding),
		// not the field — doesn't track field-level init.
		return analyze(n.lhs, std::move(state));
	case AstTag::Index: {
		auto r = analyze(n.lhs, std::move(state));
		if (r.terminated) return r;
		return analyze(n.rhs, std::move(r.state));
	}
	case AstTag::Slice: {
		// `base[start..end]` — walk the base, then the start/end bound
		// expressions (held in the extra payload) for variable reads.
		auto r = analyze(n.lhs, std::move(state));
		if (r.terminated) return r;
		ExtraIdx extra = n.rhs;
		NodeIdx startIdx = static_cast<NodeIdx>(nodes_.getExtra(extra));
		NodeIdx endIdx = static_cast<NodeIdx>(nodes_.getExtra(extra + 1));
		r = analyze(startIdx, std::move(r.state));
		if (r.terminated) return r;
		return analyze(endIdx, std::move(r.state));
	}
	case AstTag::Deref:
		// `p.*` — read the pointer. (Whether the pointee was initialized
		// is a separate concern it does not track.)
		return analyze(n.lhs, std::move(state));
	case AstTag::AddressOf:
		// `&x` — taking an address does not read the binding's value.
		// Ignore the operand for read-check purposes. it imprecisely
		// allows `&x` even when `x` is Uninit; pointer-write through it
		// is a separate analysis layer.
		return Result{std::move(state), false};
	case AstTag::AsCast:
		return analyze(n.lhs, std::move(state));
	case AstTag::StructLit:
		return analyzeStructLit(idx, std::move(state));
	case AstTag::ArrayLit: {
		// Walk every element expression so any variable reads inside the
		// array literal are checked against the init state.
		ExtraIdx extra = n.rhs;
		uint32_t count = nodes_.getExtra(extra);
		Result r{std::move(state), false};
		for (uint32_t i = 0; i < count; i++) {
			NodeIdx elemIdx =
			    static_cast<NodeIdx>(nodes_.getExtra(extra + 1 + i));
			r = analyze(elemIdx, std::move(r.state));
			if (r.terminated) return r;
		}
		return r;
	}
	case AstTag::ArrayRepeat: {
		// Walk the value expression. The count is a constant-only NumberLit
		// (enforced at codegen) — analyzing it is a no-op on init state.
		ExtraIdx extra = n.rhs;
		NodeIdx valueIdx = static_cast<NodeIdx>(nodes_.getExtra(extra));
		NodeIdx countIdx = static_cast<NodeIdx>(nodes_.getExtra(extra + 1));
		auto r = analyze(valueIdx, std::move(state));
		if (r.terminated) return r;
		return analyze(countIdx, std::move(r.state));
	}

	// Literals + comptime intrinsics — no init effect on bindings.
	// `@sizeOf(T)` / `@alignOf(T)` etc. are evaluated at codegen
	// time; they cannot read or write any runtime binding state.
	case AstTag::NumberLit:
	case AstTag::BoolLit:
	case AstTag::StringLit:
	case AstTag::ImportLit:
	case AstTag::AtCall:
	// `struct {...}` / `enum { ... }` expressions evaluate
	// to a value of type `type` at compile time. The bodies live in
	// ModuleAST and are processed by the substitution engine — the
	// analyzer doesn't see them because generic functions skip analysis
	// (they're not in mainModuleEmits). If we ever do reach these cases
	// it's a no-op.
	case AstTag::StructExpr:
	case AstTag::EnumExpr:
		return Result{std::move(state), false};

	// Pattern atoms appear only inside MatchNode arms; they don't read
	// bindings (the patterns are integer literals / ranges / wildcards).
	case AstTag::PatLit:
	case AstTag::PatRange:
	case AstTag::PatWildcard:
	case AstTag::PatOr:
	case AstTag::PatEnumVariant:
		return Result{std::move(state), false};

	case AstTag::Invalid:
	case AstTag::Count:
		return Result{std::move(state), false};
	}

	return Result{std::move(state), false};
}

// Statement / declaration cases.

Result Analyzer::analyzeVarDecl(NodeIdx idx, NameMap state) {
	// VarDecl: d.lhs = ExtraIdx -> [StringIdx name, TypeIdx type, NodeIdx init]
	const AstNode &n = nodes_.get(idx);
	ExtraIdx extra = n.lhs;
	StringIdx nameIdx = nodes_.getExtra(extra + 0);
	TypeIdx typeIdx = static_cast<TypeIdx>(nodes_.getExtra(extra + 1));
	NodeIdx initIdx = nodes_.getExtra(extra + 2);
	const std::string &name = strings_.get(nameIdx);
	varTypes_[name] = typeIdx;

	if (initIdx == kNoNode) {
		// Should not happen — Jam syntactically requires an initializer
		// for `var` declarations.
		state[name] = InitState::Init;
		return Result{std::move(state), false};
	}

	auto r = analyze(initIdx, std::move(state));
	if (r.terminated) return r;
	r.state[name] = InitState::Init;
	return r;
}

Result Analyzer::analyzeAssign(NodeIdx idx, NameMap state) {
	const AstNode &n = nodes_.get(idx);
	// Right-hand side is evaluated first (Rust semantics: drop old, then
	// store new). Sub-expressions of the RHS may read other bindings.
	auto r = analyze(n.rhs, std::move(state));
	if (r.terminated) return r;
	return analyzeAssignTarget(n.lhs, std::move(r.state));
}

// Walk an lvalue (assignment target). The "base binding" — the leftmost
// Variable in the path — is being *written to*, not read, so it does not
// trigger a read check. Sub-expressions like array indices *are* reads.
//
// In any write to a sub-path (`arr[i] = x`, `s.field = x`) marks the
// whole base binding as Init. This is intentionally imprecise: field-level
// init isn't tracked. A later phase can refine this to only mark
// the touched sub-path.
Result Analyzer::analyzeAssignTarget(NodeIdx idx, NameMap state) {
	if (idx == kNoNode) return Result{std::move(state), false};
	const AstNode &n = nodes_.get(idx);
	switch (n.tag) {
	case AstTag::Variable: {
		StringIdx nameIdx = n.lhs;
		const std::string &name = strings_.get(nameIdx);
		state[name] = InitState::Init;
		return Result{std::move(state), false};
	}
	case AstTag::MemberAccess:
		// `s.f = ...` — recurse on `s` as the lvalue base.
		return analyzeAssignTarget(n.lhs, std::move(state));
	case AstTag::Index: {
		// `a[i] = ...` — `a` is the lvalue base; `i` is read normally.
		auto r = analyze(n.rhs, std::move(state));
		if (r.terminated) return r;
		return analyzeAssignTarget(n.lhs, std::move(r.state));
	}
	case AstTag::Deref:
		// `p.* = ...` — we read `p` (the pointer) to know where to write.
		// The pointee is not a binding we track at the source level.
		return analyze(n.lhs, std::move(state));
	default:
		// Unexpected lvalue shape — fall back to a normal walk so we at
		// least surface read-checks on any sub-expressions.
		return analyze(idx, std::move(state));
	}
}

Result Analyzer::analyzeIf(NodeIdx idx, NameMap state) {
	// IfNode: d.lhs = NodeIdx (cond)
	//         d.rhs = ExtraIdx -> [thenCount, elseCount, then0, ..., else0,
	//         ...]
	const AstNode &n = nodes_.get(idx);
	auto r = analyze(n.lhs, std::move(state));
	if (r.terminated) return r;

	ExtraIdx extra = n.rhs;
	uint32_t thenCount = nodes_.getExtra(extra + 0);
	uint32_t elseCount = nodes_.getExtra(extra + 1);

	NameMap stateBeforeBranch = r.state;  // copy for the else path

	// Then-branch: walks under thenR, reusing r.state to avoid an extra copy.
	Result thenR{std::move(r.state), false};
	for (uint32_t i = 0; i < thenCount; i++) {
		if (thenR.terminated) break;
		NodeIdx s = nodes_.getExtra(extra + 2 + i);
		thenR = analyze(s, std::move(thenR.state));
	}

	// Else-branch: starts from the saved pre-branch state.
	Result elseR{std::move(stateBeforeBranch), false};
	for (uint32_t i = 0; i < elseCount; i++) {
		if (elseR.terminated) break;
		NodeIdx s = nodes_.getExtra(extra + 2 + thenCount + i);
		elseR = analyze(s, std::move(elseR.state));
	}

	// Merge.
	if (thenR.terminated && elseR.terminated) return Result{NameMap{}, true};
	if (thenR.terminated) return elseR;
	if (elseR.terminated) return thenR;
	return Result{mergeMaps(thenR.state, elseR.state), false};
}

Result Analyzer::analyzeWhile(NodeIdx idx, NameMap state) {
	// WhileNode: d.lhs = NodeIdx (cond), d.rhs = ExtraIdx ->
	//            [bodyCount, body0, body1, ...]
	const AstNode &n = nodes_.get(idx);
	auto r = analyze(n.lhs, std::move(state));
	if (r.terminated) return r;

	NameMap stateBefore = r.state;  // body may execute zero times

	ExtraIdx extra = n.rhs;
	uint32_t bodyCount = nodes_.getExtra(extra);
	Result bodyR{std::move(r.state), false};
	for (uint32_t i = 0; i < bodyCount; i++) {
		if (bodyR.terminated) break;
		NodeIdx s = nodes_.getExtra(extra + 1 + i);
		bodyR = analyze(s, std::move(bodyR.state));
	}

	// Loop body executes zero or more times. The post-loop state is the
	// merge of "body never ran" (stateBefore) with "body ran at least once"
	// (bodyR.state). If bodyR terminated (via break/return), the merge
	// degenerates to stateBefore.
	if (bodyR.terminated) return Result{std::move(stateBefore), false};
	return Result{mergeMaps(stateBefore, bodyR.state), false};
}

Result Analyzer::analyzeFor(NodeIdx idx, NameMap state) {
	// ForNode: d.lhs = ExtraIdx -> [StringIdx var, NodeIdx start, NodeIdx end,
	//                              bodyCount, body0, body1, ...]
	const AstNode &n = nodes_.get(idx);
	ExtraIdx extra = n.lhs;
	StringIdx varIdx = nodes_.getExtra(extra + 0);
	NodeIdx startIdx = nodes_.getExtra(extra + 1);
	NodeIdx endIdx = nodes_.getExtra(extra + 2);
	uint32_t bodyCount = nodes_.getExtra(extra + 3);

	auto r = analyze(startIdx, std::move(state));
	if (r.terminated) return r;
	r = analyze(endIdx, std::move(r.state));
	if (r.terminated) return r;

	NameMap stateBefore = r.state;
	const std::string &varName = strings_.get(varIdx);
	r.state[varName] = InitState::Init;

	Result bodyR{std::move(r.state), false};
	for (uint32_t i = 0; i < bodyCount; i++) {
		if (bodyR.terminated) break;
		NodeIdx s = nodes_.getExtra(extra + 4 + i);
		bodyR = analyze(s, std::move(bodyR.state));
	}

	// For-over-range: it assumes the body executes at least once. This
	// matches existing Jam idioms (e.g., `for i in 0:N { arr[i] = ... }`
	// followed by reading `arr`). Strictly speaking the body may run
	// zero times for an empty range; accepting that imprecision keeps
	// the analyzer practical on real code. A later phase can refine
	// when start/end are known constants.
	//
	// If the body terminated via `break`, fall back to stateBefore — we
	// can't tell whether the break ran on the first iteration or after
	// the body had effects. Conservative: pretend body never ran.
	if (bodyR.terminated) return Result{std::move(stateBefore), false};
	return bodyR;
}

Result Analyzer::analyzeMatch(NodeIdx idx, NameMap state) {
	// MatchNode: d.lhs = NodeIdx (scrutinee), d.rhs = ExtraIdx ->
	//   [armCount,
	//    arm0_patIdx, arm0_bodyCount, arm0_body...,
	//    arm1_patIdx, arm1_bodyCount, arm1_body..., ...]
	// Catch-all is the wildcard pattern `_`; there is no `else` arm.
	const AstNode &n = nodes_.get(idx);
	auto r = analyze(n.lhs, std::move(state));
	if (r.terminated) return r;

	ExtraIdx extra = n.rhs;
	uint32_t armCount = nodes_.getExtra(extra + 0);

	NameMap stateBefore = r.state;

	// Start as "no arm taken yet"; merge each arm's outcome below.
	Result mergedR{NameMap{}, true};

	uint32_t cursor = 1;
	for (uint32_t a = 0; a < armCount; a++) {
		// Arm pattern (index ignored for init analysis — patterns don't
		// touch existing-binding state).
		cursor++;  // patIdx
		uint32_t armBodyCount = nodes_.getExtra(extra + cursor);
		cursor++;

		Result armR{stateBefore, false};
		for (uint32_t i = 0; i < armBodyCount; i++) {
			if (armR.terminated) break;
			NodeIdx s = nodes_.getExtra(extra + cursor + i);
			armR = analyze(s, std::move(armR.state));
		}
		cursor += armBodyCount;

		if (mergedR.terminated && armR.terminated) {
			// both paths terminated; stay terminated
		} else if (mergedR.terminated) {
			mergedR = std::move(armR);
		} else if (armR.terminated) {
			// keep mergedR
		} else {
			mergedR.state = mergeMaps(mergedR.state, armR.state);
		}
	}

	return mergedR;
}

Result Analyzer::analyzeReturn(NodeIdx idx, NameMap state) {
	const AstNode &n = nodes_.get(idx);
	// scope-escape check before any reads of the operand. Returning
	// `&` of a borrow-shaped (mut/undefined) parameter would extend the
	// borrow's lifetime past the call frame. Returning the parameter as
	// a value (without `&`) is a copy and is fine — that's value semantics.
	if (n.lhs != kNoNode) checkScopeEscape(n.lhs);

	Result r{std::move(state), false};
	if (n.lhs != kNoNode) { r = analyze(n.lhs, std::move(r.state)); }
	// every drop-bearing local must be Init at every return path —
	// codegen will emit drop on it at this exit, and dropping uninit
	// memory is UB.
	checkDropBearingLocalsInit(r.state, idx);
	r.terminated = true;
	return r;
}

Result Analyzer::analyzeCall(NodeIdx idx, NameMap state) {
	// Call: d.lhs = StringIdx (callee), d.rhs = ExtraIdx -> [argCount, args...]
	const AstNode &n = nodes_.get(idx);
	const FunctionAST *callee = nullptr;
	if ((n.flags & 1) == 0) {
		StringIdx calleeIdx = n.lhs;
		const std::string &calleeName = strings_.get(calleeIdx);
		if (registry_) {
			auto it = registry_->find(calleeName);
			if (it != registry_->end()) callee = it->second;
		}
	}

	ExtraIdx extra = n.rhs;
	uint32_t argCount = nodes_.getExtra(extra);

	// Pre-pass: collect (path, mode) for each arg so we can run the
	// exclusivity check before applying any state transitions.
	// Doing this first means a flagged conflict still surfaces even if
	// one of the args would have terminated the analysis (rare).
	struct ArgInfo {
		NodeIdx argIdx;
		ParamMode mode;
		BorrowPath path;
	};
	std::vector<ArgInfo> argInfos;
	argInfos.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = nodes_.getExtra(extra + 1 + i);
		ParamMode mode = ParamMode::Let;
		if (callee && i < callee->Args.size()) { mode = callee->Args[i].Mode; }
		argInfos.push_back({argIdx, mode, extractPath(argIdx)});
	}

	// exclusivity rule. For each pair of args with overlapping paths,
	// the borrow set is OK only if both modes are Let (multiple readers).
	// Any other combination on overlapping paths is rejected.
	for (std::size_t i = 0; i < argInfos.size(); i++) {
		for (std::size_t j = i + 1; j < argInfos.size(); j++) {
			const ArgInfo &a = argInfos[i];
			const ArgInfo &b = argInfos[j];
			if (a.mode == ParamMode::Let && b.mode == ParamMode::Let) continue;
			if (!pathsOverlap(a.path, b.path)) continue;
			if (a.path.base == kNoString) continue;  // un-tracked, skip
			const std::string &name = strings_.get(a.path.base);
			emitError("conflicting borrows of `" + name +
			              "` in the same call: at least one access is "
			              "exclusive (mut/move/undefined)",
			          a.argIdx, name);
		}
	}

	Result r{std::move(state), false};

	// walk the callee subtree when present so its read-checks fire
	// (e.g. `arr[i].method()` must verify `arr` is initialized).
	if (n.flags & 1) {
		r = analyze(static_cast<NodeIdx>(n.lhs), std::move(r.state));
		if (r.terminated) return r;
	}

	for (const ArgInfo &info : argInfos) {
		if (r.terminated) return r;
		// Walk the arg expression. For `let`/`mut`/`move` modes the walk
		// includes a read-check on the base binding (it must already be
		// Init).
		r = analyze(info.argIdx, std::move(r.state));
		if (r.terminated) return r;

		// Post-call mode effect on the caller's binding.
		//   move      — caller's binding moves into the callee; becomes Uninit.
		//   let / mut — no caller-side state change.
		if (info.mode == ParamMode::Move) {
			if (info.path.base != kNoString) {
				const std::string &name = strings_.get(info.path.base);

				// reject `move` on a drop-bearing binding
				// until move-aware drop tracking lands. Without
				// it, codegen would emit drop on the moved-out slot at
				// scope exit — a double-free.
				if (lookupDropFor(name) != nullptr) {
					emitError("cannot `move` binding `" + name +
					              "` of drop-bearing type — drop+move "
					              "tracking is not yet implemented; "
					              "consider passing as `mut` or `let` "
					              "instead",
					          info.argIdx, name);
				}

				r.state[name] = InitState::Uninit;
			}
		}
	}
	return r;
}

StringIdx Analyzer::findBasePathBinding(NodeIdx argIdx) const {
	NodeIdx cur = argIdx;
	while (cur != kNoNode) {
		const AstNode &n = nodes_.get(cur);
		switch (n.tag) {
		case AstTag::Variable:
			return n.lhs;
		case AstTag::AddressOf:
		case AstTag::MemberAccess:
		case AstTag::Index:
		case AstTag::Deref:
			cur = n.lhs;
			break;
		default:
			// Not a simple lvalue chain — no trackable base binding.
			return kNoString;
		}
	}
	return kNoString;
}

// Extract a borrow path from a call-arg expression. The expression must
// be a chain of `&`, `.field`, `[idx]`, `.*` rooted at a Variable node;
// any other shape leaves `base = kNoString` and the caller skips the
// exclusivity check for that arg (we can't reason about its identity).
//
// `&` is transparent — it indicates a non-default borrow at the callsite
// but does not contribute a step to the path. The path's exclusivity
// status comes from the parameter mode, not from the `&` token.
Analyzer::BorrowPath Analyzer::extractPath(NodeIdx argIdx) const {
	BorrowPath path;
	NodeIdx cur = argIdx;
	while (cur != kNoNode) {
		const AstNode &n = nodes_.get(cur);
		switch (n.tag) {
		case AstTag::Variable:
			path.base = n.lhs;
			std::reverse(path.steps.begin(), path.steps.end());
			return path;
		case AstTag::AddressOf:
			// Pass through; not a path step.
			cur = n.lhs;
			break;
		case AstTag::MemberAccess: {
			PathStep s;
			s.kind = PathStep::Field;
			s.indexIsConst = false;
			s.fieldName = n.rhs;
			s.constIndex = 0;
			path.steps.push_back(s);
			cur = n.lhs;
			break;
		}
		case AstTag::Index: {
			PathStep s;
			s.kind = PathStep::Index;
			s.fieldName = 0;
			NodeIdx idxExpr = n.rhs;
			const AstNode &idxNode = nodes_.get(idxExpr);
			if (idxNode.tag == AstTag::NumberLit) {
				// d.lhs = lo32, d.rhs = hi32, flags bit 0 = isNeg
				uint64_t mag =
				    uint64_t(idxNode.lhs) | (uint64_t(idxNode.rhs) << 32);
				s.indexIsConst = ((idxNode.flags & 1u) == 0);
				s.constIndex = mag;
			} else {
				s.indexIsConst = false;
				s.constIndex = 0;
			}
			path.steps.push_back(s);
			cur = n.lhs;
			break;
		}
		case AstTag::Deref: {
			PathStep s;
			s.kind = PathStep::Deref;
			s.indexIsConst = false;
			s.fieldName = 0;
			s.constIndex = 0;
			path.steps.push_back(s);
			cur = n.lhs;
			break;
		}
		default:
			// Computed expression — not a tracked path. Leave base as
			// kNoString so the exclusivity check skips it.
			path.steps.clear();
			return path;
		}
	}
	return path;
}

// Two paths overlap when one is a prefix of the other (or they are
// equal). Per MVS.md §4.1:
//   - Same base required; different bases -> disjoint.
//   - Step-by-step compare on common prefix:
//       Field vs Field: same name? continue : disjoint
//       Index vs Index: both const & equal? continue : differ-const? disjoint
//                       at least one dynamic? conservatively overlap
//       Deref vs Deref: continue
//       different kinds: disjoint
//   - When all common steps match and one path runs out: overlap (prefix).
bool Analyzer::pathsOverlap(const BorrowPath &a, const BorrowPath &b) {
	if (a.base == kNoString || b.base == kNoString) return false;
	if (a.base != b.base) return false;
	std::size_t common = std::min(a.steps.size(), b.steps.size());
	for (std::size_t i = 0; i < common; i++) {
		const PathStep &sa = a.steps[i];
		const PathStep &sb = b.steps[i];
		if (sa.kind != sb.kind) return false;
		switch (sa.kind) {
		case PathStep::Field:
			if (sa.fieldName != sb.fieldName) return false;
			break;
		case PathStep::Index:
			if (sa.indexIsConst && sb.indexIsConst) {
				if (sa.constIndex != sb.constIndex) return false;
			} else {
				// At least one dynamic index — conservatively overlap.
				return true;
			}
			break;
		case PathStep::Deref:
			break;  // both deref through the same pointer position
		}
	}
	return true;
}

Result Analyzer::analyzeStructLit(NodeIdx idx, NameMap state) {
	// StructLit: d.rhs = ExtraIdx ->
	//   [fieldCount, fieldName0, fieldExpr0, fieldName1, fieldExpr1, ...]
	const AstNode &n = nodes_.get(idx);
	ExtraIdx extra = n.rhs;
	uint32_t fieldCount = nodes_.getExtra(extra);
	Result r{std::move(state), false};
	for (uint32_t i = 0; i < fieldCount; i++) {
		if (r.terminated) return r;
		NodeIdx exprIdx = nodes_.getExtra(extra + 1 + 2 * i + 1);
		r = analyze(exprIdx, std::move(r.state));
	}
	return r;
}

// Read check + diagnostics.

void Analyzer::checkVariableRead(NodeIdx idx, const NameMap &state) {
	const AstNode &n = nodes_.get(idx);
	StringIdx nameIdx = n.lhs;
	const std::string &name = strings_.get(nameIdx);

	auto it = state.find(name);
	if (it == state.end()) {
		// Not in our tracking — likely a global, an imported name, or a
		// for-loop variable that escaped its scope. it conservatively
		// assumes Init for unseen names so it doesn't false-fire.
		return;
	}

	switch (it->second) {
	case InitState::Init:
	case InitState::Unknown:
		return;  // OK
	case InitState::Uninit:
		emitError("use of uninitialized binding `" + name + "`", idx, name);
		return;
	case InitState::MaybeInit:
		emitError(
		    "binding `" + name +
		        "` may not be initialized on every path that reaches here",
		    idx, name);
		return;
	}
}

void Analyzer::checkDropBearingLocalsInit(const NameMap &state,
                                          NodeIdx anchor) {
	// Walk only the bindings that are *currently in scope* — i.e.,
	// present in the merged state map at this exit point. Bindings
	// declared inside a now-exited inner block have already been dropped
	// at their own scope end, and the merge has removed
	// them from the state map; checking them here would produce false
	// positives because varTypes_ retains every name we've ever seen.
	//
	// For each in-scope binding whose type has a registered drop fn,
	// the codegen will synthesize a drop call at this exit. If the
	// binding is Uninit or MaybeInit, drop runs on garbage memory.
	if (!drops_ || !types_) return;
	for (const auto &sv : state) {
		const std::string &name = sv.first;
		// Skip parameters — drops for owning-mode params would belong to
		// the caller's binding, not the callee. Today's codegen only
		// drops `var` locals.
		bool isParam = false;
		if (args_) {
			for (const Param &p : *args_) {
				if (p.Name == name) {
					isParam = true;
					break;
				}
			}
		}
		if (isParam) continue;

		auto vt = varTypes_.find(name);
		if (vt == varTypes_.end()) continue;
		const TypeKey &k = types_->get(vt->second);
		if (k.kind != TypeKind::Struct && k.kind != TypeKind::Named) continue;
		StringIdx structNameIdx = static_cast<StringIdx>(k.a);
		if (structNameIdx == kNoString) continue;
		const std::string &structName = strings_.get(structNameIdx);
		if (drops_->find(structName) == drops_->end()) continue;

		InitState s = sv.second;
		if (s == InitState::Init) continue;

		std::string msg =
		    "drop-bearing binding `" + name + "` of type `" + structName + "` ";
		if (s == InitState::Uninit) {
			msg += "must be initialized before this exit; drop runs on it "
			       "and would otherwise read uninit memory";
		} else {
			msg += "may not be initialized on every path that reaches this "
			       "exit; drop runs on it on every path";
		}
		emitError(std::move(msg), anchor, name);
	}
}

void Analyzer::checkScopeEscape(NodeIdx exprIdx) {
	// Only `&`-rooted expressions can carry a borrow out of the function.
	// Returning a parameter "by value" (`return p;`, `return p.x;`) is
	// runtime-equivalent to a copy, so we don't flag it.
	if (exprIdx == kNoNode || args_ == nullptr) return;
	const AstNode &top = nodes_.get(exprIdx);
	if (top.tag != AstTag::AddressOf) return;

	// Walk the path under the `&` to find the base Variable. We descend
	// through MemberAccess / Index (so `&p.field` and `&arr[i]` are
	// covered) but stop at Deref (the dereferenced pointee is itself
	// data, not the borrow path).
	NodeIdx cur = top.lhs;
	while (cur != kNoNode) {
		const AstNode &m = nodes_.get(cur);
		switch (m.tag) {
		case AstTag::Variable: {
			StringIdx nameId = m.lhs;
			const std::string &name = strings_.get(nameId);
			for (const Param &p : *args_) {
				if (p.Name != name) continue;
				if (p.Mode != ParamMode::Mut) return;
				std::string msg = "cannot return `&` of `mut`-mode "
				                  "parameter `" +
				                  name +
				                  "` — borrows are second-class and "
				                  "cannot escape the function";
				emitError(std::move(msg), exprIdx, name);
				return;
			}
			return;  // not a parameter at all
		}
		case AstTag::MemberAccess:
		case AstTag::Index:
			cur = m.lhs;
			break;
		default:
			// Deref or anything else: not a borrow path on the parameter.
			return;
		}
	}
}

int Analyzer::lineOf(NodeIdx idx) const {
	if (idx == kNoNode) return 0;
	const AstNode &n = nodes_.get(idx);
	if (n.mainToken >= tokens_.size()) return 0;
	return tokens_[n.mainToken].line;
}

void Analyzer::emitError(std::string message, NodeIdx anchor,
                         std::string varName) {
	Diagnostic d;
	d.message = std::move(message);
	d.line = lineOf(anchor);
	d.varName = std::move(varName);
	diagnostics_.push_back(std::move(d));
}

// Lattice merge.

NameMap Analyzer::mergeMaps(const NameMap &a, const NameMap &b) {
	NameMap result = a;
	for (const auto &kv : b) {
		auto it = result.find(kv.first);
		if (it == result.end()) {
			result[kv.first] = kv.second;
		} else {
			it->second = mergeState(it->second, kv.second);
		}
	}
	return result;
}

}  // namespace

// Public entry point.

std::vector<Diagnostic> analyze(const FunctionAST &fn, const NodeStore &nodes,
                                const StringPool &strings,
                                const std::vector<Token> &tokens,
                                const FunctionRegistry *registry,
                                const drops::DropRegistry *drops,
                                const TypePool *types) {
	Analyzer a(nodes, strings, tokens, registry, drops, types);
	return a.run(fn);
}

const FunctionAST *
Analyzer::lookupDropFor(const std::string &bindingName) const {
	if (!drops_ || !types_) return nullptr;
	auto vit = varTypes_.find(bindingName);
	if (vit == varTypes_.end()) return nullptr;
	const TypeKey &k = types_->get(vit->second);
	if (k.kind != TypeKind::Struct && k.kind != TypeKind::Named) return nullptr;
	StringIdx structNameIdx = static_cast<StringIdx>(k.a);
	if (structNameIdx == kNoString) return nullptr;
	const std::string &structName = strings_.get(structNameIdx);
	auto dit = drops_->find(structName);
	return (dit == drops_->end()) ? nullptr : dit->second;
}

}  // namespace init_analysis
}  // namespace jam
