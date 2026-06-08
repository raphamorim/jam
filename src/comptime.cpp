/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "comptime.h"

#include "jam_llvm.h"
#include "target.h"

namespace jam {

// ─── ComptimeValue constructors ──────────────────────────────────

ComptimeValue ComptimeValue::makeNone() {
	ComptimeValue v;
	v.kind = Kind::None;
	return v;
}

ComptimeValue ComptimeValue::makeInt(uint64_t bits, uint16_t width,
                                     bool isSigned) {
	ComptimeValue v;
	v.kind = Kind::Int;
	v.intVal.bits = bits;
	v.intVal.width = width;
	v.intVal.isSigned = isSigned;
	return v;
}

ComptimeValue ComptimeValue::makeFloat(double value, uint16_t width) {
	ComptimeValue v;
	v.kind = Kind::Float;
	v.floatVal.value = value;
	v.floatVal.width = width;
	return v;
}

ComptimeValue ComptimeValue::makeBool(bool b) {
	ComptimeValue v;
	v.kind = Kind::Bool;
	v.boolVal = b;
	return v;
}

ComptimeValue ComptimeValue::makeStr(StringIdx s) {
	ComptimeValue v;
	v.kind = Kind::Str;
	v.strVal = s;
	return v;
}

ComptimeValue ComptimeValue::makeType(TypeIdx t) {
	ComptimeValue v;
	v.kind = Kind::Type;
	v.typeVal = t;
	return v;
}

ComptimeValue ComptimeValue::makeAggregate(std::vector<ComptimeValue> fields) {
	ComptimeValue v;
	v.kind = Kind::Aggregate;
	v.aggFields = std::move(fields);
	return v;
}

int64_t ComptimeValue::asI64() const {
	if (kind != Kind::Int) return 0;
	// Sign-extend from the value's bit-width so `i8(-1)` returns -1
	// (not 255). Unsigned values are returned as-is up to 63 bits.
	uint64_t b = intVal.bits;
	if (intVal.isSigned && intVal.width < 64) {
		uint64_t signBit = 1ULL << (intVal.width - 1);
		if (b & signBit) {
			uint64_t mask = ~((1ULL << intVal.width) - 1);
			b |= mask;
		}
	}
	return static_cast<int64_t>(b);
}

uint64_t ComptimeValue::asU64() const {
	if (kind != Kind::Int) return 0;
	// Mask to the value's bit-width so over-wide bit patterns can't
	// leak. `i8(-1)` returns 0xFF here, `u8(255)` also returns 0xFF.
	if (intVal.width >= 64) return intVal.bits;
	return intVal.bits & ((1ULL << intVal.width) - 1);
}

// ─── ComptimeScope ──────────────────────────────────────────────

void ComptimeScope::bind(const std::string &name, ComptimeValue value) {
	bindings_[name] = std::move(value);
}

bool ComptimeScope::set(const std::string &name, ComptimeValue value) {
	// Walk up the parent chain to find the scope where `name` was
	// originally bound. Mutate there. Returns false if `name` isn't
	// bound anywhere in the chain — caller decides how to surface
	// (typically a diagnostic).
	ComptimeScope *cur = this;
	while (cur != nullptr) {
		auto it = cur->bindings_.find(name);
		if (it != cur->bindings_.end()) {
			it->second = std::move(value);
			return true;
		}
		cur = cur->parent_;
	}
	return false;
}

const ComptimeValue *ComptimeScope::lookup(const std::string &name) const {
	auto it = bindings_.find(name);
	if (it != bindings_.end()) return &it->second;
	if (parent_ != nullptr) return parent_->lookup(name);
	return nullptr;
}

// ─── Evaluator ──────────────────────────────────────────────────

ComptimeEvaluator::ComptimeEvaluator(const NodeStore &nodes,
                                     const StringPool &strings,
                                     const TypePool &types)
    : nodes_(nodes), strings_(strings), types_(types) {}

ComptimeValue ComptimeEvaluator::eval(NodeIdx expr,
                                      const ComptimeScope &scope) const {
	if (expr == kNoNode) return ComptimeValue::makeNone();
	const AstNode &n = nodes_.get(expr);
	switch (n.tag) {
	case AstTag::NumberLit:
		return evalNumberLit(n);
	case AstTag::BoolLit:
		return evalBoolLit(n);
	case AstTag::StringLit:
		return evalStringLit(n);
	case AstTag::Variable:
		return evalVariable(n, scope);
	case AstTag::UnaryOp:
		return evalUnaryOp(n, scope);
	case AstTag::BinaryOp:
		return evalBinaryOp(n, scope);
	case AstTag::Index:
		return evalIndex(n, scope);
	case AstTag::MemberAccess:
		return evalMemberAccess(n, scope);
	case AstTag::AtCall:
		return evalAtCall(n, scope);
	case AstTag::Call:
		return evalCall(n, scope);
	default:
		// Operator / construct we don't fold yet. Returning None keeps
		// optional-fold callers (peephole constant folding) silent;
		// `evalRequired` will turn it into a diagnostic.
		return ComptimeValue::makeNone();
	}
}

ComptimeValue ComptimeEvaluator::evalCall(const AstNode &n,
                                          const ComptimeScope &scope) const {
	// Only direct calls fold (flags bit 0 set = indirect method call on
	// a runtime value — never comptime). Direct form: lhs = callee
	// StringIdx, rhs = ExtraIdx -> [argCount, arg0, ...].
	if ((n.flags & 1) != 0) return ComptimeValue::makeNone();
	if (resolver_ == nullptr) return ComptimeValue::makeNone();

	const std::string &name = strings_.get(static_cast<StringIdx>(n.lhs));
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = nodes_.getExtra(extra);
	std::vector<ComptimeValue> argVals;
	argVals.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(nodes_.getExtra(extra + 1 + i));
		ComptimeValue v = eval(argIdx, scope);
		if (v.isNone()) {
			if (diags_ != nullptr) {
				diags_->error(loc_, "argument #" + std::to_string(i) +
				                        " to comptime call `" + name +
				                        "` is not a compile-time constant");
			}
			return ComptimeValue::makeNone();
		}
		argVals.push_back(std::move(v));
	}

	Diagnostics dummy;
	Diagnostics &d = diags_ != nullptr ? *diags_ : dummy;
	return resolver_->resolveCall(name, argVals, d, loc_);
}

ComptimeValue ComptimeEvaluator::evalAtCall(const AstNode &n,
                                            const ComptimeScope &scope) const {
	// No-arg / type-arg intrinsics (flags bit 0 clear). The target-OS
	// predicates fold to a comptime bool here so `comp if (@isDarwin())`
	// works — they're pure queries of the (host) target, the same
	// source astgenAtCall reads. `@sizeOf` / `@alignOf` need the
	// codegen context's layout tables, which this evaluator doesn't
	// hold; they stay None here and fold via astgen (Stage 5 adds a
	// hook to share them). `@os()` returns a string and isn't used for
	// branching, so it's astgen-only too.
	if ((n.flags & 1) == 0) {
		const std::string &iname = strings_.get(static_cast<StringIdx>(n.lhs));
		if (iname == "isDarwin" || iname == "isLinux" || iname == "isWindows" ||
		    iname == "isUnix") {
			jam::OS os = jam::Target::getHostTarget().os;
			bool v = (iname == "isDarwin")  ? (os == jam::OS::MacOS)
			         : (iname == "isLinux") ? (os == jam::OS::Linux)
			         : (iname == "isWindows")
			             ? (os == jam::OS::Windows)
			             : (os == jam::OS::MacOS || os == jam::OS::Linux ||
			                os == jam::OS::FreeBSD);  // isUnix
			return ComptimeValue::makeBool(v);
		}
		return ComptimeValue::makeNone();
	}

	// Expr-arg multi-form: rhs = ExtraIdx -> [argCount, arg0, ...].
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = nodes_.getExtra(extra);
	std::vector<ComptimeValue> argVals;
	argVals.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(nodes_.getExtra(extra + 1 + i));
		ComptimeValue v = eval(argIdx, scope);
		if (v.isNone()) {
			if (diags_ != nullptr) {
				diags_->error(loc_, "@-emit argument must be a compile-time "
				                    "constant (arg #" +
				                        std::to_string(i) + ")");
			}
			return ComptimeValue::makeNone();
		}
		argVals.push_back(std::move(v));
	}

	if (emitter_ == nullptr) {
		// No emitter installed — running outside a cfn dispatcher
		// context. Silently return None; the caller (test harness,
		// peephole folder) decides what to do.
		return ComptimeValue::makeNone();
	}

	const std::string &name = strings_.get(static_cast<StringIdx>(n.lhs));
	Diagnostics dummyDiags;
	Diagnostics &diagsRef = diags_ != nullptr ? *diags_ : dummyDiags;
	ExecResult r = emitter_->handleAtCall(name, argVals, diagsRef, loc_);
	(void)r;  // emit-style intrinsics return Continue on success;
	          // errors surface via pushed diagnostics + the
	          // caller's `Diagnostics::hasErrors()` post-check.
	// @-emit calls produce side effects, not values.
	return ComptimeValue::makeNone();
}

ComptimeValue ComptimeEvaluator::evalRequired(NodeIdx expr,
                                              const ComptimeScope &scope,
                                              Diagnostics &diags,
                                              SrcLoc loc) const {
	ComptimeValue v = eval(expr, scope);
	if (v.isNone()) {
		diags.error(std::move(loc),
		            "expression cannot be evaluated at compile time");
	}
	return v;
}

ComptimeValue ComptimeEvaluator::evalNumberLit(const AstNode &n) const {
	bool isNeg = (n.flags & 1) != 0;
	bool isFloat = (n.flags & 2) != 0;
	if (isFloat) {
		// The comptime float value is f64. Either the literal is stored inline
		// as f64 bits (lossless), or as full f128 in the extra pool (flag bit
		// 2) which we round once to f64 (f128→f64 equals decimal→f64; no
		// double- rounding). See the parser / astgenNumberLit.
		double v;
		if ((n.flags & 4) != 0) {
			ExtraIdx ei = static_cast<ExtraIdx>(n.lhs);
			uint32_t quad[4] = {nodes_.getExtra(ei), nodes_.getExtra(ei + 1),
			                    nodes_.getExtra(ei + 2),
			                    nodes_.getExtra(ei + 3)};
			v = JamLLVMQuadToTargetAsDouble(quad, /*toF32=*/false);
		} else {
			uint64_t bits = static_cast<uint64_t>(n.lhs) |
			                (static_cast<uint64_t>(n.rhs) << 32);
			__builtin_memcpy(&v, &bits, sizeof(v));
		}
		if (isNeg) v = -v;
		return ComptimeValue::makeFloat(v, 64);
	}
	uint64_t bits =
	    static_cast<uint64_t>(n.lhs) | (static_cast<uint64_t>(n.rhs) << 32);
	// Default integer width: u64 (or i64 if negative). Callers can
	// narrow via the surrounding type context, but at this evaluator
	// layer we keep the literal at full width to preserve precision
	// during folding.
	if (isNeg) {
		uint64_t magnitude = bits;
		uint64_t signedBits =
		    static_cast<uint64_t>(-static_cast<int64_t>(magnitude));
		return ComptimeValue::makeInt(signedBits, 64, /*isSigned=*/true);
	}
	return ComptimeValue::makeInt(bits, 64, /*isSigned=*/false);
}

ComptimeValue ComptimeEvaluator::evalBoolLit(const AstNode &n) const {
	return ComptimeValue::makeBool(n.lhs != 0);
}

ComptimeValue ComptimeEvaluator::evalStringLit(const AstNode &n) const {
	return ComptimeValue::makeStr(static_cast<StringIdx>(n.lhs));
}

ComptimeValue
ComptimeEvaluator::evalVariable(const AstNode &n,
                                const ComptimeScope &scope) const {
	const std::string &name = strings_.get(static_cast<StringIdx>(n.lhs));
	const ComptimeValue *v = scope.lookup(name);
	if (v == nullptr) return ComptimeValue::makeNone();
	return *v;
}

ComptimeValue ComptimeEvaluator::evalUnaryOp(const AstNode &n,
                                             const ComptimeScope &scope) const {
	NodeIdx operand = static_cast<NodeIdx>(n.lhs);
	ComptimeValue v = eval(operand, scope);
	if (v.isNone()) return v;
	UnaryOp op = static_cast<UnaryOp>(n.op);
	switch (op) {
	case UnaryOp::Neg:
		if (v.kind == ComptimeValue::Kind::Int) {
			uint64_t neg =
			    static_cast<uint64_t>(-static_cast<int64_t>(v.asU64()));
			return ComptimeValue::makeInt(neg, v.intVal.width, true);
		}
		if (v.kind == ComptimeValue::Kind::Float) {
			return ComptimeValue::makeFloat(-v.floatVal.value,
			                                v.floatVal.width);
		}
		return ComptimeValue::makeNone();
	case UnaryOp::LogNot:
		if (v.kind != ComptimeValue::Kind::Bool)
			return ComptimeValue::makeNone();
		return ComptimeValue::makeBool(!v.boolVal);
	case UnaryOp::BitNot:
		if (v.kind != ComptimeValue::Kind::Int)
			return ComptimeValue::makeNone();
		return ComptimeValue::makeInt(~v.intVal.bits, v.intVal.width,
		                              v.intVal.isSigned);
	default:
		return ComptimeValue::makeNone();
	}
}

ComptimeValue
ComptimeEvaluator::evalBinaryOp(const AstNode &n,
                                const ComptimeScope &scope) const {
	NodeIdx lhsIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx rhsIdx = static_cast<NodeIdx>(n.rhs);
	BinOp op = static_cast<BinOp>(n.op);

	// Short-circuit eval for LogAnd/LogOr — match the runtime semantics
	// and avoid evaluating the RHS when the LHS settles the answer.
	if (op == BinOp::LogAnd || op == BinOp::LogOr) {
		ComptimeValue l = eval(lhsIdx, scope);
		if (l.kind != ComptimeValue::Kind::Bool)
			return ComptimeValue::makeNone();
		bool lb = l.boolVal;
		if (op == BinOp::LogAnd && !lb) return ComptimeValue::makeBool(false);
		if (op == BinOp::LogOr && lb) return ComptimeValue::makeBool(true);
		ComptimeValue r = eval(rhsIdx, scope);
		if (r.kind != ComptimeValue::Kind::Bool)
			return ComptimeValue::makeNone();
		return ComptimeValue::makeBool(r.boolVal);
	}

	ComptimeValue l = eval(lhsIdx, scope);
	if (l.isNone()) return l;
	ComptimeValue r = eval(rhsIdx, scope);
	if (r.isNone()) return r;

	// Integer arithmetic + bitwise + comparison. Strict width matching
	// for arithmetic / bitwise (catches accidental mixed-width math);
	// comparisons coerce mismatched-width int literals so user code
	// like `fmt[i] == 123` (u8 vs default-u64 literal) works without
	// explicit width casts. Coercion direction: widen the narrower
	// operand to the wider one; require matching signedness, otherwise
	// give up and return None.
	if (l.kind == ComptimeValue::Kind::Int &&
	    r.kind == ComptimeValue::Kind::Int) {
		bool isComparison = op == BinOp::Eq || op == BinOp::Ne ||
		                    op == BinOp::Lt || op == BinOp::Le ||
		                    op == BinOp::Gt || op == BinOp::Ge;
		if (l.intVal.width != r.intVal.width) {
			if (!isComparison || l.intVal.isSigned != r.intVal.isSigned) {
				return ComptimeValue::makeNone();
			}
			// Widen the narrower operand. Bit-pattern preserved
			// because both sides are unsigned-or-signed alike at this
			// point.
			if (l.intVal.width < r.intVal.width) {
				l.intVal.width = r.intVal.width;
			} else {
				r.intVal.width = l.intVal.width;
			}
		}
		if (l.intVal.isSigned != r.intVal.isSigned) {
			return ComptimeValue::makeNone();
		}
		uint16_t w = l.intVal.width;
		bool sgn = l.intVal.isSigned;
		uint64_t a = l.intVal.bits;
		uint64_t b = r.intVal.bits;
		switch (op) {
		case BinOp::Add:
			return ComptimeValue::makeInt(a + b, w, sgn);
		case BinOp::Sub:
			return ComptimeValue::makeInt(a - b, w, sgn);
		case BinOp::Mul:
			return ComptimeValue::makeInt(a * b, w, sgn);
		case BinOp::Div:
			if (b == 0) return ComptimeValue::makeNone();
			if (sgn) {
				return ComptimeValue::makeInt(
				    static_cast<uint64_t>(static_cast<int64_t>(a) /
				                          static_cast<int64_t>(b)),
				    w, sgn);
			}
			return ComptimeValue::makeInt(a / b, w, sgn);
		case BinOp::Mod:
			if (b == 0) return ComptimeValue::makeNone();
			if (sgn) {
				return ComptimeValue::makeInt(
				    static_cast<uint64_t>(static_cast<int64_t>(a) %
				                          static_cast<int64_t>(b)),
				    w, sgn);
			}
			return ComptimeValue::makeInt(a % b, w, sgn);
		case BinOp::BitAnd:
			return ComptimeValue::makeInt(a & b, w, sgn);
		case BinOp::BitOr:
			return ComptimeValue::makeInt(a | b, w, sgn);
		case BinOp::BitXor:
			return ComptimeValue::makeInt(a ^ b, w, sgn);
		case BinOp::Shl:
			return ComptimeValue::makeInt(a << b, w, sgn);
		case BinOp::Shr:
			if (sgn) {
				return ComptimeValue::makeInt(
				    static_cast<uint64_t>(static_cast<int64_t>(a) >> b), w,
				    sgn);
			}
			return ComptimeValue::makeInt(a >> b, w, sgn);
		case BinOp::Eq:
			return ComptimeValue::makeBool(a == b);
		case BinOp::Ne:
			return ComptimeValue::makeBool(a != b);
		case BinOp::Lt:
			return sgn ? ComptimeValue::makeBool(static_cast<int64_t>(a) <
			                                     static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a < b);
		case BinOp::Le:
			return sgn ? ComptimeValue::makeBool(static_cast<int64_t>(a) <=
			                                     static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a <= b);
		case BinOp::Gt:
			return sgn ? ComptimeValue::makeBool(static_cast<int64_t>(a) >
			                                     static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a > b);
		case BinOp::Ge:
			return sgn ? ComptimeValue::makeBool(static_cast<int64_t>(a) >=
			                                     static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a >= b);
		default:
			return ComptimeValue::makeNone();
		}
	}

	// String equality — used by format-string parsing (matching
	// field names in the args tuple by the parsed `{name}` slice).
	if (l.kind == ComptimeValue::Kind::Str &&
	    r.kind == ComptimeValue::Kind::Str) {
		bool eq = l.strVal == r.strVal;
		if (op == BinOp::Eq) return ComptimeValue::makeBool(eq);
		if (op == BinOp::Ne) return ComptimeValue::makeBool(!eq);
		return ComptimeValue::makeNone();
	}

	// Bool equality — `comp const flag = true; comp if (flag == false) ...`
	if (l.kind == ComptimeValue::Kind::Bool &&
	    r.kind == ComptimeValue::Kind::Bool) {
		if (op == BinOp::Eq)
			return ComptimeValue::makeBool(l.boolVal == r.boolVal);
		if (op == BinOp::Ne)
			return ComptimeValue::makeBool(l.boolVal != r.boolVal);
		return ComptimeValue::makeNone();
	}

	// Type equality — the linchpin of `comp if (@TypeOf(arg) == i32)`.
	if (l.kind == ComptimeValue::Kind::Type &&
	    r.kind == ComptimeValue::Kind::Type) {
		bool eq = l.typeVal == r.typeVal;
		if (op == BinOp::Eq) return ComptimeValue::makeBool(eq);
		if (op == BinOp::Ne) return ComptimeValue::makeBool(!eq);
		return ComptimeValue::makeNone();
	}

	return ComptimeValue::makeNone();
}

ComptimeValue
ComptimeEvaluator::evalMemberAccess(const AstNode &n,
                                    const ComptimeScope &scope) const {
	// Member access on a comp value. v1 supports `.length` on a comp
	// str (returns u32 byte count). Future: struct field access on a
	// comp-known aggregate.
	NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
	StringIdx memberId = static_cast<StringIdx>(n.rhs);
	const std::string &member = strings_.get(memberId);
	ComptimeValue base = eval(baseIdx, scope);
	if (base.isNone()) return base;
	if (base.kind == ComptimeValue::Kind::Str && member == "length") {
		const std::string &s = strings_.get(base.strVal);
		return ComptimeValue::makeInt(static_cast<uint64_t>(s.length()), 64,
		                              /*isSigned=*/false);
	}
	return ComptimeValue::makeNone();
}

ExecResult ComptimeEvaluator::execStmt(NodeIdx stmt, ComptimeScope &scope,
                                       uint32_t &iterCounter, uint32_t iterCap,
                                       ComptimeValue &outReturnValue,
                                       Diagnostics &diags, SrcLoc loc) const {
	if (stmt == kNoNode) return ExecResult::Continue;
	const AstNode &n = nodes_.get(stmt);

	switch (n.tag) {
	case AstTag::VarDecl: {
		// extra: [name StringIdx, type TypeIdx, init NodeIdx]
		ExtraIdx extra = static_cast<ExtraIdx>(n.lhs);
		StringIdx nameId = static_cast<StringIdx>(nodes_.getExtra(extra));
		// type slot at extra+1 — ignored at comp time; the comp value
		// carries its own width/signedness. Width-checking against the
		// declared type can come later.
		NodeIdx initIdx = static_cast<NodeIdx>(nodes_.getExtra(extra + 2));
		ComptimeValue v = eval(initIdx, scope);
		if (v.isNone()) {
			diags.error(loc, "comp var-decl initializer must be a "
			                 "compile-time-known value");
			return ExecResult::Error;
		}
		scope.bind(strings_.get(nameId), std::move(v));
		return ExecResult::Continue;
	}

	case AstTag::Assign: {
		NodeIdx targetIdx = static_cast<NodeIdx>(n.lhs);
		NodeIdx valueIdx = static_cast<NodeIdx>(n.rhs);
		const AstNode &target = nodes_.get(targetIdx);
		if (target.tag != AstTag::Variable) {
			diags.error(loc, "comp assignment target must be a bare "
			                 "variable (member-access / index targets are "
			                 "not supported in v1)");
			return ExecResult::Error;
		}
		StringIdx nameId = static_cast<StringIdx>(target.lhs);
		const std::string &name = strings_.get(nameId);
		ComptimeValue v = eval(valueIdx, scope);
		if (v.isNone()) {
			diags.error(loc, "comp assignment value must be a compile-"
			                 "time-known value");
			return ExecResult::Error;
		}
		if (!scope.set(name, std::move(v))) {
			diags.error(loc, "assignment to undeclared variable `" + name +
			                     "` (declare with `var` first)");
			return ExecResult::Error;
		}
		return ExecResult::Continue;
	}

	case AstTag::IfNode: {
		NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
		ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
		uint32_t thenCount = nodes_.getExtra(extra);
		uint32_t elseCount = nodes_.getExtra(extra + 1);
		ComptimeValue c = eval(condIdx, scope);
		if (c.kind != ComptimeValue::Kind::Bool) {
			diags.error(loc, "comp `if` condition must fold to bool");
			return ExecResult::Error;
		}
		// Run the chosen arm in a nested scope so its locals don't
		// leak to the outer block.
		ComptimeScope inner(&scope);
		if (c.boolVal) {
			std::vector<NodeIdx> stmts;
			stmts.reserve(thenCount);
			for (uint32_t i = 0; i < thenCount; i++) {
				stmts.push_back(
				    static_cast<NodeIdx>(nodes_.getExtra(extra + 2 + i)));
			}
			return execBlock(stmts.data(), stmts.size(), inner, iterCounter,
			                 iterCap, outReturnValue, diags, loc);
		}
		std::vector<NodeIdx> stmts;
		stmts.reserve(elseCount);
		for (uint32_t i = 0; i < elseCount; i++) {
			stmts.push_back(static_cast<NodeIdx>(
			    nodes_.getExtra(extra + 2 + thenCount + i)));
		}
		return execBlock(stmts.data(), stmts.size(), inner, iterCounter,
		                 iterCap, outReturnValue, diags, loc);
	}

	case AstTag::WhileNode: {
		NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
		ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
		uint32_t bodyCount = nodes_.getExtra(extra);
		std::vector<NodeIdx> body;
		body.reserve(bodyCount);
		for (uint32_t i = 0; i < bodyCount; i++) {
			body.push_back(
			    static_cast<NodeIdx>(nodes_.getExtra(extra + 1 + i)));
		}
		while (true) {
			ComptimeValue c = eval(condIdx, scope);
			if (c.kind != ComptimeValue::Kind::Bool) {
				diags.error(loc, "comp `while` condition must fold to bool");
				return ExecResult::Error;
			}
			if (!c.boolVal) break;
			if (++iterCounter > iterCap) {
				diags.error(loc, "comp evaluation iteration cap (" +
				                     std::to_string(iterCap) +
				                     ") exceeded — possible infinite loop");
				return ExecResult::IterationCap;
			}
			ComptimeScope iter(&scope);
			ExecResult r =
			    execBlock(body.data(), body.size(), iter, iterCounter, iterCap,
			              outReturnValue, diags, loc);
			if (r != ExecResult::Continue) return r;
		}
		return ExecResult::Continue;
	}

	case AstTag::Return: {
		NodeIdx valIdx = static_cast<NodeIdx>(n.lhs);
		if (valIdx == kNoNode) {
			outReturnValue = ComptimeValue::makeNone();
		} else {
			outReturnValue = eval(valIdx, scope);
			if (outReturnValue.isNone()) {
				diags.error(loc, "comp `return` expression must fold to a "
				                 "value");
				return ExecResult::Error;
			}
		}
		return ExecResult::Returned;
	}

	default:
		// Expression statement (e.g. an @-emit intrinsic call once
		// Phase 4 lands). v1 evaluator just evaluates and discards;
		// when @emit intrinsics arrive they'll be dispatched via a
		// caller-context hook before reaching this default branch.
		(void)eval(stmt, scope);
		return ExecResult::Continue;
	}
}

ExecResult ComptimeEvaluator::execBlock(const NodeIdx *stmts, std::size_t count,
                                        ComptimeScope &scope,
                                        uint32_t &iterCounter, uint32_t iterCap,
                                        ComptimeValue &outReturnValue,
                                        Diagnostics &diags, SrcLoc loc) const {
	for (std::size_t i = 0; i < count; i++) {
		ExecResult r = execStmt(stmts[i], scope, iterCounter, iterCap,
		                        outReturnValue, diags, loc);
		if (r != ExecResult::Continue) return r;
	}
	return ExecResult::Continue;
}

ComptimeValue ComptimeEvaluator::evalIndex(const AstNode &n,
                                           const ComptimeScope &scope) const {
	NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx idxIdx = static_cast<NodeIdx>(n.rhs);
	ComptimeValue base = eval(baseIdx, scope);
	if (base.isNone()) return base;
	ComptimeValue idx = eval(idxIdx, scope);
	if (idx.isNone() || idx.kind != ComptimeValue::Kind::Int) {
		return ComptimeValue::makeNone();
	}
	uint64_t i = idx.asU64();
	if (base.kind == ComptimeValue::Kind::Str) {
		const std::string &s = strings_.get(base.strVal);
		if (i >= s.length()) return ComptimeValue::makeNone();
		// Result is the byte value as a u8.
		return ComptimeValue::makeInt(
		    static_cast<uint64_t>(static_cast<uint8_t>(s[i])), 8,
		    /*isSigned=*/false);
	}
	if (base.kind == ComptimeValue::Kind::Aggregate) {
		if (i >= base.aggFields.size()) return ComptimeValue::makeNone();
		return base.aggFields[i];
	}
	return ComptimeValue::makeNone();
}

}  // namespace jam
