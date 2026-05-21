/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "comptime.h"

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

ComptimeValue
ComptimeValue::makeAggregate(std::vector<ComptimeValue> fields) {
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
	default:
		// Operator / construct we don't fold yet. Returning None keeps
		// optional-fold callers (peephole constant folding) silent;
		// `evalRequired` will turn it into a diagnostic.
		return ComptimeValue::makeNone();
	}
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
	uint64_t bits = static_cast<uint64_t>(n.lhs) |
	                (static_cast<uint64_t>(n.rhs) << 32);
	bool isNeg = (n.flags & 1) != 0;
	bool isFloat = (n.flags & 2) != 0;
	if (isFloat) {
		double v;
		__builtin_memcpy(&v, &bits, sizeof(v));
		if (isNeg) v = -v;
		return ComptimeValue::makeFloat(v, 64);
	}
	// Default integer width: u64 (or i64 if negative). Callers can
	// narrow via the surrounding type context, but at this evaluator
	// layer we keep the literal at full width to preserve precision
	// during folding.
	if (isNeg) {
		uint64_t magnitude = bits;
		uint64_t signedBits = static_cast<uint64_t>(
		    -static_cast<int64_t>(magnitude));
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

ComptimeValue
ComptimeEvaluator::evalUnaryOp(const AstNode &n,
                                const ComptimeScope &scope) const {
	NodeIdx operand = static_cast<NodeIdx>(n.lhs);
	ComptimeValue v = eval(operand, scope);
	if (v.isNone()) return v;
	UnaryOp op = static_cast<UnaryOp>(n.op);
	switch (op) {
	case UnaryOp::Neg:
		if (v.kind == ComptimeValue::Kind::Int) {
			uint64_t neg = static_cast<uint64_t>(-static_cast<int64_t>(v.asU64()));
			return ComptimeValue::makeInt(neg, v.intVal.width, true);
		}
		if (v.kind == ComptimeValue::Kind::Float) {
			return ComptimeValue::makeFloat(-v.floatVal.value, v.floatVal.width);
		}
		return ComptimeValue::makeNone();
	case UnaryOp::LogNot:
		if (v.kind != ComptimeValue::Kind::Bool) return ComptimeValue::makeNone();
		return ComptimeValue::makeBool(!v.boolVal);
	case UnaryOp::BitNot:
		if (v.kind != ComptimeValue::Kind::Int) return ComptimeValue::makeNone();
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
		if (l.kind != ComptimeValue::Kind::Bool) return ComptimeValue::makeNone();
		bool lb = l.boolVal;
		if (op == BinOp::LogAnd && !lb) return ComptimeValue::makeBool(false);
		if (op == BinOp::LogOr && lb) return ComptimeValue::makeBool(true);
		ComptimeValue r = eval(rhsIdx, scope);
		if (r.kind != ComptimeValue::Kind::Bool) return ComptimeValue::makeNone();
		return ComptimeValue::makeBool(r.boolVal);
	}

	ComptimeValue l = eval(lhsIdx, scope);
	if (l.isNone()) return l;
	ComptimeValue r = eval(rhsIdx, scope);
	if (r.isNone()) return r;

	// Integer arithmetic + bitwise — both operands must be Int, and
	// for now we require matching width/signedness. Mixed-width is a
	// codegen-level concern; the comp evaluator stays strict.
	if (l.kind == ComptimeValue::Kind::Int &&
	    r.kind == ComptimeValue::Kind::Int) {
		if (l.intVal.width != r.intVal.width ||
		    l.intVal.isSigned != r.intVal.isSigned) {
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
				    static_cast<uint64_t>(static_cast<int64_t>(a) >> b), w, sgn);
			}
			return ComptimeValue::makeInt(a >> b, w, sgn);
		case BinOp::Eq:
			return ComptimeValue::makeBool(a == b);
		case BinOp::Ne:
			return ComptimeValue::makeBool(a != b);
		case BinOp::Lt:
			return sgn
			           ? ComptimeValue::makeBool(static_cast<int64_t>(a) <
			                                       static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a < b);
		case BinOp::Le:
			return sgn
			           ? ComptimeValue::makeBool(static_cast<int64_t>(a) <=
			                                       static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a <= b);
		case BinOp::Gt:
			return sgn
			           ? ComptimeValue::makeBool(static_cast<int64_t>(a) >
			                                       static_cast<int64_t>(b))
			           : ComptimeValue::makeBool(a > b);
		case BinOp::Ge:
			return sgn
			           ? ComptimeValue::makeBool(static_cast<int64_t>(a) >=
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
ComptimeEvaluator::evalIndex(const AstNode &n,
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
		return ComptimeValue::makeInt(static_cast<uint64_t>(
		                                  static_cast<uint8_t>(s[i])),
		                               8, /*isSigned=*/false);
	}
	if (base.kind == ComptimeValue::Kind::Aggregate) {
		if (i >= base.aggFields.size()) return ComptimeValue::makeNone();
		return base.aggFields[i];
	}
	return ComptimeValue::makeNone();
}

}  // namespace jam
