/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

// Flat, tag-dispatched AST. AstNode is a 16-byte plain struct, indices
// replace pointers, complex payloads spill into a side `extra` pool.
// The flat layout keeps nodes packed in a single contiguous vector so
// codegen walks a cache-friendly array via switch dispatch on tag.
//
// Conventions:
//   - NodeIdx 0 is the "null"/absent node. The store reserves slot 0 with
//     tag = AstTag::Invalid so a 0-init AstNode is a sentinel.
//   - For nodes that need more than two u32 payload slots, lhs/rhs index
//     into NodeStore.extra; layout per tag is documented below.
//   - StringPool / TypePool ids are 32-bit. 0 = empty string / void type.

#ifndef AST_FLAT_H
#define AST_FLAT_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using NodeIdx = uint32_t;
using StringIdx = uint32_t;
using TypeIdx = uint32_t;
using TokenIndex = uint32_t;
using ExtraIdx = uint32_t;

constexpr NodeIdx kNoNode = 0;
constexpr StringIdx kNoString = 0;
constexpr TypeIdx kNoType = 0;

// Node tags. One byte; switched on in codegen.

enum class AstTag : uint8_t {
	Invalid = 0,

	// Literals
	NumberLit,  // d.lhs = lo32(val), d.rhs = hi32(val); flags bit 0 = isNeg
	BoolLit,    // d.lhs = 0|1
	StringLit,  // d.lhs = StringIdx

	// Lvalues / refs
	Variable,      // d.lhs = StringIdx (name)
	MemberAccess,  // d.lhs = NodeIdx (object), d.rhs = StringIdx (member)
	Index,         // d.lhs = NodeIdx (object), d.rhs = NodeIdx (index)
	Deref,         // d.lhs = NodeIdx (operand)
	AddressOf,     // d.lhs = NodeIdx (operand)

	// Operators
	UnaryOp,   // d.lhs = NodeIdx (operand); op kind in `op`
	BinaryOp,  // d.lhs = NodeIdx (lhs), d.rhs = NodeIdx (rhs); op in `op`

	// Calls
	// d.lhs = StringIdx (callee fully qualified, e.g. "std.fmt.println")
	// d.rhs = ExtraIdx → [argCount, arg0, arg1, ...]
	Call,

	// Statements
	Return,  // d.lhs = NodeIdx (operand) or kNoNode for bare `return;`
	Assign,  // d.lhs = NodeIdx (target), d.rhs = NodeIdx (value)
	// d.lhs = ExtraIdx → [StringIdx name, TypeIdx type, NodeIdx init]
	// d.rhs = flags (bit 0 = isConst)
	VarDecl,
	// d.lhs = NodeIdx (cond), d.rhs = ExtraIdx →
	//   [thenCount, elseCount, then0, then1, ..., else0, else1, ...]
	IfNode,
	// d.lhs = NodeIdx (cond), d.rhs = ExtraIdx → [bodyCount, body0, body1, ...]
	WhileNode,
	// d.lhs = ExtraIdx → [StringIdx var, NodeIdx start, NodeIdx end,
	//                     bodyCount, body0, body1, ...]
	ForNode,
	Break,
	Continue,

	// Module-level
	ImportLit,  // d.lhs = StringIdx (module path)
	// d.lhs = TypeIdx (struct type, kNoType if inferred from var-decl context)
	// d.rhs = ExtraIdx → [fieldCount, fieldName0, fieldExpr0, fieldName1, ...]
	StructLit,
	// `[a, b, c, ...]` array literal in expression position. Element type
	// inferred from var-decl target type (kNoType if unknown — codegen
	// rejects).
	// d.lhs = TypeIdx (target element type, kNoType if unbound)
	// d.rhs = ExtraIdx → [count, elem0, elem1, ...]
	ArrayLit,
	// `[expr; N]` array repeat literal. N is a constant integer expression.
	// d.lhs = TypeIdx (target array type, kNoType if unbound)
	// d.rhs = ExtraIdx → [valueNode, countNode]
	ArrayRepeat,
	// a `struct { fields, methods }` expression, evaluated at
	// compile time to a value of type `type`. The expression's body lives
	// in ModuleAST::AnonStructs as a regular StructDeclAST with a
	// synthetic name; this node carries the index so the substitution
	// engine in can find it.
	// d.lhs = u32 (index into ModuleAST::AnonStructs)
	StructExpr,

	// Generic enum expression: `enum { Variant1, Variant2(T), ... }` in
	// expression position. Mirror of StructExpr for tagged unions. The
	// expression's body lives in ModuleAST::AnonEnums as a regular
	// EnumDeclAST with a synthetic name; this node carries the index
	// so the substitution engine can instantiate variant payload types
	// against the generic args. Enables `fn Option(T: type) type {
	// return enum { Some(T), None }; }` and friends.
	// d.lhs = u32 (index into ModuleAST::AnonEnums)
	EnumExpr,

	// Pattern match (integer literals, ranges, or-patterns, wildcard).
	// The catch-all is the wildcard pattern `_`; there is no `else` arm.
	// d.lhs = NodeIdx (scrutinee expression)
	// d.rhs = ExtraIdx → [armCount,
	//                     arm0_patIdx, arm0_bodyCount, arm0_body...,
	//                     arm1_patIdx, arm1_bodyCount, arm1_body..., ...]
	MatchNode,

	// Explicit type cast `expr as T`. d.lhs = NodeIdx (operand),
	// d.rhs = TypeIdx (target type). Lowers to integer cast, float
	// cast, integer↔float conversion, or — for enum-to-integer —
	// tag extraction.
	AsCast,

	// Comptime intrinsic call: `@name(T)`. Resolved to a constant at
	// codegen time; LLVM never sees a call instruction. Stage 1 only
	// supports single-TYPE-arg intrinsics (sizeOf, alignOf); generalize
	// to multi-arg shapes when user-defined cfn + CTFE lands.
	// d.lhs = StringIdx (intrinsic name, e.g. "sizeOf")
	// d.rhs = TypeIdx (the type argument)
	AtCall,

	// Static method call on a generic-call type receiver:
	//   Vec(i32).empty()          → struct static method
	//   Option(i32).Some(42)      → enum variant constructor
	//   Option(i32).None()        → unit-variant constructor
	// The parser detects the `IDENT(args).IDENT(args)` shape via
	// paren-balanced peek-ahead, parses the inner args as TYPES, and
	// builds a GenericCall TypeIdx for the receiver. Codegen resolves
	// the receiver type (instantiating if necessary) and dispatches
	// to the appropriate static-method or variant-constructor path.
	// d.lhs = TypeIdx (receiver type — typically GenericCall)
	// d.rhs = ExtraIdx → [methodNameId, argCount, arg0, arg1, ...]
	TypeMethodCall,

	// Pattern atoms — internal nodes used inside MatchNode arms. Never
	// reachable from regular expression / statement positions.

	// d.lhs = lo32 of value, d.rhs = hi32 of value, flags bit 0 = isNeg.
	PatLit,
	// d.lhs = lo32 of low bound, d.rhs = lo32 of high bound (limited to
	// values that fit in u32; may later extend via ExtraIdx for u64).
	PatRange,
	// No payload.
	PatWildcard,
	// d.lhs = ExtraIdx → [count, sub0, sub1, ...]; each subN is a NodeIdx
	// of a PatLit / PatRange / PatWildcard / PatEnumVariant.
	PatOr,
	// Enum-variant pattern (`EnumName.VariantName` in a match arm).
	// d.lhs = StringIdx (enum name), d.rhs = StringIdx (variant name).
	// Resolved to the variant's discriminant at codegen time.
	PatEnumVariant,

	// Sentinel for table sizing
	Count,
};

// Binary operator kinds (one slot in AstNode.op for BinaryOp tag).
enum class BinOp : uint8_t {
	Invalid = 0,
	Add,
	Sub,
	Mul,
	Div,
	Mod,
	BitAnd,
	BitOr,
	BitXor,
	Shl,
	Shr,
	LogAnd,
	LogOr,
	Eq,
	Ne,
	Lt,
	Le,
	Gt,
	Ge,
};

// Unary operator kinds (one slot in AstNode.op for UnaryOp tag).
enum class UnaryOp : uint8_t {
	Invalid = 0,
	Neg,     // -x
	LogNot,  // !x
	BitNot,  // ~x
};

// 16 bytes total: tag + op + flags + main_token + two u32 data slots.
struct AstNode {
	AstTag tag;
	uint8_t op;            // BinOp / UnaryOp encoded as u8
	uint16_t flags;        // tag-specific bit field (e.g. NumberLit isNeg)
	TokenIndex mainToken;  // token index for error reporting
	uint32_t lhs;          // generic data slot
	uint32_t rhs;          // generic data slot
};

static_assert(sizeof(AstNode) == 16,
              "AstNode is sized to one cache-friendly slot");

// Storage. NodeStore owns the flat node array + the `extra` u32 pool used
// for variadic payloads (call args, block bodies, struct literal fields).

class NodeStore {
	std::vector<AstNode> nodes_;
	std::vector<uint32_t> extra_;
	// Parallel array recording the source line for each node, indexed
	// the same as `nodes_`. Populated by the parser at emit-time from
	// the lexer's token line. Consulted by the codegen error formatter
	// so user-visible diagnostics carry `file:line:` prefixes.
	std::vector<int> lines_;

  public:
	NodeStore() {
		// Reserve slot 0 as the null sentinel so kNoNode is a no-op.
		nodes_.push_back(AstNode{AstTag::Invalid, 0, 0, 0, 0, 0});
		lines_.push_back(0);
	}

	NodeIdx addNode(AstNode n) {
		NodeIdx id = static_cast<NodeIdx>(nodes_.size());
		nodes_.push_back(n);
		lines_.push_back(0);
		return id;
	}

	// Variant used by the parser when it knows the source line.
	NodeIdx addNodeAt(AstNode n, int line) {
		NodeIdx id = static_cast<NodeIdx>(nodes_.size());
		nodes_.push_back(n);
		lines_.push_back(line);
		return id;
	}

	int getLine(NodeIdx id) const {
		return id < lines_.size() ? lines_[id] : 0;
	}

	const AstNode &get(NodeIdx id) const { return nodes_[id]; }
	AstNode &getMut(NodeIdx id) { return nodes_[id]; }

	std::size_t size() const { return nodes_.size(); }

	// Append a single u32 to the extra pool, return its index.
	ExtraIdx pushExtra(uint32_t v) {
		ExtraIdx i = static_cast<ExtraIdx>(extra_.size());
		extra_.push_back(v);
		return i;
	}

	// Append `len` u32s starting at `data`, return the index of the first.
	ExtraIdx pushExtraSpan(const uint32_t *data, std::size_t len) {
		ExtraIdx start = static_cast<ExtraIdx>(extra_.size());
		extra_.insert(extra_.end(), data, data + len);
		return start;
	}

	// Reserve `len` u32s; caller fills via setExtra(idx + i, ...).
	ExtraIdx reserveExtra(std::size_t len) {
		ExtraIdx start = static_cast<ExtraIdx>(extra_.size());
		extra_.resize(extra_.size() + len, 0);
		return start;
	}

	uint32_t getExtra(ExtraIdx i) const { return extra_[i]; }
	void setExtra(ExtraIdx i, uint32_t v) { extra_[i] = v; }

	const std::vector<uint32_t> &extra() const { return extra_; }
};

// Interned identifiers + string literals. deque<string> keeps references
// stable across pushes so the string_view keys in `idx_` don't dangle.

class StringPool {
	std::deque<std::string> strings_;
	std::unordered_map<std::string_view, uint32_t> idx_;

  public:
	StringPool() {
		// Slot 0 reserved for kNoString (the empty string).
		strings_.emplace_back();
		idx_.emplace(strings_.back(), 0);
	}

	StringIdx intern(std::string_view s) {
		auto it = idx_.find(s);
		if (it != idx_.end()) return it->second;
		StringIdx id = static_cast<StringIdx>(strings_.size());
		strings_.emplace_back(s);
		idx_.emplace(strings_.back(), id);
		return id;
	}

	const std::string &get(StringIdx id) const { return strings_[id]; }
	std::size_t size() const { return strings_.size(); }
};

// Type interning. Types are first-class structured values, never strings.
// Built-ins (void, bool, u8..u64, i8..i64, f32, f64, str) live at fixed
// pre-interned indices so callers can refer to them without a lookup.

enum class TypeKind : uint8_t {
	Invalid = 0,
	Void,
	NoReturn,
	Bool,
	Int,        // intT.bits, intT.isSigned
	Float,      // floatT.bits
	PtrSingle,  // *T   — ptrT.elem (no indexing on this kind)
	PtrMany,    // [*]T — ptrT.elem (indexable)
	Slice,      // []T  — sliceT.elem  (lowered to {ptr, len} struct)
	Array,      // [N]T — arrayT.elem, arrayT.len
	Struct,     // structT.name (StringIdx)
	Enum,       // enumT.name (StringIdx); see EnumDeclAST for variants
	Union,      // unionT.name (StringIdx); see UnionDeclAST for fields
	// Parser-emitted "user-named type, kind resolution deferred." The
	// parser sees `MyType` in a position where it must produce a
	// TypeIdx but doesn't yet know whether `MyType` is a struct, union,
	// or enum. The codegen resolves Named values by consulting the
	// three registries, in that order.
	Named,  // namedT.name (StringIdx)
	// the type of types — a value of this kind is itself
	// a TypeIdx, used at compile time only. A function parameter
	// declared `T: type` accepts a type argument when called. Has no
	// runtime ABI; the codegen never lowers it to an LLVM type.
	Type,
	// deferred generic instantiation, parsed from
	// `Identifier(arg, ...)` in a type position. The TypeKey carries
	// the callee name in `a` (StringIdx) and an index into
	// TypePool::genericArgs in `b`. The codegen resolves this lazily
	// via the substitution engine when an LLVM type is requested or
	// when a binding's static TypeIdx is needed.
	GenericCall,
};

struct TypeKey {
	TypeKind kind;
	uint8_t pad0;
	uint16_t pad1;
	uint32_t a;
	uint32_t b;

	// Layout per kind (a/b serve as named slots — accessed via the
	// helpers below to keep the discriminated semantics explicit).
	//   Int:    a = bits, b = isSigned (0/1)
	//   Float:  a = bits (32 or 64)
	//   PtrSingle / PtrMany: a = elem TypeIdx
	//   Slice:  a = elem TypeIdx
	//   Array:  a = elem TypeIdx, b = length
	//   Struct: a = StringIdx (struct name)
};

inline bool operator==(const TypeKey &x, const TypeKey &y) {
	if (x.kind != y.kind) return false;
	switch (x.kind) {
	case TypeKind::Invalid:
	case TypeKind::Void:
	case TypeKind::NoReturn:
	case TypeKind::Bool:
		return true;
	case TypeKind::Int:
		return x.a == y.a && x.b == y.b;
	case TypeKind::Float:
		return x.a == y.a;
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
	case TypeKind::Slice:
		return x.a == y.a;
	case TypeKind::Array:
		return x.a == y.a && x.b == y.b;
	case TypeKind::Struct:
	case TypeKind::Enum:
	case TypeKind::Union:
	case TypeKind::Named:
		return x.a == y.a;
	case TypeKind::Type:
		// Singleton meta-type: every TypeKey of kind Type is equal.
		return true;
	case TypeKind::GenericCall:
		// equal iff callee name AND args-list index match. The
		// args index is canonical because the side table interns
		// args lists.
		return x.a == y.a && x.b == y.b;
	}
	return false;
}

struct TypeKeyHash {
	std::size_t operator()(const TypeKey &k) const noexcept {
		// Simple FNV-style mix; uniqueness only needs to be per-key.
		std::size_t h = static_cast<std::size_t>(k.kind) * 1099511628211ULL;
		h ^= static_cast<std::size_t>(k.a);
		h *= 1099511628211ULL;
		h ^= static_cast<std::size_t>(k.b);
		return h;
	}
};

// Built-in type indices. The TypePool constructor pre-interns these so
// callers can refer to them without going through intern().
namespace BuiltinType {
constexpr TypeIdx Void = 1;
constexpr TypeIdx Bool = 2;
constexpr TypeIdx U8 = 3;
constexpr TypeIdx I8 = 4;
constexpr TypeIdx U16 = 5;
constexpr TypeIdx I16 = 6;
constexpr TypeIdx U32 = 7;
constexpr TypeIdx I32 = 8;
constexpr TypeIdx U64 = 9;
constexpr TypeIdx I64 = 10;
constexpr TypeIdx F32 = 11;
constexpr TypeIdx F64 = 12;
constexpr TypeIdx Type = 13;
constexpr TypeIdx NoReturn = 14;
constexpr TypeIdx U1 = Bool;
}  // namespace BuiltinType

class TypePool {
	std::vector<TypeKey> keys_;
	std::unordered_map<TypeKey, TypeIdx, TypeKeyHash> idx_;
	// side table for generic-call argument lists. A
	// `TypeKind::GenericCall` TypeKey stores the index into this vector
	// in `b`. Two `Maybe(File)` references at distinct sites resolve to
	// the same TypeIdx because the args list `[File]` is interned here
	// once.
	std::vector<std::vector<TypeIdx>> genericArgs_;
	std::map<std::vector<TypeIdx>, uint32_t> genericArgsIdx_;

	TypeIdx pushKey(TypeKey k) {
		TypeIdx id = static_cast<TypeIdx>(keys_.size());
		keys_.push_back(k);
		idx_.emplace(k, id);
		return id;
	}

  public:
	TypePool() {
		// Slot 0 = invalid sentinel.
		keys_.push_back(TypeKey{TypeKind::Invalid, 0, 0, 0, 0});
		// Pre-intern built-ins. Order matches the constants above so the
		// TypeIdx values are stable.
		pushKey(TypeKey{TypeKind::Void, 0, 0, 0, 0});
		pushKey(TypeKey{TypeKind::Bool, 0, 0, 0, 0});
		pushKey(TypeKey{TypeKind::Int, 0, 0, 8, 0});   // u8
		pushKey(TypeKey{TypeKind::Int, 0, 0, 8, 1});   // i8
		pushKey(TypeKey{TypeKind::Int, 0, 0, 16, 0});  // u16
		pushKey(TypeKey{TypeKind::Int, 0, 0, 16, 1});  // i16
		pushKey(TypeKey{TypeKind::Int, 0, 0, 32, 0});  // u32
		pushKey(TypeKey{TypeKind::Int, 0, 0, 32, 1});  // i32
		pushKey(TypeKey{TypeKind::Int, 0, 0, 64, 0});  // u64
		pushKey(TypeKey{TypeKind::Int, 0, 0, 64, 1});  // i64
		pushKey(TypeKey{TypeKind::Float, 0, 0, 32, 0});
		pushKey(TypeKey{TypeKind::Float, 0, 0, 64, 0});
		pushKey(TypeKey{TypeKind::Type, 0, 0, 0, 0});
		pushKey(TypeKey{TypeKind::NoReturn, 0, 0, 0, 0});
	}

	TypeIdx intern(TypeKey k) {
		auto it = idx_.find(k);
		if (it != idx_.end()) return it->second;
		return pushKey(k);
	}

	const TypeKey &get(TypeIdx i) const { return keys_[i]; }
	std::size_t size() const { return keys_.size(); }

	// Convenience constructors that intern in one call.
	TypeIdx internInt(uint16_t bits, bool isSigned) {
		return intern(TypeKey{TypeKind::Int, 0, 0, bits, isSigned ? 1u : 0u});
	}
	TypeIdx internFloat(uint16_t bits) {
		return intern(TypeKey{TypeKind::Float, 0, 0, bits, 0});
	}
	TypeIdx internPtrSingle(TypeIdx elem) {
		return intern(TypeKey{TypeKind::PtrSingle, 0, 0, elem, 0});
	}
	TypeIdx internPtrMany(TypeIdx elem) {
		return intern(TypeKey{TypeKind::PtrMany, 0, 0, elem, 0});
	}
	TypeIdx internSlice(TypeIdx elem) {
		return intern(TypeKey{TypeKind::Slice, 0, 0, elem, 0});
	}
	TypeIdx internArray(TypeIdx elem, uint32_t len) {
		return intern(TypeKey{TypeKind::Array, 0, 0, elem, len});
	}
	TypeIdx internStruct(StringIdx nameId) {
		return intern(TypeKey{TypeKind::Struct, 0, 0, nameId, 0});
	}

	// intern a `Identifier(arg, ...)` generic call as a
	// TypeIdx. The args list itself is interned in the side table so two
	// identical calls share an args index and consequently the same
	// TypeIdx.
	TypeIdx internGenericCall(StringIdx nameId, std::vector<TypeIdx> args) {
		uint32_t argsIdx;
		auto it = genericArgsIdx_.find(args);
		if (it != genericArgsIdx_.end()) {
			argsIdx = it->second;
		} else {
			argsIdx = static_cast<uint32_t>(genericArgs_.size());
			genericArgs_.push_back(args);
			genericArgsIdx_.emplace(std::move(args), argsIdx);
		}
		return intern(TypeKey{TypeKind::GenericCall, 0, 0, nameId, argsIdx});
	}

	const std::vector<TypeIdx> &genericArgsAt(uint32_t idx) const {
		return genericArgs_[idx];
	}
	TypeIdx internEnum(StringIdx nameId) {
		return intern(TypeKey{TypeKind::Enum, 0, 0, nameId, 0});
	}
	TypeIdx internUnion(StringIdx nameId) {
		return intern(TypeKey{TypeKind::Union, 0, 0, nameId, 0});
	}
	TypeIdx internNamed(StringIdx nameId) {
		return intern(TypeKey{TypeKind::Named, 0, 0, nameId, 0});
	}
};

#endif  // AST_FLAT_H
