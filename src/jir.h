/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JIR_H
#define JIR_H

#include "ast_flat.h"
#include "param_mode.h"

#include <cstdint>
#include <string>
#include <vector>

class FunctionAST;

// JIR (Jam IR) — a single typed flat intermediate representation
// produced by AstGen from the parsed AST and consumed by codegen to
// emit LLVM IR. The IR is typed from the start: Jam doesn't have
// comptime-as-values, so a separate untyped lowering pass isn't
// needed.
//
// Pipeline (target):
//   Source → Tokens → AST → AstGen → JIR → Codegen → LLVM IR
//
// Each `JirInst` carries a typed result (or `kNoType` for control-flow
// instructions) plus a source line for diagnostics. Variable-width
// payloads (call arg lists, struct field values, switch cases, etc.)
// live in a per-function `extra` pool, just like the AST's NodeStore.

// Ref into a function's instruction list. Index `0` is reserved as a
// sentinel "no value" / invalid ref so a zero-initialized field reads
// as null.
using JirRef = uint32_t;
constexpr JirRef kNoJirRef = 0;

// Ref into a function's block list. Same shape as `JirRef`.
using JirBlockRef = uint32_t;
constexpr JirBlockRef kNoJirBlock = 0;

// Ref into a function's extra-data pool.
using JirExtraIdx = uint32_t;

enum class JirTag : uint8_t {
	Invalid = 0,

	// Constants
	// Int: `a` = lo32, `b` = hi32 of u64 value; `ty` = int type.
	// Float: `a` = lo32, `b` = hi32 of f32/f64 bit pattern; `ty` = f32/f64.
	// Bool: `a` = 0 or 1; `ty` = i1.
	// Str: `a` = StringIdx; `ty` = []u8.
	Int,
	Float,
	Bool,
	Str,

	// Storage
	// Alloca: `ty` = pointee type. Result type is implicitly ptr-to-`ty`.
	// Load:   `a` = ptr ref; `ty` = loaded type.
	// Store:  `a` = ptr ref, `b` = value ref. No result.
	Alloca,
	Load,
	Store,

	// Integer arithmetic
	// Binary form: `a` = lhs ref, `b` = rhs ref, `ty` = result type.
	Add,
	Sub,
	Mul,
	SDiv,
	UDiv,
	SRem,
	URem,

	// Float arithmetic
	FAdd,
	FSub,
	FMul,
	FDiv,
	FRem,
	FNeg,  // unary: `a` = operand

	// Integer comparison
	// Binary form; `ty` is always i1.
	ICmpEq,
	ICmpNe,
	ICmpSlt,
	ICmpSle,
	ICmpSgt,
	ICmpSge,
	ICmpUlt,
	ICmpUle,
	ICmpUgt,
	ICmpUge,

	// Float comparison
	// Ordered predicates (NaN inputs ⇒ false) to match Jam's strict
	// float typing policy. `ty` is always i1.
	FCmpOeq,
	FCmpOne,
	FCmpOlt,
	FCmpOle,
	FCmpOgt,
	FCmpOge,

	// Bitwise / shift
	BitAnd,
	BitOr,
	BitXor,
	BitNot,
	Shl,
	AShr,
	LShr,

	// Logical (short-circuit handled by control flow)
	LogNot,  // unary

	// Type conversions
	// All take `a` = operand ref, `ty` = destination type.
	ZExt,
	SExt,
	Trunc,
	SIToFP,
	UIToFP,
	FPToSI,
	FPToUI,
	FPExt,
	FPTrunc,
	BitCast,

	// Control flow
	// Br:          `a` = JirBlockRef target. No result.
	// CondBr:      `a` = cond ref; `b` = ExtraIdx → [thenBlock, elseBlock].
	// Switch:      `a` = scrut ref; `b` = ExtraIdx →
	//                  [defaultBlock, caseCount,
	//                   case0_lo, case0_hi, case0_signed, case0_block,
	//                   case1_..., ...]
	//              Canonical multi-way branch. astgen emits Switch
	//              when every arm pattern is a single integer or
	//              unit-enum-variant; jir_codegen specialises down
	//              to icmp+cond_br for trivial shapes and otherwise
	//              emits LLVM `switch`.
	// Ret:         `a` = value ref (or kNoJirRef for void).
	// Unreachable: no operands.
	Br,
	CondBr,
	Switch,
	Ret,
	Unreachable,

	// Function call
	// Call: `a` = StringIdx (callee qualified name);
	//       `b` = ExtraIdx → [argCount, arg0, arg1, ...]
	//       `ty` = return type (kNoType for void).
	Call,

	// Function parameter access
	// Param: `a` = parameter index; `ty` = param type.
	Param,

	// Aggregates
	// StructLit:    `b` = ExtraIdx → [fieldCount, field0_val, field1_val, ...];
	//               `ty` = struct type.
	// FieldAccess:  `a` = base ref, `b` = field index; `ty` = field type.
	// ExtractValue: `a` = aggregate ref, `b` = field index; `ty` = field type.
	// ArrayLit:     `b` = ExtraIdx → [count, elem0, elem1, ...];
	//               `ty` = array type.
	// Index:        `a` = base ref, `b` = index ref; `ty` = element type.
	StructLit,
	FieldAccess,
	ExtractValue,
	ArrayLit,
	Index,
	// FieldAddr: `a` = base pointer ref; `b` = field index;
	//            `ty` = pointer-to-field-type. Lowers to a StructGEP.
	//            Used by lvalue assignments to struct fields.
	FieldAddr,
	// IndexAddr: `a` = base pointer ref; `b` = index value ref;
	//            `ty` = pointer-to-element-type. Lowers to an
	//            ArrayGEP / PtrGEP / slice.ptr+GEP depending on the
	//            base's source-level type kind. For lvalue writes to
	//            array/slice/ptr-many elements.
	IndexAddr,

	// Address-of / dereference
	AddrOf,  // `a` = lvalue ref
	Deref,   // `a` = ptr ref

	// Pattern binding payload extraction
	// For enum-variant pattern bindings, the codegen needs to load
	// the bound payload field. Encoded explicitly in JIR.
	EnumPayload,  // `a` = enum value ref, `b` = field index

	// Drop
	// Explicit destructor call for a tracked binding. Emitted by
	// AstGen at scope-exit points (return / break / continue /
	// fall-through). `a` = the binding's alloca JirRef; `b` =
	// StringIdx of the LLVM symbol to call (e.g. `__drop_T` for
	// top-level drop fns, `T.drop` for instantiated methods).
	// Codegen unconditionally lowers to `call void <symbol>(ptr)`
	// — no metadata-driven emission, no scope-tracking outside
	// AstGen. ty is kNoType (void).
	DropBinding,

	// Error recovery
	// Poison value: AstGen pushed a diagnostic for the expression
	// at this position but synthesized a typed placeholder so the
	// rest of the function can keep being analyzed. Lowers to LLVM
	// `undef` at codegen time. Programs that produce any Poison
	// instructions must not reach codegen — the driver checks
	// `Diagnostics::hasErrors()` and bails first.
	// `ty` = the type the consumer expected (kNoType is legal but
	// degrades downstream type checking).
	Poison,
};

// A single JIR instruction.
//
// Layout chosen to be compact (16 bytes on 64-bit) and trivially
// copyable so dense instruction arrays are cache-friendly.
struct JirInst {
	JirTag tag = JirTag::Invalid;
	uint8_t _pad = 0;      // padding to keep `flags` 2-byte aligned
	uint16_t flags = 0;    // tag-specific flags
	uint32_t srcLine = 0;  // for diagnostics: file:line:
	JirRef a = kNoJirRef;
	JirRef b = kNoJirRef;
	TypeIdx ty = kNoType;  // result type (kNoType for void / control)
};

// A basic block within a JirFunction. Holds a list of JirRef
// instruction indices into the function's `insts` array.
struct JirBlock {
	std::string name;  // diagnostic label
	std::vector<JirRef> insts;
};

// A single function lowered to JIR. Self-contained: its `insts`
// array holds every instruction in the function body (across all
// blocks), and the blocks reference instructions by index. Variable-
// width payloads (call args, struct fields, switch cases) live in
// `extra` so JirInst stays fixed-width.
class JirFunction {
  public:
	std::string name;
	std::vector<JirInst> insts;
	std::vector<uint32_t> extra;
	std::vector<JirBlock> blocks;
	std::vector<TypeIdx> paramTypes;
	// Parallel to `paramTypes`: the source-level parameter mode (Let
	// / Mut / Move). The backend (jir_codegen + any future non-LLVM
	// target) decides the ABI from this — today's LLVM backend
	// lowers Mut and Move as by-pointer; a future C backend or
	// native register-allocator could choose differently for small
	// primitive types. JIR shouldn't bake the by-pointer decision
	// in; carrying intent keeps the door open.
	std::vector<ParamMode> paramModes;
	TypeIdx returnType = kNoType;
	bool isExtern = false;
	bool isExport = false;
	bool isPub = false;
	bool isTest = false;
	bool isVarArgs = false;

	JirFunction() {
		// Slot 0 is the sentinel "no instruction" so kNoJirRef = 0
		// reads as invalid.
		insts.push_back(JirInst{});
		blocks.push_back(JirBlock{"<sentinel>", {}});
	}

	// Append an instruction; returns its ref (1-based, 0 reserved).
	JirRef pushInst(const JirInst &inst) {
		JirRef r = static_cast<JirRef>(insts.size());
		insts.push_back(inst);
		return r;
	}

	// Append a basic block; returns its ref.
	JirBlockRef pushBlock(const std::string &label) {
		JirBlockRef r = static_cast<JirBlockRef>(blocks.size());
		blocks.push_back(JirBlock{label, {}});
		return r;
	}

	// Append `len` u32s to the extra pool; returns the start index.
	JirExtraIdx pushExtra(const uint32_t *data, std::size_t len) {
		JirExtraIdx start = static_cast<JirExtraIdx>(extra.size());
		extra.insert(extra.end(), data, data + len);
		return start;
	}

	JirExtraIdx reserveExtra(std::size_t len) {
		JirExtraIdx start = static_cast<JirExtraIdx>(extra.size());
		extra.resize(extra.size() + len, 0);
		return start;
	}

	const JirInst &getInst(JirRef r) const { return insts[r]; }
	JirInst &getInstMut(JirRef r) { return insts[r]; }
	const JirBlock &getBlock(JirBlockRef r) const { return blocks[r]; }
	JirBlock &getBlockMut(JirBlockRef r) { return blocks[r]; }
	uint32_t getExtra(JirExtraIdx i) const { return extra[i]; }
	void setExtra(JirExtraIdx i, uint32_t v) { extra[i] = v; }
};

// A whole-program JIR module: a collection of JirFunctions, plus
// references back to the shared frontend pools.
class JirModule {
  public:
	std::vector<JirFunction> functions;
};

#endif  // JIR_H
