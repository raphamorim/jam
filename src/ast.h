/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef AST_H
#define AST_H

#include "ast_flat.h"
#include "jam_llvm.h"
#include "param_mode.h"
#include <memory>
#include <string>
#include <vector>

class JamCodegenContext;

// Parameter passing mode under MVS (mutable value semantics).
//
// Default (Let)      — read-only borrow; param is initialized at entry,
//                      cannot be reassigned, caller's binding unchanged.
// Mut                — exclusive read-write borrow; param is initialized
//                      at entry, may be read and written, caller's binding
//                      stays initialized after the call.
// Move               — consume ownership; caller's binding becomes
//                      uninitialized after the call.

// One function parameter. Mode defaults to Let when not annotated at the
// declaration site (the common case for read-only parameters).
struct Param {
	std::string Name;
	TypeIdx Type;
	ParamMode Mode = ParamMode::Let;
	// Declared with the source-level `comp` keyword (e.g. `fn f(comp n:
	// u32) ...`). A comp param's value must be known at the call site
	// and is bound into the function's substitution map at instantiation
	// time, *not* lowered as an LLVM parameter. Each unique comp-arg
	// combination produces a fresh monomorphisation via the same machine-
	// ry that handles `T: type` generics.
	bool isComp = false;
};

// Function declaration. The body is a sequence of flat-AST node indices
// owned by the shared NodeStore on JamCodegenContext.
class FunctionAST {
  public:
	std::string Name;
	std::vector<Param> Args;
	TypeIdx ReturnType;  // kNoType if void / unspecified
	std::vector<NodeIdx> Body;
	bool isExtern;
	bool isExport;
	bool isPub;
	bool isTest;
	bool isVarArgs;
	// Declared with `cfn` instead of `fn`. The keyword has two
	// meanings depending on declaration position:
	//   * Inside a struct body — opts the method into the compiler-
	//     synthesized-call set (drop / at / default). A regular `fn`
	//     shaped like one of those names is just a method; `cfn` is
	//     what wires it to the compiler's hooks. `isCfn=true,
	//     isCompTimeFn=false`.
	//   * At top level (free function) — declares a compile-time
	//     function: the body executes at compile time at each call
	//     site, with @-emit intrinsics generating runtime code into
	//     the caller. `isCfn=false, isCompTimeFn=true`. Used by
	//     `std/fmt.jam` to ship `print` / `eprint` as real Jam
	//     source.
	// The two meanings are mutually exclusive — set by the
	// parseFunction-caller based on declaration context.
	bool isCfn;
	bool isCompTimeFn = false;
	// Name of the enclosing struct, if this is a method declared inside
	// a `const T = struct { fn name(...) ... }` body. Empty for free
	// functions and for clones of generic-instantiated methods (those
	// already carry a qualified Name like `Vec__i32.push`, so the
	// mangler keeps that as the LLVM symbol).
	//
	// The mangler reads this to emit unique LLVM symbols for methods
	// across structs — without it, two structs that both define
	// `fn init()` would collapse to the same `_init` symbol and the
	// linker would silently pick one definition for both call sites.
	std::string parentStruct;

	// Path of the module this function was declared in (e.g. "timer"
	// for code in timer.jam). Empty for the entry module (whose
	// root file scope is the unqualified
	// namespace) and for generic clones that already carry qualified
	// names. ModuleResolver stamps this when a module is loaded.
	//
	// Combined with parentStruct, the mangler can emit
	// dotted LLVM symbols (`timer.Timer.read32`) so same-named
	// methods or free fns in different modules don't collide.
	std::string modulePath;

	FunctionAST(std::string Name, std::vector<Param> Args, TypeIdx ReturnType,
	            std::vector<NodeIdx> Body, bool isExtern = false,
	            bool isExport = false, bool isPub = false, bool isTest = false,
	            bool isVarArgs = false, bool isCfn = false)
	    : Name(std::move(Name)), Args(std::move(Args)), ReturnType(ReturnType),
	      Body(std::move(Body)), isExtern(isExtern), isExport(isExport),
	      isPub(isPub), isTest(isTest), isVarArgs(isVarArgs), isCfn(isCfn) {}

	// a function is generic iff any of its parameters has
	// type `type` (the meta-type) or its return type is `type`. Generic
	// functions are not lowered to LLVM at decl time — instead they are
	// registered for compile-time instantiation at each call site that
	// supplies concrete type arguments.
	bool isGeneric() const {
		if (ReturnType == BuiltinType::Type) return true;
		// Compile-time functions are always per-call-site instantiated
		// — the body runs at compile time and emits caller-specific
		// code. Same dispatch shape as type/comp-value generics.
		if (isCompTimeFn) return true;
		for (const Param &p : Args) {
			// Type-parameter generics (`T: type`) and value-parameter
			// generics (`comp n: u32`) both monomorphize per call
			// site — the same machinery handles both, so they fall
			// under the same generic predicate.
			if (p.Type == BuiltinType::Type) return true;
			if (p.isComp) return true;
		}
		return false;
	}
};

// Top-level struct declaration: const Vec3 = struct { x: f32, y: f32 };
//
// Methods declared inside the struct body — `fn name(self: ..., ...) ...`
// — live in `Methods`. The method's `FunctionAST::Name` stays as the user
// wrote it (e.g. `"drop"`); qualification (`Vec3.drop`) is applied at
// registration time in module dispatch. See docs/STRUCT_METHODS.md.
class StructDeclAST {
  public:
	std::string Name;
	std::vector<std::pair<std::string, TypeIdx>> Fields;  // (name, type)
	std::vector<std::unique_ptr<FunctionAST>> Methods;
	bool isPub = false;

	StructDeclAST(std::string Name,
	              std::vector<std::pair<std::string, TypeIdx>> Fields,
	              std::vector<std::unique_ptr<FunctionAST>> Methods = {})
	    : Name(std::move(Name)), Fields(std::move(Fields)),
	      Methods(std::move(Methods)) {}
};

// One variant of an enum declaration. Unit variants (no payload) have
// an empty `PayloadTypes` vector; tagged variants carry a sequence of
// positional payload types: `Variant(T1, T2, ...)`.
//
// Discriminants default to "previous variant's value + 1" (starting
// from zero for the first variant). An explicit `Variant = N` overrides
// that for the variant in question, and subsequent variants resume
// counting from N+1.
struct EnumVariantAST {
	std::string Name;
	std::vector<TypeIdx> PayloadTypes;
	uint32_t Discriminant = 0;  // resolved value, set during parse
};

// Top-level enum declaration:
//   const Direction = enum { Up, Down, Left, Right };
//   const Op = enum { Nop, LdR8R8(u8, u8), Jp(u16) };
//
// Variants get sequential discriminants assigned in declaration order
// (Up=0, Down=1, …). Unit variants have no payload; tagged variants
// carry positional fields. Codegen lowers the enum to:
//   - `i8` if every variant is unit
//   - `{i8 tag, [maxPayloadSize x i8]}` aligned to max payload align
//     if any variant has a payload
class EnumDeclAST {
  public:
	std::string Name;
	std::vector<EnumVariantAST> Variants;
	bool isPub = false;

	EnumDeclAST(std::string Name, std::vector<EnumVariantAST> Variants)
	    : Name(std::move(Name)), Variants(std::move(Variants)) {}
};

// Top-level union declaration:
//   const FloatBits = union { i: u32, f: f32 };
//
// Untagged C-style union — every field shares the same address; the
// program tracks which field is "live" out-of-band. Reading a field
// other than the most recently written one reinterprets the bits.
//
// Layout: a struct of `{ alignedType, [pad x i8] }` where alignedType
// is the field with the largest alignment requirement and `pad` is the
// extra bytes needed to reach the largest field's size. This gives the
// union the alignment of the most-aligned field and the size of the
// largest field — matching the C ABI convention.
class UnionDeclAST {
  public:
	std::string Name;
	std::vector<std::pair<std::string, TypeIdx>> Fields;  // (name, type)
	bool isPub = false;

	UnionDeclAST(std::string Name,
	             std::vector<std::pair<std::string, TypeIdx>> Fields)
	    : Name(std::move(Name)), Fields(std::move(Fields)) {}
};

// Module-scope value binding: `const NAME[: T]? = expr;`.
//
// Bound to a single integer/bool/float literal (or constant expression
// evaluable at codegen time). At each use site, the init expression is
// codegen'd inline with `DeclaredType` as the expected type — a small
// inlining pass that costs nothing at runtime and lets the optimizer
// fold across uses just like a literal would.
class ConstDeclAST {
  public:
	std::string Name;
	TypeIdx DeclaredType;  // kNoType when omitted; init drives the type
	NodeIdx InitExpr;
	// if non-kNoType, this const is a type alias (RHS is a
	// generic-instantiation expression like `Box(i32)`). InitExpr is
	// kNoNode in that case. main.cpp processes type-alias consts
	// before regular consts and registers the alias in the codegen
	// context's type-alias table.
	TypeIdx AliasedType = kNoType;
	bool isPub = false;
	// `comp const NAME = expr;` — the initializer is required to fold
	// at compile time. Registration eagerly validates the fold (a
	// plain const with a non-foldable init only errors when a comp
	// position consumes it; a comp const errors at the declaration).
	bool isComp = false;

	ConstDeclAST(std::string Name, TypeIdx DeclaredType, NodeIdx InitExpr)
	    : Name(std::move(Name)), DeclaredType(DeclaredType),
	      InitExpr(InitExpr) {}
};

// const std = import("std");
//
// When written as `pub const X = import(...)` at module scope, the
// handle becomes a re-export: a downstream importer that names *this*
// module can then reach `module.X.member` to descend into the re-
// exported module. Tracked via `isPub`.
//
// `chain` captures trailing `.seg.seg` access on the RHS — e.g. `const
// fmt = import("std").fmt;` parses with `Path="std"` and `chain=["fmt"]`.
// Each segment walks a `pub const X = import(...)` re-export in the
// preceding module's namespace; the final path is what `Name` binds to.
class ImportDeclAST {
  public:
	std::string Name;
	std::string Path;
	std::vector<std::string> chain;
	bool isPub = false;

	ImportDeclAST(std::string Name, std::string Path)
	    : Name(std::move(Name)), Path(std::move(Path)) {}
};

// `chain` captures trailing `.seg.seg` access on the RHS — e.g.
// `const {print} = import("std").fmt;` parses with `Path="std"` and
// `chain=["fmt"]`. Resolution walks pub-import re-exports in each
// intermediate module's namespace, ending at the module each
// destructured name is pulled from.
class DestructuringImportDeclAST {
  public:
	std::vector<std::string> Names;
	std::string Path;
	std::vector<std::string> chain;

	DestructuringImportDeclAST(std::vector<std::string> Names, std::string Path)
	    : Names(std::move(Names)), Path(std::move(Path)) {}
};

class ModuleAST {
  public:
	std::vector<std::unique_ptr<ImportDeclAST>> Imports;
	std::vector<std::unique_ptr<DestructuringImportDeclAST>>
	    DestructuringImports;
	std::vector<std::unique_ptr<StructDeclAST>> Structs;
	std::vector<std::unique_ptr<UnionDeclAST>> Unions;
	std::vector<std::unique_ptr<EnumDeclAST>> Enums;
	std::vector<std::unique_ptr<ConstDeclAST>> Consts;
	std::vector<std::unique_ptr<FunctionAST>> Functions;
	// bodies of `struct { ... }` expressions. Each entry is
	// a regular StructDeclAST with a synthetic name (`__anon_struct_<N>`).
	// Index N comes from the AnonStructs vector at parse time and is
	// stored in the StructExpr AST node's d.lhs slot. The substitution
	// engine in reads from here to instantiate the struct with
	// concrete type arguments.
	std::vector<std::unique_ptr<StructDeclAST>> AnonStructs;
	std::vector<std::unique_ptr<EnumDeclAST>> AnonEnums;

	ModuleAST() = default;
};

#endif  // AST_H
