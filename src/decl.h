/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_DECL_H
#define JAM_DECL_H

#include "abi.h"
#include "ast_flat.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class FunctionAST;
class StructDeclAST;
class EnumDeclAST;
class UnionDeclAST;
class ConstDeclAST;

namespace jam {

// ABI signature of a non-generic function, computed once by the
// analyzer and cached so callers don't re-classify at every site.
// Generic functions skip this cache — each instantiation is its own
// FunctionAST with its own substituted param TypeIdxs, so the cache
// would never hit. `computed` distinguishes "analyzer ran" from the
// default-constructed empty-vector state.
struct FnSignature {
	std::vector<jam::abi::ParamABI> params;
	jam::abi::ReturnABI returnAbi{};
	bool computed = false;
};

// Stable index into the per-compilation DeclTable. Index 0 is reserved
// as kNoDecl so callers can use it as a sentinel.
using DeclIndex = uint32_t;
constexpr DeclIndex kNoDecl = 0;

// What sort of source-level binding produced this Decl. Determines
// which AST pointer is non-null and which analysis branch runs.
enum class DeclKind : uint8_t {
	Invalid = 0,
	Function,   // fn name(...) ret { ... }  + extern + tfn
	Struct,     // const Name = struct { ... };
	Enum,       // const Name = enum { ... };
	Union,      // const Name = union { ... };
	Const,      // const NAME[: T]? = expr;  (value, not type alias)
	TypeAlias,  // const NAME = SomeType  /  GenericCall, evaluated to a type
};

// Where this Decl sits in the analysis lifecycle. The chokepoint state
// is InProgress: re-entering it triggers cycle detection. Trimmed to
// the states a non-incremental compiler needs.
enum class DeclAnalysis : uint8_t {
	Unreferenced,       // not yet touched
	InProgress,         // analyzer running, cycle-detector key
	Complete,           // value populated
	AnalysisFailure,    // this decl's own analysis failed
	DependencyFailure,  // a Decl we depended on failed
};

// A Struct's *body* lifecycle, distinct from its Decl's value. The
// Decl can be Complete (the Struct's typed value is `(type, NamedTy)`)
// long before the Struct's fields are materialised. Field materialisation
// only happens when something — `getLLVMType`, `@sizeOf`, a field
// access — actually demands it.
enum class StructStatus : uint8_t {
	None,            // declared, fields not yet computed
	FieldTypesWIP,   // computing fields; cycle-detector key
	HaveFieldTypes,  // field TypeIdxs known; LLVM body may still be empty
	LayoutWIP,       // computing LLVM body / size / align
	HaveLayout,      // LLVM body emitted; size/align cached
};

// Enum / Union body lifecycles. Same shape as StructStatus but kept
// in their own enums so a tagged-union variant's status can't be
// silently treated as a struct's. Both follow None -> BodyWIP ->
// HaveBody; the WIP state is the cycle-detector key for
// `resolveTypeFieldsEnum` / `resolveTypeFieldsUnion`.
enum class EnumStatus : uint8_t {
	None,
	BodyWIP,
	HaveBody,
};

enum class UnionStatus : uint8_t {
	None,
	BodyWIP,
	HaveBody,
};

// The value a fully-analyzed Decl resolves to. Tiny mini-Value: Jam's
// generic args today are only types, so we don't need comptime int /
// big-int / pointer-decl-ref variants. Adding them later is just new
// enum cases + payload.
struct DeclValue {
	enum class Kind : uint8_t {
		None,      // not yet computed / failed
		Type,      // a type value (struct, enum, alias result, builtin)
		Function,  // a fn reference
	};
	Kind kind = Kind::None;
	TypeIdx type = kNoType;                 // when kind == Type
	const FunctionAST *function = nullptr;  // when kind == Function
};

// One source-level declaration. Owned by DeclTable; held by index so
// stable across reallocation.
struct Decl {
	std::string name;  // user-visible identifier (no namespacing)
	DeclKind kind = DeclKind::Invalid;
	DeclAnalysis analysis = DeclAnalysis::Unreferenced;
	StructStatus structStatus = StructStatus::None;
	EnumStatus enumStatus = EnumStatus::None;
	UnionStatus unionStatus = UnionStatus::None;

	// Exactly one of these is non-null; which one is determined by kind.
	const FunctionAST *fnAst = nullptr;
	const StructDeclAST *structAst = nullptr;
	const EnumDeclAST *enumAst = nullptr;
	const UnionDeclAST *unionAst = nullptr;
	const ConstDeclAST *constAst = nullptr;

	// Populated when analysis == Complete. For Struct/Enum/Union/TypeAlias
	// this carries the resolved TypeIdx; for Function it carries the
	// FunctionAST*; for Const it stays kind==None and the existing
	// JamCodegenContext::registerModuleConst path handles the value.
	DeclValue value;

	// Cached ABI signature for non-generic functions. Set by
	// `Analyzer::analyzeFunction`; reads must check `signature.computed`.
	// See `FnSignature` above.
	FnSignature signature;

	// Best-effort source position for diagnostics. Resolved at register
	// time from the AST node's first token.
	int line = 0;
	std::string file;

	// Shallow dependency graph: who I depend on and who depends on me.
	// Symmetric: declareDependency(A, B) appends to both sides.
	std::vector<DeclIndex> dependencies;
	std::vector<DeclIndex> dependants;
};

// Central registry — one per compilation. Holds every named top-level
// binding from the main module and every imported module's pub items.
// Index 0 holds a sentinel Invalid Decl so kNoDecl reads as "not found".
class DeclTable {
  public:
	DeclTable();

	// Create a new Decl. Caller fills the *Ast pointer matching `kind`
	// after the call. Returns the new index.
	DeclIndex create(DeclKind kind, std::string name);

	Decl &get(DeclIndex idx);
	const Decl &get(DeclIndex idx) const;

	// Find a Decl by its source-level name. Returns kNoDecl if not
	// present. Used by the register pass to wire dependencies and by
	// the demand pass to translate a name reference into an index.
	DeclIndex findByName(const std::string &name) const;

	// Kind-filtered lookup. Jam permits a function and a type to
	// share a name (e.g. `fn Counter(T:type) type` lives in the
	// function namespace; `const Counter = struct {...}` lives in
	// the type namespace). Pass the kind you want; returns kNoDecl
	// if no decl of that kind has the given name.
	DeclIndex findByNameAndKind(const std::string &name, DeclKind kind) const;

	// Wire a symmetric `from -> to` dependency edge so trace output
	// can walk either direction. Idempotent: a duplicate edge is a
	// no-op.
	void declareDependency(DeclIndex from, DeclIndex to);

	// Number of registered decls excluding the sentinel.
	std::size_t size() const { return decls_.size() - 1; }

	// Iteration helpers; caller is expected to skip kNoDecl.
	const std::vector<Decl> &all() const { return decls_; }

  private:
	std::vector<Decl> decls_;
	std::unordered_map<std::string, DeclIndex> byName_;
};

}  // namespace jam

#endif  // JAM_DECL_H
