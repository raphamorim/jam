/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef CODEGEN_H
#define CODEGEN_H

#include "ast_flat.h"
#include "drop_registry.h"
#include "jam_llvm.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations for AST types referenced by pointer/reference
// here. Full definitions live in ast.h, included by ast.cpp / codegen.cpp.
class FunctionAST;
class StructDeclAST;
class EnumDeclAST;

class JamCodegenContext {
  public:
	JamCodegenContext(const char *moduleName);
	~JamCodegenContext();

	// Non-copyable
	JamCodegenContext(const JamCodegenContext &) = delete;
	JamCodegenContext &operator=(const JamCodegenContext &) = delete;

	// Accessors
	JamContextRef getContext() const { return ctx; }
	JamModuleRef getModule() const { return mod; }
	JamBuilderRef getBuilder() const { return builder; }

	// Type helpers
	JamTypeRef getInt1Type() const { return JamLLVMInt1Type(ctx); }
	JamTypeRef getInt8Type() const { return JamLLVMInt8Type(ctx); }
	JamTypeRef getInt16Type() const { return JamLLVMInt16Type(ctx); }
	JamTypeRef getInt32Type() const { return JamLLVMInt32Type(ctx); }
	JamTypeRef getInt64Type() const { return JamLLVMInt64Type(ctx); }
	JamTypeRef getFloatType() const { return JamLLVMFloatType(ctx); }
	JamTypeRef getDoubleType() const { return JamLLVMDoubleType(ctx); }
	JamTypeRef getVoidType() const { return JamLLVMVoidType(ctx); }

	// Get type from Jam type string. Internally parses once, interns into
	// the TypePool, then resolves to an LLVM type via getLLVMType(TypeIdx).
	JamTypeRef getTypeFromString(const std::string &typeStr) const;

	// First-class type-pool entry points. internFromString parses a type
	// syntax fragment into a canonical TypeIdx; getLLVMType resolves that
	// TypeIdx to its LLVM type (cached). Once the parser/AST migrate to
	// TypeIdx, getTypeFromString becomes a thin wrapper.
	TypeIdx internFromString(const std::string &typeStr) const;
	JamTypeRef getLLVMType(TypeIdx ty) const;
	TypePool &getTypePool() { return typePool; }
	const TypePool &getTypePool() const { return typePool; }
	StringPool &getStringPool() { return stringPool; }
	const StringPool &getStringPool() const { return stringPool; }
	NodeStore &getNodeStore() { return nodeStore; }
	const NodeStore &getNodeStore() const { return nodeStore; }

	// Struct registry. Field names stay as std::string for now (struct
	// fields are queried by name); field types are TypeIdx.
	struct StructInfo {
		std::string name;
		JamTypeRef type;
		std::vector<std::pair<std::string, TypeIdx>> fields;
	};
	void
	registerStruct(const std::string &name, JamTypeRef type,
	               std::vector<std::pair<std::string, TypeIdx>> fields) const;
	const StructInfo *getStruct(const std::string &name) const;
	// Convenience: resolve a TypeIdx of kind Struct back to its StructInfo.
	const StructInfo *lookupStruct(TypeIdx ty) const;
	int getFieldIndex(const std::string &structName,
	                  const std::string &fieldName) const;

	// Union registry. Untagged unions: every field shares the same
	// address. UnionInfo carries the union's LLVM storage type plus the
	// list of (fieldName, fieldType) pairs for member-access lookup.
	struct UnionInfo {
		std::string name;
		JamTypeRef type;
		std::vector<std::pair<std::string, TypeIdx>> fields;
	};
	void registerUnion(const std::string &name, JamTypeRef type,
	                   std::vector<std::pair<std::string, TypeIdx>> fields);
	const UnionInfo *getUnion(const std::string &name) const;
	const UnionInfo *lookupUnion(TypeIdx ty) const;
	TypeIdx getUnionFieldType(const std::string &unionName,
	                          const std::string &fieldName) const;

	// Enum registry. Each variant has a name and zero or more payload
	// fields (positional). When every variant is payload-less ("unit"),
	// the enum lowers to plain `i8`. When at least one variant carries
	// a payload, the enum lowers to a struct
	//     { i8 tag, [maxPayloadSize x i8] payload }
	// aligned to the max alignment of any variant's payload struct.
	struct EnumVariantInfo {
		std::string name;
		std::vector<TypeIdx> payloadTypes;  // empty for unit variants
		uint32_t discriminant = 0;          // wire / runtime tag value
	};
	struct EnumInfo {
		std::string name;
		JamTypeRef type = nullptr;  // LLVM type (i8 or {i8, [N x i8]})
		std::vector<EnumVariantInfo> variants;
		// True if at least one variant carries a payload. Determines
		// whether construction / pattern-matching follow the unit-only
		// fast path or the tagged-union path.
		bool hasPayloadVariant = false;
		// Max payload size in bytes; 0 if every variant is unit.
		uint64_t maxPayloadSize = 0;
		// Max payload alignment; 1 if every variant is unit.
		uint64_t maxPayloadAlign = 1;
	};
	void registerEnum(const std::string &name,
	                  std::vector<EnumVariantInfo> variants) const;
	void setEnumLLVMType(const std::string &name, JamTypeRef llvmType,
	                     uint64_t maxPayloadSize, uint64_t maxPayloadAlign,
	                     bool hasPayloadVariant) const;
	const EnumInfo *getEnum(const std::string &name) const;
	const EnumInfo *lookupEnum(TypeIdx ty) const;
	// Reverse-lookup: given an LLVM struct type, return the enum whose
	// payloaded layout matches, or nullptr. Used to disambiguate enum
	// scrutinees from slice-shaped structs in `match`.
	const EnumInfo *findEnumByLLVMType(JamTypeRef ty) const;
	// Index of `variantName` within `enumName`, or -1 if not found.
	int getEnumVariantIndex(const std::string &enumName,
	                        const std::string &variantName) const;

	// Type-size helpers used by union layout. Returns size/alignment in
	// bytes for any TypeIdx we currently support. Throws for types whose
	// size we don't know how to compute (e.g. user types with
	// unresolvable bodies).
	uint64_t typeSize(TypeIdx ty) const;
	uint64_t typeAlign(TypeIdx ty) const;

	// Module-scope `const NAME[: T]? = expr;` registry. Stores the
	// parsed init expression (NodeIdx) and the declared type (kNoType
	// when omitted). At each Variable use site we re-codegen the init
	// expression in place — effectively a per-use inline expansion that
	// the optimizer collapses just like a literal.
	struct ModuleConstInfo {
		NodeIdx initExpr;
		TypeIdx declaredType;
	};
	void registerModuleConst(const std::string &name, NodeIdx init,
	                         TypeIdx declared);
	const ModuleConstInfo *getModuleConst(const std::string &name) const;

	// drop emission state.
	//
	// The DropRegistry pointer is set once per module before codegen begins
	// and lives for the duration of the run. When VarDecl codegen sees a
	// declaration whose type has a registered drop fn, the binding is
	// pushed to `drops_`. At every Return — and at the implicit
	// fall-through end of a function body — the codegen emits a call to
	// each entry's drop fn in reverse declaration order, then clears the
	// list for the next function.
	struct DropEntry {
		std::string name;
		JamValueRef alloca;
		JamTypeRef llvmType;        // LLVM type of the binding's storage
		const FunctionAST *dropFn;  // borrowed; lives on the ModuleAST
	};
	void setDropRegistry(const jam::drops::DropRegistry *r) {
		dropRegistry = r;
	}
	const jam::drops::DropRegistry *getDropRegistry() const {
		return dropRegistry;
	}
	// look up a drop method for an instantiated struct
	// (e.g. Box__i32). Falls back to the pre-built drop registry.
	// Returns nullptr if the struct has no drop method.
	const FunctionAST *lookupDropFn(const std::string &structName) const {
		auto it = instantiatedDrops_.find(structName);
		if (it != instantiatedDrops_.end()) return it->second;
		if (dropRegistry) {
			auto rit = dropRegistry->find(structName);
			if (rit != dropRegistry->end()) return rit->second;
		}
		return nullptr;
	}
  private:
	JamContextRef ctx;
	JamModuleRef mod;
	JamBuilderRef builder;
	// `structs` is mutable because lazily instantiates new
	// struct types (e.g. `Maybe(File)`) on demand from inside the
	// otherwise-const `resolveGenericCall` / `getLLVMType` paths.
	mutable std::map<std::string, StructInfo> structs;
	std::map<std::string, UnionInfo> unions;
	mutable std::map<std::string, EnumInfo> enums;
	std::map<std::string, ModuleConstInfo> moduleConsts;
	mutable TypePool typePool;
	mutable StringPool stringPool;
	mutable NodeStore nodeStore;
	// Lazy LLVM type per TypeIdx (built once, reused). Indexed by TypeIdx.
	mutable std::vector<JamTypeRef> llvmTypeCache;

	// Pre-built map of struct → drop fn (built before any function is
	// lowered). AstGen consults this to wire drop calls at scope exit;
	// drop tracking itself lives in the per-function `AstGenCtx`.
	const jam::drops::DropRegistry *dropRegistry = nullptr;

	// callsite ABI: when codegen for a call expression needs to know
	// the callee's parameter modes (e.g. to decide whether to auto-take
	// the address of an arg for a large `let` aggregate), it looks up
	// the FunctionAST here. Populated by main.cpp during prototype
	// declaration; lifetime tied to the parsed module.
  public:
	void registerFunctionAST(const std::string &name, const FunctionAST *fn);
	const FunctionAST *getFunctionAST(const std::string &name) const;

	struct ImportHandleInfo {
		std::string modulePath;
		std::unordered_set<std::string> privateNames;
	};
	void registerImportHandle(const std::string &handle,
	                          const std::string &modulePath);
	void registerPrivateName(const std::string &handle,
	                         const std::string &name);
	const ImportHandleInfo *getImportHandle(const std::string &handle) const;
	std::string formatNamespaceLookupError(const std::string &kind,
	                                       const std::string &qualified) const;

  private:
	std::unordered_map<std::string, const FunctionAST *> functionAsts;
	std::unordered_map<std::string, ImportHandleInfo> importHandles_;

	// `genericResolutions_` memoizes per-callsite: every unique
	// `TypeKind::GenericCall` TypeIdx maps to the resolved TypeIdx.
	// Fast path for repeated lookups of the exact same syntactic call.
	//
	// `genericInstances_` is the identity cache: it deduplicates
	// instantiations of the *same* FunctionAST with the *same* concrete
	// args across syntactic paths (e.g. bare `Vec(i32)` and namespace-
	// qualified `c.Vec(i32)` both resolve to the same Vec FunctionAST
	// and produce a single Vec__i32 struct).
	mutable std::unordered_map<TypeIdx, TypeIdx> genericResolutions_;
	struct GenericInstanceKey {
		const FunctionAST *fn;
		std::vector<TypeIdx> args;
		bool operator==(const GenericInstanceKey &o) const {
			return fn == o.fn && args == o.args;
		}
	};
	struct GenericInstanceKeyHash {
		size_t operator()(const GenericInstanceKey &k) const {
			size_t h = std::hash<const void *>{}(k.fn);
			for (TypeIdx t : k.args) {
				h ^=
				    std::hash<uint32_t>{}(t) + 0x9e3779b9 + (h << 6) + (h >> 2);
			}
			return h;
		}
	};
	mutable std::unordered_map<GenericInstanceKey, TypeIdx,
	                           GenericInstanceKeyHash>
	    genericInstances_;

	// borrowed pointer to the parsed module's anonymous
	// struct bodies (those produced by `struct { ... }` expressions).
	// Set by main.cpp before any codegen runs. The substitution engine
	// reads from here when resolving generic calls whose bodies contain
	// `return struct {...};`.
	const std::vector<std::unique_ptr<StructDeclAST>> *anonStructs_ = nullptr;

	// Mirror of anonStructs_ for `enum { ... }` expressions. Read by
	// instantiateEnumExpr to materialize generic sum types.
	const std::vector<std::unique_ptr<EnumDeclAST>> *anonEnums_ = nullptr;

	// type alias table. `const BoxI32 = Box(i32);` registers
	// `BoxI32 → resolved-TypeIdx-of-Box(i32)`. Consulted by lookupStruct
	// (and by getLLVMType via the recursive lookup path) so a binding
	// declared `var b: BoxI32` finds the same struct that `Box(i32)`
	// would produce.
	mutable std::map<std::string, TypeIdx> typeAliases_;

	// clones of FunctionASTs produced by method
	// instantiation. Each clone has substituted parameter and return
	// types and a unique source-level name (`Box__i32.unwrap`) so the
	// existing function registry / LLVM symbol pipeline handles them
	// as ordinary functions.
	mutable std::vector<std::unique_ptr<FunctionAST>> instantiatedMethods_;

	// drop methods on instantiated types. The pre-built
	// drop registry is borrowed via const pointer and was populated
	// before lazy instantiation. Drop methods produced by
	// instantiateStructExpr go here and are consulted alongside the
	// pre-built registry by var-decl codegen.
	mutable std::unordered_map<std::string, const FunctionAST *>
	    instantiatedDrops_;

	// type substitution context active during codegen of an
	// instantiated method's body. Lookups of Named types (T, Self,
	// __anon_struct_N) consult this map first. Set/cleared around
	// jirDeclarePrototype + jirDefineBody calls in instantiateStructExpr.
	mutable std::unordered_map<std::string, TypeIdx> currentSubst_;

  public:
	// resolve a `TypeKind::GenericCall` TypeIdx to a concrete
	// TypeIdx by running the substitution engine on the generic
	// function's body. Result is memoized — subsequent calls with the
	// same TypeIdx hit the cache and return the same concrete TypeIdx.
	TypeIdx resolveGenericCall(TypeIdx callTy) const;

	// register the anonymous-struct table for the current
	// module so the substitution engine can find struct expression
	// bodies by their AnonStructs index.
	void setAnonStructs(const std::vector<std::unique_ptr<StructDeclAST>> *as) {
		anonStructs_ = as;
	}

	void setAnonEnums(const std::vector<std::unique_ptr<EnumDeclAST>> *ae) {
		anonEnums_ = ae;
	}

	// register a type alias (`const Name = Box(i32);`).
	// Consulted by lookupStruct when resolving a Named TypeIdx whose
	// name matches an alias.
	void registerTypeAlias(const std::string &name, TypeIdx target) {
		typeAliases_[name] = target;
	}
	TypeIdx lookupTypeAlias(const std::string &name) const {
		auto it = typeAliases_.find(name);
		if (it != typeAliases_.end()) return it->second;
		return kNoType;
	}

	// substitution context manipulators. The map is active
	// only during codegen of an instantiated method's body — set right
	// before jirDeclarePrototype/jirDefineBody, cleared right after.
	void setCurrentSubst(std::unordered_map<std::string, TypeIdx> s) const {
		currentSubst_ = std::move(s);
	}
	void clearCurrentSubst() const { currentSubst_.clear(); }
	TypeIdx lookupCurrentSubst(const std::string &name) const {
		auto it = currentSubst_.find(name);
		if (it != currentSubst_.end()) return it->second;
		return kNoType;
	}

  private:
	// instantiate a `struct {...}` expression as the result
	// of a generic call. Substitutes each field's type with the
	// concrete generic args, creates a fresh LLVM struct type with a
	// synthesized name, and returns a Named TypeIdx pointing at it.
	// Memoizes by instantiated name.
	TypeIdx instantiateStructExpr(
	    const AstNode &exprNode, const std::string &calleeName,
	    const std::vector<TypeIdx> &args,
	    const std::unordered_map<std::string, TypeIdx> &subst) const;

	// Mirror of instantiateStructExpr for `enum { ... }` expressions.
	// Substitutes each variant's payload types with the concrete
	// generic args, registers a fresh tagged-union LLVM layout, and
	// returns an Enum TypeIdx pointing at it. Memoizes by instantiated
	// name. Used to materialize `Option(i32)`, `Result(File, Errno)`,
	// and other sum-type generics on demand.
	TypeIdx instantiateEnumExpr(
	    const AstNode &exprNode, const std::string &calleeName,
	    const std::vector<TypeIdx> &args,
	    const std::unordered_map<std::string, TypeIdx> &subst) const;
};

#endif  // CODEGEN_H
