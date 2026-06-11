/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "codegen.h"

#include "analyzer.h"
#include "ast.h"
#include "astgen.h"
#include "jir_codegen.h"
#include "jir_verify.h"

#include <iostream>
#include <stdexcept>

JamCodegenContext::JamCodegenContext(const char *moduleName) {
	ctx = JamLLVMCreateContext();
	mod = JamLLVMCreateModule(moduleName, ctx);
	builder = JamLLVMCreateBuilder(ctx);
}

JamCodegenContext::~JamCodegenContext() {
	JamLLVMDisposeBuilder(builder);
	JamLLVMDisposeModule(mod);
	JamLLVMDisposeContext(ctx);
}

jam::Analyzer &JamCodegenContext::analyzer() const {
	if (!analyzer_) {
		// Lazy ctor — needs a fully-constructed JamCodegenContext to
		// reference for diagnostics / file / type pool. The DeclTable
		// is a sibling member, so its address is stable.
		analyzer_ = std::make_unique<jam::Analyzer>(
		    const_cast<JamCodegenContext &>(*this), declTable_);
	}
	return *analyzer_;
}

// Parse a type-syntax string into the canonical TypeIdx (recursive). The
// parser will eventually produce TypeIdx directly and this function will
// only be called by legacy callers that still hold strings.
TypeIdx JamCodegenContext::internFromString(const std::string &typeStr) const {
	// `const T` is purely semantic — strip and continue.
	if (typeStr.length() >= 6 && typeStr.substr(0, 6) == "const ") {
		return internFromString(typeStr.substr(6));
	}
	// Built-in scalars resolve to pre-interned indices in the pool.
	if (typeStr == "u8") return BuiltinType::U8;
	if (typeStr == "i8") return BuiltinType::I8;
	if (typeStr == "u16") return BuiltinType::U16;
	if (typeStr == "i16") return BuiltinType::I16;
	if (typeStr == "u32") return BuiltinType::U32;
	if (typeStr == "i32") return BuiltinType::I32;
	if (typeStr == "u64") return BuiltinType::U64;
	if (typeStr == "i64") return BuiltinType::I64;
	if (typeStr == "f32") return BuiltinType::F32;
	if (typeStr == "f64") return BuiltinType::F64;
	if (typeStr == "bool" || typeStr == "u1") return BuiltinType::Bool;
	// `str` is a slice of u8.
	if (typeStr == "str") { return typePool.internSlice(BuiltinType::U8); }
	// User-defined struct (looked up in the registry by name).
	if (getStruct(typeStr)) {
		StringIdx nameId = stringPool.intern(typeStr);
		return typePool.internStruct(nameId);
	}
	// Pointers: *T  or  [*]T.
	if (typeStr.length() >= 3 && typeStr.substr(0, 3) == "[*]") {
		return typePool.internPtrMany(internFromString(typeStr.substr(3)));
	}
	if (typeStr.length() >= 2 && typeStr[0] == '*') {
		return typePool.internPtrSingle(internFromString(typeStr.substr(1)));
	}
	// Slice []T (`[]const T` already had its const stripped above).
	if (typeStr.length() >= 2 && typeStr.substr(0, 2) == "[]") {
		return typePool.internSlice(internFromString(typeStr.substr(2)));
	}
	// Fixed-size array [N]T.
	if (typeStr.length() >= 3 && typeStr[0] == '[') {
		size_t closeBracket = typeStr.find(']');
		if (closeBracket == std::string::npos || closeBracket == 1) {
			throw std::runtime_error("Malformed array type: " + typeStr);
		}
		uint32_t len = static_cast<uint32_t>(
		    std::stoull(typeStr.substr(1, closeBracket - 1)));
		TypeIdx elem = internFromString(typeStr.substr(closeBracket + 1));
		return typePool.internArray(elem, len);
	}
	throw std::runtime_error("Unknown type: " + typeStr);
}

JamTypeRef JamCodegenContext::getLLVMType(TypeIdx ty) const {
	// Resolve a bare user-type reference to its module-qualified TypeIdx
	// before anything else, so the per-TypeIdx LLVM-type cache below keys
	// each module's same-named type separately. Declaration-level types
	// (struct fields, etc.) are already qualified at registration, so
	// this only fires for body-level references, where currentBodyModule
	// names the type's owning module. Qualified / primitive types are
	// returned unchanged, so this is a cheap no-op in the common case.
	ty = requalifyType(ty, currentBodyModule());
	if (ty >= llvmTypeCache.size()) { llvmTypeCache.resize(ty + 1, nullptr); }
	if (llvmTypeCache[ty]) return llvmTypeCache[ty];

	const TypeKey &k = typePool.get(ty);
	JamTypeRef result = nullptr;
	switch (k.kind) {
	case TypeKind::Invalid:
	case TypeKind::Void:
	case TypeKind::NoReturn:
		result = getVoidType();
		break;
	case TypeKind::Bool:
		result = getInt1Type();
		break;
	case TypeKind::Int: {
		switch (k.a) {
		case 8:
			result = getInt8Type();
			break;
		case 16:
			result = getInt16Type();
			break;
		case 32:
			result = getInt32Type();
			break;
		case 64:
			result = getInt64Type();
			break;
		default:
			throw std::runtime_error("Unsupported int width");
		}
		break;
	}
	case TypeKind::Float:
		result = (k.a == 32) ? getFloatType() : getDoubleType();
		break;
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany: {
		JamTypeRef elem = getLLVMType(static_cast<TypeIdx>(k.a));
		result = JamLLVMPointerType(elem, 0);
		break;
	}
	case TypeKind::Slice: {
		JamTypeRef elem = getLLVMType(static_cast<TypeIdx>(k.a));
		JamTypeRef elemPtr = JamLLVMPointerType(elem, 0);
		JamTypeRef usize = getInt64Type();
		JamTypeRef parts[2] = {elemPtr, usize};
		result = JamLLVMStructType(ctx, parts, 2, false);
		break;
	}
	case TypeKind::Array: {
		JamTypeRef elem = getLLVMType(static_cast<TypeIdx>(k.a));
		result = JamLLVMArrayType(elem, k.b);
		break;
	}
	case TypeKind::ArrayExpr: {
		// lazily fold the length expression to a canonical Array
		// key, then recurse. The resolution is memoized in
		// arrayExprResolutions_ (same shape as GenericCall below).
		result = getLLVMType(resolveArrayExpr(ty));
		break;
	}
	case TypeKind::Struct: {
		const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
		const auto *sinfo = getStruct(name);
		if (!sinfo) {
			throw std::runtime_error("Unknown struct type: " + name);
		}
		result = sinfo->type;
		break;
	}
	case TypeKind::Named: {
		// Parser-deferred user type. Resolution order:
		//   1. substitution context (T, Self, __anon_struct_N)
		//   2. struct/union/enum registries
		//   3. type alias map
		const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
		TypeIdx substTarget = lookupCurrentSubst(name);
		if (substTarget != kNoType) {
			result = getLLVMType(substTarget);
			break;
		}
		bool privacyBlocked = handlePrivacyBlocked(name);
		if (const auto *sinfo = privacyBlocked ? nullptr : getStruct(name)) {
			result = sinfo->type;
		} else if (const auto *uinfo =
		               privacyBlocked ? nullptr : getUnion(name)) {
			result = uinfo->type;
		} else if (const auto *einfo =
		               privacyBlocked ? nullptr : getEnum(name)) {
			// Unit-only enums lower to i8. Payloaded enums lower
			// to {i8, [N x i8]} via the named struct type set during
			// declaration.
			result = einfo->hasPayloadVariant ? einfo->type : getInt8Type();
		} else {
			TypeIdx aliasTarget = lookupTypeAlias(name);
			if (aliasTarget != kNoType) {
				result = getLLVMType(aliasTarget);
				break;
			}
			// 3+ segment chain through module re-exports.
			if (name.find('.') != std::string::npos) {
				TypeIdx chained = resolveChainedType(name);
				if (chained != kNoType) {
					result = getLLVMType(chained);
					break;
				}
			}
			throw std::runtime_error(
			    formatNamespaceLookupError("user-defined type", name));
		}
		break;
	}
	case TypeKind::Union: {
		const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
		const auto *info = getUnion(name);
		if (!info) { throw std::runtime_error("Unknown union type: " + name); }
		result = info->type;
		break;
	}
	case TypeKind::Enum: {
		// enums lower to u8 — one byte per discriminant.
		result = getInt8Type();
		break;
	}
	case TypeKind::Type:
		// the meta-type has no runtime representation.
		// Reaching this path means a generic function leaked to LLVM
		// codegen without being instantiated first.
		throw std::runtime_error(
		    "internal: cannot lower `type` to LLVM (generic was not "
		    "instantiated before codegen)");
	case TypeKind::GenericCall: {
		// lazily resolve the call to a concrete TypeIdx
		// via the substitution engine, then recurse on the result.
		// The resolution is memoized in genericResolutions so each
		// distinct call site only does the work once.
		TypeIdx resolved = resolveGenericCall(ty);
		result = getLLVMType(resolved);
		break;
	}
	case TypeKind::Fn: {
		// Function-typed values lower to LLVM opaque pointers. The
		// signature isn't part of the LLVM value type (LLVM 15+ uses
		// opaque pointers everywhere); the Jam type system tracks the
		// signature for call-site type-checking. The actual LLVM
		// function type (return + params) is built per indirect call
		// site by jir_codegen, reading the signature out of the
		// TypeKey + fnParamsAt(k.b).
		result = JamLLVMPointerType(getInt8Type(), 0);
		break;
	}
	case TypeKind::Module:
		// Module values have no runtime representation. Reaching here
		// means a module-typed JIR ref leaked to LLVM lowering — the
		// MemberAccess / Call paths should consume it before codegen
		// sees the value. Diagnostic gives the path-string for
		// debugging.
		throw std::runtime_error(
		    "internal: module value `" +
		    stringPool.get(static_cast<StringIdx>(k.a)) +
		    "` reached LLVM codegen (modules are compile-time only)");
	}
	llvmTypeCache[ty] = result;
	return result;
}

JamTypeRef
JamCodegenContext::getTypeFromString(const std::string &typeStr) const {
	return getLLVMType(internFromString(typeStr));
}

void JamCodegenContext::registerStruct(
    const std::string &name, JamTypeRef type,
    std::vector<std::pair<std::string, TypeIdx>> fields) const {
	StructInfo info;
	info.name = name;
	info.type = type;
	info.fields = std::move(fields);
	structs[name] = std::move(info);
}

const JamCodegenContext::StructInfo *
JamCodegenContext::lookupStruct(TypeIdx ty) const {
	if (ty == kNoType) return nullptr;
	// Qualify a bare body-level reference to its owning module so the
	// registry lookup below hits this module's struct, not a same-named
	// one elsewhere. No-op for already-qualified / non-user types.
	ty = requalifyType(ty, currentBodyModule());
	const TypeKey &k = typePool.get(ty);
	// a `GenericCall` TypeIdx resolves to a concrete type
	// (typically a Named struct produced by instantiation). Recurse on
	// the resolved TypeIdx so downstream lookups behave as if the user
	// had written the instantiated name directly.
	if (k.kind == TypeKind::GenericCall) {
		return lookupStruct(resolveGenericCall(ty));
	}
	// Accept TypeKind::Struct (explicit) or TypeKind::Named (parser-
	// deferred user type that resolves to a struct).
	if (k.kind != TypeKind::Struct && k.kind != TypeKind::Named) {
		return nullptr;
	}
	const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
	// substitution context wins (T, Self, __anon_struct_N
	// resolved per-instantiation during method body codegen).
	TypeIdx substTarget = lookupCurrentSubst(name);
	if (substTarget != kNoType) { return lookupStruct(substTarget); }
	// A handle-spelled reference to another module's PRIVATE type must
	// miss (the caller's error path then reports `is not exported`),
	// even when the handle name coincides with the module identity the
	// registry keys by.
	if (handlePrivacyBlocked(name)) { return nullptr; }
	if (const StructInfo *direct = getStruct(name)) { return direct; }
	// try the type alias table — `const BoxI32 = Box(i32);`
	// maps `BoxI32` to the instantiated struct's TypeIdx.
	TypeIdx aliasTarget = lookupTypeAlias(name);
	if (aliasTarget != kNoType) { return lookupStruct(aliasTarget); }
	// 3+ segment chain through module re-exports: `w.lib.Point`.
	if (name.find('.') != std::string::npos) {
		TypeIdx chained = resolveChainedType(name);
		if (chained != kNoType) return lookupStruct(chained);
	}
	return nullptr;
}

const JamCodegenContext::StructInfo *
JamCodegenContext::getStruct(const std::string &name) const {
	auto it = structs.find(name);
	if (it != structs.end()) return &it->second;
	return nullptr;
}

int JamCodegenContext::getFieldIndex(const std::string &structName,
                                     const std::string &fieldName) const {
	const StructInfo *info = getStruct(structName);
	if (!info) return -1;
	for (size_t i = 0; i < info->fields.size(); i++) {
		if (info->fields[i].first == fieldName) return static_cast<int>(i);
	}
	return -1;
}

void JamCodegenContext::registerTypeOwner(const std::string &ctxModule,
                                          const std::string &typeName,
                                          const std::string &ownerModule) {
	typeModuleOf_[ctxModule][typeName] = ownerModule;
	// Ownership feeds every memoized resolution below; registration
	// only happens during the up-front pass, so these clears are
	// effectively free.
	requalifyMemo_.clear();
	dropMemo_.clear();
	sizeMemo_.clear();
	alignMemo_.clear();
}

uint32_t JamCodegenContext::moduleIdOf(const std::string &m) const {
	auto it = moduleIds_.find(m);
	if (it != moduleIds_.end()) return it->second;
	uint32_t id = static_cast<uint32_t>(moduleIds_.size()) + 1;
	moduleIds_.emplace(m, id);
	return id;
}

TypeIdx JamCodegenContext::requalifyType(TypeIdx ty,
                                         const std::string &ctxModule) const {
	if (ty == kNoType) return ty;
	// Kinds that can never carry a user-type name skip everything —
	// the common case for primitive-typed expressions.
	switch (typePool.get(ty).kind) {
	case TypeKind::Named:
	case TypeKind::Struct:
	case TypeKind::Enum:
	case TypeKind::Union:
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
	case TypeKind::Slice:
	case TypeKind::Array:
	case TypeKind::ArrayExpr:
	case TypeKind::GenericCall:
		break;
	default:
		return ty;
	}
	// Subst contexts change which names stay un-qualified — compute
	// fresh there (bounded: only generic-instantiation lowering).
	if (!currentSubst_.empty()) return requalifyTypeUncached(ty, ctxModule);
	uint64_t key = (static_cast<uint64_t>(moduleIdOf(ctxModule)) << 32) |
	               static_cast<uint64_t>(ty);
	auto it = requalifyMemo_.find(key);
	if (it != requalifyMemo_.end()) return it->second;
	TypeIdx r = requalifyTypeUncached(ty, ctxModule);
	requalifyMemo_.emplace(key, r);
	return r;
}

TypeIdx
JamCodegenContext::requalifyTypeUncached(TypeIdx ty,
                                         const std::string &ctxModule) const {
	if (ty == kNoType) return ty;
	// By value, not by reference: the recursive calls below intern new
	// types, which can reallocate the TypePool's storage and dangle a
	// reference into it.
	TypeKey k = typePool.get(ty);
	switch (k.kind) {
	case TypeKind::Named:
	case TypeKind::Struct:
	case TypeKind::Enum:
	case TypeKind::Union: {
		const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
		// Already qualified / handle.Type / chained: leave it for the
		// alias and chained-resolution paths in lookupStruct.
		if (name.find('.') != std::string::npos) return ty;
		// An active generic-parameter substitution (T, Self) wins over a
		// same-named module-level type: the body meant the parameter.
		if (lookupCurrentSubst(name) != kNoType) return ty;
		auto mIt = typeModuleOf_.find(ctxModule);
		if (mIt == typeModuleOf_.end()) return ty;
		auto tIt = mIt->second.find(name);
		if (tIt == mIt->second.end()) return ty;  // std / generic param / Self
		// Entry-module owner: the qualified name IS the bare name —
		// skip the string build entirely.
		if (tIt->second.empty()) return ty;
		std::string q = qualifyTypeName(tIt->second, name);
		TypeKey nk = k;
		nk.a = static_cast<uint32_t>(stringPool.intern(q));
		return typePool.intern(nk);
	}
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
	case TypeKind::Slice:
	case TypeKind::Array:
	case TypeKind::ArrayExpr: {
		// Wrapper kinds carry the element TypeIdx in `a`; `b` (length /
		// length-expr) is preserved untouched.
		TypeIdx elem = requalifyType(static_cast<TypeIdx>(k.a), ctxModule);
		if (elem == static_cast<TypeIdx>(k.a)) return ty;
		TypeKey nk = k;
		nk.a = static_cast<uint32_t>(elem);
		return typePool.intern(nk);
	}
	case TypeKind::GenericCall: {
		// Qualify BOTH the callee and the type arguments. The callee
		// (`Vec`) qualifies to its owning module's identity
		// (`std/collections.Vec`) exactly like a concrete type name —
		// two modules' same-named generic factories must never share a
		// GenericCall TypeIdx, or the per-TypeIdx resolution cache hands
		// one module the other's instantiation. Args qualify so the
		// instantiation substitutes the right concrete types
		// (`Option(File)` -> `Option(std/fs.File)`).
		//
		// Copy the name and args BEFORE recursing: the recursive
		// requalifyType calls below may intern new types, reallocating
		// the TypePool's internal vectors and invalidating any reference
		// into them (`k` and the args span).
		StringIdx nameId = static_cast<StringIdx>(k.a);
		std::vector<TypeIdx> args =
		    typePool.genericArgsAt(static_cast<uint32_t>(k.b));
		bool changed = false;
		{
			const std::string &callee = stringPool.get(nameId);
			if (callee.find('.') == std::string::npos &&
			    lookupCurrentSubst(callee) == kNoType) {
				auto mIt = typeModuleOf_.find(ctxModule);
				if (mIt != typeModuleOf_.end()) {
					auto tIt = mIt->second.find(callee);
					if (tIt != mIt->second.end() && !tIt->second.empty()) {
						nameId = stringPool.intern(
						    qualifyTypeName(tIt->second, callee));
						changed = true;
					}
				}
			}
		}
		std::vector<TypeIdx> newArgs;
		newArgs.reserve(args.size());
		for (TypeIdx a : args) {
			TypeIdx r = requalifyType(a, ctxModule);
			if (r != a) changed = true;
			newArgs.push_back(r);
		}
		if (!changed) return ty;
		return typePool.internGenericCall(nameId, std::move(newArgs));
	}
	default:
		return ty;
	}
}

// Union registry

void JamCodegenContext::registerUnion(
    const std::string &name, JamTypeRef type,
    std::vector<std::pair<std::string, TypeIdx>> fields) {
	UnionInfo info;
	info.name = name;
	info.type = type;
	info.fields = std::move(fields);
	unions[name] = std::move(info);
}

const JamCodegenContext::UnionInfo *
JamCodegenContext::getUnion(const std::string &name) const {
	auto it = unions.find(name);
	if (it != unions.end()) return &it->second;
	return nullptr;
}

const JamCodegenContext::UnionInfo *
JamCodegenContext::lookupUnion(TypeIdx ty) const {
	if (ty == kNoType) return nullptr;
	ty = requalifyType(ty, currentBodyModule());
	const TypeKey &k = typePool.get(ty);
	// Accept TypeKind::Union (explicit) or TypeKind::Named (parser-
	// deferred user type that resolves to a union).
	if (k.kind != TypeKind::Union && k.kind != TypeKind::Named) {
		return nullptr;
	}
	const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
	if (handlePrivacyBlocked(name)) return nullptr;
	if (const UnionInfo *direct = getUnion(name)) return direct;
	if (name.find('.') != std::string::npos) {
		TypeIdx chained = resolveChainedType(name);
		if (chained != kNoType) return lookupUnion(chained);
	}
	return nullptr;
}

TypeIdx
JamCodegenContext::getUnionFieldType(const std::string &unionName,
                                     const std::string &fieldName) const {
	const UnionInfo *info = getUnion(unionName);
	if (!info) return kNoType;
	for (const auto &f : info->fields) {
		if (f.first == fieldName) return f.second;
	}
	return kNoType;
}

// Enum registry

void JamCodegenContext::registerEnum(
    const std::string &name, std::vector<EnumVariantInfo> variants) const {
	EnumInfo info;
	info.name = name;
	info.variants = std::move(variants);
	for (const auto &v : info.variants) {
		if (!v.payloadTypes.empty()) {
			info.hasPayloadVariant = true;
			break;
		}
	}
	enums[name] = std::move(info);
}

void JamCodegenContext::setEnumLLVMType(const std::string &name,
                                        JamTypeRef llvmType,
                                        uint64_t maxPayloadSize,
                                        uint64_t maxPayloadAlign,
                                        bool hasPayloadVariant) const {
	auto it = enums.find(name);
	if (it == enums.end()) {
		throw std::runtime_error("setEnumLLVMType: unknown enum " + name);
	}
	it->second.type = llvmType;
	it->second.maxPayloadSize = maxPayloadSize;
	it->second.maxPayloadAlign = maxPayloadAlign;
	it->second.hasPayloadVariant = hasPayloadVariant;
}

const JamCodegenContext::EnumInfo *
JamCodegenContext::getEnum(const std::string &name) const {
	auto it = enums.find(name);
	if (it != enums.end()) return &it->second;
	return nullptr;
}

const JamCodegenContext::EnumInfo *
JamCodegenContext::lookupEnum(TypeIdx ty) const {
	if (ty == kNoType) return nullptr;
	ty = requalifyType(ty, currentBodyModule());
	const TypeKey &k = typePool.get(ty);
	// a GenericCall TypeIdx resolves to a concrete type;
	// recurse on the resolved TypeIdx so generic enum instantiations
	// (e.g. `Option(i32)` -> `Option__i32`) resolve uniformly.
	if (k.kind == TypeKind::GenericCall) {
		return lookupEnum(resolveGenericCall(ty));
	}
	// Accept TypeKind::Enum (explicit) or TypeKind::Named (parser-
	// deferred user type that resolves to an enum).
	if (k.kind != TypeKind::Enum && k.kind != TypeKind::Named) {
		return nullptr;
	}
	const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
	if (handlePrivacyBlocked(name)) return nullptr;
	if (const EnumInfo *direct = getEnum(name)) return direct;
	// try the type alias table — `const OptI32 =
	// Option(i32);` maps `OptI32` to the instantiated enum's TypeIdx.
	TypeIdx aliasTarget = lookupTypeAlias(name);
	if (aliasTarget != kNoType) { return lookupEnum(aliasTarget); }
	if (name.find('.') != std::string::npos) {
		TypeIdx chained = resolveChainedType(name);
		if (chained != kNoType) return lookupEnum(chained);
	}
	return nullptr;
}

int JamCodegenContext::getEnumVariantIndex(
    const std::string &enumName, const std::string &variantName) const {
	const EnumInfo *info = getEnum(enumName);
	if (!info) return -1;
	for (size_t i = 0; i < info->variants.size(); i++) {
		if (info->variants[i].name == variantName) {
			return static_cast<int>(i);
		}
	}
	return -1;
}

const JamCodegenContext::EnumInfo *
JamCodegenContext::findEnumByLLVMType(JamTypeRef ty) const {
	for (const auto &kv : enums) {
		if (kv.second.type == ty) return &kv.second;
	}
	return nullptr;
}

void JamCodegenContext::registerModuleConst(const std::string &name,
                                            const std::string &bareName,
                                            const std::string &owner,
                                            NodeIdx init, TypeIdx declared,
                                            bool isComp, std::string file) {
	moduleConsts[name] = ModuleConstInfo{
	    init, declared, isComp, std::move(file), bareName, owner};
	// The seeded-scope cache derives from the const set.
	comptimeSeedCache_.clear();
}

void JamCodegenContext::validateCompConsts() {
	jam::ComptimeEvaluator ev(nodeStore, stringPool, typePool);
	for (const auto &kv : moduleConsts) {
		if (!kv.second.isComp) continue;
		// Evaluate each comp const against ITS OWN module's scope — the
		// initializer may reference sibling consts by bare name, and
		// those must be the owner's, not another module's.
		pushBodyModule(kv.second.owner);
		jam::ComptimeScope scope;
		seedComptimeScope(scope);
		jam::ComptimeValue v = ev.eval(kv.second.initExpr, scope);
		popBodyModule();
		if (v.isNone()) {
			jam::SrcLoc loc{kv.second.file,
			                nodeStore.getLine(kv.second.initExpr)};
			diagnostics().error(
			    loc, "comp const `" + kv.second.bareName +
			             "` must be compile-time evaluable — its "
			             "initializer depends on a runtime value or an "
			             "unsupported construct");
		}
	}
}

// function-AST lookup. main.cpp registers each Jam-defined function
// by source-level name so call codegen can recover the parameter modes
// and route ByPointer-classified Let/Move args through implicit
// address-of at the call site.
void JamCodegenContext::registerFunctionAST(const std::string &name,
                                            const FunctionAST *fn) {
	functionAsts[name] = fn;
}

const FunctionAST *
JamCodegenContext::getFunctionAST(const std::string &name) const {
	// When lowering the body of a generic instantiated from a different
	// module, resolve identifiers against the DEFINING MODULE's
	// namespace first — never the caller's. A body like Vec(T).drop
	// calling `free` must find `pub extern fn free` in
	// std/collections.jam (where Vec lives) regardless of whether the
	// caller module independently imports it.
	//
	// The final global map below holds only NON-import entries: plain fns,
	// struct methods, and generic instantiations registered by mangled name
	// (`Vec__u32.withCapacity`). Import-handle flats (`handle.X`) live in the
	// per-module `moduleImports_` tables instead, so an imported body can't
	// reach the entry module's imports through this fallback.
	if (!bodyModuleStack_.empty()) {
		const std::string &defMod = bodyModuleStack_.back();
		if (!defMod.empty()) {
			auto nsIt = moduleNamespaces_.find(defMod);
			if (nsIt != moduleNamespaces_.end()) {
				auto fIt = nsIt->second.functions.find(name);
				if (fIt != nsIt->second.functions.end()) { return fIt->second; }
			}
		}
	}
	// Qualified handle calls (`std.fmt.print`, `fmt.print`) resolve against
	// the CURRENT body's module imports -- the entry module / entry-defined
	// generics use the "" key. A body never sees another file's imports.
	{
		auto mIt = moduleImports_.find(currentBodyModule());
		if (mIt != moduleImports_.end()) {
			auto qIt = mIt->second.qualFns.find(name);
			if (qIt != mIt->second.qualFns.end()) return qIt->second;
		}
	}
	auto it = functionAsts.find(name);
	return (it == functionAsts.end()) ? nullptr : it->second;
}

void JamCodegenContext::pushBodyModule(const std::string &modulePath) {
	bodyModuleStack_.push_back(modulePath);
}

void JamCodegenContext::popBodyModule() {
	if (!bodyModuleStack_.empty()) bodyModuleStack_.pop_back();
}

const std::string &JamCodegenContext::currentBodyModule() const {
	static const std::string kEmpty;
	return bodyModuleStack_.empty() ? kEmpty : bodyModuleStack_.back();
}

const JamCodegenContext::ImportHandleInfo *
JamCodegenContext::getImportHandle(const std::string &handle) const {
	// Resolve against the CURRENT body's module imports only -- the entry
	// module (and entry-defined generics) use the "" key; an imported body
	// (e.g. Bus.drop in bus.jam, lowered under pushBodyModule("bus")) uses its
	// own key. There is NO cross-module fallback: a handle absent from the
	// declaring module's imports is an error, never resolved against the
	// entry/root module's imports.
	auto mIt = moduleImports_.find(currentBodyModule());
	if (mIt != moduleImports_.end()) {
		auto hIt = mIt->second.handles.find(handle);
		if (hIt != mIt->second.handles.end()) return &hIt->second;
	}
	return nullptr;
}

void JamCodegenContext::registerModuleNamespace(ModuleNamespace ns) {
	std::string key = ns.path;
	moduleNamespaces_[std::move(key)] = std::move(ns);
}

const JamCodegenContext::ModuleNamespace *
JamCodegenContext::getModuleNamespace(const std::string &path) const {
	auto it = moduleNamespaces_.find(path);
	return (it == moduleNamespaces_.end()) ? nullptr : &it->second;
}

// Split `dotted` on `.` into segments. Shared by chained function /
// type resolution.
static std::vector<std::string> splitDotted(const std::string &dotted) {
	std::vector<std::string> segs;
	size_t start = 0;
	for (size_t i = 0; i <= dotted.size(); ++i) {
		if (i == dotted.size() || dotted[i] == '.') {
			segs.push_back(dotted.substr(start, i - start));
			start = i + 1;
		}
	}
	return segs;
}

// Walk `segs[0..segs.size()-1]` through ModuleNamespace re-exports
// (`moduleAliases`), starting from the import handle named by
// `segs[0]`. Returns the leaf namespace + the trailing segment name,
// or `{nullptr, ""}` on any miss. Both `resolveChainedFunction` and
// `resolveChainedType` share this prefix walk and only diverge on how
// they look up the final segment.
struct ChainWalkResult {
	const JamCodegenContext::ModuleNamespace *leaf;
	std::string lastSeg;
};
static ChainWalkResult walkChain(const JamCodegenContext &ctx,
                                 const std::string &dotted) {
	std::vector<std::string> segs = splitDotted(dotted);
	if (segs.size() < 3) return {nullptr, {}};
	const auto *handle = ctx.getImportHandle(segs.front());
	if (!handle) return {nullptr, {}};
	const auto *ns = ctx.getModuleNamespace(handle->modulePath);
	if (!ns) return {nullptr, {}};
	for (size_t i = 1; i + 1 < segs.size(); ++i) {
		auto it = ns->moduleAliases.find(segs[i]);
		if (it == ns->moduleAliases.end()) return {nullptr, {}};
		const TypeKey &k = ctx.getTypePool().get(it->second);
		if (k.kind != TypeKind::Module) return {nullptr, {}};
		const std::string &nextPath =
		    ctx.getStringPool().get(static_cast<StringIdx>(k.a));
		ns = ctx.getModuleNamespace(nextPath);
		if (!ns) return {nullptr, {}};
	}
	return {ns, segs.back()};
}

const FunctionAST *
JamCodegenContext::resolveChainedFunction(const std::string &dotted) const {
	auto r = walkChain(*this, dotted);
	if (!r.leaf) return nullptr;
	auto fit = r.leaf->functions.find(r.lastSeg);
	return (fit == r.leaf->functions.end()) ? nullptr : fit->second;
}

TypeIdx JamCodegenContext::resolveChainedType(const std::string &dotted) const {
	auto r = walkChain(*this, dotted);
	if (!r.leaf) return kNoType;
	auto tit = r.leaf->types.find(r.lastSeg);
	if (tit == r.leaf->types.end()) return kNoType;
	// No-progress guard: when an import handle shares its name with the
	// module's identity (`const m = import("m")`), the chain resolves
	// `m.Inner` to the SAME qualified Named that was asked about. Every
	// caller tail-recurses on the result, so report "nothing further to
	// resolve" instead of handing back the input.
	TypeIdx asNamed = typePool.internNamed(stringPool.intern(dotted));
	if (tit->second == asNamed) return kNoType;
	return tit->second;
}

std::string JamCodegenContext::formatNamespaceLookupError(
    const std::string &kind, const std::string &qualified) const {
	size_t dotPos = qualified.find('.');
	if (dotPos == std::string::npos) {
		return "Unknown " + kind + ": " + qualified;
	}
	std::string handle = qualified.substr(0, dotPos);
	std::string bare = qualified.substr(dotPos + 1);
	const auto *info = getImportHandle(handle);
	if (!info) {
		return "unknown module handle `" + handle + "` in `" + qualified + "`";
	}
	if (info->privateNames.count(bare)) {
		return "symbol `" + bare + "` is not exported from module `" +
		       info->modulePath + "`";
	}
	return "symbol `" + bare + "` does not exist in module `" +
	       info->modulePath + "`";
}

const JamCodegenContext::ModuleConstInfo *
JamCodegenContext::getModuleConst(const std::string &name) const {
	// Resolve the bare spelling through the current module's ownership
	// map to the owner-qualified registry key — a body reads its OWN
	// module's const (or one it destructure-imported), never another
	// module's same-named one.
	auto mIt = typeModuleOf_.find(currentBodyModule());
	if (mIt != typeModuleOf_.end()) {
		auto oIt = mIt->second.find(name);
		if (oIt != mIt->second.end()) {
			auto it = moduleConsts.find(qualifyTypeName(oIt->second, name));
			if (it != moduleConsts.end()) return &it->second;
		}
	}
	// Entry-module bodies fall back to the bare key (entry consts
	// register bare); other modules' unresolved names are NOT consts.
	if (currentBodyModule().empty()) {
		auto it = moduleConsts.find(name);
		if (it != moduleConsts.end()) return &it->second;
	}
	return nullptr;
}

// Size of a type in bytes. Used by union layout computation. The
// numbers assume a 64-bit target — pointers and slice lengths are 8
// bytes. Struct sizes do not currently account for inter-field padding;
// callers needing exact struct sizes should ask LLVM via the data
// layout instead. For union fields the simple sum is enough because we
// pick the field with the largest size as the layout type.
namespace {
// Round `off` up to the next multiple of `align`. `align` must be a power of
// two (which every Jam alignment value is, by construction). Used by all
// aggregate-size paths so jam's typeSize matches LLVM's getTypeAllocSize for
// the same struct/union/enum body.
inline uint64_t alignUp(uint64_t off, uint64_t align) {
	return (off + align - 1) / align * align;
}
}  // namespace

uint64_t JamCodegenContext::typeSize(TypeIdx ty) const {
	ty = requalifyType(ty, currentBodyModule());
	// Layout never changes for a concrete requalified type, and the
	// recursive aggregate walk below re-asks for every field on every
	// query — memoize per TypeIdx. Subst contexts bypass (a `T` field
	// sizes differently per instantiation).
	bool cacheable = currentSubst_.empty();
	if (cacheable) {
		auto it = sizeMemo_.find(ty);
		if (it != sizeMemo_.end()) return it->second;
	}
	uint64_t r = typeSizeImpl(ty);
	if (cacheable) sizeMemo_.emplace(ty, r);
	return r;
}

uint64_t JamCodegenContext::typeSizeImpl(TypeIdx ty) const {
	const TypeKey &k = typePool.get(ty);
	switch (k.kind) {
	case TypeKind::Invalid:
	case TypeKind::Void:
	case TypeKind::NoReturn:
		return 0;
	case TypeKind::Bool:
		return 1;
	case TypeKind::Int:
	case TypeKind::Float:
		return k.a / 8;
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
		return 8;
	case TypeKind::Slice:
		return 16;  // (ptr, len)
	case TypeKind::Array:
		return static_cast<uint64_t>(k.b) * typeSize(static_cast<TypeIdx>(k.a));
	case TypeKind::ArrayExpr:
		return typeSize(resolveArrayExpr(ty));
	case TypeKind::Struct:
	case TypeKind::Named: {
		// a Named type may be a substitution-context
		// reference to a parameter (T -> i32) or to Self. Resolve
		// through the substitution map first; if found and the
		// target is a primitive (Int/Float/etc.), the recursive
		// typeSize handles it. Same shape as getLLVMType.
		const std::string &substName =
		    stringPool.get(static_cast<StringIdx>(k.a));
		if (TypeIdx subTarget = lookupCurrentSubst(substName);
		    subTarget != kNoType) {
			return typeSize(subTarget);
		}
		// User-named types resolve through any of the three registries.
		if (const StructInfo *info = lookupStruct(ty)) {
			// Mirror LLVM's struct layout: each field starts at the next
			// multiple of its own alignment, and the struct's total size
			// is padded to its overall alignment so arrays-of-struct
			// land elements at correctly aligned offsets. The old plain
			// sum disagreed with LLVM for any struct containing a
			// less-aligned trailing field — e.g. `{i64, i8}` would
			// report 9 bytes while LLVM lays it out as 16.
			uint64_t off = 0, maxAlign = 1;
			for (const auto &f : info->fields) {
				uint64_t a = typeAlign(f.second);
				off = alignUp(off, a);
				off += typeSize(f.second);
				if (a > maxAlign) maxAlign = a;
			}
			return alignUp(off, maxAlign);
		}
		if (const UnionInfo *info = lookupUnion(ty)) {
			uint64_t maxSize = 0, maxAlign = 1;
			for (const auto &f : info->fields) {
				uint64_t s = typeSize(f.second);
				uint64_t a = typeAlign(f.second);
				if (s > maxSize) maxSize = s;
				if (a > maxAlign) maxAlign = a;
			}
			return alignUp(maxSize, maxAlign);
		}
		if (const EnumInfo *info = lookupEnum(ty)) {
			if (!info->hasPayloadVariant) return 1;
			// Layout matches `fillEnumBodies` in main.cpp: an
			// `{i8 tag, alignDriver, [extraBytes x i8]}` named struct
			// where alignDriver is the scalar whose alignment == enum
			// alignment. LLVM places alignDriver at offset maxAlign
			// (padding the tag out), then the optional trailing array,
			// then pads the whole thing to maxAlign. The old formula
			// `1 + (align-1) + maxPayloadSize` under-counted when the
			// payload size wasn't already a multiple of the alignment.
			uint64_t mPA = info->maxPayloadAlign;
			uint64_t paddedPayload = alignUp(info->maxPayloadSize, mPA);
			if (paddedPayload == 0) return 2 * mPA;
			return mPA + paddedPayload;
		}
		// alias lookup fallback, same as typeAlign.
		if (TypeIdx aliasTarget = lookupTypeAlias(substName);
		    aliasTarget != kNoType) {
			return typeSize(aliasTarget);
		}
		throw std::runtime_error(
		    "typeSize: " +
		    formatNamespaceLookupError("user-defined type", substName));
	}
	case TypeKind::Union: {
		const UnionInfo *info = lookupUnion(ty);
		if (!info) {
			const std::string &name =
			    stringPool.get(static_cast<StringIdx>(k.a));
			throw std::runtime_error("typeSize: " +
			                         formatNamespaceLookupError("union", name));
		}
		uint64_t maxSize = 0, maxAlign = 1;
		for (const auto &f : info->fields) {
			uint64_t s = typeSize(f.second);
			uint64_t a = typeAlign(f.second);
			if (s > maxSize) maxSize = s;
			if (a > maxAlign) maxAlign = a;
		}
		return alignUp(maxSize, maxAlign);
	}
	case TypeKind::Enum:
		return 1;  // Unit-only enums lower to u8
	case TypeKind::Type:
		// Meta-type has no runtime size.
		return 0;
	case TypeKind::GenericCall:
		// resolve and recurse.
		return typeSize(resolveGenericCall(ty));
	case TypeKind::Fn:
		// Function value = code pointer = pointer width.
		return 8;
	case TypeKind::Module:
		// Modules are compile-time-only values.
		return 0;
	}
	throw std::runtime_error("typeSize: unhandled type kind");
}

// Alignment requirement of a type. Equal to size for primitive scalars
// on every target we care about. For aggregates, the alignment is the
// max of the constituent alignments.
uint64_t JamCodegenContext::typeAlign(TypeIdx ty) const {
	ty = requalifyType(ty, currentBodyModule());
	bool cacheable = currentSubst_.empty();
	if (cacheable) {
		auto it = alignMemo_.find(ty);
		if (it != alignMemo_.end()) return it->second;
	}
	uint64_t r = typeAlignImpl(ty);
	if (cacheable) alignMemo_.emplace(ty, r);
	return r;
}

uint64_t JamCodegenContext::typeAlignImpl(TypeIdx ty) const {
	const TypeKey &k = typePool.get(ty);
	switch (k.kind) {
	case TypeKind::Invalid:
	case TypeKind::Void:
	case TypeKind::NoReturn:
		return 1;
	case TypeKind::Bool:
		return 1;
	case TypeKind::Int:
	case TypeKind::Float:
		return k.a / 8;
	case TypeKind::PtrSingle:
	case TypeKind::PtrMany:
	case TypeKind::Slice:
		return 8;
	case TypeKind::Array:
		return typeAlign(static_cast<TypeIdx>(k.a));
	case TypeKind::ArrayExpr:
		return typeAlign(resolveArrayExpr(ty));
	case TypeKind::Struct:
	case TypeKind::Named: {
		// substitution context wins. A Named type may
		// be a parameter reference (T -> i32) or Self that resolves
		// to a non-aggregate; the recursive call handles primitives.
		const std::string &substName =
		    stringPool.get(static_cast<StringIdx>(k.a));
		if (TypeIdx subTarget = lookupCurrentSubst(substName);
		    subTarget != kNoType) {
			return typeAlign(subTarget);
		}
		if (const StructInfo *info = lookupStruct(ty)) {
			uint64_t maxAlign = 1;
			for (const auto &f : info->fields) {
				uint64_t a = typeAlign(f.second);
				if (a > maxAlign) maxAlign = a;
			}
			return maxAlign;
		}
		if (const UnionInfo *info = lookupUnion(ty)) {
			uint64_t maxAlign = 1;
			for (const auto &f : info->fields) {
				uint64_t a = typeAlign(f.second);
				if (a > maxAlign) maxAlign = a;
			}
			return maxAlign;
		}
		if (const EnumInfo *info = lookupEnum(ty)) {
			return info->hasPayloadVariant ? info->maxPayloadAlign : 1;
		}
		// a Named TypeIdx may be a type alias produced by
		// `const Foo = Bar(args);`. Resolve through the alias table
		// and recurse — matches lookupStruct's behavior.
		if (TypeIdx aliasTarget = lookupTypeAlias(substName);
		    aliasTarget != kNoType) {
			return typeAlign(aliasTarget);
		}
		throw std::runtime_error(
		    "typeAlign: " +
		    formatNamespaceLookupError("user-defined type", substName));
	}
	case TypeKind::Union: {
		const UnionInfo *info = lookupUnion(ty);
		if (!info) {
			const std::string &name =
			    stringPool.get(static_cast<StringIdx>(k.a));
			throw std::runtime_error("typeAlign: " +
			                         formatNamespaceLookupError("union", name));
		}
		uint64_t maxAlign = 1;
		for (const auto &f : info->fields) {
			uint64_t a = typeAlign(f.second);
			if (a > maxAlign) maxAlign = a;
		}
		return maxAlign;
	}
	case TypeKind::Enum:
		return 1;
	case TypeKind::Type:
		return 1;
	case TypeKind::GenericCall:
		return typeAlign(resolveGenericCall(ty));
	case TypeKind::Fn:
		// Function pointer alignment.
		return 8;
	case TypeKind::Module:
		return 1;
	}
	throw std::runtime_error("typeAlign: unhandled type kind");
}

// substitution engine for `Identifier(arg, ...)` types.

namespace {

// Recursively rewrite a TypeIdx, replacing parameter Named-types with their
// bound concrete TypeIdx. Other compound types (pointers, slices, arrays,
// nested generic calls) are reconstructed with substituted children.
TypeIdx substituteType(TypeIdx ty,
                       const std::unordered_map<std::string, TypeIdx> &subst,
                       TypePool &types, const StringPool &strings) {
	const TypeKey &k = types.get(ty);
	switch (k.kind) {
	case TypeKind::Named: {
		const std::string &name = strings.get(static_cast<StringIdx>(k.a));
		auto it = subst.find(name);
		if (it != subst.end()) return it->second;
		return ty;
	}
	case TypeKind::PtrSingle:
		return types.internPtrSingle(
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings));
	case TypeKind::PtrMany:
		return types.internPtrMany(
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings));
	case TypeKind::Slice:
		return types.internSlice(
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings));
	case TypeKind::Array:
		return types.internArray(
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings),
		    k.b);
	case TypeKind::ArrayExpr:
		// Substitute the element type; the length expression is
		// type-free and rides along unchanged.
		return types.internArrayExpr(
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings),
		    static_cast<NodeIdx>(k.b));
	case TypeKind::GenericCall: {
		// Recurse into args: a generic call inside a generic body
		// (e.g. `Box(Maybe(T))`) substitutes T in the inner call.
		const auto &args = types.genericArgsAt(k.b);
		std::vector<TypeIdx> newArgs;
		newArgs.reserve(args.size());
		for (TypeIdx a : args) {
			newArgs.push_back(substituteType(a, subst, types, strings));
		}
		return types.internGenericCall(static_cast<StringIdx>(k.a),
		                               std::move(newArgs));
	}
	case TypeKind::Fn: {
		// Function-typed value: substitute the return type AND every
		// param type. A `fn(T) T` field inside a generic struct sees
		// T -> concrete on instantiation, so the field's TypeIdx must
		// rebuild with substituted children.
		TypeIdx retSub =
		    substituteType(static_cast<TypeIdx>(k.a), subst, types, strings);
		const auto &params = types.fnParamsAt(k.b);
		std::vector<TypeIdx> newParams;
		newParams.reserve(params.size());
		for (TypeIdx p : params) {
			newParams.push_back(substituteType(p, subst, types, strings));
		}
		return types.internFn(retSub, std::move(newParams));
	}
	default:
		return ty;
	}
}

// Stable spelling of a generic argument type, used to build the
// instantiated name (`Vec__i32`, `Holder__lib/m.Inner`). The spelling
// is the memoization key, so every distinct argument type MUST spell
// distinctly — a shared catch-all would silently hand `Pair(f64)` the
// struct layout already instantiated for `Pair(f32)`. Compound types
// spell recursively (`Vec(*mut u8)` -> `Vec__pm_u8`).
static std::string genericArgSpelling(const TypePool &typePool,
                                      const StringPool &stringPool, TypeIdx t) {
	const TypeKey ak = typePool.get(t);
	switch (ak.kind) {
	case TypeKind::Int: {
		char buf[16];
		std::snprintf(buf, sizeof(buf), "%c%u", ak.b ? 'i' : 'u', ak.a);
		return buf;
	}
	case TypeKind::Bool:
		return "bool";
	case TypeKind::Float:
		return ak.a == 32 ? "f32" : "f64";
	case TypeKind::Struct:
	case TypeKind::Named:
	case TypeKind::Enum:
	case TypeKind::Union:
		return stringPool.get(static_cast<StringIdx>(ak.a));
	case TypeKind::PtrSingle:
		return "p_" + genericArgSpelling(typePool, stringPool,
		                                 static_cast<TypeIdx>(ak.a));
	case TypeKind::PtrMany:
		return "pm_" + genericArgSpelling(typePool, stringPool,
		                                  static_cast<TypeIdx>(ak.a));
	case TypeKind::Slice:
		return "sl_" + genericArgSpelling(typePool, stringPool,
		                                  static_cast<TypeIdx>(ak.a));
	case TypeKind::Array:
		return "arr" + std::to_string(ak.b) + "_" +
		       genericArgSpelling(typePool, stringPool,
		                          static_cast<TypeIdx>(ak.a));
	default:
		// No stable spelling — include the TypeIdx so distinct types
		// never share an instantiation.
		return "T" + std::to_string(t);
	}
}

}  // namespace

TypeIdx JamCodegenContext::resolveGenericCall(TypeIdx callTy) const {
	// Qualify the callee + args to their owning modules first (no-op for
	// already-qualified or entry-owned names). Some callers requalify
	// before calling; the ones that can't (jir_verify's hook) rely on
	// this. Without it the per-TypeIdx caches below would conflate two
	// modules' same-named generics.
	callTy = requalifyType(callTy, currentBodyModule());
	// Apply the active substitution context to the GenericCall's args
	// before resolving. The same `Inner(T)` TypeIdx appears in every
	// instantiation of an outer generic that mentions it — Wrap(i32)
	// must see `[i32]`, Wrap(f32) must see `[f32]`. Without this
	// substitution the args stay as the placeholder Named("T") and
	// the inner instantiation either resolves to a meaningless
	// Inner__T or loops because the placeholder never collapses.
	//
	// The cache key is the *substituted* GenericCall TypeIdx so two
	// outer substitutions don't conflate.
	const TypeKey &k0 = typePool.get(callTy);
	const auto &rawArgs = typePool.genericArgsAt(k0.b);
	TypeIdx effectiveTy = callTy;
	if (!currentSubst_.empty()) {
		std::vector<TypeIdx> substArgs;
		substArgs.reserve(rawArgs.size());
		bool anyChanged = false;
		for (TypeIdx a : rawArgs) {
			TypeIdx s = substituteType(a, currentSubst_, typePool, stringPool);
			if (s != a) anyChanged = true;
			substArgs.push_back(s);
		}
		if (anyChanged) {
			effectiveTy = typePool.internGenericCall(
			    static_cast<StringIdx>(k0.a), std::move(substArgs));
		}
	}

	auto cached = genericResolutions_.find(effectiveTy);
	if (cached != genericResolutions_.end()) return cached->second;

	const TypeKey &k = typePool.get(effectiveTy);
	const std::string calleeName = stringPool.get(static_cast<StringIdx>(k.a));

	// Canonicalize the args to their owner identities so the identity
	// cache and the instantiated name don't depend on how the caller
	// SPELLED an argument: a handle-qualified `P.Pair` resolves through
	// the scoped alias table to the same `lib/p.Pair` a requalified
	// bare reference produces — one instantiation, not two.
	std::vector<TypeIdx> args = typePool.genericArgsAt(k.b);
	for (TypeIdx &a : args) {
		const TypeKey &ak = typePool.get(a);
		if (ak.kind != TypeKind::Named) continue;
		const std::string &an = stringPool.get(static_cast<StringIdx>(ak.a));
		if (an.find('.') == std::string::npos) continue;
		TypeIdx aliased = lookupTypeAlias(an);
		if (aliased == kNoType) {
			TypeIdx chained = resolveChainedType(an);
			if (chained != kNoType) aliased = chained;
		}
		if (aliased != kNoType && aliased != a) {
			a = requalifyType(aliased, currentBodyModule());
		}
	}

	// Demand-driven: consult the Analyzer first so cycle detection
	// kicks in and any cross-decl dependency the user introduced is
	// recorded. The analyzer's stub for Function decls just confirms
	// the FunctionAST is reachable — the heavy lifting (resolving
	// args, instantiating the struct) stays here. Step 4 will hoist
	// the instantiation logic into the analyzer itself.
	jam::DeclIndex calleeDecl = declTable_.findByName(calleeName);
	if (calleeDecl != jam::kNoDecl) {
		analyzer().ensureDeclAnalyzed(calleeDecl);
	}

	// Identity-based lookup: namespace-qualified and bare callees both
	// resolve to the same FunctionAST (main.cpp registers both forms).
	// Look it up by whichever name the caller used.
	const FunctionAST *generic = getFunctionAST(calleeName);
	if (!generic && calleeDecl != jam::kNoDecl) {
		// Demand-driven resolution: the eager import pass may
		// not have reached this generic's defining module yet
		// (getLoadedModules() is an unordered_map), so resolve it on
		// reference straight from the decl index, which registerTopLevelDecls
		// populated for every imported module before any prototype is declared.
		generic = declTable_.get(calleeDecl).fnAst;
	}
	if (!generic) {
		throw std::runtime_error(
		    formatNamespaceLookupError("generic", calleeName));
	}
	if (!generic->isGeneric()) {
		throw std::runtime_error(
		    "Identifier `" + calleeName +
		    "` is a non-generic function used in a type position");
	}

	// Identity cache: if the same (FunctionAST, concrete args) has
	// already been instantiated via a different syntactic path, reuse
	// that result. Keeps `c.Vec(i32)` and `Vec(i32)` pointing at the
	// same Vec__i32 struct.
	GenericInstanceKey idKey{generic, std::vector<TypeIdx>(args)};
	auto idCached = genericInstances_.find(idKey);
	if (idCached != genericInstances_.end()) {
		genericResolutions_[effectiveTy] = idCached->second;
		return idCached->second;
	}
	if (args.size() != generic->Args.size()) {
		throw std::runtime_error("Generic `" + calleeName + "` expects " +
		                         std::to_string(generic->Args.size()) +
		                         " type argument(s), got " +
		                         std::to_string(args.size()));
	}

	// v1 only supports `T: type` parameters (no comptime values yet).
	for (size_t i = 0; i < generic->Args.size(); i++) {
		if (generic->Args[i].Type != BuiltinType::Type) {
			throw std::runtime_error(
			    "Generic `" + calleeName +
			    "` has a non-type parameter (comptime values are v2)");
		}
	}

	// Build the substitution map from parameter names to concrete args.
	std::unordered_map<std::string, TypeIdx> subst;
	for (size_t i = 0; i < generic->Args.size(); i++) {
		subst[generic->Args[i].Name] = args[i];
	}

	// Walk the function body looking for the return statement. v1
	// supports two return shapes: (1) `return T;` where T is a type
	// parameter or a named type, and (2) `return struct {...};` where
	// the body declares the instantiated struct's fields.
	TypeIdx result = kNoType;
	for (NodeIdx stmt : generic->Body) {
		const AstNode &n = nodeStore.get(stmt);
		if (n.tag != AstTag::Return) continue;
		NodeIdx valueIdx = static_cast<NodeIdx>(n.lhs);
		const AstNode &value = nodeStore.get(valueIdx);
		if (value.tag == AstTag::Variable) {
			const std::string &name =
			    stringPool.get(static_cast<StringIdx>(value.lhs));
			auto it = subst.find(name);
			if (it != subst.end()) {
				result = it->second;
				break;
			}
			// Not a parameter — treat as a named type reference and
			// substitute through (handles forwarding generics that
			// return a non-parameter named type).
			TypeIdx asNamed = typePool.internNamed(stringPool.intern(name));
			result = substituteType(asNamed, subst, typePool, stringPool);
			break;
		}
		if (value.tag == AstTag::StructExpr) {
			// The instantiated struct's name starts from the generic's
			// OWNER-QUALIFIED identity (`lib/a.Pair__i32`), not the
			// syntactic spelling: prefixes like `c.Vec` converge on the
			// owner, while two modules' same-named factories stay
			// distinct. Pass the generic's modulePath so the cloned
			// methods and the anon struct's member types resolve against
			// the DEFINING module — anon-struct bodies returned from
			// `pub fn Vec(T)` aren't visible to module_resolver, so the
			// originating module is propagated here.
			result = instantiateStructExpr(
			    value, qualifyTypeName(generic->modulePath, generic->Name),
			    args, subst, generic->modulePath);
			break;
		}
		if (value.tag == AstTag::EnumExpr) {
			result = instantiateEnumExpr(
			    value, qualifyTypeName(generic->modulePath, generic->Name),
			    args, subst, generic->modulePath);
			break;
		}
		throw std::runtime_error(
		    "Generic body's return value shape not supported in v1 "
		    "(only `return T;` or `return struct {...};` are "
		    "implemented)");
	}

	if (result == kNoType) {
		throw std::runtime_error("Generic `" + calleeName +
		                         "` has no return statement to evaluate");
	}

	genericInstances_[idKey] = result;
	genericResolutions_[effectiveTy] = result;
	return result;
}

void JamCodegenContext::seedComptimeScope(jam::ComptimeScope &scope) const {
	// Fold the consts VISIBLE TO THE CURRENT MODULE into scope, bound
	// by their bare source spelling: a body references `N`, and `N`
	// must be its own module's (or a destructure-imported) const —
	// never another module's same-named one, and never a private const
	// of an unrelated module. Consts may reference each other
	// (`const B = A * 2;`), so seed to a fixpoint: each pass folds the
	// consts whose dependencies folded in an earlier pass. Consts that
	// never fold (runtime-only inits) simply stay unbound — an
	// expression referencing one comes back None.
	//
	// The fixpoint re-evaluates const initializer ASTs and used to run
	// for EVERY function body and every comptime fold — cache the
	// seeded scope per module (consts are immutable once registered;
	// registerModuleConst clears the cache).
	const std::string &cur = currentBodyModule();
	auto cacheIt = comptimeSeedCache_.find(cur);
	if (cacheIt == comptimeSeedCache_.end()) {
		jam::ComptimeScope seeded;
		jam::ComptimeEvaluator ev(nodeStore, stringPool, typePool);
		auto ownIt = typeModuleOf_.find(cur);
		auto visible = [&](const ModuleConstInfo &info) -> bool {
			if (ownIt != typeModuleOf_.end()) {
				auto oIt = ownIt->second.find(info.bareName);
				if (oIt != ownIt->second.end())
					return oIt->second == info.owner;
			}
			// Entry bodies additionally see entry consts that predate
			// the ownership map (bare-registered, owner "").
			return cur.empty() && info.owner.empty();
		};
		bool progress = true;
		while (progress) {
			progress = false;
			for (const auto &kv : moduleConsts) {
				if (!visible(kv.second)) continue;
				if (seeded.lookup(kv.second.bareName) != nullptr) continue;
				jam::ComptimeValue v = ev.eval(kv.second.initExpr, seeded);
				if (!v.isNone()) {
					seeded.bind(kv.second.bareName, v);
					progress = true;
				}
			}
		}
		cacheIt = comptimeSeedCache_.emplace(cur, std::move(seeded)).first;
	}
	cacheIt->second.copyBindingsInto(scope);
	// Active comp-param substitutions shadow same-named module consts —
	// the same precedence astgenVariable applies (comp subst is checked
	// before getModuleConst). Bound last so they overwrite.
	for (const auto &kv : currentCompSubst_) {
		scope.bind(kv.first, kv.second);
	}
}

jam::ComptimeValue
CompCallResolverImpl::resolveCall(const std::string &name,
                                  const std::vector<jam::ComptimeValue> &args,
                                  jam::Diagnostics &diags, jam::SrcLoc loc) {
	const FunctionAST *fn = ctx_.getFunctionAST(name);
	// Only compile-time functions are callable from comptime in v1.
	// A plain `fn` that EXISTS is a definitive misuse — say so. An
	// unknown name returns None silently (the surrounding context, e.g.
	// "array length must be comptime-known", surfaces it). Plain-fn
	// CTFE is Stage 7.
	if (fn != nullptr && !fn->isCompTimeFn) {
		diags.error(loc, "`" + name +
		                     "` is a runtime function and cannot be called "
		                     "at compile time — only `cfn` functions are "
		                     "comptime-evaluable");
		return jam::ComptimeValue::makeNone();
	}
	if (fn == nullptr) { return jam::ComptimeValue::makeNone(); }
	if (args.size() != fn->Args.size()) {
		diags.error(loc, "cfn `" + name + "` expects " +
		                     std::to_string(fn->Args.size()) + " arg(s), got " +
		                     std::to_string(args.size()));
		return jam::ComptimeValue::makeNone();
	}
	if (++totalCalls_ > kMaxTotalCalls) {
		diags.error(loc, "comptime total-call cap (" +
		                     std::to_string(kMaxTotalCalls) +
		                     ") exceeded — comptime evaluation does too much "
		                     "work (runaway recursion?)");
		return jam::ComptimeValue::makeNone();
	}
	if (++depth_ > kMaxDepth) {
		diags.error(loc, "cfn call depth cap (" + std::to_string(kMaxDepth) +
		                     ") exceeded — possible infinite recursion in `" +
		                     name + "`");
		depth_--;
		return jam::ComptimeValue::makeNone();
	}

	// Fresh scope: a cfn body sees only its parameters (lexical, not
	// dynamic — it cannot reach the caller's locals by name).
	jam::ComptimeScope callScope;
	for (std::size_t i = 0; i < args.size(); i++) {
		callScope.bind(fn->Args[i].Name, args[i]);
	}
	uint32_t iter = 0;
	jam::ComptimeValue ret;
	std::vector<NodeIdx> body(fn->Body.begin(), fn->Body.end());
	jam::ExecResult r =
	    ev_.execBlock(body.data(), body.size(), callScope, iter,
	                  jam::ComptimeEvaluator::kDefaultIterCap, ret, diags, loc);
	depth_--;
	if (r == jam::ExecResult::Error || r == jam::ExecResult::IterationCap) {
		return jam::ComptimeValue::makeNone();
	}
	// A `return;` with no value, or fall-through, yields None — the
	// caller decides whether a value was required.
	return ret;
}

jam::ComptimeValue JamCodegenContext::foldComptimeExpr(NodeIdx expr) const {
	jam::ComptimeEvaluator ev(nodeStore, stringPool, typePool);
	jam::ComptimeScope scope;
	seedComptimeScope(scope);
	// Install a call resolver so a cfn can appear in a comptime
	// expression position — `[bufLen(512)]u8`, `comp const N =
	// f(3)`. Diagnostics route to this context's channel; the
	// SrcLoc is best-effort (the expr's own line is not threaded here).
	CompCallResolverImpl resolver(*this, ev);
	ev.setCallResolver(&resolver);
	jam::SrcLoc loc{currentFile(), nodeStore.getLine(expr)};
	ev.setCallContext(nullptr, &diagnostics(), loc);
	jam::ComptimeValue v = ev.eval(expr, scope);
	ev.clearCallContext();
	ev.clearCallResolver();
	return v;
}

TypeIdx JamCodegenContext::resolveArrayExpr(TypeIdx ty) const {
	auto cached = arrayExprResolutions_.find(ty);
	if (cached != arrayExprResolutions_.end()) return cached->second;

	const TypeKey &k = typePool.get(ty);
	jam::ComptimeValue len = foldComptimeExpr(static_cast<NodeIdx>(k.b));
	if (len.isNone()) {
		throw std::runtime_error("array length must be comptime-known");
	}
	if (!len.isInt() || (len.intVal.isSigned && len.asI64() < 0)) {
		throw std::runtime_error("array size must be a non-negative integer");
	}
	uint64_t lenVal = len.asU64();
	if (lenVal > UINT32_MAX) {
		// A negative fold wraps to a huge unsigned value; report it as
		// the sign error it is rather than a baffling range overflow.
		if (static_cast<int64_t>(lenVal) < 0) {
			throw std::runtime_error(
			    "array size must be a non-negative integer");
		}
		throw std::runtime_error("array size " + std::to_string(lenVal) +
		                         " exceeds u32 range");
	}

	TypeIdx resolved = typePool.internArray(static_cast<TypeIdx>(k.a),
	                                        static_cast<uint32_t>(lenVal));
	arrayExprResolutions_.emplace(ty, resolved);
	return resolved;
}

// Instantiate a `struct {...}` expression appearing in a generic body's
// return statement. Substitutes each field's TypeIdx with the concrete
// generic args, creates a fresh LLVM struct type with a unique name, and
// returns a Named TypeIdx pointing at the new struct. Methods are not
// instantiated in v1.
TypeIdx JamCodegenContext::instantiateStructExpr(
    const AstNode &exprNode, const std::string &calleeName,
    const std::vector<TypeIdx> &args,
    const std::unordered_map<std::string, TypeIdx> &subst,
    const std::string &definingModulePath_) const {
	if (!anonStructs_) {
		throw std::runtime_error(
		    "internal: anonymous struct table not registered on "
		    "codegen context");
	}
	uint32_t anonIdx = exprNode.lhs;
	if (anonIdx >= anonStructs_->size()) {
		throw std::runtime_error(
		    "internal: StructExpr references missing AnonStructs[" +
		    std::to_string(anonIdx) + "]");
	}
	const StructDeclAST *anon = (*anonStructs_)[anonIdx].get();

	// Build the instantiated struct's name from the callee's qualified
	// identity + arg spellings: `Maybe(File)` -> `Maybe__std/fs.File`.
	// See genericArgSpelling for why every distinct arg type must spell
	// distinctly.
	std::string instName = calleeName;
	for (TypeIdx a : args) {
		instName += "__";
		instName += genericArgSpelling(typePool, stringPool, a);
	}

	// Memoize on the instantiated name. If we've already produced this
	// struct, return its TypeIdx without re-creating the LLVM type.
	if (const StructInfo *existing = getStruct(instName)) {
		(void)existing;
		return typePool.internNamed(stringPool.intern(instName));
	}

	// Build the full substitution map: parameter names -> concrete args,
	// plus the anon-struct's synthetic name (which is what `Self`
	// resolved to in *type* positions at parse time) -> the new
	// instantiated struct's Named TypeIdx. We also alias the literal
	// string "Self" to the same target so codegen sites that see
	// the parser's stringified `Self.method(...)` (an expression-
	// position member access on the Self identifier) can resolve
	// it via the same map. Used for field types, method signatures,
	// and method body codegen.
	std::unordered_map<std::string, TypeIdx> bodySubst = subst;
	TypeIdx instNamed = typePool.internNamed(stringPool.intern(instName));
	bodySubst[anon->Name] = instNamed;
	bodySubst["Self"] = instNamed;

	// Substitute each field's type, then declare + fill the LLVM struct.
	// After substitution, qualify against the DEFINING module: a field
	// like `inner: Inner` names the generic's sibling type, not whatever
	// `Inner` happens to mean at the trigger site. Anon-struct bodies
	// never pass through bindDeclTypes, so this is their decl-time
	// qualification.
	std::vector<std::pair<std::string, TypeIdx>> instFields;
	instFields.reserve(anon->Fields.size());
	for (const auto &f : anon->Fields) {
		TypeIdx subbed =
		    substituteType(f.second, bodySubst, typePool, stringPool);
		instFields.emplace_back(f.first,
		                        requalifyType(subbed, definingModulePath_));
	}

	JamTypeRef llvmStruct =
	    JamLLVMStructCreateNamed(getContext(), instName.c_str());
	registerStruct(instName, llvmStruct, instFields);

	// Field-type lowering and Pass 1 signature declaration resolve in
	// the generic's DEFINING module scope — a field naming the owner's
	// private sibling type must not trip the trigger-site module's
	// handle-privacy gate. (Pass 2 bodies push the same module
	// per-method below.)
	const_cast<JamCodegenContext &>(*this).pushBodyModule(definingModulePath_);

	std::vector<JamTypeRef> fieldLLVM;
	fieldLLVM.reserve(instFields.size());
	for (const auto &f : instFields) {
		fieldLLVM.push_back(getLLVMType(f.second));
	}
	JamLLVMStructSetBody(llvmStruct, fieldLLVM.data(),
	                     static_cast<unsigned>(fieldLLVM.size()), false);

	// instantiate methods. Two passes so methods on the
	// same struct can call each other regardless of declaration order
	// — the first pass clones + registers + declares LLVM prototypes
	// for every method (so any later self.method() lookup succeeds);
	// the second pass defines bodies (which may emit calls to
	// other-method prototypes registered in pass 1). Without the
	// split, a method body that calls another method declared later
	// in the struct body emits "Unknown function referenced: ..."
	// because the callee's prototype isn't yet in the LLVM module.
	if (!anon->Methods.empty()) {
		struct InstMethod {
			FunctionAST *clonePtr;
			// Carries Pass 1's metadata JirFunction so Pass 2 can
			// append the body without redoing param-mode + return-
			// type lookups.
			JirFunction passOneJir;
		};
		std::vector<InstMethod> insts;
		insts.reserve(anon->Methods.size());

		JamCodegenContext &mutCtx = const_cast<JamCodegenContext &>(*this);

		// Pass 1: clone + register + jirDeclarePrototype for every method.
		// Signature types qualify against the DEFINING module after
		// substitution (same as the field types above): a method like
		// `fn makeInner(self: Self) Inner` names the generic's sibling
		// type regardless of what the trigger site calls `Inner`.
		for (const auto &origMethod : anon->Methods) {
			std::vector<Param> instArgs;
			instArgs.reserve(origMethod->Args.size());
			for (const auto &p : origMethod->Args) {
				Param sp = p;
				sp.Type = requalifyType(
				    substituteType(p.Type, bodySubst, typePool, stringPool),
				    definingModulePath_);
				instArgs.push_back(std::move(sp));
			}
			TypeIdx instReturn = origMethod->ReturnType;
			if (instReturn != kNoType) {
				instReturn = requalifyType(
				    substituteType(instReturn, bodySubst, typePool, stringPool),
				    definingModulePath_);
			}

			std::string instMethodName = instName + "." + origMethod->Name;
			auto cloned = std::make_unique<FunctionAST>(
			    instMethodName, std::move(instArgs), instReturn,
			    origMethod->Body, origMethod->isExtern, origMethod->isExport,
			    origMethod->isPub, origMethod->isTest, origMethod->isVarArgs,
			    origMethod->isCfn);
			// Note: modulePath stays empty on the clone — stamping it
			// would alter the mangled symbol name (mangling.h:60
			// includes modulePath in the FQN). Body-scope identifier
			// resolution is carried separately via bodyModuleStack_,
			// pushed at the astgenBodyInto call site below.
			FunctionAST *clonePtr = cloned.get();
			instantiatedMethods_.push_back(std::move(cloned));

			if (origMethod->Name == "drop" && origMethod->Args.size() == 1 &&
			    origMethod->Args[0].Name == "self" &&
			    origMethod->Args[0].Mode == ParamMode::Mut) {
				instantiatedDrops_[instName] = clonePtr;
			}

			mutCtx.registerFunctionAST(instMethodName, clonePtr);
			// Declarations need the substitution context for any
			// nested type expressions in the signature.
			setCurrentSubst(bodySubst);
			// Pass 1 builds a signature-only JirFunction and emits
			// the prototype with JIR's ABI (mut/move -> ptr). The
			// metadata is cached on the InstMethod entry so Pass 2
			// can continue from here instead of rebuilding it.
			JirFunction passOneJir = astgenMetadata(*clonePtr, mutCtx);
			passOneJir.name = clonePtr->Name;
			jirDeclarePrototype(passOneJir, mutCtx);
			clearCurrentSubst();
			insts.push_back({clonePtr, std::move(passOneJir)});
		}

		// Pass 2: define bodies. All methods are now declared, so
		// `self.method()` calls between them resolve cleanly. The
		// cloned bodies go through astgen + JIR codegen — the same
		// typed pipeline main-module functions use, so generics
		// aren't a second-class path. Save the builder's insertion
		// block before each `jirDefineBody` re-positions it, so the
		// caller (which may be mid-emission of the trigger expression
		// that asked for instantiation) finds the builder where it
		// left off.
		JamBasicBlockRef savedBB = JamLLVMGetInsertBlock(getBuilder());
		for (auto &im : insts) {
			setCurrentSubst(bodySubst);
			// Push a reference-trace frame so any astgen diagnostic
			// raised while lowering this instantiation's body is
			// annotated with the chain "in instantiation of
			// `instName.method`". The frame is popped automatically
			// when the iteration ends. We don't have a precise call
			// site here (instantiation is triggered lazily inside
			// type resolution); the file/line on the frame stays
			// zero and the formatter skips it.
			jam::Diagnostic::Trace traceFrame{/*loc=*/{},
			                                  /*decl=*/im.clonePtr->Name};
			jam::RefTraceFrame guard(refTrace_, std::move(traceFrame));
			// Push the generic's defining module so bare-identifier
			// lookups in the body (e.g. `free` in Vec(T).drop)
			// resolve against that module's namespace. RAII via
			// try/catch: pop on every exit path.
			mutCtx.pushBodyModule(definingModulePath_);
			// CONDITIONAL METHODS: every instantiated method except
			// `cfn drop` exists only for the type arguments its body
			// compiles AND analyzes for (Rust's `impl<T: Clone> Clone
			// for Vec<T>` shape, generalized). A failed body withdraws
			// its diagnostics, records the reason, and the method is
			// simply not provided — calling it reports "not available
			// for this instantiation: <reason>" at the call site.
			// `cfn drop` stays unconditional: silently withdrawing a
			// destructor would change ownership semantics.
			bool conditional = !(im.clonePtr->Name.size() >= 5 &&
			                     im.clonePtr->Name.rfind(".drop") ==
			                         im.clonePtr->Name.size() - 5);
			std::size_t diagMark = mutCtx.diagnostics().size();
			auto withdraw = [&](const std::string &reason) {
				mutCtx.diagnostics().truncateTo(diagMark);
				mutCtx.unregisterFunctionAST(im.clonePtr->Name);
				mutCtx.recordWithdrawnMethod(im.clonePtr->Name, reason);
			};
			try {
				astgenBodyInto(im.passOneJir, *im.clonePtr, mutCtx);
			} catch (const AstGenAnalysisFail &) {
				mutCtx.popBodyModule();
				clearCurrentSubst();
				if (conditional) {
					std::string reason = "does not compile for these "
					                     "type arguments";
					const auto &all = mutCtx.diagnostics().all();
					if (all.size() > diagMark) {
						reason = all[diagMark].message;
					}
					withdraw(reason);
					continue;
				}
				// diagnostic already pushed; trace was attached via
				// the helper. Continue with the next method so the
				// user sees every error in this instantiation.
				continue;
			}
			// Body compiled — now run the mode-aware analysis on the
			// CLONE (substituted param types/modes), so std generic
			// bodies obey the same move/ownership rules user code does.
			// Failures withdraw the method the same way.
			if (mutCtx.analysisFns() != nullptr) {
				static const std::vector<Token> kNoTokens;
				auto adiags = jam::init_analysis::analyze(
				    *im.clonePtr, nodeStore, stringPool, kNoTokens,
				    mutCtx.analysisFns(), mutCtx.getDropRegistry(), &typePool,
				    mutCtx.analysisEnums(), mutCtx.analysisHooks());
				if (!adiags.empty()) {
					mutCtx.popBodyModule();
					clearCurrentSubst();
					if (conditional) {
						withdraw(adiags[0].message);
						continue;
					}
					for (auto &d : adiags) {
						jam::SrcLoc loc{currentFile_, d.line};
						mutCtx.diagnostics().error(loc, d.message);
					}
					continue;
				}
			}
			mutCtx.popBodyModule();
			auto diags = verifyJirFunction(
			    im.passOneJir, &typePool, &stringPool,
			    +[](void *c, TypeIdx t) -> TypeIdx {
				    auto *cc = static_cast<JamCodegenContext *>(c);
				    const TypeKey &k = cc->getTypePool().get(t);
				    if (k.kind == TypeKind::GenericCall) {
					    TypeIdx r = cc->resolveGenericCall(t);
					    if (r != kNoType) return r;
				    }
				    if (k.kind == TypeKind::ArrayExpr) {
					    return cc->resolveArrayExpr(t);
				    }
				    return t;
			    },
			    &mutCtx);
			for (auto &d : diags) {
				if (d.loc.file.empty()) d.loc.file = currentFile_;
				mutCtx.diagnostics().push(std::move(d));
			}
			jirDefineBody(im.passOneJir, mutCtx);
			clearCurrentSubst();
		}
		if (savedBB) { JamLLVMPositionBuilderAtEnd(getBuilder(), savedBB); }
	}

	const_cast<JamCodegenContext &>(*this).popBodyModule();
	return typePool.internNamed(stringPool.intern(instName));
}

// Instantiate an `enum { ... }` expression appearing in a generic
// body's return statement. Substitutes each variant's payload TypeIdx
// list, registers a fresh enum, computes the tagged-union LLVM
// layout, and returns an Enum TypeIdx pointing at it. Memoizes on the
// instantiated name (`Option__i32`, etc).
TypeIdx JamCodegenContext::instantiateEnumExpr(
    const AstNode &exprNode, const std::string &calleeName,
    const std::vector<TypeIdx> &args,
    const std::unordered_map<std::string, TypeIdx> &subst,
    const std::string &definingModulePath) const {
	if (!anonEnums_) {
		throw std::runtime_error(
		    "internal: anonymous enum table not registered on "
		    "codegen context");
	}
	uint32_t anonIdx = exprNode.lhs;
	if (anonIdx >= anonEnums_->size()) {
		throw std::runtime_error(
		    "internal: EnumExpr references missing AnonEnums[" +
		    std::to_string(anonIdx) + "]");
	}
	const EnumDeclAST *anon = (*anonEnums_)[anonIdx].get();

	// Build instantiated name `Option__i32` etc — same shape (and same
	// shared speller) as the struct path.
	std::string instName = calleeName;
	for (TypeIdx a : args) {
		instName += "__";
		instName += genericArgSpelling(typePool, stringPool, a);
	}

	// Memoize. Return as a Named TypeIdx so the rest of codegen resolves
	// through the existing Named -> EnumInfo path (handles size/align,
	// match dispatch, etc., uniformly with non-generic enum references).
	if (const EnumInfo *existing = getEnum(instName)) {
		(void)existing;
		return typePool.internNamed(stringPool.intern(instName));
	}

	// Substitute variant payload types.
	std::unordered_map<std::string, TypeIdx> bodySubst = subst;
	TypeIdx instEnumTy = typePool.internNamed(stringPool.intern(instName));
	bodySubst[anon->Name] = instEnumTy;
	bodySubst["Self"] = instEnumTy;

	std::vector<EnumVariantInfo> variants;
	variants.reserve(anon->Variants.size());
	bool hasPayload = false;
	for (const auto &v : anon->Variants) {
		EnumVariantInfo vi;
		vi.name = v.Name;
		vi.discriminant = v.Discriminant;
		for (TypeIdx ty : v.PayloadTypes) {
			// Qualify against the DEFINING module after substitution —
			// a payload naming the generic's sibling type must not
			// capture the trigger site's same-named type. Mirrors the
			// struct path's field qualification.
			vi.payloadTypes.push_back(requalifyType(
			    substituteType(ty, bodySubst, typePool, stringPool),
			    definingModulePath));
		}
		if (!vi.payloadTypes.empty()) hasPayload = true;
		variants.push_back(std::move(vi));
	}

	registerEnum(instName, std::move(variants));

	// For unit-only enums the LLVM type is plain i8 — no body to set.
	// Done.
	if (!hasPayload) { return instEnumTy; }

	// Payloaded enum: layout mirrors main.cpp's fillEnumBodies path —
	// create a named struct {i8 tag, alignDriver, [extraBytes x i8]},
	// then set on the EnumInfo via setEnumLLVMType.
	JamTypeRef llvmStruct =
	    JamLLVMStructCreateNamed(getContext(), instName.c_str());
	setEnumLLVMType(instName, llvmStruct, 0, 1, true);

	const EnumInfo *info = getEnum(instName);
	uint64_t maxSize = 0, maxAlign = 1;
	for (const auto &v : info->variants) {
		uint64_t off = 0, varAlign = 1;
		for (TypeIdx t : v.payloadTypes) {
			uint64_t s = typeSize(t);
			uint64_t a = typeAlign(t);
			off = (off + a - 1) / a * a;
			off += s;
			if (a > varAlign) varAlign = a;
		}
		if (varAlign > 1) { off = (off + varAlign - 1) / varAlign * varAlign; }
		if (off > maxSize) maxSize = off;
		if (varAlign > maxAlign) maxAlign = varAlign;
	}

	JamTypeRef alignDriver;
	uint64_t alignDriverSize;
	switch (maxAlign) {
	case 1:
		alignDriver = getInt8Type();
		alignDriverSize = 1;
		break;
	case 2:
		alignDriver = getInt16Type();
		alignDriverSize = 2;
		break;
	case 4:
		alignDriver = getInt32Type();
		alignDriverSize = 4;
		break;
	case 8:
		alignDriver = getInt64Type();
		alignDriverSize = 8;
		break;
	default:
		throw std::runtime_error(
		    "Enum `" + instName +
		    "` requires alignment > 8, which is not yet supported");
	}

	uint64_t paddedSize = (maxSize + maxAlign - 1) / maxAlign * maxAlign;
	uint64_t extraBytes =
	    (paddedSize > alignDriverSize) ? paddedSize - alignDriverSize : 0;

	std::vector<JamTypeRef> bodyTypes;
	bodyTypes.push_back(getInt8Type());
	bodyTypes.push_back(alignDriver);
	if (extraBytes > 0) {
		bodyTypes.push_back(
		    JamLLVMArrayType(getInt8Type(), static_cast<unsigned>(extraBytes)));
	}
	JamLLVMStructSetBody(llvmStruct, bodyTypes.data(),
	                     static_cast<unsigned>(bodyTypes.size()), false);
	setEnumLLVMType(instName, llvmStruct, maxSize, maxAlign, true);

	return instEnumTy;
}
