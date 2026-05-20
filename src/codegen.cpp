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
		if (const auto *sinfo = getStruct(name)) {
			result = sinfo->type;
		} else if (const auto *uinfo = getUnion(name)) {
			result = uinfo->type;
		} else if (const auto *einfo = getEnum(name)) {
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
	if (const StructInfo *direct = getStruct(name)) { return direct; }
	// try the type alias table — `const BoxI32 = Box(i32);`
	// maps `BoxI32` to the instantiated struct's TypeIdx.
	TypeIdx aliasTarget = lookupTypeAlias(name);
	if (aliasTarget != kNoType) { return lookupStruct(aliasTarget); }
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
	const TypeKey &k = typePool.get(ty);
	// Accept TypeKind::Union (explicit) or TypeKind::Named (parser-
	// deferred user type that resolves to a union).
	if (k.kind != TypeKind::Union && k.kind != TypeKind::Named) {
		return nullptr;
	}
	const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
	return getUnion(name);
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
	const TypeKey &k = typePool.get(ty);
	// a GenericCall TypeIdx resolves to a concrete type;
	// recurse on the resolved TypeIdx so generic enum instantiations
	// (e.g. `Option(i32)` → `Option__i32`) resolve uniformly.
	if (k.kind == TypeKind::GenericCall) {
		return lookupEnum(resolveGenericCall(ty));
	}
	// Accept TypeKind::Enum (explicit) or TypeKind::Named (parser-
	// deferred user type that resolves to an enum).
	if (k.kind != TypeKind::Enum && k.kind != TypeKind::Named) {
		return nullptr;
	}
	const std::string &name = stringPool.get(static_cast<StringIdx>(k.a));
	if (const EnumInfo *direct = getEnum(name)) return direct;
	// try the type alias table — `const OptI32 =
	// Option(i32);` maps `OptI32` to the instantiated enum's TypeIdx.
	TypeIdx aliasTarget = lookupTypeAlias(name);
	if (aliasTarget != kNoType) { return lookupEnum(aliasTarget); }
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
                                            NodeIdx init, TypeIdx declared) {
	moduleConsts[name] = ModuleConstInfo{init, declared};
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
	auto it = functionAsts.find(name);
	return (it == functionAsts.end()) ? nullptr : it->second;
}

void JamCodegenContext::registerImportHandle(const std::string &handle,
                                             const std::string &modulePath) {
	importHandles_[handle].modulePath = modulePath;
}

void JamCodegenContext::registerPrivateName(const std::string &handle,
                                            const std::string &name) {
	importHandles_[handle].privateNames.insert(name);
}

const JamCodegenContext::ImportHandleInfo *
JamCodegenContext::getImportHandle(const std::string &handle) const {
	auto it = importHandles_.find(handle);
	return (it == importHandles_.end()) ? nullptr : &it->second;
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
	auto it = moduleConsts.find(name);
	if (it != moduleConsts.end()) return &it->second;
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
	case TypeKind::Struct:
	case TypeKind::Named: {
		// a Named type may be a substitution-context
		// reference to a parameter (T → i32) or to Self. Resolve
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
	}
	throw std::runtime_error("typeSize: unhandled type kind");
}

// Alignment requirement of a type. Equal to size for primitive scalars
// on every target we care about. For aggregates, the alignment is the
// max of the constituent alignments.
uint64_t JamCodegenContext::typeAlign(TypeIdx ty) const {
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
	case TypeKind::Struct:
	case TypeKind::Named: {
		// substitution context wins. A Named type may
		// be a parameter reference (T → i32) or Self that resolves
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
	default:
		return ty;
	}
}

}  // namespace

TypeIdx JamCodegenContext::resolveGenericCall(TypeIdx callTy) const {
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
	const std::string &calleeName = stringPool.get(static_cast<StringIdx>(k.a));
	const auto &args = typePool.genericArgsAt(k.b);

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
			// Use generic->Name (the bare source-level name) for the
			// instantiated struct name so syntactic prefixes like
			// `c.Vec` don't bake into `Vec__i32`.
			result = instantiateStructExpr(value, generic->Name, args, subst);
			break;
		}
		if (value.tag == AstTag::EnumExpr) {
			result = instantiateEnumExpr(value, generic->Name, args, subst);
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

// Instantiate a `struct {...}` expression appearing in a generic body's
// return statement. Substitutes each field's TypeIdx with the concrete
// generic args, creates a fresh LLVM struct type with a unique name, and
// returns a Named TypeIdx pointing at the new struct. Methods are not
// instantiated in v1.
TypeIdx JamCodegenContext::instantiateStructExpr(
    const AstNode &exprNode, const std::string &calleeName,
    const std::vector<TypeIdx> &args,
    const std::unordered_map<std::string, TypeIdx> &subst) const {
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

	// Build the instantiated struct's name from the callee + arg names.
	// `Maybe(File)` → `Maybe__File`. Pointer/array types lower through
	// substituteType; we only need a stable spelling for the canonical
	// non-compound cases here. v1's stdlib won't pass non-named types
	// as generic args, so this is enough to get the demo running.
	std::string instName = calleeName;
	for (TypeIdx a : args) {
		instName += "__";
		const TypeKey &ak = typePool.get(a);
		switch (ak.kind) {
		case TypeKind::Int: {
			char buf[16];
			std::snprintf(buf, sizeof(buf), "%c%u", ak.b ? 'i' : 'u', ak.a);
			instName += buf;
			break;
		}
		case TypeKind::Bool:
			instName += "bool";
			break;
		case TypeKind::Struct:
		case TypeKind::Named:
			instName += stringPool.get(static_cast<StringIdx>(ak.a));
			break;
		default:
			instName += "T";  // catch-all; v2 spec needed
			break;
		}
	}

	// Memoize on the instantiated name. If we've already produced this
	// struct, return its TypeIdx without re-creating the LLVM type.
	if (const StructInfo *existing = getStruct(instName)) {
		(void)existing;
		return typePool.internNamed(stringPool.intern(instName));
	}

	// Build the full substitution map: parameter names → concrete args,
	// plus the anon-struct's synthetic name (which is what `Self`
	// resolved to in *type* positions at parse time) → the new
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
	std::vector<std::pair<std::string, TypeIdx>> instFields;
	instFields.reserve(anon->Fields.size());
	for (const auto &f : anon->Fields) {
		instFields.emplace_back(
		    f.first, substituteType(f.second, bodySubst, typePool, stringPool));
	}

	JamTypeRef llvmStruct =
	    JamLLVMStructCreateNamed(getContext(), instName.c_str());
	registerStruct(instName, llvmStruct, instFields);

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
		for (const auto &origMethod : anon->Methods) {
			std::vector<Param> instArgs;
			instArgs.reserve(origMethod->Args.size());
			for (const auto &p : origMethod->Args) {
				Param sp = p;
				sp.Type =
				    substituteType(p.Type, bodySubst, typePool, stringPool);
				instArgs.push_back(std::move(sp));
			}
			TypeIdx instReturn = origMethod->ReturnType;
			if (instReturn != kNoType) {
				instReturn =
				    substituteType(instReturn, bodySubst, typePool, stringPool);
			}

			std::string instMethodName = instName + "." + origMethod->Name;
			auto cloned = std::make_unique<FunctionAST>(
			    instMethodName, std::move(instArgs), instReturn,
			    origMethod->Body, origMethod->isExtern, origMethod->isExport,
			    origMethod->isPub, origMethod->isTest, origMethod->isVarArgs);
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
			// the prototype with JIR's ABI (mut/move → ptr). The
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
			try {
				astgenBodyInto(im.passOneJir, *im.clonePtr, mutCtx);
			} catch (const AstGenAnalysisFail &) {
				// diagnostic already pushed; trace was attached via
				// the helper. Continue with the next method so the
				// user sees every error in this instantiation.
				clearCurrentSubst();
				continue;
			}
			auto diags = verifyJirFunction(
			    im.passOneJir, &typePool, &stringPool,
			    +[](void *c, TypeIdx t) -> TypeIdx {
				    auto *cc = static_cast<JamCodegenContext *>(c);
				    const TypeKey &k = cc->getTypePool().get(t);
				    if (k.kind == TypeKind::GenericCall) {
					    TypeIdx r = cc->resolveGenericCall(t);
					    if (r != kNoType) return r;
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
    const std::unordered_map<std::string, TypeIdx> &subst) const {
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

	// Build instantiated name `Option__i32` etc — same shape as struct.
	std::string instName = calleeName;
	for (TypeIdx a : args) {
		instName += "__";
		const TypeKey &ak = typePool.get(a);
		switch (ak.kind) {
		case TypeKind::Int: {
			char buf[16];
			std::snprintf(buf, sizeof(buf), "%c%u", ak.b ? 'i' : 'u', ak.a);
			instName += buf;
			break;
		}
		case TypeKind::Bool:
			instName += "bool";
			break;
		case TypeKind::Struct:
		case TypeKind::Enum:
		case TypeKind::Named:
			instName += stringPool.get(static_cast<StringIdx>(ak.a));
			break;
		default:
			instName += "T";
			break;
		}
	}

	// Memoize. Return as a Named TypeIdx so the rest of codegen resolves
	// through the existing Named → EnumInfo path (handles size/align,
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
			vi.payloadTypes.push_back(
			    substituteType(ty, bodySubst, typePool, stringPool));
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
