/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "ast.h"
#include "abi.h"
#include "ast_flat.h"
#include "codegen.h"
#include "jam_llvm.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

JamBasicBlockRef CurrentLoopContinue = nullptr;
JamBasicBlockRef CurrentLoopBreak = nullptr;
// index into the drop-scope stack of the *currently enclosing
// loop body's scope*. On `break` / `continue`, the codegen emits drops
// for every scope from the top down to (and including) this index, so
// locals declared inside the loop body — possibly under nested
// if/match scopes — get their drops fired before the branch leaves the
// loop. SIZE_MAX means "no enclosing loop body active here".
size_t CurrentLoopBodyScopeIdx = SIZE_MAX;

// Forward declaration so codegenCall (auto-address path) can call
// codegenAddressOf which is defined further down in the file.
static JamValueRef codegenAddressOf(JamCodegenContext &ctx, const AstNode &n);

// Forward declaration so the struct-method dispatch in codegenCall can
// translate a qualified source-level name (`File.drop`) into the LLVM
// symbol (`__drop_File`). The implementation lives near declarePrototype.
static std::string mangledFunctionName(const FunctionAST &fn,
                                       const TypePool &types,
                                       const StringPool &strings);

static JamValueRef coerceTo(JamCodegenContext &ctx, JamValueRef val,
                            JamTypeRef expected) {
	JamTypeRef actual = JamLLVMTypeOf(val);
	if (actual == expected) return val;
	if (JamLLVMTypeIsFloat(expected) && JamLLVMTypeIsInteger(actual)) {
		// note:
		// implicit int-to-float coercion is rejected: a float-typed
		// destination needs a float literal (e.g. `3.0`) or an explicit
		// `as` cast. Catches `var x: f32 = 3;` early. without this check
		// would silently emit `store i8 into f32` and produce malformed
		// IR.
		throw std::runtime_error(
		    "cannot assign integer to float-typed destination; use a "
		    "float literal (e.g. `3.0`) or an explicit `as` cast");
	}
	if (JamLLVMTypeIsFloat(expected) && JamLLVMTypeIsFloat(actual)) {
		return JamLLVMBuildFPCast(ctx.getBuilder(), val, expected,
		                          "assign_fpcast");
	}
	if (JamLLVMTypeIsInteger(expected) && JamLLVMTypeIsInteger(actual)) {
		return JamLLVMBuildIntCast(ctx.getBuilder(), val, expected, false,
		                           "assign_icast");
	}
	if (JamLLVMTypeIsPointer(expected) && JamLLVMTypeIsStruct(actual)) {
		// `[]T` slice (LLVM `{ptr, i64}`) passed where the callee declared
		// `*const[] T` / `*const T`. Extract field 0 (the data pointer).
		// Without this the slice's `len` half gets passed in the next
		// register, polluting whatever the callee reads next — on Linux
		// AArch64 this means snprintf's first vararg slot ends up holding
		// the format slice's length instead of the user's first vararg.
		return JamLLVMBuildExtractValue(ctx.getBuilder(), val, 0,
		                                "slice.to.ptr");
	}
	return val;
}

// Returns the pointee TypeIdx of `*T` or `[*]T`; kNoType otherwise.
static TypeIdx pointeeTypeOf(const TypePool &tp, TypeIdx ptrType) {
	if (ptrType == kNoType) return kNoType;
	const TypeKey &k = tp.get(ptrType);
	if (k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany) {
		return static_cast<TypeIdx>(k.a);
	}
	return kNoType;
}

// Reconstruct the u64 numeric value from a NumberLit node (lhs holds low 32
// bits, rhs holds high 32 bits; flags bit 0 marks negation).
static uint64_t numberLitValue(const AstNode &n) {
	return static_cast<uint64_t>(n.lhs) | (static_cast<uint64_t>(n.rhs) << 32);
}

// Resolve a PatEnumVariant AST node to its canonical enum name + variant
// info. Handles all four encoding combinations:
//   flags & 1 == 0: no bindings (lhs = StringIdx OR TypeIdx, rhs = variant)
//   flags & 1 == 1: with bindings (lhs = ExtraIdx → [recv, variant, count,
//   ...]) flags & 2 == 0: receiver is StringIdx (source-level name,
//   alias-resolve) flags & 2 == 1: receiver is TypeIdx (GenericCall — resolved
//   via lookupEnum)
//
// Used by every codegen / exhaustiveness / payload-binding site that
// consumes a PatEnumVariant. Centralizes the alias / generic-call
// resolution so future encodings only touch this function.
struct DecodedPatEnumVariant {
	std::string enumName;  // canonical (`Option__i32`, not alias)
	std::string variantName;
	const JamCodegenContext::EnumInfo *einfo = nullptr;
	int variantIndex = -1;  // -1 if variant missing
	bool hasBindings = false;
	ExtraIdx bindingsStart =
	    0;  // start of [count, b0, b1, ...] when hasBindings
	uint32_t bindingCount = 0;
};

static DecodedPatEnumVariant decodePatEnumVariant(JamCodegenContext &ctx,
                                                  const AstNode &pn) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	DecodedPatEnumVariant d;
	bool hasBindings = (pn.flags & 1) != 0;
	bool typeIdxReceiver = (pn.flags & 2) != 0;

	uint32_t recvSlot;
	StringIdx variantNameId;
	if (hasBindings) {
		ExtraIdx ex = static_cast<ExtraIdx>(pn.lhs);
		recvSlot = ns.getExtra(ex);
		variantNameId = static_cast<StringIdx>(ns.getExtra(ex + 1));
		d.bindingCount = ns.getExtra(ex + 2);
		d.bindingsStart = ex + 3;
		d.hasBindings = true;
	} else {
		recvSlot = pn.lhs;
		variantNameId = static_cast<StringIdx>(pn.rhs);
	}
	d.variantName = sp.get(variantNameId);

	if (typeIdxReceiver) {
		TypeIdx ty = static_cast<TypeIdx>(recvSlot);
		if (const auto *info = ctx.lookupEnum(ty)) {
			d.enumName = info->name;
			d.einfo = info;
		}
	} else {
		std::string recvName = sp.get(static_cast<StringIdx>(recvSlot));
		d.enumName = recvName;
		d.einfo = ctx.getEnum(recvName);
		if (!d.einfo) {
			// Source-level alias (`const OptI32 = Option(i32);`).
			TypeIdx aliasTarget = ctx.lookupTypeAlias(recvName);
			if (aliasTarget != kNoType) {
				if (const auto *info = ctx.lookupEnum(aliasTarget)) {
					d.enumName = info->name;
					d.einfo = info;
				}
			}
		}
	}
	if (d.einfo) {
		d.variantIndex = ctx.getEnumVariantIndex(d.enumName, d.variantName);
	}
	return d;
}

// Materialize a NumberLit node directly at a known integer type. Falls back
// to natural smallest-fit width when expectedType is null or non-integer.
// Float literals (flags bit 1) lower via LLVMConstReal at the expected
// width or default f64; the parser packs the host `double`'s bit pattern
// into the same low/high u32 slots.
static JamValueRef numberLitConst(JamCodegenContext &ctx, const AstNode &n,
                                  JamTypeRef expectedType) {
	uint64_t val = numberLitValue(n);
	bool isNegative = (n.flags & 1) != 0;
	bool isFloat = (n.flags & 2) != 0;

	if (isFloat) {
		// Reinterpret the packed u64 as a host double.
		double d = 0.0;
		std::memcpy(&d, &val, sizeof(d));
		if (isNegative) d = -d;
		// Default to f64; honour explicit f32 from context.
		JamTypeRef ty = (expectedType && JamLLVMTypeIsFloat(expectedType))
		                    ? expectedType
		                    : ctx.getDoubleType();
		return JamLLVMConstReal(ty, d);
	}

	if (expectedType && JamLLVMTypeIsInteger(expectedType)) {
		if (isNegative) {
			int64_t signedVal = -static_cast<int64_t>(val);
			return JamLLVMConstInt(expectedType,
			                       static_cast<uint64_t>(signedVal), true);
		}
		return JamLLVMConstInt(expectedType, val, false);
	}

	JamTypeRef IntType;
	if (isNegative) {
		if (val <= 128) IntType = ctx.getInt8Type();
		else if (val <= 32768) IntType = ctx.getInt16Type();
		else if (val <= 2147483648ULL) IntType = ctx.getInt32Type();
		else IntType = ctx.getInt64Type();
		int64_t signedVal = -static_cast<int64_t>(val);
		return JamLLVMConstInt(IntType, static_cast<uint64_t>(signedVal), true);
	}
	if (val <= 255) IntType = ctx.getInt8Type();
	else if (val <= 65535) IntType = ctx.getInt16Type();
	else if (val <= 4294967295ULL) IntType = ctx.getInt32Type();
	else IntType = ctx.getInt64Type();
	return JamLLVMConstInt(IntType, val, false);
}

// Walk a MemberAccess/Variable chain back to its root variable. Fills `path`
// with member names (outermost-first → leaf-last). Returns the StringIdx of
// the root variable name, or kNoString if the root is not a Variable.
static StringIdx collectMemberChain(const NodeStore &nodes, NodeIdx idx,
                                    std::vector<StringIdx> &outPath) {
	std::vector<StringIdx> reversed;
	NodeIdx cur = idx;
	while (true) {
		const AstNode &n = nodes.get(cur);
		if (n.tag == AstTag::MemberAccess) {
			reversed.push_back(static_cast<StringIdx>(n.rhs));
			cur = static_cast<NodeIdx>(n.lhs);
			continue;
		}
		if (n.tag == AstTag::Variable) {
			for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
				outPath.push_back(*it);
			}
			return static_cast<StringIdx>(n.lhs);
		}
		return kNoString;
	}
}

static TypeIdx inferLValueType(JamCodegenContext &ctx, NodeIdx nodeIdx);

struct LValueAddress {
	JamValueRef ptr;
	TypeIdx leafType;
	JamTypeRef leafLLVMType;
};

// resolveLValueAddress:
// Walk a chain of `Variable` / `MemberAccess` / `Index` / `Deref` nodes
// and return a pointer at the leaf. Supports arbitrary interleaving so
// `arr[i].field`, `g.things[i].pc`, `slice[i].sub.x`, and `xs[i].*` all
// reach the right storage. The resolver also handles slice projection
// (`s.ptr` / `s.len`), slice indexing, many-pointer indexing, and union
// member access (all union fields share the base address).
//
// Index sub-expressions are codegen'd inline; callers must NOT codegen
// the index themselves.
static LValueAddress resolveLValueAddress(JamCodegenContext &ctx,
                                          NodeIdx nodeIdx) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	const AstNode &n = ns.get(nodeIdx);

	if (n.tag == AstTag::Variable) {
		const std::string &name = sp.get(static_cast<StringIdx>(n.lhs));
		JamValueRef alloca = ctx.getVariable(name);
		if (!alloca) { throw std::runtime_error("Unknown variable: " + name); }
		TypeIdx ty = ctx.getVariableType(name);
		LValueAddress r;
		r.ptr = alloca;
		r.leafType = ty;
		r.leafLLVMType = ctx.getLLVMType(ty);
		return r;
	}

	if (n.tag == AstTag::MemberAccess) {
		LValueAddress base =
		    resolveLValueAddress(ctx, static_cast<NodeIdx>(n.lhs));
		const std::string &fieldName = sp.get(static_cast<StringIdx>(n.rhs));
		const TypeKey &k = ctx.getTypePool().get(base.leafType);

		// `s.ptr` and `s.len`. The slice is lowered as
		// `{ptr, len}` so each field has a static struct index.
		if (k.kind == TypeKind::Slice) {
			TypeIdx elemTy = static_cast<TypeIdx>(k.a);
			if (fieldName == "ptr") {
				LValueAddress r;
				r.ptr =
				    JamLLVMBuildStructGEP(ctx.getBuilder(), base.leafLLVMType,
				                          base.ptr, 0, "slice.ptr");
				r.leafType = ctx.getTypePool().internPtrMany(elemTy);
				r.leafLLVMType = ctx.getLLVMType(r.leafType);
				return r;
			}
			if (fieldName == "len") {
				LValueAddress r;
				r.ptr =
				    JamLLVMBuildStructGEP(ctx.getBuilder(), base.leafLLVMType,
				                          base.ptr, 1, "slice.len");
				r.leafType = BuiltinType::U64;
				r.leafLLVMType = ctx.getLLVMType(r.leafType);
				return r;
			}
			throw std::runtime_error("Slice has no field '" + fieldName +
			                         "' (only .ptr and .len)");
		}

		// Union: every field shares the same address.
		if (const auto *uinfo = ctx.lookupUnion(base.leafType)) {
			TypeIdx fieldTy = ctx.getUnionFieldType(uinfo->name, fieldName);
			if (fieldTy == kNoType) {
				throw std::runtime_error("Union `" + uinfo->name +
				                         "` has no field `" + fieldName + "`");
			}
			LValueAddress r;
			r.ptr = base.ptr;
			r.leafType = fieldTy;
			r.leafLLVMType = ctx.getLLVMType(fieldTy);
			return r;
		}

		// Struct field.
		const auto *info = ctx.lookupStruct(base.leafType);
		if (!info) {
			throw std::runtime_error("Cannot access field '" + fieldName +
			                         "' on non-struct type");
		}
		int fldIdx = ctx.getFieldIndex(info->name, fieldName);
		if (fldIdx < 0) {
			throw std::runtime_error("Unknown field '" + fieldName +
			                         "' in struct " + info->name);
		}
		TypeIdx fldType = info->fields[fldIdx].second;
		LValueAddress r;
		r.ptr = JamLLVMBuildStructGEP(ctx.getBuilder(), base.leafLLVMType,
		                              base.ptr, static_cast<unsigned>(fldIdx),
		                              fieldName.c_str());
		r.leafType = fldType;
		r.leafLLVMType = ctx.getLLVMType(fldType);
		return r;
	}

	if (n.tag == AstTag::Index) {
		JamValueRef idxVal = codegenNode(ctx, static_cast<NodeIdx>(n.rhs));
		if (!idxVal) {
			throw std::runtime_error("failed to codegen index expression");
		}
		idxVal = coerceTo(ctx, idxVal, ctx.getInt64Type());

		LValueAddress base =
		    resolveLValueAddress(ctx, static_cast<NodeIdx>(n.lhs));
		const TypeKey &k = ctx.getTypePool().get(base.leafType);

		if (k.kind == TypeKind::Array) {
			TypeIdx elemTy = static_cast<TypeIdx>(k.a);
			LValueAddress r;
			r.leafType = elemTy;
			r.leafLLVMType = ctx.getLLVMType(elemTy);
			r.ptr = JamLLVMBuildArrayGEP(ctx.getBuilder(), base.leafLLVMType,
			                             base.ptr, idxVal, "idxgep");
			return r;
		}
		if (k.kind == TypeKind::Slice) {
			TypeIdx elemTy = static_cast<TypeIdx>(k.a);
			JamValueRef ptrSlot =
			    JamLLVMBuildStructGEP(ctx.getBuilder(), base.leafLLVMType,
			                          base.ptr, 0, "slice.ptr.slot");
			TypeIdx ptrType = ctx.getTypePool().internPtrMany(elemTy);
			JamTypeRef ptrLLVMType = ctx.getLLVMType(ptrType);
			JamValueRef loadedPtr = JamLLVMBuildLoad(
			    ctx.getBuilder(), ptrLLVMType, ptrSlot, "slice.ptr");
			LValueAddress r;
			r.leafType = elemTy;
			r.leafLLVMType = ctx.getLLVMType(elemTy);
			r.ptr = JamLLVMBuildPtrGEP(ctx.getBuilder(), r.leafLLVMType,
			                           loadedPtr, idxVal, "ptrgep");
			return r;
		}
		if (k.kind == TypeKind::PtrMany) {
			TypeIdx elemTy = static_cast<TypeIdx>(k.a);
			JamValueRef loadedPtr = JamLLVMBuildLoad(
			    ctx.getBuilder(), base.leafLLVMType, base.ptr, "ptrload");
			LValueAddress r;
			r.leafType = elemTy;
			r.leafLLVMType = ctx.getLLVMType(elemTy);
			r.ptr = JamLLVMBuildPtrGEP(ctx.getBuilder(), r.leafLLVMType,
			                           loadedPtr, idxVal, "ptrgep");
			return r;
		}
		if (k.kind == TypeKind::PtrSingle) {
			throw std::runtime_error(
			    "Cannot index single-item pointer; use `.*` to "
			    "dereference, or declare as `*mut[] T` for many-item "
			    "indexing");
		}
		throw std::runtime_error("Cannot index value of this type");
	}

	if (n.tag == AstTag::Deref) {
		// `ptr.*` as an lvalue: the pointer's value IS the address of
		// the pointee. Load the pointer, hand it back as the leaf
		// pointer with the pointee type.
		LValueAddress base =
		    resolveLValueAddress(ctx, static_cast<NodeIdx>(n.lhs));
		const TypeKey &k = ctx.getTypePool().get(base.leafType);
		if (k.kind != TypeKind::PtrSingle && k.kind != TypeKind::PtrMany) {
			throw std::runtime_error("Cannot dereference non-pointer type");
		}
		TypeIdx pointeeTy = static_cast<TypeIdx>(k.a);
		JamValueRef loadedPtr = JamLLVMBuildLoad(
		    ctx.getBuilder(), base.leafLLVMType, base.ptr, "deref.ptr");
		LValueAddress r;
		r.ptr = loadedPtr;
		r.leafType = pointeeTy;
		r.leafLLVMType = ctx.getLLVMType(pointeeTy);
		return r;
	}

	if (n.tag == AstTag::Call) {
		// ok so, cases like `foo().field`, `foo()[i]`, `arr[i].method().sub`.
		// to keep the chain going we materialize the call, spill it to a stack
		// temp, and hand back a pointer to the temp.
		//
		// To allocate the right-sized temp and to find the right struct
		// for further `.field` GEPs, we mirror codegenCall's dispatch to
		// figure out the source-level return type:
		//   StringIdx callee, free function   ->  registry["foo"]
		//   StringIdx callee, `inst.method`   ->  walk to Struct.method
		//   StringIdx callee, `Struct.method` ->  registry["Struct.method"]
		//   NodeIdx callee, `…method`         ->  receiver-type method
		TypeIdx retType = kNoType;
		if ((n.flags & 1) == 0) {
			const std::string &callee = sp.get(static_cast<StringIdx>(n.lhs));
			if (const FunctionAST *fn = ctx.getFunctionAST(callee)) {
				retType = fn->ReturnType;
			} else {
				size_t firstDot = callee.find('.');
				size_t lastDot = callee.rfind('.');
				if (firstDot != std::string::npos && firstDot == lastDot) {
					std::string instName = callee.substr(0, firstDot);
					std::string methodPart = callee.substr(firstDot + 1);
					if (ctx.hasVariable(instName)) {
						TypeIdx instTy = ctx.getVariableType(instName);
						if (const auto *sinfo = ctx.lookupStruct(instTy)) {
							std::string qualified =
							    sinfo->name + "." + methodPart;
							if (const FunctionAST *fn =
							        ctx.getFunctionAST(qualified)) {
								retType = fn->ReturnType;
							}
						}
					}
				}
			}
		} else {
			NodeIdx calleeNodeIdx = static_cast<NodeIdx>(n.lhs);
			const AstNode &cn = ns.get(calleeNodeIdx);
			if (cn.tag == AstTag::MemberAccess) {
				TypeIdx recvTy =
				    inferLValueType(ctx, static_cast<NodeIdx>(cn.lhs));
				if (const auto *sinfo = ctx.lookupStruct(recvTy)) {
					const std::string &methodName =
					    sp.get(static_cast<StringIdx>(cn.rhs));
					std::string qualified = sinfo->name + "." + methodName;
					if (const FunctionAST *fn = ctx.getFunctionAST(qualified)) {
						retType = fn->ReturnType;
					}
				}
			}
		}
		if (retType == kNoType) {
			throw std::runtime_error(
			    "Cannot chain through a call whose return type the "
			    "codegen can't statically resolve — assign the call "
			    "result to a `var` first");
		}
		JamValueRef val = codegenNode(ctx, nodeIdx);
		if (!val) {
			throw std::runtime_error("call produced no value to chain");
		}
		JamTypeRef retLLVMType = ctx.getLLVMType(retType);
		JamValueRef tmp = JamLLVMBuildAlloca(
		    ctx.getBuilder(), retLLVMType, ctx.typeAlign(retType), "call.tmp");
		JamLLVMBuildStore(ctx.getBuilder(), val, tmp);
		LValueAddress r;
		r.ptr = tmp;
		r.leafType = retType;
		r.leafLLVMType = retLLVMType;
		return r;
	}

	throw std::runtime_error(
	    "lvalue chain contains an unsupported node — only "
	    "Variable, MemberAccess, Index, Deref, and Call are "
	    "addressable; assign the value to a `var` first if you need to "
	    "chain through a more complex expression");
}

static TypeIdx inferLValueType(JamCodegenContext &ctx, NodeIdx nodeIdx) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	const AstNode &n = ns.get(nodeIdx);

	if (n.tag == AstTag::Variable) {
		const std::string &name = sp.get(static_cast<StringIdx>(n.lhs));
		return ctx.getVariableType(name);
	}
	if (n.tag == AstTag::MemberAccess) {
		TypeIdx baseTy = inferLValueType(ctx, static_cast<NodeIdx>(n.lhs));
		if (baseTy == kNoType) return kNoType;
		const std::string &fieldName = sp.get(static_cast<StringIdx>(n.rhs));
		const TypeKey &k = ctx.getTypePool().get(baseTy);
		if (k.kind == TypeKind::Slice) {
			if (fieldName == "ptr") {
				return ctx.getTypePool().internPtrMany(
				    static_cast<TypeIdx>(k.a));
			}
			if (fieldName == "len") return BuiltinType::U64;
			return kNoType;
		}
		if (const auto *uinfo = ctx.lookupUnion(baseTy)) {
			return ctx.getUnionFieldType(uinfo->name, fieldName);
		}
		const auto *info = ctx.lookupStruct(baseTy);
		if (!info) return kNoType;
		int idx = ctx.getFieldIndex(info->name, fieldName);
		if (idx < 0) return kNoType;
		return info->fields[idx].second;
	}
	if (n.tag == AstTag::Index) {
		TypeIdx baseTy = inferLValueType(ctx, static_cast<NodeIdx>(n.lhs));
		if (baseTy == kNoType) return kNoType;
		const TypeKey &k = ctx.getTypePool().get(baseTy);
		if (k.kind == TypeKind::Array || k.kind == TypeKind::Slice ||
		    k.kind == TypeKind::PtrMany) {
			return static_cast<TypeIdx>(k.a);
		}
		return kNoType;
	}
	if (n.tag == AstTag::Deref) {
		TypeIdx baseTy = inferLValueType(ctx, static_cast<NodeIdx>(n.lhs));
		if (baseTy == kNoType) return kNoType;
		const TypeKey &k = ctx.getTypePool().get(baseTy);
		if (k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany) {
			return static_cast<TypeIdx>(k.a);
		}
		return kNoType;
	}
	return kNoType;
}

static JamValueRef resolveIndexedElementPtr(JamCodegenContext &ctx,
                                            NodeIdx objectIdx,
                                            JamValueRef idxVal,
                                            JamTypeRef &outElemType) {
	LValueAddress base = resolveLValueAddress(ctx, objectIdx);
	const TypeKey &k = ctx.getTypePool().get(base.leafType);

	if (k.kind == TypeKind::Array) {
		TypeIdx elemTy = static_cast<TypeIdx>(k.a);
		outElemType = ctx.getLLVMType(elemTy);
		return JamLLVMBuildArrayGEP(ctx.getBuilder(), base.leafLLVMType,
		                            base.ptr, idxVal, "idxgep");
	}
	if (k.kind == TypeKind::Slice) {
		TypeIdx elemTy = static_cast<TypeIdx>(k.a);
		outElemType = ctx.getLLVMType(elemTy);
		JamValueRef ptrSlot = JamLLVMBuildStructGEP(
		    ctx.getBuilder(), base.leafLLVMType, base.ptr, 0, "slice.ptr.slot");
		TypeIdx ptrType = ctx.getTypePool().internPtrMany(elemTy);
		JamValueRef loadedPtr = JamLLVMBuildLoad(
		    ctx.getBuilder(), ctx.getLLVMType(ptrType), ptrSlot, "slice.ptr");
		return JamLLVMBuildPtrGEP(ctx.getBuilder(), outElemType, loadedPtr,
		                          idxVal, "ptrgep");
	}
	if (k.kind == TypeKind::PtrMany) {
		TypeIdx elemTy = static_cast<TypeIdx>(k.a);
		outElemType = ctx.getLLVMType(elemTy);
		JamValueRef loadedPtr = JamLLVMBuildLoad(
		    ctx.getBuilder(), base.leafLLVMType, base.ptr, "ptrload");
		return JamLLVMBuildPtrGEP(ctx.getBuilder(), outElemType, loadedPtr,
		                          idxVal, "ptrgep");
	}
	if (k.kind == TypeKind::PtrSingle) {
		throw std::runtime_error(
		    "Cannot index single-item pointer; use `.*` to dereference, "
		    "or declare as `*mut[] T` for many-item indexing");
	}
	throw std::runtime_error("Cannot index value of this type");
}

static JamValueRef genPrintCall(JamCodegenContext &ctx,
                                const std::string &callee, const uint32_t *args,
                                uint32_t argCount) {
	JamFunctionRef printfFunc = JamLLVMGetFunction(ctx.getModule(), "printf");
	if (!printfFunc) {
		JamTypeRef i8PtrType = JamLLVMPointerType(ctx.getInt8Type(), 0);
		JamTypeRef printfRetType = ctx.getInt32Type();
		JamTypeRef printfParamTypes[1] = {i8PtrType};
		JamTypeRef printfType =
		    JamLLVMFunctionType(printfRetType, printfParamTypes, 1, true);
		printfFunc = JamLLVMAddFunction(ctx.getModule(), "printf", printfType);
		JamLLVMApplyDefaultFnAttrs(printfFunc, /*isExtern=*/true);
	}

	JamFunctionRef putsFunc = JamLLVMGetFunction(ctx.getModule(), "puts");
	if (!putsFunc) {
		JamTypeRef i8PtrType = JamLLVMPointerType(ctx.getInt8Type(), 0);
		JamTypeRef putsRetType = ctx.getInt32Type();
		JamTypeRef putsParamTypes[1] = {i8PtrType};
		JamTypeRef putsType =
		    JamLLVMFunctionType(putsRetType, putsParamTypes, 1, false);
		putsFunc = JamLLVMAddFunction(ctx.getModule(), "puts", putsType);
		JamLLVMApplyDefaultFnAttrs(putsFunc, /*isExtern=*/true);
	}

	if (callee == "std.fmt.println" && argCount == 1) {
		JamValueRef arg = codegenNode(ctx, args[0]);
		if (!arg) return nullptr;
		if (JamLLVMTypeIsStruct(JamLLVMTypeOf(arg))) {
			arg = JamLLVMBuildExtractValue(ctx.getBuilder(), arg, 0, "str_ptr");
		}
		JamValueRef callArgs[1] = {arg};
		return JamLLVMBuildCall(ctx.getBuilder(), putsFunc, callArgs, 1,
		                        "puts_call");
	}
	if (callee == "std.fmt.print" && argCount == 1) {
		JamValueRef arg = codegenNode(ctx, args[0]);
		if (!arg) return nullptr;
		if (JamLLVMTypeIsStruct(JamLLVMTypeOf(arg))) {
			arg = JamLLVMBuildExtractValue(ctx.getBuilder(), arg, 0, "str_ptr");
		}
		JamValueRef formatPtr =
		    JamLLVMBuildGlobalStringPtr(ctx.getBuilder(), "%s", "print_fmt");
		JamValueRef callArgs[2] = {formatPtr, arg};
		return JamLLVMBuildCall(ctx.getBuilder(), printfFunc, callArgs, 2,
		                        "printf_call");
	}
	throw std::runtime_error("Complex print formatting not yet implemented");
}

static JamValueRef genSleepCall(JamCodegenContext &ctx, const uint32_t *args,
                                uint32_t argCount) {
	if (argCount != 1) {
		throw std::runtime_error(
		    "std.thread.sleep expects exactly 1 argument (milliseconds)");
	}
	JamFunctionRef usleepFunc = JamLLVMGetFunction(ctx.getModule(), "usleep");
	if (!usleepFunc) {
		JamTypeRef usleepRetType = ctx.getInt32Type();
		JamTypeRef usleepParamTypes[1] = {ctx.getInt32Type()};
		JamTypeRef usleepType =
		    JamLLVMFunctionType(usleepRetType, usleepParamTypes, 1, false);
		usleepFunc = JamLLVMAddFunction(ctx.getModule(), "usleep", usleepType);
		JamLLVMApplyDefaultFnAttrs(usleepFunc, /*isExtern=*/true);
	}

	JamValueRef msArg = codegenNode(ctx, args[0]);
	if (!msArg) return nullptr;

	JamTypeRef i64Type = ctx.getInt64Type();
	JamValueRef msArg64 =
	    JamLLVMBuildIntCast(ctx.getBuilder(), msArg, i64Type, false, "ms_cast");
	JamValueRef thousand = JamLLVMConstInt(i64Type, 1000, false);
	JamValueRef usArg64 =
	    JamLLVMBuildMul(ctx.getBuilder(), msArg64, thousand, "us_mul");
	JamValueRef usArg = JamLLVMBuildIntCast(
	    ctx.getBuilder(), usArg64, ctx.getInt32Type(), false, "us_trunc");
	JamValueRef callArgs[1] = {usArg};
	return JamLLVMBuildCall(ctx.getBuilder(), usleepFunc, callArgs, 1,
	                        "usleep_call");
}

static JamValueRef genAssertCall(JamCodegenContext &ctx, const uint32_t *args,
                                 uint32_t argCount) {
	if (argCount != 2) {
		throw std::runtime_error(
		    "assert expects exactly 2 arguments (actual, expected)");
	}

	JamValueRef actual = codegenNode(ctx, args[0]);
	if (!actual) return nullptr;
	JamTypeRef actualType = JamLLVMTypeOf(actual);

	// Materialize integer-literal "expected" arg directly at actual's type
	// so we never need to guess sext vs zext.
	const AstNode &expectedNode = ctx.getNodeStore().get(args[1]);
	JamTypeRef expectedHint = nullptr;
	if (expectedNode.tag == AstTag::NumberLit &&
	    JamLLVMTypeIsInteger(actualType)) {
		expectedHint = actualType;
	}
	JamValueRef expected = codegenNode(ctx, args[1], expectedHint);
	if (!expected) return nullptr;

	JamTypeRef expectedType = JamLLVMTypeOf(expected);
	if (actualType != expectedType) {
		if (JamLLVMTypeIsInteger(actualType) &&
		    JamLLVMTypeIsInteger(expectedType)) {
			unsigned aw = JamLLVMGetIntTypeWidth(actualType);
			unsigned ew = JamLLVMGetIntTypeWidth(expectedType);
			if (aw > ew) {
				expected =
				    JamLLVMBuildIntCast(ctx.getBuilder(), expected, actualType,
				                        false, "assert_cast");
			} else {
				actual =
				    JamLLVMBuildIntCast(ctx.getBuilder(), actual, expectedType,
				                        false, "assert_cast");
			}
		}
	}

	JamValueRef cmpResult = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_EQ,
	                                         actual, expected, "assert_cmp");

	JamBasicBlockRef currentBlock = JamLLVMGetInsertBlock(ctx.getBuilder());
	JamFunctionRef TheFunction = JamLLVMGetBasicBlockParent(currentBlock);
	JamBasicBlockRef failBlock =
	    JamLLVMAppendBasicBlock(TheFunction, "assert.fail");
	JamBasicBlockRef passBlock =
	    JamLLVMAppendBasicBlock(TheFunction, "assert.pass");

	JamLLVMBuildCondBr(ctx.getBuilder(), cmpResult, passBlock, failBlock);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), failBlock);

	JamFunctionRef printfFunc = JamLLVMGetFunction(ctx.getModule(), "printf");
	if (!printfFunc) {
		JamTypeRef i8PtrType = JamLLVMPointerType(ctx.getInt8Type(), 0);
		JamTypeRef printfParamTypes[1] = {i8PtrType};
		JamTypeRef printfType =
		    JamLLVMFunctionType(ctx.getInt32Type(), printfParamTypes, 1, true);
		printfFunc = JamLLVMAddFunction(ctx.getModule(), "printf", printfType);
		JamLLVMApplyDefaultFnAttrs(printfFunc, /*isExtern=*/true);
	}
	JamFunctionRef exitFunc = JamLLVMGetFunction(ctx.getModule(), "exit");
	if (!exitFunc) {
		JamTypeRef exitParamTypes[1] = {ctx.getInt32Type()};
		JamTypeRef exitType =
		    JamLLVMFunctionType(ctx.getVoidType(), exitParamTypes, 1, false);
		exitFunc = JamLLVMAddFunction(ctx.getModule(), "exit", exitType);
		JamLLVMApplyDefaultFnAttrs(exitFunc, /*isExtern=*/true);
	}

	JamValueRef fmtStr = JamLLVMBuildGlobalStringPtr(
	    ctx.getBuilder(), "assertion failed\n", "assert_fail_msg");
	JamValueRef printArgs[1] = {fmtStr};
	JamLLVMBuildCall(ctx.getBuilder(), printfFunc, printArgs, 1, "");

	JamValueRef exitCode = JamLLVMConstInt(ctx.getInt32Type(), 1, false);
	JamValueRef exitArgs[1] = {exitCode};
	JamLLVMBuildCall(ctx.getBuilder(), exitFunc, exitArgs, 1, "");
	JamLLVMBuildUnreachable(ctx.getBuilder());

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), passBlock);

	return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
}

// Main switch-dispatched codegen

static JamValueRef codegenStringLit(JamCodegenContext &ctx,
                                    const std::string &val) {
	JamValueRef StrConstant =
	    JamLLVMConstString(ctx.getContext(), val.c_str(), val.length(), true);
	JamTypeRef strArrayType =
	    JamLLVMArrayType(ctx.getInt8Type(), val.length() + 1);
	JamValueRef StrGlobal =
	    JamLLVMAddGlobal(ctx.getModule(), strArrayType, "str");
	JamLLVMSetGlobalConstant(StrGlobal, true);
	JamLLVMSetInitializer(StrGlobal, StrConstant);

	JamTypeRef i8PtrType = JamLLVMPointerType(ctx.getInt8Type(), 0);
	JamTypeRef usizeType = ctx.getInt64Type();
	JamTypeRef sliceTypes[2] = {i8PtrType, usizeType};
	JamTypeRef sliceType =
	    JamLLVMStructType(ctx.getContext(), sliceTypes, 2, false);

	JamValueRef StrPtr =
	    JamLLVMBuildBitCast(ctx.getBuilder(), StrGlobal, i8PtrType, "str_ptr");
	JamValueRef SliceStruct = JamLLVMGetUndef(sliceType);
	SliceStruct = JamLLVMBuildInsertValue(ctx.getBuilder(), SliceStruct, StrPtr,
	                                      0, "slice_ptr");
	SliceStruct = JamLLVMBuildInsertValue(
	    ctx.getBuilder(), SliceStruct,
	    JamLLVMConstInt(usizeType, val.length(), false), 1, "slice_len");
	return SliceStruct;
}

// Best-effort signedness inference for an integer-typed expression.
// LLVM doesn't carry signed/unsigned in its types, so the compiler
// has to recover it from the source-level Jam type. Used by relational
// operators (< <= > >=) to pick between signed and unsigned ICMP
// predicates — `i32(-1) >= 0` must use SGE, not UGE.
//
// Returns true iff the expression is known to produce a signed
// integer value. Returns false for unsigned, non-integer, or
// unknown types (the unsigned predicate is the safer default for
// the cases we can't classify).
static bool isSignedIntExpr(JamCodegenContext &ctx, NodeIdx idx) {
	if (idx == kNoNode) return false;
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	const AstNode &n = ns.get(idx);
	auto isSignedTy = [&](TypeIdx ty) -> bool {
		if (ty == kNoType) return false;
		const TypeKey &k = ctx.getTypePool().get(ty);
		return k.kind == TypeKind::Int && k.b != 0;
	};
	switch (n.tag) {
	case AstTag::NumberLit:
		// A negative literal is inherently signed. A positive literal
		// adopts its consumer's type — be conservative and say
		// unsigned so non-negative literals don't force SGE; the
		// other operand decides.
		return (n.flags & 1) != 0;
	case AstTag::Variable: {
		const std::string &name = sp.get(static_cast<StringIdx>(n.lhs));
		return isSignedTy(ctx.getVariableType(name));
	}
	case AstTag::Call: {
		const std::string &callee = sp.get(static_cast<StringIdx>(n.lhs));
		if (const FunctionAST *fn = ctx.getFunctionAST(callee)) {
			return isSignedTy(fn->ReturnType);
		}
		return false;
	}
	case AstTag::AsCast:
		return isSignedTy(static_cast<TypeIdx>(n.rhs));
	case AstTag::BinaryOp:
	case AstTag::UnaryOp:
		// Recurse on the LHS (operand types match for arithmetic).
		return isSignedIntExpr(ctx, static_cast<NodeIdx>(n.lhs));
	case AstTag::MemberAccess: {
		// `slice.len` is u32 (unsigned). Generic field access:
		// resolve the struct + field type.
		NodeIdx base = static_cast<NodeIdx>(n.lhs);
		StringIdx member = static_cast<StringIdx>(n.rhs);
		const AstNode &bn = ns.get(base);
		if (bn.tag == AstTag::Variable) {
			const std::string &varName = sp.get(static_cast<StringIdx>(bn.lhs));
			TypeIdx baseTy = ctx.getVariableType(varName);
			if (const auto *info = ctx.lookupStruct(baseTy)) {
				const std::string &memName = sp.get(member);
				for (const auto &f : info->fields) {
					if (f.first == memName) { return isSignedTy(f.second); }
				}
			}
		}
		return false;
	}
	default:
		return false;
	}
}

static JamValueRef codegenBinaryOp(JamCodegenContext &ctx, const AstNode &n) {
	BinOp op = static_cast<BinOp>(n.op);
	const NodeStore &ns = ctx.getNodeStore();
	NodeIdx lhsIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx rhsIdx = static_cast<NodeIdx>(n.rhs);

	// Short-circuit logical operators.
	if (op == BinOp::LogAnd || op == BinOp::LogOr) {
		JamValueRef L = codegenNode(ctx, lhsIdx);
		if (!L) return nullptr;

		JamBasicBlockRef currentBlock = JamLLVMGetInsertBlock(ctx.getBuilder());
		JamFunctionRef TheFunction = JamLLVMGetBasicBlockParent(currentBlock);
		JamBasicBlockRef rhsBlock = JamLLVMAppendBasicBlock(
		    TheFunction, op == BinOp::LogAnd ? "and.rhs" : "or.rhs");
		JamBasicBlockRef mergeBlock = JamLLVMAppendBasicBlock(
		    TheFunction, op == BinOp::LogAnd ? "and.end" : "or.end");

		if (op == BinOp::LogAnd) {
			JamLLVMBuildCondBr(ctx.getBuilder(), L, rhsBlock, mergeBlock);
		} else {
			JamLLVMBuildCondBr(ctx.getBuilder(), L, mergeBlock, rhsBlock);
		}

		JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), rhsBlock);
		JamValueRef R = codegenNode(ctx, rhsIdx);
		if (!R) return nullptr;
		JamLLVMBuildBr(ctx.getBuilder(), mergeBlock);
		rhsBlock = JamLLVMGetInsertBlock(ctx.getBuilder());

		JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), mergeBlock);
		JamValueRef phi =
		    JamLLVMBuildPhi(ctx.getBuilder(), ctx.getInt1Type(),
		                    op == BinOp::LogAnd ? "and.result" : "or.result");

		JamValueRef shortCircuitVal = JamLLVMConstInt(
		    ctx.getInt1Type(), op == BinOp::LogAnd ? 0 : 1, false);
		JamValueRef incomingVals[2] = {shortCircuitVal, R};
		JamBasicBlockRef incomingBlocks[2] = {currentBlock, rhsBlock};
		JamLLVMAddIncoming(phi, incomingVals, incomingBlocks, 2);
		return phi;
	}

	// For mixed literal/concrete, materialize the literal at the concrete
	// side's type so the binary op sees aligned widths from the start.
	bool lIsLit = ns.get(lhsIdx).tag == AstTag::NumberLit;
	bool rIsLit = ns.get(rhsIdx).tag == AstTag::NumberLit;
	JamValueRef L = nullptr, R = nullptr;
	if (lIsLit && !rIsLit) {
		R = codegenNode(ctx, rhsIdx);
		JamTypeRef expected = R ? JamLLVMTypeOf(R) : nullptr;
		L = codegenNode(ctx, lhsIdx, expected);
	} else if (!lIsLit && rIsLit) {
		L = codegenNode(ctx, lhsIdx);
		JamTypeRef expected = L ? JamLLVMTypeOf(L) : nullptr;
		R = codegenNode(ctx, rhsIdx, expected);
	} else {
		L = codegenNode(ctx, lhsIdx);
		R = codegenNode(ctx, rhsIdx);
	}
	if (!L || !R) return nullptr;

	// Width-align if still mismatched (two literals of different widths, or
	// two vars of different declared widths).
	JamTypeRef lt = JamLLVMTypeOf(L);
	JamTypeRef rt = JamLLVMTypeOf(R);
	if (lt != rt && JamLLVMTypeIsInteger(lt) && JamLLVMTypeIsInteger(rt)) {
		unsigned lw = JamLLVMGetIntTypeWidth(lt);
		unsigned rw = JamLLVMGetIntTypeWidth(rt);
		if (lw > rw) R = coerceTo(ctx, R, lt);
		else L = coerceTo(ctx, L, rt);
	}
	// note: reject any remaining numeric-width mismatch where at least one
	// side is a float. The int-widening block above already aligned
	// <int->int pairs; what survives here is either `f32 + f64` or a
	// stray <int->float pair. Both would lower to malformed IR
	// (`fadd float, double`, `fadd i32, float`). Implicit widening
	// would silently rebit the narrower operand, so we make the user
	// spell the conversion with `as`.
	if (lt != rt && (JamLLVMTypeIsFloat(lt) || JamLLVMTypeIsFloat(rt))) {
		throw std::runtime_error(
		    "mismatched operand types in binary operation; insert an "
		    "`as` cast so both sides have the same type");
	}

	// route arithmetic to LLVM's F* opcodes and comparisons to FCmp
	// with ordered predicates so NaN never compares true.
	const bool isFloat = JamLLVMTypeIsFloat(lt);

	if (isFloat) {
		switch (op) {
		case BinOp::Add:
			return JamLLVMBuildFAdd(ctx.getBuilder(), L, R, "faddtmp");
		case BinOp::Sub:
			return JamLLVMBuildFSub(ctx.getBuilder(), L, R, "fsubtmp");
		case BinOp::Mul:
			return JamLLVMBuildFMul(ctx.getBuilder(), L, R, "fmultmp");
		case BinOp::Div:
			return JamLLVMBuildFDiv(ctx.getBuilder(), L, R, "fdivtmp");
		case BinOp::Mod:
			return JamLLVMBuildFRem(ctx.getBuilder(), L, R, "fremtmp");
		case BinOp::Eq:
			// OEQ: false when either side is NaN. Matches IEEE-754 and C.
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_OEQ, L, R,
			                        "fcmp");
		case BinOp::Ne:
			// UNE: true when either side is NaN OR operands differ. The
			// alternative (ONE) returns false on NaN inputs, which would
			// make `nan != nan` evaluate false, which is surprising and wrong.
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_UNE, L, R,
			                        "fcmp");
		case BinOp::Lt:
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_OLT, L, R,
			                        "fcmp");
		case BinOp::Le:
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_OLE, L, R,
			                        "fcmp");
		case BinOp::Gt:
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_OGT, L, R,
			                        "fcmp");
		case BinOp::Ge:
			return JamLLVMBuildFCmp(ctx.getBuilder(), JAM_FCMP_OGE, L, R,
			                        "fcmp");
		case BinOp::BitAnd:
		case BinOp::BitOr:
		case BinOp::BitXor:
		case BinOp::Shl:
		case BinOp::Shr:
			throw std::runtime_error(
			    "bitwise / shift operators are not defined for float operands");
		default:
			throw std::runtime_error("Invalid float binary operator");
		}
	}

	switch (op) {
	case BinOp::Add:
		return JamLLVMBuildAdd(ctx.getBuilder(), L, R, "addtmp");
	case BinOp::Sub:
		return JamLLVMBuildSub(ctx.getBuilder(), L, R, "subtmp");
	case BinOp::Mul:
		return JamLLVMBuildMul(ctx.getBuilder(), L, R, "multmp");
	case BinOp::Div: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return signed_ ? JamLLVMBuildSDiv(ctx.getBuilder(), L, R, "divtmp")
		               : JamLLVMBuildUDiv(ctx.getBuilder(), L, R, "divtmp");
	}
	case BinOp::Mod: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return signed_ ? JamLLVMBuildSRem(ctx.getBuilder(), L, R, "remtmp")
		               : JamLLVMBuildURem(ctx.getBuilder(), L, R, "remtmp");
	}
	case BinOp::BitAnd:
		return JamLLVMBuildAnd(ctx.getBuilder(), L, R, "andtmp");
	case BinOp::BitOr:
		return JamLLVMBuildOr(ctx.getBuilder(), L, R, "ortmp");
	case BinOp::BitXor:
		return JamLLVMBuildXor(ctx.getBuilder(), L, R, "xortmp");
	case BinOp::Shl:
		return JamLLVMBuildShl(ctx.getBuilder(), L, R, "shltmp");
	case BinOp::Shr: {
		// this one was a pain because:
		// arithmetic shift right for signed operands so the sign bit
		// propagates (matches C's behaviour for signed >>); logical
		// shift for unsigned. Without this an `i64` value of -16 >> 12
		// became a huge positive number, which broke the psone emulator
		// IR-clamp pipeline and made every 3D scene render as a
		// stretched streak.
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return signed_ ? JamLLVMBuildAShr(ctx.getBuilder(), L, R, "shrtmp")
		               : JamLLVMBuildLShr(ctx.getBuilder(), L, R, "shrtmp");
	}
	case BinOp::Eq:
		return JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_EQ, L, R, "cmptmp");
	case BinOp::Ne:
		return JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_NE, L, R, "cmptmp");
	case BinOp::Lt: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return JamLLVMBuildICmp(ctx.getBuilder(),
		                        signed_ ? JAM_ICMP_SLT : JAM_ICMP_ULT, L, R,
		                        "cmptmp");
	}
	case BinOp::Le: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return JamLLVMBuildICmp(ctx.getBuilder(),
		                        signed_ ? JAM_ICMP_SLE : JAM_ICMP_ULE, L, R,
		                        "cmptmp");
	}
	case BinOp::Gt: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return JamLLVMBuildICmp(ctx.getBuilder(),
		                        signed_ ? JAM_ICMP_SGT : JAM_ICMP_UGT, L, R,
		                        "cmptmp");
	}
	case BinOp::Ge: {
		bool signed_ =
		    isSignedIntExpr(ctx, lhsIdx) || isSignedIntExpr(ctx, rhsIdx);
		return JamLLVMBuildICmp(ctx.getBuilder(),
		                        signed_ ? JAM_ICMP_SGE : JAM_ICMP_UGE, L, R,
		                        "cmptmp");
	}
	default:
		throw std::runtime_error("Invalid binary operator");
	}
}

static JamValueRef codegenUnaryOp(JamCodegenContext &ctx, const AstNode &n) {
	UnaryOp op = static_cast<UnaryOp>(n.op);
	JamValueRef operandVal = codegenNode(ctx, static_cast<NodeIdx>(n.lhs));
	if (!operandVal) return nullptr;
	switch (op) {
	case UnaryOp::LogNot:
		return JamLLVMBuildXor(ctx.getBuilder(), operandVal,
		                       JamLLVMConstInt(ctx.getInt1Type(), 1, false),
		                       "nottmp");
	case UnaryOp::BitNot: {
		JamTypeRef ty = JamLLVMTypeOf(operandVal);
		JamValueRef allOnes = JamLLVMConstInt(ty, ~0ULL, false);
		return JamLLVMBuildXor(ctx.getBuilder(), operandVal, allOnes, "nottmp");
	}
	case UnaryOp::Neg: {
		JamTypeRef ty = JamLLVMTypeOf(operandVal);
		if (JamLLVMTypeIsFloat(ty)) {
			return JamLLVMBuildFNeg(ctx.getBuilder(), operandVal, "fnegtmp");
		}
		JamValueRef zero = JamLLVMConstInt(ty, 0, false);
		return JamLLVMBuildSub(ctx.getBuilder(), zero, operandVal, "negtmp");
	}
	default:
		throw std::runtime_error("Invalid unary operator");
	}
}

// Comptime intrinsic dispatch (`@name(T)`). Stage 1 supports two
// intrinsics — `sizeOf` returns the byte size of a type as u64;
// `alignOf` returns the alignment as u8. Both consult the codegen
// context's typeSize/typeAlign which already compute these for
// struct layout and ABI purposes. The emitted IR is a literal
// constant; LLVM never sees a call. Stage 2 will extend this
// dispatch to user-defined cfn bodies once CTFE lands.
static JamValueRef codegenAtCall(JamCodegenContext &ctx, const AstNode &n) {
	const std::string &name =
	    ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	TypeIdx tyArg = static_cast<TypeIdx>(n.rhs);

	if (name == "sizeOf") {
		uint64_t bytes = ctx.typeSize(tyArg);
		return JamLLVMConstInt(ctx.getInt64Type(), bytes, false);
	}
	if (name == "alignOf") {
		uint64_t a = ctx.typeAlign(tyArg);
		return JamLLVMConstInt(ctx.getInt8Type(), a, false);
	}
	throw std::runtime_error("Unknown comptime intrinsic: @" + name);
}

// Static method / variant call on a generic-call type receiver
// (`Vec(i32).empty()`, `Option(i32).Some(x)`, etc.). The receiver
// TypeIdx is parser-built via internGenericCall. We resolve it to a
// concrete struct or enum (instantiating the generic if needed),
// then synthesize a regular Call AST node with the canonical
// qualified name and delegate to codegenCall.
static JamValueRef codegenCall(JamCodegenContext &ctx, const AstNode &n);
static JamValueRef codegenTypeMethodCall(JamCodegenContext &ctx,
                                         const AstNode &n) {
	TypeIdx recvTy = static_cast<TypeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	StringIdx methodNameId =
	    static_cast<StringIdx>(ctx.getNodeStore().getExtra(extra));
	uint32_t argCount = ctx.getNodeStore().getExtra(extra + 1);
	const std::string &methodName = ctx.getStringPool().get(methodNameId);

	// Resolve receiver to a struct or enum name. lookupStruct /
	// lookupEnum handle GenericCall TypeIdx via resolveGenericCall.
	std::string receiverName;
	if (const auto *sinfo = ctx.lookupStruct(recvTy)) {
		receiverName = sinfo->name;
	} else if (const auto *einfo = ctx.lookupEnum(recvTy)) {
		receiverName = einfo->name;
	} else {
		throw std::runtime_error(
		    "Type-method-call receiver doesn't resolve to a struct "
		    "or enum");
	}

	// Synthesize a regular Call AST node with the canonical qualified
	// name. The args (NodeIdx list in the TypeMethodCall extra) are
	// reused as-is for the synthesized Call. Then codegenCall handles
	// static-method dispatch + enum-variant construction uniformly.
	std::string qualified = receiverName + "." + methodName;
	StringIdx calleeId = ctx.getStringPool().intern(qualified);
	ExtraIdx callExtra = ctx.getNodeStore().reserveExtra(1 + argCount);
	ctx.getNodeStore().setExtra(callExtra, argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		ctx.getNodeStore().setExtra(callExtra + 1 + i,
		                            ctx.getNodeStore().getExtra(extra + 2 + i));
	}
	AstNode synth{AstTag::Call, 0, 0, 0, static_cast<uint32_t>(calleeId),
	              callExtra};
	return codegenCall(ctx, synth);
}

static JamValueRef codegenCall(JamCodegenContext &ctx, const AstNode &n) {
	if (n.flags & 1) {
		NodeIdx calleeNodeIdx = static_cast<NodeIdx>(n.lhs);
		const AstNode &cn = ctx.getNodeStore().get(calleeNodeIdx);
		if (cn.tag != AstTag::MemberAccess) {
			throw std::runtime_error(
			    "Indirect calls are only supported on method receivers — "
			    "the callee must end in `.methodName`");
		}
		NodeIdx receiverIdx = static_cast<NodeIdx>(cn.lhs);
		StringIdx methodNameId = static_cast<StringIdx>(cn.rhs);
		const std::string &methodName = ctx.getStringPool().get(methodNameId);

		TypeIdx recvTy = inferLValueType(ctx, receiverIdx);
		if (recvTy == kNoType) {
			throw std::runtime_error("Cannot infer receiver type for method `" +
			                         methodName + "`");
		}
		const auto *sinfo = ctx.lookupStruct(recvTy);
		if (!sinfo) {
			throw std::runtime_error(
			    "Method call receiver is not a struct (method `" + methodName +
			    "`)");
		}
		std::string qualified = sinfo->name + "." + methodName;
		const FunctionAST *methodAST = ctx.getFunctionAST(qualified);
		if (!methodAST || methodAST->Args.empty()) {
			throw std::runtime_error("Method `" + methodName +
			                         "` not found on struct `" + sinfo->name +
			                         "`");
		}

		NodeIdx recvArgNode = receiverIdx;
		ParamMode mode = methodAST->Args[0].Mode;
		if (mode == ParamMode::Mut || mode == ParamMode::Move) {
			recvArgNode = ctx.getNodeStore().addNode(
			    AstNode{AstTag::AddressOf, 0, 0, 0,
			            static_cast<uint32_t>(receiverIdx), 0});
		}

		ExtraIdx origExtra = static_cast<ExtraIdx>(n.rhs);
		uint32_t origCount = ctx.getNodeStore().getExtra(origExtra);
		StringIdx calleeId = ctx.getStringPool().intern(qualified);
		ExtraIdx callExtra = ctx.getNodeStore().reserveExtra(2 + origCount);
		ctx.getNodeStore().setExtra(callExtra, 1 + origCount);
		ctx.getNodeStore().setExtra(callExtra + 1,
		                            static_cast<uint32_t>(recvArgNode));
		for (uint32_t i = 0; i < origCount; i++) {
			ctx.getNodeStore().setExtra(
			    callExtra + 2 + i,
			    ctx.getNodeStore().getExtra(origExtra + 1 + i));
		}
		AstNode synth{AstTag::Call, 0, 0, 0, static_cast<uint32_t>(calleeId),
		              callExtra};
		return codegenCall(ctx, synth);
	}

	// `callee` may be rewritten below (e.g. instance-receiver method
	// dispatch turns `inst.method` into `StructName.method`), so it's
	// a value not a reference.
	std::string callee = ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = ctx.getNodeStore().getExtra(extra);
	std::vector<uint32_t> args(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		args[i] = ctx.getNodeStore().getExtra(extra + 1 + i);
	}

	if (callee == "std.fmt.print" || callee == "std.fmt.println") {
		return genPrintCall(ctx, callee, args.data(), argCount);
	}
	if (callee == "std.thread.sleep") {
		return genSleepCall(ctx, args.data(), argCount);
	}
	if (callee == "assert") {
		return genAssertCall(ctx, args.data(), argCount);
	}

	// Enum-variant constructor: `EnumName.VariantName(arg0, arg1, …)`
	// builds a tagged enum value. The qualified callee name has shape
	// `<EnumName>.<VariantName>` — split it and check the registry.
	{
		size_t dot = callee.find('.');
		if (dot != std::string::npos &&
		    callee.find('.', dot + 1) == std::string::npos) {
			std::string enumName = callee.substr(0, dot);
			std::string varName = callee.substr(dot + 1);
			// Resolve aliases (`const OptI32 = Option(i32);`) to the
			// canonical enum name so generic enum variant construction
			// works through the alias.
			if (!ctx.getEnum(enumName)) {
				TypeIdx aliasTarget = ctx.lookupTypeAlias(enumName);
				if (aliasTarget != kNoType) {
					if (const auto *info = ctx.lookupEnum(aliasTarget)) {
						enumName = info->name;
					}
				}
			}
			if (const auto *einfo = ctx.getEnum(enumName)) {
				int idx = ctx.getEnumVariantIndex(enumName, varName);
				if (idx < 0) {
					throw std::runtime_error("Enum `" + enumName +
					                         "` has no variant `" + varName +
					                         "`");
				}
				const auto &v = einfo->variants[idx];
				if (argCount != v.payloadTypes.size()) {
					throw std::runtime_error(
					    "Variant `" + enumName + "." + varName + "` expects " +
					    std::to_string(v.payloadTypes.size()) +
					    " payload arg(s), got " + std::to_string(argCount));
				}
				// unit-only enum: just the tag (discriminant value).
				if (!einfo->hasPayloadVariant) {
					return JamLLVMConstInt(
					    ctx.getInt8Type(),
					    static_cast<uint64_t>(v.discriminant), false);
				}
				JamTypeRef enumLLVMType = einfo->type;
				// Allocate an enum-typed slot, store the tag at offset
				// 0, and store each payload arg at its computed offset
				// within the payload area. Then load the whole struct
				// to return as an SSA value.
				uint64_t enumAlign =
				    einfo->maxPayloadAlign > 1 ? einfo->maxPayloadAlign : 1;
				JamValueRef alloca = JamLLVMBuildAlloca(
				    ctx.getBuilder(), enumLLVMType, enumAlign, "enum.tmp");
				// Store tag at field 0.
				JamValueRef tagPtr = JamLLVMBuildStructGEP(
				    ctx.getBuilder(), enumLLVMType, alloca, 0, "enum.tag");
				JamLLVMBuildStore(
				    ctx.getBuilder(),
				    JamLLVMConstInt(ctx.getInt8Type(),
				                    static_cast<uint64_t>(v.discriminant),
				                    false),
				    tagPtr);
				// If variant has payload, store fields. Payload area is
				// the LAST field of the enum struct (index 1 if no
				// padding, index 2 if padding bytes inserted).
				if (!v.payloadTypes.empty()) {
					// Payload starts at struct field 1 (the
					// alignment-driving scalar); larger payloads spill
					// into field 2 via byte-offset GEP from field 1.
					unsigned payloadFieldIdx = 1;
					JamValueRef payloadAreaPtr = JamLLVMBuildStructGEP(
					    ctx.getBuilder(), enumLLVMType, alloca, payloadFieldIdx,
					    "enum.payload");
					(void)einfo;
					uint64_t off = 0;
					for (size_t i = 0; i < v.payloadTypes.size(); i++) {
						uint64_t s = ctx.typeSize(v.payloadTypes[i]);
						uint64_t a = ctx.typeAlign(v.payloadTypes[i]);
						off = (off + a - 1) / a * a;
						// GEP into the payload bytes at offset `off`.
						JamValueRef i64Off =
						    JamLLVMConstInt(ctx.getInt64Type(), off, false);
						JamValueRef fieldPtr = JamLLVMBuildPtrGEP(
						    ctx.getBuilder(), ctx.getInt8Type(), payloadAreaPtr,
						    i64Off, "enum.field.ptr");
						JamTypeRef fieldTy = ctx.getLLVMType(v.payloadTypes[i]);
						JamValueRef argVal = codegenNode(ctx, args[i], fieldTy);
						if (!argVal) return nullptr;
						argVal = coerceTo(ctx, argVal, fieldTy);
						JamLLVMBuildStore(ctx.getBuilder(), argVal, fieldPtr);
						off += s;
					}
				}
				// Load the constructed enum value from the alloca.
				return JamLLVMBuildLoad(ctx.getBuilder(), enumLLVMType, alloca,
				                        "enum.val");
			}
		}
	}

	// Struct-method qualified call: `Struct.method(args...)`. Methods are
	// registered in the function table under their qualified source-level
	// name; the LLVM symbol is whatever mangledFunctionName produces
	// (drop methods share the existing `__drop_T` mangling). The rest of
	// this function uses `callee` (source-level) for FunctionAST lookups
	// and `llvmName` for the LLVM symbol — they only differ for methods.
	std::string llvmName = callee;
	std::string lookupName = callee;

	// Multi-dot instance-method dispatch: `self.field.method(args)`,
	// `outer.inner.fn(args)`, etc. The single-dot block below handles
	// the simple `instance.method` and `Type.method` shapes; this
	// block extends the dispatch to walk a chain of field accesses
	// from a root variable to the receiver value, then look up the
	// method on the receiver's type. Synthesizes a MemberAccess chain
	// for the receiver AST so the rest of codegenCall sees it as a
	// regular first argument.
	{
		size_t firstDot = callee.find('.');
		size_t lastDot = callee.rfind('.');
		if (firstDot != std::string::npos && lastDot != firstDot) {
			std::string rootVar = callee.substr(0, firstDot);
			std::string methodPart = callee.substr(lastDot + 1);
			std::string middle =
			    callee.substr(firstDot + 1, lastDot - firstDot - 1);
			// Split the middle "a.b.c" into ["a", "b", "c"].
			std::vector<std::string> fieldChain;
			{
				size_t p = 0;
				while (p <= middle.size()) {
					size_t n = middle.find('.', p);
					if (n == std::string::npos) {
						fieldChain.push_back(middle.substr(p));
						break;
					}
					fieldChain.push_back(middle.substr(p, n - p));
					p = n + 1;
				}
			}

			if (ctx.getVariable(rootVar) != nullptr) {
				TypeIdx recvTy = ctx.getVariableType(rootVar);
				// Walk each field, descending into nested structs.
				for (const auto &fld : fieldChain) {
					const auto *info = ctx.lookupStruct(recvTy);
					if (!info) {
						recvTy = kNoType;
						break;
					}
					bool found = false;
					for (const auto &f : info->fields) {
						if (f.first == fld) {
							recvTy = f.second;
							found = true;
							break;
						}
					}
					if (!found) {
						recvTy = kNoType;
						break;
					}
				}

				if (recvTy != kNoType) {
					if (const auto *sinfo = ctx.lookupStruct(recvTy)) {
						std::string canonicalCallee =
						    sinfo->name + "." + methodPart;
						const FunctionAST *methodAST =
						    ctx.getFunctionAST(canonicalCallee);
						if (methodAST && !methodAST->Args.empty()) {
							// Synthesize the receiver as a chain of
							// MemberAccess nodes rooted at Variable.
							StringIdx rootId =
							    ctx.getStringPool().intern(rootVar);
							NodeIdx recvNode = ctx.getNodeStore().addNode(
							    AstNode{AstTag::Variable, 0, 0, 0, rootId, 0});
							for (const auto &fld : fieldChain) {
								StringIdx fldId =
								    ctx.getStringPool().intern(fld);
								recvNode = ctx.getNodeStore().addNode(AstNode{
								    AstTag::MemberAccess, 0, 0, 0,
								    static_cast<uint32_t>(recvNode), fldId});
							}
							ParamMode m = methodAST->Args[0].Mode;
							if (m == ParamMode::Mut || m == ParamMode::Move) {
								recvNode = ctx.getNodeStore().addNode(AstNode{
								    AstTag::AddressOf, 0, 0, 0,
								    static_cast<uint32_t>(recvNode), 0});
							}
							args.insert(args.begin(), recvNode);
							argCount++;
							callee = canonicalCallee;
						}
					}
				}
			}
		}
	}

	{
		size_t dot = callee.find('.');
		if (dot != std::string::npos &&
		    callee.find('.', dot + 1) == std::string::npos) {
			std::string typeName = callee.substr(0, dot);
			std::string methodPart = callee.substr(dot + 1);

			// Instance-receiver method call: `instance.method(args)`.
			// The LHS is a value (variable, including `self` inside
			// another method's body), not a type name. Look up its
			// type, find the method on that struct, and prepend a
			// synthesized AST node for the receiver to the arg list
			// so the rest of this function sees the call shape
			// `StructName.method(receiver, args...)`.
			//
			// The receiver node is either `Variable(name)` (for
			// `let` self — pass by value, codegen auto-addresses if
			// the ABI demands it for size > 16B aggregates) or
			// `AddressOf(Variable(name))` (for `mut` / `move` /
			// `undefined` self — the call site needs an explicit
			// address to match the user-visible borrow shape).
			if (ctx.getVariable(typeName) != nullptr) {
				TypeIdx recvTy = ctx.getVariableType(typeName);
				if (recvTy != kNoType) {
					if (const auto *sinfo = ctx.lookupStruct(recvTy)) {
						// Look up the method AST to pick the right
						// receiver shape based on its first-param
						// mode (let vs mut/move/undefined).
						std::string canonicalCallee =
						    sinfo->name + "." + methodPart;
						const FunctionAST *methodAST =
						    ctx.getFunctionAST(canonicalCallee);
						if (methodAST && !methodAST->Args.empty()) {
							StringIdx recvName =
							    ctx.getStringPool().intern(typeName);
							NodeIdx recvNode =
							    ctx.getNodeStore().addNode(AstNode{
							        AstTag::Variable, 0, 0, 0, recvName, 0});
							ParamMode m = methodAST->Args[0].Mode;
							if (m == ParamMode::Mut || m == ParamMode::Move) {
								// Wrap in AddressOf — the method
								// expects a pointer-shaped self.
								recvNode = ctx.getNodeStore().addNode(AstNode{
								    AstTag::AddressOf, 0, 0, 0,
								    static_cast<uint32_t>(recvNode), 0});
							}
							args.insert(args.begin(), recvNode);
							argCount++;

							// Rewrite callee so the canonical-type
							// path below picks up the right method.
							typeName = sinfo->name;
							callee = canonicalCallee;
						}
					}
				}
			}

			// if `typeName` is `Self` and we're inside
			// the body of an instantiated method, the substitution
			// context maps `Self` to the synthetic anon-struct name,
			// which in turn maps to the instantiated struct's Named
			// TypeIdx. Resolve it to the canonical instantiated
			// struct name (e.g. `Box__i32`).
			//
			// We also handle the synthetic name directly because the
			// parser already rewrote `Self` to `__anon_struct_<N>`
			// in *type* positions (see parseType), but not in
			// *expression* positions like `Self.make(...)`.
			if (typeName == "Self") {
				TypeIdx selfTy = ctx.lookupCurrentSubst("Self");
				if (selfTy == kNoType) {
					// Fall through to the regular lookup; the
					// parser-time anon-struct name might have been
					// substituted via the value-position path.
				} else if (const auto *sinfo = ctx.lookupStruct(selfTy)) {
					typeName = sinfo->name;
				}
			} else {
				TypeIdx ctxTy = ctx.lookupCurrentSubst(typeName);
				if (ctxTy != kNoType) {
					if (const auto *sinfo = ctx.lookupStruct(ctxTy)) {
						typeName = sinfo->name;
					}
				}
			}

			// if the LHS is a type alias
			// (`const BoxI32 = Box(i32);`), resolve it to the
			// canonical instantiated struct name (`Box__i32`) and
			// look up the method there. Both `BoxI32.unwrap` and
			// `Box__i32.unwrap` dispatch to the same instantiated
			// method.
			std::string canonicalType = typeName;
			TypeIdx aliasTarget = ctx.lookupTypeAlias(typeName);
			if (aliasTarget != kNoType) {
				if (const auto *sinfo = ctx.lookupStruct(aliasTarget)) {
					canonicalType = sinfo->name;
				}
			}

			if (ctx.getStruct(canonicalType)) {
				std::string canonical = canonicalType + "." + methodPart;
				const FunctionAST *methodAST = ctx.getFunctionAST(canonical);
				if (!methodAST) { methodAST = ctx.getFunctionAST(callee); }
				if (methodAST) {
					lookupName = canonical;
					llvmName = mangledFunctionName(
					    *methodAST, ctx.getTypePool(), ctx.getStringPool());
				} else {
					// LHS resolved to a real struct, but it has no
					// method by that name. Report this specifically —
					// the most common case is a generic instantiation
					// where the substituted T lacks an expected method
					// (e.g. `Maybe(NoDefault).default()` body calling
					// `T.default()` when NoDefault has no default()).
					throw std::runtime_error("type `" + canonicalType +
					                         "` has no method `" + methodPart +
					                         "`");
				}
			}
		}
	}

	JamFunctionRef CalleeF =
	    JamLLVMGetFunction(ctx.getModule(), llvmName.c_str());
	if (!CalleeF && llvmName == callee) {
		size_t dotPos = callee.find('.');
		if (dotPos != std::string::npos &&
		    ctx.getFunctionAST(callee) != nullptr) {
			std::string bare = callee.substr(dotPos + 1);
			CalleeF = JamLLVMGetFunction(ctx.getModule(), bare.c_str());
			if (CalleeF) llvmName = bare;
		}
	}
	if (!CalleeF) {
		throw std::runtime_error(
		    ctx.formatNamespaceLookupError("function", callee));
	}

	// when the callee's parameter is a Let/Move-mode aggregate that
	// the ABI classifier sends ByPointer (size > 16 B), the call site
	// implicitly passes the address of the arg's storage rather than the
	// value. The user does NOT write `&` for these — by-value semantics
	// are preserved at the source level; the pointer is purely an ABI
	// optimization. (Mut and Undefined modes still require explicit `&`
	// at the call site since their borrow shape is user-visible.)
	//
	// lookupName is the resolved name (alias-canonicalized)
	// for instantiated methods. For non-method calls it equals callee.
	const FunctionAST *calleeAST = ctx.getFunctionAST(lookupName);
	if (!calleeAST) calleeAST = ctx.getFunctionAST(callee);

	// sret: when the callee returns a large aggregate, the call
	// site allocates a result slot and passes its address as the
	// leading argument. The call itself returns void; the "value" of
	// the call expression is a load from the slot.
	JamValueRef calleeSretSlot = nullptr;
	JamTypeRef calleeSretPointee = nullptr;
	if (calleeAST && !calleeAST->isExtern && calleeAST->ReturnType != kNoType) {
		jam::abi::ReturnABI calleeRabi =
		    jam::abi::classifyReturn(calleeAST->ReturnType, ctx);
		if (calleeRabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
			calleeSretPointee = ctx.getLLVMType(calleeAST->ReturnType);
			calleeSretSlot = JamLLVMBuildAlloca(
			    ctx.getBuilder(), calleeSretPointee,
			    ctx.typeAlign(calleeAST->ReturnType), "sretslot");
		}
	}

	// The user's arg count needs to match the callee's *source-level*
	// parameter count. The LLVM declaredParamCount also includes the
	// sret slot when applicable; compensate so the user-facing error
	// message matches what the user wrote.
	unsigned declaredParamCount = JamLLVMCountParams(CalleeF);
	unsigned userParamCount =
	    declaredParamCount - (calleeSretSlot != nullptr ? 1u : 0u);
	bool isVarArg = JamLLVMFunctionIsVarArg(CalleeF);
	if (isVarArg ? argCount < userParamCount : argCount != userParamCount) {
		throw std::runtime_error("Incorrect number of arguments passed to " +
		                         callee);
	}

	const unsigned calleeSretOffset = (calleeSretSlot != nullptr) ? 1u : 0u;

	std::vector<JamValueRef> ArgsV;
	if (calleeSretSlot != nullptr) { ArgsV.push_back(calleeSretSlot); }
	for (unsigned i = 0; i < argCount; i++) {
		if (i < userParamCount) {
			JamTypeRef expected =
			    JamLLVMTypeOf(JamLLVMGetParam(CalleeF, i + calleeSretOffset));

			bool autoAddress = false;
			// extern callees follow the C ABI literally; never
			// auto-address. The user wrote the FFI-facing parameter
			// types directly, and LLVM's backend handles any byval
			// passing per the platform's MEMORY classification.
			if (calleeAST && !calleeAST->isExtern &&
			    i < calleeAST->Args.size()) {
				const Param &p = calleeAST->Args[i];
				// `mut` params are always ByPointer (see classifyParam);
				// the call site needs to forward an lvalue address.
				// `let` / `move` get the same treatment only when the
				// aggregate's size pushes the ABI past kByValueMaxBytes.
				if (p.Mode == ParamMode::Let || p.Mode == ParamMode::Move ||
				    p.Mode == ParamMode::Mut) {
					auto pabi = jam::abi::classifyParam(p.Mode, p.Type, ctx);
					if (pabi.kind == jam::abi::ParamABI::Kind::ByPointer) {
						autoAddress = true;
					}
				}
			}

			JamValueRef argVal;
			if (autoAddress) {
				const AstNode &argNode = ctx.getNodeStore().get(args[i]);
				if (argNode.tag == AstTag::AddressOf) {
					argVal = codegenNode(ctx, args[i]);
					if (!argVal) return nullptr;
				} else if (argNode.tag == AstTag::Variable ||
				           argNode.tag == AstTag::Index ||
				           argNode.tag == AstTag::MemberAccess) {
					// lvalue: take its address directly via the same
					// path that the explicit `&` operator uses.
					AstNode fakeAddrOf{AstTag::AddressOf,
					                   0,
					                   0,
					                   0,
					                   static_cast<uint32_t>(args[i]),
					                   0};
					argVal = codegenAddressOf(ctx, fakeAddrOf);
				} else {
					// rvalue: codegen as a value, store into a fresh
					// stack slot, pass the slot's address.
					JamTypeRef llvmTy =
					    ctx.getLLVMType(calleeAST->Args[i].Type);
					JamValueRef tmp = JamLLVMBuildAlloca(
					    ctx.getBuilder(), llvmTy,
					    ctx.typeAlign(calleeAST->Args[i].Type), "argtmp");
					JamValueRef val = codegenNode(ctx, args[i], llvmTy);
					if (!val) return nullptr;
					val = coerceTo(ctx, val, llvmTy);
					JamLLVMBuildStore(ctx.getBuilder(), val, tmp);
					argVal = tmp;
				}
			} else {
				argVal = codegenNode(ctx, args[i], expected);
				if (!argVal) return nullptr;
				argVal = coerceTo(ctx, argVal, expected);
			}
			ArgsV.push_back(argVal);
		} else {
			JamValueRef argVal = codegenNode(ctx, args[i]);
			if (!argVal) return nullptr;
			ArgsV.push_back(argVal);
		}
	}

	const char *callName =
	    JamLLVMTypeIsVoid(JamLLVMGetReturnType(CalleeF)) ? "" : "calltmp";
	JamValueRef callResult = JamLLVMBuildCall(
	    ctx.getBuilder(), CalleeF, ArgsV.data(), ArgsV.size(), callName);

	// when sret was used, the call returned void and the actual
	// result lives in the pre-allocated slot. Load it so the surrounding
	// expression sees the value.
	if (calleeSretSlot != nullptr) {
		return JamLLVMBuildLoad(ctx.getBuilder(), calleeSretPointee,
		                        calleeSretSlot, "sretload");
	}
	// callee declared `noreturn`: control never returns. Terminate the
	// current block with `unreachable` so any code emitted after this
	// expression statement is dead from the CFG's perspective.
	if (calleeAST && calleeAST->ReturnType == BuiltinType::NoReturn) {
		JamLLVMBuildUnreachable(ctx.getBuilder());
	}
	return callResult;
}

// Forward decls — these helpers are defined further down (alongside
// FunctionAST codegen for mangledFunctionName, alongside codegenVarDecl
// for the drop emitters) so the source order can stay readable without
// re-ordering large blocks.
static std::string mangledFunctionName(const FunctionAST &fn,
                                       const TypePool &types,
                                       const StringPool &strings);
static void emitTopScopeDrops(JamCodegenContext &ctx);
static void emitAllScopeDrops(JamCodegenContext &ctx);

static JamValueRef codegenReturn(JamCodegenContext &ctx, const AstNode &n) {
	NodeIdx retIdx = static_cast<NodeIdx>(n.lhs);
	if (retIdx == kNoNode) {
		emitAllScopeDrops(ctx);
		JamLLVMBuildRetVoid(ctx.getBuilder());
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	}

	// If the return value is a struct literal with no explicit target
	// type (the parser leaves d.lhs == kNoType), patch it from the
	// enclosing function's return TypeIdx so codegenStructLit knows
	// what to construct. Mirrors what codegenVarDecl does for var
	// decls with struct-literal initializers.
	const AstNode &retNode = ctx.getNodeStore().get(retIdx);
	if (retNode.tag == AstTag::StructLit &&
	    static_cast<TypeIdx>(retNode.lhs) == kNoType) {
		TypeIdx retTy = ctx.getCurrentReturnType();
		if (retTy != kNoType && ctx.lookupStruct(retTy)) {
			ctx.getNodeStore().getMut(retIdx).lhs = retTy;
		}
	}
	if ((retNode.tag == AstTag::ArrayLit ||
	     retNode.tag == AstTag::ArrayRepeat) &&
	    static_cast<TypeIdx>(retNode.lhs) == kNoType) {
		TypeIdx retTy = ctx.getCurrentReturnType();
		if (retTy != kNoType) {
			const TypeKey &k = ctx.getTypePool().get(retTy);
			if (k.kind == TypeKind::Array) {
				ctx.getNodeStore().getMut(retIdx).lhs = retTy;
			}
		}
	}

	// when the enclosing function returns via sret, the codegen
	// stores the return value into the caller-provided slot rather than
	// returning by value. The slot's LLVM pointee type tells us what
	// shape the codegen of the return expression should produce.
	JamValueRef sretSlot = ctx.getSretSlot();
	if (sretSlot != nullptr) {
		JamFunctionRef func =
		    JamLLVMGetBasicBlockParent(JamLLVMGetInsertBlock(ctx.getBuilder()));
		(void)func;  // sret functions return void; LLVM-level RetType is void
		JamValueRef RetVal = codegenNode(ctx, retIdx);
		if (!RetVal) return nullptr;
		JamLLVMBuildStore(ctx.getBuilder(), RetVal, sretSlot);
		emitAllScopeDrops(ctx);
		JamLLVMBuildRetVoid(ctx.getBuilder());
		return RetVal;
	}

	JamFunctionRef func =
	    JamLLVMGetBasicBlockParent(JamLLVMGetInsertBlock(ctx.getBuilder()));
	JamTypeRef expected = JamLLVMGetReturnType(func);
	// Codegen the return value BEFORE emitting drops so reads of any
	// drop-bearing binding still see live storage. Then emit drops for
	// every active scope (innermost first), then the actual ret.
	JamValueRef RetVal = codegenNode(ctx, retIdx, expected);
	if (!RetVal) return nullptr;
	if (JamLLVMTypeIsInteger(expected)) {
		RetVal = coerceTo(ctx, RetVal, expected);
	}
	emitAllScopeDrops(ctx);
	JamLLVMBuildRet(ctx.getBuilder(), RetVal);
	return RetVal;
}

static JamValueRef codegenAssign(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	NodeIdx targetIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx valueIdx = static_cast<NodeIdx>(n.rhs);
	const AstNode &target = ns.get(targetIdx);

	// x = value
	if (target.tag == AstTag::Variable) {
		const std::string &name = sp.get(static_cast<StringIdx>(target.lhs));
		JamValueRef alloca = ctx.getVariable(name);
		if (!alloca) { throw std::runtime_error("Unknown variable: " + name); }
		JamTypeRef expected = ctx.getLLVMType(ctx.getVariableType(name));
		JamValueRef rhsVal = codegenNode(ctx, valueIdx, expected);
		if (!rhsVal) return nullptr;
		rhsVal = coerceTo(ctx, rhsVal, expected);
		JamLLVMBuildStore(ctx.getBuilder(), rhsVal, alloca);
		return rhsVal;
	}

	// p.* = value
	if (target.tag == AstTag::Deref) {
		const AstNode &op = ns.get(static_cast<NodeIdx>(target.lhs));
		if (op.tag != AstTag::Variable) {
			throw std::runtime_error(
			    "`.* =` is only supported on a pointer-typed variable");
		}
		const std::string &name = sp.get(static_cast<StringIdx>(op.lhs));
		JamValueRef alloca = ctx.getVariable(name);
		if (!alloca) { throw std::runtime_error("Unknown variable: " + name); }
		TypeIdx pointee =
		    pointeeTypeOf(ctx.getTypePool(), ctx.getVariableType(name));
		if (pointee == kNoType) {
			throw std::runtime_error("Cannot dereference non-pointer: " + name);
		}
		JamTypeRef pointeeType = ctx.getLLVMType(pointee);
		JamTypeRef ptrType = ctx.getLLVMType(ctx.getVariableType(name));
		JamValueRef ptrVal =
		    JamLLVMBuildLoad(ctx.getBuilder(), ptrType, alloca, name.c_str());
		JamValueRef rhsVal = codegenNode(ctx, valueIdx, pointeeType);
		if (!rhsVal) return nullptr;
		rhsVal = coerceTo(ctx, rhsVal, pointeeType);
		JamLLVMBuildStore(ctx.getBuilder(), rhsVal, ptrVal);
		return rhsVal;
	}

	// Any other lvalue shape (Index, MemberAccess, or arbitrary interleavings
	// like `arr[i].field`, `g.things[i].pc`, `s.ptr[i]`) flows through the
	// shared resolver.
	if (target.tag == AstTag::Index || target.tag == AstTag::MemberAccess) {
		JamValueRef rhsVal = codegenNode(ctx, valueIdx);
		if (!rhsVal) return nullptr;
		LValueAddress addr = resolveLValueAddress(ctx, targetIdx);
		rhsVal = coerceTo(ctx, rhsVal, addr.leafLLVMType);
		JamLLVMBuildStore(ctx.getBuilder(), rhsVal, addr.ptr);
		return rhsVal;
	}

	throw std::runtime_error("Invalid assignment target");
}

static JamValueRef codegenIndex(JamCodegenContext &ctx, const AstNode &n) {
	JamValueRef idxVal = codegenNode(ctx, static_cast<NodeIdx>(n.rhs));
	if (!idxVal) return nullptr;
	idxVal = coerceTo(ctx, idxVal, ctx.getInt64Type());
	JamTypeRef elemType = nullptr;
	JamValueRef elemPtr = resolveIndexedElementPtr(
	    ctx, static_cast<NodeIdx>(n.lhs), idxVal, elemType);
	return JamLLVMBuildLoad(ctx.getBuilder(), elemType, elemPtr, "idxload");
}

static JamValueRef codegenDeref(JamCodegenContext &ctx, const AstNode &n) {
	const AstNode &op = ctx.getNodeStore().get(static_cast<NodeIdx>(n.lhs));
	if (op.tag != AstTag::Variable) {
		throw std::runtime_error(
		    "`.* ` is only supported on a pointer-typed variable");
	}
	const std::string &name =
	    ctx.getStringPool().get(static_cast<StringIdx>(op.lhs));
	JamValueRef alloca = ctx.getVariable(name);
	if (!alloca) { throw std::runtime_error("Unknown variable: " + name); }
	TypeIdx pointee =
	    pointeeTypeOf(ctx.getTypePool(), ctx.getVariableType(name));
	if (pointee == kNoType) {
		throw std::runtime_error("Cannot dereference non-pointer: " + name);
	}
	JamTypeRef pointeeType = ctx.getLLVMType(pointee);
	JamTypeRef ptrType = ctx.getLLVMType(ctx.getVariableType(name));
	JamValueRef ptrVal =
	    JamLLVMBuildLoad(ctx.getBuilder(), ptrType, alloca, name.c_str());
	return JamLLVMBuildLoad(ctx.getBuilder(), pointeeType, ptrVal, "deref");
}

static JamValueRef codegenAddressOf(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	NodeIdx operandIdx = static_cast<NodeIdx>(n.lhs);
	const AstNode &op = ns.get(operandIdx);

	if (op.tag == AstTag::Variable) {
		const std::string &name = sp.get(static_cast<StringIdx>(op.lhs));
		JamValueRef alloca = ctx.getVariable(name);
		if (!alloca) { throw std::runtime_error("Unknown variable: " + name); }
		return alloca;
	}

	// All other lvalue shapes (Index, MemberAccess, Deref, plus chains like
	// `arr[i].field`) go through the shared resolver. `&p.*` resolves to
	// `p`'s value (the resolver loads the pointer and returns it as the
	// addressable leaf), which matches the source-level semantics.
	if (op.tag == AstTag::Index || op.tag == AstTag::MemberAccess ||
	    op.tag == AstTag::Deref) {
		return resolveLValueAddress(ctx, operandIdx).ptr;
	}

	throw std::runtime_error("Cannot take address of this expression");
}

static JamValueRef codegenVarDecl(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	ExtraIdx extra = static_cast<ExtraIdx>(n.lhs);
	StringIdx nameId = static_cast<StringIdx>(ns.getExtra(extra));
	TypeIdx type = static_cast<TypeIdx>(ns.getExtra(extra + 1));
	NodeIdx initIdx = static_cast<NodeIdx>(ns.getExtra(extra + 2));
	const std::string &name = sp.get(nameId);

	JamTypeRef VarType = ctx.getLLVMType(type);
	JamValueRef Alloca = JamLLVMBuildAlloca(ctx.getBuilder(), VarType,
	                                        ctx.typeAlign(type), name.c_str());

	// Every var declaration carries an initializer (the parser rejects
	// the no-init form). Patch nested literals with the declared target
	// type so they can resolve fields and coerce values during codegen.
	const AstNode &initNode = ns.get(initIdx);
	if (initNode.tag == AstTag::StructLit || initNode.tag == AstTag::ArrayLit ||
	    initNode.tag == AstTag::ArrayRepeat) {
		ctx.getNodeStore().getMut(initIdx).lhs = type;
	}
	// Register the binding's address BEFORE evaluating the init expression
	// so `&self` inside a struct literal (recursive type initializer) can
	// resolve to this alloca. The init analyzer rejects reads-before-init
	// separately.
	ctx.setVariable(name, Alloca);
	ctx.setVariableType(name, type);
	JamValueRef InitVal = codegenNode(ctx, initIdx, VarType);
	if (!InitVal) return nullptr;
	InitVal = coerceTo(ctx, InitVal, VarType);
	JamLLVMBuildStore(ctx.getBuilder(), InitVal, Alloca);

	// Register the binding for drop emission at scope exit if its type has
	// a user-defined `fn drop(self: mut T)`. With every var carrying a
	// concrete initializer, the binding is always Init at this point —
	// the drop call fires unconditionally at every exit reaching it.
	{
		// Resolve through aliases / generic instantiations to the
		// canonical struct so the drop-fn lookup uses the actual
		// instantiated name (e.g. `Box__i32`), not the alias the user
		// wrote (`BoxI32`). lookupStruct handles GenericCall and
		// alias resolution; the resulting StructInfo's `name` is
		// canonical. lookupDropFn checks the pre-built drop registry
		// AND the per-instantiation drop map populated by
		// instantiateStructExpr.
		const auto *sinfo = ctx.lookupStruct(type);
		if (sinfo) {
			const std::string &structName = sinfo->name;
			if (const FunctionAST *dropFn = ctx.lookupDropFn(structName)) {
				ctx.registerLocalDrop(name, Alloca, VarType, dropFn);
			}
		}
	}
	return Alloca;
}

// Emit a single drop call for one tracked binding. The
// mode-aware ABI now lowers `fn drop(self: mut T)` as a pointer-typed
// parameter, so the call site passes the binding's storage address
// directly — no load-value-then-pass-by-value workaround. The drop fn
// reads/writes self through the pointer and the caller's storage is
// genuinely affected.
//
// With `undefined` removed, every drop-bearing binding is statically Init
// at every exit reaching this drop point, so the call fires unconditionally.
static void emitOneDrop(JamCodegenContext &ctx,
                        const JamCodegenContext::DropEntry &e) {
	std::string mangled =
	    mangledFunctionName(*e.dropFn, ctx.getTypePool(), ctx.getStringPool());
	JamFunctionRef dropFn =
	    JamLLVMGetFunction(ctx.getModule(), mangled.c_str());
	if (!dropFn) return;
	JamValueRef args[1] = {e.alloca};
	JamLLVMBuildCall(ctx.getBuilder(), dropFn, args, 1, "");
}

// emit drops for the topmost active scope, in reverse declaration
// order. Used at the end of nested blocks (if/else arms, while/for body,
// match arm body, function body fall-through) just before branching to
// the merge / cond / next-statement BB.
static void emitTopScopeDrops(JamCodegenContext &ctx) {
	const auto &scopes = ctx.getDropScopes();
	if (scopes.empty()) return;
	const auto &top = scopes.back();
	for (auto it = top.rbegin(); it != top.rend(); ++it) {
		emitOneDrop(ctx, *it);
	}
}

// emit drops for every active scope, innermost-first. Used at every
// `return` statement so a return inside a nested block drops both the
// block-scoped locals and the outer function-scoped locals.
static void emitAllScopeDrops(JamCodegenContext &ctx) {
	const auto &scopes = ctx.getDropScopes();
	for (auto sit = scopes.rbegin(); sit != scopes.rend(); ++sit) {
		for (auto eit = sit->rbegin(); eit != sit->rend(); ++eit) {
			emitOneDrop(ctx, *eit);
		}
	}
}

// emit drops for every scope from the innermost down to (and
// including) `targetScopeIdx`. Used by `break` and `continue` to drop
// the loop body's locals — and any deeper nested scopes — before
// branching out of (or to the next iteration of) the loop. The function
// scope and any other scopes outside the target index are preserved.
static void emitDropsThroughScope(JamCodegenContext &ctx,
                                  std::size_t targetScopeIdx) {
	const auto &scopes = ctx.getDropScopes();
	if (scopes.empty() || targetScopeIdx >= scopes.size()) return;
	for (std::size_t i = scopes.size(); i > targetScopeIdx; i--) {
		const auto &scope = scopes[i - 1];
		for (auto it = scope.rbegin(); it != scope.rend(); ++it) {
			emitOneDrop(ctx, *it);
		}
	}
}

static JamValueRef codegenIf(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t thenCount = ns.getExtra(extra);
	uint32_t elseCount = ns.getExtra(extra + 1);

	JamValueRef CondV = codegenNode(ctx, condIdx);
	if (!CondV) return nullptr;
	JamTypeRef condType = JamLLVMTypeOf(CondV);
	CondV = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_NE, CondV,
	                         JamLLVMConstInt(condType, 0, false), "ifcond");

	JamBasicBlockRef CurrentBB = JamLLVMGetInsertBlock(ctx.getBuilder());
	JamFunctionRef TheFunction = JamLLVMGetBasicBlockParent(CurrentBB);
	JamBasicBlockRef ThenBB = JamLLVMAppendBasicBlock(TheFunction, "then");
	JamBasicBlockRef ElseBB = JamLLVMAppendBasicBlock(TheFunction, "else");
	JamBasicBlockRef MergeBB = JamLLVMAppendBasicBlock(TheFunction, "ifcont");

	JamLLVMBuildCondBr(ctx.getBuilder(), CondV, ThenBB, ElseBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), ThenBB);
	ctx.pushDropScope();
	for (uint32_t i = 0; i < thenCount; i++) {
		codegenNode(ctx, ns.getExtra(extra + 2 + i));
	}
	if (!JamLLVMGetBasicBlockTerminator(
	        JamLLVMGetInsertBlock(ctx.getBuilder()))) {
		emitTopScopeDrops(ctx);
		JamLLVMBuildBr(ctx.getBuilder(), MergeBB);
	}
	ctx.popDropScope();

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), ElseBB);
	ctx.pushDropScope();
	for (uint32_t i = 0; i < elseCount; i++) {
		codegenNode(ctx, ns.getExtra(extra + 2 + thenCount + i));
	}
	if (!JamLLVMGetBasicBlockTerminator(
	        JamLLVMGetInsertBlock(ctx.getBuilder()))) {
		emitTopScopeDrops(ctx);
		JamLLVMBuildBr(ctx.getBuilder(), MergeBB);
	}
	ctx.popDropScope();

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), MergeBB);
	return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
}

static JamValueRef codegenWhile(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t bodyCount = ns.getExtra(extra);

	JamBasicBlockRef CurrentBB = JamLLVMGetInsertBlock(ctx.getBuilder());
	JamFunctionRef TheFunction = JamLLVMGetBasicBlockParent(CurrentBB);
	JamBasicBlockRef CondBB = JamLLVMAppendBasicBlock(TheFunction, "whilecond");
	JamBasicBlockRef LoopBB = JamLLVMAppendBasicBlock(TheFunction, "whileloop");
	JamBasicBlockRef AfterBB =
	    JamLLVMAppendBasicBlock(TheFunction, "afterloop");

	JamBasicBlockRef PrevContinue = CurrentLoopContinue;
	JamBasicBlockRef PrevBreak = CurrentLoopBreak;
	std::size_t PrevLoopScope = CurrentLoopBodyScopeIdx;
	CurrentLoopContinue = CondBB;
	CurrentLoopBreak = AfterBB;

	JamLLVMBuildBr(ctx.getBuilder(), CondBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), CondBB);
	JamValueRef CondV = codegenNode(ctx, condIdx);
	if (!CondV) {
		CurrentLoopContinue = PrevContinue;
		CurrentLoopBreak = PrevBreak;
		CurrentLoopBodyScopeIdx = PrevLoopScope;
		return nullptr;
	}
	JamTypeRef condType = JamLLVMTypeOf(CondV);
	CondV = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_NE, CondV,
	                         JamLLVMConstInt(condType, 0, false), "whilecond");
	JamLLVMBuildCondBr(ctx.getBuilder(), CondV, LoopBB, AfterBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), LoopBB);
	// the loop body's scope is the next one to be pushed.
	CurrentLoopBodyScopeIdx = ctx.getDropScopes().size();
	ctx.pushDropScope();
	for (uint32_t i = 0; i < bodyCount; i++) {
		codegenNode(ctx, ns.getExtra(extra + 1 + i));
	}
	if (!JamLLVMGetBasicBlockTerminator(
	        JamLLVMGetInsertBlock(ctx.getBuilder()))) {
		emitTopScopeDrops(ctx);
		JamLLVMBuildBr(ctx.getBuilder(), CondBB);
	}
	ctx.popDropScope();

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), AfterBB);
	CurrentLoopContinue = PrevContinue;
	CurrentLoopBreak = PrevBreak;
	CurrentLoopBodyScopeIdx = PrevLoopScope;
	return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
}

static JamValueRef codegenFor(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	ExtraIdx extra = static_cast<ExtraIdx>(n.lhs);
	StringIdx varNameId = static_cast<StringIdx>(ns.getExtra(extra));
	NodeIdx startIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1));
	NodeIdx endIdx = static_cast<NodeIdx>(ns.getExtra(extra + 2));
	uint32_t bodyCount = ns.getExtra(extra + 3);
	const std::string &varName = sp.get(varNameId);

	JamBasicBlockRef CurrentBB = JamLLVMGetInsertBlock(ctx.getBuilder());
	JamFunctionRef TheFunction = JamLLVMGetBasicBlockParent(CurrentBB);

	JamValueRef StartVal = codegenNode(ctx, startIdx);
	JamValueRef EndVal = codegenNode(ctx, endIdx);
	if (!StartVal || !EndVal) return nullptr;

	JamTypeRef VarType = JamLLVMTypeOf(StartVal);
	JamTypeRef endType = JamLLVMTypeOf(EndVal);
	if (endType != VarType) {
		if (JamLLVMTypeIsInteger(VarType) && JamLLVMTypeIsInteger(endType)) {
			EndVal = JamLLVMBuildIntCast(ctx.getBuilder(), EndVal, VarType,
			                             true, "endcast");
		} else {
			throw std::runtime_error("Type mismatch in for loop range");
		}
	}

	// For-loop induction variable is always an integer here; alignment is
	// the integer's byte width.
	uint64_t loopVarAlign = JamLLVMGetIntTypeWidth(VarType) / 8;
	JamValueRef Alloca = JamLLVMBuildAlloca(ctx.getBuilder(), VarType,
	                                        loopVarAlign, varName.c_str());
	JamLLVMBuildStore(ctx.getBuilder(), StartVal, Alloca);

	JamValueRef OldVal = ctx.getVariable(varName);
	ctx.setVariable(varName, Alloca);
	// Record a source-level TypeIdx for the loop variable so subsequent
	// reads via codegenNode(Variable) can recover the LLVM load type.
	// The variable's type is whatever integer the range bounds inferred
	// to; treat it as unsigned by default (range syntax is `0:N`).
	if (JamLLVMTypeIsInteger(VarType)) {
		ctx.setVariableType(
		    varName,
		    ctx.getTypePool().internInt(
		        static_cast<uint16_t>(JamLLVMGetIntTypeWidth(VarType)), false));
	}

	JamBasicBlockRef CondBB = JamLLVMAppendBasicBlock(TheFunction, "forcond");
	JamBasicBlockRef LoopBB = JamLLVMAppendBasicBlock(TheFunction, "forloop");
	JamBasicBlockRef IncrBB = JamLLVMAppendBasicBlock(TheFunction, "forincr");
	JamBasicBlockRef AfterBB =
	    JamLLVMAppendBasicBlock(TheFunction, "afterloop");

	JamBasicBlockRef PrevContinue = CurrentLoopContinue;
	JamBasicBlockRef PrevBreak = CurrentLoopBreak;
	std::size_t PrevLoopScope = CurrentLoopBodyScopeIdx;
	CurrentLoopContinue = IncrBB;
	CurrentLoopBreak = AfterBB;

	JamLLVMBuildBr(ctx.getBuilder(), CondBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), CondBB);
	JamValueRef CurVar =
	    JamLLVMBuildLoad(ctx.getBuilder(), VarType, Alloca, varName.c_str());
	JamValueRef CondV = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_SLT, CurVar,
	                                     EndVal, "forcond");
	JamLLVMBuildCondBr(ctx.getBuilder(), CondV, LoopBB, AfterBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), LoopBB);
	CurrentLoopBodyScopeIdx = ctx.getDropScopes().size();
	ctx.pushDropScope();
	for (uint32_t i = 0; i < bodyCount; i++) {
		codegenNode(ctx, ns.getExtra(extra + 4 + i));
	}
	if (!JamLLVMGetBasicBlockTerminator(
	        JamLLVMGetInsertBlock(ctx.getBuilder()))) {
		emitTopScopeDrops(ctx);
		JamLLVMBuildBr(ctx.getBuilder(), IncrBB);
	}
	ctx.popDropScope();

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), IncrBB);
	JamValueRef CurVarForInc =
	    JamLLVMBuildLoad(ctx.getBuilder(), VarType, Alloca, varName.c_str());
	JamValueRef StepVal = JamLLVMConstInt(VarType, 1, false);
	JamValueRef NextVar =
	    JamLLVMBuildAdd(ctx.getBuilder(), CurVarForInc, StepVal, "nextvar");
	JamLLVMBuildStore(ctx.getBuilder(), NextVar, Alloca);
	JamLLVMBuildBr(ctx.getBuilder(), CondBB);

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), AfterBB);

	if (OldVal) ctx.setVariable(varName, OldVal);

	CurrentLoopContinue = PrevContinue;
	CurrentLoopBreak = PrevBreak;
	CurrentLoopBodyScopeIdx = PrevLoopScope;
	return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
}

// `match` codegen — (integer literals, inclusive ranges, or-patterns,
// wildcard, `else`).
//
// Strategy: a sequential icmp-cascade. For each arm, build a boolean
// expression representing "this pattern matches the scrutinee", branch to
// the arm body if true and to the next arm's test (or `else` / fallthrough)
// if false. LLVM's downstream simplifycfg pass collapses chains of equality
// tests on the same value into a `switch` instruction when profitable, so
// the pure-literal opcode-dispatch case still gets jump-table codegen
// without us emitting `switch` ourselves.
//
// The Maranget decision-tree formalism degenerates to "specialize the only
// column" in (one scrutinee, one column); the cascade is the canonical
// one-column lowering. A multi-column tree will replace this later.

// Build a boolean (i1) value that is true iff the scrutinee matches the
// given pattern node. `scrut` is the already-loaded scrutinee value;
// `scrutType` is its LLVM type, used to materialize comparison constants.
static JamValueRef emitPatternTest(JamCodegenContext &ctx, NodeIdx patIdx,
                                   JamValueRef scrut, JamTypeRef scrutType) {
	const NodeStore &ns = ctx.getNodeStore();
	const AstNode &pn = ns.get(patIdx);
	switch (pn.tag) {
	case AstTag::PatWildcard:
		return JamLLVMConstInt(ctx.getInt1Type(), 1, false);
	case AstTag::PatLit: {
		uint64_t val = static_cast<uint64_t>(pn.lhs) |
		               (static_cast<uint64_t>(pn.rhs) << 32);
		bool isNeg = (pn.flags & 1) != 0;
		uint64_t materialized =
		    isNeg ? static_cast<uint64_t>(-static_cast<int64_t>(val)) : val;
		JamValueRef k = JamLLVMConstInt(scrutType, materialized, isNeg);
		return JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_EQ, scrut, k,
		                        "pat.eq");
	}
	case AstTag::PatRange: {
		JamValueRef lo = JamLLVMConstInt(scrutType, pn.lhs, false);
		JamValueRef hi = JamLLVMConstInt(scrutType, pn.rhs, false);
		// Unsigned bounds test: scrut >= lo && scrut <= hi.
		JamValueRef geLo = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_UGE,
		                                    scrut, lo, "pat.ge");
		JamValueRef leHi = JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_ULE,
		                                    scrut, hi, "pat.le");
		return JamLLVMBuildAnd(ctx.getBuilder(), geLo, leHi, "pat.range");
	}
	case AstTag::PatOr: {
		ExtraIdx extra = static_cast<ExtraIdx>(pn.lhs);
		uint32_t count = ns.getExtra(extra);
		JamValueRef acc = nullptr;
		for (uint32_t i = 0; i < count; i++) {
			NodeIdx sub = static_cast<NodeIdx>(ns.getExtra(extra + 1 + i));
			JamValueRef one = emitPatternTest(ctx, sub, scrut, scrutType);
			if (!acc) {
				acc = one;
			} else {
				acc = JamLLVMBuildOr(ctx.getBuilder(), acc, one, "pat.or");
			}
		}
		return acc ? acc : JamLLVMConstInt(ctx.getInt1Type(), 0, false);
	}
	case AstTag::PatEnumVariant: {
		// Encoded via PatEnumVariant flags — see decodePatEnumVariant
		// for the four-way encoding cross product (bindings × TypeIdx
		// receiver) and alias resolution.
		DecodedPatEnumVariant d = decodePatEnumVariant(ctx, pn);
		const std::string &enumName = d.enumName;
		const std::string &variantName = d.variantName;
		const auto *einfo = d.einfo;
		int idx = d.variantIndex;
		if (!einfo || idx < 0) {
			throw std::runtime_error("Enum `" + enumName +
			                         "` has no variant `" + variantName +
			                         "` (in match pattern)");
		}
		uint32_t discrim = einfo->variants[idx].discriminant;
		// Extract the tag if the scrutinee is a {tag, payload} struct.
		// For unit-only enums the scrutinee is already i8.
		JamValueRef tagVal = scrut;
		JamTypeRef tagType = scrutType;
		if (JamLLVMTypeIsStruct(scrutType)) {
			tagVal =
			    JamLLVMBuildExtractValue(ctx.getBuilder(), scrut, 0, "pat.tag");
			tagType = ctx.getInt8Type();
		}
		JamValueRef k =
		    JamLLVMConstInt(tagType, static_cast<uint64_t>(discrim), false);
		return JamLLVMBuildICmp(ctx.getBuilder(), JAM_ICMP_EQ, tagVal, k,
		                        "pat.variant");
	}
	default:
		throw std::runtime_error("Unsupported pattern node kind in codegen");
	}
}

static JamValueRef codegenMatch(JamCodegenContext &ctx, const AstNode &n) {
	const NodeStore &ns = ctx.getNodeStore();
	NodeIdx scrutIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);

	uint32_t armCount = ns.getExtra(extra);

	// Arm records begin immediately after the armCount slot.
	uint32_t armsOff = 1;

	// compile-time exhaustiveness + reachability over enum
	// variants. We scan all arm patterns once, accumulating which
	// variants are covered and detecting duplicates (an arm that
	// would never run because an earlier arm already covers it).
	// Exhaustiveness fires when no catch-all `_` is present.
	{
		std::string foundEnum;
		std::vector<std::string> coveredVariants;
		bool sawCatchAll = false;
		// Recursive walk over a pattern node. Side effects: appends
		// variant names to `coveredVariants` for enum patterns whose
		// enum matches the (first-seen) `foundEnum`; sets
		// `sawCatchAll` if a wildcard makes the match exhaustive.
		// Return value: whether the pattern provably catches its
		// position (wildcard, or or-pattern whose any sub is a
		// catch-all).
		std::function<bool(NodeIdx)> walkPattern;
		walkPattern = [&](NodeIdx p) -> bool {
			const AstNode &pp = ns.get(p);
			switch (pp.tag) {
			case AstTag::PatWildcard:
				return true;
			case AstTag::PatEnumVariant: {
				DecodedPatEnumVariant d = decodePatEnumVariant(ctx, pp);
				if (foundEnum.empty()) foundEnum = d.enumName;
				if (foundEnum == d.enumName) {
					coveredVariants.push_back(d.variantName);
				}
				return false;
			}
			case AstTag::PatOr: {
				ExtraIdx ex = static_cast<ExtraIdx>(pp.lhs);
				uint32_t cnt = ns.getExtra(ex);
				bool catchAll = false;
				for (uint32_t i = 0; i < cnt; i++) {
					NodeIdx sub = static_cast<NodeIdx>(ns.getExtra(ex + 1 + i));
					if (walkPattern(sub)) catchAll = true;
				}
				return catchAll;
			}
			default:
				// Literal / range / anything else — doesn't catch
				// everything; exhaustiveness must come from elsewhere.
				return false;
			}
		};
		uint32_t scanPos = armsOff;
		for (uint32_t i = 0; i < armCount; i++) {
			NodeIdx patIdx = static_cast<NodeIdx>(ns.getExtra(extra + scanPos));
			uint32_t bodyCount = ns.getExtra(extra + scanPos + 1);
			if (walkPattern(patIdx)) sawCatchAll = true;
			scanPos = scanPos + 2 + bodyCount;
		}

		// Reachability: duplicate enum-variant pattern across arms is
		// always unreachable (the earlier arm consumes the value).
		// Runs regardless of `else` presence — `else` doesn't excuse
		// dead code.
		for (size_t i = 0; i < coveredVariants.size(); i++) {
			for (size_t j = i + 1; j < coveredVariants.size(); j++) {
				if (coveredVariants[i] == coveredVariants[j]) {
					throw std::runtime_error(
					    "Unreachable match arm: variant `" + foundEnum + "." +
					    coveredVariants[j] +
					    "` is already covered by an earlier arm");
				}
			}
		}

		// Exhaustiveness: enforced when no catch-all `_` is present.
		if (!foundEnum.empty() && !sawCatchAll) {
			const auto *einfo = ctx.getEnum(foundEnum);
			if (einfo) {
				std::vector<std::string> missing;
				for (const auto &v : einfo->variants) {
					bool covered = false;
					for (const auto &c : coveredVariants) {
						if (c == v.name) {
							covered = true;
							break;
						}
					}
					if (!covered) missing.push_back(v.name);
				}
				if (!missing.empty()) {
					std::string msg = "Non-exhaustive match on enum `" +
					                  foundEnum + "`; missing variant(s): ";
					for (size_t i = 0; i < missing.size(); i++) {
						if (i > 0) msg += ", ";
						msg += missing[i];
					}
					throw std::runtime_error(msg);
				}
			}
		}
	}

	// Evaluate the scrutinee once. The value is reused for every pattern
	// test below; LLVM mem2reg promotes the local to an SSA value for
	// free. We also alloca a slot holding the scrutinee so payload
	// bindings can GEP into the {tag, payload} struct.
	JamValueRef scrut = codegenNode(ctx, scrutIdx);
	if (!scrut) return nullptr;
	JamTypeRef scrutType = JamLLVMTypeOf(scrut);
	// Classify the scrutinee. Enum scrutinees may be either an integer
	// (unit-only enums lower to i8) or a struct (payloaded enums
	// lower to a named struct). For struct scrutinees we MUST verify
	// the type matches a registered enum, otherwise a slice (which is
	// also a struct shape `{ptr, len}`) would silently route through
	// the tag-extraction path and produce nonsense.
	bool scrutIsEnum = false;
	if (JamLLVMTypeIsStruct(scrutType)) {
		if (ctx.findEnumByLLVMType(scrutType) != nullptr) {
			scrutIsEnum = true;
		} else {
			throw std::runtime_error(
			    "`match` on a struct-shaped value is only supported when "
			    "the value is an enum; got a different struct type");
		}
	} else if (!JamLLVMTypeIsInteger(scrutType)) {
		throw std::runtime_error(
		    "`match` only supports integer or enum scrutinees");
	}
	JamValueRef scrutPtr = nullptr;
	if (scrutIsEnum) {
		const auto *einfo = ctx.findEnumByLLVMType(scrutType);
		uint64_t scrutAlign =
		    einfo && einfo->maxPayloadAlign > 1 ? einfo->maxPayloadAlign : 1;
		scrutPtr = JamLLVMBuildAlloca(ctx.getBuilder(), scrutType, scrutAlign,
		                              "match.scrut");
		JamLLVMBuildStore(ctx.getBuilder(), scrut, scrutPtr);
	}

	JamBasicBlockRef curBB = JamLLVMGetInsertBlock(ctx.getBuilder());
	JamFunctionRef func = JamLLVMGetBasicBlockParent(curBB);
	JamBasicBlockRef mergeBB = JamLLVMAppendBasicBlock(func, "match.end");

	// Per-arm result for the phi: pairs (basicBlock, value) where
	// the arm's body ended without terminating control flow. The
	// match's expression-form value is a phi over these.
	std::vector<std::pair<JamBasicBlockRef, JamValueRef>> armResults;

	// Pre-create a "test block" for every arm beyond the first, plus a
	// "fall block" for the else arm (or merge if no else). This lets each
	// arm body branch back to merge cleanly without lookahead.
	std::vector<JamBasicBlockRef> testBBs;
	testBBs.reserve(armCount);
	std::vector<JamBasicBlockRef> bodyBBs;
	bodyBBs.reserve(armCount);
	for (uint32_t i = 0; i < armCount; i++) {
		if (i == 0) {
			testBBs.push_back(curBB);
		} else {
			testBBs.push_back(JamLLVMAppendBasicBlock(func, "match.test"));
		}
		bodyBBs.push_back(JamLLVMAppendBasicBlock(func, "match.arm"));
	}
	// Where do we go if every arm fails? Always the merge block — match
	// has no `else` arm, so a non-exhaustive match falls straight through
	// (the analyzer rejects this for enum scrutinees).
	JamBasicBlockRef fallBB = mergeBB;

	// Emit each arm's pattern test in its dedicated test block.
	uint32_t pos = armsOff;
	for (uint32_t i = 0; i < armCount; i++) {
		NodeIdx patIdx = static_cast<NodeIdx>(ns.getExtra(extra + pos));
		uint32_t bodyCount = ns.getExtra(extra + pos + 1);
		uint32_t bodyStart = pos + 2;

		JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), testBBs[i]);
		JamValueRef matched = emitPatternTest(ctx, patIdx, scrut, scrutType);
		JamBasicBlockRef nextTest =
		    (i + 1 < armCount) ? testBBs[i + 1] : fallBB;
		JamLLVMBuildCondBr(ctx.getBuilder(), matched, bodyBBs[i], nextTest);

		// Emit the arm body.
		JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), bodyBBs[i]);

		// Materialize payload bindings if the pattern carries any.
		// Decoded via the shared helper which handles all four
		// encoding combinations + alias / generic-call resolution.
		const AstNode &pn = ns.get(patIdx);
		if (scrutIsEnum && pn.tag == AstTag::PatEnumVariant &&
		    (pn.flags & 1) != 0) {
			DecodedPatEnumVariant d = decodePatEnumVariant(ctx, pn);
			if (!d.einfo || d.variantIndex < 0) {
				throw std::runtime_error(
				    "Pattern references unknown variant `" + d.enumName + "." +
				    d.variantName + "`");
			}
			const auto *einfo = d.einfo;
			const std::string &enumName = d.enumName;
			const std::string &variantName = d.variantName;
			uint32_t bcount = d.bindingCount;
			ExtraIdx ex = d.bindingsStart - 3;  // anchor for backward-compat
			const auto &v = einfo->variants[d.variantIndex];
			if (bcount != v.payloadTypes.size()) {
				throw std::runtime_error("Pattern for `" + enumName + "." +
				                         variantName + "` binds " +
				                         std::to_string(bcount) +
				                         " payload field(s); variant has " +
				                         std::to_string(v.payloadTypes.size()));
			}
			// Payload area always starts at struct field 1 (the
			// alignment-driving scalar; larger payloads spill into
			// field 2 via byte-offset GEP from there).
			unsigned payloadFieldIdx = 1;
			JamValueRef payloadAreaPtr =
			    JamLLVMBuildStructGEP(ctx.getBuilder(), scrutType, scrutPtr,
			                          payloadFieldIdx, "match.payload");
			(void)einfo;
			uint64_t off = 0;
			for (uint32_t b = 0; b < bcount; b++) {
				StringIdx nameId =
				    static_cast<StringIdx>(ns.getExtra(ex + 3 + b));
				const std::string &name = ctx.getStringPool().get(nameId);
				TypeIdx ty = v.payloadTypes[b];
				uint64_t s = ctx.typeSize(ty);
				uint64_t a = ctx.typeAlign(ty);
				off = (off + a - 1) / a * a;
				JamValueRef i64Off =
				    JamLLVMConstInt(ctx.getInt64Type(), off, false);
				JamValueRef fieldPtr = JamLLVMBuildPtrGEP(
				    ctx.getBuilder(), ctx.getInt8Type(), payloadAreaPtr, i64Off,
				    "match.field.ptr");
				JamTypeRef fieldLLVM = ctx.getLLVMType(ty);
				JamValueRef fieldVal = JamLLVMBuildLoad(
				    ctx.getBuilder(), fieldLLVM, fieldPtr, name.c_str());
				JamValueRef bindAlloca =
				    JamLLVMBuildAlloca(ctx.getBuilder(), fieldLLVM,
				                       ctx.typeAlign(ty), name.c_str());
				JamLLVMBuildStore(ctx.getBuilder(), fieldVal, bindAlloca);
				ctx.setVariable(name, bindAlloca);
				ctx.setVariableType(name, ty);
				off += s;
			}
		}

		// Track each arm's last produced value (for match-as-expression
		// support). Arms that terminate via `return` / `break` /
		// `continue` don't fall through to merge and aren't recorded.
		JamValueRef lastValue = nullptr;
		ctx.pushDropScope();
		for (uint32_t b = 0; b < bodyCount; b++) {
			lastValue = codegenNode(ctx, ns.getExtra(extra + bodyStart + b));
		}
		JamBasicBlockRef armEndBB = JamLLVMGetInsertBlock(ctx.getBuilder());
		if (!JamLLVMGetBasicBlockTerminator(armEndBB)) {
			emitTopScopeDrops(ctx);
			armResults.push_back({armEndBB, lastValue});
			JamLLVMBuildBr(ctx.getBuilder(), mergeBB);
		}
		ctx.popDropScope();

		pos = bodyStart + bodyCount;
	}

	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), mergeBB);

	// Build a phi over arm-produced values when all arms agree on a
	// non-void type. This is the value of the `match` expression.
	// When every arm terminated (e.g. all arms `return`), no path
	// reaches the merge block; leave it untermited and let the
	// surrounding code (defineBody, or subsequent statements that
	// emit into the block) handle the dead-block case. Emitting an
	// `unreachable` here would be incorrect when more user code
	// follows the match — the builder would insert instructions
	// after the terminator and produce malformed IR.
	if (armResults.empty()) {
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	}
	// Pick the phi type from the first arm that produced a non-void
	// value. Empty bodies and void-typed values (e.g. statement-form
	// match arms ending in a void fn call) get a typed-zero substitute
	// below; LLVM rejects a phi with void type, so we skip those when
	// choosing. If every arm is void/empty, skip the phi entirely.
	JamTypeRef phiType = nullptr;
	for (auto &r : armResults) {
		if (!r.second) continue;
		JamTypeRef ty = JamLLVMTypeOf(r.second);
		if (!ty || JamLLVMTypeIsVoid(ty)) continue;
		phiType = ty;
		break;
	}
	if (!phiType) {
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	}
	JamValueRef phi =
	    JamLLVMBuildPhi(ctx.getBuilder(), phiType, "match.result");
	std::vector<JamValueRef> phiVals;
	std::vector<JamBasicBlockRef> phiBlocks;
	for (auto &r : armResults) {
		JamValueRef v = r.second;
		if (!v || JamLLVMTypeOf(v) != phiType) {
			// Empty body or type mismatch: substitute a typed zero so
			// the phi remains well-formed. Mismatched arm-types in an
			// expression-form match are a user bug; the resulting
			// value is unspecified rather than blowing up the build.
			v = JamLLVMConstInt(phiType, 0, false);
		}
		phiVals.push_back(v);
		phiBlocks.push_back(r.first);
	}
	JamLLVMAddIncoming(phi, phiVals.data(), phiBlocks.data(),
	                   static_cast<unsigned>(phiVals.size()));
	return phi;
}

// `[a, b, c]`: build the SSA aggregate via insertvalue chain. The target
// type must have been bound to n.lhs by the caller (var-decl, return,
// struct-field init, etc.); reaching codegen with kNoType is a missing-
// context error.
//
// Targets:
//   • [N]T   — element list must have N entries.
//   • []T    — only the empty form `[]` is accepted; produces an empty
//              slice {ptr=null, len=0}. Non-empty literals targeting a
//              slice would need a backing storage decision the language
//              hasn't made yet.
static JamValueRef codegenArrayLit(JamCodegenContext &ctx, const AstNode &n) {
	TypeIdx targetType = static_cast<TypeIdx>(n.lhs);
	if (targetType == kNoType) {
		throw std::runtime_error(
		    "Array literal used without a known target type");
	}
	const TypeKey &k = ctx.getTypePool().get(targetType);
	const NodeStore &ns = ctx.getNodeStore();
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t count = ns.getExtra(extra);

	if (k.kind == TypeKind::Slice) {
		if (count != 0) {
			throw std::runtime_error(
			    "Non-empty array literal cannot target a slice type "
			    "directly — use a fixed-size array and slice it");
		}
		// {ptr=null, len=0} as a {ptr, i64} aggregate.
		JamTypeRef sliceLLVM = ctx.getLLVMType(targetType);
		JamValueRef sliceVal = JamLLVMGetUndef(sliceLLVM);
		TypeIdx elemTy = static_cast<TypeIdx>(k.a);
		JamTypeRef elemLLVM = ctx.getLLVMType(elemTy);
		JamTypeRef nullPtr = JamLLVMPointerType(elemLLVM, 0);
		JamValueRef nullPtrVal = JamLLVMConstNull(nullPtr);
		sliceVal = JamLLVMBuildInsertValue(ctx.getBuilder(), sliceVal,
		                                   nullPtrVal, 0, "slice.ptr");
		JamValueRef zeroLen = JamLLVMConstInt(ctx.getInt64Type(), 0, false);
		sliceVal = JamLLVMBuildInsertValue(ctx.getBuilder(), sliceVal, zeroLen,
		                                   1, "slice.len");
		return sliceVal;
	}

	if (k.kind != TypeKind::Array) {
		throw std::runtime_error(
		    "Array literal target type is not an array or slice");
	}
	TypeIdx elemTy = static_cast<TypeIdx>(k.a);
	uint32_t expectedLen = k.b;
	JamTypeRef arrLLVM = ctx.getLLVMType(targetType);
	JamTypeRef elemLLVM = ctx.getLLVMType(elemTy);

	if (count != expectedLen) {
		throw std::runtime_error("Array literal has " + std::to_string(count) +
		                         " elements but type expects " +
		                         std::to_string(expectedLen));
	}

	JamValueRef arrVal = JamLLVMGetUndef(arrLLVM);
	for (uint32_t i = 0; i < count; i++) {
		NodeIdx elemIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1 + i));
		const AstNode &elemNode = ns.get(elemIdx);
		if ((elemNode.tag == AstTag::ArrayLit ||
		     elemNode.tag == AstTag::ArrayRepeat ||
		     elemNode.tag == AstTag::StructLit) &&
		    static_cast<TypeIdx>(elemNode.lhs) == kNoType) {
			ctx.getNodeStore().getMut(elemIdx).lhs = elemTy;
		}
		JamValueRef elemVal = codegenNode(ctx, elemIdx, elemLLVM);
		if (!elemVal) return nullptr;
		elemVal = coerceTo(ctx, elemVal, elemLLVM);
		arrVal = JamLLVMBuildInsertValue(ctx.getBuilder(), arrVal, elemVal, i,
		                                 "elem");
	}
	return arrVal;
}

// `[expr; N]`: repeat `expr` N times. v1 requires N to be a constant
// integer literal so we can build a fixed-size array; symbolic-sized
// arrays would need a Slice or runtime fill loop, neither of which is
// in scope yet. Codegen emits the repeat as an insertvalue chain — the
// simplest IR; LLVM constant-folds when expr is constant.
static JamValueRef codegenArrayRepeat(JamCodegenContext &ctx,
                                      const AstNode &n) {
	TypeIdx arrType = static_cast<TypeIdx>(n.lhs);
	if (arrType == kNoType) {
		throw std::runtime_error(
		    "Array repeat literal used without a known target type");
	}
	const TypeKey &k = ctx.getTypePool().get(arrType);
	if (k.kind != TypeKind::Array) {
		throw std::runtime_error(
		    "Array repeat literal target type is not an array");
	}
	TypeIdx elemTy = static_cast<TypeIdx>(k.a);
	uint32_t expectedLen = k.b;
	JamTypeRef arrLLVM = ctx.getLLVMType(arrType);
	JamTypeRef elemLLVM = ctx.getLLVMType(elemTy);

	const NodeStore &ns = ctx.getNodeStore();
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	NodeIdx valueIdx = static_cast<NodeIdx>(ns.getExtra(extra));
	NodeIdx countIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1));

	const AstNode &countNode = ns.get(countIdx);
	if (countNode.tag != AstTag::NumberLit) {
		throw std::runtime_error(
		    "Array repeat count must be an integer literal");
	}
	uint64_t countVal = numberLitValue(countNode);
	if (countVal != static_cast<uint64_t>(expectedLen)) {
		throw std::runtime_error(
		    "Array repeat count " + std::to_string(countVal) +
		    " does not match declared length " + std::to_string(expectedLen));
	}

	const AstNode &valNode = ns.get(valueIdx);
	if ((valNode.tag == AstTag::ArrayLit ||
	     valNode.tag == AstTag::ArrayRepeat ||
	     valNode.tag == AstTag::StructLit) &&
	    static_cast<TypeIdx>(valNode.lhs) == kNoType) {
		ctx.getNodeStore().getMut(valueIdx).lhs = elemTy;
	}
	JamValueRef elemVal = codegenNode(ctx, valueIdx, elemLLVM);
	if (!elemVal) return nullptr;
	elemVal = coerceTo(ctx, elemVal, elemLLVM);

	JamValueRef arrVal = JamLLVMGetUndef(arrLLVM);
	for (uint32_t i = 0; i < expectedLen; i++) {
		arrVal = JamLLVMBuildInsertValue(ctx.getBuilder(), arrVal, elemVal, i,
		                                 "rep");
	}
	return arrVal;
}

static JamValueRef codegenStructLit(JamCodegenContext &ctx, const AstNode &n) {
	TypeIdx structType = static_cast<TypeIdx>(n.lhs);
	if (structType == kNoType) {
		throw std::runtime_error(
		    "Struct literal used without a known target type");
	}
	const auto *info = ctx.lookupStruct(structType);
	if (!info) {
		// Union targets share the `{ field: value }` literal form. The
		// untagged-union shape means exactly one field's bits are
		// initialized; the literal must list one field and we materialize
		// the union value via alloca/store/load (LLVM has no SSA insertvalue
		// for the cross-field reinterpret pattern).
		if (const auto *uinfo = ctx.lookupUnion(structType)) {
			const NodeStore &ns = ctx.getNodeStore();
			const StringPool &sp = ctx.getStringPool();
			ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
			uint32_t fieldCount = ns.getExtra(extra);
			if (fieldCount != 1) {
				throw std::runtime_error(
				    "Union literal must list exactly one field, got " +
				    std::to_string(fieldCount));
			}
			StringIdx fldNameId =
			    static_cast<StringIdx>(ns.getExtra(extra + 1));
			NodeIdx fldExprIdx = static_cast<NodeIdx>(ns.getExtra(extra + 2));
			const std::string &fieldName = sp.get(fldNameId);
			TypeIdx fieldTy = ctx.getUnionFieldType(uinfo->name, fieldName);
			if (fieldTy == kNoType) {
				throw std::runtime_error("Union `" + uinfo->name +
				                         "` has no field `" + fieldName + "`");
			}
			JamTypeRef fieldLLVM = ctx.getLLVMType(fieldTy);
			JamValueRef fieldVal = codegenNode(ctx, fldExprIdx, fieldLLVM);
			if (!fieldVal) return nullptr;
			fieldVal = coerceTo(ctx, fieldVal, fieldLLVM);
			// Alloca the union, write the chosen field, load the union back
			// as an aggregate value.
			JamValueRef alloca = JamLLVMBuildAlloca(
			    ctx.getBuilder(), uinfo->type, ctx.typeAlign(structType),
			    (uinfo->name + ".lit").c_str());
			JamLLVMBuildStore(ctx.getBuilder(), fieldVal, alloca);
			return JamLLVMBuildLoad(ctx.getBuilder(), uinfo->type, alloca,
			                        "union.val");
		}
		throw std::runtime_error("Struct literal target is not a struct");
	}

	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t fieldCount = ns.getExtra(extra);

	JamValueRef structVal = JamLLVMGetUndef(info->type);
	for (uint32_t i = 0; i < fieldCount; i++) {
		StringIdx fldNameId =
		    static_cast<StringIdx>(ns.getExtra(extra + 1 + i * 2));
		NodeIdx fldExprIdx =
		    static_cast<NodeIdx>(ns.getExtra(extra + 2 + i * 2));
		const std::string &fieldName = sp.get(fldNameId);
		int idx = ctx.getFieldIndex(info->name, fieldName);
		if (idx < 0) {
			throw std::runtime_error("Unknown field '" + fieldName +
			                         "' in struct " + info->name);
		}

		// Propagate target struct/array type into nested literals.
		TypeIdx declaredFieldType = info->fields[idx].second;
		const AstNode &fldNode = ns.get(fldExprIdx);
		if (fldNode.tag == AstTag::StructLit &&
		    ctx.lookupStruct(declaredFieldType)) {
			ctx.getNodeStore().getMut(fldExprIdx).lhs = declaredFieldType;
		}
		if ((fldNode.tag == AstTag::ArrayLit ||
		     fldNode.tag == AstTag::ArrayRepeat) &&
		    static_cast<TypeIdx>(fldNode.lhs) == kNoType) {
			const TypeKey &fk = ctx.getTypePool().get(declaredFieldType);
			if (fk.kind == TypeKind::Array) {
				ctx.getNodeStore().getMut(fldExprIdx).lhs = declaredFieldType;
			}
		}

		JamTypeRef expectedType = ctx.getLLVMType(declaredFieldType);
		JamValueRef fieldVal = codegenNode(ctx, fldExprIdx, expectedType);
		if (!fieldVal) return nullptr;

		JamTypeRef actualType = JamLLVMTypeOf(fieldVal);
		if (actualType != expectedType) {
			if (JamLLVMTypeIsFloat(expectedType) &&
			    JamLLVMTypeIsInteger(actualType)) {
				fieldVal = JamLLVMBuildSIToFP(ctx.getBuilder(), fieldVal,
				                              expectedType, "fld_si2fp");
			} else if (JamLLVMTypeIsFloat(expectedType) &&
			           JamLLVMTypeIsFloat(actualType)) {
				fieldVal = JamLLVMBuildFPCast(ctx.getBuilder(), fieldVal,
				                              expectedType, "fld_fpcast");
			} else if (JamLLVMTypeIsInteger(expectedType) &&
			           JamLLVMTypeIsInteger(actualType)) {
				fieldVal =
				    JamLLVMBuildIntCast(ctx.getBuilder(), fieldVal,
				                        expectedType, false, "fld_icast");
			}
		}

		structVal =
		    JamLLVMBuildInsertValue(ctx.getBuilder(), structVal, fieldVal,
		                            static_cast<unsigned>(idx), "fld_set");
	}
	return structVal;
}

static JamValueRef codegenMemberAccess(JamCodegenContext &ctx, const AstNode &n,
                                       NodeIdx selfIdx) {
	const NodeStore &ns = ctx.getNodeStore();
	const StringPool &sp = ctx.getStringPool();

	std::vector<StringIdx> path;
	StringIdx rootName = collectMemberChain(ns, selfIdx, path);
	if (rootName == kNoString) {
		// Chain contains a non-Variable root (typically an `Index` node,
		// as in `arr[i].field` or `g.things[i].pc`). Walk through the
		// shared lvalue resolver to get a pointer at the leaf, then load
		// the leaf value.
		LValueAddress addr = resolveLValueAddress(ctx, selfIdx);
		return JamLLVMBuildLoad(ctx.getBuilder(), addr.leafLLVMType, addr.ptr,
		                        "field");
	}

	// Enum-variant value: `EnumName.Variant` (no parens) resolves to
	// the discriminant for unit variants. For payloaded enums (those with at
	// least one payloaded variant), the result is a fully-formed enum
	// struct with tag set and payload bytes undefined — equivalent to
	// `EnumName.Variant()` with no payload args.
	if (path.size() == 1) {
		std::string enumName = sp.get(rootName);
		const std::string &variantName = sp.get(path[0]);
		const auto *einfo = ctx.getEnum(enumName);
		if (!einfo) {
			// Resolve type alias (`const OptI32 = Option(i32);`) so
			// generic enum unit-variant references work.
			TypeIdx aliasTarget = ctx.lookupTypeAlias(enumName);
			if (aliasTarget != kNoType) {
				if (const auto *info = ctx.lookupEnum(aliasTarget)) {
					enumName = info->name;
					einfo = info;
				}
			}
		}
		if (einfo) {
			int idx = ctx.getEnumVariantIndex(enumName, variantName);
			if (idx < 0) {
				throw std::runtime_error("Enum `" + enumName +
				                         "` has no variant `" + variantName +
				                         "`");
			}
			const auto &v = einfo->variants[idx];
			JamValueRef tagConst =
			    JamLLVMConstInt(ctx.getInt8Type(),
			                    static_cast<uint64_t>(v.discriminant), false);
			if (!einfo->hasPayloadVariant) {
				// path: enum is just i8.
				return tagConst;
			}
			if (!v.payloadTypes.empty()) {
				throw std::runtime_error(
				    "Variant `" + enumName + "." + variantName +
				    "` carries a payload; use `" + enumName + "." +
				    variantName + "(...)` to construct it");
			}
			// unit variant: build a {tag, payload-undef} struct value.
			JamTypeRef enumLLVMType = einfo->type;
			uint64_t enumAlign =
			    einfo->maxPayloadAlign > 1 ? einfo->maxPayloadAlign : 1;
			JamValueRef alloca = JamLLVMBuildAlloca(
			    ctx.getBuilder(), enumLLVMType, enumAlign, "enum.unit");
			JamValueRef tagPtr = JamLLVMBuildStructGEP(
			    ctx.getBuilder(), enumLLVMType, alloca, 0, "enum.tag");
			JamLLVMBuildStore(ctx.getBuilder(), tagConst, tagPtr);
			return JamLLVMBuildLoad(ctx.getBuilder(), enumLLVMType, alloca,
			                        "enum.val");
		}
	}

	if (!ctx.hasVariable(sp.get(rootName))) {
		std::string memberChain;
		for (StringIdx fld : path) {
			memberChain += ".";
			memberChain += sp.get(fld);
		}
		throw std::runtime_error("Unknown variable `" + sp.get(rootName) +
		                         "` in `" + sp.get(rootName) + memberChain +
		                         "`");
	}

	const std::string &varName = sp.get(rootName);
	TypeIdx varTy = ctx.getVariableType(varName);
	const TypeKey &varKey = ctx.getTypePool().get(varTy);

	// `slice.ptr` / `slice.len` projection (single-level).
	if (varKey.kind == TypeKind::Slice && path.size() == 1) {
		const std::string &member = sp.get(path[0]);
		if (member != "ptr" && member != "len") {
			throw std::runtime_error("Slice has no field '" + member +
			                         "' (only .ptr and .len)");
		}
		JamValueRef alloca = ctx.getVariable(varName);
		JamTypeRef sliceType = ctx.getLLVMType(varTy);
		JamValueRef sliceVal = JamLLVMBuildLoad(ctx.getBuilder(), sliceType,
		                                        alloca, varName.c_str());
		unsigned fieldIdx = (member == "ptr") ? 0 : 1;
		return JamLLVMBuildExtractValue(ctx.getBuilder(), sliceVal, fieldIdx,
		                                member.c_str());
	}

	// Union member read (single-level only). All fields share the
	// same address; reading uses the field's type at the union's
	// allocation. Reading a different field than the most recently
	// written one reinterprets the bits — that's the union's whole job.
	//
	// The parser interns user-named types as TypeKind::Named without
	// distinguishing struct from union, so dispatch on the registry
	// (lookupUnion accepts either Struct or Union kinds).
	if (path.size() == 1) {
		if (const auto *uinfo = ctx.lookupUnion(varTy)) {
			const std::string &member = sp.get(path[0]);
			TypeIdx fieldTy = ctx.getUnionFieldType(uinfo->name, member);
			if (fieldTy == kNoType) {
				throw std::runtime_error("Union `" + uinfo->name +
				                         "` has no field `" + member + "`");
			}
			JamValueRef alloca = ctx.getVariable(varName);
			JamTypeRef fieldLLVMType = ctx.getLLVMType(fieldTy);
			// Opaque-pointer LLVM: the alloca is just `ptr`; we load
			// the requested field type from it directly. No bitcast
			// required.
			return JamLLVMBuildLoad(ctx.getBuilder(), fieldLLVMType, alloca,
			                        member.c_str());
		}
	}

	const auto *info = ctx.lookupStruct(varTy);
	if (!info) {
		throw std::runtime_error("Cannot access field of non-struct `" +
		                         varName + "`");
	}

	// Walk the field chain with struct GEPs and load only the leaf field —
	// the previous code loaded the whole struct and walked it with
	// extractvalue, which is N×fieldSize of pointless memory traffic in
	// debug builds and reads poison from any uninitialized fields.
	JamValueRef alloca = ctx.getVariable(varName);
	JamValueRef leafPtr = alloca;
	JamTypeRef leafLLVMType = info->type;
	TypeIdx currentType = varTy;
	for (StringIdx fldId : path) {
		const auto *curInfo = ctx.lookupStruct(currentType);
		if (!curInfo) {
			throw std::runtime_error("Cannot access field '" + sp.get(fldId) +
			                         "' on non-struct type");
		}
		const std::string &fieldName = sp.get(fldId);
		int idx = ctx.getFieldIndex(curInfo->name, fieldName);
		if (idx < 0) {
			throw std::runtime_error("Unknown field '" + fieldName +
			                         "' in struct " + curInfo->name);
		}
		leafPtr = JamLLVMBuildStructGEP(ctx.getBuilder(), leafLLVMType, leafPtr,
		                                static_cast<unsigned>(idx),
		                                fieldName.c_str());
		currentType = curInfo->fields[idx].second;
		leafLLVMType = ctx.getLLVMType(currentType);
	}
	return JamLLVMBuildLoad(ctx.getBuilder(), leafLLVMType, leafPtr, "field");
	(void)n;
}

JamValueRef codegenNode(JamCodegenContext &ctx, NodeIdx node,
                        JamTypeRef expectedType) {
	const AstNode &n = ctx.getNodeStore().get(node);
	switch (n.tag) {
	case AstTag::Invalid:
		throw std::runtime_error("Codegen on invalid node");
	case AstTag::NumberLit:
		return numberLitConst(ctx, n, expectedType);
	case AstTag::BoolLit:
		return JamLLVMConstInt(ctx.getInt1Type(), n.lhs ? 1 : 0, false);
	case AstTag::StringLit:
		return codegenStringLit(
		    ctx, ctx.getStringPool().get(static_cast<StringIdx>(n.lhs)));
	case AstTag::StructExpr:
		// a `struct {...}` expression evaluates to a `type`
		// value at compile time. The substitution engine in consumes
		// these from ModuleAST::AnonStructs. Reaching this case during
		// regular codegen means a generic function leaked through —
		// generics should be skipped at the function-emit pass.
		throw std::runtime_error(
		    "internal: struct expression reached LLVM codegen "
		    "(generic was not instantiated before lowering)");
	case AstTag::EnumExpr:
		// Same story for `enum { ... }` expressions — the substitution
		// engine consumes these from ModuleAST::AnonEnums.
		throw std::runtime_error(
		    "internal: enum expression reached LLVM codegen "
		    "(generic was not instantiated before lowering)");
	case AstTag::Variable: {
		const std::string &name =
		    ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
		JamValueRef V = ctx.getVariable(name);
		if (V) {
			JamTypeRef LoadType = ctx.getLLVMType(ctx.getVariableType(name));
			return JamLLVMBuildLoad(ctx.getBuilder(), LoadType, V,
			                        name.c_str());
		}
		// Fall back to module-scope `const NAME[: T]? = expr;` — re-codegen
		// the init expression in place. With the declared type passed as
		// expectedType, integer literals adopt the declared width naturally.
		if (const auto *mc = ctx.getModuleConst(name)) {
			JamTypeRef expected = mc->declaredType != kNoType
			                          ? ctx.getLLVMType(mc->declaredType)
			                          : nullptr;
			return codegenNode(ctx, mc->initExpr, expected);
		}
		throw std::runtime_error("Unknown variable name: " + name);
	}
	case AstTag::MemberAccess:
		return codegenMemberAccess(ctx, n, node);
	case AstTag::Index:
		return codegenIndex(ctx, n);
	case AstTag::Deref:
		return codegenDeref(ctx, n);
	case AstTag::AddressOf:
		return codegenAddressOf(ctx, n);
	case AstTag::UnaryOp:
		return codegenUnaryOp(ctx, n);
	case AstTag::BinaryOp:
		return codegenBinaryOp(ctx, n);
	case AstTag::Call:
		return codegenCall(ctx, n);
	case AstTag::AtCall:
		return codegenAtCall(ctx, n);
	case AstTag::TypeMethodCall:
		return codegenTypeMethodCall(ctx, n);
	case AstTag::Return:
		return codegenReturn(ctx, n);
	case AstTag::Assign:
		return codegenAssign(ctx, n);
	case AstTag::VarDecl:
		return codegenVarDecl(ctx, n);
	case AstTag::IfNode:
		return codegenIf(ctx, n);
	case AstTag::WhileNode:
		return codegenWhile(ctx, n);
	case AstTag::ForNode:
		return codegenFor(ctx, n);
	case AstTag::Break:
		if (!CurrentLoopBreak) {
			throw std::runtime_error("break statement not inside a loop");
		}
		// drop all scopes inside (and including) the enclosing
		// loop body before exiting the loop.
		emitDropsThroughScope(ctx, CurrentLoopBodyScopeIdx);
		JamLLVMBuildBr(ctx.getBuilder(), CurrentLoopBreak);
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	case AstTag::Continue:
		if (!CurrentLoopContinue) {
			throw std::runtime_error("continue statement not inside a loop");
		}
		// drop all scopes inside (and including) the enclosing
		// loop body before jumping to the next iteration.
		emitDropsThroughScope(ctx, CurrentLoopBodyScopeIdx);
		JamLLVMBuildBr(ctx.getBuilder(), CurrentLoopContinue);
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	case AstTag::ImportLit:
		return JamLLVMConstInt(ctx.getInt8Type(), 0, false);
	case AstTag::StructLit:
		return codegenStructLit(ctx, n);
	case AstTag::ArrayLit:
		return codegenArrayLit(ctx, n);
	case AstTag::ArrayRepeat:
		return codegenArrayRepeat(ctx, n);
	case AstTag::MatchNode:
		return codegenMatch(ctx, n);
	case AstTag::AsCast: {
		// `expr as Type` — explicit conversion. Handles:
		//   • integer <-> integer (truncate/extend)
		//   • integer <-> float (siToFP / fpToSI)
		//   • float <-> float (FPCast)
		//   • enum <-> integer: extracts the tag for payloaded enums (which
		//     are {tag, payload} structs); identity for unit-only enums
		//     (which are already i8).
		NodeIdx operandIdx = static_cast<NodeIdx>(n.lhs);
		TypeIdx targetTy = static_cast<TypeIdx>(n.rhs);
		JamTypeRef targetLLVM = ctx.getLLVMType(targetTy);
		const AstNode &operandNode = ctx.getNodeStore().get(operandIdx);
		JamTypeRef innerExpected = (operandNode.tag == AstTag::NumberLit &&
		                            JamLLVMTypeIsInteger(targetLLVM))
		                               ? targetLLVM
		                               : nullptr;
		JamValueRef val = codegenNode(ctx, operandIdx, innerExpected);
		if (!val) return nullptr;
		JamTypeRef srcLLVM = JamLLVMTypeOf(val);
		if (srcLLVM == targetLLVM) return val;
		// Enum-to-integer: extract the tag (struct field 0) when the
		// source is a struct, then cast that i8 to the target width.
		if (JamLLVMTypeIsStruct(srcLLVM) &&
		    ctx.findEnumByLLVMType(srcLLVM) != nullptr &&
		    JamLLVMTypeIsInteger(targetLLVM)) {
			JamValueRef tag =
			    JamLLVMBuildExtractValue(ctx.getBuilder(), val, 0, "as.tag");
			if (JamLLVMTypeOf(tag) == targetLLVM) return tag;
			return JamLLVMBuildIntCast(ctx.getBuilder(), tag, targetLLVM, false,
			                           "as.tag.cast");
		}
		if (JamLLVMTypeIsInteger(srcLLVM) && JamLLVMTypeIsInteger(targetLLVM)) {
			// sign-extend when widening a signed source so e.g. an i8
			// holding -16 becomes i64 -16 rather than i64 +240.
			bool signedSrc = isSignedIntExpr(ctx, operandIdx);
			return JamLLVMBuildIntCast(ctx.getBuilder(), val, targetLLVM,
			                           signedSrc, "as.icast");
		}
		if (JamLLVMTypeIsInteger(srcLLVM) && JamLLVMTypeIsFloat(targetLLVM)) {
			// Pick signed vs unsigned conversion from the source operand.
			// Otherwise `255u as f32` would round-trip as -1.0 because
			// SIToFP treats the high bit as a sign.
			bool signedSrc = isSignedIntExpr(ctx, operandIdx);
			return signedSrc ? JamLLVMBuildSIToFP(ctx.getBuilder(), val,
			                                      targetLLVM, "as.si2fp")
			                 : JamLLVMBuildUIToFP(ctx.getBuilder(), val,
			                                      targetLLVM, "as.ui2fp");
		}
		if (JamLLVMTypeIsFloat(srcLLVM) && JamLLVMTypeIsInteger(targetLLVM)) {
			// Conversion truncates toward zero (matches C semantics).
			// Dest signedness picks FPToSI vs FPToUI. TypeKey.b is set
			// on signed integer types (see ast_flat.h's seed of i8/i16/…).
			const TypeKey &tk = ctx.getTypePool().get(targetTy);
			bool signedDst = tk.kind == TypeKind::Int && tk.b != 0;
			return signedDst ? JamLLVMBuildFPToSI(ctx.getBuilder(), val,
			                                      targetLLVM, "as.fp2si")
			                 : JamLLVMBuildFPToUI(ctx.getBuilder(), val,
			                                      targetLLVM, "as.fp2ui");
		}
		if (JamLLVMTypeIsFloat(srcLLVM) && JamLLVMTypeIsFloat(targetLLVM)) {
			return JamLLVMBuildFPCast(ctx.getBuilder(), val, targetLLVM,
			                          "as.fpcast");
		}
		throw std::runtime_error("Unsupported `as` cast between these types");
	}
	case AstTag::PatLit:
	case AstTag::PatRange:
	case AstTag::PatWildcard:
	case AstTag::PatOr:
	case AstTag::PatEnumVariant:
		throw std::runtime_error(
		    "Pattern node reached top-level codegen; patterns are only "
		    "valid inside a `match` arm");
	case AstTag::Count:
		break;
	}
	throw std::runtime_error("Unhandled AST tag in codegen");
	(void)expectedType;
}

JamValueRef resolveLvaluePtr(JamCodegenContext &ctx, NodeIdx node,
                             JamTypeRef &outElemType) {
	(void)ctx;
	(void)node;
	(void)outElemType;
	throw std::runtime_error("resolveLvaluePtr is not yet exposed");
}

// FunctionAST codegen — split into declarePrototype + defineBody so the
// driver can register every function's signature before it starts emitting
// any body. That's what lets `main` (or any caller) appear above its
// callees in source order.

// name-mangling for `fn drop(self: mut T)`. When the source-level
// name "drop" coincides with a recognizable drop-fn signature, mangle to
// "__drop_<TypeName>" at the LLVM level so multiple types can each have
// their own drop fn without colliding. Used by declarePrototype,
// defineBody, and the drop-emission helper so all three see the same
// LLVM function name.
static std::string mangledFunctionName(const FunctionAST &fn,
                                       const TypePool &types,
                                       const StringPool &strings) {
	if (fn.isTest) return "__test_" + fn.Name;
	if (fn.Name == "drop" && fn.Args.size() == 1) {
		const Param &p = fn.Args[0];
		if (p.Name == "self" && p.Mode == ParamMode::Mut) {
			const TypeKey &k = types.get(p.Type);
			if (k.kind == TypeKind::Struct || k.kind == TypeKind::Named) {
				StringIdx ni = static_cast<StringIdx>(k.a);
				if (ni != kNoString) { return "__drop_" + strings.get(ni); }
			}
		}
	}
	return fn.Name;
}

JamFunctionRef FunctionAST::declarePrototype(JamCodegenContext &ctx) {
	std::string funcName =
	    mangledFunctionName(*this, ctx.getTypePool(), ctx.getStringPool());

	// return ABI: large aggregates returned by Jam-defined fns
	// (not extern, not test) are sret — caller passes a leading
	// `ptr sret(%T) align A` arg, callee stores into it and returns
	// void. Small returns stay direct.
	jam::abi::ReturnABI rabi =
	    (isTest || isExtern)
	        ? jam::abi::ReturnABI{jam::abi::ReturnABI::Kind::Direct,
	                              (isTest || ReturnType == kNoType)
	                                  ? ctx.getVoidType()
	                                  : ctx.getLLVMType(ReturnType),
	                              0}
	        : jam::abi::classifyReturn(ReturnType, ctx);

	std::vector<JamTypeRef> ArgTypes;
	if (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
		// sret slot leads the LLVM arg list.
		ArgTypes.push_back(JamLLVMPointerType(ctx.getLLVMType(ReturnType), 0));
	}
	if (!isTest) {
		for (const auto &arg : Args) {
			if (isExtern) {
				// extern fns follow the C ABI literally. The user
				// already wrote the parameter types as they want them
				// to appear at the FFI boundary (e.g. `*const T` for a
				// pointer arg, `u32` for a scalar). LLVM's backend
				// handles `byval` for large aggregates per the
				// platform's MEMORY classification; we do NOT
				// re-classify with mode-aware rules.
				ArgTypes.push_back(ctx.getLLVMType(arg.Type));
				continue;
			}
			// mode-aware ABI. classifyParam decides per-(mode, type)
			// whether the parameter is passed by value (the type's natural
			// LLVM representation) or by pointer.
			jam::abi::ParamABI pabi =
			    jam::abi::classifyParam(arg.Mode, arg.Type, ctx);
			if (pabi.kind == jam::abi::ParamABI::Kind::ByPointer) {
				ArgTypes.push_back(
				    JamLLVMPointerType(ctx.getLLVMType(arg.Type), 0));
			} else {
				ArgTypes.push_back(pabi.llvmType);
			}
		}
	}

	JamTypeRef RetType = (rabi.kind == jam::abi::ReturnABI::Kind::Indirect)
	                         ? ctx.getVoidType()
	                         : rabi.directType;

	JamTypeRef FT = JamLLVMFunctionType(RetType, ArgTypes.data(),
	                                    ArgTypes.size(), isVarArgs);

	JamFunctionRef F =
	    JamLLVMAddFunction(ctx.getModule(), funcName.c_str(), FT);
	JamLLVMApplyDefaultFnAttrs(F, isExtern);
	if (ReturnType == BuiltinType::NoReturn) {
		JamLLVMSetFunctionNoReturn(F);
	}

	// Apply sret attributes to the leading parameter when applicable.
	if (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
		JamLLVMAddParamAttrSret(F, 0, ctx.getLLVMType(ReturnType),
		                        rabi.sretAlign);
	}

	if (isExtern || isExport || Name == "main") {
		JamLLVMSetLinkage((JamValueRef)F, JAM_LINKAGE_EXTERNAL);
	} else {
		JamLLVMSetLinkage((JamValueRef)F, JAM_LINKAGE_INTERNAL);
	}

	// when the function uses sret, the user's parameter at source
	// index `i` lives at LLVM index `i + 1` (the sret slot is index 0).
	const unsigned argOffset =
	    (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) ? 1u : 0u;

	if (isExtern || isExport || Name == "main") {
		JamLLVMSetFunctionCallConv(F, JAM_CALLCONV_C);
		for (unsigned i = 0; i < Args.size(); i++) {
			if (Args[i].Type == BuiltinType::Bool) {
				JamLLVMAddParamAttrZeroExt(F, i + argOffset);
			}
		}
		if (ReturnType == BuiltinType::Bool) { JamLLVMAddRetAttrZeroExt(F); }
	}

	for (unsigned i = 0; i < Args.size(); i++) {
		JamValueRef param = JamLLVMGetParam(F, i + argOffset);
		JamLLVMSetValueName(param, Args[i].Name.c_str());
	}

	return F;
}

//  - `return`, `break`, `continue`               — diverge by definition
//  - `Call(callee)` where callee is `noreturn`   — diverge
//  - `if/else` where both branches diverge       — diverge
//  - `if (true) { … diverges }`                  — diverge (const-fold)
//  - `if (false) ... else { … diverges }`         — diverge (const-fold)
//  - `match` where every arm diverges            — diverge
//  - `while (true) { … }` with no reachable break — diverge
//
// anything else (var-decls, assignments, plain calls, …) reports false,
// meaning "this statement could fall through." The function-tail check
// in `defineBody` uses this to:
//   - emit `unreachable` after a diverging tail (so the IR verifies);
//   - error on a non-diverging tail in a non-void / `noreturn` fn.

static bool stmtDiverges(JamCodegenContext &ctx, NodeIdx idx);

// Returns true iff `last-of-sequence` diverges. An empty sequence
// trivially does not diverge (control falls through to the caller).
static bool seqDiverges(JamCodegenContext &ctx, ExtraIdx base,
                        uint32_t count) {
	if (count == 0) return false;
	NodeIdx last =
	    static_cast<NodeIdx>(ctx.getNodeStore().getExtra(base + count - 1));
	return stmtDiverges(ctx, last);
}

// Compile-time bool literal extractor. Sets `*val` and returns true
// when `idx` is a `BoolLit` node; returns false otherwise.
static bool tryEvalCompileTimeBool(JamCodegenContext &ctx, NodeIdx idx,
                                   bool *val) {
	const AstNode &n = ctx.getNodeStore().get(idx);
	if (n.tag == AstTag::BoolLit) {
		*val = (n.lhs != 0);
		return true;
	}
	return false;
}

// True iff a `Call` node's target function is declared `noreturn`.
// Dynamic / unresolved calls return false (conservative).
static bool calleeIsNoReturn(JamCodegenContext &ctx, const AstNode &n) {
	if (n.flags & 1) return false;  // dynamic call
	const std::string &name =
	    ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	const FunctionAST *fn = ctx.getFunctionAST(name);
	return fn && fn->ReturnType == BuiltinType::NoReturn;
}

// True iff `idx` contains a `break` reachable from the current
// loop's lexical level (i.e., not nested inside another while/for).
// Used by `while (true)` to decide whether the loop can exit.
static bool containsReachableBreak(JamCodegenContext &ctx, NodeIdx idx) {
	const NodeStore &ns = ctx.getNodeStore();
	const AstNode &n = ns.get(idx);
	switch (n.tag) {
	case AstTag::Break:
		return true;
	case AstTag::WhileNode:
	case AstTag::ForNode:
		return false;  // break inside a nested loop belongs to that loop
	case AstTag::IfNode: {
		ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
		uint32_t thenCount = ns.getExtra(ex);
		uint32_t elseCount = ns.getExtra(ex + 1);
		ExtraIdx then0 = ex + 2;
		ExtraIdx else0 = then0 + thenCount;
		for (uint32_t i = 0; i < thenCount; i++) {
			if (containsReachableBreak(
			        ctx, static_cast<NodeIdx>(ns.getExtra(then0 + i)))) {
				return true;
			}
		}
		for (uint32_t i = 0; i < elseCount; i++) {
			if (containsReachableBreak(
			        ctx, static_cast<NodeIdx>(ns.getExtra(else0 + i)))) {
				return true;
			}
		}
		return false;
	}
	case AstTag::MatchNode: {
		ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
		uint32_t armCount = ns.getExtra(ex);
		uint32_t pos = 1;
		for (uint32_t a = 0; a < armCount; a++) {
			uint32_t bodyCount = ns.getExtra(ex + pos + 1);
			ExtraIdx bodyStart = ex + pos + 2;
			for (uint32_t b = 0; b < bodyCount; b++) {
				if (containsReachableBreak(
				        ctx, static_cast<NodeIdx>(ns.getExtra(bodyStart + b)))) {
					return true;
				}
			}
			pos = pos + 2 + bodyCount;
		}
		return false;
	}
	default:
		return false;
	}
}

static bool stmtDiverges(JamCodegenContext &ctx, NodeIdx idx) {
	const NodeStore &ns = ctx.getNodeStore();
	const AstNode &n = ns.get(idx);
	switch (n.tag) {
	case AstTag::Return:
	case AstTag::Break:
	case AstTag::Continue:
		return true;
	case AstTag::Call:
		return calleeIsNoReturn(ctx, n);
	case AstTag::IfNode: {
		NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
		ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
		uint32_t thenCount = ns.getExtra(ex);
		uint32_t elseCount = ns.getExtra(ex + 1);
		ExtraIdx then0 = ex + 2;
		ExtraIdx else0 = then0 + thenCount;
		bool condVal;
		if (tryEvalCompileTimeBool(ctx, condIdx, &condVal)) {
			return condVal ? seqDiverges(ctx, then0, thenCount)
			               : seqDiverges(ctx, else0, elseCount);
		}
		// Runtime cond: both branches must diverge for the whole
		// expression to. A missing else (elseCount == 0) means the
		// false path falls through.
		if (elseCount == 0) return false;
		return seqDiverges(ctx, then0, thenCount) &&
		       seqDiverges(ctx, else0, elseCount);
	}
	case AstTag::MatchNode: {
		ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
		uint32_t armCount = ns.getExtra(ex);
		uint32_t pos = 1;
		for (uint32_t a = 0; a < armCount; a++) {
			uint32_t bodyCount = ns.getExtra(ex + pos + 1);
			ExtraIdx bodyStart = ex + pos + 2;
			if (!seqDiverges(ctx, bodyStart, bodyCount)) return false;
			pos = pos + 2 + bodyCount;
		}
		return armCount > 0;
	}
	case AstTag::WhileNode: {
		NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
		bool condVal;
		if (!tryEvalCompileTimeBool(ctx, condIdx, &condVal) || !condVal) {
			return false;
		}
		ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
		uint32_t bodyCount = ns.getExtra(ex);
		ExtraIdx body0 = ex + 1;
		for (uint32_t i = 0; i < bodyCount; i++) {
			if (containsReachableBreak(
			        ctx, static_cast<NodeIdx>(ns.getExtra(body0 + i)))) {
				return false;
			}
		}
		return true;
	}
	default:
		return false;
	}
}

void FunctionAST::defineBody(JamCodegenContext &ctx) {
	if (isExtern) return;

	std::string funcName =
	    mangledFunctionName(*this, ctx.getTypePool(), ctx.getStringPool());
	JamFunctionRef F = JamLLVMGetFunction(ctx.getModule(), funcName.c_str());
	if (!F) {
		throw std::runtime_error("defineBody: prototype not declared for " +
		                         funcName);
	}

	JamBasicBlockRef BB = JamLLVMAppendBasicBlock(F, "entry");
	JamLLVMPositionBuilderAtEnd(ctx.getBuilder(), BB);

	ctx.clearVariables();
	ctx.clearDrops();
	ctx.pushDropScope();  // function-level scope

	// if this function uses sret, the leading LLVM arg is the
	// caller-provided result slot. Record it so codegenReturn writes
	// through it; user parameters shift one slot to the right.
	jam::abi::ReturnABI rabi =
	    isExtern ? jam::abi::ReturnABI{jam::abi::ReturnABI::Kind::Direct,
	                                   ReturnType == kNoType
	                                       ? ctx.getVoidType()
	                                       : ctx.getLLVMType(ReturnType),
	                                   0}
	             : jam::abi::classifyReturn(ReturnType, ctx);
	unsigned argOffset = 0;
	if (rabi.kind == jam::abi::ReturnABI::Kind::Indirect) {
		ctx.setSretSlot(JamLLVMGetParam(F, 0));
		argOffset = 1;
	} else {
		ctx.setSretSlot(nullptr);
	}

	// Record the function's source-level return TypeIdx so codegenReturn
	// can patch struct-literal-shaped return expressions whose target
	// type is parser-time kNoType.
	ctx.setCurrentReturnType(ReturnType);

	for (unsigned i = 0; i < Args.size(); i++) {
		// mode-aware ABI: ByValue parameters are stored to a local
		// alloca on entry (matching the existing pattern for value
		// semantics). ByPointer parameters are *already* pointers to
		// caller-owned storage; we register the parameter directly as
		// the variable's place — reads will load through it, writes
		// will store through it, and the caller observes the mutations.
		jam::abi::ParamABI pabi =
		    jam::abi::classifyParam(Args[i].Mode, Args[i].Type, ctx);
		JamValueRef param = JamLLVMGetParam(F, i + argOffset);
		if (pabi.kind == jam::abi::ParamABI::Kind::ByPointer) {
			ctx.setVariable(Args[i].Name, param);
		} else {
			JamTypeRef ArgType = ctx.getLLVMType(Args[i].Type);
			JamValueRef Alloca = JamLLVMBuildAlloca(ctx.getBuilder(), ArgType,
			                                        ctx.typeAlign(Args[i].Type),
			                                        Args[i].Name.c_str());
			JamLLVMBuildStore(ctx.getBuilder(), param, Alloca);
			ctx.setVariable(Args[i].Name, Alloca);
		}
		ctx.setVariableType(Args[i].Name, Args[i].Type);
	}

	for (NodeIdx stmt : Body) { codegenNode(ctx, stmt); }

	if (!JamLLVMGetBasicBlockTerminator(
	        JamLLVMGetInsertBlock(ctx.getBuilder()))) {
		//   1. Void return type — emit `ret void`.
		//   2. `noreturn` — body must diverge. Falls-through is an error.
		//   3. Non-void, body diverges — emit `unreachable`.
		//   4. Non-void, body may fall through — error (missing return).
		// Divergence is determined by AST-level analysis on the last
		// statement (see stmtDiverges), modeling Zig's noreturn-type
		// propagation through expressions.
		bool diverges = !Body.empty() && stmtDiverges(ctx, Body.back());
		if (ReturnType == kNoType) {
			emitTopScopeDrops(ctx);
			JamLLVMBuildRetVoid(ctx.getBuilder());
		} else if (ReturnType == BuiltinType::NoReturn) {
			if (!diverges) {
				throw std::runtime_error(
				    "function `" + Name +
				    "` is declared `noreturn` but its body may complete "
				    "without diverging");
			}
			emitTopScopeDrops(ctx);
			JamLLVMBuildUnreachable(ctx.getBuilder());
		} else if (diverges) {
			emitTopScopeDrops(ctx);
			JamLLVMBuildUnreachable(ctx.getBuilder());
		} else {
			throw std::runtime_error(
			    "function `" + Name +
			    "` has a non-void return type but its body may complete "
			    "without returning a value");
		}
	}
	ctx.popDropScope();
	ctx.clearDrops();
}

JamFunctionRef FunctionAST::codegen(JamCodegenContext &ctx) {
	JamFunctionRef F = declarePrototype(ctx);
	defineBody(ctx);
	return F;
}
