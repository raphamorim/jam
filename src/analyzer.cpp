/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "analyzer.h"

#include "abi.h"
#include "ast.h"
#include "codegen.h"

#include <algorithm>
#include <sstream>

namespace jam {

Analyzer::Analyzer(JamCodegenContext &ctx, DeclTable &decls)
    : ctx_(ctx), decls_(decls) {}

SrcLoc Analyzer::locOf(const Decl &d) const {
	SrcLoc loc;
	loc.file = d.file.empty() ? ctx_.currentFile() : d.file;
	loc.line = d.line;
	return loc;
}

void Analyzer::pushCycleError(DeclIndex repeated) {
	const Decl &repeatedDecl = decls_.get(repeated);
	std::ostringstream chain;
	// The cycle is the suffix of analysisStack_ that begins with the
	// first occurrence of `repeated`. Walk from there to the end so
	// the user reads the loop in source order: A → B → A.
	auto first =
	    std::find(analysisStack_.begin(), analysisStack_.end(), repeated);
	chain << "dependency loop detected: ";
	for (auto it = first; it != analysisStack_.end(); ++it) {
		if (it != first) chain << " → ";
		chain << "`" << decls_.get(*it).name << "`";
	}
	chain << " → `" << repeatedDecl.name << "`";
	// The chain itself is in the message, so the reference trace
	// would just duplicate it. Skip the trace here; other diagnostics
	// that don't print the chain get it via emitErrorWithRefTrace.
	ctx_.diagnostics().error(locOf(repeatedDecl), chain.str());
}

std::vector<Diagnostic::Trace>
Analyzer::buildReferenceTrace(bool excludeTop) const {
	std::vector<Diagnostic::Trace> trace;
	if (analysisStack_.empty()) return trace;
	// Walk the stack from outermost to innermost so the user reads
	// "referenced by `outer` ... referenced by `middle`" in causal
	// order. `excludeTop` drops the innermost frame, used when the
	// error message already names that decl.
	std::size_t end = analysisStack_.size();
	if (excludeTop && end > 0) --end;
	for (std::size_t i = 0; i < end; ++i) {
		const Decl &d = decls_.get(analysisStack_[i]);
		Diagnostic::Trace t;
		t.kind = Diagnostic::Trace::Kind::Reference;
		t.decl = d.name;
		t.loc = locOf(d);
		trace.push_back(std::move(t));
	}
	return trace;
}

void Analyzer::emitErrorWithRefTrace(SrcLoc loc, std::string message,
                                     bool excludeTop) {
	auto trace = buildReferenceTrace(excludeTop);
	if (trace.empty()) {
		ctx_.diagnostics().error(std::move(loc), std::move(message));
		return;
	}
	ctx_.diagnostics().errorWithTrace(std::move(loc), std::move(message),
	                                  std::move(trace));
}

DeclValue Analyzer::ensureDeclAnalyzed(DeclIndex idx) {
	if (idx == kNoDecl) return {};
	Decl &d = decls_.get(idx);

	switch (d.analysis) {
	case DeclAnalysis::Complete:
		return d.value;
	case DeclAnalysis::InProgress:
		pushCycleError(idx);
		// Don't mark this decl as failed — the cycle is the *caller's*
		// problem too. Returning a poison value lets the caller decide
		// whether to bail. The next ensureDeclAnalyzed for this same
		// idx (after the stack unwinds) will hit `Complete` cleanly.
		return {};
	case DeclAnalysis::AnalysisFailure:
	case DeclAnalysis::DependencyFailure:
		return {};
	case DeclAnalysis::Unreferenced:
		break;
	}

	// Move to InProgress, dispatch, then to Complete (or failure).
	d.analysis = DeclAnalysis::InProgress;
	analysisStack_.push_back(idx);
	DeclValue value = analyzeDecl(idx);
	analysisStack_.pop_back();

	// Re-fetch — analyzeDecl may have appended new decls and
	// invalidated our reference. Decl& is stable across creates only
	// when std::vector doesn't reallocate; safer to look up again.
	Decl &again = decls_.get(idx);
	if (again.analysis == DeclAnalysis::InProgress) {
		// Branch didn't set a terminal state: default to Complete with
		// whatever value was returned. (A branch that pushed a cycle
		// error returns None but stays InProgress so the unwind path
		// here marks it Complete-with-None — that's a *resolved
		// state*, callers see "no value" but no infinite recursion on
		// retry.)
		again.analysis = DeclAnalysis::Complete;
		again.value = value;
	}
	return again.value;
}

bool Analyzer::resolveTypeFieldsStruct(DeclIndex idx) {
	if (idx == kNoDecl) return false;
	Decl &d = decls_.get(idx);
	if (d.kind != DeclKind::Struct) return false;

	switch (d.structStatus) {
	case StructStatus::HaveFieldTypes:
	case StructStatus::HaveLayout:
		return true;
	case StructStatus::FieldTypesWIP: {
		// Distinct error from the Decl-level "dependency loop" — this
		// is specifically a struct whose own fields reference itself
		// (via something that wasn't a pointer or slice indirection).
		// Build a Reference trace from the struct-fill stack so the
		// user sees the chain `B referenced by A` when a multi-step
		// cycle hits (the message itself only names the cycle's
		// closing decl).
		std::vector<Diagnostic::Trace> trace;
		for (DeclIndex anc : structFillStack_) {
			if (anc == idx) continue;  // skip the closing decl
			const Decl &ad = decls_.get(anc);
			Diagnostic::Trace t;
			t.kind = Diagnostic::Trace::Kind::Reference;
			t.decl = ad.name;
			t.loc = locOf(ad);
			trace.push_back(std::move(t));
		}
		if (trace.empty()) {
			ctx_.diagnostics().error(locOf(d), "struct `" + d.name +
			                                       "` depends on itself");
		} else {
			ctx_.diagnostics().errorWithTrace(
			    locOf(d), "struct `" + d.name + "` depends on itself",
			    std::move(trace));
		}
		return false;
	}
	case StructStatus::LayoutWIP:
		// Layout depends on field types, so if we're here the field
		// types are already known; treat as success.
		return true;
	case StructStatus::None:
		break;
	}

	d.structStatus = StructStatus::FieldTypesWIP;
	structFillStack_.push_back(idx);

	// Walk every field and materialise its LLVM type. Each call to
	// `getLLVMType` is a chance to re-enter the analyzer (when the
	// field's TypeIdx is a Named struct whose body isn't yet set, or
	// a GenericCall): that re-entry hits the chokepoint and either
	// detects the FieldTypesWIP cycle (self-ref) or recurses into a
	// different struct's resolution. The previous eager
	// `fillStructBodies` in main.cpp did this inline; routing through
	// the analyzer is what makes the cycle detection structural.
	if (d.structAst == nullptr) {
		// No AST attached — Decl was created with a name but no
		// body. Treat as already-resolved so callers don't loop.
		d.structStatus = StructStatus::HaveFieldTypes;
		structFillStack_.pop_back();
		return true;
	}
	const auto &fields = d.structAst->Fields;
	std::vector<JamTypeRef> fieldTypes;
	fieldTypes.reserve(fields.size());
	bool fieldFillFailed = false;
	for (const auto &f : fields) {
		fieldTypes.push_back(ctx_.getLLVMType(f.second));
		// Cycle detection sits at the field position, not in
		// getLLVMType — because getLLVMType for `*mut S` also recurses
		// into S, and a pointer field does NOT make S sized-dependent
		// on itself. By peeking at the TypeKey here we only trigger
		// the matching ensure*Body when the field is a *direct* named
		// type (TypeKind::Named/Struct/Enum/Union, not PtrSingle/
		// PtrMany/Slice/Array), so linked-list idioms keep working
		// while `S { x: S }` (or struct→union→struct chains) correctly
		// trips the WIP cycle check.
		const TypeKey &fk = ctx_.getTypePool().get(f.second);
		if (fk.kind == TypeKind::Named || fk.kind == TypeKind::Struct ||
		    fk.kind == TypeKind::Enum || fk.kind == TypeKind::Union) {
			const std::string &fieldName =
			    ctx_.getStringPool().get(static_cast<StringIdx>(fk.a));
			bool nestedOk = true;
			if (ctx_.getStruct(fieldName)) {
				nestedOk = ensureStructBody(fieldName);
			} else if (ctx_.getEnum(fieldName)) {
				nestedOk = ensureEnumBody(fieldName);
			} else if (ctx_.getUnion(fieldName)) {
				nestedOk = ensureUnionBody(fieldName);
			}
			if (!nestedOk) fieldFillFailed = true;
		}
	}
	const auto *info = ctx_.getStruct(d.name);
	if (info != nullptr && info->type != nullptr) {
		if (fieldFillFailed) {
			// Downstream pushed a diagnostic; leave a single-i8 body
			// so callers can still query a size without UB.
			JamTypeRef stub = ctx_.getInt8Type();
			JamLLVMStructSetBody(info->type, &stub, 1, false);
		} else {
			JamLLVMStructSetBody(info->type, fieldTypes.data(),
			                     static_cast<unsigned>(fieldTypes.size()),
			                     false);
		}
	}
	d.structStatus = StructStatus::HaveFieldTypes;
	structFillStack_.pop_back();
	return !fieldFillFailed;
}

bool Analyzer::ensureStructBody(const std::string &name) {
	// Jam permits a function and a struct to share a name (the type
	// shows up in type position, the function in call position).
	// Use the kind-aware finder so we land on the Struct decl even
	// when a same-name function was registered first.
	DeclIndex idx = decls_.findByNameAndKind(name, DeclKind::Struct);
	if (idx == kNoDecl) return false;
	return resolveTypeFieldsStruct(idx);
}

bool Analyzer::ensureEnumBody(const std::string &name) {
	DeclIndex idx = decls_.findByNameAndKind(name, DeclKind::Enum);
	if (idx == kNoDecl) return false;
	return resolveTypeFieldsEnum(idx);
}

bool Analyzer::ensureUnionBody(const std::string &name) {
	DeclIndex idx = decls_.findByNameAndKind(name, DeclKind::Union);
	if (idx == kNoDecl) return false;
	return resolveTypeFieldsUnion(idx);
}

DeclValue Analyzer::resolveDecl(const std::string &name) {
	DeclIndex idx = decls_.findByName(name);
	if (idx == kNoDecl) return {};
	return ensureDeclAnalyzed(idx);
}

bool Analyzer::resolveTypeFieldsEnum(DeclIndex idx) {
	if (idx == kNoDecl) return false;
	Decl &d = decls_.get(idx);
	if (d.kind != DeclKind::Enum) return false;

	switch (d.enumStatus) {
	case EnumStatus::HaveBody:
		return true;
	case EnumStatus::BodyWIP: {
		// Self-cycle: a payload type references the enum itself.
		std::vector<Diagnostic::Trace> trace;
		for (DeclIndex anc : enumFillStack_) {
			if (anc == idx) continue;
			const Decl &ad = decls_.get(anc);
			Diagnostic::Trace t;
			t.kind = Diagnostic::Trace::Kind::Reference;
			t.decl = ad.name;
			t.loc = locOf(ad);
			trace.push_back(std::move(t));
		}
		if (trace.empty()) {
			ctx_.diagnostics().error(locOf(d),
			                         "enum `" + d.name + "` depends on itself");
		} else {
			ctx_.diagnostics().errorWithTrace(
			    locOf(d), "enum `" + d.name + "` depends on itself",
			    std::move(trace));
		}
		return false;
	}
	case EnumStatus::None:
		break;
	}

	// Layout matches main.cpp's prior fillEnumBodies: only payloaded
	// enums need a struct body; unit-only enums lower to plain i8 and
	// have no LLVM struct to set. The math computes per-variant size /
	// alignment, picks an alignDriver scalar matching the strictest
	// variant's alignment, and pads the trailing slot up to the worst-
	// case payload size so an array of enums tiles correctly.
	d.enumStatus = EnumStatus::BodyWIP;
	enumFillStack_.push_back(idx);

	const auto *info = ctx_.getEnum(d.name);
	if (info == nullptr || !info->hasPayloadVariant) {
		d.enumStatus = EnumStatus::HaveBody;
		enumFillStack_.pop_back();
		return true;
	}

	uint64_t maxSize = 0;
	uint64_t maxAlign = 1;
	bool payloadFillFailed = false;
	for (const auto &v : info->variants) {
		if (payloadFillFailed) break;
		uint64_t off = 0;
		uint64_t varAlign = 1;
		for (TypeIdx t : v.payloadTypes) {
			// Trigger nested body fills for direct named-type
			// payloads so typeSize/typeAlign see a fully-laid-out
			// LLVM type. Pointer payloads (PtrSingle/PtrMany) skip
			// this — the pointee size doesn't depend on the
			// pointee's body, so they don't need (and shouldn't
			// trigger) the WIP cycle check.
			const TypeKey &tk = ctx_.getTypePool().get(t);
			if (tk.kind == TypeKind::Named || tk.kind == TypeKind::Struct ||
			    tk.kind == TypeKind::Enum || tk.kind == TypeKind::Union) {
				const std::string &payloadName =
				    ctx_.getStringPool().get(static_cast<StringIdx>(tk.a));
				bool nestedOk = true;
				if (ctx_.getStruct(payloadName)) {
					nestedOk = ensureStructBody(payloadName);
				} else if (ctx_.getEnum(payloadName)) {
					nestedOk = ensureEnumBody(payloadName);
				} else if (ctx_.getUnion(payloadName)) {
					nestedOk = ensureUnionBody(payloadName);
				}
				if (!nestedOk) {
					payloadFillFailed = true;
					break;
				}
			}
			uint64_t s = ctx_.typeSize(t);
			uint64_t a = ctx_.typeAlign(t);
			off = (off + a - 1) / a * a;
			off += s;
			if (a > varAlign) varAlign = a;
		}
		if (varAlign > 1) { off = (off + varAlign - 1) / varAlign * varAlign; }
		if (off > maxSize) maxSize = off;
		if (varAlign > maxAlign) maxAlign = varAlign;
	}
	if (payloadFillFailed) {
		// Stub body: {i8 tag, i8 payload}. Downstream queries see a
		// well-formed (if useless) LLVM type so the build doesn't
		// crash before the diagnostics get printed.
		JamTypeRef body[2] = {ctx_.getInt8Type(), ctx_.getInt8Type()};
		JamLLVMStructSetBody(info->type, body, 2, false);
		ctx_.setEnumLLVMType(d.name, info->type, 0, 1, true);
		d.enumStatus = EnumStatus::HaveBody;
		enumFillStack_.pop_back();
		return false;
	}

	JamTypeRef alignDriver;
	uint64_t alignDriverSize;
	switch (maxAlign) {
	case 1:
		alignDriver = ctx_.getInt8Type();
		alignDriverSize = 1;
		break;
	case 2:
		alignDriver = ctx_.getInt16Type();
		alignDriverSize = 2;
		break;
	case 4:
		alignDriver = ctx_.getInt32Type();
		alignDriverSize = 4;
		break;
	case 8:
		alignDriver = ctx_.getInt64Type();
		alignDriverSize = 8;
		break;
	default:
		ctx_.diagnostics().error(
		    locOf(d),
		    "enum `" + d.name +
		        "` requires alignment > 8, which is not yet supported");
		d.enumStatus = EnumStatus::HaveBody;
		enumFillStack_.pop_back();
		return false;
	}

	uint64_t paddedSize = (maxSize + maxAlign - 1) / maxAlign * maxAlign;
	uint64_t extraBytes =
	    paddedSize > alignDriverSize ? paddedSize - alignDriverSize : 0;

	std::vector<JamTypeRef> bodyTypes;
	bodyTypes.push_back(ctx_.getInt8Type());  // tag
	bodyTypes.push_back(alignDriver);
	if (extraBytes > 0) {
		bodyTypes.push_back(JamLLVMArrayType(
		    ctx_.getInt8Type(), static_cast<unsigned>(extraBytes)));
	}

	JamLLVMStructSetBody(info->type, bodyTypes.data(),
	                     static_cast<unsigned>(bodyTypes.size()), false);
	ctx_.setEnumLLVMType(d.name, info->type, maxSize, maxAlign, true);

	d.enumStatus = EnumStatus::HaveBody;
	enumFillStack_.pop_back();
	return true;
}

bool Analyzer::resolveTypeFieldsUnion(DeclIndex idx) {
	if (idx == kNoDecl) return false;
	Decl &d = decls_.get(idx);
	if (d.kind != DeclKind::Union) return false;

	switch (d.unionStatus) {
	case UnionStatus::HaveBody:
		return true;
	case UnionStatus::BodyWIP: {
		std::vector<Diagnostic::Trace> trace;
		for (DeclIndex anc : unionFillStack_) {
			if (anc == idx) continue;
			const Decl &ad = decls_.get(anc);
			Diagnostic::Trace t;
			t.kind = Diagnostic::Trace::Kind::Reference;
			t.decl = ad.name;
			t.loc = locOf(ad);
			trace.push_back(std::move(t));
		}
		if (trace.empty()) {
			ctx_.diagnostics().error(locOf(d), "union `" + d.name +
			                                       "` depends on itself");
		} else {
			ctx_.diagnostics().errorWithTrace(
			    locOf(d), "union `" + d.name + "` depends on itself",
			    std::move(trace));
		}
		return false;
	}
	case UnionStatus::None:
		break;
	}

	// Layout: { alignedField, [paddingBytes x i8] } where alignedField
	// is the field with the strictest alignment. Padding pads up to the
	// largest field's size so the union has the right size and
	// alignment for any stored variant.
	d.unionStatus = UnionStatus::BodyWIP;
	unionFillStack_.push_back(idx);

	const auto *info = ctx_.getUnion(d.name);
	if (info == nullptr || info->fields.empty()) {
		ctx_.diagnostics().error(
		    locOf(d), "union `" + d.name + "` must have at least one field");
		d.unionStatus = UnionStatus::HaveBody;
		unionFillStack_.pop_back();
		return false;
	}

	uint64_t maxSize = 0;
	uint64_t maxAlign = 1;
	std::size_t alignFieldIdx = 0;
	bool fieldFillFailed = false;
	for (std::size_t i = 0; i < info->fields.size(); ++i) {
		TypeIdx t = info->fields[i].second;
		const TypeKey &tk = ctx_.getTypePool().get(t);
		if (tk.kind == TypeKind::Named || tk.kind == TypeKind::Struct ||
		    tk.kind == TypeKind::Enum || tk.kind == TypeKind::Union) {
			const std::string &fieldName =
			    ctx_.getStringPool().get(static_cast<StringIdx>(tk.a));
			bool nestedOk = true;
			if (ctx_.getStruct(fieldName)) {
				nestedOk = ensureStructBody(fieldName);
			} else if (ctx_.getEnum(fieldName)) {
				nestedOk = ensureEnumBody(fieldName);
			} else if (ctx_.getUnion(fieldName)) {
				nestedOk = ensureUnionBody(fieldName);
			}
			if (!nestedOk) {
				// A downstream cycle or failure already pushed a
				// diagnostic. Calling typeSize/typeAlign on a still-
				// empty LLVM struct would crash, so bail with a
				// stub body for this union.
				fieldFillFailed = true;
				break;
			}
		}
		uint64_t s = ctx_.typeSize(t);
		uint64_t a = ctx_.typeAlign(t);
		if (s > maxSize) maxSize = s;
		if (a > maxAlign) {
			maxAlign = a;
			alignFieldIdx = i;
		}
	}
	if (fieldFillFailed) {
		// Emit a single-i8 body so downstream code that asks for the
		// union's size doesn't see undefined behavior. The diagnostic
		// already exists from the recursive call.
		JamTypeRef i8 = ctx_.getInt8Type();
		JamLLVMStructSetBody(info->type, &i8, 1, false);
		d.unionStatus = UnionStatus::HaveBody;
		unionFillStack_.pop_back();
		return false;
	}
	uint64_t allocSize = (maxSize + maxAlign - 1) / maxAlign * maxAlign;
	JamTypeRef alignedTy = ctx_.getLLVMType(info->fields[alignFieldIdx].second);
	uint64_t alignedSz = ctx_.typeSize(info->fields[alignFieldIdx].second);
	uint64_t paddingBytes = allocSize > alignedSz ? allocSize - alignedSz : 0;

	std::vector<JamTypeRef> bodyTypes;
	bodyTypes.push_back(alignedTy);
	if (paddingBytes > 0) {
		bodyTypes.push_back(JamLLVMArrayType(
		    ctx_.getInt8Type(), static_cast<unsigned>(paddingBytes)));
	}
	JamLLVMStructSetBody(info->type, bodyTypes.data(),
	                     static_cast<unsigned>(bodyTypes.size()), false);

	d.unionStatus = UnionStatus::HaveBody;
	unionFillStack_.pop_back();
	return true;
}

DeclValue Analyzer::analyzeDecl(DeclIndex idx) {
	Decl &d = decls_.get(idx);
	switch (d.kind) {
	case DeclKind::Function:
		return analyzeFunction(idx);
	case DeclKind::Struct:
		return analyzeStruct(idx);
	case DeclKind::Enum:
		return analyzeEnum(idx);
	case DeclKind::Union:
		return analyzeUnion(idx);
	case DeclKind::Const:
		return analyzeConst(idx);
	case DeclKind::TypeAlias:
		// Same shape as Const for now; the const-or-alias decision is
		// made inside analyzeConst by inspecting the InitExpr.
		return analyzeConst(idx);
	case DeclKind::Invalid:
		return {};
	}
	return {};
}

DeclValue Analyzer::analyzeStruct(DeclIndex idx) {
	Decl &d = decls_.get(idx);
	// Body fill is part of the struct's own analysis — calling
	// resolveTypeFieldsStruct here means ensureDeclAnalyzed(structIdx)
	// is now the single entry point that both proves the cycle-free
	// status AND materialises the LLVM body. If the body walk hits
	// FieldTypesWIP it already pushed an error; the Decl still ends
	// up Complete (with the placeholder Named TypeIdx) so callers
	// receive a usable type and the error appears once.
	resolveTypeFieldsStruct(idx);
	DeclValue v;
	v.kind = DeclValue::Kind::Type;
	v.type =
	    ctx_.getTypePool().internNamed(ctx_.getStringPool().intern(d.name));
	return v;
}

DeclValue Analyzer::analyzeEnum(DeclIndex idx) {
	resolveTypeFieldsEnum(idx);
	Decl &d = decls_.get(idx);
	DeclValue v;
	v.kind = DeclValue::Kind::Type;
	v.type =
	    ctx_.getTypePool().internNamed(ctx_.getStringPool().intern(d.name));
	return v;
}

DeclValue Analyzer::analyzeUnion(DeclIndex idx) {
	resolveTypeFieldsUnion(idx);
	Decl &d = decls_.get(idx);
	DeclValue v;
	v.kind = DeclValue::Kind::Type;
	v.type =
	    ctx_.getTypePool().internNamed(ctx_.getStringPool().intern(d.name));
	return v;
}

DeclValue Analyzer::analyzeFunction(DeclIndex idx) {
	Decl &d = decls_.get(idx);
	// Cache the ABI signature so downstream codegen + JIR + call sites
	// can ask one source for "how does this fn get lowered?" instead
	// of re-running classifyParam/classifyReturn at each site. Generic
	// functions skip the cache: each instantiation has its own
	// substituted TypeIdxs that arrive only at call time, so caching
	// at the decl would never hit.
	if (d.fnAst != nullptr && !d.fnAst->isGeneric() && !d.signature.computed) {
		d.signature.params.reserve(d.fnAst->Args.size());
		for (const auto &p : d.fnAst->Args) {
			d.signature.params.push_back(
			    jam::abi::classifyParam(p.Mode, p.Type, ctx_));
		}
		d.signature.returnAbi =
		    jam::abi::classifyReturn(d.fnAst->ReturnType, ctx_);
		d.signature.computed = true;
	}
	DeclValue v;
	v.kind = DeclValue::Kind::Function;
	v.function = d.fnAst;
	return v;
}

DeclValue Analyzer::analyzeConst(DeclIndex idx) {
	Decl &d = decls_.get(idx);
	// `AliasedType` is set by `registerConsts` in main.cpp when it
	// detects that the const's RHS is a type-resolving expression
	// (e.g. `const Box = Vec(i32)` → AliasedType = GenericCall ty).
	// Reading that field here lets the analyzer surface the alias
	// as a Type-kind DeclValue, so future call sites that ask the
	// analyzer "what is X?" get a TypeIdx back instead of None.
	//
	// Consts that stay as values (e.g. `const PI = 3.14`) keep
	// AliasedType == kNoType — the analyzer returns a None-valued
	// DeclValue and the existing JamCodegenContext::registerModuleConst
	// path handles the runtime value.
	if (d.constAst != nullptr && d.constAst->AliasedType != kNoType) {
		DeclValue v;
		v.kind = DeclValue::Kind::Type;
		v.type = d.constAst->AliasedType;
		return v;
	}
	return {};
}

}  // namespace jam
