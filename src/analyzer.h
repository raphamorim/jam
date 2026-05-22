/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_ANALYZER_H
#define JAM_ANALYZER_H

#include "decl.h"
#include "diagnostics.h"

#include <vector>

class JamCodegenContext;

namespace jam {

// Demand-driven decl analysis. Replaces the explicit phase pipeline
// that used to live in main.cpp::compileAndRun. Two state machines
// stack on top of the DeclTable:
//
//   * `Decl.analysis` (Unreferenced/InProgress/Complete/...) — the
//     chokepoint at `ensureDeclAnalyzed`. A decl whose value is
//     currently being computed is `InProgress`; re-entering it from
//     anywhere is a dependency loop.
//
//   * `Struct.status` (None/FieldTypesWIP/HaveFieldTypes/...) — a
//     separate lifecycle for *struct bodies*. A struct Decl can be
//     `Complete` (value = a NamedType TypeIdx) long before its
//     fields are materialized; `resolveTypeFieldsStruct` is what
//     forces the field types when something — `getLLVMType`,
//     `@sizeOf`, a field access — actually needs them.
//
// The split lets `struct Container { x: Vec(i32) }` work even when
// `Vec`'s Decl hasn't been analysed yet: Container is `Complete` as
// soon as its NamedType is interned; the demand for `x.ty` later
// pulls in `Vec` recursively without any fixed phase ordering.
class Analyzer {
  public:
	Analyzer(JamCodegenContext &ctx, DeclTable &decls);

	// Ensure `idx`'s value has been computed. Returns the value (a
	// type, function, etc.); on cycle / failure returns a None-valued
	// DeclValue and the Decl's analysis state reflects the failure.
	// Diagnostics are pushed to `ctx.diagnostics()`.
	//
	// Idempotent: a Complete decl returns the cached value with no
	// work. An InProgress decl triggers the cycle-detector.
	DeclValue ensureDeclAnalyzed(DeclIndex idx);

	// Force struct-field types to be materialized. Distinct from
	// ensureDeclAnalyzed: a struct's Decl can be Complete (its value
	// is the NamedType) while its body is still uncomputed. Returns
	// false on `FieldTypesWIP` re-entry (struct depends on itself)
	// or any field-eval failure.
	bool resolveTypeFieldsStruct(DeclIndex idx);

	// Same shape as resolveTypeFieldsStruct, but for enums and
	// unions. Computes the LLVM body (variant layout / tagged-union
	// padding) and stamps EnumStatus::HaveBody / UnionStatus::HaveBody
	// when done. Re-entry into a BodyWIP decl is a self-cycle.
	bool resolveTypeFieldsEnum(DeclIndex idx);
	bool resolveTypeFieldsUnion(DeclIndex idx);

	// Look up a struct/enum/union by source-level name and route the
	// body-fill through the matching `resolveTypeFields*`. Used by
	// JamCodegenContext::getLLVMType so any code path that asks for
	// an LLVM type whose body hasn't been filled triggers demand-
	// driven materialisation + cycle detection. Returns true if the
	// body is set after the call.
	bool ensureStructBody(const std::string &name);
	bool ensureEnumBody(const std::string &name);
	bool ensureUnionBody(const std::string &name);

	// Demand-driven name lookup: find the decl, force its analysis,
	// and return the resolved DeclValue. Returns a None-kind value if
	// the name doesn't exist in the DeclTable. The eager Step E loop
	// pre-analyzes all type-shaped decls, so today this is mostly a
	// chokepoint — but any future flip to lazy decl analysis (e.g.
	// per-module or incremental rebuilds) keeps astgen callers
	// working without ordering assumptions.
	DeclValue resolveDecl(const std::string &name);

	// Public read of the current analysis stack — used by Step 5's
	// reference-trace formatter and by tests.
	const std::vector<DeclIndex> &analysisStack() const {
		return analysisStack_;
	}

  private:
	// Dispatch on Decl::kind. Each branch writes the resolved
	// DeclValue back into the Decl and returns it. Errors are pushed
	// to diagnostics; the branch returns a None-valued DeclValue and
	// the Decl's analysis state is set to AnalysisFailure /
	// DependencyFailure. Branches receive the DeclIndex (not just
	// Decl&) so they can call back into ensureDeclAnalyzed for
	// dependencies and into resolveTypeFieldsStruct without a
	// secondary name lookup.
	DeclValue analyzeDecl(DeclIndex idx);
	DeclValue analyzeStruct(DeclIndex idx);
	DeclValue analyzeEnum(DeclIndex idx);
	DeclValue analyzeUnion(DeclIndex idx);
	DeclValue analyzeFunction(DeclIndex idx);
	DeclValue analyzeConst(DeclIndex idx);

	// Push a "dependency loop detected: A -> B -> A" diagnostic, using
	// the current analysis stack as the chain.
	void pushCycleError(DeclIndex repeated);

	// Build a reference trace from the analysis stack — every ancestor
	// of the current decl, oldest-first ("referenced by `outer`
	// referenced by `middle` …"). `excludeTop` skips the topmost stack
	// entry when it would just restate the diagnostic's own location
	// (e.g. a self-cycle error already names the struct in its message).
	std::vector<Diagnostic::Trace>
	buildReferenceTrace(bool excludeTop = false) const;

	// Push an error with the reference trace pre-attached.
	void emitErrorWithRefTrace(SrcLoc loc, std::string message,
	                           bool excludeTop = false);

	// SrcLoc for the Decl, used by diagnostic helpers.
	SrcLoc locOf(const Decl &d) const;

	JamCodegenContext &ctx_;
	DeclTable &decls_;
	// Stack of currently-analyzing Decl indices. Pushed on entry to
	// ensureDeclAnalyzed (when not yet Complete), popped on exit. A
	// decl already on the stack when asked again is a cycle.
	std::vector<DeclIndex> analysisStack_;
	// Parallel stack for struct-body fills. resolveTypeFieldsStruct
	// has its own lifecycle (StructStatus::FieldTypesWIP) that
	// doesn't go through ensureDeclAnalyzed, so analysisStack_ is
	// empty for struct-field cycles. This stack captures the
	// "ancestor structs whose field walks led to here" so the cycle
	// error can show a chain like `B referenced by A`.
	std::vector<DeclIndex> structFillStack_;
	std::vector<DeclIndex> enumFillStack_;
	std::vector<DeclIndex> unionFillStack_;
};

}  // namespace jam

#endif  // JAM_ANALYZER_H
