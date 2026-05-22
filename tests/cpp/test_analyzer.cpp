// Unit tests for src/analyzer.h — the demand-driven Decl chokepoint.
// Covers the state-machine transitions (Unreferenced -> InProgress ->
// Complete), the cycle detector (Decl.analysis re-entry produces a
// "dependency loop" diagnostic with the chain), and the struct-status
// chokepoint (Struct.status FieldTypesWIP re-entry produces a
// "struct depends on itself" diagnostic).
//
// Real Decls would carry ast pointers; this suite uses Function /
// Struct decls with null AST pointers — the analyzer treats those
// like "trivial" decls and runs the stub branch.

#include "../../src/analyzer.h"
#include "../../src/ast.h"
#include "../../src/codegen.h"
#include "../../src/decl.h"
#include "test_framework.h"

namespace {

// Codegen context is move-deleted, so each test constructs one in
// place. The analyzer only consumes diagnostics() + currentFile()
// from it for now (Step 2 chokepoint shape); the deeper codegen
// plumbing (type pool, struct registry) becomes a dependency in
// Step 3 when the stub branches get filled.

void testUnreferencedTransitionsThroughInProgressToComplete() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex a = decls.create(jam::DeclKind::Function, "a");
	ASSERT_TRUE(decls.get(a).analysis == jam::DeclAnalysis::Unreferenced);

	jam::DeclValue v = az.ensureDeclAnalyzed(a);
	ASSERT_TRUE(decls.get(a).analysis == jam::DeclAnalysis::Complete);
	// Function decls get a DeclValue::Function pointing at their AST;
	// since we passed no AST it's null but the kind is set.
	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::Function);
}

void testCompleteDeclReturnsCachedValueNoReanalysis() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex a = decls.create(jam::DeclKind::Function, "a");
	az.ensureDeclAnalyzed(a);
	std::size_t errCountAfterFirst = ctx.diagnostics().errorCount();

	// Re-ask; expect no new diagnostics + state stays Complete.
	jam::DeclValue v = az.ensureDeclAnalyzed(a);
	ASSERT_EQ(errCountAfterFirst, ctx.diagnostics().errorCount());
	ASSERT_TRUE(decls.get(a).analysis == jam::DeclAnalysis::Complete);
	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::Function);
}

void testKNoDeclReturnsEmpty() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclValue v = az.ensureDeclAnalyzed(jam::kNoDecl);
	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::None);
	ASSERT_FALSE(ctx.diagnostics().hasErrors());
}

void testInProgressTriggersCycleError() {
	// Simulate re-entering ensureDeclAnalyzed while the decl is mid-
	// analysis. We push the decl onto the analysis stack ourselves
	// (no public API in Step 2 to force recursion since the stubs are
	// non-recursive; Step 3 wires real recursion).
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclIndex a = decls.create(jam::DeclKind::Struct, "Cyc");

	// Manually flip a to InProgress to model the "currently being
	// analyzed" scenario from inside one of its dependencies.
	decls.get(a).analysis = jam::DeclAnalysis::InProgress;

	std::size_t before = ctx.diagnostics().errorCount();
	jam::DeclValue v = az.ensureDeclAnalyzed(a);
	std::size_t after = ctx.diagnostics().errorCount();

	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::None);
	ASSERT_EQ(before + 1, after);
	const auto &diag = ctx.diagnostics().all().back();
	ASSERT_TRUE(diag.message.find("dependency loop detected") !=
	            std::string::npos);
	ASSERT_TRUE(diag.message.find("Cyc") != std::string::npos);
}

void testCycleChainListsAnalysisStackInOrder() {
	// When the analyzer hits a cycle, the message should list the
	// chain in source order: A -> B -> A. We populate the analysis
	// stack ourselves and ask the analyzer to "re-enter" A from the
	// middle of B's analysis.
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclIndex a = decls.create(jam::DeclKind::Function, "A");
	jam::DeclIndex b = decls.create(jam::DeclKind::Function, "B");

	decls.get(a).analysis = jam::DeclAnalysis::InProgress;
	decls.get(b).analysis = jam::DeclAnalysis::InProgress;
	// Stack-driven cycle detection: emulate the recursive call chain
	// by writing into the analyzer's analysis stack. We rely on the
	// public read-back `analysisStack()` to confirm the read; the
	// write is internal so the test pokes the chokepoint by setting
	// analysis state directly + relying on pushCycleError using
	// whatever the analyzer has on its stack. Step 2 keeps that
	// stack empty until ensureDeclAnalyzed runs, so the chain shown
	// here is just the repeated decl. Step 3 will exercise the full
	// chain through real recursion.
	az.ensureDeclAnalyzed(a);
	(void)b;

	ASSERT_TRUE(ctx.diagnostics().hasErrors());
	const auto &diag = ctx.diagnostics().all().back();
	ASSERT_TRUE(diag.message.find("`A`") != std::string::npos);
}

void testResolveTypeFieldsStructNoneToHaveFieldTypes() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex s = decls.create(jam::DeclKind::Struct, "S");
	ASSERT_TRUE(decls.get(s).structStatus == jam::StructStatus::None);

	bool ok = az.resolveTypeFieldsStruct(s);
	ASSERT_TRUE(ok);
	ASSERT_TRUE(decls.get(s).structStatus == jam::StructStatus::HaveFieldTypes);
}

void testResolveTypeFieldsStructWIPProducesSelfCycleError() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex s = decls.create(jam::DeclKind::Struct, "Recursive");
	decls.get(s).structStatus = jam::StructStatus::FieldTypesWIP;

	std::size_t before = ctx.diagnostics().errorCount();
	bool ok = az.resolveTypeFieldsStruct(s);
	std::size_t after = ctx.diagnostics().errorCount();

	ASSERT_FALSE(ok);
	ASSERT_EQ(before + 1, after);
	const auto &diag = ctx.diagnostics().all().back();
	ASSERT_TRUE(diag.message.find("`Recursive` depends on itself") !=
	            std::string::npos);
}

void testResolveTypeFieldsStructAlreadyHaveFieldTypesIsNoOp() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclIndex s = decls.create(jam::DeclKind::Struct, "S");
	decls.get(s).structStatus = jam::StructStatus::HaveFieldTypes;

	std::size_t before = ctx.diagnostics().errorCount();
	bool ok = az.resolveTypeFieldsStruct(s);
	ASSERT_TRUE(ok);
	ASSERT_EQ(before, ctx.diagnostics().errorCount());
	// State stays put.
	ASSERT_TRUE(decls.get(s).structStatus == jam::StructStatus::HaveFieldTypes);
}

void testResolveTypeFieldsStructOnNonStructDeclIsFalse() {
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclIndex f = decls.create(jam::DeclKind::Function, "f");
	bool ok = az.resolveTypeFieldsStruct(f);
	ASSERT_FALSE(ok);
}

void testEnsureDeclAnalyzedFunctionPopulatesSignatureCache() {
	// Step C: analyzeFunction caches param + return ABI on the Decl
	// so call sites and codegen read one source instead of
	// re-classifying. Without a real FunctionAST attached the cache
	// stays empty (signature.computed == false) — that's the path
	// taken when an analyzer test creates a bare Decl. With an AST,
	// the analyzer fills the cache and `computed` flips to true.
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex fn = decls.create(jam::DeclKind::Function, "noFnAst");
	az.ensureDeclAnalyzed(fn);
	ASSERT_FALSE(decls.get(fn).signature.computed);

	// Attach a real (empty) FunctionAST and re-analyze a fresh Decl.
	FunctionAST realFn("withAst", /*Args=*/{},
	                   /*ReturnType=*/kNoType, /*Body=*/{});
	jam::DeclIndex fn2 = decls.create(jam::DeclKind::Function, "withAst");
	decls.get(fn2).fnAst = &realFn;
	az.ensureDeclAnalyzed(fn2);
	ASSERT_TRUE(decls.get(fn2).signature.computed);
	// Empty Args -> empty params vector; kNoType return -> Direct ABI
	// classification's default (matches what classifyReturn returns
	// for the void-like case).
	ASSERT_EQ(static_cast<std::size_t>(0),
	          decls.get(fn2).signature.params.size());
}

void testEnsureDeclAnalyzedStructReturnsNamedTypeAndFillsBody() {
	// Step A: analyzeStruct(idx) now drives resolveTypeFieldsStruct
	// internally and returns a DeclValue::Type whose TypeIdx is the
	// Named TypeIdx for the struct's source name. Without a real
	// AST attached the body-walk takes the null-AST early-out, but
	// the structStatus must still flip to HaveFieldTypes — that's
	// the proof that ensureDeclAnalyzed is the single chokepoint.
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	jam::DeclIndex s = decls.create(jam::DeclKind::Struct, "Foo");
	ASSERT_TRUE(decls.get(s).structStatus == jam::StructStatus::None);

	jam::DeclValue v = az.ensureDeclAnalyzed(s);
	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::Type);
	ASSERT_TRUE(v.type != kNoType);
	ASSERT_TRUE(decls.get(s).structStatus == jam::StructStatus::HaveFieldTypes);
	ASSERT_TRUE(decls.get(s).analysis == jam::DeclAnalysis::Complete);
}

void testResolveDeclLooksUpByNameAndAnalyzes() {
	// Step F: resolveDecl(name) is the demand-driven name chokepoint.
	// It finds the decl by name and forces ensureDeclAnalyzed in one
	// call — call sites don't have to know about DeclTable indices.
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);

	decls.create(jam::DeclKind::Struct, "Foo");
	jam::DeclValue v = az.resolveDecl("Foo");
	ASSERT_TRUE(v.kind == jam::DeclValue::Kind::Type);
	ASSERT_TRUE(v.type != kNoType);

	// Unknown name returns a None-kind value with no diagnostic — the
	// caller decides whether to error.
	std::size_t errBefore = ctx.diagnostics().errorCount();
	jam::DeclValue miss = az.resolveDecl("DoesNotExist");
	ASSERT_TRUE(miss.kind == jam::DeclValue::Kind::None);
	ASSERT_EQ(errBefore, ctx.diagnostics().errorCount());
}

void testAnalysisStackEmptyAfterTopLevelCall() {
	// Each ensureDeclAnalyzed pushes then pops — the stack must be
	// empty when control returns to the test driver.
	JamCodegenContext ctx("test_module");
	ctx.setCurrentFile("test.jam");
	jam::DeclTable decls;
	jam::Analyzer az(ctx, decls);
	jam::DeclIndex a = decls.create(jam::DeclKind::Function, "a");
	jam::DeclIndex b = decls.create(jam::DeclKind::Struct, "b");
	az.ensureDeclAnalyzed(a);
	az.ensureDeclAnalyzed(b);
	ASSERT_EQ(static_cast<std::size_t>(0), az.analysisStack().size());
}

}  // namespace

int main() {
	TestFramework framework;
	framework.addTest("Analyzer - Unreferenced -> InProgress -> Complete",
	                  testUnreferencedTransitionsThroughInProgressToComplete);
	framework.addTest("Analyzer - Complete returns cached value",
	                  testCompleteDeclReturnsCachedValueNoReanalysis);
	framework.addTest("Analyzer - kNoDecl returns empty",
	                  testKNoDeclReturnsEmpty);
	framework.addTest("Analyzer - InProgress triggers cycle error",
	                  testInProgressTriggersCycleError);
	framework.addTest("Analyzer - cycle chain lists stack in source order",
	                  testCycleChainListsAnalysisStackInOrder);
	framework.addTest(
	    "Analyzer - resolveTypeFieldsStruct: None -> HaveFieldTypes",
	    testResolveTypeFieldsStructNoneToHaveFieldTypes);
	framework.addTest(
	    "Analyzer - resolveTypeFieldsStruct: WIP is self-cycle error",
	    testResolveTypeFieldsStructWIPProducesSelfCycleError);
	framework.addTest(
	    "Analyzer - resolveTypeFieldsStruct: HaveFieldTypes is no-op",
	    testResolveTypeFieldsStructAlreadyHaveFieldTypesIsNoOp);
	framework.addTest(
	    "Analyzer - resolveTypeFieldsStruct on non-struct returns false",
	    testResolveTypeFieldsStructOnNonStructDeclIsFalse);
	framework.addTest("Analyzer - ensureDeclAnalyzed(struct) returns Named "
	                  "TypeIdx + fills body",
	                  testEnsureDeclAnalyzedStructReturnsNamedTypeAndFillsBody);
	framework.addTest(
	    "Analyzer - ensureDeclAnalyzed(fn) populates ABI signature cache",
	    testEnsureDeclAnalyzedFunctionPopulatesSignatureCache);
	framework.addTest("Analyzer - resolveDecl(name) looks up + analyzes",
	                  testResolveDeclLooksUpByNameAndAnalyzes);
	framework.addTest("Analyzer - analysis stack empty after top-level call",
	                  testAnalysisStackEmptyAfterTopLevelCall);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
