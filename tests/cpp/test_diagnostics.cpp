// Diagnostic-pipeline tests.
//
// Covers the structured-diagnostic system introduced to mirror Zig's
// AstGen errors (`file:line: error: message` format with secondary
// notes and reference traces). Tests run jam.out as a subprocess and
// assert on its stderr; this keeps the suite stable across pure-IR
// refactors and matches the existing test_codegen_errors.cpp shape.

#include "test_framework.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace {

struct CompileResult {
	int exitCode;
	std::string stderr_;
};

CompileResult compileSource(const std::string &name,
                            const std::string &source) {
	std::string path = "/tmp/" + name + ".jam";
	{
		std::ofstream out(path);
		out << source;
	}
	std::string cmd = "./jam.out " + path + " 2>&1";

	std::string output;
	FILE *pipe = popen(cmd.c_str(), "r");
	if (!pipe) { throw std::runtime_error("popen failed: " + cmd); }
	char buf[256];
	while (fgets(buf, sizeof(buf), pipe) != nullptr) output += buf;
	int status = pclose(pipe);

	int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	return {exitCode, std::move(output)};
}

bool stderrContains(const CompileResult &r, const std::string &substr) {
	return r.stderr_.find(substr) != std::string::npos;
}

// Count occurrences of a substring — used by the multi-error tests
// to assert that EVERY error in a batch is reported (not just the
// first one).
std::size_t countOccurrences(const std::string &hay,
                             const std::string &needle) {
	std::size_t n = 0;
	for (std::size_t pos = 0;
	     (pos = hay.find(needle, pos)) != std::string::npos; ++pos) {
		++n;
	}
	return n;
}

// ── Line-number tests ──────────────────────────────────────────

void testUnknownFunctionHasLine() {
	auto r = compileSource("diag_unknown_fn", "fn main() i32 {\n"
	                                          "    foobar();\n"
	                                          "    return 0;\n"
	                                          "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, "unknown function `foobar`"));
}

void testUnknownVariableHasLine() {
	auto r = compileSource("diag_unknown_var", "fn main() i32 {\n"
	                                           "    return badName;\n"
	                                           "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, "unknown variable `badName`"));
}

void testBreakOutsideLoopHasLine() {
	auto r = compileSource("diag_break_loop", "fn main() i32 {\n"
	                                          "    break;\n"
	                                          "    return 0;\n"
	                                          "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, "`break` outside of loop"));
}

void testContinueOutsideLoopHasLine() {
	auto r = compileSource("diag_continue_loop", "fn main() i32 {\n"
	                                             "    continue;\n"
	                                             "    return 0;\n"
	                                             "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, "`continue` outside of loop"));
}

void testUnknownFieldHasLine() {
	auto r = compileSource("diag_unknown_field",
	                       "const Point = struct { x: i32, y: i32 };\n"
	                       "fn main() i32 {\n"
	                       "    var p: Point = Point { x: 1, y: 2 };\n"
	                       "    return p.zz;\n"
	                       "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":4: error:"));
}

// ── Init-analysis flows through the same channel ───────────────

void testUseAfterMoveHasLine() {
	auto r = compileSource("diag_use_after_move",
	                       "fn consume(x: move i32) i32 { return x; }\n"
	                       "fn main() i32 {\n"
	                       "    var x: i32 = 1;\n"
	                       "    var a: i32 = consume(x);\n"
	                       "    var b: i32 = consume(x);\n"
	                       "    return a + b;\n"
	                       "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":5: error:"));
	ASSERT_TRUE(stderrContains(r, "uninitialized binding"));
}

// ── Multi-error reporting ───────────────────────────────────────

void testMultipleErrorsAllReported() {
	auto r = compileSource("diag_multi", "fn first() i32 { return foo(); }\n"
	                                     "fn second() i32 { return bar(); }\n"
	                                     "fn third() i32 { break; return 0; }\n"
	                                     "fn main() i32 { return 0; }\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Each of the three error lines should appear once.
	ASSERT_TRUE(stderrContains(r, ":1: error:"));
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, ":3: error:"));
	// At least 3 distinct "error:" entries (more allowed in case
	// init-analysis flags a downstream issue too).
	ASSERT_TRUE(countOccurrences(r.stderr_, "error:") >= 3);
}

void testMultiErrorsSortedByLine() {
	auto r =
	    compileSource("diag_multi_sorted", "fn a() i32 { return zzz(); }\n"
	                                       "fn b() i32 { return aaa(); }\n"
	                                       "fn main() i32 { return 0; }\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Line 1's error must appear before line 2's, regardless of
	// function name alphabet — Diagnostics::emit sorts by location.
	auto p1 = r.stderr_.find(":1: error:");
	auto p2 = r.stderr_.find(":2: error:");
	ASSERT_TRUE(p1 != std::string::npos && p2 != std::string::npos);
	ASSERT_TRUE(p1 < p2);
}

// ── Reference trace for generic instantiation ──────────────────

void testGenericInstantiationCarriesRefTrace() {
	auto r =
	    compileSource("diag_ref_trace", "fn Box(T: type) type {\n"
	                                    "    return struct {\n"
	                                    "        val: T,\n"
	                                    "        fn pickBad(self: Self) i32 {\n"
	                                    "            return self.notAField;\n"
	                                    "        }\n"
	                                    "    };\n"
	                                    "}\n"
	                                    "fn main() i32 {\n"
	                                    "    var b: Box(i32) = Box(i32) { val: 7 };\n"
	                                    "    return b.pickBad();\n"
	                                    "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Underlying error inside the instantiated body — line 5 in
	// source. Reference trace adds "in instantiation of
	// `Box__i32.pickBad`".
	ASSERT_TRUE(stderrContains(r, ":5: error:"));
	ASSERT_TRUE(stderrContains(r, "in instantiation of `Box__i32.pickBad`"));
}

// ── Per-decl recovery (one broken decl doesn't suppress others) ──

void testBrokenDeclDoesNotMaskNextDecl() {
	auto r = compileSource(
	    "diag_recovery", "fn broken() i32 { return missingFn(); }\n"
	                     "fn alsoBroken() i32 { return anotherMissingFn(); }\n"
	                     "fn main() i32 { return 0; }\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Both decls must produce an error (a single throw would have
	// reported only the first).
	ASSERT_TRUE(stderrContains(r, "missingFn"));
	ASSERT_TRUE(stderrContains(r, "anotherMissingFn"));
}

// ── Within-decl multi-error (Poison-based recovery) ──────────────

// Three unknown-function calls in the same function should each
// produce a distinct diagnostic. Without Poison-based recovery the
// first failNode would bail the whole decl and we'd see only one
// error — Zig's recoverable-error model lets us report them all.
void testRecoveryWithinSingleDecl() {
	auto r = compileSource("diag_recover_within", "fn main() i32 {\n"
	                                              "    var a: i32 = foo();\n"
	                                              "    var b: i32 = bar();\n"
	                                              "    var c: i32 = baz();\n"
	                                              "    return a + b + c;\n"
	                                              "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, ":3: error:"));
	ASSERT_TRUE(stderrContains(r, ":4: error:"));
	ASSERT_TRUE(stderrContains(r, "`foo`"));
	ASSERT_TRUE(stderrContains(r, "`bar`"));
	ASSERT_TRUE(stderrContains(r, "`baz`"));
}

void testRecoveryAcrossDifferentErrorClasses() {
	auto r =
	    compileSource("diag_recover_mixed", "fn main() i32 {\n"
	                                        "    var p: i32 = unknownThing;\n"
	                                        "    var q: i32 = alsoMissing();\n"
	                                        "    return p + q;\n"
	                                        "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2: error:"));
	ASSERT_TRUE(stderrContains(r, ":3: error:"));
	ASSERT_TRUE(stderrContains(r, "unknown variable `unknownThing`"));
	ASSERT_TRUE(stderrContains(r, "unknown function `alsoMissing`"));
}

void testUnknownMethodIsRecoverable() {
	auto r = compileSource("diag_recover_method",
	                       "const Point = struct { x: i32 };\n"
	                       "fn other() i32 { return 99; }\n"
	                       "fn main() i32 {\n"
	                       "    var p: Point = Point { x: 1 };\n"
	                       "    var a: i32 = p.bogusMethod();\n"
	                       "    var b: i32 = p.alsoBogus();\n"
	                       "    return a + b;\n"
	                       "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Both unknown-method calls in the same function should report.
	ASSERT_TRUE(stderrContains(r, "bogusMethod"));
	ASSERT_TRUE(stderrContains(r, "alsoBogus"));
}

// ── Parser errors now carry :line: too ─────────────────────────

void testParserErrorHasLine() {
	auto r = compileSource("diag_parser_line", "fn main() i32 {\n"
	                                           "    return *;\n"
	                                           "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2:"));
}

void testRedeclarationInSameScopeRejected() {
	auto r = compileSource("diag_redecl_same_scope", "fn main() {\n"
	                                                 "    const a = true;\n"
	                                                 "    var a = true;\n"
	                                                 "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":3:"));
	ASSERT_TRUE(stderrContains(r, "redeclaration of `a`"));
}

void testRedeclarationAcrossSiblingScopesAllowed() {
	auto r =
	    compileSource("diag_redecl_siblings", "fn main(x: i32) i32 {\n"
	                                          "    if (x == 0) {\n"
	                                          "        const op: u32 = 1;\n"
	                                          "        return op as i32;\n"
	                                          "    }\n"
	                                          "    if (x == 1) {\n"
	                                          "        const op: u32 = 2;\n"
	                                          "        return op as i32;\n"
	                                          "    }\n"
	                                          "    return 99;\n"
	                                          "}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testTypeMismatchBoolEqualsFloatRejected() {
	auto r = compileSource("diag_bool_eq_float", "fn main() {\n"
	                                             "    const a: bool = 1.0;\n"
	                                             "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2:"));
	ASSERT_TRUE(stderrContains(r, "type mismatch in `a`"));
}

void testTypeInferenceAllocatesCorrectWidth() {
	auto r = compileSource("diag_var_infer", "fn main() i32 {\n"
	                                         "    var b = true;\n"
	                                         "    var f = 3.14;\n"
	                                         "    return 0;\n"
	                                         "}\n");
	// Both inferred-type bindings should compile cleanly; before
	// the fix they silently allocated a 1-byte slot regardless of
	// init type. We can't introspect alloca widths from stderr, so
	// just assert the program built — IR-level coverage lives in the
	// hand-test `cat | --emit-ir` flow.
	ASSERT_TRUE(r.exitCode == 0);
}

void testSelfReferentialStructIsCaught() {
	// A struct whose own field is the same struct (without a pointer
	// or slice indirection) is infinite-sized. The analyzer's
	// `resolveTypeFieldsStruct` flips the struct's status to
	// `FieldTypesWIP` before walking fields; re-entering during the
	// field walk produces a "struct ... depends on itself" error.
	auto r = compileSource("diag_selfref_struct", "const S = struct { x: S };\n"
	                                              "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "`S` depends on itself"));
}

void testSelfReferentialStructViaPointerIsAllowed() {
	// Indirect self-reference (through a pointer) is fine — the
	// struct's size doesn't depend on its own size. This shape is
	// the standard linked-list / tree node idiom.
	auto r = compileSource("diag_selfref_struct_ptr",
	                       "const Node = struct { next: *mut Node };\n"
	                       "fn main() {}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testMultiStepStructCycleShowsReferenceTrace() {
	// A two-step cycle: `A { b: B }`, `B { a: A }`. The closing
	// error names `A`; the trace shows `B` as the intermediate
	// reference. The user reading just the message wouldn't know
	// the loop closed via B — the trace is what makes the cycle
	// shape legible.
	auto r = compileSource("diag_multistep_struct_cycle",
	                       "const A = struct { b: B };\n"
	                       "const B = struct { a: A };\n"
	                       "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "`A` depends on itself"));
	ASSERT_TRUE(stderrContains(r, "referenced by `B`"));
}

void testSelfReferentialEnumIsCaught() {
	// A payloaded enum variant that carries the enum itself is
	// infinite-sized — the analyzer's resolveTypeFieldsEnum walks
	// payload TypeIdxs and, when it sees a direct enum/struct/union
	// payload, triggers ensureEnumBody/Struct/Union which trips the
	// matching BodyWIP / FieldTypesWIP cycle key.
	auto r = compileSource("diag_selfref_enum",
	                       "const E = enum { Nothing, Just(E) };\n"
	                       "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "`E` depends on itself"));
}

void testSelfReferentialUnionIsCaught() {
	auto r = compileSource("diag_selfref_union",
	                       "const U = union { a: U, b: i32 };\n"
	                       "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "`U` depends on itself"));
}

void testStructViaUnionCycleIsCaught() {
	// Mixed struct→union→struct chain. The cycle key here is on the
	// inner struct's FieldTypesWIP — the union's body fill triggers
	// ensureStructBody, which re-enters S while it's still WIP.
	auto r = compileSource("diag_struct_via_union",
	                       "const U = union { s: S, b: i32 };\n"
	                       "const S = struct { u: U };\n"
	                       "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "depends on itself"));
}

void testThreeStepStructCycleShowsFullChain() {
	// `A { b: B }`, `B { c: C }`, `C { a: A }`. The chain in the
	// trace is the full ancestor list — both `B` and `C` show up.
	auto r =
	    compileSource("diag_three_step_cycle", "const A = struct { b: B };\n"
	                                           "const B = struct { c: C };\n"
	                                           "const C = struct { a: A };\n"
	                                           "fn main() {}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, "`A` depends on itself"));
	ASSERT_TRUE(stderrContains(r, "referenced by `B`"));
	ASSERT_TRUE(stderrContains(r, "referenced by `C`"));
}

void testChainedMethodCallOnLocal() {
	// `t.field.method(args)` — multi-segment receiver where the root
	// is a local. Pre-fix the parser collapsed the chain into the
	// qualified name `t.field.method`, then the codegen looked for a
	// function with that name and errored "unknown module handle `t`".
	// The fix: parser emits the indirect-call form when the chain
	// root isn't a known import handle and the chain has 2+ dots.
	// Inner-with-methods would trip the top-level-struct method
	// restriction (only drop/default allowed); we use two generic
	// wrappers so methods are permitted on both layers.
	auto r = compileSource(
	    "diag_chained_call_local",
	    "fn Inner(T: type) type {\n"
	    "    return struct {\n"
	    "        val: T,\n"
	    "        fn get(self: Self) T { return self.val; }\n"
	    "    };\n"
	    "}\n"
	    "fn Wrap(T: type) type {\n"
	    "    return struct {\n"
	    "        inner: Inner(T),\n"
	    "        fn make(v: T) Self {\n"
	    "            return Self { inner: Inner(T) { val: v } };\n"
	    "        }\n"
	    "    };\n"
	    "}\n"
	    "fn main() i32 {\n"
	    "    var w: Wrap(i32) = Wrap(i32).make(7);\n"
	    "    return w.inner.get();\n"
	    "}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testTwoDotStaticCallStillWorks() {
	// `Random.default()` style — single dot, type-static dispatch.
	// Must keep working (regression guard for the parser fix).
	auto r = compileSource("diag_static_call_one_dot",
	                       "const Random = struct {\n"
	                       "    n: i32,\n"
	                       "    fn default() Self { return Self { n: 0 }; }\n"
	                       "};\n"
	                       "fn main() i32 {\n"
	                       "    var r: Random = Random.default();\n"
	                       "    return r.n;\n"
	                       "}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testGenericMethodChainedThroughLocalField() {
	// The case from the user's bug report: a struct has a field of
	// generic-instantiated type, and an instance method is invoked
	// through `local.field.method(args)`. The chain has two dots and
	// `local` is in the locals table — must take the indirect-call
	// path.
	auto r = compileSource(
	    "diag_generic_via_field",
	    "const {Vec} = import(\"collections\");\n"
	    "const Random = struct {\n"
	    "    inner: Vec(i32),\n"
	    "    fn default() Self { return Self { inner: Vec(i32).empty() }; }\n"
	    "};\n"
	    "fn main() i32 {\n"
	    "    var t = Random.default();\n"
	    "    t.inner.push(1);\n"
	    "    t.inner.push(2);\n"
	    "    return 0;\n"
	    "}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testFunctionAndStructMayShareName() {
	// `fn Counter(T: type) type` and `const Counter = struct {...}`
	// can coexist — the function lives in the function namespace,
	// the struct in the type namespace. The analyzer's kind-aware
	// finder lands on the right Decl depending on the lookup
	// context, so the struct's body still gets materialised even
	// when the same-name function was registered first.
	auto r = compileSource(
	    "diag_fn_struct_same_name",
	    "fn Counter(T: type) type { return struct { value: T }; }\n"
	    "const Counter = struct {\n"
	    "    n: i32,\n"
	    "    fn default() Self { return Self { n: 0 }; }\n"
	    "};\n"
	    "fn main() i32 {\n"
	    "    var c: Counter = Counter.default();\n"
	    "    return c.n;\n"
	    "}\n");
	ASSERT_TRUE(r.exitCode == 0);
}

void testIntegerOverflowLiteralHasLine() {
	auto r =
	    compileSource("diag_intover", "fn main() i32 {\n"
	                                  "    var v: u32 = 99999999999999999999;\n"
	                                  "    return 0;\n"
	                                  "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	ASSERT_TRUE(stderrContains(r, ":2:"));
	ASSERT_TRUE(stderrContains(r, "exceeds u64 range"));
}

// ── Output format is stable ────────────────────────────────────

void testDiagnosticFormatIsFileLineError() {
	auto r = compileSource("diag_format", "fn main() i32 {\n"
	                                      "    return undefined_thing;\n"
	                                      "}\n");
	ASSERT_TRUE(r.exitCode != 0);
	// Format: "<path>:<line>: error: <message>"
	// — file colon line colon error colon message.
	bool foundShape = false;
	std::istringstream iss(r.stderr_);
	std::string line;
	while (std::getline(iss, line)) {
		// Look for a line that has "/tmp/" (the file prefix) and
		// ":<digits>: error:" somewhere after it.
		auto tmp = line.find("/tmp/");
		auto err = line.find(": error:");
		if (tmp != std::string::npos && err != std::string::npos && err > tmp) {
			foundShape = true;
			break;
		}
	}
	ASSERT_TRUE(foundShape);
}

}  // namespace

class DiagnosticTests {
  public:
	static void registerAllTests(TestFramework &framework) {
		framework.addTest("Diagnostics - unknown function carries :line:",
		                  testUnknownFunctionHasLine);
		framework.addTest("Diagnostics - unknown variable carries :line:",
		                  testUnknownVariableHasLine);
		framework.addTest("Diagnostics - break outside loop carries :line:",
		                  testBreakOutsideLoopHasLine);
		framework.addTest("Diagnostics - continue outside loop carries :line:",
		                  testContinueOutsideLoopHasLine);
		framework.addTest("Diagnostics - unknown field carries :line:",
		                  testUnknownFieldHasLine);
		framework.addTest(
		    "Diagnostics - init-analysis use-after-move carries :line:",
		    testUseAfterMoveHasLine);
		framework.addTest("Diagnostics - multiple errors all reported",
		                  testMultipleErrorsAllReported);
		framework.addTest("Diagnostics - multi-error sorted by line",
		                  testMultiErrorsSortedByLine);
		framework.addTest(
		    "Diagnostics - generic instantiation carries ref trace",
		    testGenericInstantiationCarriesRefTrace);
		framework.addTest("Diagnostics - broken decl does not mask next decl",
		                  testBrokenDeclDoesNotMaskNextDecl);
		framework.addTest("Diagnostics - within-decl multi-error (Poison)",
		                  testRecoveryWithinSingleDecl);
		framework.addTest("Diagnostics - recovery across error classes",
		                  testRecoveryAcrossDifferentErrorClasses);
		framework.addTest("Diagnostics - unknown method is recoverable",
		                  testUnknownMethodIsRecoverable);
		framework.addTest("Diagnostics - parser error carries :line:",
		                  testParserErrorHasLine);
		framework.addTest(
		    "Diagnostics - integer-overflow literal carries :line:",
		    testIntegerOverflowLiteralHasLine);
		framework.addTest("Diagnostics - redeclaration in same scope rejected",
		                  testRedeclarationInSameScopeRejected);
		framework.addTest(
		    "Diagnostics - redeclaration across sibling scopes OK",
		    testRedeclarationAcrossSiblingScopesAllowed);
		framework.addTest("Diagnostics - bool var = float literal rejected",
		                  testTypeMismatchBoolEqualsFloatRejected);
		framework.addTest(
		    "Diagnostics - inferred var width correct for bool / float",
		    testTypeInferenceAllocatesCorrectWidth);
		framework.addTest("Diagnostics - self-referential struct rejected",
		                  testSelfReferentialStructIsCaught);
		framework.addTest(
		    "Diagnostics - self-referential struct via pointer ok",
		    testSelfReferentialStructViaPointerIsAllowed);
		framework.addTest("Diagnostics - function and struct may share a name",
		                  testFunctionAndStructMayShareName);
		framework.addTest(
		    "Diagnostics - multi-step struct cycle shows ref trace",
		    testMultiStepStructCycleShowsReferenceTrace);
		framework.addTest("Diagnostics - three-step struct cycle full chain",
		                  testThreeStepStructCycleShowsFullChain);
		framework.addTest("Diagnostics - self-referential enum rejected",
		                  testSelfReferentialEnumIsCaught);
		framework.addTest("Diagnostics - self-referential union rejected",
		                  testSelfReferentialUnionIsCaught);
		framework.addTest("Diagnostics - struct-via-union cycle rejected",
		                  testStructViaUnionCycleIsCaught);
		framework.addTest("Parser - chained method call on local works",
		                  testChainedMethodCallOnLocal);
		framework.addTest("Parser - single-dot static call still works",
		                  testTwoDotStaticCallStillWorks);
		framework.addTest("Parser - local.field.method() on generic field",
		                  testGenericMethodChainedThroughLocalField);
		framework.addTest("Diagnostics - format is file:line: error:",
		                  testDiagnosticFormatIsFileLineError);
	}
};

int main() {
	TestFramework framework;
	DiagnosticTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
