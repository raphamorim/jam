// Codegen-time must-fail tests.
//
// The init-analyzer test suite (test_init_analysis.cpp) covers errors
// reported by analysis on a parsed module. This file covers errors that
// surface during *codegen* — primarily generic instantiation failures
// like "type T has no method default" — by invoking the jam.out
// binary as a subprocess and asserting on its stderr.
//
// Subprocess approach (rather than driving JamCodegenContext in-process)
// avoids replicating the LLVM-init / target-machine / drop-registry
// scaffolding that main.cpp builds end-to-end. Each test writes a Jam
// source file to /tmp, runs the compiler, captures stderr+exit, and
// asserts on the returned message.

#include "test_framework.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace {

struct CompileResult {
	int exitCode;
	std::string stderr_;
};

// Locate the compiler binary across build layouts. The Makefile tree
// puts it at ./output/jam.out; CI builds with CMake and symlinks
// `ln -sf build/jam jam.out` in the repo root (see ci.yaml). A JAM_BIN
// env var wins over both so a caller can point anywhere.
const std::string &jamBinary() {
	static const std::string bin = [] {
		if (const char *env = std::getenv("JAM_BIN")) {
			return std::string(env);
		}
		if (std::filesystem::exists("output/jam.out")) {
			return std::string("./output/jam.out");
		}
		if (std::filesystem::exists("jam.out")) {
			return std::string("./jam.out");
		}
		// Default to the Makefile layout; the popen failure surfaces
		// the missing binary loudly either way.
		return std::string("./output/jam.out");
	}();
	return bin;
}

// Run jam.out on a one-off source string. The binary must already be
// built (the Makefile target depends on `build`). We invoke from the
// project root so jam.out's relative paths resolve correctly.
CompileResult compileSource(const std::string &name,
                            const std::string &source) {
	std::string path = "/tmp/" + name + ".jam";
	{
		std::ofstream out(path);
		out << source;
	}

	// Redirect stderr->stdout so popen captures both. jam.out usually
	// only writes to stderr on error, but this is robust either way.
	// Explicit `-o` avoids jam's default output name (`./output`)
	// colliding with the build tree's `output/` directory when the
	// tests run from the project root.
	std::string outBin = "/tmp/" + name + ".bin";
	std::string cmd = jamBinary() + " -o " + outBin + " " + path + " 2>&1";

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

// `jam.out test` variant: compiles AND RUNS the file's tfn tests, so it
// exercises the test-runner harness itself (exit-status decoding of the
// spawned binary included). Runs from /tmp so the default output name
// (`./output`) can't collide with the build tree's `output/` directory.
CompileResult runTestMode(const std::string &name, const std::string &source) {
	std::string path = "/tmp/" + name + ".jam";
	{
		std::ofstream out(path);
		out << source;
	}
	std::string jamBin = std::filesystem::absolute(jamBinary()).string();
	std::string cmd = "cd /tmp && " + jamBin + " test " + name + ".jam 2>&1";

	std::string output;
	FILE *pipe = popen(cmd.c_str(), "r");
	if (!pipe) { throw std::runtime_error("popen failed: " + cmd); }
	char buf[256];
	while (fgets(buf, sizeof(buf), pipe) != nullptr) output += buf;
	int status = pclose(pipe);

	int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	return {exitCode, std::move(output)};
}

// `--emit-ir` variant. Skips the link step (so it doesn't try to drop
// the binary into `./output`, which is the build directory) and pipes
// LLVM IR back through the same stdout/stderr channel. Positive tests
// match on IR substrings; negative tests match on the diagnostic.
CompileResult compileSourceIR(const std::string &name,
                              const std::string &source) {
	std::string path = "/tmp/" + name + ".jam";
	{
		std::ofstream out(path);
		out << source;
	}
	std::string cmd = jamBinary() + " --emit-ir " + path + " 2>&1";

	std::string output;
	FILE *pipe = popen(cmd.c_str(), "r");
	if (!pipe) { throw std::runtime_error("popen failed: " + cmd); }
	char buf[256];
	while (fgets(buf, sizeof(buf), pipe) != nullptr) output += buf;
	int status = pclose(pipe);

	int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	return {exitCode, std::move(output)};
}

// Multi-file variant: writes `main.jam` and `lib.jam` into a fresh
// /tmp directory, then runs jam.out on main.jam. The module resolver
// uses main.jam's directory as `baseDir`, so `import("lib")` from
// main resolves to the sibling lib.jam without any std-lib lookup.
CompileResult compileWithLib(const std::string &name,
                             const std::string &mainSource,
                             const std::string &libSource) {
	std::string dir = "/tmp/jam_test_" + name;
	mkdir(dir.c_str(), 0755);
	{
		std::ofstream out(dir + "/lib.jam");
		out << libSource;
	}
	std::string mainPath = dir + "/main.jam";
	{
		std::ofstream out(mainPath);
		out << mainSource;
	}
	std::string outBin = dir + "/main.bin";
	std::string cmd = jamBinary() + " -o " + outBin + " " + mainPath + " 2>&1";

	std::string output;
	FILE *pipe = popen(cmd.c_str(), "r");
	if (!pipe) { throw std::runtime_error("popen failed: " + cmd); }
	char buf[256];
	while (fgets(buf, sizeof(buf), pipe) != nullptr) output += buf;
	int status = pclose(pipe);

	int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	return {exitCode, std::move(output)};
}

}  // namespace

class CodegenErrorTests {
  public:
	// ─── comptime (COMPTIME_PLAN.md Stages 1–2) ──────────────────────

	// A `comp const` whose initializer depends on a runtime value is
	// rejected eagerly at the declaration.
	static void testCompConstRuntimeInitRejected() {
		auto r = compileSource("comp_const_runtime", R"(
fn rt() u64 { return 5; }
comp const BAD = rt();
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "comp const `BAD`"));
		ASSERT_TRUE(stderrContains(r, "compile-time"));
	}

	// A function-local `comp` binding whose initializer isn't foldable
	// is rejected at the declaration.
	static void testCompLocalRuntimeInitRejected() {
		auto r = compileSource("comp_local_runtime", R"(
fn rt() u64 { return 5; }
fn main() {
    comp const x = rt();
    var y: u64 = x;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "comp initializer of `x`"));
	}

	// Reassigning a `comp const` is rejected.
	static void testCompConstReassignRejected() {
		auto r = compileSource("comp_const_reassign", R"(
fn main() {
    comp const x = 1;
    x = 2;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "cannot assign to comp const `x`"));
	}

	// Mutating a `comp var` from inside runtime conditional control
	// flow is rejected — a comp value can't depend on a runtime branch.
	static void testCompVarRuntimeCondAssignRejected() {
		auto r = compileSource("comp_var_rt_cond", R"(
fn flag() bool { return true; }
fn main() {
    comp var x = 1;
    if (flag()) {
        x = 2;
    }
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "comp binding `x`"));
		ASSERT_TRUE(stderrContains(r, "runtime"));
	}

	// A `comp var` value that overflows its established int width is
	// rejected at the assignment.
	static void testCompVarOverflowRejected() {
		auto r = compileSource("comp_var_overflow", R"(
fn main() {
    comp var x: u8 = 1;
    x = 300;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "does not fit"));
	}

	// `comp if` with a non-comptime condition is rejected.
	static void testCompIfRuntimeCondRejected() {
		auto r = compileSource("comp_if_runtime", R"(
fn flag() bool { return true; }
fn main() {
    comp if (flag()) {
        var x: u64 = 1;
    } else {
        var y: u64 = 2;
    }
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "comp if"));
		ASSERT_TRUE(stderrContains(r, "compile-time"));
	}

	// A symbol referenced ONLY in the TAKEN arm of a comp-if is a real
	// reference and must still resolve (the elision only spares the
	// dead arm).
	static void testCompIfTakenArmStillChecked() {
		auto r = compileSource("comp_if_taken_checked", R"(
fn main() {
    comp if (true) {
        var x: u64 = doesNotExist();
    } else {
        var y: u64 = 0;
    }
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "doesNotExist"));
	}

	// A value-returning cfn whose body doesn't `return` a value is
	// rejected at the call site.
	static void testCfnMissingReturnRejected() {
		auto r = compileSource("cfn_missing_return", R"(
cfn noReturn(x: u64) u64 {
    var y: u64 = x + 1;
}
fn main() {
    var z: u64 = noReturn(5);
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "noReturn"));
	}

	// Infinite cfn recursion is caught by the depth cap.
	static void testCfnInfiniteRecursionRejected() {
		auto r = compileSource("cfn_infinite", R"(
cfn forever(n: u64) u64 {
    return forever(n + 1);
}
fn main() {
    var z: u64 = forever(0);
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "depth cap"));
	}

	// Calling a plain (runtime) fn from comptime is rejected.
	static void testCfnCallsRuntimeFnRejected() {
		auto r = compileSource("cfn_calls_runtime", R"(
fn runtimeFn(x: u64) u64 { return x * 2; }
cfn wrap(x: u64) u64 {
    return runtimeFn(x);
}
fn main() {
    var z: u64 = wrap(5);
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "runtime function"));
	}

	// A cfn arg that isn't comptime-known is rejected.
	static void testCfnRuntimeArgRejected() {
		auto r = compileSource("cfn_runtime_arg", R"(
cfn dbl(x: u64) u64 { return x * 2; }
fn main() {
    var n: u64 = 3;
    var z: u64 = dbl(n);
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "compile-time constant"));
	}

	static void registerAllTests(TestFramework &framework) {
		framework.addTest("Comptime - cfn missing return rejected",
		                  testCfnMissingReturnRejected);
		framework.addTest("Comptime - cfn infinite recursion rejected",
		                  testCfnInfiniteRecursionRejected);
		framework.addTest("Comptime - cfn calls runtime fn rejected",
		                  testCfnCallsRuntimeFnRejected);
		framework.addTest("Comptime - cfn runtime arg rejected",
		                  testCfnRuntimeArgRejected);
		framework.addTest("Comptime - comp const runtime init rejected",
		                  testCompConstRuntimeInitRejected);
		framework.addTest("Comptime - comp local runtime init rejected",
		                  testCompLocalRuntimeInitRejected);
		framework.addTest("Comptime - comp const reassign rejected",
		                  testCompConstReassignRejected);
		framework.addTest("Comptime - comp var runtime-cond assign rejected",
		                  testCompVarRuntimeCondAssignRejected);
		framework.addTest("Comptime - comp var overflow rejected",
		                  testCompVarOverflowRejected);
		framework.addTest("Comptime - comp if runtime cond rejected",
		                  testCompIfRuntimeCondRejected);
		framework.addTest("Comptime - comp if taken arm still checked",
		                  testCompIfTakenArmStillChecked);
		framework.addTest("Codegen - Maybe(T) where T lacks default()",
		                  testMaybeOfTypeWithoutDefault);
		// Three rejected-method validation tests removed: the compiler
		// doesn't currently enforce `default()`-shape contracts or
		// reject non-drop / non-default top-level methods. Re-add
		// when (and if) those checks land — for now the tests just
		// assert behavior that doesn't exist.
		framework.addTest("Codegen - int literal in float-typed destination",
		                  testIntToFloatRejected);
		framework.addTest("Codegen - mixed-width float binary op without cast",
		                  testMixedFloatWidthRejected);
		framework.addTest("Codegen - destructured non-pub symbol rejected",
		                  testDestructuredNonPubRejected);
		framework.addTest("Codegen - `pub` re-export compiles",
		                  testPubReexportCompiles);
		framework.addTest("Codegen - `pub` on destructuring import rejected",
		                  testPubOnDestructuringRejected);
		framework.addTest("Codegen - namespace access to non-pub type rejected",
		                  testNamespaceNonPubType);
		framework.addTest(
		    "Codegen - namespace access to missing symbol rejected",
		    testNamespaceMissingSymbol);
		framework.addTest(
		    "Codegen - namespace access via unknown handle rejected",
		    testNamespaceUnknownHandle);
		framework.addTest("Codegen - namespace access to non-pub fn rejected",
		                  testNamespaceNonPubFn);
		framework.addTest(
		    "Codegen - bare-name access to non-pub imported type blocked",
		    testBareNamePrivateBlocked);
		framework.addTest(
		    "Codegen - non-void fn that forgets to return is rejected",
		    testForgottenReturnRejected);
		framework.addTest(
		    "Codegen - non-void fn with missing else branch is rejected",
		    testMissingElseReturnRejected);
		framework.addTest(
		    "Codegen - noreturn fn whose body may fall through is rejected",
		    testNoreturnFallsThroughRejected);
		// Fn-as-value (Rust-style item coercion) + ptr↔int casts.
		framework.addTest("FnRef - bare fn name lowers to ptrtoint @fn",
		                  testFnRefBareName);
		framework.addTest("FnRef - explicit `fn as u64` lowers to ptrtoint",
		                  testFnRefAsU64);
		framework.addTest("FnRef - ptr ↔ u64 round-trips via ptrtoint/inttoptr",
		                  testPtrU64RoundTrip);
		framework.addTest("FnRef - extern fn name resolves to its address",
		                  testFnRefExternFn);
		framework.addTest(
		    "FnRef - generic fn rejected with `cannot take address` diagnostic",
		    testFnRefGenericRejected);
		framework.addTest("FnRef - ptr as u32 (narrower than u64) is rejected",
		                  testPtrAsNarrowIntRejected);
		framework.addTest(
		    "FnRef - u32 as *mut[] u8 (int->ptr) zero-extends + inttoptr",
		    testNarrowIntAsPtrZeroExtends);
		framework.addTest(
		    "FnRef - truly unknown variable still errors (no fn fallback)",
		    testUnknownVariableStillErrors);
		framework.addTest(
		    "XMod - error in imported body blames the DEFINING file",
		    testImportedBodyErrorBlamesDefiningFile);
		framework.addTest(
		    "XMod - imported body cannot see the entry module's imports",
		    testImportedBodyCannotSeeEntryImports);
		// Array sizes in type position + literal-length agreement.
		framework.addTest(
		    "Array - hex size in type position lowers to [2048 x i8]",
		    testHexArraySizeLowersCorrectly);
		framework.addTest("Array - repeat count != array length rejected",
		                  testArrayRepeatCountMismatchRejected);
		framework.addTest("Array - list literal longer than array rejected",
		                  testArrayLitTooLongRejected);
		framework.addTest("Array - list literal shorter than array rejected",
		                  testArrayLitTooShortRejected);
		framework.addTest("Array - float array size rejected",
		                  testFloatArraySizeRejected);
		framework.addTest("Array - array size beyond u32 rejected",
		                  testHugeArraySizeRejected);
		framework.addTest("Array - empty literal into sized array rejected",
		                  testEmptyLitIntoSizedArrayRejected);
		framework.addTest("Array - size literal grammar matches value position",
		                  testArraySizeLiteralGrammarParity);
		framework.addTest("Array - const size folds to [2048 x i8] in IR",
		                  testConstArraySizeLowersCorrectly);
		framework.addTest("Array - non-comptime size rejected",
		                  testNonComptimeSizeRejected);
		framework.addTest("Array - negative const size rejected",
		                  testNegativeConstSizeRejected);
		framework.addTest("Array - float const size rejected",
		                  testFloatConstSizeRejected);
		framework.addTest("Array - length mismatch via const size rejected",
		                  testConstSizeMismatchRejected);
		framework.addTest(
		    "Harness - signal-killed test binary reported as failure",
		    testSignalKilledBinaryReported);
		// Move tracking for drop-bearing locals (Swift-style static
		// resolution: ambiguous moves are rejected, no runtime flags).
		framework.addTest("Move - use after move rejected",
		                  testUseAfterMoveRejected);
		framework.addTest("Move - conditional move rejected",
		                  testConditionalMoveRejected);
		framework.addTest("Move - reassign after move rejected",
		                  testReassignAfterMoveRejected);
		framework.addTest("Move - move inside loop rejected",
		                  testLoopMoveRejected);
		framework.addTest("Move - conditional method-call move rejected",
		                  testConditionalMethodMoveRejected);
		framework.addTest("Move - use after method-call move rejected",
		                  testUseAfterMethodMoveRejected);
		framework.addTest("Move - move out of borrowed param rejected",
		                  testMoveOutOfBorrowedParamRejected);
		framework.addTest("Modes - `&` on a mode argument rejected",
		                  testAddrOfOnModeArgRejected);
		framework.addTest("Move - drop-bearing field extraction rejected",
		                  testFieldExtractRejected);
		framework.addTest("Move - drop-bearing repeat literal rejected",
		                  testDropBearingRepeatRejected);
		framework.addTest("Move - double capture in array literal rejected",
		                  testDoubleArrayCaptureRejected);
		framework.addTest("Move - use after enum payload capture rejected",
		                  testUseAfterEnumCaptureRejected);
		framework.addTest("Move - conditional enum payload capture rejected",
		                  testConditionalEnumCaptureRejected);
		framework.addTest(
		    "Move - imported-module body analyzed, blames defining file",
		    testImportedBodyMoveAnalyzed);
		framework.addTest("Clone - owns-resources without cfn clone rejected",
		                  testCloneOwnsResourcesRejected);
		framework.addTest(
		    "Clone - container clone conditional on element cloneability",
		    testConditionalContainerClone);
		framework.addTest("Clone - payloaded enum clone fenced",
		                  testEnumCloneFenced);
		framework.addTest(
		    "Conditional - withdrawn method replays reason at call site",
		    testWithdrawnMethodReplays);
		framework.addTest("MatchMove - rejections (use-after/borrowed/cond)",
		                  testMatchMoveRejections);
		framework.addTest("MatchMove - own-cfn-drop enum edge (E0509 analog)",
		                  testMatchMoveOwnDropEnum);
	}

  private:
	// A generic body that calls T.default() must instantiate to a
	// concrete T that has a default() method. NoDefault doesn't, so
	// instantiation should error with a precise message naming both
	// the missing method and the type that's missing it.
	static void testMaybeOfTypeWithoutDefault() {
		auto r = compileSource("must_fail_no_default", R"(
const NoDefault = struct {
    n: i32,
};

fn Maybe(T: type) type {
    return struct {
        storage: T,
        valid: bool,
        fn default() Self {
            return Self { storage: T.default(), valid: false };
        }
    };
}

const MaybeND = Maybe(NoDefault);

fn main() i32 {
    var m: MaybeND = MaybeND.default();
    return m.storage.n;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "NoDefault"));
		ASSERT_TRUE(stderrContains(r, "default"));
	}

	// Float-typed destinations need a float literal (`3.0`) or an
	// explicit `as` cast. Implicit int->float coercion is rejected so
	// the source spells out every bit-pattern change.
	static void testIntToFloatRejected() {
		auto r = compileSource("must_fail_int_to_float", R"(
fn main() {
    var x: f32 = 3;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "float"));
		ASSERT_TRUE(stderrContains(r, "as"));
	}

	// Binary ops over mismatched float widths (`f32 + f64`) need an
	// explicit `as` cast on the narrower side. Without it the codegen
	// would emit `fadd float, double`, which is malformed IR.
	static void testMixedFloatWidthRejected() {
		auto r = compileSource("must_fail_mixed_float_width", R"(
fn add(a: f32, b: f64) f64 { return a + b; }
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "mismatched"));
		ASSERT_TRUE(stderrContains(r, "as"));
	}

	// `const { X } = import("lib")` requires X to be `pub` in lib.
	// Non-pub triggers a precise "is not exported" diagnostic.
	static void testDestructuredNonPubRejected() {
		auto r = compileWithLib("must_fail_destructured_nonpub",
		                        R"(
const { Hidden } = import("lib");
fn main() {}
)",
		                        "const Hidden = struct { n: i32, };\n");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "Hidden"));
		ASSERT_TRUE(stderrContains(r, "is not exported from module"));
	}

	// `pub const X = import(...)` is a re-export. The module that
	// writes it surfaces the imported module under its own namespace
	// so a downstream importer can chain through (`outer.X.member`).
	// Just check that the form parses + compiles cleanly — exercising
	// the re-export chain end-to-end is the smoke test in
	// `test_print.cpp` and the unit suite.
	static void testPubReexportCompiles() {
		auto r = compileSource("ok_pub_reexport", R"(
pub const fmt = import("fmt");
fn main() {}
)");
		ASSERT_TRUE(r.exitCode == 0);
	}

	// `pub const { X } = import(...)` is rejected explicitly so the
	// destructuring path doesn't silently drop the pub modifier.
	static void testPubOnDestructuringRejected() {
		auto r = compileSource("must_fail_pub_destructuring", R"(
pub const { Vec } = import("std/collections");
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "pub"));
		ASSERT_TRUE(stderrContains(r, "destructuring"));
	}

	// `mod.Private` where Private isn't pub: emits the precise
	// "is not exported" diagnostic instead of a generic "Unknown".
	static void testNamespaceNonPubType() {
		// The type access itself is what we're testing; using
		// `lib.Private` as a fn-param type is enough to fire the
		// "is not exported" diagnostic without needing a struct
		// literal of that type.
		auto r = compileWithLib("must_fail_ns_nonpub_type",
		                        R"(
const lib = import("lib");
fn takes(p: lib.Private) {}
fn main() {}
)",
		                        "const Private = struct { n: i32, };\n");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "Private"));
		ASSERT_TRUE(stderrContains(r, "is not exported from module"));
	}

	// `mod.Nope` where Nope doesn't exist at all in `lib`: distinct
	// "does not exist" diagnostic, separable from the non-pub case.
	static void testNamespaceMissingSymbol() {
		auto r = compileWithLib("must_fail_ns_missing",
		                        R"(
const lib = import("lib");
fn takes(p: lib.Nope) {}
fn main() {}
)",
		                        "pub const Val = struct { n: i32, };\n");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "Nope"));
		ASSERT_TRUE(stderrContains(r, "does not exist in module"));
	}

	// `unknown.X` where `unknown` was never bound to an import.
	static void testNamespaceUnknownHandle() {
		auto r = compileSource("must_fail_ns_unknown_handle", R"(
fn takes(p: unknown.X) {}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "unknown module handle"));
		ASSERT_TRUE(stderrContains(r, "unknown.X"));
	}

	// `mod.fn()` where fn isn't pub: namespace fn-call path also
	// produces the precise "not exported" diagnostic via the shared
	// formatNamespaceLookupError helper.
	static void testNamespaceNonPubFn() {
		auto r = compileWithLib("must_fail_ns_nonpub_fn",
		                        R"(
const lib = import("lib");
fn main() { var r: i32 = lib.priv(7); }
)",
		                        "fn priv(x: i32) i32 { return x; }\n");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "priv"));
		ASSERT_TRUE(stderrContains(r, "is not exported from module"));
	}

	// Non-void function whose body may complete without returning a
	// value. Previously emitted `unreachable` silently — calling the
	// fn was undefined behavior at runtime. Now rejected at compile
	// time via the analyzer's noreturn-tail check.
	static void testForgottenReturnRejected() {
		auto r = compileSource("must_fail_forgotten_return", R"(
fn forgotReturn() i32 { var x: i32 = 42; }
fn main() i32 { return forgotReturn(); }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "forgotReturn"));
		ASSERT_TRUE(stderrContains(r, "non-void return type"));
		ASSERT_TRUE(stderrContains(r, "without returning a value"));
	}

	// A `noreturn` function must diverge on every path. A body that
	// could complete normally (e.g. a single var-decl with no later
	// statement) violates the contract and is rejected.
	static void testNoreturnFallsThroughRejected() {
		auto r = compileSource("must_fail_noreturn_fallthrough", R"(
fn faker() noreturn {
    var x: i32 = 1;
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "faker"));
		ASSERT_TRUE(stderrContains(r, "noreturn"));
		ASSERT_TRUE(stderrContains(r, "without diverging"));
	}

	// `if (cond) { return X; }` with no else falls through on the
	// false branch. The same diagnostic should fire even though the
	// body contains a `return` statement.
	static void testMissingElseReturnRejected() {
		auto r = compileSource("must_fail_missing_else_return", R"(
fn maybe(cond: bool) i32 {
    if (cond) { return 1; }
}
fn main() i32 { return maybe(true); }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "maybe"));
		ASSERT_TRUE(stderrContains(r, "non-void return type"));
	}

	// Non-pub structs from imported modules must not leak via the
	// bare-name path. Previously `declareStructs` registered all
	// imported structs globally regardless of pub; the publicOnly
	// gate closes that hole.
	static void testBareNamePrivateBlocked() {
		auto r = compileWithLib("must_fail_bare_name_private",
		                        R"(
const lib = import("lib");
fn takes(p: Private) {}
fn main() {}
)",
		                        "const Private = struct { n: i32, };\n");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "Private"));
	}

	// Fn-as-value (Rust-style item coercion) + ptr↔int casts
	//
	// These exercise the `export fn` callback workflow needed for
	// SDL_AudioSpec-style C-ABI callbacks. The bare-name form mirrors
	// Rust's implicit fn-item coercion; the `as u64` form is the
	// explicit cast. Both must lower to LLVM `ptrtoint`.

	// Bare fn name in expression position binds as a u64 — the
	// "coercion" branch of the AsCast early-exits when src == dst so
	// we should see a direct ptrtoint store with no extra cast IR.
	static void testFnRefBareName() {
		auto r = compileSourceIR("fnref_bare", R"(
export fn cb(ud: u64, s: *mut[] u8, len: i32) { s[0] = 1; }
fn main() {
    var addr: u64 = cb;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		// Function should be externally linked, C-ABI (no `internal`).
		ASSERT_TRUE(stderrContains(r, "define void @cb"));
		// The fn-ref must lower to ptrtoint of the named symbol.
		ASSERT_TRUE(stderrContains(r, "ptrtoint (ptr @cb to i64)"));
	}

	// Explicit `fn as u64` cast. With FnRef typed as u64 the cast is
	// a no-op at the JIR level — same ptrtoint instruction.
	static void testFnRefAsU64() {
		auto r = compileSourceIR("fnref_as_u64", R"(
export fn cb(ud: u64, s: *mut[] u8, len: i32) { s[0] = 1; }
fn main() {
    var addr: u64 = cb as u64;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "ptrtoint (ptr @cb to i64)"));
	}

	// Pointer ↔ u64 round-trip. Confirms both ptrtoint and inttoptr
	// branches in astgenAsCast / jirCodegen exist and produce the
	// matching LLVM instructions.
	static void testPtrU64RoundTrip() {
		auto r = compileSourceIR("ptr_u64_round_trip", R"(
extern fn malloc(size: u64) *mut[] u8;
fn main() {
    var p: *mut[] u8 = malloc(16);
    var a: u64 = p as u64;
    var p2: *mut[] u8 = a as *mut[] u8;
    p2[0] = 99;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "ptrtoint ptr"));
		ASSERT_TRUE(stderrContains(r, "inttoptr i64"));
	}

	// Pure `extern fn` (no body) should still be referenceable by
	// name — the symbol resolves to the LLVM `declare` placeholder
	// and ptrtoint folds it just like a defined function.
	static void testFnRefExternFn() {
		auto r = compileSourceIR("fnref_extern", R"(
extern fn malloc(size: u64) *mut[] u8;
fn main() {
    var addr: u64 = malloc;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "ptrtoint (ptr @malloc to i64)"));
	}

	// Taking the address of a generic fn is meaningless before
	// monomorphization — no concrete LLVM symbol exists yet. AstGen
	// surfaces a precise diagnostic naming both the action and the
	// fn.
	static void testFnRefGenericRejected() {
		auto r = compileSource("fnref_generic", R"(
fn identity(T: type, x: T) T { return x; }
fn main() {
    var a: u64 = identity as u64;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "cannot take address of generic fn"));
		ASSERT_TRUE(stderrContains(r, "identity"));
	}

	// Pointers are 64-bit on every target Jam supports. Casting to a
	// narrower int width would silently truncate the upper bits and
	// is rejected up front rather than letting LLVM emit a lossy
	// truncate. The user can always do `(p as u64) as u32` if they
	// really want the lower 32 bits.
	static void testPtrAsNarrowIntRejected() {
		auto r = compileSource("ptr_as_narrow_int", R"(
extern fn malloc(size: u64) *mut[] u8;
fn main() {
    var p: *mut[] u8 = malloc(8);
    var a: u32 = p as u32;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "unsupported `as` cast"));
	}

	// The OTHER direction is intentionally NOT symmetric: an int->ptr cast
	// permits ANY integer width into a THIN pointer -- the int is an address,
	// and narrower-than-pointer sources just widen to pointer width. (Only the
	// reverse, ptr->int, is width-checked: u64 only.) So `u32 as *mut[] u8` is
	// accepted and lowers to zext-to-i64 + inttoptr. `*mut[] u8` is a thin
	// many-ptr, not a slice, so it is a legal target.
	static void testNarrowIntAsPtrZeroExtends() {
		auto r = compileSourceIR("narrow_int_as_ptr", R"(
extern fn sink(p: *mut[] u8);
fn main() {
    var n: u32 = 0xFF;
    var p: *mut[] u8 = n as *mut[] u8;
    sink(p);
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "inttoptr"));
		ASSERT_TRUE(stderrContains(r, "zext"));
	}

	// Regression guard: the fn-name fallback in astgenVariable must
	// NOT swallow the existing "unknown variable" diagnostic. If a
	// name is neither a local, a module const, nor a function, the
	// error must still fire — otherwise downstream codegen will
	// crash trying to resolve a non-existent symbol.
	static void testUnknownVariableStillErrors() {
		auto r = compileSource("fnref_unknown_var", R"(
fn main() {
    var a: u64 = nonexistent_thing;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "unknown variable"));
		ASSERT_TRUE(stderrContains(r, "nonexistent_thing"));
	}

	// An error inside an IMPORTED module's body (here, a `cfn drop` in
	// lib.jam) must be attributed to lib.jam — the file the code was
	// WRITTEN in — not to main.jam, the entry file being compiled.
	//
	// The imported-body codegen pass runs with the entry file as the
	// global currentFile(); the fix sets currentFile() to the defining
	// module for that pass. The node's line is already correct (one global
	// NodeStore), so the bug was a wrong FILENAME with a right line.
	static void testImportedBodyErrorBlamesDefiningFile() {
		auto r = compileWithLib("xmod_body_err",
		                        // main.jam: just imports + uses the type so
		                        // lib.jam's drop body gets compiled.
		                        R"(
const { makeBad } = import("lib");
fn main() {
    var b = makeBad();
}
)",
		                        // lib.jam: the bad reference is on line 4.
		                        R"(pub const Bad = struct {
    x: u32,
    cfn drop(self: mut Self) {
        self.x = unknownNameXyz;
    }
};
pub fn makeBad() Bad { return Bad { x: 0 }; }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "unknown variable"));
		// The whole point: the diagnostic names the DEFINING file...
		ASSERT_TRUE(stderrContains(r, "lib.jam:4"));
		// ...and never blames the entry file for this imported-body error.
		ASSERT_TRUE(!stderrContains(r, "main.jam:"));
	}

	// Import-scope strictness: a body resolves qualified names against ITS OWN
	// module's imports, never the entry module's. Here main.jam binds
	// `p = import("lib")`, but lib.jam's drop body uses `p.ping()` without
	// importing `p` itself. It must be rejected (not silently resolved against
	// main's `p`), and the error must point at lib.jam.
	static void testImportedBodyCannotSeeEntryImports() {
		auto r = compileWithLib("xmod_no_entry_leak",
		                        R"(
const p = import("lib");
const { makeThing } = import("lib");
fn main() {
    var n: u32 = 0;
    var t: Thing = makeThing(&n);
}
)",
		                        R"(pub fn ping() u32 { return 7; }
pub const Thing = struct {
    sink: *mut u32,
    cfn drop(self: mut Self) {
        var q: *mut u32 = self.sink;
        q.* = p.ping();
    }
};
pub fn makeThing(s: *mut u32) Thing { return Thing { sink: s }; }
)");
		ASSERT_TRUE(r.exitCode != 0);
		// `p` is undeclared in lib.jam's namespace -> rejected, not leaked.
		ASSERT_TRUE(stderrContains(r, "unknown module handle"));
		ASSERT_TRUE(stderrContains(r, "lib.jam:"));
		ASSERT_TRUE(!stderrContains(r, "main.jam:"));
	}

	// `[0x800]u8` must intern a 2048-element array type. The size token
	// goes through the full number-literal parser; a base-10-only scan
	// (std::stoul) stops at the `x` and silently produces `[0]u8`, after
	// which the repeat literal memsets 2048 bytes into a 0-byte slot.
	static void testHexArraySizeLowersCorrectly() {
		auto r = compileSourceIR("array_hex_size_ir", R"(
const Block = struct {
    cells: [0x800]u8,
};
fn main() {
    var b = Block{cells: [0; 0x800]};
    b.cells[0] = 1;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "[2048 x i8]"));
		ASSERT_TRUE(!stderrContains(r, "[0 x i8]"));
	}

	// A repeat literal whose count disagrees with the destination array
	// length must be a compile error — the memset/store lowering sizes
	// from the count, so letting it through writes past the slot.
	static void testArrayRepeatCountMismatchRejected() {
		auto r = compileSource("array_repeat_mismatch", R"(
fn main() {
    const a: [4]u8 = [0; 0x800];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    r, "array repeat count 2048 does not match array type length 4"));
	}

	static void testArrayLitTooLongRejected() {
		auto r = compileSource("array_lit_too_long", R"(
fn main() {
    const a: [2]u8 = [1, 2, 3];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    r, "array literal has 3 element(s) but the array type expects 2"));
	}

	static void testArrayLitTooShortRejected() {
		auto r = compileSource("array_lit_too_short", R"(
fn main() {
    const a: [4]u8 = [1, 2];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    r, "array literal has 2 element(s) but the array type expects 4"));
	}

	static void testFloatArraySizeRejected() {
		auto r = compileSource("array_float_size", R"(
fn main() {
    var a: [1.5]u8 = [0; 1];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(r, "array size must be a non-negative integer"));
	}

	static void testHugeArraySizeRejected() {
		auto r = compileSource("array_huge_size", R"(
fn main() {
    var a: [0x1_0000_0000]u8 = [0; 1];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "exceeds u32 range"));
	}

	// Zig errors `expected 8 array elements; found 0` here; jam's `[]`
	// empty literal must hit the same wall instead of leaving the array
	// uninitialized.
	static void testEmptyLitIntoSizedArrayRejected() {
		auto r = compileSource("array_empty_lit", R"(
fn main() {
    const a: [8]u8 = [];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    r, "array literal has 0 element(s) but the array type expects 8"));
	}

	// The size token in `[N]T` inherits the FULL number-literal grammar,
	// not just "digits the old std::stoul happened to eat": uppercase
	// base prefixes, underscores glued to the prefix, and C-style
	// leading zeros are rejected with the same diagnostics value
	// position produces.
	static void testArraySizeLiteralGrammarParity() {
		auto upper = compileSource("array_size_upper_base", R"(
fn main() {
    var a: [0X800]u8 = [0; 2048];
}
)");
		ASSERT_TRUE(upper.exitCode != 0);
		ASSERT_TRUE(stderrContains(upper, "base prefix must be lowercase"));

		auto glued = compileSource("array_size_underscore_after_base", R"(
fn main() {
    var a: [0x_800]u8 = [0; 2048];
}
)");
		ASSERT_TRUE(glued.exitCode != 0);
		ASSERT_TRUE(stderrContains(glued, "underscore not allowed"));

		auto leading = compileSource("array_size_leading_zero", R"(
fn main() {
    var a: [07]u8 = [0; 7];
}
)");
		ASSERT_TRUE(leading.exitCode != 0);
		ASSERT_TRUE(stderrContains(leading, "leading zero is not allowed"));
	}

	// `[SIZE]u8` with a module const folds at type-resolution time and
	// lowers identically to the literal spelling — the IR must carry
	// the folded length. Mirrors how a comptime-known length resolves
	// before lowering in the reference compiler.
	static void testConstArraySizeLowersCorrectly() {
		auto r = compileSourceIR("array_const_size_ir", R"(
const SIZE = 0x800;
const Block = struct {
    cells: [SIZE]u8,
};
fn main() {
    var b = Block{cells: [0; SIZE]};
    b.cells[0] = 1;
}
)");
		ASSERT_TRUE(r.exitCode == 0);
		ASSERT_TRUE(stderrContains(r, "[2048 x i8]"));
		ASSERT_TRUE(!stderrContains(r, "[0 x i8]"));
	}

	// Undefined names and runtime variables can't fold — both must be
	// rejected with the comptime-known diagnostic, not silently sized.
	static void testNonComptimeSizeRejected() {
		auto undef = compileSource("array_size_undefined", R"(
fn main() {
    var a: [BOGUS]u8 = [0; 4];
}
)");
		ASSERT_TRUE(undef.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(undef, "array length must be comptime-known"));

		auto runtime = compileSource("array_size_runtime_var", R"(
fn main() {
    var n: u32 = 5;
    var a: [n]u8 = [0; 5];
}
)");
		ASSERT_TRUE(runtime.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(runtime, "array length must be comptime-known"));
	}

	static void testNegativeConstSizeRejected() {
		auto r = compileSource("array_size_negative_const", R"(
const NEG = 0 - 4;
fn main() {
    var a: [NEG]u8 = [0; 4];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(r, "array size must be a non-negative integer"));
	}

	static void testFloatConstSizeRejected() {
		auto r = compileSource("array_size_float_const", R"(
const F = 1.5;
fn main() {
    var a: [F]u8 = [0; 1];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(r, "array size must be a non-negative integer"));
	}

	// The folded length feeds the same literal-length checks as a
	// written-out size: a 3-element literal into `[N]u8` with N = 4
	// reports the mismatch with the RESOLVED length.
	static void testConstSizeMismatchRejected() {
		auto r = compileSource("array_size_const_mismatch", R"(
const N = 4;
fn main() {
    const a: [N]u8 = [1, 2, 3];
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    r, "array literal has 3 element(s) but the array type expects 4"));
	}

	// A test binary killed by a signal has no exit status; decoding it
	// with WEXITSTATUS alone reads 0 and reports the crashed run as
	// "passed" — with the child's unflushed "testing ..." line lost, the
	// failure would be completely invisible. The runner must surface the
	// signal and return shell-convention 128+sig.
	static void testSignalKilledBinaryReported() {
		auto r = runTestMode("harness_signal_crash", R"(
tfn crashByWildStore() {
    var addr: u64 = 1;
    var p: *mut[] u8 = addr as *mut[] u8;
    p[0] = 9;
}
)");
		ASSERT_TRUE(r.exitCode > 128);
		ASSERT_TRUE(stderrContains(r, "terminated by signal"));
	}

	// Shared prelude for the move-tracking rejections: a drop-bearing
	// Counter and a `move`-mode consumer.
	static std::string movePrelude() {
		return R"(
const Counter = struct {
    value: u32,
    sink: *mut u32,
};

cfn drop(self: mut Counter) {
    var p: *mut u32 = self.sink;
    p.* = p.* + 1;
}

fn consume(c: move Counter) {
    var owned: Counter = c;
}
)";
	}

	static void testUseAfterMoveRejected() {
		auto r = compileSource("move_use_after", movePrelude() + R"(
fn bad(sink: *mut u32) u32 {
    var c: Counter = Counter { value: 1, sink: sink };
    consume(c);
    return c.value;
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "use of moved binding `c`"));
	}

	// rustc would insert a runtime drop flag here (DropStyle::
	// Conditional); jam rejects, Swift SE-0390 style, keeping codegen
	// flag-free.
	static void testConditionalMoveRejected() {
		auto r = compileSource("move_conditional", movePrelude() + R"(
fn bad(sink: *mut u32, doIt: bool) {
    var c: Counter = Counter { value: 1, sink: sink };
    if (doIt) {
        consume(c);
    }
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "move it on all control-flow paths"));
	}

	static void testReassignAfterMoveRejected() {
		auto r = compileSource("move_reassign", movePrelude() + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    consume(c);
    c = Counter { value: 2, sink: sink };
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(
		    stderrContains(r, "cannot assign to `c` after it was moved"));
	}

	// A loop body runs zero or more times, so a move inside of an outer
	// binding is always a maybe-move.
	static void testLoopMoveRejected() {
		auto r = compileSource("move_in_loop", movePrelude() + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    var i: u32 = 0;
    while (i < 3) {
        consume(c);
        i = i + 1;
    }
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "move it on all control-flow paths"));
	}

	// Method-call moves (`v.push(c)` with `push(value: move T)`) follow
	// the same rules — the analysis resolves the callee through the
	// receiver's static type.
	static void testConditionalMethodMoveRejected() {
		auto r = compileSource("move_cond_push", R"(
const { Vec } = import("std/collections");
)" + movePrelude() + R"(
fn bad(sink: *mut u32, doIt: bool) {
    var v: Vec(Counter) = Vec(Counter).empty();
    var c: Counter = Counter { value: 1, sink: sink };
    if (doIt) {
        v.push(c);
    }
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "move it on all control-flow paths"));
	}

	static void testUseAfterMethodMoveRejected() {
		auto r = compileSource("move_use_after_push", R"(
const { Vec } = import("std/collections");
)" + movePrelude() + R"(
fn bad(sink: *mut u32) u32 {
    var v: Vec(Counter) = Vec(Counter).empty();
    var c: Counter = Counter { value: 1, sink: sink };
    v.push(c);
    return c.value;
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "use of moved binding `c`"));
	}

	// A `let`/`mut` parameter is borrowed, not owned: moving a
	// drop-bearing value out of it (here via a var-init move) would
	// leave the caller and the new owner both dropping the payload.
	static void testMoveOutOfBorrowedParamRejected() {
		auto r = compileSource("move_steal_from_let", movePrelude() + R"(
fn steal(c: Counter) {
    var mine: Counter = c;
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "borrowed, not owned"));
		ASSERT_TRUE(stderrContains(r, "declare the parameter `move`"));
	}

	// Ownership is tracked per whole binding: extracting a drop-bearing
	// value out of a pure field path duplicates ownership (the field
	// copy and the aggregate's drop glue both drop the payload).
	// rustc models per-field move paths; jam rejects in every
	// ownership-transferring position.
	static void testFieldExtractRejected() {
		std::string holder = movePrelude() + R"(
const Holder = struct {
    c: Counter,
};
)";
		auto moveArg = compileSource("field_extract_move", holder + R"(
fn bad(sink: *mut u32) {
    var h: Holder = Holder { c: Counter { value: 1, sink: sink } };
    consume(h.c);
}
fn main() {}
)");
		ASSERT_TRUE(moveArg.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    moveArg, "cannot move a drop-bearing field out of its aggregate"));

		auto ret = compileSource("field_extract_return", holder + R"(
fn takeOut(h: move Holder) Counter {
    return h.c;
}
fn main() {}
)");
		ASSERT_TRUE(ret.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    ret, "cannot return a drop-bearing field out of its aggregate"));

		auto varInit = compileSource("field_extract_varinit", holder + R"(
fn bad(sink: *mut u32) {
    var h: Holder = Holder { c: Counter { value: 1, sink: sink } };
    var x: Counter = h.c;
}
fn main() {}
)");
		ASSERT_TRUE(varInit.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    varInit, "cannot copy a drop-bearing field out of its aggregate"));

		// Whole-binding moves stay legal — the rule only bites paths.
		auto whole = compileSource("field_extract_whole_ok", holder + R"(
fn consumeHolder(h: move Holder) {}
fn ok(sink: *mut u32) {
    var h: Holder = Holder { c: Counter { value: 1, sink: sink } };
    consumeHolder(h);
}
fn main() {}
)");
		ASSERT_TRUE(whole.exitCode == 0);
	}

	// `[c; N]` duplicates one value into N slots — N owners of one
	// payload for drop-bearing types. rustc requires `T: Copy`; jam
	// rejects.
	static void testDropBearingRepeatRejected() {
		auto r = compileSource("array_repeat_dropbearing", movePrelude() + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    var arr: [2]Counter = [c; 2];
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "owners of one drop-bearing value"));
	}

	// `[c, c]` — the first element moves c, the second is use-after-move.
	static void testDoubleArrayCaptureRejected() {
		auto r = compileSource("array_double_capture", movePrelude() + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    var arr: [2]Counter = [c, c];
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "use of moved binding `c`"));
	}

	// Enum payload construction is a MOVE of bare drop-bearing args:
	// the enum's tag-dispatched glue owns the drop.
	static void testUseAfterEnumCaptureRejected() {
		auto r = compileSource("enum_use_after_capture", movePrelude() + R"(
const Maybe = enum {
    None,
    Some(Counter),
};
fn bad(sink: *mut u32) u32 {
    var c: Counter = Counter { value: 1, sink: sink };
    var m: Maybe = Maybe.Some(c);
    return c.value;
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "use of moved binding `c`"));
	}

	static void testConditionalEnumCaptureRejected() {
		auto r = compileSource("enum_cond_capture", movePrelude() + R"(
const Maybe = enum {
    None,
    Some(Counter),
};
fn bad(sink: *mut u32, doIt: bool) {
    var c: Counter = Counter { value: 1, sink: sink };
    if (doIt) {
        var m: Maybe = Maybe.Some(c);
    }
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "move it on all"));
	}

	// The mode-aware callsite analysis covers IMPORTED module bodies
	// (including their calls to private same-module helpers), with
	// diagnostics attributed to the defining file. Regression: std
	// generic bodies escaping analysis is how Box.init's double-drop
	// shipped.
	static void testImportedBodyMoveAnalyzed() {
		auto r = compileWithLib("move_imported_body",
		                        R"(
const { badHelper } = import("lib");
fn main() {
    var hits: u32 = 0;
    badHelper(&hits);
}
)",
		                        R"(pub const Counter = struct {
    value: u32,
    sink: *mut u32,
};

cfn drop(self: mut Counter) {
    var p: *mut u32 = self.sink;
    p.* = p.* + 1;
}

fn consume(c: move Counter) {
}

pub fn badHelper(sink: *mut u32) u32 {
    var c: Counter = Counter { value: 1, sink: sink };
    consume(c);
    return c.value;
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "use of moved binding `c`"));
		ASSERT_TRUE(stderrContains(r, "lib.jam:"));
	}

	// clone(): a type with its own cfn drop but no cfn clone cannot be
	// cloned — the compiler can clone structure, not resources.
	static void testCloneOwnsResourcesRejected() {
		auto r = compileSource("clone_owns_resources", movePrelude() + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    var d: Counter = c.clone();
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "owns resources"));
		ASSERT_TRUE(stderrContains(r, "define `cfn clone"));
	}

	// Vec(T)'s clone instantiates CONDITIONALLY (Rust's
	// `impl<T: Clone> Clone for Vec<T>`): Vec(Counter) itself works,
	// but calling .clone() on it reports owns-resources at the call
	// site, naming the instantiation.
	static void testConditionalContainerClone() {
		auto r = compileSource("clone_vec_conditional", R"(
const { Vec } = import("std/collections");
)" + movePrelude() + R"(
fn bad(sink: *mut u32) {
    var v: Vec(Counter) = Vec(Counter).empty();
    v.push(Counter { value: 1, sink: sink });
    var w: Vec(Counter) = v.clone();
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "Vec__Counter"));
		ASSERT_TRUE(stderrContains(r, "owns resources"));
	}

	// Enum payload clone recurses into the payload's clone tier: a
	// non-cloneable payload (own cfn drop, no cfn clone) reports
	// owns-resources at the clone site. (Cloneable payloads deep-clone
	// — pinned in tests/unit/test_match_move.jam.)
	static void testEnumCloneFenced() {
		auto r = compileSource("clone_enum_fenced", movePrelude() + R"(
const Maybe = enum {
    None,
    Some(Counter),
};
fn bad(sink: *mut u32) {
    var m: Maybe = Maybe.Some(Counter { value: 1, sink: sink });
    var n: Maybe = m.clone();
}
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "owns resources"));
		ASSERT_TRUE(stderrContains(r, "define `cfn clone"));
	}

	// Conditional generic methods: a method whose body fails astgen or
	// the mode-aware analysis for these type args is WITHDRAWN; calling
	// it replays the recorded reason. Three flavors: the `v[i]` sugar's
	// `at` (astgen withdrawal: element not cloneable), `get` (same),
	// and `filled` (ANALYSIS withdrawal: the fill duplicates a borrowed
	// drop-bearing value).
	static void testWithdrawnMethodReplays() {
		std::string prelude = std::string(R"(
const { Vec } = import("std/collections");
)") + movePrelude();
		auto idx = compileSource("cond_at_replay", prelude + R"(
fn bad(sink: *mut u32) u32 {
    var v: Vec(Counter) = Vec(Counter).empty();
    v.push(Counter { value: 1, sink: sink });
    return v[0].value;
}
fn main() {}
)");
		ASSERT_TRUE(idx.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    idx, "`Vec__Counter.at` is not available for this instantiation"));
		ASSERT_TRUE(stderrContains(idx, "owns resources"));

		auto filled = compileSource("cond_filled_replay", prelude + R"(
fn bad(sink: *mut u32) {
    var c: Counter = Counter { value: 1, sink: sink };
    var v: Vec(Counter) = Vec(Counter).filled(c, 3);
}
fn main() {}
)");
		ASSERT_TRUE(filled.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    filled,
		    "`Vec__Counter.filled` is not available for this instantiation"));
		ASSERT_TRUE(stderrContains(filled, "borrowed, not owned"));

		auto get = compileSource("cond_get_replay", prelude + R"(
fn bad(sink: *mut u32) u32 {
    var v: Vec(Counter) = Vec(Counter).empty();
    v.push(Counter { value: 1, sink: sink });
    var got = v.get(0);
    return 0;
}
fn main() {}
)");
		ASSERT_TRUE(get.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    get, "`Vec__Counter.get` is not available for this instantiation"));
	}

	// MATCH-MOVE rejections: matching a drop-bearing enum consumes the
	// scrutinee, so re-matching is use-after-move, a borrowed (let
	// param) scrutinee can't be consumed, and a conditional match of an
	// outer binding violates the depth rule.
	static void testMatchMoveRejections() {
		std::string en = movePrelude() + R"(
const Maybe = enum {
    None,
    Some(Counter),
};
)";
		auto twice = compileSource("matchmove_twice", en + R"(
fn bad(sink: *mut u32) u32 {
    var m: Maybe = Maybe.Some(Counter { value: 1, sink: sink });
    match (m) { _ { } }
    match (m) { _ { return 1; } }
    return 0;
}
fn main() {}
)");
		ASSERT_TRUE(twice.exitCode != 0);
		ASSERT_TRUE(stderrContains(twice, "use of moved binding `m`"));

		auto borrowed = compileSource("matchmove_borrowed", en + R"(
fn bad(m: Maybe) u32 {
    match (m) { _ { return 1; } }
    return 0;
}
fn main() {}
)");
		ASSERT_TRUE(borrowed.exitCode != 0);
		ASSERT_TRUE(stderrContains(borrowed, "borrowed, not owned"));

		auto cond = compileSource("matchmove_cond", en + R"(
fn bad(sink: *mut u32, doIt: bool) {
    var m: Maybe = Maybe.Some(Counter { value: 1, sink: sink });
    if (doIt) {
        match (m) { _ { } }
    }
}
fn main() {}
)");
		ASSERT_TRUE(cond.exitCode != 0);
		ASSERT_TRUE(stderrContains(cond, "move it on all control-flow paths"));
	}

	// Enums with their OWN cfn drop: match-consume applies even when
	// payload-less (the analyzer/codegen gates must agree — found by
	// the rustc-comparison fleet), and binding the payload out from
	// under the enum's own drop is rejected (Rust's E0509).
	static void testMatchMoveOwnDropEnum() {
		auto twice = compileSource("matchmove_owndrop_twice", R"(
extern fn puts(s: *const[] u8) i32;
const Token = enum {
    Red,
    Blue,
};
cfn drop(self: mut Token) {
    puts("D");
}
fn bad() u32 {
    var t: Token = Token.Red();
    match (t) { _ { } }
    match (t) { _ { return 1; } }
    return 0;
}
fn main() {}
)");
		ASSERT_TRUE(twice.exitCode != 0);
		ASSERT_TRUE(stderrContains(twice, "use of moved binding `t`"));

		auto bind = compileSource("matchmove_owndrop_bind", R"(
extern fn puts(s: *const[] u8) i32;
const Wrapped = enum {
    None,
    Some(u32),
};
cfn drop(self: mut Wrapped) {
    puts("D");
}
fn bad() u32 {
    var w: Wrapped = Wrapped.Some(5);
    match (w) {
        Wrapped.Some(x) { return x; }
        _ { return 0; }
    }
}
fn main() {}
)");
		ASSERT_TRUE(bind.exitCode != 0);
		ASSERT_TRUE(stderrContains(
		    bind, "cannot bind the payload out of an enum that has its "
		          "own `cfn drop`"));
	}

	// Call sites are sigil-free for parameter modes; `&` is address-of
	// (a pointer value) and only valid where the parameter's TYPE is a
	// pointer. `&x` on a mode parameter is rejected with guidance.
	static void testAddrOfOnModeArgRejected() {
		auto r = compileSource("addrof_on_mode_arg", R"(
fn bump(x: mut u32) {
    x = x + 1;
}
fn main() {
    var n: u32 = 1;
    bump(&n);
}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "`&` makes a pointer"));
		ASSERT_TRUE(stderrContains(r, "pass it plainly"));
	}
};

int main() {
	TestFramework framework;
	CodegenErrorTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
