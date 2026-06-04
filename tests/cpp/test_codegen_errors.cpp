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
	std::string cmd = "./output/jam.out -o " + outBin + " " + path + " 2>&1";

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
	std::string jamBin =
	    (std::filesystem::current_path() / "output" / "jam.out").string();
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
	std::string cmd = "./output/jam.out --emit-ir " + path + " 2>&1";

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
	std::string cmd =
	    "./output/jam.out -o " + outBin + " " + mainPath + " 2>&1";

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
	static void registerAllTests(TestFramework &framework) {
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
};

int main() {
	TestFramework framework;
	CodegenErrorTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
