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

	// Redirect stderr→stdout so popen captures both. jam.out usually
	// only writes to stderr on error, but this is robust either way.
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
	std::string cmd = "./jam.out " + mainPath + " 2>&1";

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
		framework.addTest("Codegen - default() with parameters rejected",
		                  testDefaultWithParameters);
		framework.addTest("Codegen - default() with wrong return type",
		                  testDefaultWrongReturnType);
		framework.addTest("Codegen - non-drop non-default method on top-level",
		                  testForbiddenTopLevelMethod);
		framework.addTest("Codegen - int literal in float-typed destination",
		                  testIntToFloatRejected);
		framework.addTest("Codegen - mixed-width float binary op without cast",
		                  testMixedFloatWidthRejected);
		framework.addTest("Codegen - destructured non-pub symbol rejected",
		                  testDestructuredNonPubRejected);
		framework.addTest("Codegen - `pub` on import rejected",
		                  testPubOnImportRejected);
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

	// `default` on a top-level struct must take no parameters. The
	// validation in main.cpp specifically checks Args.empty().
	static void testDefaultWithParameters() {
		auto r = compileSource("must_fail_default_with_params", R"(
const Bad = struct {
    n: i32,
    fn default(self: mut Self) Self {
        return Self { n: 0 };
    }
};

fn main() i32 { return 0; }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "default"));
		ASSERT_TRUE(stderrContains(r, "no parameters"));
	}

	// `default` must return Self (the enclosing struct's type).
	// Returning anything else is a typing error in the contract.
	static void testDefaultWrongReturnType() {
		auto r = compileSource("must_fail_default_wrong_return", R"(
const Bad = struct {
    n: i32,
    fn default() i32 {
        return 0;
    }
};

fn main() i32 { return 0; }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "default"));
		ASSERT_TRUE(stderrContains(r, "Self"));
	}

	// Float-typed destinations need a float literal (`3.0`) or an
	// explicit `as` cast. Implicit int→float coercion is rejected so
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

	// Top-level structs only allow `drop` and `default` methods. Other
	// names (e.g. `unwrap`) get a clear "not allowed" error so users
	// don't think method-as-namespace works on plain structs.
	static void testForbiddenTopLevelMethod() {
		auto r = compileSource("must_fail_other_method", R"(
const Bad = struct {
    n: i32,
    fn unwrap(self: mut Self) i32 {
        return self.n;
    }
};

fn main() i32 { return 0; }
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "drop"));
		ASSERT_TRUE(stderrContains(r, "default"));
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

	// `pub const x = import(...)` is rejected; re-exports aren't a
	// feature yet.
	static void testPubOnImportRejected() {
		auto r = compileSource("must_fail_pub_import", R"(
pub const std = import("std");
fn main() {}
)");
		ASSERT_TRUE(r.exitCode != 0);
		ASSERT_TRUE(stderrContains(r, "pub"));
		ASSERT_TRUE(stderrContains(r, "imports"));
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
};

int main() {
	TestFramework framework;
	CodegenErrorTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
