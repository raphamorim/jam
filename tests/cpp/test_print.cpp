// End-to-end tests for the @-emit + cfn-based print pipeline.
// Each test compiles a small Jam program that uses `fmt.print` /
// `fmt.eprint`, runs the resulting binary with stdout + stderr
// redirected to separate files, and asserts on the captured bytes.
//
// stdout/stderr separation lets us verify that `@eprint` actually
// reaches fd=2 — important because the @-emit dispatch passes the fd
// as a comp-known literal and we want to catch any wiring mistake
// that lands stderr output on stdout (or vice versa).

#include "test_framework.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace {

struct RunResult {
	// Exit code from the compile step. -1 means compile died via signal.
	int compileExit;
	// Exit code from running the compiled binary. -1 if compile failed
	// and we never ran the binary.
	int runExit;
	// Captured streams. `compileOut` is the compile step's combined
	// stdout+stderr; `stdout_` and `stderr_` are the runtime streams.
	std::string compileOut;
	std::string stdout_;
	std::string stderr_;
};

// Slurp a whole file into a std::string. Used to read the captured
// stdout/stderr from /tmp after the binary runs.
std::string readFile(const std::string &path) {
	std::ifstream in(path);
	std::stringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

// Compile `source` to a binary and run it. Captures the runtime
// stdout/stderr separately via shell redirects so we can assert on
// each independently.
RunResult compileAndRun(const std::string &name, const std::string &source) {
	std::string srcPath = "/tmp/jam_print_" + name + ".jam";
	std::string binPath = "/tmp/jam_print_" + name + ".bin";
	std::string compileLog = "/tmp/jam_print_" + name + ".compile";
	std::string stdoutFile = "/tmp/jam_print_" + name + ".out";
	std::string stderrFile = "/tmp/jam_print_" + name + ".err";

	{
		std::ofstream out(srcPath);
		out << source;
	}

	std::string compileCmd = "./output/jam.out -o " + binPath + " " + srcPath +
	                           " >" + compileLog + " 2>&1";
	int compileStatus = std::system(compileCmd.c_str());
	int compileExit =
	    WIFEXITED(compileStatus) ? WEXITSTATUS(compileStatus) : -1;
	std::string compileOut = readFile(compileLog);
	if (compileExit != 0) {
		return {compileExit, -1, std::move(compileOut), "", ""};
	}

	std::string runCmd =
	    binPath + " >" + stdoutFile + " 2>" + stderrFile;
	int runStatus = std::system(runCmd.c_str());
	int runExit = WIFEXITED(runStatus) ? WEXITSTATUS(runStatus) : -1;
	std::string stdoutContent = readFile(stdoutFile);
	std::string stderrContent = readFile(stderrFile);
	return {0, runExit, std::move(compileOut), std::move(stdoutContent),
	        std::move(stderrContent)};
}

// ── Per-built-in-type tests ──────────────────────────────────────

void testPrintIntI32() {
	auto r = compileAndRun("int_i32", R"(
const fmt = import("fmt");
fn main() i32 {
    var x: i32 = 42;
    fmt.print("x={x}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("x=42\n"), r.stdout_);
	ASSERT_EQ(std::string(""), r.stderr_);
}

void testPrintIntI64Negative() {
	auto r = compileAndRun("int_i64_neg", R"(
const fmt = import("fmt");
fn main() i32 {
    var x: i64 = -123;
    fmt.print("{x}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("-123\n"), r.stdout_);
}

void testPrintIntU64() {
	auto r = compileAndRun("int_u64", R"(
const fmt = import("fmt");
fn main() i32 {
    var x: u64 = 9999999999;
    fmt.print("{x}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("9999999999\n"), r.stdout_);
}

void testPrintBoolTrue() {
	auto r = compileAndRun("bool_true", R"(
const fmt = import("fmt");
fn main() i32 {
    var t: bool = true;
    fmt.print("{t}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("true\n"), r.stdout_);
}

void testPrintBoolFalse() {
	auto r = compileAndRun("bool_false", R"(
const fmt = import("fmt");
fn main() i32 {
    var f: bool = false;
    fmt.print("{f}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("false\n"), r.stdout_);
}

void testPrintStr() {
	auto r = compileAndRun("str", R"(
const fmt = import("fmt");
fn main() i32 {
    var name: str = "Alice";
    fmt.print("hello, {name}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("hello, Alice\n"), r.stdout_);
}

// ── Multi-interp + escapes ───────────────────────────────────────

void testPrintMultipleInterps() {
	auto r = compileAndRun("multi", R"(
const fmt = import("fmt");
fn main() i32 {
    var name: str = "Bob";
    var age: i32 = 30;
    var ok: bool = true;
    fmt.print("{name} is {age} (active={ok})\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("Bob is 30 (active=true)\n"), r.stdout_);
}

void testBraceEscapeOpen() {
	auto r = compileAndRun("brace_open", R"(
const fmt = import("fmt");
fn main() i32 {
    fmt.print("set is {{here\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("set is {here\n"), r.stdout_);
}

void testBraceEscapeClose() {
	auto r = compileAndRun("brace_close", R"(
const fmt = import("fmt");
fn main() i32 {
    fmt.print("done}}here\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("done}here\n"), r.stdout_);
}

void testBraceEscapesMixedWithInterp() {
	auto r = compileAndRun("brace_mixed", R"(
const fmt = import("fmt");
fn main() i32 {
    var n: i32 = 3;
    fmt.print("set is {{1, 2, {n}}}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("set is {1, 2, 3}\n"), r.stdout_);
}

// ── stderr routing ───────────────────────────────────────────────

void testEprintGoesToStderr() {
	auto r = compileAndRun("eprint", R"(
const fmt = import("fmt");
fn main() i32 {
    fmt.print("on stdout\n");
    fmt.eprint("on stderr\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("on stdout\n"), r.stdout_);
	ASSERT_EQ(std::string("on stderr\n"), r.stderr_);
}

void testEprintInterpAndRouting() {
	auto r = compileAndRun("eprint_interp", R"(
const fmt = import("fmt");
fn main() i32 {
    var code: i32 = 42;
    var msg: str = "boom";
    fmt.eprint("error code={code} msg={msg}\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string(""), r.stdout_);
	ASSERT_EQ(std::string("error code=42 msg=boom\n"), r.stderr_);
}

// ── Nested-module access shapes ──────────────────────────────────
// `std/std.jam` re-exports each std module via `pub const X =
// import(...)`. These tests pin the three user-visible forms for
// reaching `print` through that re-export so a regression in
// chain resolution surfaces as a missed printout.

void testStdChainedAccess() {
	auto r = compileAndRun("std_chain", R"(
const std = import("std");
fn main() i32 {
    std.fmt.print("std.fmt.print\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("std.fmt.print\n"), r.stdout_);
}

void testStdDestructureFromChain() {
	auto r = compileAndRun("std_destruct_chain", R"(
const { print } = import("std").fmt;
fn main() i32 {
    print("destructured from std.fmt\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("destructured from std.fmt\n"), r.stdout_);
}

// Pins literal-batching shape: a long literal run between two
// placeholders, plus leading-only / trailing-only-placeholder
// shapes. The cfn collapses each contiguous literal span into a
// single `@emitWriteBytes` -> dprintf("%.*s", …) call. Run-time
// correctness is the visible signal; the IR-level reduction is a
// secondary win covered by inspection of `--emit-ir`.

void testLongLiteralRunBatched() {
	auto r = compileAndRun("long_run", R"(
const fmt = import("std").fmt;
fn main() i32 {
    var x: i32 = 42;
    fmt.print("hello, world! the answer is {x}, have a nice day\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string(
	             "hello, world! the answer is 42, have a nice day\n"),
	          r.stdout_);
}

void testLeadingPlaceholderOnly() {
	auto r = compileAndRun("lead_ph", R"(
const fmt = import("std").fmt;
fn main() i32 {
    var name: str = "Bob";
    fmt.print("{name} is here\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("Bob is here\n"), r.stdout_);
}

void testTrailingPlaceholderOnly() {
	auto r = compileAndRun("trail_ph", R"(
const fmt = import("std").fmt;
fn main() i32 {
    var n: i32 = 7;
    fmt.print("count: {n}");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("count: 7"), r.stdout_);
}

void testStdDestructureModule() {
	auto r = compileAndRun("std_destruct_mod", R"(
const { fmt } = import("std");
fn main() i32 {
    fmt.print("fmt destructured from std\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("fmt destructured from std\n"), r.stdout_);
}

// ── Negative tests: compile-time errors ──────────────────────────

void testUnknownIdentInInterpFails() {
	auto r = compileAndRun("unknown_ident", R"(
const fmt = import("fmt");
fn main() i32 {
    fmt.print("hello {nope}\n");
    return 0;
}
)");
	ASSERT_TRUE(r.compileExit != 0);
	ASSERT_TRUE(r.compileOut.find("nope") != std::string::npos);
}

void testMixedTypeIntsAndStr() {
	auto r = compileAndRun("mixed", R"(
const fmt = import("fmt");
fn main() i32 {
    var label: str = "count";
    var n: u32 = 7;
    var flag: bool = false;
    fmt.print("{label}: {n} ({flag})\n");
    return 0;
}
)");
	ASSERT_EQ(0, r.runExit);
	ASSERT_EQ(std::string("count: 7 (false)\n"), r.stdout_);
}

}  // namespace

int main() {
	TestFramework framework;
	framework.addTest("Print - i32", testPrintIntI32);
	framework.addTest("Print - i64 negative", testPrintIntI64Negative);
	framework.addTest("Print - u64", testPrintIntU64);
	framework.addTest("Print - bool true", testPrintBoolTrue);
	framework.addTest("Print - bool false", testPrintBoolFalse);
	framework.addTest("Print - str", testPrintStr);
	framework.addTest("Print - multiple interps", testPrintMultipleInterps);
	framework.addTest("Print - {{ escape", testBraceEscapeOpen);
	framework.addTest("Print - }} escape", testBraceEscapeClose);
	framework.addTest("Print - escapes + interp mixed",
	                  testBraceEscapesMixedWithInterp);
	framework.addTest("Print - eprint routes to stderr",
	                  testEprintGoesToStderr);
	framework.addTest("Print - eprint interp + stderr routing",
	                  testEprintInterpAndRouting);
	framework.addTest("Print - unknown identifier fails compile",
	                  testUnknownIdentInInterpFails);
	framework.addTest("Print - mixed types (str + u32 + bool)",
	                  testMixedTypeIntsAndStr);
	framework.addTest("Print - std.fmt.print (chained access)",
	                  testStdChainedAccess);
	framework.addTest("Print - const { print } = import(\"std\").fmt",
	                  testStdDestructureFromChain);
	framework.addTest("Print - const { fmt } = import(\"std\")",
	                  testStdDestructureModule);
	framework.addTest("Print - long literal run batched",
	                  testLongLiteralRunBatched);
	framework.addTest("Print - leading placeholder only",
	                  testLeadingPlaceholderOnly);
	framework.addTest("Print - trailing placeholder only",
	                  testTrailingPlaceholderOnly);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
