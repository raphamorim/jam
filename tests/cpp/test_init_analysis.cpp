// In-process tests for the MVS init analyzer.
//
// Each test compiles a Jam source string through lexer + parser, runs
// init_analysis::analyze on every function in the parsed module, and
// asserts on the returned Diagnostic vector. No fork+exec, no stderr
// scraping — the analyzer's structured output is checked directly.
//
// This complements the .jam-based must-pass suite under tests/unit/ by
// covering the rejected-program cases that require precise diagnostic
// shape (count, message substring, varName) rather than a binary
// "did the compile fail?" oracle.

#include "../../src/ast.h"
#include "../../src/ast_flat.h"
#include "../../src/init_analysis.h"
#include "../../src/lexer.h"
#include "../../src/parser.h"
#include "test_framework.h"

#include <memory>
#include <string>
#include <vector>

namespace {

// One run's outputs: the diagnostics from every analyzed function plus
// the supporting state kept alive long enough for the diagnostics to
// reference (the strings live in StringPool, etc.).
struct AnalyzeResult {
	std::unique_ptr<NodeStore> nodes;
	std::unique_ptr<StringPool> strings;
	std::unique_ptr<TypePool> types;
	std::unique_ptr<ModuleAST> module;
	std::vector<jam::init_analysis::Diagnostic> diagnostics;
};

AnalyzeResult analyzeSource(const std::string &src) {
	AnalyzeResult result;
	result.nodes = std::make_unique<NodeStore>();
	result.strings = std::make_unique<StringPool>();
	result.types = std::make_unique<TypePool>();

	Lexer lexer(src);
	auto tokens = lexer.scanTokens();

	Parser parser(tokens, lexer.sourceBuffer(), *result.types, *result.strings,
	              *result.nodes);
	result.module = parser.parse();

	jam::init_analysis::FunctionRegistry registry;
	for (auto &fn : result.module->Functions) { registry[fn->Name] = fn.get(); }

	jam::drops::DropRegistry drops = jam::drops::buildDropRegistry(
	    *result.module, *result.types, *result.strings);

	for (auto &fn : result.module->Functions) {
		if (fn->isExtern) continue;
		auto diags = jam::init_analysis::analyze(
		    *fn, *result.nodes, *result.strings, tokens, &registry, &drops,
		    result.types.get());
		for (auto &d : diags) result.diagnostics.push_back(std::move(d));
	}
	return result;
}

bool diagsContain(const std::vector<jam::init_analysis::Diagnostic> &diags,
                  const std::string &msgSubstring) {
	for (const auto &d : diags) {
		if (d.message.find(msgSubstring) != std::string::npos) return true;
	}
	return false;
}

bool diagsAbout(const std::vector<jam::init_analysis::Diagnostic> &diags,
                const std::string &varName) {
	for (const auto &d : diags) {
		if (d.varName == varName) return true;
	}
	return false;
}

// Callsite mode propagation: how a `move` arg drains the caller's binding.

void testReadAfterMove() {
	auto r = analyzeSource(R"(
fn consume(buf: move u32) u32 { return buf; }

fn caller() u32 {
    var x: u32 = 100;
    consume(x);
    return x;
}
)");
	ASSERT_EQ(static_cast<size_t>(1), r.diagnostics.size());
	ASSERT_CONTAINS(r.diagnostics[0].message, "uninitialized");
	ASSERT_EQ(std::string("x"), r.diagnostics[0].varName);
}

void testDoubleMove() {
	auto r = analyzeSource(R"(
fn consume(buf: move u32) u32 { return buf; }

fn caller() u32 {
    var x: u32 = 100;
    consume(x);
    return consume(x);
}
)");
	ASSERT_EQ(static_cast<size_t>(1), r.diagnostics.size());
	ASSERT_CONTAINS(r.diagnostics[0].message, "uninitialized");
	ASSERT_EQ(std::string("x"), r.diagnostics[0].varName);
}

void testMoveThenSeparateBindingOK() {
	auto r = analyzeSource(R"(
fn consume(buf: move u32) u32 { return buf; }

fn caller() u32 {
    var x: u32 = 5;
    consume(x);
    var y: u32 = 10;
    return y;
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

// Exclusivity rule: at most one `mut` borrow per path, no `mut` + `let`
// or `mut` + `move` simultaneously, no overlapping field paths.

void testExclusivityMutLet() {
	auto r = analyzeSource(R"(
fn modify(x: mut u32, y: u32) u32 {
    x = x + y;
    return x;
}

fn caller() u32 {
    var n: u32 = 5;
    return modify(n, n);
}
)");
	ASSERT_TRUE(diagsContain(r.diagnostics, "conflicting borrows"));
	ASSERT_TRUE(diagsAbout(r.diagnostics, "n"));
}

void testExclusivityTwoMoves() {
	auto r = analyzeSource(R"(
fn consumeTwo(a: move u32, b: move u32) u32 { return a + b; }

fn caller() u32 {
    var x: u32 = 5;
    return consumeTwo(x, x);
}
)");
	ASSERT_TRUE(diagsContain(r.diagnostics, "conflicting borrows"));
	ASSERT_TRUE(diagsAbout(r.diagnostics, "x"));
}

void testExclusivityOverlappingPath() {
	auto r = analyzeSource(R"(
const Pair = struct { a: u32, b: u32 };

fn modifyAndRead(whole: mut Pair, part: u32) u32 { return part; }

fn caller() u32 {
    var p: Pair = Pair { a: 1, b: 2 };
    return modifyAndRead(p, p.a);
}
)");
	ASSERT_TRUE(diagsContain(r.diagnostics, "conflicting borrows"));
	ASSERT_TRUE(diagsAbout(r.diagnostics, "p"));
}

void testExclusivityDisjointFieldsOK() {
	auto r = analyzeSource(R"(
const Pair = struct { a: u32, b: u32 };

fn add(a: u32, b: u32) u32 { return a + b; }

fn caller() u32 {
    var p: Pair = Pair { a: 10, b: 20 };
    return add(p.a, p.b);
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

// Scope-escape check: `&mut` to a function-local can't outlive the
// frame via a return / out-param.

void testEscapeMutParam() {
	auto r = analyzeSource(R"(
fn dangle(p: mut u32) *mut u32 {
    return &p;
}
)");
	ASSERT_EQ(static_cast<size_t>(1), r.diagnostics.size());
	ASSERT_CONTAINS(r.diagnostics[0].message,
	                "cannot return `&` of `mut`-mode");
	ASSERT_EQ(std::string("p"), r.diagnostics[0].varName);
}

void testEscapeMutField() {
	auto r = analyzeSource(R"(
const Pair = struct { a: u32, b: u32 };

fn dangleField(p: mut Pair) *mut u32 {
    return &p.a;
}
)");
	ASSERT_EQ(static_cast<size_t>(1), r.diagnostics.size());
	ASSERT_CONTAINS(r.diagnostics[0].message,
	                "cannot return `&` of `mut`-mode");
}

void testReturnMutParamByValueOK() {
	// `return p;` (no `&`) is a value copy under value semantics — not an
	// escape. The check must NOT flag this.
	auto r = analyzeSource(R"(
fn doubleIt(x: mut u32) u32 {
    x = x + x;
    return x;
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

// Drop registry interaction: `move` on a drop-bearing binding is the
// hazard the analyzer prevents (without move-aware drop tracking the
// codegen would auto-fire drop on a moved-out slot — double-free).

void testMoveOnDropBearingRejected() {
	// A type with a user-defined `cfn drop(self: mut T)` is "drop-
	// bearing" — `cfn` is the explicit opt-in that hands the
	// destructor call to the compiler's auto-fire path (see
	// drop_registry.cpp's `considerDropCandidate`). The analyzer
	// rejects `move` on a drop-bearing binding so the codegen can't
	// emit drop on a moved-out slot.
	auto r = analyzeSource(R"(
const File = struct {
    fd: i32,
};

cfn drop(self: mut File) {
    self.fd = 0;
}

fn consume(f: move File) i32 {
    return f.fd;
}

fn caller() i32 {
    var f: File = File { fd: 7 };
    return consume(f);
}
)");
	ASSERT_TRUE(diagsContain(r.diagnostics, "drop-bearing type"));
	ASSERT_TRUE(diagsAbout(r.diagnostics, "f"));
}

void testLetOnDropBearingOK() {
	// Passing a drop-bearing binding by `let` (read-only borrow, default)
	// or `mut` is fine — only `move` is rejected.
	auto r = analyzeSource(R"(
const File = struct {
    fd: i32,
};

fn drop(self: mut File) {
    self.fd = 0;
}

fn read(f: File) i32 {
    return f.fd;
}

fn caller() i32 {
    var f: File = File { fd: 7 };
    return read(f);
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

void testMutOnDropBearingOK() {
	auto r = analyzeSource(R"(
const File = struct {
    fd: i32,
};

fn drop(self: mut File) {
    self.fd = 0;
}

fn close(f: mut File) {
    f.fd = -1;
}

fn caller() i32 {
    var f: File = File { fd: 7 };
    close(&f);
    return f.fd;
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

void testNonDropBearingMoveOK() {
	// Non-drop-bearing struct can still be moved freely (no double-free
	// risk because no drop runs at scope exit).
	auto r = analyzeSource(R"(
const Plain = struct {
    a: u32,
    b: u32,
};

fn consume(p: move Plain) u32 {
    return p.a + p.b;
}

fn caller() u32 {
    var p: Plain = Plain { a: 1, b: 2 };
    return consume(p);
}
)");
	ASSERT_EQ(static_cast<size_t>(0), r.diagnostics.size());
}

}  // namespace

class InitAnalysisTests {
  public:
	static void registerAllTests(TestFramework &framework) {
		// Callsite mode propagation
		framework.addTest("InitAnalysis - read after move", testReadAfterMove);
		framework.addTest("InitAnalysis - double move", testDoubleMove);
		framework.addTest("InitAnalysis - move then separate binding OK",
		                  testMoveThenSeparateBindingOK);

		// Exclusivity
		framework.addTest("InitAnalysis - mut + let same binding",
		                  testExclusivityMutLet);
		framework.addTest("InitAnalysis - two moves same binding",
		                  testExclusivityTwoMoves);
		framework.addTest("InitAnalysis - overlapping path",
		                  testExclusivityOverlappingPath);
		framework.addTest("InitAnalysis - disjoint fields OK",
		                  testExclusivityDisjointFieldsOK);

		// Scope escape
		framework.addTest("InitAnalysis - escape &mut param",
		                  testEscapeMutParam);
		framework.addTest("InitAnalysis - escape &mut field",
		                  testEscapeMutField);
		framework.addTest("InitAnalysis - mut param by-value return OK",
		                  testReturnMutParamByValueOK);

		// Drop registry interaction
		framework.addTest("InitAnalysis - move on drop-bearing rejected",
		                  testMoveOnDropBearingRejected);
		framework.addTest("InitAnalysis - let on drop-bearing OK",
		                  testLetOnDropBearingOK);
		framework.addTest("InitAnalysis - mut on drop-bearing OK",
		                  testMutOnDropBearingOK);
		framework.addTest("InitAnalysis - move on non-drop OK",
		                  testNonDropBearingMoveOK);
	}
};

int main() {
	TestFramework framework;
	InitAnalysisTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
