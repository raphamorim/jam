// Unit tests for src/comptime.h — the comp-time expression evaluator.
//
// Each test builds a small AST by hand (NodeStore + StringPool +
// TypePool), invokes ComptimeEvaluator::eval, and asserts on the
// returned ComptimeValue. No subprocess fork; the evaluator runs in
// the test's address space.

#include "../../src/ast_flat.h"
#include "../../src/comptime.h"
#include "test_framework.h"

#include <cstdint>

namespace {

// Helper: emit a NumberLit AstNode for `n`.
NodeIdx mkInt(NodeStore &ns, uint64_t n, bool isNeg = false) {
	AstNode node{};
	node.tag = AstTag::NumberLit;
	node.lhs = static_cast<uint32_t>(n & 0xFFFFFFFFu);
	node.rhs = static_cast<uint32_t>(n >> 32);
	if (isNeg) node.flags |= 1;
	return ns.addNode(node);
}

NodeIdx mkBool(NodeStore &ns, bool b) {
	AstNode node{};
	node.tag = AstTag::BoolLit;
	node.lhs = b ? 1u : 0u;
	return ns.addNode(node);
}

NodeIdx mkStr(NodeStore &ns, StringPool &sp, const std::string &s) {
	AstNode node{};
	node.tag = AstTag::StringLit;
	node.lhs = static_cast<uint32_t>(sp.intern(s));
	return ns.addNode(node);
}

NodeIdx mkVar(NodeStore &ns, StringPool &sp, const std::string &name) {
	AstNode node{};
	node.tag = AstTag::Variable;
	node.lhs = static_cast<uint32_t>(sp.intern(name));
	return ns.addNode(node);
}

NodeIdx mkBinOp(NodeStore &ns, BinOp op, NodeIdx l, NodeIdx r) {
	AstNode node{};
	node.tag = AstTag::BinaryOp;
	node.op = static_cast<uint8_t>(op);
	node.lhs = static_cast<uint32_t>(l);
	node.rhs = static_cast<uint32_t>(r);
	return ns.addNode(node);
}

NodeIdx mkUnaryOp(NodeStore &ns, UnaryOp op, NodeIdx operand) {
	AstNode node{};
	node.tag = AstTag::UnaryOp;
	node.op = static_cast<uint8_t>(op);
	node.lhs = static_cast<uint32_t>(operand);
	return ns.addNode(node);
}

NodeIdx mkIndex(NodeStore &ns, NodeIdx base, NodeIdx idx) {
	AstNode node{};
	node.tag = AstTag::Index;
	node.lhs = static_cast<uint32_t>(base);
	node.rhs = static_cast<uint32_t>(idx);
	return ns.addNode(node);
}

// ── Literal folds ────────────────────────────────────────────────

void testNumberLitFoldsToInt() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	NodeIdx n = mkInt(ns, 42);
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::ComptimeValue v = e.eval(n, scope);
	ASSERT_TRUE(v.isInt());
	ASSERT_EQ(static_cast<uint64_t>(42), v.asU64());
}

void testBoolLitFolds() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	NodeIdx t = mkBool(ns, true);
	NodeIdx f = mkBool(ns, false);
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::ComptimeValue tv = e.eval(t, scope);
	jam::ComptimeValue fv = e.eval(f, scope);
	ASSERT_TRUE(tv.isBool());
	ASSERT_TRUE(tv.boolVal);
	ASSERT_TRUE(fv.isBool());
	ASSERT_FALSE(fv.boolVal);
}

void testStringLitFolds() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	NodeIdx s = mkStr(ns, sp, "hello");
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::ComptimeValue v = e.eval(s, scope);
	ASSERT_TRUE(v.isStr());
	ASSERT_EQ(std::string("hello"), sp.get(v.strVal));
}

// ── Scope lookups ────────────────────────────────────────────────

void testVariableLookupFromScope() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	NodeIdx v = mkVar(ns, sp, "x");
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	scope.bind("x", jam::ComptimeValue::makeInt(99, 32, true));
	jam::ComptimeValue r = e.eval(v, scope);
	ASSERT_TRUE(r.isInt());
	ASSERT_EQ(static_cast<uint64_t>(99), r.asU64());
}

void testVariableMissingReturnsNone() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	NodeIdx v = mkVar(ns, sp, "missing");
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::ComptimeValue r = e.eval(v, scope);
	ASSERT_TRUE(r.isNone());
}

void testNestedScopeFallsThroughToParent() {
	jam::ComptimeScope parent;
	parent.bind("x", jam::ComptimeValue::makeInt(7, 64, false));
	jam::ComptimeScope child(&parent);
	child.bind("y", jam::ComptimeValue::makeInt(11, 64, false));

	const jam::ComptimeValue *x = child.lookup("x");
	const jam::ComptimeValue *y = child.lookup("y");
	const jam::ComptimeValue *z = child.lookup("z");
	ASSERT_TRUE(x != nullptr);
	ASSERT_EQ(static_cast<uint64_t>(7), x->asU64());
	ASSERT_TRUE(y != nullptr);
	ASSERT_EQ(static_cast<uint64_t>(11), y->asU64());
	ASSERT_TRUE(z == nullptr);
}

// ── Arithmetic ───────────────────────────────────────────────────

void testIntAddSubMul() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx a = mkInt(ns, 10);
	NodeIdx b = mkInt(ns, 3);
	ASSERT_EQ(static_cast<uint64_t>(13),
	          e.eval(mkBinOp(ns, BinOp::Add, a, b), scope).asU64());
	ASSERT_EQ(static_cast<uint64_t>(7),
	          e.eval(mkBinOp(ns, BinOp::Sub, a, b), scope).asU64());
	ASSERT_EQ(static_cast<uint64_t>(30),
	          e.eval(mkBinOp(ns, BinOp::Mul, a, b), scope).asU64());
}

void testIntDivByZeroIsNone() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx a = mkInt(ns, 10);
	NodeIdx b = mkInt(ns, 0);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Div, a, b), scope).isNone());
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Mod, a, b), scope).isNone());
}

void testIntComparisons() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx a = mkInt(ns, 5);
	NodeIdx b = mkInt(ns, 7);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Lt, a, b), scope).boolVal);
	ASSERT_FALSE(e.eval(mkBinOp(ns, BinOp::Gt, a, b), scope).boolVal);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Ne, a, b), scope).boolVal);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Eq, a, a), scope).boolVal);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Le, a, a), scope).boolVal);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Ge, b, a), scope).boolVal);
}

void testBitwise() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx a = mkInt(ns, 0b1100);
	NodeIdx b = mkInt(ns, 0b1010);
	ASSERT_EQ(static_cast<uint64_t>(0b1000),
	          e.eval(mkBinOp(ns, BinOp::BitAnd, a, b), scope).asU64());
	ASSERT_EQ(static_cast<uint64_t>(0b1110),
	          e.eval(mkBinOp(ns, BinOp::BitOr, a, b), scope).asU64());
	ASSERT_EQ(static_cast<uint64_t>(0b0110),
	          e.eval(mkBinOp(ns, BinOp::BitXor, a, b), scope).asU64());
}

// ── Logical operators (short-circuit) ────────────────────────────

void testLogAndShortCircuits() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx f = mkBool(ns, false);
	// RHS references missing variable — would return None if eval'd.
	NodeIdx missing = mkVar(ns, sp, "z");
	jam::ComptimeValue v =
	    e.eval(mkBinOp(ns, BinOp::LogAnd, f, missing), scope);
	ASSERT_TRUE(v.isBool());
	ASSERT_FALSE(v.boolVal);  // LHS=false short-circuits, never touched RHS
}

void testLogOrShortCircuits() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx t = mkBool(ns, true);
	NodeIdx missing = mkVar(ns, sp, "z");
	jam::ComptimeValue v = e.eval(mkBinOp(ns, BinOp::LogOr, t, missing), scope);
	ASSERT_TRUE(v.isBool());
	ASSERT_TRUE(v.boolVal);
}

// ── Unary ────────────────────────────────────────────────────────

void testUnaryNegateInt() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx five = mkInt(ns, 5);
	jam::ComptimeValue v = e.eval(mkUnaryOp(ns, UnaryOp::Neg, five), scope);
	ASSERT_TRUE(v.isInt());
	// asI64 must sign-extend from the value's width.
	ASSERT_EQ(static_cast<int64_t>(-5), v.asI64());
}

void testUnaryLogNotBool() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx t = mkBool(ns, true);
	jam::ComptimeValue v = e.eval(mkUnaryOp(ns, UnaryOp::LogNot, t), scope);
	ASSERT_TRUE(v.isBool());
	ASSERT_FALSE(v.boolVal);
}

// ── String indexing ──────────────────────────────────────────────

void testStringIndexReturnsByte() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx s = mkStr(ns, sp, "abc");
	NodeIdx i = mkInt(ns, 1);
	jam::ComptimeValue v = e.eval(mkIndex(ns, s, i), scope);
	ASSERT_TRUE(v.isInt());
	ASSERT_EQ(static_cast<uint64_t>('b'), v.asU64());
}

void testStringIndexOutOfBoundsIsNone() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx s = mkStr(ns, sp, "abc");
	NodeIdx i = mkInt(ns, 99);
	jam::ComptimeValue v = e.eval(mkIndex(ns, s, i), scope);
	ASSERT_TRUE(v.isNone());
}

// ── Type equality (the @TypeOf-comparison primitive) ────────────

void testTypeEquality() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	scope.bind("T", jam::ComptimeValue::makeType(BuiltinType::I32));
	scope.bind("U", jam::ComptimeValue::makeType(BuiltinType::I32));
	scope.bind("V", jam::ComptimeValue::makeType(BuiltinType::F32));

	NodeIdx t = mkVar(ns, sp, "T");
	NodeIdx u = mkVar(ns, sp, "U");
	NodeIdx v = mkVar(ns, sp, "V");
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Eq, t, u), scope).boolVal);
	ASSERT_FALSE(e.eval(mkBinOp(ns, BinOp::Eq, t, v), scope).boolVal);
	ASSERT_TRUE(e.eval(mkBinOp(ns, BinOp::Ne, t, v), scope).boolVal);
}

// ── Aggregate indexing (for tuples in Phase 4) ──────────────────

void testAggregateIndexing() {
	std::vector<jam::ComptimeValue> fields;
	fields.push_back(jam::ComptimeValue::makeInt(10, 32, true));
	fields.push_back(jam::ComptimeValue::makeInt(20, 32, true));
	fields.push_back(jam::ComptimeValue::makeInt(30, 32, true));
	jam::ComptimeValue agg =
	    jam::ComptimeValue::makeAggregate(std::move(fields));

	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	scope.bind("t", agg);

	NodeIdx t = mkVar(ns, sp, "t");
	NodeIdx i = mkInt(ns, 1);
	jam::ComptimeValue v = e.eval(mkIndex(ns, t, i), scope);
	ASSERT_TRUE(v.isInt());
	ASSERT_EQ(static_cast<uint64_t>(20), v.asU64());
}

// ── Required-fold diagnostics ───────────────────────────────────

void testEvalRequiredPushesDiagnosticOnFail() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx miss = mkVar(ns, sp, "missing");
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	jam::ComptimeValue v = e.evalRequired(miss, scope, diags, loc);
	ASSERT_TRUE(v.isNone());
	ASSERT_TRUE(diags.hasErrors());
	ASSERT_EQ(static_cast<std::size_t>(1), diags.errorCount());
}

void testEvalRequiredSilentOnSuccess() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	NodeIdx n = mkInt(ns, 42);
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	jam::ComptimeValue v = e.evalRequired(n, scope, diags, loc);
	ASSERT_TRUE(v.isInt());
	ASSERT_FALSE(diags.hasErrors());
}

// ── Mutable scope ────────────────────────────────────────────────

void testScopeSetMutatesExistingBinding() {
	jam::ComptimeScope scope;
	scope.bind("x", jam::ComptimeValue::makeInt(5, 32, true));
	bool ok = scope.set("x", jam::ComptimeValue::makeInt(10, 32, true));
	ASSERT_TRUE(ok);
	ASSERT_EQ(static_cast<uint64_t>(10), scope.lookup("x")->asU64());
}

void testScopeSetFailsForUnknownName() {
	jam::ComptimeScope scope;
	bool ok = scope.set("nope", jam::ComptimeValue::makeBool(false));
	ASSERT_FALSE(ok);
}

void testScopeSetWalksUpToParent() {
	jam::ComptimeScope parent;
	parent.bind("x", jam::ComptimeValue::makeInt(5, 32, true));
	jam::ComptimeScope child(&parent);
	// `x` lives in parent; child.set should mutate parent's binding.
	bool ok = child.set("x", jam::ComptimeValue::makeInt(99, 32, true));
	ASSERT_TRUE(ok);
	ASSERT_EQ(static_cast<uint64_t>(99), parent.lookup("x")->asU64());
}

// ── Helper builders for statement nodes ──────────────────────────

NodeIdx mkVarDecl(NodeStore &ns, StringPool &sp, const std::string &name,
                  NodeIdx init) {
	ExtraIdx extra = ns.reserveExtra(3);
	ns.setExtra(extra, sp.intern(name));
	ns.setExtra(extra + 1, kNoType);  // type — ignored by comp evaluator
	ns.setExtra(extra + 2, init);
	AstNode node{};
	node.tag = AstTag::VarDecl;
	node.lhs = extra;
	node.rhs = 0;  // flags
	return ns.addNode(node);
}

NodeIdx mkAssign(NodeStore &ns, NodeIdx target, NodeIdx value) {
	AstNode node{};
	node.tag = AstTag::Assign;
	node.lhs = target;
	node.rhs = value;
	return ns.addNode(node);
}

NodeIdx mkIfNode(NodeStore &ns, NodeIdx cond,
                 const std::vector<NodeIdx> &thenStmts,
                 const std::vector<NodeIdx> &elseStmts) {
	std::size_t total = 2 + thenStmts.size() + elseStmts.size();
	ExtraIdx extra = ns.reserveExtra(total);
	ns.setExtra(extra, static_cast<uint32_t>(thenStmts.size()));
	ns.setExtra(extra + 1, static_cast<uint32_t>(elseStmts.size()));
	for (std::size_t i = 0; i < thenStmts.size(); i++) {
		ns.setExtra(extra + 2 + i, thenStmts[i]);
	}
	for (std::size_t i = 0; i < elseStmts.size(); i++) {
		ns.setExtra(extra + 2 + thenStmts.size() + i, elseStmts[i]);
	}
	AstNode node{};
	node.tag = AstTag::IfNode;
	node.lhs = cond;
	node.rhs = extra;
	return ns.addNode(node);
}

NodeIdx mkWhileNode(NodeStore &ns, NodeIdx cond,
                    const std::vector<NodeIdx> &body) {
	ExtraIdx extra = ns.reserveExtra(1 + body.size());
	ns.setExtra(extra, static_cast<uint32_t>(body.size()));
	for (std::size_t i = 0; i < body.size(); i++) {
		ns.setExtra(extra + 1 + i, body[i]);
	}
	AstNode node{};
	node.tag = AstTag::WhileNode;
	node.lhs = cond;
	node.rhs = extra;
	return ns.addNode(node);
}

NodeIdx mkMemberAccess(NodeStore &ns, StringPool &sp, NodeIdx base,
                       const std::string &member) {
	AstNode node{};
	node.tag = AstTag::MemberAccess;
	node.lhs = base;
	node.rhs = sp.intern(member);
	return ns.addNode(node);
}

// ── execStmt: VarDecl + Assign ───────────────────────────────────

void testExecVarDeclAddsBinding() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	// var x = 42;
	NodeIdx decl = mkVarDecl(ns, sp, "x", mkInt(ns, 42));
	jam::ExecResult r = e.execStmt(decl, scope, iter, 1000, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Continue);
	ASSERT_TRUE(scope.lookup("x") != nullptr);
	ASSERT_EQ(static_cast<uint64_t>(42), scope.lookup("x")->asU64());
}

void testExecAssignMutatesBinding() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	scope.bind("x", jam::ComptimeValue::makeInt(5, 64, false));
	NodeIdx tgt = mkVar(ns, sp, "x");
	NodeIdx asn = mkAssign(ns, tgt, mkInt(ns, 99));
	jam::ExecResult r = e.execStmt(asn, scope, iter, 1000, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Continue);
	ASSERT_EQ(static_cast<uint64_t>(99), scope.lookup("x")->asU64());
}

void testExecAssignToUndeclaredErrors() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	NodeIdx tgt = mkVar(ns, sp, "neverDeclared");
	NodeIdx asn = mkAssign(ns, tgt, mkInt(ns, 1));
	jam::ExecResult r = e.execStmt(asn, scope, iter, 1000, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Error);
	ASSERT_TRUE(diags.hasErrors());
}

// ── execStmt: control flow ───────────────────────────────────────

void testExecIfPicksTrueBranch() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	scope.bind("x", jam::ComptimeValue::makeInt(0, 64, false));
	// if (true) { x = 1; } else { x = 2; }
	NodeIdx thenAsn = mkAssign(ns, mkVar(ns, sp, "x"), mkInt(ns, 1));
	NodeIdx elseAsn = mkAssign(ns, mkVar(ns, sp, "x"), mkInt(ns, 2));
	NodeIdx ifNode = mkIfNode(ns, mkBool(ns, true), {thenAsn}, {elseAsn});
	jam::ExecResult r = e.execStmt(ifNode, scope, iter, 1000, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Continue);
	ASSERT_EQ(static_cast<uint64_t>(1), scope.lookup("x")->asU64());
}

void testExecIfPicksFalseBranch() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	scope.bind("x", jam::ComptimeValue::makeInt(0, 64, false));
	NodeIdx thenAsn = mkAssign(ns, mkVar(ns, sp, "x"), mkInt(ns, 1));
	NodeIdx elseAsn = mkAssign(ns, mkVar(ns, sp, "x"), mkInt(ns, 2));
	NodeIdx ifNode = mkIfNode(ns, mkBool(ns, false), {thenAsn}, {elseAsn});
	e.execStmt(ifNode, scope, iter, 1000, ret, diags, loc);
	ASSERT_EQ(static_cast<uint64_t>(2), scope.lookup("x")->asU64());
}

void testExecWhileLoopCountsUp() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	// var i: u64 = 0;
	scope.bind("i", jam::ComptimeValue::makeInt(0, 64, false));
	// while (i < 5) { i = i + 1; }
	NodeIdx cond = mkBinOp(ns, BinOp::Lt, mkVar(ns, sp, "i"), mkInt(ns, 5));
	NodeIdx inc = mkBinOp(ns, BinOp::Add, mkVar(ns, sp, "i"), mkInt(ns, 1));
	NodeIdx asn = mkAssign(ns, mkVar(ns, sp, "i"), inc);
	NodeIdx whileNode = mkWhileNode(ns, cond, {asn});
	jam::ExecResult r =
	    e.execStmt(whileNode, scope, iter, 1000, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Continue);
	ASSERT_EQ(static_cast<uint64_t>(5), scope.lookup("i")->asU64());
}

void testExecWhileIterationCapTrips() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	scope.bind("i", jam::ComptimeValue::makeInt(0, 64, false));
	// while (true) { i = i + 1; } — cap at 100 iterations.
	NodeIdx inc = mkBinOp(ns, BinOp::Add, mkVar(ns, sp, "i"), mkInt(ns, 1));
	NodeIdx asn = mkAssign(ns, mkVar(ns, sp, "i"), inc);
	NodeIdx whileNode = mkWhileNode(ns, mkBool(ns, true), {asn});
	jam::ExecResult r =
	    e.execStmt(whileNode, scope, iter, /*iterCap=*/100, ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::IterationCap);
	ASSERT_TRUE(diags.hasErrors());
	// 100 iterations ran (i went 0→1→…→100); the 101st loop attempt
	// trips the cap before the body runs.
	ASSERT_EQ(static_cast<uint64_t>(100), scope.lookup("i")->asU64());
}

void testExecBlockShortCircuitsOnError() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	jam::Diagnostics diags;
	jam::SrcLoc loc{"test.jam", 1};
	uint32_t iter = 0;
	jam::ComptimeValue ret;

	// Block: var x = 0; <bad-assign-to-y>; x = 99;
	// The bad assign should stop execution before x = 99 runs.
	NodeIdx d = mkVarDecl(ns, sp, "x", mkInt(ns, 0));
	NodeIdx bad = mkAssign(ns, mkVar(ns, sp, "y"), mkInt(ns, 1));
	NodeIdx good = mkAssign(ns, mkVar(ns, sp, "x"), mkInt(ns, 99));
	std::vector<NodeIdx> body = {d, bad, good};
	jam::ExecResult r = e.execBlock(body.data(), body.size(), scope, iter, 1000,
	                                ret, diags, loc);
	ASSERT_TRUE(r == jam::ExecResult::Error);
	// x stays at 0 (the good assign never ran).
	ASSERT_EQ(static_cast<uint64_t>(0), scope.lookup("x")->asU64());
}

void testMemberAccessStrLength() {
	NodeStore ns;
	StringPool sp;
	TypePool tp;
	jam::ComptimeEvaluator e(ns, sp, tp);
	jam::ComptimeScope scope;
	scope.bind("s", jam::ComptimeValue::makeStr(sp.intern("hello")));

	NodeIdx access = mkMemberAccess(ns, sp, mkVar(ns, sp, "s"), "length");
	jam::ComptimeValue v = e.eval(access, scope);
	ASSERT_TRUE(v.isInt());
	ASSERT_EQ(static_cast<uint64_t>(5), v.asU64());
}

}  // namespace

int main() {
	TestFramework framework;
	framework.addTest("Comptime - NumberLit folds to Int",
	                  testNumberLitFoldsToInt);
	framework.addTest("Comptime - BoolLit folds", testBoolLitFolds);
	framework.addTest("Comptime - StringLit folds", testStringLitFolds);
	framework.addTest("Comptime - Variable lookup from scope",
	                  testVariableLookupFromScope);
	framework.addTest("Comptime - missing variable returns None",
	                  testVariableMissingReturnsNone);
	framework.addTest("Comptime - nested scope falls through to parent",
	                  testNestedScopeFallsThroughToParent);
	framework.addTest("Comptime - int add/sub/mul", testIntAddSubMul);
	framework.addTest("Comptime - int div-by-zero returns None",
	                  testIntDivByZeroIsNone);
	framework.addTest("Comptime - int comparisons", testIntComparisons);
	framework.addTest("Comptime - bitwise and/or/xor", testBitwise);
	framework.addTest("Comptime - logical and short-circuits",
	                  testLogAndShortCircuits);
	framework.addTest("Comptime - logical or short-circuits",
	                  testLogOrShortCircuits);
	framework.addTest("Comptime - unary negate int", testUnaryNegateInt);
	framework.addTest("Comptime - unary log-not bool", testUnaryLogNotBool);
	framework.addTest("Comptime - string index returns byte",
	                  testStringIndexReturnsByte);
	framework.addTest("Comptime - string index out-of-bounds None",
	                  testStringIndexOutOfBoundsIsNone);
	framework.addTest("Comptime - type equality", testTypeEquality);
	framework.addTest("Comptime - aggregate indexing", testAggregateIndexing);
	framework.addTest("Comptime - evalRequired pushes diagnostic on fail",
	                  testEvalRequiredPushesDiagnosticOnFail);
	framework.addTest("Comptime - scope set mutates existing",
	                  testScopeSetMutatesExistingBinding);
	framework.addTest("Comptime - scope set returns false for unknown",
	                  testScopeSetFailsForUnknownName);
	framework.addTest("Comptime - scope set walks up to parent",
	                  testScopeSetWalksUpToParent);
	framework.addTest("Comptime - exec VarDecl adds binding",
	                  testExecVarDeclAddsBinding);
	framework.addTest("Comptime - exec Assign mutates",
	                  testExecAssignMutatesBinding);
	framework.addTest("Comptime - exec Assign to undeclared errors",
	                  testExecAssignToUndeclaredErrors);
	framework.addTest("Comptime - exec If picks true branch",
	                  testExecIfPicksTrueBranch);
	framework.addTest("Comptime - exec If picks false branch",
	                  testExecIfPicksFalseBranch);
	framework.addTest("Comptime - exec While counts up to bound",
	                  testExecWhileLoopCountsUp);
	framework.addTest("Comptime - exec While iteration cap trips",
	                  testExecWhileIterationCapTrips);
	framework.addTest("Comptime - exec Block short-circuits on error",
	                  testExecBlockShortCircuitsOnError);
	framework.addTest("Comptime - member-access str.length",
	                  testMemberAccessStrLength);
	framework.addTest("Comptime - evalRequired silent on success",
	                  testEvalRequiredSilentOnSuccess);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
