// Unit tests for the JIR skeleton: data-structure invariants of jir.h
// (sentinels, monotonic refs, extra pool, instruction size).
//
// End-to-end coverage of `--via-jir` is intentionally *not* duplicated
// here: once codegen has fully migrated to JIR, the existing unit and
// codegen-error suites will exercise it as the only path. Adding
// parallel JIR-flagged copies of the same programs would just be
// churn we'd delete later.

#include "../../src/jir.h"
#include "test_framework.h"

namespace {

void testFunctionStartsWithSentinels() {
	JirFunction fn;
	// kNoJirRef (0) is reserved as sentinel; the first real instruction
	// gets ref 1. Same for blocks.
	ASSERT_TRUE(fn.insts.size() == 1);
	ASSERT_TRUE(fn.blocks.size() == 1);
	ASSERT_TRUE(fn.insts[0].tag == JirTag::Invalid);
}

void testPushInstReturnsMonotonicRefs() {
	JirFunction fn;
	JirInst c1{};
	c1.tag = JirTag::Int;
	c1.a = 42;
	c1.ty = BuiltinType::I32;
	JirRef r1 = fn.pushInst(c1);

	JirInst c2{};
	c2.tag = JirTag::Bool;
	c2.a = 1;
	c2.ty = BuiltinType::Bool;
	JirRef r2 = fn.pushInst(c2);

	ASSERT_TRUE(r1 == 1);
	ASSERT_TRUE(r2 == 2);
	ASSERT_TRUE(fn.getInst(r1).tag == JirTag::Int);
	ASSERT_TRUE(fn.getInst(r2).tag == JirTag::Bool);
	ASSERT_TRUE(fn.getInst(r1).a == 42);
}

void testPushBlockReturnsMonotonicRefs() {
	JirFunction fn;
	JirBlockRef entry = fn.pushBlock("entry");
	JirBlockRef body = fn.pushBlock("body");
	ASSERT_TRUE(entry == 1);
	ASSERT_TRUE(body == 2);
	ASSERT_TRUE(fn.getBlock(entry).name == "entry");
	ASSERT_TRUE(fn.getBlock(body).name == "body");
}

void testExtraPoolAppendsAndReads() {
	JirFunction fn;
	uint32_t buf[3] = {10, 20, 30};
	JirExtraIdx start = fn.pushExtra(buf, 3);
	ASSERT_TRUE(fn.getExtra(start) == 10);
	ASSERT_TRUE(fn.getExtra(start + 1) == 20);
	ASSERT_TRUE(fn.getExtra(start + 2) == 30);
}

void testInstSizeIsSmall() {
	// The struct's layout impacts cache behaviour for dense
	// instruction arrays. Pin down its current shape so a careless
	// addition doesn't accidentally blow it up.
	ASSERT_TRUE(sizeof(JirInst) <= 24);
}

}  // namespace

class JirSkeletonTests {
  public:
	static void registerAllTests(TestFramework &framework) {
		framework.addTest("JIR - function starts with sentinels",
		                  testFunctionStartsWithSentinels);
		framework.addTest("JIR - pushInst returns monotonic refs",
		                  testPushInstReturnsMonotonicRefs);
		framework.addTest("JIR - pushBlock returns monotonic refs",
		                  testPushBlockReturnsMonotonicRefs);
		framework.addTest("JIR - extra pool appends and reads",
		                  testExtraPoolAppendsAndReads);
		framework.addTest("JIR - instruction size is small", testInstSizeIsSmall);
	}
};

int main() {
	TestFramework framework;
	JirSkeletonTests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
