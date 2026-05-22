// Unit tests for src/decl.h — DeclTable's identity invariants, the
// symmetric declareDependency edges, and the sentinel slot. The
// Analyzer layer (cycle detection, ensureDeclAnalyzed, struct
// field materialization) gets its own test suite once it lands.

#include "../../src/decl.h"
#include "test_framework.h"

namespace {

void testSentinelSlot() {
	jam::DeclTable t;
	// Index 0 is the sentinel: findByName for an unknown name returns
	// kNoDecl, which IS 0 — callers can treat that as "not found".
	ASSERT_EQ(jam::kNoDecl, t.findByName("anything"));
	ASSERT_EQ(static_cast<std::size_t>(0), t.size());
}

void testCreateAssignsMonotonicIndices() {
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	jam::DeclIndex b = t.create(jam::DeclKind::Struct, "b");
	jam::DeclIndex c = t.create(jam::DeclKind::Const, "c");
	ASSERT_NE(jam::kNoDecl, a);
	ASSERT_NE(jam::kNoDecl, b);
	ASSERT_NE(jam::kNoDecl, c);
	ASSERT_TRUE(a < b);
	ASSERT_TRUE(b < c);
	ASSERT_EQ(static_cast<std::size_t>(3), t.size());
}

void testFindByName() {
	jam::DeclTable t;
	jam::DeclIndex vec = t.create(jam::DeclKind::Function, "Vec");
	jam::DeclIndex box = t.create(jam::DeclKind::Function, "Box");
	ASSERT_EQ(vec, t.findByName("Vec"));
	ASSERT_EQ(box, t.findByName("Box"));
	ASSERT_EQ(jam::kNoDecl, t.findByName("Missing"));
}

void testFindByNameAndKindLandsOnRequestedKind() {
	// Jam permits a function and a struct to share a name. The first
	// registration wins in `findByName`; `findByNameAndKind` walks
	// past it and lands on the kind the caller actually wants.
	jam::DeclTable t;
	jam::DeclIndex fnCounter = t.create(jam::DeclKind::Function, "Counter");
	jam::DeclIndex structCounter = t.create(jam::DeclKind::Struct, "Counter");
	// findByName returns the first (function).
	ASSERT_EQ(fnCounter, t.findByName("Counter"));
	// Kind-aware lookup returns each individually.
	ASSERT_EQ(fnCounter,
	          t.findByNameAndKind("Counter", jam::DeclKind::Function));
	ASSERT_EQ(structCounter,
	          t.findByNameAndKind("Counter", jam::DeclKind::Struct));
	ASSERT_EQ(jam::kNoDecl,
	          t.findByNameAndKind("Counter", jam::DeclKind::Enum));
	ASSERT_EQ(jam::kNoDecl,
	          t.findByNameAndKind("Missing", jam::DeclKind::Struct));
}

void testDependencyEdgesAreSymmetric() {
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	jam::DeclIndex b = t.create(jam::DeclKind::Function, "b");
	jam::DeclIndex c = t.create(jam::DeclKind::Function, "c");

	// a -> b, a -> c, b -> c
	t.declareDependency(a, b);
	t.declareDependency(a, c);
	t.declareDependency(b, c);

	const auto &ad = t.get(a);
	const auto &bd = t.get(b);
	const auto &cd = t.get(c);

	// a's dependencies: {b, c}; a is no one's dependant.
	ASSERT_EQ(static_cast<std::size_t>(2), ad.dependencies.size());
	ASSERT_EQ(static_cast<std::size_t>(0), ad.dependants.size());
	// b: one dep (c), one dependant (a).
	ASSERT_EQ(static_cast<std::size_t>(1), bd.dependencies.size());
	ASSERT_EQ(c, bd.dependencies[0]);
	ASSERT_EQ(static_cast<std::size_t>(1), bd.dependants.size());
	ASSERT_EQ(a, bd.dependants[0]);
	// c: two dependants (a and b), no deps.
	ASSERT_EQ(static_cast<std::size_t>(0), cd.dependencies.size());
	ASSERT_EQ(static_cast<std::size_t>(2), cd.dependants.size());
}

void testDuplicateDependencyIsIdempotent() {
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	jam::DeclIndex b = t.create(jam::DeclKind::Function, "b");
	t.declareDependency(a, b);
	t.declareDependency(a, b);
	t.declareDependency(a, b);
	ASSERT_EQ(static_cast<std::size_t>(1), t.get(a).dependencies.size());
	ASSERT_EQ(static_cast<std::size_t>(1), t.get(b).dependants.size());
}

void testSelfDependencyIgnored() {
	// Decl A depending on itself isn't useful — the cycle detector
	// runs on the analysis stack, not the dep graph. The graph
	// rejects self-loops up front so trace walks don't infinite-loop.
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	t.declareDependency(a, a);
	ASSERT_EQ(static_cast<std::size_t>(0), t.get(a).dependencies.size());
	ASSERT_EQ(static_cast<std::size_t>(0), t.get(a).dependants.size());
}

void testKNoDecCallsAreNoOps() {
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	// Either side kNoDecl: silently ignored, no crash, no edges added.
	t.declareDependency(jam::kNoDecl, a);
	t.declareDependency(a, jam::kNoDecl);
	ASSERT_EQ(static_cast<std::size_t>(0), t.get(a).dependencies.size());
	ASSERT_EQ(static_cast<std::size_t>(0), t.get(a).dependants.size());
}

void testGetReturnsMutableReference() {
	// Verify the mutable accessor lets the analyzer flip analysis state.
	// This is exercised by the Analyzer layer; we just sanity-check
	// the reference semantics here.
	jam::DeclTable t;
	jam::DeclIndex a = t.create(jam::DeclKind::Function, "a");
	ASSERT_TRUE(t.get(a).analysis == jam::DeclAnalysis::Unreferenced);
	t.get(a).analysis = jam::DeclAnalysis::InProgress;
	ASSERT_TRUE(t.get(a).analysis == jam::DeclAnalysis::InProgress);
}

}  // namespace

int main() {
	TestFramework framework;
	framework.addTest("DeclTable - sentinel slot present", testSentinelSlot);
	framework.addTest("DeclTable - create assigns monotonic indices",
	                  testCreateAssignsMonotonicIndices);
	framework.addTest("DeclTable - findByName lookup", testFindByName);
	framework.addTest("DeclTable - findByNameAndKind disambiguates",
	                  testFindByNameAndKindLandsOnRequestedKind);
	framework.addTest("DeclTable - dependency edges are symmetric",
	                  testDependencyEdgesAreSymmetric);
	framework.addTest("DeclTable - duplicate dependency is idempotent",
	                  testDuplicateDependencyIsIdempotent);
	framework.addTest("DeclTable - self dependency is ignored",
	                  testSelfDependencyIgnored);
	framework.addTest("DeclTable - kNoDecl edges are no-ops",
	                  testKNoDecCallsAreNoOps);
	framework.addTest("DeclTable - get() returns mutable reference",
	                  testGetReturnsMutableReference);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
