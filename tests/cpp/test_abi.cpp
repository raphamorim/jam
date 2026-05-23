// In-process unit tests for the ABI classifier (P9.1).
//
// The classifier is a pure function of (parameter mode, type) for
// parameters and (type) for returns. These tests construct a
// JamCodegenContext (it owns the TypePool and the lazy LLVM type
// cache), build types via the existing TypePool API, and assert on
// the resulting ParamABI / ReturnABI shape.
//
// Reference behavior (from docs/ABI.md §3):
//   mut                    -> ByPointer (any size)
//   let / move, scalar T   -> ByValue
//   let / move, aggregate
//     size <= 16 bytes     -> ByValue
//     size  > 16 bytes     -> ByPointer

#include "../../src/abi.h"
#include "../../src/ast.h"
#include "../../src/codegen.h"
#include "test_framework.h"

#include <vector>

namespace {

// Build a struct type with the given field types and register it in the
// codegen context's struct registry, so typeSize / typeAlign / etc. can
// resolve it. Returns the TypeIdx for the struct.
//
// We have to build the LLVM struct type explicitly here rather than
// calling ctx.getLLVMType(ty) — that path looks up the struct registry,
// which we're about to populate. Lookup-before-register would throw
// "Unknown struct type". So: build the LLVM type via JamLLVMStructType
// against the already-known field types, then register.
TypeIdx
buildStruct(JamCodegenContext &ctx, const std::string &name,
            const std::vector<std::pair<std::string, TypeIdx>> &fields) {
	StringIdx nameId = ctx.getStringPool().intern(name);
	TypeIdx ty = ctx.getTypePool().internStruct(nameId);
	std::vector<JamTypeRef> fieldTypes;
	fieldTypes.reserve(fields.size());
	for (const auto &f : fields) {
		fieldTypes.push_back(ctx.getLLVMType(f.second));
	}
	JamTypeRef llvmTy =
	    JamLLVMStructType(ctx.getContext(), fieldTypes.data(),
	                      static_cast<unsigned>(fieldTypes.size()), false);
	ctx.registerStruct(name, llvmTy, fields);
	return ty;
}

// Parameter classification

void testLetU32IsByValueScalar() {
	JamCodegenContext ctx("test");
	auto a = jam::abi::classifyParam(ParamMode::Let, BuiltinType::U32, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByValue);
	ASSERT_TRUE(a.llvmType != nullptr);
}

void testMutU32IsByPointer() {
	JamCodegenContext ctx("test");
	auto a = jam::abi::classifyParam(ParamMode::Mut, BuiltinType::U32, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(4), a.pointerAlign);
}

void testMoveU8IsByValueScalar() {
	JamCodegenContext ctx("test");
	auto a = jam::abi::classifyParam(ParamMode::Move, BuiltinType::U8, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByValue);
}

void testLetSmallStructIsByPointer() {
	// Every aggregate is byref under the current ABI — `let` of an
	// 8-byte struct lands ByPointer same as a 24-byte one. LLVM
	// mem2reg/SROA re-promotes the storage back to registers when
	// the bytes don't escape, so the perf cost is opt-time only.
	JamCodegenContext ctx("test");
	TypeIdx pair = buildStruct(
	    ctx, "Pair", {{"a", BuiltinType::U32}, {"b", BuiltinType::U32}});
	auto a = jam::abi::classifyParam(ParamMode::Let, pair, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(4), a.pointerAlign);
}

void testLet16ByteStructIsByPointer() {
	// 16-byte struct: still byref (no size threshold).
	JamCodegenContext ctx("test");
	TypeIdx pair = buildStruct(
	    ctx, "Pair64", {{"a", BuiltinType::U64}, {"b", BuiltinType::U64}});
	auto a = jam::abi::classifyParam(ParamMode::Let, pair, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(8), a.pointerAlign);
}

void testLetLargeStructIsByPointer() {
	// { u64, u64, u64 } = 24 bytes > 16
	JamCodegenContext ctx("test");
	TypeIdx big = buildStruct(ctx, "Big",
	                          {{"a", BuiltinType::U64},
	                           {"b", BuiltinType::U64},
	                           {"c", BuiltinType::U64}});
	auto a = jam::abi::classifyParam(ParamMode::Let, big, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(8), a.pointerAlign);
}

void testMoveLargeStructIsByPointer() {
	// Move follows the same threshold as Let on the callee side.
	JamCodegenContext ctx("test");
	TypeIdx big = buildStruct(ctx, "Big2",
	                          {{"a", BuiltinType::U64},
	                           {"b", BuiltinType::U64},
	                           {"c", BuiltinType::U64}});
	auto a = jam::abi::classifyParam(ParamMode::Move, big, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(8), a.pointerAlign);
}

void testMutLargeStructIsByPointer() {
	JamCodegenContext ctx("test");
	TypeIdx big = buildStruct(ctx, "Big3",
	                          {{"a", BuiltinType::U64},
	                           {"b", BuiltinType::U64},
	                           {"c", BuiltinType::U64}});
	auto a = jam::abi::classifyParam(ParamMode::Mut, big, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
	ASSERT_EQ(static_cast<uint32_t>(8), a.pointerAlign);
}

void testMutSmallStructIsByPointer() {
	// `mut` is always ByPointer regardless of size.
	JamCodegenContext ctx("test");
	TypeIdx pair = buildStruct(
	    ctx, "PairMut", {{"a", BuiltinType::U32}, {"b", BuiltinType::U32}});
	auto a = jam::abi::classifyParam(ParamMode::Mut, pair, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByPointer);
}

void testLetBoolIsByValueScalar() {
	JamCodegenContext ctx("test");
	auto a = jam::abi::classifyParam(ParamMode::Let, BuiltinType::Bool, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByValue);
}

void testLetF64IsByValueScalar() {
	JamCodegenContext ctx("test");
	auto a = jam::abi::classifyParam(ParamMode::Let, BuiltinType::F64, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByValue);
}

void testLetPointerIsByValueScalar() {
	JamCodegenContext ctx("test");
	TypeIdx ptr = ctx.getTypePool().internPtrSingle(BuiltinType::U8);
	auto a = jam::abi::classifyParam(ParamMode::Let, ptr, ctx);
	ASSERT_TRUE(a.kind == jam::abi::ParamABI::Kind::ByValue);
}

// Return classification

void testReturnVoidIsDirect() {
	JamCodegenContext ctx("test");
	auto r = jam::abi::classifyReturn(kNoType, ctx);
	ASSERT_TRUE(r.kind == jam::abi::ReturnABI::Kind::Direct);
}

void testReturnU32IsDirect() {
	JamCodegenContext ctx("test");
	auto r = jam::abi::classifyReturn(BuiltinType::U32, ctx);
	ASSERT_TRUE(r.kind == jam::abi::ReturnABI::Kind::Direct);
}

void testReturn16ByteAggregateIsIndirect() {
	// Aggregate returns are always sret (no size threshold). Caller
	// owns the slot; callee writes through the hidden first arg.
	JamCodegenContext ctx("test");
	TypeIdx pair = buildStruct(
	    ctx, "PairR", {{"a", BuiltinType::U64}, {"b", BuiltinType::U64}});
	auto r = jam::abi::classifyReturn(pair, ctx);
	ASSERT_TRUE(r.kind == jam::abi::ReturnABI::Kind::Indirect);
	ASSERT_EQ(static_cast<uint32_t>(8), r.sretAlign);
}

void testReturnLargeAggregateIsIndirect() {
	JamCodegenContext ctx("test");
	TypeIdx big = buildStruct(ctx, "BigR",
	                          {{"a", BuiltinType::U64},
	                           {"b", BuiltinType::U64},
	                           {"c", BuiltinType::U64}});
	auto r = jam::abi::classifyReturn(big, ctx);
	ASSERT_TRUE(r.kind == jam::abi::ReturnABI::Kind::Indirect);
	ASSERT_EQ(static_cast<uint32_t>(8), r.sretAlign);
}

}  // namespace

class ABITests {
  public:
	static void registerAllTests(TestFramework &framework) {
		// classifyParam
		framework.addTest("ABI classifyParam - let u32 ByValue",
		                  testLetU32IsByValueScalar);
		framework.addTest("ABI classifyParam - mut u32 ByPointer align 4",
		                  testMutU32IsByPointer);
		framework.addTest("ABI classifyParam - move u8 ByValue",
		                  testMoveU8IsByValueScalar);
		framework.addTest("ABI classifyParam - let small struct ByPointer",
		                  testLetSmallStructIsByPointer);
		framework.addTest("ABI classifyParam - let 16-byte struct ByPointer",
		                  testLet16ByteStructIsByPointer);
		framework.addTest("ABI classifyParam - let 24-byte struct ByPointer",
		                  testLetLargeStructIsByPointer);
		framework.addTest("ABI classifyParam - move 24-byte struct ByPointer",
		                  testMoveLargeStructIsByPointer);
		framework.addTest("ABI classifyParam - mut 24-byte struct ByPointer",
		                  testMutLargeStructIsByPointer);
		framework.addTest(
		    "ABI classifyParam - mut small struct ByPointer (size irrelevant)",
		    testMutSmallStructIsByPointer);
		framework.addTest("ABI classifyParam - let bool ByValue",
		                  testLetBoolIsByValueScalar);
		framework.addTest("ABI classifyParam - let f64 ByValue",
		                  testLetF64IsByValueScalar);
		framework.addTest("ABI classifyParam - let *u8 ByValue",
		                  testLetPointerIsByValueScalar);

		// classifyReturn
		framework.addTest("ABI classifyReturn - void Direct",
		                  testReturnVoidIsDirect);
		framework.addTest("ABI classifyReturn - u32 Direct",
		                  testReturnU32IsDirect);
		framework.addTest("ABI classifyReturn - 16-byte aggregate Indirect",
		                  testReturn16ByteAggregateIsIndirect);
		framework.addTest("ABI classifyReturn - 24-byte aggregate Indirect",
		                  testReturnLargeAggregateIsIndirect);
	}
};

int main() {
	TestFramework framework;
	ABITests::registerAllTests(framework);
	framework.runAll();
	return framework.allPassed() ? 0 : 1;
}
