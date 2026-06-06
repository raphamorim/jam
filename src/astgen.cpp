/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "astgen.h"

#include "abi.h"
#include "ast.h"
#include "codegen.h"
#include "comptime.h"
#include "jir_codegen.h"
#include "mangling.h"
#include "target.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// Per-function lowering state. The current insertion block tracks
// where new instructions are appended; AstGen helpers push refs into
// `jfn.getBlock(currentBlock).insts`. `locals` maps source-level
// variable names to the JirRef of their owning Alloca, so reads emit
// Load(alloca) and writes emit Store(alloca, value).
// Loop scope: tracks the continue-target (cond / loop-head) and the
// break-target (loop-exit) so a nested `break` / `continue` can branch
// to the right block. Mirrors what FunctionAST::codegenWhile carried
// via a member stack in the legacy path.
struct LoopFrame {
	JirBlockRef continueBlock;
	JirBlockRef breakBlock;
	// Index into `dropScopes` where the loop body's scope sits.
	// `break` and `continue` drop everything from the top down
	// through this scope so locals declared inside the body are
	// destroyed before exiting / reiterating. Loop-var scopes
	// above this index (e.g. the for-loop induction variable)
	// are preserved across iterations.
	std::size_t bodyScopeIdx = 0;
};

// Drop-bearing local: a mut/move-mode binding whose source-level type
// declares `fn drop(self: mut T)`. We collect these in declaration
// order so drop calls are emitted in REVERSE order at scope exit.
struct DropTrack {
	std::string varName;
	JirRef slot;   // alloca for the variable
	TypeIdx type;  // source-level type
	std::string
	    llvmFnName;  // canonical drop fn — FQN like `T.drop` / `m.T.drop`
};

struct AstGenCtx {
	JirFunction &jfn;
	JamCodegenContext &ctx;
	JirBlockRef currentBlock;
	std::unordered_map<std::string, JirRef> locals;
	std::unordered_map<std::string, TypeIdx> localTypes;
	std::vector<LoopFrame> loopStack;
	// Drop tracking: a stack of scopes, each holding drop-bearing
	// bindings declared at that scope level (in declaration order).
	// Scope 0 is the function body. Structured forms (if / while /
	// for / match arm) push and pop scopes around their bodies,
	// emitting drops for the popped scope's bindings on exit. break
	// / continue drop *down through* the enclosing loop's body
	// scope; return drops every scope.
	std::vector<std::vector<DropTrack>> dropScopes;
	// Names declared at each currently-active lexical scope. Parallel
	// to `dropScopes` so push/pop happen together — the redeclaration
	// check in `astgenVarDecl` only consults the top frame, so sibling
	// blocks (`if { const op = ...; }` and `if { const op = ...; }`
	// at the same level) don't trip it. The function-body's frame is
	// pushed by `astgenBodyInto`.
	std::vector<std::unordered_set<std::string>> localScopes;
	// Most-recently-entered AST node — `astgenExpr` updates this on
	// every entry so error helpers without an explicit NodeIdx in
	// scope (`failHere` and friends) can still emit a SrcLoc.
	NodeIdx currentNode = 0;
};

// `AstGenAnalysisFail` is declared in astgen.h so main.cpp and
// codegen.cpp can catch it at decl / generic-instantiation
// boundaries. The diagnostic itself was pushed to
// `gctx.ctx.diagnostics()` before the throw.

// Source location for an AST node. Falls back to line 0 when the
// parser didn't record a line — never seen in practice.
static jam::SrcLoc locOf(AstGenCtx &gctx, NodeIdx node) {
	jam::SrcLoc loc;
	loc.file = gctx.ctx.currentFile();
	loc.line = gctx.ctx.getNodeStore().getLine(node);
	return loc;
}

// Build a Diagnostic with the current reference-trace stack copied
// in. Every astgen error helper funnels through here so the chain of
// generic instantiations (if any) is preserved.
static jam::Diagnostic makeDiag(AstGenCtx &gctx, NodeIdx node,
                                std::string message,
                                std::vector<jam::Diagnostic> notes) {
	jam::Diagnostic d;
	d.loc = locOf(gctx, node);
	d.severity = jam::Diagnostic::Severity::Error;
	d.message = std::move(message);
	d.notes = std::move(notes);
	d.referenceTrace = gctx.ctx.refTrace();
	return d;
}

// Append a Diagnostic anchored at `node` and keep walking. The
// caller is responsible for synthesising a poison value if the
// expression position needs one.
static void appendErrorNode(AstGenCtx &gctx, NodeIdx node,
                            std::string message) {
	gctx.ctx.diagnostics().push(makeDiag(gctx, node, std::move(message), {}));
}

// As `appendErrorNode`, but the diagnostic carries secondary notes.
// Notes are typically built by the caller from related source
// positions ("X was declared here", "did you mean Y?").
static void appendErrorNodeNotes(AstGenCtx &gctx, NodeIdx node,
                                 std::string message,
                                 std::vector<jam::Diagnostic> notes) {
	gctx.ctx.diagnostics().push(
	    makeDiag(gctx, node, std::move(message), std::move(notes)));
}

// Append + bail the current decl. The catch site (one per function,
// struct method, or generic instantiation) drops back to compiling
// siblings, so a single broken function doesn't suppress diagnostics
// from the rest of the file.
[[noreturn]] static void failNode(AstGenCtx &gctx, NodeIdx node,
                                  std::string message) {
	appendErrorNode(gctx, node, std::move(message));
	throw AstGenAnalysisFail{};
}

[[noreturn]] static void failNodeNotes(AstGenCtx &gctx, NodeIdx node,
                                       std::string message,
                                       std::vector<jam::Diagnostic> notes) {
	appendErrorNodeNotes(gctx, node, std::move(message), std::move(notes));
	throw AstGenAnalysisFail{};
}

// Versions that anchor at `gctx.currentNode` — used by helpers buried
// deep inside the call tree that don't have the offending NodeIdx in
// scope but know astgenExpr most recently entered some node.
[[noreturn]] static void failHere(AstGenCtx &gctx, std::string message) {
	failNode(gctx, gctx.currentNode, std::move(message));
}

static void appendErrorHere(AstGenCtx &gctx, std::string message) {
	appendErrorNode(gctx, gctx.currentNode, std::move(message));
}

// Forward decl (defined near the recovery helpers below).
static JirRef recoverHere(AstGenCtx &gctx, std::string message, TypeIdx ty);

// Method-miss reporting that understands CONDITIONAL methods: an
// instantiated generic method withdrawn for these type arguments
// replays the recorded reason instead of a bare "unknown method".
static JirRef reportMethodMiss(AstGenCtx &gctx,
                               const std::string &qualified) {
	if (const std::string *why = gctx.ctx.getWithdrawnMethod(qualified)) {
		failHere(gctx, "method `" + qualified +
		                   "` is not available for this instantiation — " +
		                   *why);
	}
	return recoverHere(gctx, "unknown method `" + qualified + "`", kNoType);
}

// Canonicalize a parser-deferred `[expr]T` (TypeKind::ArrayExpr) before
// the type becomes astgen state — binding types, expected hints, field
// reads. The codegen-side resolver throws when the length expression
// doesn't comptime-fold; convert that into a per-decl diagnostic
// anchored at the current node so sibling decls keep compiling.
static TypeIdx resolveDeferredArrayTy(AstGenCtx &gctx, TypeIdx ty) {
	if (ty == kNoType) return ty;
	if (gctx.ctx.getTypePool().get(ty).kind != TypeKind::ArrayExpr) return ty;
	try {
		return gctx.ctx.resolveArrayExpr(ty);
	} catch (const std::exception &e) { failHere(gctx, e.what()); }
}

// Forward-declared so the recovery helpers can build Poison; the
// real definition follows below alongside `emitAllocaHoisted`.
static JirRef emit(AstGenCtx &gctx, JirInst inst);

// Synthesize a typed Poison placeholder at the current cursor. Used
// after `appendError*` to keep walking the tree without committing
// to a specific value — codegen lowers Poison to LLVM `undef`, but
// the driver short-circuits before codegen runs whenever
// `Diagnostics::hasErrors()` is true, so the undef is unreachable
// in well-formed builds and serves only as a placeholder for
// downstream typecheck.
static JirRef emitPoison(AstGenCtx &gctx, TypeIdx ty) {
	JirInst inst{};
	inst.tag = JirTag::Poison;
	inst.ty = ty;
	return emit(gctx, inst);
}

// Combined: push a recoverable diagnostic anchored at `node` and
// hand back a typed Poison so the caller can continue typechecking.
static JirRef recoverNode(AstGenCtx &gctx, NodeIdx node, std::string message,
                          TypeIdx ty) {
	appendErrorNode(gctx, node, std::move(message));
	return emitPoison(gctx, ty);
}

// As `recoverNode` but anchored at `gctx.currentNode`.
static JirRef recoverHere(AstGenCtx &gctx, std::string message, TypeIdx ty) {
	appendErrorHere(gctx, std::move(message));
	return emitPoison(gctx, ty);
}

// Helper: append `inst` to the function's instruction array AND to
// the current block's insts list. Returns the new ref.
static JirRef emit(AstGenCtx &gctx, JirInst inst) {
	JirRef r = gctx.jfn.pushInst(inst);
	gctx.jfn.getBlockMut(gctx.currentBlock).insts.push_back(r);
	return r;
}

// Allocate an Alloca in the function's *entry* block (block 1)
// regardless of where the current emitter is. Mirrors what
// `JamLLVMBuildAlloca` does at the LLVM level — allocas are stack
// slots, they belong at function entry. Without this hoisting, an
// Alloca emitted inside a structured form (matchbind, if-body, etc.)
// lives in a later block than its uses, violating the JIR's
// def-before-use invariant even though LLVM codegen handles it via
// late hoisting.
//
// Inserts just before any terminator already in entry, otherwise
// appends. The JirRef is valid immediately; callers use it like any
// other ref.
static JirRef emitAllocaHoisted(AstGenCtx &gctx, JirInst alloca) {
	JirRef r = gctx.jfn.pushInst(alloca);
	auto &entryInsts = gctx.jfn.getBlockMut(/*entry=*/1).insts;
	std::size_t insertAt = entryInsts.size();
	while (insertAt > 0) {
		JirTag prev = gctx.jfn.getInst(entryInsts[insertAt - 1]).tag;
		if (prev != JirTag::Br && prev != JirTag::CondBr &&
		    prev != JirTag::Switch && prev != JirTag::Ret &&
		    prev != JirTag::Unreachable) {
			break;
		}
		insertAt--;
	}
	entryInsts.insert(
	    entryInsts.begin() + static_cast<std::ptrdiff_t>(insertAt), r);
	return r;
}

// True if the block's last instruction is a terminator (Ret /
// Unreachable / Br / CondBr / Switch). Mirrors LLVM's
// hasTerminator(). Used at function-tail to decide whether to emit
// an implicit `ret void` / `unreachable`.
static bool blockHasTerminator(const JirBlock &block, const JirFunction &jfn) {
	if (block.insts.empty()) return false;
	JirRef last = block.insts.back();
	JirTag t = jfn.getInst(last).tag;
	return t == JirTag::Ret || t == JirTag::Unreachable || t == JirTag::Br ||
	       t == JirTag::CondBr || t == JirTag::Switch;
}

// Count blocks that branch into `target` via Br / CondBr / Switch.
// Used by the function-end fallthrough check: a block with zero
// predecessors (and not the entry block) is unreachable, so the
// "non-void function falls through" diagnostic should not fire —
// dead post-merge / post-loop blocks come up naturally when every
// arm / iteration returns.
static std::size_t predecessorCount(const JirFunction &jfn,
                                    JirBlockRef target) {
	std::size_t count = 0;
	for (std::size_t b = 1; b < jfn.blocks.size(); b++) {
		if (static_cast<JirBlockRef>(b) == target) continue;
		const JirBlock &blk = jfn.blocks[b];
		if (blk.insts.empty()) continue;
		JirRef lastRef = blk.insts.back();
		const JirInst &last = jfn.getInst(lastRef);
		switch (last.tag) {
		case JirTag::Br:
			if (static_cast<JirBlockRef>(last.a) == target) count++;
			break;
		case JirTag::CondBr: {
			JirExtraIdx ex = last.b;
			if (ex + 2 <= jfn.extra.size()) {
				if (static_cast<JirBlockRef>(jfn.extra[ex]) == target) count++;
				if (static_cast<JirBlockRef>(jfn.extra[ex + 1]) == target)
					count++;
			}
			break;
		}
		case JirTag::Switch: {
			JirExtraIdx ex = last.b;
			if (ex + 2 > jfn.extra.size()) break;
			if (static_cast<JirBlockRef>(jfn.extra[ex]) == target) count++;
			uint32_t caseCount = jfn.extra[ex + 1];
			for (uint32_t i = 0; i < caseCount; i++) {
				JirExtraIdx caseSlot = ex + 2 + i * 4 + 3;
				if (caseSlot < jfn.extra.size() &&
				    static_cast<JirBlockRef>(jfn.extra[caseSlot]) == target) {
					count++;
				}
			}
			break;
		}
		default:
			break;
		}
	}
	return count;
}

// Result-location protocol. Callers tell `astgenExpr` whether they
// want the expression's *value* (default) or a *pointer* to its
// storage. Pointer requests on lvalue-shaped nodes (Variable,
// MemberAccess, Index, Deref) emit only the address-producing JIR
// (alloca slot / FieldAddr / IndexAddr) — never a Load of the whole
// value. Pointer requests on non-lvalue nodes spill the value to a
// fresh alloca and return that pointer. The producer's job is to
// honor the consumer's request: each lvalue-shaped case branches
// on `loc` in the main switch so one entry point covers both
// value and address forms uniformly.
enum class ResultLoc { Value, Pointer };

// Forward decl: AstGen for an arbitrary AST node in expression
// position. Returns the JirRef of the produced value (or pointer-
// to-value when `loc == ResultLoc::Pointer`), or kNoJirRef for
// statement-form / void expressions.
//
// When `loc == ResultLoc::Pointer` and `outLeafTy` is non-null, the
// callee writes the pointee type into `*outLeafTy`. The JIR ty of
// the returned JirRef is inconsistent for pointer results (alloca
// refs carry `ty=leaf`; FieldAddr / IndexAddr / BitCast carry
// `ty=PtrSingle(leaf)`), so callers that need the leaf without
// casework should pass `outLeafTy`.
static JirRef astgenExpr(AstGenCtx &gctx, NodeIdx node, TypeIdx expected,
                         ResultLoc loc = ResultLoc::Value,
                         TypeIdx *outLeafTy = nullptr);
// Forward decls: branch emitters, defined below alongside the
// control-flow helpers.
static void emitBr(AstGenCtx &gctx, JirBlockRef target);
static void emitCondBr(AstGenCtx &gctx, JirRef cond, JirBlockRef thenB,
                       JirBlockRef elseB);
static void emitDrops(AstGenCtx &gctx, const std::vector<DropTrack> &bindings);
// Field-walking drop-glue helpers — used both by emitDrops (auto field
// drop at scope exit for structs whose fields need drop) and by the
// `@dropInPlace(ptr)` intrinsic. Defined later; declared here so
// emitDrops + astgenVarDecl can call them.
static void emitDropInPlace(AstGenCtx &gctx, JirRef ptrRef, TypeIdx pointeeTy);
bool typeNeedsDropInner(JamCodegenContext &ctx, TypeIdx ty);
static void emitEnumPayloadDrops(AstGenCtx &gctx, JirRef ptrRef,
                                 const JamCodegenContext::EnumInfo *einfo);
static void rejectDropBearingFieldExtract(AstGenCtx &gctx, NodeIdx exprIdx,
                                          TypeIdx resultTy, const char *verb);
static void rejectAddrOfOnModeArg(AstGenCtx &gctx, NodeIdx argIdx,
                                  const Param &p);
static bool assignTargetIsValueWorld(AstGenCtx &gctx, NodeIdx targetIdx);
static JirRef emitCall(AstGenCtx &gctx, const FunctionAST *fn,
                       const std::vector<JirRef> &argRefs,
                       JirRef destPtr = kNoJirRef);
// Forward decls for the Call / TypeMethodCall paths so the
// astgenExprIntoPtr place-into-destination helper can drive them
// before their definitions appear later in this file.
static JirRef astgenCall(AstGenCtx &gctx, const AstNode &n,
                         JirRef destPtr = kNoJirRef);
static JirRef astgenTypeMethodCall(AstGenCtx &gctx, const AstNode &n,
                                   JirRef destPtr = kNoJirRef);
// Forward decl: place-into-destination dispatcher used by VarDecl,
// Return, and StructLitInto. See its definition for which expression
// shapes it recognises (StructLit, sret Calls).
static bool astgenExprIntoPtr(AstGenCtx &gctx, NodeIdx exprIdx,
                              TypeIdx expectedTy, JirRef destPtr);

// Emit a `JirTag::SretArg` referencing this function's hidden sret
// pointer. The pointer's pointee type is the function's return type.
// Used as the result_ptr for byref returns so a `return X` writes X
// straight into the caller-owned slot.
static JirRef emitSretArg(AstGenCtx &gctx, TypeIdx retTy) {
	TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrSingle, 0, 0, retTy, 0});
	JirInst s{};
	s.tag = JirTag::SretArg;
	s.ty = ptrTy;
	return emit(gctx, s);
}
// `v[i]` desugar dispatch — see `emitStructCfnDispatch` for the body.
// Forward-declared so astgenAssign (in this file, above the
// definition) can call it for `v[i] = x` -> setAt routing.
static JirRef emitStructCfnDispatch(AstGenCtx &gctx,
                                    const JamCodegenContext::StructInfo *sinfo,
                                    const char *methodName, JirRef recv,
                                    JirRef idx,
                                    const std::vector<JirRef> &extraArgs);

// Push an empty drop scope (called when entering a structured body).
// We push a parallel `localScopes` frame so each lexical block has
// its own redeclaration namespace — sibling `if` / `else` arms can
// each declare their own `const op`.
static inline void pushDropScope(AstGenCtx &gctx) {
	gctx.dropScopes.emplace_back();
	gctx.localScopes.emplace_back();
}

// Emit drops for every scope from the top of `dropScopes` down to (and
// including) index `targetIdx`. Does NOT pop — used for break /
// continue / return, where the original scope stack is still in flight
// in the calling AST walker (and will pop on its own as the structured
// bodies return).
static inline void emitDropsThroughScope(AstGenCtx &gctx,
                                         std::size_t targetIdx) {
	if (gctx.dropScopes.size() <= targetIdx) return;
	for (std::size_t i = gctx.dropScopes.size(); i > targetIdx; i--) {
		emitDrops(gctx, gctx.dropScopes[i - 1]);
	}
}

// If `exprIdx` is a bare `Variable` referring to a tracked drop local,
// remove that local from `gctx.dropScopes` so the scope-exit drop won't
// fire for it. The local's value has been captured by the surrounding
// owner (a struct-literal field, an array element, etc.); the new owner
// is responsible for the drop. Mirrors Rust's move semantics for non-
// Copy types — without this, every `Foo { f: v }` pattern leaves the
// local `v` to drop at scope exit, double-freeing the storage the
// struct field now holds.
//
// Permanent removal (rather than the path-local filter used for return-
// move) matches Rust's "definitely moved at this point" model for
// straight-line code. Conditionally-moved vars in branchy code would
// need drop-flag tracking; that's deferred. Practical impact for now:
// `if (cond) Foo { f: v } else { /* doesn't move v */ }` will silently
// leak `v` in the else branch instead of crashing.
static void consumeMovedVariable(AstGenCtx &gctx, NodeIdx exprIdx) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &expr = ns.get(exprIdx);
	if (expr.tag != AstTag::Variable) return;
	StringIdx nameId = static_cast<StringIdx>(expr.lhs);
	const std::string &name = gctx.ctx.getStringPool().get(nameId);
	for (auto &scope : gctx.dropScopes) {
		for (auto it = scope.begin(); it != scope.end(); ++it) {
			if (it->varName == name) {
				scope.erase(it);
				return;
			}
		}
	}
}

// Like emitDropsThroughScope but skips the drop for one specific tracked
// variable. Used by return-as-move: when the return expression is a bare
// `Variable` referring to a tracked local, that local's ownership has
// been transferred to the caller's binding, so the function-scope drop
// must NOT fire (otherwise the caller holds a freshly-freed pointer).
// We rebuild a filtered DropTrack vector per scope rather than mutating
// `gctx.dropScopes`, so other return paths in the same function (e.g.
// a branch that returns a different value) still see and drop the local
// correctly. Mirrors Rust's per-return move analysis.
static inline void emitDropsThroughScopeMovedOut(AstGenCtx &gctx,
                                                 std::size_t targetIdx,
                                                 const std::string &movedVar) {
	if (gctx.dropScopes.size() <= targetIdx) return;
	for (std::size_t i = gctx.dropScopes.size(); i > targetIdx; i--) {
		const auto &orig = gctx.dropScopes[i - 1];
		std::vector<DropTrack> filtered;
		filtered.reserve(orig.size());
		bool removed = false;
		for (const auto &d : orig) {
			if (!removed && d.varName == movedVar) {
				removed = true;
				continue;  // skip this entry: ownership moved to caller
			}
			filtered.push_back(d);
		}
		emitDrops(gctx, filtered);
	}
}

// Pop the top drop scope after emitting its drops (called when leaving
// a structured body via normal fall-through). The caller is responsible
// for not double-dropping when control reached the end via a return /
// break / continue that already emitted drops.
static inline void popDropScopeEmitting(AstGenCtx &gctx) {
	if (gctx.dropScopes.empty()) return;
	const auto &scope = gctx.dropScopes.back();
	if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock), gctx.jfn)) {
		emitDrops(gctx, scope);
	}
	gctx.dropScopes.pop_back();
	if (!gctx.localScopes.empty()) gctx.localScopes.pop_back();
}

// Pop the top scope WITHOUT emitting drops (used when the divergent
// terminator already emitted them, or when no drop-bearing local was
// declared in the scope).
static inline void popDropScope(AstGenCtx &gctx) {
	if (!gctx.dropScopes.empty()) gctx.dropScopes.pop_back();
	if (!gctx.localScopes.empty()) gctx.localScopes.pop_back();
}

// AstGen for a `NumberLit` AST node, materializing the appropriate
// JIR `Int` or `Float` constant.
static JirRef astgenNumberLit(AstGenCtx &gctx, const AstNode &n,
                              TypeIdx expected) {
	uint64_t val =
	    static_cast<uint64_t>(n.lhs) | (static_cast<uint64_t>(n.rhs) << 32);
	bool isNeg = (n.flags & 1) != 0;
	bool isFloat = (n.flags & 2) != 0;

	JirInst inst{};
	inst.srcLine =
	    gctx.jfn.insts.empty() ? 0 : static_cast<uint32_t>(0);  // patched below

	if (isFloat) {
		// This is the coercion point (we know `expected`), so produce the
		// value rounded ONCE to the target f32/f64, avoiding the
		// decimal→f64→f32 double-round. Codegen is unchanged: it emits the
		// already-correctly-rounded value via ConstReal.
		bool toF32 = (expected != kNoType && expected == BuiltinType::F32);
		double d;
		if ((n.flags & 4) != 0) {
			// Full f128 in the extra pool — round it to the target here.
			const NodeStore &ns = gctx.ctx.getNodeStore();
			ExtraIdx ei = static_cast<ExtraIdx>(n.lhs);
			uint32_t quad[4] = {ns.getExtra(ei), ns.getExtra(ei + 1),
			                    ns.getExtra(ei + 2), ns.getExtra(ei + 3)};
			d = JamLLVMQuadToTargetAsDouble(quad, toF32);
		} else {
			// f64 stored inline, and it is the exact f128 value (lossless), so
			// codegen's ConstReal(target, d) rounds it once to the target with
			// no double-rounding (f64→f32 here == f128→f32).
			uint64_t bits = static_cast<uint64_t>(n.lhs) |
			                (static_cast<uint64_t>(n.rhs) << 32);
			__builtin_memcpy(&d, &bits, sizeof(d));
		}
		uint64_t bits;
		__builtin_memcpy(&bits, &d, sizeof(bits));
		inst.tag = JirTag::Float;
		inst.a = static_cast<uint32_t>(bits & 0xFFFFFFFFu);
		inst.b = static_cast<uint32_t>(bits >> 32);
		if (isNeg) inst.flags |= 1;  // sign bit applied at codegen
		inst.ty = toF32 ? BuiltinType::F32 : BuiltinType::F64;
		return emit(gctx, inst);
	}

	inst.tag = JirTag::Int;
	inst.a = static_cast<uint32_t>(val & 0xFFFFFFFFu);
	inst.b = static_cast<uint32_t>(val >> 32);
	if (isNeg) inst.flags |= 1;

	// Pick the result type. With an integer `expected` we honour it
	// directly so literals lower at the consumer's width (this is the
	// peer-type-propagation contract). When the expected type is a
	// generic call or alias that resolves to an integer (e.g.
	// `Identity(i32)` -> i32), resolve it first so the contract holds
	// transitively. Otherwise fall back to the smallest-fit width.
	if (expected != kNoType) {
		TypeIdx resolved = expected;
		const TypeKey &k0 = gctx.ctx.getTypePool().get(resolved);
		if (k0.kind == TypeKind::GenericCall) {
			TypeIdx r = gctx.ctx.resolveGenericCall(resolved);
			if (r != kNoType) resolved = r;
		}
		const TypeKey &k = gctx.ctx.getTypePool().get(resolved);
		if (k.kind == TypeKind::Int) {
			inst.ty = resolved;
			return emit(gctx, inst);
		}
		// Int literal into a float destination: settle the int at a
		// width that fits, then have the consumer emit a SIToFP /
		// UIToFP. Struct-field and array-element init handle the
		// coercion silently to match what the legacy struct codegen
		// did (BuildSIToFP after a literal at expected = f32).
		// Bare `var x: f32 = 3;` is rejected at the init site, not
		// here, so cross-context literals stay permissive.
	}
	if (isNeg) {
		if (val <= 128) inst.ty = BuiltinType::I8;
		else if (val <= 32768) inst.ty = BuiltinType::I16;
		else if (val <= 2147483648ULL) inst.ty = BuiltinType::I32;
		else inst.ty = BuiltinType::I64;
	} else {
		if (val <= 255) inst.ty = BuiltinType::U8;
		else if (val <= 65535) inst.ty = BuiltinType::U16;
		else if (val <= 4294967295ULL) inst.ty = BuiltinType::U32;
		else inst.ty = BuiltinType::U64;
	}
	return emit(gctx, inst);
}

// AstGen for `StringLit`. The result type is normally a slice-of-u8
// (`[]u8`); the slice TypeKey is interned in the TypePool on first use so
// a stable TypeIdx flows through the IR. The actual {ptr,len} struct is
// materialised at jir-codegen time using the StringIdx in `a`.
//
// Pointer decay: a string literal is a NUL-terminated `u8` array constant,
// so when the use site expects a many-/single-item pointer to u8
// (`*const[] u8`, `*mut[] u8`, `*const u8`) it lowers to the bare global
// pointer instead of a fat slice. This mirrors Zig, where a literal's type
// `*const [N:0]u8` coerces to `[*]const u8` / `[*c]const u8` (Sema.zig's
// `src_array_ptr` array-pointer decay). It is what lets a literal be passed
// straight to C FFI — `snprintf(.., "n=%d", ..)` — without an explicit
// `.ptr`, while a runtime `[]u8` slice (which carries no static NUL
// guarantee) still requires `.ptr`. The decay is keyed on the EXPECTED
// type, so it fires uniformly for call args, `var` initialisers, and
// `return` values; jir_codegen emits just the global address when the Str
// inst's type is a pointer kind.
static JirRef astgenStringLit(AstGenCtx &gctx, const AstNode &n,
                              TypeIdx expected) {
	StringIdx s = static_cast<StringIdx>(n.lhs);
	TypeIdx resultTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
	if (expected != kNoType) {
		const TypeKey &ek = gctx.ctx.getTypePool().get(expected);
		if ((ek.kind == TypeKind::PtrMany || ek.kind == TypeKind::PtrSingle) &&
		    ek.a == BuiltinType::U8) {
			resultTy = expected;
		}
	}
	JirInst inst{};
	inst.tag = JirTag::Str;
	inst.a = s;
	inst.ty = resultTy;
	return emit(gctx, inst);
}

// AstGen for `BoolLit`.
static JirRef astgenBoolLit(AstGenCtx &gctx, const AstNode &n) {
	JirInst inst{};
	inst.tag = JirTag::Bool;
	inst.a = (n.lhs != 0) ? 1u : 0u;
	inst.ty = BuiltinType::Bool;
	return emit(gctx, inst);
}

// AstGen for `Return`. The arm body's tail expression (when present)
// receives the function's declared return type as expected hint.
// Drop calls for every drop-tracked binding currently in scope are
// emitted before the Ret instruction, in reverse declaration order.
// Detect `return <var>` where <var> names a tracked drop-bearing local —
// the var's ownership is moved to the caller's binding, so the function-
// scope drop must not fire for it. Returns the var's name or "" if the
// returned expression isn't a bare Variable referencing a tracked local.
// Only the simple `return v` shape is recognized — partial moves out of
// fields (`return v.field`) or struct literals (`return Foo { v: v }`)
// stay on the conservative path and fire v.drop. The struct-literal
// case is the next compiler task; once that lands here, the unsafe
// `var v = ...; return Foo { v: v }` pattern stops double-freeing too.
static std::string detectReturnMove(AstGenCtx &gctx, NodeIdx valIdx) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &returned = ns.get(valIdx);
	if (returned.tag != AstTag::Variable) return std::string();
	StringIdx nameId = static_cast<StringIdx>(returned.lhs);
	const std::string &name = gctx.ctx.getStringPool().get(nameId);
	for (const auto &scope : gctx.dropScopes) {
		for (const auto &d : scope) {
			if (d.varName == name) return name;
		}
	}
	return std::string();
}

static void astgenReturn(AstGenCtx &gctx, const AstNode &n) {
	JirRef valRef = kNoJirRef;
	std::string movedVar;
	if (n.lhs != 0) {
		NodeIdx valIdx = static_cast<NodeIdx>(n.lhs);
		movedVar = detectReturnMove(gctx, valIdx);
		// When the function returns via sret, hand the sret slot
		// pointer to the value expression as a result_ptr. Byref-
		// producing exprs (StructLit / ArrayLit / sret Call) write
		// fields directly into the slot and return kNoJirRef — no
		// temp alloca, no trailing memcpy. Byval returns still flow
		// the value through and let Ret's standard store handle it.
		TypeIdx retTy = gctx.jfn.returnType;
		// `return h.c` copies the field while the aggregate's drop glue
		// still owns it — the caller's copy and the glue would both
		// drop the payload. Reject (pure field paths only).
		rejectDropBearingFieldExtract(gctx, valIdx, retTy, "return");
		bool sretFn = retTy != kNoType &&
		              jam::abi::classifyReturn(retTy, gctx.ctx).kind ==
		                  jam::abi::ReturnABI::Kind::Indirect;
		if (sretFn &&
		    astgenExprIntoPtr(gctx, valIdx, retTy, emitSretArg(gctx, retTy))) {
			if (movedVar.empty()) {
				emitDropsThroughScope(gctx, 0);
			} else {
				emitDropsThroughScopeMovedOut(gctx, 0, movedVar);
			}
			JirInst ret{};
			ret.tag = JirTag::Ret;
			ret.a = kNoJirRef;
			emit(gctx, ret);
			return;
		}
		valRef = astgenExpr(gctx, valIdx, retTy);
	}
	// Drop every active scope before exiting the function — minus the
	// moved-out var, if any.
	if (movedVar.empty()) {
		emitDropsThroughScope(gctx, 0);
	} else {
		emitDropsThroughScopeMovedOut(gctx, 0, movedVar);
	}
	JirInst ret{};
	ret.tag = JirTag::Ret;
	ret.a = valRef;
	emit(gctx, ret);
}

// Resolve the canonical drop function name for a Jam type. Returns
// empty when the type has no drop fn registered. Routes through
// `mangledFunctionName` so the LLVM symbol picked here is the same
// one the drop fn's definition gets at codegen time (FQN like
// `T.drop` or `m.T.drop`). Falls back to the codegen context's
// instantiated-drops table so generic struct/enum instantiations
// (Vec(i32), Holder(i32), ...) fire drops too.
static std::string lookupDropFnLLVMName(JamCodegenContext &ctx, TypeIdx ty) {
	const TypeKey &k = ctx.getTypePool().get(ty);
	std::string typeName;
	if (k.kind == TypeKind::Struct || k.kind == TypeKind::Named ||
	    k.kind == TypeKind::Enum) {
		typeName = ctx.getStringPool().get(static_cast<StringIdx>(k.a));
	} else {
		return "";
	}
	// Resolve to the registered FunctionAST for the type's drop fn,
	// then defer the LLVM-symbol decision to the shared
	// `mangledFunctionName` helper. Single source of truth — emitDrops
	// and the legacy drop-emitter both look up the same way, so a
	// mangling change can't silently desync.
	auto resolveName = [&](const std::string &name) -> std::string {
		const FunctionAST *fn = nullptr;
		const jam::drops::DropRegistry *reg = ctx.getDropRegistry();
		if (reg != nullptr) {
			auto it = reg->find(name);
			if (it != reg->end()) fn = it->second;
		}
		if (fn == nullptr) fn = ctx.lookupDropFn(name);
		if (fn == nullptr) return "";
		return mangledFunctionName(*fn, ctx.getTypePool(),
		                           ctx.getStringPool());
	};
	std::string r = resolveName(typeName);
	if (!r.empty()) return r;
	TypeIdx aliasTarget = ctx.lookupTypeAlias(typeName);
	if (aliasTarget != kNoType) {
		const TypeKey &ak0 = ctx.getTypePool().get(aliasTarget);
		if (ak0.kind == TypeKind::GenericCall) {
			TypeIdx resolved = ctx.resolveGenericCall(aliasTarget);
			if (resolved != kNoType) aliasTarget = resolved;
		}
		const TypeKey &ak = ctx.getTypePool().get(aliasTarget);
		if (ak.kind == TypeKind::Named || ak.kind == TypeKind::Struct ||
		    ak.kind == TypeKind::Enum) {
			std::string aliasName =
			    ctx.getStringPool().get(static_cast<StringIdx>(ak.a));
			r = resolveName(aliasName);
			if (!r.empty()) return r;
		}
	}
	return "";
}

// Emit drop calls for every binding in `bindings` in REVERSE order
// (LIFO — last-pushed dropped first). Each binding gets the full Rust-
// style `drop_in_place::<T>` sequence: if T has its own drop fn, it
// fires first; then T's droppable fields fire in declaration order.
// Routing both the with-fn and no-fn paths through emitDropInPlace
// keeps the semantics consistent — a tracked var of any droppable
// type drops the same way whether the user wrote `cfn drop` or only
// has droppable fields.
static void emitDrops(AstGenCtx &gctx, const std::vector<DropTrack> &bindings) {
	for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
		emitDropInPlace(gctx, it->slot, it->type);
	}
}

// AstGen for `VarDecl`. Lowers to Alloca + Store(init). The variable's
// name -> alloca ref binding is recorded so subsequent reads emit
// Load(alloca).
static void astgenVarDecl(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx extra = static_cast<ExtraIdx>(n.lhs);
	StringIdx nameId = static_cast<StringIdx>(ns.getExtra(extra));
	TypeIdx declared = static_cast<TypeIdx>(ns.getExtra(extra + 1));
	NodeIdx initIdx = static_cast<NodeIdx>(ns.getExtra(extra + 2));
	const std::string &name = gctx.ctx.getStringPool().get(nameId);

	// Reject re-declaration within the same lexical scope only. We
	// inspect the innermost `localScopes` frame, so `const a = X;
	// var a = Y;` at the same level errors, but
	//     fn f() { var x = 1; if (c) { var x = 2; } }
	// still compiles — intentional inner-block shadowing is allowed.
	if (!gctx.localScopes.empty() && gctx.localScopes.back().count(name) != 0) {
		failHere(gctx, "redeclaration of `" + name + "` in the same scope");
	}

	// Type resolution.
	//   * declared (`var x: T = E;`) — lower E with T as the expected
	//     hint so literal-narrowing settles at T; verify the result
	//     type matches T.
	//   * inferred (`var x = E;`) — lower E with no hint, take the
	//     result's type as the binding's type. Self-referential inits
	//     like `var head = Node { next: &head };` need a declared
	//     type so the slot pre-exists; we reject them with a clear
	//     error when inferred.
	//
	// Inference commits to whatever single type comes back from
	// astgenExpr — sufficient for the v1 grammar (no `if`-expression
	// in value position outside `match`). Cases that would need
	// peer-typing across multiple stores into the same slot must
	// use an explicit `: T` annotation.
	TypeIdx type;
	JirRef allocaRef;
	JirRef initRef;
	if (declared == kNoType) {
		// Lower init first so we have a concrete type to allocate.
		initRef = astgenExpr(gctx, initIdx, kNoType);
		type = gctx.jfn.getInst(initRef).ty;
		if (type == kNoType) {
			failHere(gctx, "could not infer type of `" + name +
			                   "`; add an explicit `: T` annotation");
		}
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = type;
		allocaRef = emitAllocaHoisted(gctx, alloca);
		gctx.locals[name] = allocaRef;
		gctx.localTypes[name] = type;
		if (!gctx.localScopes.empty()) { gctx.localScopes.back().insert(name); }
	} else {
		// Deferred `[expr]T` annotations canonicalize here so the
		// binding's type — and everything downstream that switches on
		// TypeKind::Array (index lowering, literal-length checks) —
		// sees the resolved length.
		type = resolveDeferredArrayTy(gctx, declared);
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = type;
		allocaRef = emitAllocaHoisted(gctx, alloca);
		// Register the binding *before* lowering the initializer so
		// self-referential inits like
		//     var head: Node = Node { next: &head };
		// can find the slot for `&head`. The slot's bytes are
		// undefined until the Store below; users are responsible
		// for the resulting semantics.
		gctx.locals[name] = allocaRef;
		gctx.localTypes[name] = type;
		if (!gctx.localScopes.empty()) { gctx.localScopes.back().insert(name); }

		// Try the place-into-destination path first: StructLit and
		// sret Calls can write directly into `allocaRef`, skipping
		// the SSA aggregate (and the insertvalue chain) the value-
		// form would build. When placed, `initRef` stays kNoJirRef
		// so the trailing Store below short-circuits.
		if (astgenExprIntoPtr(gctx, initIdx, type, allocaRef)) {
			initRef = kNoJirRef;
		} else {
			initRef = astgenExpr(gctx, initIdx, type);

			// Type-check the init against the declared type. The
			// astgenNumberLit path already narrows integer literals to
			// the declared int width when `expected` is an Int, so
			// `var x: i32 = 5;` lands as ik=I32. Anything else that
			// doesn't match is a real mismatch — including
			// `var x: f32 = 3;` (int into float), `var x: bool = 1.0;`
			// (float into bool), `var x: u8 = "s";` (slice into u8),
			// etc.
			//
			// Both sides are resolved through generic-call / type-alias
			// chains first so `var a: Identity(i32) = 42;` compares
			// `i32 == i32` instead of the unresolved `GenericCall ≠ Int`.
			// Pointer-target types are also resolved per-side so
			// `var p: *const T = &x` (where `T` is an alias) matches.
			std::function<TypeIdx(TypeIdx)> resolveForCmp =
			    [&](TypeIdx t) -> TypeIdx {
				if (t == kNoType) return t;
				const TypeKey &k = gctx.ctx.getTypePool().get(t);
				// Generic substitution wins (inside an instantiated
				// method body, `T` resolves to whatever the
				// instantiation supplied).
				if (k.kind == TypeKind::Named) {
					const std::string &name = gctx.ctx.getStringPool().get(
					    static_cast<StringIdx>(k.a));
					TypeIdx sub = gctx.ctx.lookupCurrentSubst(name);
					if (sub != kNoType) return resolveForCmp(sub);
				}
				if (k.kind == TypeKind::GenericCall) {
					TypeIdx r = gctx.ctx.resolveGenericCall(t);
					if (r != kNoType) return resolveForCmp(r);
				}
				if (k.kind == TypeKind::Named) {
					const std::string &nm = gctx.ctx.getStringPool().get(
					    static_cast<StringIdx>(k.a));
					TypeIdx a = gctx.ctx.lookupTypeAlias(nm);
					if (a != kNoType) return resolveForCmp(a);
					// 3+ segment chain through module re-exports — collapse
					// `w.leaf.Point` to the canonical `Point` so it matches
					// values produced by `w.leaf.makePoint(...)` whose return
					// type was registered with the single-segment name.
					if (nm.find('.') != std::string::npos) {
						TypeIdx c = gctx.ctx.resolveChainedType(nm);
						if (c != kNoType) return resolveForCmp(c);
					}
				}
				return t;
			};
			TypeIdx initTy = gctx.jfn.getInst(initRef).ty;
			TypeIdx declRes = resolveForCmp(type);
			TypeIdx initRes = resolveForCmp(initTy);
			// Pointer-shape leniency: PtrSingle(T) and PtrMany(T) share
			// the runtime representation (a plain `ptr`), so `&arr[i]`
			// (which lowers to PtrSingle(T)) is accepted in a PtrMany(T)
			// slot as a zero-cost retag. The rule is permissive because
			// we don't currently distinguish pointer-to-array from
			// pointer-to-element at the JIR level — revisit when an
			// Array-pointer kind lands and the source can carry its
			// length statically.
			auto pointerCompatible = [&](TypeIdx a, TypeIdx b) -> bool {
				if (a == kNoType || b == kNoType) return false;
				const TypeKey &ka = gctx.ctx.getTypePool().get(a);
				const TypeKey &kb = gctx.ctx.getTypePool().get(b);
				bool aPtr = ka.kind == TypeKind::PtrSingle ||
				            ka.kind == TypeKind::PtrMany;
				bool bPtr = kb.kind == TypeKind::PtrSingle ||
				            kb.kind == TypeKind::PtrMany;
				return aPtr && bPtr && ka.a == kb.a;
			};
			bool typesMatch =
			    (declRes == initRes) || pointerCompatible(declRes, initRes);
			if (initTy != kNoType && !typesMatch) {
				const TypeKey &dk = gctx.ctx.getTypePool().get(declRes);
				const TypeKey &ik = gctx.ctx.getTypePool().get(initRes);
				// Specialize the message for the two patterns users hit
				// most often; everything else gets the generic mismatch.
				if (dk.kind == TypeKind::Float && ik.kind == TypeKind::Int) {
					failHere(gctx, "cannot assign integer to float-typed `" +
					                   name +
					                   "`; use a float literal (e.g. `3.0`) "
					                   "or an explicit `as` cast");
				}
				failHere(gctx,
				         "type mismatch in `" + name +
				             "`: declared and initialised values disagree");
			}
		}  // close: place-into-destination else
	}

	if (initRef != kNoJirRef) {
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = allocaRef;
		store.b = initRef;
		emit(gctx, store);
	}

	// `var owned = c;` with a bare drop-bearing `c` is a MOVE: the new
	// binding owns the value, the source's scope-exit drop is
	// suppressed. Without this, both bindings drop the same payload.
	// `var x = h.c` (field extraction) has no per-field suppression to
	// offer — reject it instead.
	rejectDropBearingFieldExtract(gctx, initIdx, type, "copy");
	consumeMovedVariable(gctx, initIdx);

	// If this binding's type owns heap, track it on the top of the
	// drop-scope stack so scope-exit (or the function-end Return)
	// fires the cleanup. Two flavors share the same DropTrack slot:
	//   - Types with a registered `cfn drop`: emit a direct call to
	//     that fn (existing path, `llvmFnName` non-empty).
	//   - Types whose own struct doesn't have `cfn drop` but whose
	//     fields recursively do (e.g. `Bus { dma: Vec(u32), ... }`):
	//     `llvmFnName` stays empty; emitDrops invokes emitDropInPlace
	//     to walk the fields at the drop site. Matches Rust's
	//     `needs_drop<T>()` which auto-drops a struct when any of its
	//     fields does, even without an explicit Drop impl.
	std::string dropName = lookupDropFnLLVMName(gctx.ctx, type);
	if (!dropName.empty()) {
		if (gctx.dropScopes.empty()) pushDropScope(gctx);
		gctx.dropScopes.back().push_back({name, allocaRef, type, dropName});
	} else if (typeNeedsDrop(gctx.ctx, type)) {
		// Struct without its own `cfn drop` but with droppable fields
		// (e.g. `Bus { dma: Vec(u32), ... }`). Track the binding so
		// scope-exit invokes `emitDropInPlace(slot, type)` which walks
		// fields. Safe in combination with return-move and struct-
		// literal-capture-move: the source local of every `Struct
		// { f: src }` and `return src` is consumed, so the only firing
		// is on the final owner. Matches Rust's `needs_drop<T>()`.
		if (gctx.dropScopes.empty()) pushDropScope(gctx);
		gctx.dropScopes.back().push_back(
		    {name, allocaRef, type, std::string()});
	}
}

// AstGen for `Variable` (read). Falls back to module-level constants
// when the name isn't a local — these were declared as `const X: T =
// initExpr;` at the module level and lower by re-evaluating the init
// expression at each read site. Constant folding makes this cheap;
// non-constant inits would require a runtime global slot (deferred).
static JirRef astgenVariable(AstGenCtx &gctx, const AstNode &n,
                             TypeIdx expected = kNoType) {
	const std::string &name =
	    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	auto it = gctx.locals.find(name);
	if (it != gctx.locals.end()) {
		JirInst load{};
		load.tag = JirTag::Load;
		load.a = it->second;
		load.ty = gctx.localTypes[name];
		return emit(gctx, load);
	}
	// Comp-param substitution. When the enclosing function was
	// instantiated with `comp n: u32` bound to a concrete value, refs
	// to `n` in the body lower to that constant directly — no Param /
	// Load instruction, no LLVM argument. Locals shadow comp params
	// (the local-lookup above runs first); module consts and
	// functions are checked after, so a comp-bound name takes
	// precedence over a same-named module const.
	if (const jam::ComptimeValue *cv = gctx.ctx.lookupCurrentCompSubst(name)) {
		switch (cv->kind) {
		case jam::ComptimeValue::Kind::Int: {
			JirInst ic{};
			ic.tag = JirTag::Int;
			ic.a = static_cast<uint32_t>(cv->intVal.bits & 0xFFFFFFFFu);
			ic.b = static_cast<uint32_t>(cv->intVal.bits >> 32);
			ic.ty = gctx.ctx.getTypePool().internInt(cv->intVal.width,
			                                         cv->intVal.isSigned);
			return emit(gctx, ic);
		}
		case jam::ComptimeValue::Kind::Bool: {
			JirInst ic{};
			ic.tag = JirTag::Bool;
			ic.a = cv->boolVal ? 1u : 0u;
			ic.ty = BuiltinType::Bool;
			return emit(gctx, ic);
		}
		case jam::ComptimeValue::Kind::Str: {
			JirInst si{};
			si.tag = JirTag::Str;
			si.a = cv->strVal;
			si.ty = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
			return emit(gctx, si);
		}
		case jam::ComptimeValue::Kind::Float: {
			JirInst fl{};
			fl.tag = JirTag::Float;
			uint64_t bits;
			__builtin_memcpy(&bits, &cv->floatVal.value, sizeof(bits));
			fl.a = static_cast<uint32_t>(bits & 0xFFFFFFFFu);
			fl.b = static_cast<uint32_t>(bits >> 32);
			fl.ty = gctx.ctx.getTypePool().internFloat(cv->floatVal.width);
			return emit(gctx, fl);
		}
		case jam::ComptimeValue::Kind::Type:
			// Type values have no runtime representation. Reaching
			// here means a value position used a comp param of meta-
			// type kind — pure type values shouldn't lower to JIR.
			return recoverHere(gctx,
			                   "comp param `" + name +
			                       "` is of type `type` and has no "
			                       "runtime representation",
			                   kNoType);
		case jam::ComptimeValue::Kind::Aggregate:
		case jam::ComptimeValue::Kind::None:
			return recoverHere(
			    gctx, "comp param `" + name + "` has no runtime lowering yet",
			    kNoType);
		}
	}
	if (const auto *mc = gctx.ctx.getModuleConst(name)) {
		return astgenExpr(gctx, mc->initExpr, mc->declaredType);
	}
	// Fn-name-as-value (Rust-style item coercion). The identifier
	// resolves to a function symbol — surface its address as either
	// (a) a typed function pointer when the context expects a Fn
	// type, so `var f: fn(i32) i32 = add;` and struct fields of fn
	// type get a properly-typed value, or (b) the legacy u64 when
	// the context is untyped (writes to buffers, manual casts).
	// Generic fns are rejected because no monomorphized body exists
	// at this point in lowering.
	if (const FunctionAST *fn = gctx.ctx.getFunctionAST(name)) {
		if (fn->isGeneric()) {
			return recoverHere(
			    gctx, "cannot take address of generic fn `" + name + "`",
			    kNoType);
		}
		JirInst fnref{};
		fnref.tag = JirTag::FnRef;
		fnref.a =
		    static_cast<JirRef>(gctx.ctx.getStringPool().intern(fn->Name));
		// If the consumer asked for a Fn type, give them one; otherwise
		// fall back to u64 (legacy raw-address shape). Future cleanup:
		// always emit the typed Fn and let consumers cast to u64
		// explicitly via `as`.
		bool expectFn = false;
		if (expected != kNoType) {
			const TypeKey &ek = gctx.ctx.getTypePool().get(expected);
			expectFn = ek.kind == TypeKind::Fn;
		}
		fnref.ty = expectFn ? expected : BuiltinType::U64;
		return emit(gctx, fnref);
	}
	// Recoverable: emit a Poison so the rest of the function still
	// gets analyzed (and additional errors reported in the same pass).
	return recoverHere(gctx, "unknown variable `" + name + "`", kNoType);
}

// Resolve `node` as an *lvalue* — thin wrapper for legacy callers
// over `astgenExpr(..., Pointer, &outLeafTy)`. Real Pointer-loc
// logic lives inline in `astgenExpr`'s entry switch (Variable,
// Deref, MemberAccess, Index). Non-lvalue tags here error via the
// validation switch below before reaching `astgenExpr`'s value-
// then-spill fallback, so misuse is reported as "not assignable"
// rather than silently spilling into a temporary that dies at
// expression-end.
static JirRef astgenLvalue(AstGenCtx &gctx, NodeIdx node, TypeIdx &outLeafTy) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &n = ns.get(node);
	switch (n.tag) {
	case AstTag::Variable:
	case AstTag::Deref:
	case AstTag::MemberAccess:
	case AstTag::Index:
		break;
	default:
		failNode(gctx, node, "this expression is not assignable");
	}
	TypeIdx leaf = kNoType;
	JirRef ptr = astgenExpr(gctx, node, kNoType, ResultLoc::Pointer, &leaf);
	outLeafTy = leaf;
	return ptr;
}

// AstGen for `Assign`. Two paths:
//   1. Target is an Index on a struct value (e.g. `v[i] = x` where
//      `v: Vec(i32)`). Dispatch to the struct's `cfn setAt` method —
//      that's how value-shaped indexed assignment works without
//      producing a pointer / borrow. No `astgenLvalue` step needed
//      because there's no pointer-producing JIR op involved; the
//      setter takes (recv, i, value) by value and stores internally
//      via plain slice-indexing through its own `self.ptr`.
//   2. Anything else — the original lvalue-pointer-then-Store path.
//      Variable / Deref / MemberAccess / Index on arrays-slices-
//      ptr-many all go through here.
static void astgenAssign(AstGenCtx &gctx, const AstNode &n) {
	NodeIdx targetIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx valueIdx = static_cast<NodeIdx>(n.rhs);

	// `v[i] = x` on a struct -> `v.setAt(i, x)`.
	const AstNode &target = gctx.ctx.getNodeStore().get(targetIdx);
	if (target.tag == AstTag::Index) {
		NodeIdx baseIdx = static_cast<NodeIdx>(target.lhs);
		NodeIdx idxIdx = static_cast<NodeIdx>(target.rhs);
		// Take the base as a pointer via ResultLoc::Pointer (zero-
		// cost for Variable bases — just the alloca's JirRef). The
		// dropped alternative was `astgenExpr(baseIdx)` which for
		// array bases Loads the entire backing storage as a dead
		// SSA register and stalls LLVM at -O1+.
		//
		// Limited to Variable bases for the same reason as
		// astgenIndex's fast path: astgenLvalue throws on non-
		// struct MemberAccess (e.g. `slice.ptr`). Non-Variable LHS
		// of an indexed-assign is rare; those fall through to the
		// general lvalue-store path below.
		TypeIdx baseTy = kNoType;
		const AstNode &baseNode = gctx.ctx.getNodeStore().get(baseIdx);
		if (baseNode.tag == AstTag::Variable) {
			JirRef basePtr =
			    astgenExpr(gctx, baseIdx, kNoType, ResultLoc::Pointer);
			baseTy = gctx.jfn.getInst(basePtr).ty;
		}
		const auto *sinfo =
		    baseTy != kNoType ? gctx.ctx.lookupStruct(baseTy) : nullptr;
		if (sinfo != nullptr) {
			const std::string qualified = sinfo->name + ".setAt";
			const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
			if (method == nullptr &&
			    gctx.ctx.getWithdrawnMethod(qualified) != nullptr) {
				reportMethodMiss(gctx, qualified);
				return;
			}
			if (method != nullptr && method->isCfn &&
			    method->Args.size() >= 3) {
				// setAt's self must be mut/move (it mutates), so we
				// always hand the method a *Self pointer from
				// astgenLvalue. The peek above already restricted
				// the base shape to Variable, so this lvalue lookup
				// is just an alloca table read.
				ParamMode mode = method->Args[0].Mode;
				if (mode != ParamMode::Mut && mode != ParamMode::Move) {
					failHere(gctx, "astgen: cfn setAt on `" + sinfo->name +
					                   "` must take `self: mut Self`");
				}
				TypeIdx leafTyR = kNoType;
				JirRef recv = astgenLvalue(gctx, baseIdx, leafTyR);
				JirRef idxRef = astgenExpr(gctx, idxIdx, BuiltinType::U64);
				TypeIdx valParamTy = method->Args[2].Type;
				JirRef valRef = astgenExpr(gctx, valueIdx, valParamTy);
				emitStructCfnDispatch(gctx, sinfo, "setAt", recv, idxRef,
				                      {valRef});
				// A bare drop-bearing local/`move`-param stored into a
				// container element is MOVED — the element owns it now.
				rejectDropBearingFieldExtract(gctx, valueIdx, valParamTy,
				                              "copy");
				consumeMovedVariable(gctx, valueIdx);
				return;
			}
		}
		// Not a struct, or no `cfn setAt` defined — fall through.
	}

	TypeIdx leafTy = kNoType;
	JirRef ptrRef = astgenLvalue(gctx, targetIdx, leafTy);
	JirRef valRef = astgenExpr(gctx, valueIdx, leafTy);
	// Overwrite drop: storing over a live drop-bearing VALUE-world
	// destination (whole local, struct field, fixed-array element)
	// drops the old value first — rustc's assignment semantics.
	// Pointer-world destinations stay raw stores (see
	// assignTargetIsValueWorld). Order matches Rust: RHS evaluates
	// first, old value drops, new value stores. Reassign-after-MOVE is
	// already rejected by init_analysis, so the destination here is
	// never a moved-out slot.
	if (leafTy != kNoType && typeNeedsDrop(gctx.ctx, leafTy) &&
	    assignTargetIsValueWorld(gctx, targetIdx)) {
		emitDropInPlace(gctx, ptrRef, leafTy);
	}
	JirInst store{};
	store.tag = JirTag::Store;
	store.a = ptrRef;
	store.b = valRef;
	emit(gctx, store);
	// A bare drop-bearing local/`move`-param on the RHS is MOVED into
	// the destination — the destination owns it now (`self.ptr[i] =
	// value` in Vec.push transfers the pushed element; without this the
	// move param's own drop would double-free it). Plain copies of
	// non-drop values are untouched: consumeMovedVariable only affects
	// tracked drop-bearing bindings. Field extraction on the RHS
	// (`x = h.c`) has no per-field suppression — reject.
	rejectDropBearingFieldExtract(gctx, valueIdx, leafTy, "copy");
	consumeMovedVariable(gctx, valueIdx);
}

// AstGen for `StructLit`. The struct value is built as an SSA aggregate
// via a chain of InsertValue at codegen time. The JIR carries the
// field *values* in source order; field names are resolved against the
// struct's declared field order to permute the value list into LLVM's
// positional layout.
static JirRef astgenStructLit(AstGenCtx &gctx, const AstNode &n,
                              TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	TypeIdx ty = static_cast<TypeIdx>(n.lhs);
	if (ty == kNoType) ty = expected;
	if (ty == kNoType) {
		failHere(gctx, "astgen: struct literal without target type");
	}

	// Union literal: exactly one field, stored into a slot of the
	// union's full size. The runtime form is the chosen field's
	// type; reading other fields reinterprets the bits — that's the
	// union's whole job.
	if (const auto *uinfo = gctx.ctx.lookupUnion(ty)) {
		ExtraIdx fieldsExtra = static_cast<ExtraIdx>(n.rhs);
		uint32_t fieldCount = ns.getExtra(fieldsExtra);
		if (fieldCount != 1) {
			failHere(gctx, "astgen: union literal must list exactly one field");
		}
		StringIdx nameId = static_cast<StringIdx>(ns.getExtra(fieldsExtra + 1));
		NodeIdx exprIdx = static_cast<NodeIdx>(ns.getExtra(fieldsExtra + 2));
		const std::string &fieldName = gctx.ctx.getStringPool().get(nameId);
		TypeIdx fieldTy = gctx.ctx.getUnionFieldType(uinfo->name, fieldName);
		if (fieldTy == kNoType) {
			failHere(gctx, "astgen: union `" + uinfo->name +
			                   "` has no field `" + fieldName + "`");
		}
		JirRef fieldVal = astgenExpr(gctx, exprIdx, fieldTy);
		// Alloca the union, store the field's value into its slot,
		// load the whole union back as an aggregate value.
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = ty;
		JirRef slot = emitAllocaHoisted(gctx, alloca);
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = slot;
		store.b = fieldVal;
		emit(gctx, store);
		JirInst load{};
		load.tag = JirTag::Load;
		load.a = slot;
		load.ty = ty;
		return emit(gctx, load);
	}

	const auto *info = gctx.ctx.lookupStruct(ty);
	if (info == nullptr) {
		const TypeKey &tk = gctx.ctx.getTypePool().get(ty);
		if (tk.kind == TypeKind::Named || tk.kind == TypeKind::Struct) {
			const std::string &name =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(tk.a));
			// Qualified names (`lib.Private`) get the precise
			// "not exported" / "does not exist" / "unknown handle"
			// diagnostic; bare names fall back to the generic message.
			if (name.find('.') != std::string::npos) {
				failHere(gctx,
				         gctx.ctx.formatNamespaceLookupError("struct", name));
			}
			failHere(gctx, "unknown struct `" + name + "`");
		}
		failHere(gctx, "astgen: struct literal type is not a known struct");
	}

	ExtraIdx fieldsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t fieldCount = ns.getExtra(fieldsExtra);

	// Compile each named field; permute into positional order.
	std::vector<JirRef> ordered(info->fields.size(), kNoJirRef);
	for (uint32_t i = 0; i < fieldCount; i++) {
		StringIdx nameId =
		    static_cast<StringIdx>(ns.getExtra(fieldsExtra + 1 + i * 2));
		NodeIdx exprIdx =
		    static_cast<NodeIdx>(ns.getExtra(fieldsExtra + 2 + i * 2));
		const std::string &fieldName = gctx.ctx.getStringPool().get(nameId);
		int idx = gctx.ctx.getFieldIndex(info->name, fieldName);
		if (idx < 0) {
			// Recoverable: record the bad field name and skip it so
			// other malformed fields in the same literal still get
			// reported in this pass.
			appendErrorHere(gctx, "unknown struct field `" + fieldName + "`");
			continue;
		}
		TypeIdx expectedField =
		    resolveDeferredArrayTy(gctx, info->fields[idx].second);
		rejectDropBearingFieldExtract(gctx, exprIdx, expectedField, "capture");
		JirRef fieldVal = astgenExpr(gctx, exprIdx, expectedField);
		// Struct-literal field capture is a MOVE for drop-bearing types:
		// if the field's value came from a tracked local Variable, mark
		// that local as moved so its scope-exit drop doesn't fire — the
		// struct field now owns the value, and dropping the original
		// would double-free what the field still references.
		consumeMovedVariable(gctx, exprIdx);
		// Silent int->float widening matches the legacy struct codegen:
		// `Vec3 { x: 0 }` with `x: f32` lands an integer literal here;
		// emit SIToFP / UIToFP to settle the IR type instead of letting
		// jir_codegen pipe an integer value into a float slot of the
		// InsertValue chain.
		TypeIdx vt = gctx.jfn.getInst(fieldVal).ty;
		if (vt != expectedField && vt != kNoType) {
			const TypeKey &fk = gctx.ctx.getTypePool().get(expectedField);
			const TypeKey &vk = gctx.ctx.getTypePool().get(vt);
			if (fk.kind == TypeKind::Float && vk.kind == TypeKind::Int) {
				JirInst c{};
				c.tag = vk.b != 0 ? JirTag::SIToFP : JirTag::UIToFP;
				c.a = fieldVal;
				c.ty = expectedField;
				fieldVal = emit(gctx, c);
			}
		}
		ordered[idx] = fieldVal;
	}
	// Any positional slot not assigned? Fail loudly — full struct
	// literals must initialise every field.
	for (size_t i = 0; i < ordered.size(); i++) {
		if (ordered[i] == kNoJirRef) {
			failHere(gctx, "astgen: struct literal missing field `" +
			                   info->fields[i].first + "`");
		}
	}

	std::vector<uint32_t> packed;
	packed.reserve(1 + ordered.size());
	packed.push_back(static_cast<uint32_t>(ordered.size()));
	for (JirRef r : ordered) packed.push_back(r);
	JirExtraIdx extra = gctx.jfn.pushExtra(packed.data(), packed.size());

	JirInst inst{};
	inst.tag = JirTag::StructLit;
	inst.b = extra;
	inst.ty = ty;
	return emit(gctx, inst);
}

// Write a struct literal directly into `destPtr` (a pointer to the
// struct slot) instead of building an SSA aggregate via an
// InsertValue chain. For each field we emit FieldAddr → recursive
// place into the field's slot (StructLit / sret Call) or a value
// compile + Store fallback for everything else.
//
// Why this matters for build perf: when an outer `var b: Bus = Bus
// { ..., cdrom: Cdrom.init() }` would otherwise build %Bus via 22
// InsertValue, after O3 inlining + SROA every multi-byte field inside
// the struct gets decomposed into per-byte InsertValue (a [2352]u8
// field expands into 2352 of them). AArch64 ISel's DAGCombine then
// goes quadratic on the resulting store chain. Routing through
// per-field stores into the destination keeps those large aggregates
// in memory rather than in SSA registers.
static void astgenStructLitInto(AstGenCtx &gctx, const AstNode &n,
                                TypeIdx expectedTy, JirRef destPtr) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	TypeIdx ty = static_cast<TypeIdx>(n.lhs);
	if (ty == kNoType) ty = expectedTy;
	if (ty == kNoType) {
		failHere(gctx, "astgen: struct literal without target type");
	}

	// Unions and non-struct types: fall back to value-form StructLit
	// + Store. Unions are small enough that the perf pathology doesn't
	// trigger, and they have specialized lowering already.
	const auto *info = gctx.ctx.lookupStruct(ty);
	if (info == nullptr || gctx.ctx.lookupUnion(ty) != nullptr) {
		JirRef val = astgenStructLit(gctx, n, expectedTy);
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = destPtr;
		store.b = val;
		emit(gctx, store);
		return;
	}

	ExtraIdx fieldsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t fieldCount = ns.getExtra(fieldsExtra);

	// Map declared-field index -> source NodeIdx of the initializer.
	std::vector<NodeIdx> exprByIdx(info->fields.size(), 0);
	std::vector<uint8_t> hasField(info->fields.size(), 0);
	for (uint32_t i = 0; i < fieldCount; i++) {
		StringIdx nameId =
		    static_cast<StringIdx>(ns.getExtra(fieldsExtra + 1 + i * 2));
		NodeIdx exprIdx =
		    static_cast<NodeIdx>(ns.getExtra(fieldsExtra + 2 + i * 2));
		const std::string &fieldName = gctx.ctx.getStringPool().get(nameId);
		int idx = gctx.ctx.getFieldIndex(info->name, fieldName);
		if (idx < 0) {
			appendErrorHere(gctx, "unknown struct field `" + fieldName + "`");
			continue;
		}
		exprByIdx[idx] = exprIdx;
		hasField[idx] = 1;
	}

	for (size_t i = 0; i < info->fields.size(); i++) {
		if (!hasField[i]) {
			failHere(gctx, "astgen: struct literal missing field `" +
			                   info->fields[i].first + "`");
		}
		TypeIdx expectedField =
		    resolveDeferredArrayTy(gctx, info->fields[i].second);
		rejectDropBearingFieldExtract(gctx, exprByIdx[i], expectedField,
		                              "capture");
		TypeIdx fieldPtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, expectedField, 0});
		JirInst fieldAddr{};
		fieldAddr.tag = JirTag::FieldAddr;
		fieldAddr.a = destPtr;
		fieldAddr.b = static_cast<uint32_t>(i);
		fieldAddr.ty = fieldPtrTy;
		JirRef fieldPtr = emit(gctx, fieldAddr);

		// Try the place path (recursive StructLit or sret Call with
		// destPtr forwarded). Fall back to value-compile + Store for
		// anything else.
		if (!astgenExprIntoPtr(gctx, exprByIdx[i], expectedField, fieldPtr)) {
			JirRef val = astgenExpr(gctx, exprByIdx[i], expectedField);
			// Silent int->float widening — mirrors astgenStructLit.
			TypeIdx vt = gctx.jfn.getInst(val).ty;
			if (vt != expectedField && vt != kNoType) {
				const TypeKey &fk = gctx.ctx.getTypePool().get(expectedField);
				const TypeKey &vk = gctx.ctx.getTypePool().get(vt);
				if (fk.kind == TypeKind::Float && vk.kind == TypeKind::Int) {
					JirInst c{};
					c.tag = vk.b != 0 ? JirTag::SIToFP : JirTag::UIToFP;
					c.a = val;
					c.ty = expectedField;
					val = emit(gctx, c);
				}
			}
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = fieldPtr;
			store.b = val;
			emit(gctx, store);
		}
		// Mirror the byval path: a tracked-local Variable used as a
		// field initializer is MOVED into the struct field; suppress
		// its scope-exit drop. Applies whether the lowering went
		// through astgenExpr+Store or the byref astgenExprIntoPtr fast
		// path.
		consumeMovedVariable(gctx, exprByIdx[i]);
	}
}

// Try to lower an expression directly into `destPtr`. Returns true on
// success — when the destination has been written and no SSA value
// remains for the caller to bind. Returns false to let the caller
// take the standard astgenExpr + Store path.
//
// Recognises:
//   - StructLit              -> per-field FieldAddr + Store into destPtr
//   - ArrayLit / ArrayRepeat -> per-element IndexAddr + Store into destPtr
//   - Call / TypeMethodCall  -> forward destPtr as the sret slot so
//                                the callee writes its result through
//                                the caller's storage (no temp alloca)
//
// Everything else (scalars, ptr-typed values, small aggregates that
// the ABI passes by value) returns false. The pathology this helper
// dodges only kicks in for large-aggregate stores anyway.
static bool astgenExprIntoPtr(AstGenCtx &gctx, NodeIdx exprIdx,
                              TypeIdx expectedTy, JirRef destPtr) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &n = ns.get(exprIdx);
	switch (n.tag) {
	case AstTag::StructLit: {
		astgenStructLitInto(gctx, n, expectedTy, destPtr);
		return true;
	}
	case AstTag::ArrayLit: {
		// Per-element write into destPtr. Element type comes from
		// the surrounding context or from compiling the first elem.
		expectedTy = resolveDeferredArrayTy(gctx, expectedTy);
		TypeIdx elemTy = static_cast<TypeIdx>(n.lhs);
		if (elemTy == kNoType && expectedTy != kNoType) {
			const TypeKey &ek = gctx.ctx.getTypePool().get(expectedTy);
			if (ek.kind == TypeKind::Array) {
				elemTy = static_cast<TypeIdx>(ek.a);
			}
		}
		if (elemTy == kNoType) return false;
		ExtraIdx elemsExtra = static_cast<ExtraIdx>(n.rhs);
		uint32_t count = ns.getExtra(elemsExtra);
		// Destination is a fixed-size array: the literal must supply
		// exactly that many elements, or the per-element stores below
		// would write past (or short of) the destination slot.
		if (expectedTy != kNoType) {
			const TypeKey &ek = gctx.ctx.getTypePool().get(expectedTy);
			if (ek.kind == TypeKind::Array && ek.b != count) {
				failHere(gctx, "array literal has " + std::to_string(count) +
				                   " element(s) but the array type expects " +
				                   std::to_string(ek.b));
			}
		}
		TypeIdx u64Ty = BuiltinType::U64;
		TypeIdx elemPtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, elemTy, 0});
		for (uint32_t i = 0; i < count; i++) {
			NodeIdx eIdx =
			    static_cast<NodeIdx>(ns.getExtra(elemsExtra + 1 + i));
			JirInst idxInst{};
			idxInst.tag = JirTag::Int;
			idxInst.a = i;
			idxInst.ty = u64Ty;
			JirRef idxRef = emit(gctx, idxInst);
			JirInst ia{};
			ia.tag = JirTag::IndexAddr;
			ia.a = destPtr;
			ia.b = idxRef;
			ia.ty = elemPtrTy;
			JirRef ep = emit(gctx, ia);
			if (!astgenExprIntoPtr(gctx, eIdx, elemTy, ep)) {
				JirRef ev = astgenExpr(gctx, eIdx, elemTy);
				JirInst st{};
				st.tag = JirTag::Store;
				st.a = ep;
				st.b = ev;
				emit(gctx, st);
			}
			// Array elements own their values: a bare drop-bearing
			// local moves in (mirrors the value-path ArrayLit and
			// struct-literal capture).
			rejectDropBearingFieldExtract(gctx, eIdx, elemTy, "capture");
			consumeMovedVariable(gctx, eIdx);
		}
		return true;
	}
	// ArrayRepeat (`[expr; N]`) deliberately falls through to the
	// astgenExpr + Store path. Unrolling N stores at astgen time
	// bloats JIR (`[0; 2352]` becomes 2352 IndexAddr+Store JIR
	// instructions); JirTag::ArrayLit's byref codegen already emits
	// the same per-element stores at LLVM-IR time but keeps the JIR
	// itself compact. The trailing memcpy that the value-path leaves
	// behind is one extra instruction LLVM cleans up — much cheaper
	// than inflating astgen's per-fn IR-build cost.
	case AstTag::Call: {
		JirRef r = astgenCall(gctx, n, destPtr);
		// kNoJirRef -> emitCall used destPtr as the sret slot and the
		// result has been written in-place. A real ref -> the call
		// had a Direct (ByValue) return, so we finish the place by
		// storing the produced SSA value. Either way the field has
		// been written; signal handled.
		if (r != kNoJirRef) {
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = destPtr;
			store.b = r;
			emit(gctx, store);
		}
		return true;
	}
	case AstTag::TypeMethodCall: {
		JirRef r = astgenTypeMethodCall(gctx, n, destPtr);
		if (r != kNoJirRef) {
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = destPtr;
			store.b = r;
			emit(gctx, store);
		}
		return true;
	}
	default:
		return false;
	}
}

// AstGen for `MemberAccess`. Two cases:
//   1. `EnumName.Variant` — when `base` is a Variable whose source-level
//      name happens to be a registered enum, lower to a unit-variant
//      constructor (tag i8 for unit-only enums; full {tag,payload}
//      struct for payloaded enums).
//   2. Regular struct field projection — emit FieldAccess against the
//      loaded aggregate value.
static JirRef astgenMemberAccess(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
	StringIdx memberId = static_cast<StringIdx>(n.rhs);
	const std::string &member = gctx.ctx.getStringPool().get(memberId);

	const AstNode &baseNode = ns.get(baseIdx);
	if (baseNode.tag == AstTag::Variable) {
		const std::string &baseName =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(baseNode.lhs));
		// Enum-variant unit reference: emit JirTag::StructLit with a
		// single field (the tag) for payloaded enums; for unit-only
		// enums, the tag value IS the runtime form (i8).
		if (const auto *einfo = gctx.ctx.getEnum(baseName)) {
			int vidx = gctx.ctx.getEnumVariantIndex(baseName, member);
			if (vidx < 0) {
				failHere(gctx, "astgen: enum `" + baseName +
				                   "` has no variant `" + member + "`");
			}
			uint32_t disc = einfo->variants[vidx].discriminant;
			TypeIdx enumTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::Named, 0, 0,
			            static_cast<uint32_t>(
			                gctx.ctx.getStringPool().intern(baseName)),
			            0});
			JirInst tag{};
			tag.tag = JirTag::Int;
			tag.a = disc;
			tag.ty = BuiltinType::U8;
			JirRef tagRef = emit(gctx, tag);
			if (!einfo->hasPayloadVariant) { return tagRef; }
			// Payloaded enum: build a {tag, payload-undef} struct.
			// We carry the enum's TypeIdx so codegen materialises the
			// right struct shape.
			std::vector<uint32_t> packed = {1, tagRef};
			JirExtraIdx extra =
			    gctx.jfn.pushExtra(packed.data(), packed.size());
			JirInst sl{};
			sl.tag = JirTag::StructLit;
			sl.b = extra;
			sl.ty = enumTy;
			return emit(gctx, sl);
		}
	}

	// Addressable base -> FieldAddr + Load fieldTy. The value path
	// below would emit Load wholeStruct + ExtractValue, which for a
	// struct embedding large arrays scales as O(structSize) per field
	// read — a method reading N fields from a 1KB self struct emitted
	// N copies of a 1KB load.
	bool baseIsLvalueable = baseNode.tag == AstTag::Variable ||
	                        baseNode.tag == AstTag::MemberAccess ||
	                        baseNode.tag == AstTag::Index ||
	                        baseNode.tag == AstTag::Deref;
	if (baseIsLvalueable) {
		TypeIdx baseLeafTy = kNoType;
		JirRef basePtr = astgenLvalue(gctx, baseIdx, baseLeafTy);
		if (const auto *uinfo = gctx.ctx.lookupUnion(baseLeafTy)) {
			TypeIdx fieldTy = gctx.ctx.getUnionFieldType(uinfo->name, member);
			if (fieldTy == kNoType) {
				failHere(gctx, "astgen: union `" + uinfo->name +
				                   "` has no field `" + member + "`");
			}
			TypeIdx fieldPtrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
			JirInst bc{};
			bc.tag = JirTag::BitCast;
			bc.a = basePtr;
			bc.ty = fieldPtrTy;
			JirRef fp = emit(gctx, bc);
			JirInst ld{};
			ld.tag = JirTag::Load;
			ld.a = fp;
			ld.ty = fieldTy;
			return emit(gctx, ld);
		}
		if (const auto *info = gctx.ctx.lookupStruct(baseLeafTy)) {
			int idx = gctx.ctx.getFieldIndex(info->name, member);
			if (idx < 0) {
				return recoverHere(gctx,
				                   "unknown field `" + member + "` on `" +
				                       info->name + "`",
				                   kNoType);
			}
			TypeIdx fieldTy =
			    resolveDeferredArrayTy(gctx, info->fields[idx].second);
			TypeIdx fieldPtrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
			JirInst fa{};
			fa.tag = JirTag::FieldAddr;
			fa.a = basePtr;
			fa.b = static_cast<uint32_t>(idx);
			fa.ty = fieldPtrTy;
			JirRef fp = emit(gctx, fa);
			JirInst ld{};
			ld.tag = JirTag::Load;
			ld.a = fp;
			ld.ty = fieldTy;
			return emit(gctx, ld);
		}
		// Slice base (.ptr / .len), or some other shape — fall
		// through to the value path. The astgenLvalue call above
		// emitted an alloca/GEP lookup that may go unused; small
		// cost relative to the win for struct/union bases.
	}

	// Regular struct field projection.
	JirRef baseRef = astgenExpr(gctx, baseIdx, kNoType);
	TypeIdx baseTy = gctx.jfn.getInst(baseRef).ty;

	// Slice `.ptr` / `.len` projection — the slice runtime form is a
	// {ptr, len} struct so this is just an ExtractValue at field 0 / 1.
	const TypeKey &bk = gctx.ctx.getTypePool().get(baseTy);
	if (bk.kind == TypeKind::Slice) {
		if (member != "ptr" && member != "len") {
			failHere(gctx, "astgen: slice has no field `" + member + "`");
		}
		unsigned fieldIdx = (member == "ptr") ? 0 : 1;
		TypeIdx fieldTy;
		if (fieldIdx == 0) {
			fieldTy = gctx.ctx.getTypePool().intern(TypeKey{
			    TypeKind::PtrMany, 0, 0, static_cast<uint32_t>(bk.a), 0});
		} else {
			fieldTy = BuiltinType::U64;
		}
		JirInst inst{};
		inst.tag = JirTag::ExtractValue;
		inst.a = baseRef;
		inst.b = fieldIdx;
		inst.ty = fieldTy;
		return emit(gctx, inst);
	}

	// Union field read: reinterpret the union's bits as the chosen
	// field's type. Spill the SSA value to an alloca and Load at
	// fieldTy — opaque pointers let us hand back any field-typed
	// value from the same storage.
	if (const auto *uinfo = gctx.ctx.lookupUnion(baseTy)) {
		TypeIdx fieldTy = gctx.ctx.getUnionFieldType(uinfo->name, member);
		if (fieldTy == kNoType) {
			failHere(gctx, "astgen: union `" + uinfo->name +
			                   "` has no field `" + member + "`");
		}
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = baseTy;
		JirRef slot = emitAllocaHoisted(gctx, alloca);
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = slot;
		store.b = baseRef;
		emit(gctx, store);
		JirInst load{};
		load.tag = JirTag::Load;
		load.a = slot;
		load.ty = fieldTy;
		return emit(gctx, load);
	}

	const auto *info = gctx.ctx.lookupStruct(baseTy);
	if (info == nullptr) {
		failHere(gctx, "astgen: cannot access field of non-struct type");
	}
	int idx = gctx.ctx.getFieldIndex(info->name, member);
	if (idx < 0) {
		return recoverHere(
		    gctx, "unknown field `" + member + "` on `" + info->name + "`",
		    kNoType);
	}
	JirInst inst{};
	inst.tag = JirTag::FieldAccess;
	inst.a = baseRef;
	inst.b = static_cast<uint32_t>(idx);
	inst.ty = resolveDeferredArrayTy(gctx, info->fields[idx].second);
	return emit(gctx, inst);
}

// AstGen for `ArrayLit`. SSA array value built via a chain of
// InsertValue at codegen time. The target element type (`n.lhs`) may
// be unset; in that case we infer it from the first element.
static JirRef astgenArrayLit(AstGenCtx &gctx, const AstNode &n,
                             TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	expected = resolveDeferredArrayTy(gctx, expected);
	TypeIdx elemTy = static_cast<TypeIdx>(n.lhs);
	if (elemTy == kNoType && expected != kNoType) {
		const TypeKey &ek = gctx.ctx.getTypePool().get(expected);
		if (ek.kind == TypeKind::Array) elemTy = static_cast<TypeIdx>(ek.a);
	}
	ExtraIdx elemsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t count = ns.getExtra(elemsExtra);

	// When the destination type is a fixed-size array, the literal must
	// supply exactly that many elements. Without this check the literal
	// keeps its own length and the store writes past (or short of) the
	// destination slot.
	if (expected != kNoType) {
		const TypeKey &ek = gctx.ctx.getTypePool().get(expected);
		if (ek.kind == TypeKind::Array && ek.b != count) {
			failHere(gctx, "array literal has " + std::to_string(count) +
			                   " element(s) but the array type expects " +
			                   std::to_string(ek.b));
		}
	}

	std::vector<JirRef> elems;
	elems.reserve(count);
	for (uint32_t i = 0; i < count; i++) {
		NodeIdx e = static_cast<NodeIdx>(ns.getExtra(elemsExtra + 1 + i));
		elems.push_back(astgenExpr(gctx, e, elemTy));
		// A bare drop-bearing local used as an array element is a MOVE
		// — the array owns it now (same rule as struct-literal field
		// capture). Field extraction has no per-field suppression and
		// is rejected.
		rejectDropBearingFieldExtract(gctx, e, elemTy, "capture");
		consumeMovedVariable(gctx, e);
	}
	if (elemTy == kNoType && !elems.empty()) {
		elemTy = gctx.jfn.getInst(elems[0]).ty;
	}
	if (elemTy == kNoType) {
		failHere(gctx,
		         "astgen: array literal element type could not be inferred");
	}
	TypeIdx arrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::Array, 0, 0, elemTy, count});

	std::vector<uint32_t> packed;
	packed.reserve(1 + elems.size());
	packed.push_back(count);
	for (JirRef r : elems) packed.push_back(r);
	JirExtraIdx extra = gctx.jfn.pushExtra(packed.data(), packed.size());

	JirInst inst{};
	inst.tag = JirTag::ArrayLit;
	inst.b = extra;
	inst.ty = arrTy;
	return emit(gctx, inst);
}

// AstGen for `ArrayRepeat` (`[expr; N]`). Expands the count at astgen
// time when it's a constant integer literal, lowering to an ArrayLit
// with `N` copies of the same JirRef. Non-constant counts are
// rejected — they aren't supported by the legacy path either.
static JirRef astgenArrayRepeat(AstGenCtx &gctx, const AstNode &n,
                                TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	TypeIdx arrTy = static_cast<TypeIdx>(n.lhs);
	if (arrTy == kNoType) arrTy = expected;
	arrTy = resolveDeferredArrayTy(gctx, arrTy);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	NodeIdx valueIdx = static_cast<NodeIdx>(ns.getExtra(extra));
	NodeIdx countIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1));

	// Literal counts read straight off the node; anything else
	// (`[0; SIZE]`, `[0; 2 * 1024]`) comptime-folds against the
	// module-const scope — mirroring the deferred `[expr]T` type
	// position so both spellings stay in sync.
	const AstNode &cn = ns.get(countIdx);
	uint64_t count = 0;
	if (cn.tag == AstTag::NumberLit) {
		count = static_cast<uint64_t>(cn.lhs) |
		        (static_cast<uint64_t>(cn.rhs) << 32);
	} else {
		jam::ComptimeValue v = gctx.ctx.foldComptimeExpr(countIdx);
		if (v.isNone()) {
			failHere(gctx, "array-repeat count must be comptime-known");
		}
		if (!v.isInt() || (v.intVal.isSigned && v.asI64() < 0)) {
			failHere(gctx, "array-repeat count must be a non-negative integer");
		}
		count = v.asU64();
	}
	if (count > UINT32_MAX) {
		// A negative fold wraps to a huge unsigned value; report it as
		// the sign error it is rather than a baffling range overflow.
		if (static_cast<int64_t>(count) < 0) {
			failHere(gctx, "array-repeat count must be a non-negative integer");
		}
		failHere(gctx, "array repeat count " + std::to_string(count) +
		                   " exceeds u32 range");
	}

	// Resolve element type. From the array TypeKey if we have it; else
	// from the first compile of the value. When the destination is a
	// fixed-size array, the repeat count must match its length exactly —
	// the memset/store below sizes from the count, so a mismatch writes
	// past (or short of) the destination slot.
	TypeIdx elemTy = kNoType;
	if (arrTy != kNoType) {
		const TypeKey &k = gctx.ctx.getTypePool().get(arrTy);
		if (k.kind == TypeKind::Array) {
			elemTy = static_cast<TypeIdx>(k.a);
			if (k.b != count) {
				failHere(gctx, "array repeat count " + std::to_string(count) +
				                   " does not match array type length " +
				                   std::to_string(k.b));
			}
		}
	}
	JirRef val = astgenExpr(gctx, valueIdx, elemTy);
	if (elemTy == kNoType) elemTy = gctx.jfn.getInst(val).ty;
	if (arrTy == kNoType) {
		arrTy = gctx.ctx.getTypePool().intern(TypeKey{
		    TypeKind::Array, 0, 0, elemTy, static_cast<uint32_t>(count)});
	}
	// A repeat literal duplicates ONE value into N slots — for a
	// drop-bearing element type that is N owners of one payload, every
	// extra copy a future double-free. rustc requires `T: Copy` for
	// `[x; N]`; jam rejects until explicit clone() lands.
	if (typeNeedsDrop(gctx.ctx, elemTy)) {
		failHere(gctx,
		         "repeat literal would create " + std::to_string(count) +
		             " owners of one drop-bearing value; initialize each "
		             "element explicitly");
	}

	// Zig-style fill: a constant byte fill (especially `[0; N]`)
	// lowers to one memset instead of N unrolled IndexAddr+Store.
	// memset covers any zero fill (zeros every element type) and a
	// byte-element fill; other constant fills use the per-element path.
	{
		const JirInst &vinst = gctx.jfn.getInst(val);
		if (vinst.tag == JirTag::Int) {
			uint64_t fv = static_cast<uint64_t>(vinst.a) |
			              (static_cast<uint64_t>(vinst.b) << 32);
			uint64_t elemSz = gctx.ctx.typeSize(elemTy);
			if (fv == 0 || (elemSz == 1 && fv <= 255)) {
				JirInst al{};
				al.tag = JirTag::Alloca;
				al.ty = arrTy;
				JirRef slot = emitAllocaHoisted(gctx, al);
				JirInst ms{};
				ms.tag = JirTag::MemSet;
				ms.a = slot;
				ms.b = static_cast<uint32_t>(count * elemSz);
				ms.flags = static_cast<uint16_t>(fv & 0xFF);
				emit(gctx, ms);
				return slot;
			}
		}
	}
	std::vector<uint32_t> packed;
	packed.reserve(1 + count);
	packed.push_back(static_cast<uint32_t>(count));
	for (uint64_t i = 0; i < count; i++) packed.push_back(val);
	JirExtraIdx ext = gctx.jfn.pushExtra(packed.data(), packed.size());

	JirInst inst{};
	inst.tag = JirTag::ArrayLit;
	inst.b = ext;
	inst.ty = arrTy;
	return emit(gctx, inst);
}

// Look up and call a struct's `cfn` method by name. Returns the
// call's result JirRef (whatever type the method returns), or
// kNoJirRef when the struct doesn't define a matching cfn method.
// Used by the `v[i]` desugar to dispatch to `at` (rvalue read) and
// the `v[i] = x` desugar to dispatch to `setAt` (lvalue write).
//
// Both methods are value-shaped: `at(self, i) T` returns the
// element by value (no pointer); `setAt(self: mut Self, i, value)`
// performs the write. No pointer / address appears in any signature
// — the language's borrow-free MVS model stays intact.
//
// `recv` must be a `*Self` pointer for mut/move self, or the Self
// value for let/const self. Caller decides whether to re-lower or
// spill based on the base AST shape. `extraArgs` holds any args
// beyond (recv, idx) — empty for `at`, single-element [value] for
// `setAt`.
static JirRef emitStructCfnDispatch(AstGenCtx &gctx,
                                    const JamCodegenContext::StructInfo *sinfo,
                                    const char *methodName, JirRef recv,
                                    JirRef idx,
                                    const std::vector<JirRef> &extraArgs) {
	std::string qualified = std::string(sinfo->name) + "." + methodName;
	const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
	// Method must be declared `cfn` to opt into the compiler's
	// index-syntax dispatch. A plain `fn at` / `fn setAt` is just
	// an ordinary instance method, called explicitly by the user.
	if (method == nullptr || !method->isCfn ||
	    method->Args.size() < 2 + extraArgs.size()) {
		return kNoJirRef;
	}
	// Narrow the U64-typed index to the method's declared index
	// parameter width (conventionally u32).
	TypeIdx idxParamTy = method->Args[1].Type;
	const TypeKey &idxKey = gctx.ctx.getTypePool().get(idxParamTy);
	TypeIdx currentTy = gctx.jfn.getInst(idx).ty;
	if (currentTy != idxParamTy && idxKey.kind == TypeKind::Int) {
		const TypeKey &curKey = gctx.ctx.getTypePool().get(currentTy);
		JirInst conv{};
		conv.tag = (idxKey.a < curKey.a) ? JirTag::Trunc : JirTag::ZExt;
		conv.a = idx;
		conv.ty = idxParamTy;
		idx = emit(gctx, conv);
	}
	std::vector<JirRef> argRefs;
	argRefs.reserve(2 + extraArgs.size());
	argRefs.push_back(recv);
	argRefs.push_back(idx);
	for (JirRef r : extraArgs) argRefs.push_back(r);
	return emitCall(gctx, method, argRefs);
}

// AstGen for `Index`. Lowers to JirTag::Index whose codegen handles
// the GEP+Load (for stored Variables/Arrays) or alloca-spill (for
// SSA aggregates) shape. Element type comes from the base's TypeKey.
// Struct receivers route through `at(self, i)` — see
// `emitStructIndexAtCall`.
//
// Statically determine, without emitting any JIR, whether `node` is
// an addressable expression chain whose leaf type is Array — and if
// so, return that Array TypeIdx (else kNoType). Walks the AST
// structurally: Variable -> localTypes; MemberAccess -> recurse on the
// parent, then look up the field type via the parent's StructInfo;
// Index -> recurse on base, then unwrap one Array dimension; Deref ->
// recurse on the pointee. Anything else (slice projections, calls,
// arithmetic, literals) returns kNoType.
//
// Used by astgenIndex's fast path to gate `astgenLvalue` — calling
// astgenLvalue speculatively on shapes like `slice.ptr` throws.
// Peeking the type structurally lets us only commit to the lvalue
// path when we know it will succeed.
static TypeIdx peekAddressableArrayLeafType(AstGenCtx &gctx, NodeIdx node) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &n = ns.get(node);
	TypeIdx ty = kNoType;
	switch (n.tag) {
	case AstTag::Variable: {
		const std::string &name =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
		auto it = gctx.localTypes.find(name);
		if (it == gctx.localTypes.end()) return kNoType;
		ty = it->second;
		break;
	}
	case AstTag::MemberAccess: {
		// Resolve the parent's type, then look up the named field.
		// Recurse with a helper that handles the broader non-Array
		// leaves too — we need parent's actual type, not just "is
		// addressable-array".
		NodeIdx parentIdx = static_cast<NodeIdx>(n.lhs);
		const AstNode &parent = ns.get(parentIdx);
		TypeIdx parentTy = kNoType;
		if (parent.tag == AstTag::Variable) {
			const std::string &pname = gctx.ctx.getStringPool().get(
			    static_cast<StringIdx>(parent.lhs));
			auto it = gctx.localTypes.find(pname);
			if (it == gctx.localTypes.end()) return kNoType;
			parentTy = it->second;
		} else {
			// Recurse: parent is itself a MemberAccess / Index /
			// Deref chain. Peek its leaf type; if the leaf is an
			// addressable expression, we'd still need to know its
			// type (not just Array-ness). For simplicity, only
			// support Variable parents for now — that covers the
			// `self.field[i]` case which is what timer.jam needs.
			return kNoType;
		}
		const auto *sinfo = gctx.ctx.lookupStruct(parentTy);
		if (sinfo == nullptr) return kNoType;
		const std::string &fieldName =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.rhs));
		for (const auto &f : sinfo->fields) {
			if (f.first == fieldName) {
				ty = f.second;
				break;
			}
		}
		if (ty == kNoType) return kNoType;
		break;
	}
	default:
		return kNoType;
	}
	const TypeKey &k = gctx.ctx.getTypePool().get(ty);
	if (k.kind == TypeKind::Array) return ty;
	return kNoType;
}

// Array Variables (and other lvalueable Array bases) take a
// dedicated fast path: peek the type via astgenLvalue, then emit
// IndexAddr + Load against the storage pointer. This avoids the
// rvalue-Load-then-spill pattern that JirTag::Index uses for SSA
// aggregates — which for arrays loads the entire backing storage
// (up to several KB) into an SSA value per access and stalls LLVM
// at -O1+ when many such accesses share a function. The fast path
// takes the base by pointer and lowers `arr[i]` as a single
// IndexAddr + per-element Load.
static JirRef astgenIndex(AstGenCtx &gctx, const AstNode &n) {
	NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx idxIdx = static_cast<NodeIdx>(n.rhs);

	// Array addressable-base fast path. For Array bases reached via
	// an addressable expression chain, take the base as a pointer via
	// ResultLoc::Pointer and emit IndexAddr + Load for just the
	// element. Avoids the rvalue-Load-then-spill pattern
	// JirTag::Index uses on Array SSA values, which loads the full
	// backing storage (up to several KB) per access and stalls LLVM
	// at -O1+.
	//
	// Discovered while investigating the Timer struct perf
	// regression: `self.channels[idx].counter` was lowering to
	// `load %Timer (208B); extractvalue channels (192B); alloca; store;
	// gep; load` per *field access* — every read of a field on an
	// indexed array element of a struct field cost hundreds of bytes
	// of register traffic. The previous Variable-only fast path
	// missed this case.
	//
	// We must STATICALLY prove the base resolves to an Array before
	// calling astgenLvalue — calling it speculatively on shapes like
	// `slice.ptr` throws ("lvalue field access on non-struct").
	// `peekAddressableArrayElem` walks the AST without emitting JIR
	// and returns the array's element type when the chain is a known
	// Variable / MemberAccess-on-struct / Index / Deref leading to
	// Array, or kNoType otherwise.
	{
		TypeIdx peekedArrayTy = peekAddressableArrayLeafType(gctx, baseIdx);
		if (peekedArrayTy != kNoType) {
			TypeIdx leafTy = kNoType;
			JirRef basePtr = astgenLvalue(gctx, baseIdx, leafTy);
			const TypeKey &lk = gctx.ctx.getTypePool().get(leafTy);
			// Sanity: the peek already promised Array, but check
			// in case the lvalue path saw something different (e.g.
			// type-alias edges peek didn't unfold).
			if (lk.kind == TypeKind::Array) {
				TypeIdx elemTy = static_cast<TypeIdx>(lk.a);
				JirRef idxRef = astgenExpr(gctx, idxIdx, BuiltinType::U64);
				TypeIdx elemPtrTy = gctx.ctx.getTypePool().intern(
				    TypeKey{TypeKind::PtrSingle, 0, 0, elemTy, 0});
				JirInst ia{};
				ia.tag = JirTag::IndexAddr;
				ia.a = basePtr;
				ia.b = idxRef;
				ia.ty = elemPtrTy;
				JirRef elemPtr = emit(gctx, ia);
				JirInst ld{};
				ld.tag = JirTag::Load;
				ld.a = elemPtr;
				ld.ty = elemTy;
				return emit(gctx, ld);
			}
		}
	}

	JirRef baseRef = astgenExpr(gctx, baseIdx, kNoType);
	JirRef idxRef = astgenExpr(gctx, idxIdx, BuiltinType::U64);
	TypeIdx baseTy = gctx.jfn.getInst(baseRef).ty;
	const TypeKey &k = gctx.ctx.getTypePool().get(baseTy);

	// Struct dispatch: `v[i]` -> `v.at(i)`. The `at` method is value-
	// shaped — it returns T directly, no pointer wrapper. The call's
	// result IS the index expression's value; nothing more to do.
	// `lookupStruct` chases Struct / Named / GenericCall TypeKinds
	// (an un-aliased `Vec(u8)` use-site annotation lands as the
	// latter) and returns null for non-struct bases, so we fall
	// through cleanly to arrays / slices / ptr-many below.
	{
		const auto *sinfo = gctx.ctx.lookupStruct(baseTy);
		if (sinfo != nullptr) {
			const std::string qualified = sinfo->name + ".at";
			const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
			if (method != nullptr && method->isCfn && !method->Args.empty()) {
				// Receiver-prep mirrors the indirect-call path: for
				// mut/move self, hand the method an addressable
				// `*Self`. Re-lower addressable bases via
				// astgenLvalue; spill non-addressable rvalues to a
				// fresh alloca and use that. (`at` is typically
				// `self: Self`, but Vec.at could legitimately take
				// `mut self` if the type wants to lazily mutate on
				// read — we support both.)
				ParamMode mode = method->Args[0].Mode;
				JirRef recv = baseRef;
				if (mode == ParamMode::Mut || mode == ParamMode::Move) {
					const AstNode &baseNode =
					    gctx.ctx.getNodeStore().get(baseIdx);
					TypeIdx leafTy = kNoType;
					switch (baseNode.tag) {
					case AstTag::Variable:
					case AstTag::MemberAccess:
					case AstTag::Index:
					case AstTag::Deref:
						recv = astgenLvalue(gctx, baseIdx, leafTy);
						break;
					default: {
						JirInst alloca{};
						alloca.tag = JirTag::Alloca;
						alloca.ty = baseTy;
						JirRef slot = emitAllocaHoisted(gctx, alloca);
						JirInst store{};
						store.tag = JirTag::Store;
						store.a = slot;
						store.b = baseRef;
						emit(gctx, store);
						recv = slot;
						break;
					}
					}
				}
				JirRef atResult =
				    emitStructCfnDispatch(gctx, sinfo, "at", recv, idxRef, {});
				if (atResult != kNoJirRef) { return atResult; }
				// `at` may be a WITHDRAWN conditional method for this
				// instantiation (e.g. Vec(T) where T isn't cloneable):
				// replay the reason instead of the generic index error.
				if (gctx.ctx.getWithdrawnMethod(sinfo->name + ".at") !=
				    nullptr) {
					return reportMethodMiss(gctx, sinfo->name + ".at");
				}
			}
		}
	}

	TypeIdx elemTy = kNoType;
	if (k.kind == TypeKind::Array || k.kind == TypeKind::Slice ||
	    k.kind == TypeKind::PtrMany) {
		elemTy = static_cast<TypeIdx>(k.a);
	} else {
		failHere(gctx, "astgen: cannot index value of this type");
	}
	JirInst inst{};
	inst.tag = JirTag::Index;
	inst.a = baseRef;
	inst.b = idxRef;
	inst.ty = elemTy;
	return emit(gctx, inst);
}

// AstGen for `AddressOf` (`&x`). Delegates to `astgenLvalue` for any
// supported lvalue form (Variable, MemberAccess, Index, Deref), then
// wraps the resulting pointer in an AddrOf so the JIR Type is the
// expected pointer-to-leaf-type. Non-lvalue operands rvalue-spill
// to a fresh alloca so `&someExpr` lowers cleanly.
// AstGen for `Slice` -- `base[start..end]`. Builds a `[]T` value
// { ptr = &base[start], len = end - start }. v1 supports many-item
// pointer bases (the std.process / argv case); array and slice bases
// can layer on later. Uses MakeSlice to assemble the {ptr,len} aggregate.
static JirRef astgenSlice(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx ex = static_cast<ExtraIdx>(n.rhs);
	NodeIdx startIdx = static_cast<NodeIdx>(ns.getExtra(ex));
	NodeIdx endIdx = static_cast<NodeIdx>(ns.getExtra(ex + 1));

	JirRef baseRef = astgenExpr(gctx, baseIdx, kNoType);
	TypeIdx baseTy = gctx.jfn.getInst(baseRef).ty;
	const TypeKey &bk = gctx.ctx.getTypePool().get(baseTy);
	if (bk.kind != TypeKind::PtrMany && bk.kind != TypeKind::PtrSingle) {
		return recoverHere(
		    gctx,
		    "slice expression `base[a..b]` needs a many-item pointer base",
		    kNoType);
	}
	TypeIdx elemTy = static_cast<TypeIdx>(bk.a);

	JirRef startRef = astgenExpr(gctx, startIdx, BuiltinType::U64);
	JirRef endRef = astgenExpr(gctx, endIdx, BuiltinType::U64);

	// Slice bounds drive both the IndexAddr offset and the
	// (end - start) length, which need matching u64 operands.
	// Zero-extend narrower (unsigned) bounds; a u64 bound is as-is.
	if (gctx.jfn.getInst(startRef).ty != BuiltinType::U64) {
		JirInst zs{};
		zs.tag = JirTag::ZExt;
		zs.a = startRef;
		zs.ty = BuiltinType::U64;
		startRef = emit(gctx, zs);
	}
	if (gctx.jfn.getInst(endRef).ty != BuiltinType::U64) {
		JirInst ze{};
		ze.tag = JirTag::ZExt;
		ze.a = endRef;
		ze.ty = BuiltinType::U64;
		endRef = emit(gctx, ze);
	}

	// ptr = &base[start]
	JirInst ia{};
	ia.tag = JirTag::IndexAddr;
	ia.a = baseRef;
	ia.b = startRef;
	ia.ty = gctx.ctx.getTypePool().internPtrMany(elemTy);
	JirRef ptrRef = emit(gctx, ia);

	// len = end - start
	JirInst sub{};
	sub.tag = JirTag::Sub;
	sub.a = endRef;
	sub.b = startRef;
	sub.ty = BuiltinType::U64;
	JirRef lenRef = emit(gctx, sub);

	// { ptr, len }
	JirInst ms{};
	ms.tag = JirTag::MakeSlice;
	ms.a = ptrRef;
	ms.b = lenRef;
	ms.ty = gctx.ctx.getTypePool().internSlice(elemTy);
	return emit(gctx, ms);
}

static JirRef astgenAddressOf(AstGenCtx &gctx, const AstNode &n) {
	NodeIdx opIdx = static_cast<NodeIdx>(n.lhs);
	const AstNode &op = gctx.ctx.getNodeStore().get(opIdx);
	TypeIdx leafTy = kNoType;
	JirRef ptrRef;
	switch (op.tag) {
	case AstTag::Variable:
	case AstTag::MemberAccess:
	case AstTag::Index:
	case AstTag::Deref:
		ptrRef = astgenLvalue(gctx, opIdx, leafTy);
		break;
	default: {
		// Rvalue spill: evaluate `op`, alloca a slot, store, return slot.
		JirRef val = astgenExpr(gctx, opIdx, kNoType);
		leafTy = gctx.jfn.getInst(val).ty;
		JirInst slotAlloca{};
		slotAlloca.tag = JirTag::Alloca;
		slotAlloca.ty = leafTy;
		ptrRef = emitAllocaHoisted(gctx, slotAlloca);
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = ptrRef;
		store.b = val;
		emit(gctx, store);
		break;
	}
	}
	TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrSingle, 0, 0, leafTy, 0});
	JirInst inst{};
	inst.tag = JirTag::AddrOf;
	inst.a = ptrRef;
	inst.ty = ptrTy;
	return emit(gctx, inst);
}

// AstGen for `Deref` (`p.*`). The operand evaluates to a pointer; we
// emit a Load against it via JirTag::Deref.
static JirRef astgenDeref(AstGenCtx &gctx, const AstNode &n) {
	NodeIdx opIdx = static_cast<NodeIdx>(n.lhs);
	JirRef ptrRef = astgenExpr(gctx, opIdx, kNoType);
	TypeIdx ptrTy = gctx.jfn.getInst(ptrRef).ty;
	const TypeKey &k = gctx.ctx.getTypePool().get(ptrTy);
	if (k.kind != TypeKind::PtrSingle && k.kind != TypeKind::PtrMany) {
		failHere(gctx, "astgen: cannot dereference non-pointer");
	}
	TypeIdx pointee = static_cast<TypeIdx>(k.a);
	JirInst inst{};
	inst.tag = JirTag::Deref;
	inst.a = ptrRef;
	inst.ty = pointee;
	return emit(gctx, inst);
}

// AstGen for `AsCast` (`expr as T`). Picks the right JIR conversion
// tag from the (source, dest) type kinds and widths. The operand is
// lowered with `T` as the expected hint when both are integer types
// so a NumberLit literal can settle at the target's width instead of
// getting truncated/extended after the fact.
static JirRef astgenAsCast(AstGenCtx &gctx, const AstNode &n) {
	NodeIdx operandIdx = static_cast<NodeIdx>(n.lhs);
	TypeIdx dstTyOrig = static_cast<TypeIdx>(n.rhs);
	// Resolve GenericCall destinations (e.g. `Identity(u8)` -> u8) so
	// the downstream Int/Float branches see the concrete type.
	TypeIdx dstTy = dstTyOrig;
	{
		const TypeKey &k0 = gctx.ctx.getTypePool().get(dstTy);
		if (k0.kind == TypeKind::GenericCall) {
			TypeIdx resolved = gctx.ctx.resolveGenericCall(dstTy);
			if (resolved != kNoType) dstTy = resolved;
		}
	}
	const TypeKey &dst = gctx.ctx.getTypePool().get(dstTy);
	// Pass the destination as a peer-type hint for Int *and* Float so a
	// numeric literal operand settles directly at the target width. For
	// floats this is essential: `<lit> as f32` must round the literal to f32
	// in one step, not lower it at f64 and then FPTrunc (which double-rounds).
	TypeIdx hint = (dst.kind == TypeKind::Int || dst.kind == TypeKind::Float)
	                   ? dstTy
	                   : kNoType;
	JirRef val = astgenExpr(gctx, operandIdx, hint);
	TypeIdx srcTy = gctx.jfn.getInst(val).ty;
	if (srcTy == dstTy) return val;
	// Same resolution on the source side.
	{
		const TypeKey &k0 = gctx.ctx.getTypePool().get(srcTy);
		if (k0.kind == TypeKind::GenericCall) {
			TypeIdx resolved = gctx.ctx.resolveGenericCall(srcTy);
			if (resolved != kNoType) srcTy = resolved;
		}
	}
	if (srcTy == dstTy) return val;
	const TypeKey &src = gctx.ctx.getTypePool().get(srcTy);

	auto emitCast = [&](JirTag tag) -> JirRef {
		JirInst inst{};
		inst.tag = tag;
		inst.a = val;
		inst.ty = dstTy;
		return emit(gctx, inst);
	};

	// Bool-to-integer: i1 -> integer of any width.
	if (src.kind == TypeKind::Bool && dst.kind == TypeKind::Int) {
		JirInst inst{};
		inst.tag = JirTag::ZExt;
		inst.a = val;
		inst.ty = dstTy;
		return emit(gctx, inst);
	}
	// Integer-to-enum: store the tag into a fresh enum value. For
	// unit-only enums the runtime form is already i8, so the cast is
	// a width adjustment + retag at the JIR level. For payloaded
	// enums it builds a {tag, undef-payload} aggregate.
	if (src.kind == TypeKind::Int &&
	    (dst.kind == TypeKind::Named || dst.kind == TypeKind::Enum)) {
		if (const auto *einfo = gctx.ctx.lookupEnum(dstTy)) {
			JirRef tagRef = val;
			// Narrow source to u8 first.
			if (src.a != 8) {
				JirInst trunc{};
				trunc.tag = JirTag::Trunc;
				trunc.a = val;
				trunc.ty = BuiltinType::U8;
				tagRef = emit(gctx, trunc);
			} else if (src.b != 0) {
				JirInst bc{};
				bc.tag = JirTag::BitCast;
				bc.a = val;
				bc.ty = BuiltinType::U8;
				tagRef = emit(gctx, bc);
			}
			if (!einfo->hasPayloadVariant) return tagRef;
			// Payloaded enum: alloca, store tag at field 0, load.
			JirInst alloca{};
			alloca.tag = JirTag::Alloca;
			alloca.ty = dstTy;
			JirRef slot = emitAllocaHoisted(gctx, alloca);
			TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});
			JirInst fa{};
			fa.tag = JirTag::FieldAddr;
			fa.a = slot;
			fa.b = 0;
			fa.ty = u8PtrTy;
			JirRef tagPtr = emit(gctx, fa);
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = tagPtr;
			store.b = tagRef;
			emit(gctx, store);
			JirInst load{};
			load.tag = JirTag::Load;
			load.a = slot;
			load.ty = dstTy;
			return emit(gctx, load);
		}
	}
	// Enum-to-integer: extract the discriminant byte. For unit-only
	// enums the runtime form is already i8, so the cast is just a
	// width adjustment. For payloaded enums it's an ExtractValue(0)
	// followed by the same width adjustment.
	if ((src.kind == TypeKind::Named || src.kind == TypeKind::Enum) &&
	    dst.kind == TypeKind::Int) {
		if (const auto *einfo = gctx.ctx.lookupEnum(srcTy)) {
			JirRef tagRef = val;
			if (einfo->hasPayloadVariant) {
				JirInst ev{};
				ev.tag = JirTag::ExtractValue;
				ev.a = val;
				ev.b = 0;
				ev.ty = BuiltinType::U8;
				tagRef = emit(gctx, ev);
			}
			// Cast the u8 tag to the requested destination width.
			TypeIdx tagTy = BuiltinType::U8;
			if (tagTy == dstTy) return tagRef;
			const TypeKey &tk = gctx.ctx.getTypePool().get(tagTy);
			uint32_t sw = tk.a;
			uint32_t dw = dst.a;
			JirInst cast{};
			if (sw < dw) cast.tag = JirTag::ZExt;
			else if (sw > dw) cast.tag = JirTag::Trunc;
			else cast.tag = JirTag::BitCast;
			cast.a = tagRef;
			cast.ty = dstTy;
			return emit(gctx, cast);
		}
	}
	// Pointer ↔ pointer cast: in opaque-pointer LLVM the runtime
	// representation is identical, so just retag at the JIR level.
	auto isPtr = [](const TypeKey &k) {
		return k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany;
	};
	if (isPtr(src) && isPtr(dst)) {
		JirInst inst{};
		inst.tag = JirTag::BitCast;
		inst.a = val;
		inst.ty = dstTy;
		return emit(gctx, inst);
	}
	// Pointer ↔ integer cast — only u64 is wide enough to round-trip
	// a pointer on every supported target, so restrict to that width
	// for the ptr→int direction. `myPtr as u64` stays the canonical form.
	if (isPtr(src) && dst.kind == TypeKind::Int && dst.a == 64) {
		return emitCast(JirTag::PtrToInt);
	}
	// Int → thin pointer (PtrSingle / PtrMany). Accepts any integer
	// width — matches Rust's *addr-ptr-cast* (compiler/rustc_hir_typeck/
	// src/cast.rs:1101 check_addr_ptr_cast), which permits the cast as
	// long as the target pointer is sized/thin. Narrower-than-pointer
	// sources are zero/sign-extended to u64 first so LLVM's IntToPtr
	// gets a pointer-width operand. The slice type ([]T, which carries
	// a length field) is intentionally excluded by `isPtr` — round-
	// tripping that needs an explicit (ptr, len) pair.
	if (src.kind == TypeKind::Int && isPtr(dst)) {
		JirRef widened = val;
		if (src.a < 64) {
			JirInst ext{};
			ext.tag = (src.b != 0) ? JirTag::SExt : JirTag::ZExt;
			ext.a = val;
			ext.ty = gctx.ctx.getTypePool().internInt(64, false);
			widened = emit(gctx, ext);
		} else if (src.a > 64) {
			JirInst tr{};
			tr.tag = JirTag::Trunc;
			tr.a = val;
			tr.ty = gctx.ctx.getTypePool().internInt(64, false);
			widened = emit(gctx, tr);
		}
		JirInst i2p{};
		i2p.tag = JirTag::IntToPtr;
		i2p.a = widened;
		i2p.ty = dstTy;
		return emit(gctx, i2p);
	}
	if (src.kind == TypeKind::Int && dst.kind == TypeKind::Int) {
		uint32_t sw = src.a;
		uint32_t dw = dst.a;
		bool srcSigned = src.b != 0;
		if (sw < dw) return emitCast(srcSigned ? JirTag::SExt : JirTag::ZExt);
		if (sw > dw) return emitCast(JirTag::Trunc);
		// Same width, different signedness — no-op at the bit level,
		// just retag the JIR type.
		return emitCast(JirTag::BitCast);
	}
	if (src.kind == TypeKind::Int && dst.kind == TypeKind::Float) {
		return emitCast(src.b != 0 ? JirTag::SIToFP : JirTag::UIToFP);
	}
	if (src.kind == TypeKind::Float && dst.kind == TypeKind::Int) {
		// Signedness of the destination picks FPToSI vs FPToUI.
		return emitCast(dst.b != 0 ? JirTag::FPToSI : JirTag::FPToUI);
	}
	if (src.kind == TypeKind::Float && dst.kind == TypeKind::Float) {
		uint32_t sw = src.a;
		uint32_t dw = dst.a;
		if (sw < dw) return emitCast(JirTag::FPExt);
		if (sw > dw) return emitCast(JirTag::FPTrunc);
		return val;
	}
	failHere(gctx, "astgen: unsupported `as` cast between these types");
}

// AstGen for `UnaryOp`. Three forms:
//   `-x`  (Neg)     -> 0 - x  for ints, FNeg for floats
//   `!x`  (LogNot)  -> x == false (xor 1)
//   `~x`  (BitNot)  -> x XOR all-ones
static JirRef astgenUnaryOp(AstGenCtx &gctx, const AstNode &n,
                            TypeIdx expected) {
	NodeIdx opIdx = static_cast<NodeIdx>(n.lhs);
	JirRef operand = astgenExpr(gctx, opIdx, expected);
	UnaryOp op = static_cast<UnaryOp>(n.op);
	TypeIdx ty = gctx.jfn.getInst(operand).ty;

	switch (op) {
	case UnaryOp::Neg: {
		const TypeKey &k = gctx.ctx.getTypePool().get(ty);
		if (k.kind == TypeKind::Float) {
			JirInst inst{};
			inst.tag = JirTag::FNeg;
			inst.a = operand;
			inst.ty = ty;
			return emit(gctx, inst);
		}
		// Integer negate: emit `0 - operand` so we reuse the existing
		// Sub handler and signed/unsigned semantics fall out naturally.
		JirInst zero{};
		zero.tag = JirTag::Int;
		zero.a = 0;
		zero.b = 0;
		zero.ty = ty;
		JirRef zeroRef = emit(gctx, zero);
		JirInst sub{};
		sub.tag = JirTag::Sub;
		sub.a = zeroRef;
		sub.b = operand;
		sub.ty = ty;
		return emit(gctx, sub);
	}
	case UnaryOp::LogNot: {
		JirInst inst{};
		inst.tag = JirTag::LogNot;
		inst.a = operand;
		inst.ty = BuiltinType::Bool;
		return emit(gctx, inst);
	}
	case UnaryOp::BitNot: {
		JirInst inst{};
		inst.tag = JirTag::BitNot;
		inst.a = operand;
		inst.ty = ty;
		return emit(gctx, inst);
	}
	default:
		failHere(gctx, "astgen: unknown UnaryOp");
	}
}

// AstGen for `BinaryOp` — integer arithmetic + comparison.
// Each binary op chooses the right JirTag based on the operand type
// (integer vs float) and signedness (for SDiv/UDiv, SLT/ULT, etc.).
static JirRef astgenBinaryOp(AstGenCtx &gctx, const AstNode &n,
                             TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx lhsIdx = static_cast<NodeIdx>(n.lhs);
	NodeIdx rhsIdx = static_cast<NodeIdx>(n.rhs);
	JirRef lhsRef = astgenExpr(gctx, lhsIdx, expected);
	// Pass the LHS's resolved type as the expected hint for RHS so a
	// literal `48` in `c - 48` materialises at c's width instead of
	// at its natural u8. Otherwise the JIR Sub would be `i32 - u8`,
	// which LLVM's builders accept only by accident on small targets.
	TypeIdx lhsType = gctx.jfn.getInst(lhsRef).ty;
	JirRef rhsRef = astgenExpr(gctx, rhsIdx, lhsType);

	// Float widths must match exactly: `f32 + f64` is rejected. The
	// user has to write an explicit `as` cast.
	//
	// Integer widths are matched implicitly here — the narrower side
	// is sign / zero extended to the wider side's width. Without this
	// `while (i < a.len)` with `i: u32` and `a.len: u64` would reach
	// the verifier as a mixed-width comparison and abort. Legacy
	// did the same widening via `coerceTo` at codegen.
	{
		TypeIdx rhsType = gctx.jfn.getInst(rhsRef).ty;
		const TypeKey &lk = gctx.ctx.getTypePool().get(lhsType);
		const TypeKey &rk = gctx.ctx.getTypePool().get(rhsType);
		if (lk.kind == TypeKind::Float && rk.kind == TypeKind::Float &&
		    lk.a != rk.a) {
			failHere(gctx,
			         "mismatched float widths in binary op; use an explicit "
			         "`as` cast to align them");
		}
		if (lk.kind == TypeKind::Int && rk.kind == TypeKind::Int &&
		    lk.a != rk.a) {
			JirInst widen{};
			if (lk.a < rk.a) {
				widen.tag = lk.b != 0 ? JirTag::SExt : JirTag::ZExt;
				widen.a = lhsRef;
				widen.ty = rhsType;
				lhsRef = emit(gctx, widen);
				lhsType = rhsType;
			} else {
				widen.tag = rk.b != 0 ? JirTag::SExt : JirTag::ZExt;
				widen.a = rhsRef;
				widen.ty = lhsType;
				rhsRef = emit(gctx, widen);
			}
		}
	}

	// Result type defaults to the operand type. For comparisons we
	// override to bool below.
	TypeIdx resultTy = lhsType;

	BinOp op = static_cast<BinOp>(n.op);
	JirTag tag = JirTag::Invalid;
	bool isCmp = false;

	const TypeKey &k = gctx.ctx.getTypePool().get(resultTy);
	bool isFloat = k.kind == TypeKind::Float;
	bool isSigned = k.kind == TypeKind::Int && k.b != 0;

	// Short-circuit logical operators don't fit the
	// uniformly-evaluate-both-sides shape — handle separately. The
	// result is i1; only the LHS evaluation is unconditional.
	if (op == BinOp::LogAnd || op == BinOp::LogOr) {
		// LogAnd: result = lhs ? rhs : false  (lower as if-expression)
		// LogOr:  result = lhs ? true : rhs
		// Use an alloca for the result slot; codegen folds to phi at -O2.
		bool isAnd = op == BinOp::LogAnd;
		// Re-emit operands as bools: lhsRef is already i1 (we cast expected to
		// bool).
		(void)lhsRef;
		(void)rhsRef;
		(void)resultTy;
		// Discard the eagerly-emitted lhs/rhs above — we want to control
		// emission order for short-circuit. Reset emission by re-evaluating.
		// Simpler implementation path: emit the result slot, then a
		// CondBr based on freshly-evaluated lhs, then conditionally
		// emit rhs. The eager lhs/rhs above are already in the IR,
		// but as long as they have no side-effects beyond their JIR
		// types we can keep them — operator precedence guarantees no
		// dead writes.
		JirInst resAlloca{};
		resAlloca.tag = JirTag::Alloca;
		resAlloca.ty = BuiltinType::Bool;
		JirRef resSlot = emitAllocaHoisted(gctx, resAlloca);
		JirInst lhsStore{};
		lhsStore.tag = JirTag::Store;
		lhsStore.a = resSlot;
		lhsStore.b = lhsRef;
		emit(gctx, lhsStore);

		JirBlockRef rhsB = gctx.jfn.pushBlock(isAnd ? "and.rhs" : "or.rhs");
		JirBlockRef endB = gctx.jfn.pushBlock(isAnd ? "and.end" : "or.end");
		if (isAnd) {
			emitCondBr(gctx, lhsRef, rhsB, endB);
		} else {
			emitCondBr(gctx, lhsRef, endB, rhsB);
		}
		gctx.currentBlock = rhsB;
		JirInst rhsStore{};
		rhsStore.tag = JirTag::Store;
		rhsStore.a = resSlot;
		rhsStore.b = rhsRef;
		emit(gctx, rhsStore);
		emitBr(gctx, endB);

		gctx.currentBlock = endB;
		JirInst loadRes{};
		loadRes.tag = JirTag::Load;
		loadRes.a = resSlot;
		loadRes.ty = BuiltinType::Bool;
		return emit(gctx, loadRes);
	}

	switch (op) {
	case BinOp::Add:
		tag = isFloat ? JirTag::FAdd : JirTag::Add;
		break;
	case BinOp::Sub:
		tag = isFloat ? JirTag::FSub : JirTag::Sub;
		break;
	case BinOp::Mul:
		tag = isFloat ? JirTag::FMul : JirTag::Mul;
		break;
	case BinOp::Div:
		tag = isFloat ? JirTag::FDiv : (isSigned ? JirTag::SDiv : JirTag::UDiv);
		break;
	case BinOp::Mod:
		tag = isFloat ? JirTag::FRem : (isSigned ? JirTag::SRem : JirTag::URem);
		break;
	case BinOp::BitAnd:
		tag = JirTag::BitAnd;
		break;
	case BinOp::BitOr:
		tag = JirTag::BitOr;
		break;
	case BinOp::BitXor:
		tag = JirTag::BitXor;
		break;
	case BinOp::Shl:
		tag = JirTag::Shl;
		break;
	case BinOp::Shr:
		tag = isSigned ? JirTag::AShr : JirTag::LShr;
		break;
	case BinOp::Eq:
		tag = isFloat ? JirTag::FCmpOeq : JirTag::ICmpEq;
		isCmp = true;
		break;
	case BinOp::Ne:
		tag = isFloat ? JirTag::FCmpOne : JirTag::ICmpNe;
		isCmp = true;
		break;
	case BinOp::Lt:
		tag = isFloat ? JirTag::FCmpOlt
		              : (isSigned ? JirTag::ICmpSlt : JirTag::ICmpUlt);
		isCmp = true;
		break;
	case BinOp::Le:
		tag = isFloat ? JirTag::FCmpOle
		              : (isSigned ? JirTag::ICmpSle : JirTag::ICmpUle);
		isCmp = true;
		break;
	case BinOp::Gt:
		tag = isFloat ? JirTag::FCmpOgt
		              : (isSigned ? JirTag::ICmpSgt : JirTag::ICmpUgt);
		isCmp = true;
		break;
	case BinOp::Ge:
		tag = isFloat ? JirTag::FCmpOge
		              : (isSigned ? JirTag::ICmpSge : JirTag::ICmpUge);
		isCmp = true;
		break;
	default:
		failHere(gctx, "unsupported binary operator (internal op = " +
		                   std::to_string(static_cast<int>(op)) +
		                   ") — please file a bug");
	}

	JirInst inst{};
	inst.tag = tag;
	inst.a = lhsRef;
	inst.b = rhsRef;
	inst.ty = isCmp ? BuiltinType::Bool : resultTy;
	return emit(gctx, inst);
}

// Branch unconditionally from the current block to `target` and move
// the insertion point to `next`. Used at the bottom of structured
// scopes (then-arm, else-arm, loop-body) where the structured shape
// guarantees what the next block should be.
static void emitBr(AstGenCtx &gctx, JirBlockRef target) {
	JirInst br{};
	br.tag = JirTag::Br;
	br.a = target;
	emit(gctx, br);
}

static void emitCondBr(AstGenCtx &gctx, JirRef cond, JirBlockRef thenB,
                       JirBlockRef elseB) {
	// Fold a CondBr whose cond is a known constant `Bool` literal to a
	// plain Br to the live branch. This keeps the dead branch from
	// looking reachable (its block ends up with zero predecessors),
	// which matters for the function-end fall-through check on
	// `while (true)` / `loop { ... }` patterns: the false branch of
	// the loop-condition compare is dead, so its successor is also
	// dead, and the function tail correctly resolves as unreachable.
	const JirInst &c = gctx.jfn.getInst(cond);
	if (c.tag == JirTag::Bool) {
		emitBr(gctx, c.a != 0 ? thenB : elseB);
		return;
	}
	std::vector<uint32_t> packed = {thenB, elseB};
	JirExtraIdx extra = gctx.jfn.pushExtra(packed.data(), packed.size());
	JirInst cbr{};
	cbr.tag = JirTag::CondBr;
	cbr.a = cond;
	cbr.b = extra;
	emit(gctx, cbr);
}

// AstGen for `IfNode` (statement form). Builds then/else/merge blocks;
// `else` is optional. The merge block is reused as the next insertion
// point only if at least one branch fell through (i.e. neither branch
// ended with a terminator). Otherwise the merge block is silently
// dropped.
static void astgenIf(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t thenCount = ns.getExtra(extra);
	uint32_t elseCount = ns.getExtra(extra + 1);

	JirRef condRef = astgenExpr(gctx, condIdx, BuiltinType::Bool);
	JirBlockRef thenB = gctx.jfn.pushBlock("then");
	JirBlockRef elseB =
	    (elseCount > 0) ? gctx.jfn.pushBlock("else") : kNoJirBlock;
	JirBlockRef mergeB = gctx.jfn.pushBlock("ifend");

	emitCondBr(gctx, condRef, thenB, (elseCount > 0) ? elseB : mergeB);

	// Then arm — own scope so locals declared inside the body drop
	// at branch exit.
	gctx.currentBlock = thenB;
	pushDropScope(gctx);
	for (uint32_t i = 0; i < thenCount; i++) {
		NodeIdx s = static_cast<NodeIdx>(ns.getExtra(extra + 2 + i));
		astgenExpr(gctx, s, kNoType);
	}
	popDropScopeEmitting(gctx);
	if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock), gctx.jfn)) {
		emitBr(gctx, mergeB);
	}

	// Else arm (if present).
	if (elseCount > 0) {
		gctx.currentBlock = elseB;
		pushDropScope(gctx);
		for (uint32_t i = 0; i < elseCount; i++) {
			NodeIdx s =
			    static_cast<NodeIdx>(ns.getExtra(extra + 2 + thenCount + i));
			astgenExpr(gctx, s, kNoType);
		}
		popDropScopeEmitting(gctx);
		if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock),
		                        gctx.jfn)) {
			emitBr(gctx, mergeB);
		}
	}

	gctx.currentBlock = mergeB;
}

// AstGen for `ForNode` (`for x in start..end { body }`). Desugars
// to a while-loop with an explicit induction variable: allocate a
// slot for `x` initialised to `start`, loop while `x < end`,
// running the body and incrementing `x` at the end of each
// iteration. Mirrors how the parser would have shaped this if
// it lowered eagerly.
static void astgenFor(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx extra = static_cast<ExtraIdx>(n.lhs);
	StringIdx varNameId = static_cast<StringIdx>(ns.getExtra(extra));
	NodeIdx startIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1));
	NodeIdx endIdx = static_cast<NodeIdx>(ns.getExtra(extra + 2));
	uint32_t bodyCount = ns.getExtra(extra + 3);
	const std::string &varName = gctx.ctx.getStringPool().get(varNameId);

	// Peek a bare-Variable end bound's declared type and use it as the
	// start expression's hint: `for i in 0:n` adopts n's exact type
	// (width AND signedness) for the induction var. Complex end
	// expressions fall back to the start's own type plus the width
	// reconciliation at the comparison below.
	TypeIdx startHint = kNoType;
	{
		const AstNode &en = ns.get(endIdx);
		if (en.tag == AstTag::Variable) {
			const std::string &endName =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(en.lhs));
			auto it = gctx.localTypes.find(endName);
			if (it != gctx.localTypes.end()) { startHint = it->second; }
		}
	}
	JirRef startRef = astgenExpr(gctx, startIdx, startHint);
	TypeIdx idxTy = gctx.jfn.getInst(startRef).ty;

	// Allocate the loop-var slot and store the start value.
	JirInst alloca{};
	alloca.tag = JirTag::Alloca;
	alloca.ty = idxTy;
	JirRef slot = emitAllocaHoisted(gctx, alloca);
	JirInst store{};
	store.tag = JirTag::Store;
	store.a = slot;
	store.b = startRef;
	emit(gctx, store);
	gctx.locals[varName] = slot;
	gctx.localTypes[varName] = idxTy;

	JirBlockRef condB = gctx.jfn.pushBlock("forcond");
	JirBlockRef bodyB = gctx.jfn.pushBlock("forbody");
	JirBlockRef stepB = gctx.jfn.pushBlock("forstep");
	JirBlockRef exitB = gctx.jfn.pushBlock("forexit");
	emitBr(gctx, condB);

	gctx.currentBlock = condB;
	JirInst load{};
	load.tag = JirTag::Load;
	load.a = slot;
	load.ty = idxTy;
	JirRef loadIdx = emit(gctx, load);
	JirRef endRef = astgenExpr(gctx, endIdx, idxTy);
	// Pick signed-or-unsigned comparison from the induction var's type.
	const TypeKey &ik = gctx.ctx.getTypePool().get(idxTy);
	bool isSigned = ik.kind == TypeKind::Int && ik.b != 0;
	// `for i in 0:n` with a VARIABLE bound: the induction var's type
	// comes from the start expression while a non-literal end keeps its
	// own width (the expected-type hint only narrows literals). Widen
	// the narrower side so the comparison is well-typed — same shape as
	// astgenAssertCall's width reconciliation.
	TypeIdx endTy = gctx.jfn.getInst(endRef).ty;
	if (endTy != idxTy) {
		const TypeKey &ek = gctx.ctx.getTypePool().get(endTy);
		if (ik.kind == TypeKind::Int && ek.kind == TypeKind::Int &&
		    ik.b == ek.b) {
			JirInst ext{};
			ext.tag = isSigned ? JirTag::SExt : JirTag::ZExt;
			if (ik.a > ek.a) {
				ext.a = endRef;
				ext.ty = idxTy;
				endRef = emit(gctx, ext);
			} else {
				ext.a = loadIdx;
				ext.ty = endTy;
				loadIdx = emit(gctx, ext);
			}
		} else {
			failHere(gctx,
			         "for-range bounds have mismatched types; cast one "
			         "side with `as` so both bounds agree");
		}
	}
	JirInst cmp{};
	cmp.tag = isSigned ? JirTag::ICmpSlt : JirTag::ICmpUlt;
	cmp.a = loadIdx;
	cmp.b = endRef;
	cmp.ty = BuiltinType::Bool;
	JirRef cmpRef = emit(gctx, cmp);
	emitCondBr(gctx, cmpRef, bodyB, exitB);

	// Loop body — break exits to exitB, continue jumps to stepB. The
	// body lives in its own drop scope so per-iteration locals are
	// destroyed before each step / break.
	gctx.currentBlock = bodyB;
	pushDropScope(gctx);
	gctx.loopStack.push_back({stepB, exitB, gctx.dropScopes.size() - 1});
	for (uint32_t i = 0; i < bodyCount; i++) {
		NodeIdx s = static_cast<NodeIdx>(ns.getExtra(extra + 4 + i));
		astgenExpr(gctx, s, kNoType);
	}
	popDropScopeEmitting(gctx);
	if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock), gctx.jfn)) {
		emitBr(gctx, stepB);
	}
	gctx.loopStack.pop_back();

	// Step block: x = x + 1; br cond.
	gctx.currentBlock = stepB;
	JirInst lstep{};
	lstep.tag = JirTag::Load;
	lstep.a = slot;
	lstep.ty = idxTy;
	JirRef cur = emit(gctx, lstep);
	JirInst one{};
	one.tag = JirTag::Int;
	one.a = 1;
	one.ty = idxTy;
	JirRef oneRef = emit(gctx, one);
	JirInst inc{};
	inc.tag = JirTag::Add;
	inc.a = cur;
	inc.b = oneRef;
	inc.ty = idxTy;
	JirRef next = emit(gctx, inc);
	JirInst stepStore{};
	stepStore.tag = JirTag::Store;
	stepStore.a = slot;
	stepStore.b = next;
	emit(gctx, stepStore);
	emitBr(gctx, condB);

	gctx.currentBlock = exitB;
	gctx.locals.erase(varName);
	gctx.localTypes.erase(varName);
}

// AstGen for `WhileNode` (also handles `loop`, which the parser
// desugars to `while (true) { ... }`). Cond/body/exit blocks; body
// loops back to cond, break exits, continue re-tests.
static void astgenWhile(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx condIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	uint32_t bodyCount = ns.getExtra(extra);

	JirBlockRef condB = gctx.jfn.pushBlock("loopcond");
	JirBlockRef bodyB = gctx.jfn.pushBlock("loopbody");
	JirBlockRef exitB = gctx.jfn.pushBlock("loopexit");

	emitBr(gctx, condB);
	gctx.currentBlock = condB;
	JirRef condRef = astgenExpr(gctx, condIdx, BuiltinType::Bool);
	emitCondBr(gctx, condRef, bodyB, exitB);

	gctx.currentBlock = bodyB;
	pushDropScope(gctx);
	gctx.loopStack.push_back({condB, exitB, gctx.dropScopes.size() - 1});
	for (uint32_t i = 0; i < bodyCount; i++) {
		NodeIdx s = static_cast<NodeIdx>(ns.getExtra(extra + 1 + i));
		astgenExpr(gctx, s, kNoType);
	}
	popDropScopeEmitting(gctx);
	if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock), gctx.jfn)) {
		emitBr(gctx, condB);
	}
	gctx.loopStack.pop_back();

	gctx.currentBlock = exitB;
}

static void astgenBreak(AstGenCtx &gctx) {
	if (gctx.loopStack.empty()) {
		failHere(gctx, "astgen: `break` outside of loop");
	}
	emitDropsThroughScope(gctx, gctx.loopStack.back().bodyScopeIdx);
	emitBr(gctx, gctx.loopStack.back().breakBlock);
}

static void astgenContinue(AstGenCtx &gctx) {
	if (gctx.loopStack.empty()) {
		failHere(gctx, "astgen: `continue` outside of loop");
	}
	emitDropsThroughScope(gctx, gctx.loopStack.back().bodyScopeIdx);
	emitBr(gctx, gctx.loopStack.back().continueBlock);
}

// Match lowering (Phase 5)
//
// Each match desugars to:
//   - eval scrutinee
//   - alloca result slot (only when the match is in expression position
//     and arms produce a value)
//   - chain of compare+branch for each arm's patterns, defaulting to
//     either an explicit wildcard arm or unreachable
//   - in each arm: eval body, store tail to result slot, br merge
//   - at merge: load the result slot (or noop in statement form)
//
// Peer-type resolution scans every arm's tail expression to pick a
// unified Jam type so literals lower at the unified width with the
// right signedness from the start. Mirrors what the legacy
// `codegenMatch` did but moved up to astgen so jir_codegen stays
// mechanical.

static TypeIdx peerResolveType(TypeIdx a, TypeIdx b, const TypePool &tp) {
	if (a == kNoType) return b;
	if (b == kNoType) return a;
	if (a == b) return a;
	const TypeKey &ka = tp.get(a);
	const TypeKey &kb = tp.get(b);
	if (ka.kind == TypeKind::Int && kb.kind == TypeKind::Int) {
		uint32_t aw = ka.a;
		uint32_t bw = kb.a;
		bool aSigned = ka.b != 0;
		bool bSigned = kb.b != 0;
		uint32_t wider = aw > bw ? aw : bw;
		bool sign = aSigned || bSigned;
		return const_cast<TypePool &>(tp).intern(
		    TypeKey{TypeKind::Int, 0, 0, wider, sign ? 1u : 0u});
	}
	if (ka.kind == TypeKind::Float && kb.kind == TypeKind::Float) {
		uint32_t aw = ka.a;
		uint32_t bw = kb.a;
		return const_cast<TypePool &>(tp).intern(
		    TypeKey{TypeKind::Float, 0, 0, aw > bw ? aw : bw, 0});
	}
	return kNoType;
}

// Compile-time, source-level type inference for a leaf expression.
// Lightweight: only NumberLit / BoolLit / Variable are precise; for
// everything else we return kNoType and the caller falls back to the
// caller-supplied expected type.
static TypeIdx inferTailType(const AstGenCtx &gctx, NodeIdx idx) {
	const AstNode &n = gctx.ctx.getNodeStore().get(idx);
	switch (n.tag) {
	case AstTag::NumberLit: {
		uint64_t val =
		    static_cast<uint64_t>(n.lhs) | (static_cast<uint64_t>(n.rhs) << 32);
		bool isNeg = (n.flags & 1) != 0;
		bool isFloat = (n.flags & 2) != 0;
		if (isFloat) return BuiltinType::F64;
		if (isNeg) {
			if (val <= 128) return BuiltinType::I8;
			if (val <= 32768) return BuiltinType::I16;
			if (val <= 2147483648ULL) return BuiltinType::I32;
			return BuiltinType::I64;
		}
		if (val <= 255) return BuiltinType::U8;
		if (val <= 65535) return BuiltinType::U16;
		if (val <= 4294967295ULL) return BuiltinType::U32;
		return BuiltinType::U64;
	}
	case AstTag::BoolLit:
		return BuiltinType::Bool;
	case AstTag::Variable: {
		const std::string &name =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
		auto it = gctx.localTypes.find(name);
		if (it != gctx.localTypes.end()) return it->second;
		return kNoType;
	}
	default:
		return kNoType;
	}
}

// True if `node` is a divergent statement form (return / break /
// continue, or a call to a `noreturn` function like `abort()` /
// `panic()`), so the arm contributes no tail value to the merge
// phi/load.
static bool stmtDiverges(const AstGenCtx &gctx, NodeIdx node) {
	const AstNode &n = gctx.ctx.getNodeStore().get(node);
	if (n.tag == AstTag::Return || n.tag == AstTag::Break ||
	    n.tag == AstTag::Continue) {
		return true;
	}
	if (n.tag == AstTag::Call && (n.flags & 1) == 0) {
		const std::string &callee =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
		if (const FunctionAST *fn = gctx.ctx.getFunctionAST(callee)) {
			return fn->ReturnType == BuiltinType::NoReturn;
		}
	}
	return false;
}

// One payload binding introduced by a pattern: the source-level name,
// the alloca's JirRef, and the binding's source-level type. The arm
// body installs these into gctx.locals / gctx.localTypes before
// lowering its statements and removes them after.
struct ArmBinding {
	std::string name;
	JirRef slot;
	TypeIdx type;
};
using ArmBindings = std::vector<ArmBinding>;

// A single case row of a Switch dispatch: the discriminant value
// (zero-extended to 64 bits; sign bit recorded separately so codegen
// can pick the right LLVM const helper), and the block to jump to
// when the scrutinee matches. Multiple cases can share the same
// target — that's how or-patterns of literals collapse.
struct SwitchCase {
	uint64_t value;
	bool isSigned;
	JirBlockRef target;
};

// Recursively walk a pattern and try to express it as one or more
// Switch cases targeting `armBlock`. Returns false if anything in the
// pattern can't be a Switch case (PatRange, PatEnumVariant with
// bindings, any unrecognised tag). PatWildcard is handled by the
// caller — it doesn't contribute cases, it becomes the default block.
//
// `scrutIsEnum` selects how PatLit / PatEnumVariant are validated:
//   * integer scrutinee -> only PatLit accepted; value = literal bits.
//   * enum scrutinee   -> only PatEnumVariant accepted; value = the
//                         variant's discriminant byte. The enum's
//                         `EnumInfo` is supplied so we can resolve the
//                         variant name without re-doing the lookup
//                         later.
static bool collectSwitchCases(const NodeStore &ns, JamCodegenContext &ctx,
                               NodeIdx patIdx, JirBlockRef armBlock,
                               TypeIdx scrutTy, bool scrutIsEnum,
                               const JamCodegenContext::EnumInfo *einfo,
                               std::vector<SwitchCase> &out) {
	const AstNode &p = ns.get(patIdx);
	switch (p.tag) {
	case AstTag::PatLit: {
		if (scrutIsEnum) return false;
		uint64_t val =
		    static_cast<uint64_t>(p.lhs) | (static_cast<uint64_t>(p.rhs) << 32);
		bool isNeg = (p.flags & 1) != 0;
		if (isNeg) {
			// Two's-complement encoding at the scrut's width; the
			// codegen sign-extends to LLVM int width on AddCase.
			val = static_cast<uint64_t>(-static_cast<int64_t>(val));
		}
		const TypeKey &sk = ctx.getTypePool().get(scrutTy);
		bool signedCmp = sk.kind == TypeKind::Int && sk.b != 0;
		out.push_back(SwitchCase{val, signedCmp, armBlock});
		return true;
	}
	case AstTag::PatEnumVariant: {
		bool hasBindings = (p.flags & 1) != 0;
		bool inferReceiver = (p.flags & 4) != 0;
		// Constant pattern: a bare identifier (infer-receiver, no bindings)
		// naming a module-level const. comptime-evaluate it to an integer and
		// emit a switch case — the scalar-const -> SwitchInt path rustc takes
		// via Const::try_eval_bits. Unfoldable / non-int consts return false
		// and fall back to the per-arm compare chain (astgenPatternCompare).
		if (!scrutIsEnum && inferReceiver && !hasBindings) {
			const std::string &cname =
			    ctx.getStringPool().get(static_cast<StringIdx>(p.rhs));
			if (const auto *mc = ctx.getModuleConst(cname)) {
				jam::ComptimeEvaluator ev(ns, ctx.getStringPool(),
				                          ctx.getTypePool());
				jam::ComptimeValue v =
				    ev.eval(mc->initExpr, jam::ComptimeScope{});
				if (v.isInt()) {
					const TypeKey &sk = ctx.getTypePool().get(scrutTy);
					bool signedCmp = sk.kind == TypeKind::Int && sk.b != 0;
					out.push_back(
					    SwitchCase{v.intVal.bits, signedCmp, armBlock});
					return true;
				}
			}
			return false;
		}
		if (!scrutIsEnum || einfo == nullptr) return false;
		// Reject pattern shapes that need a binding block before the
		// arm body — Switch can only jump straight to `armBlock`.
		if (hasBindings) return false;
		StringIdx variantNameId = static_cast<StringIdx>(p.rhs);
		const std::string &variantName = ctx.getStringPool().get(variantNameId);
		int vidx = ctx.getEnumVariantIndex(einfo->name, variantName);
		if (vidx < 0) return false;
		uint64_t disc = einfo->variants[vidx].discriminant;
		out.push_back(SwitchCase{disc, false, armBlock});
		return true;
	}
	case AstTag::PatOr: {
		ExtraIdx ex = static_cast<ExtraIdx>(p.lhs);
		uint32_t cnt = ns.getExtra(ex);
		for (uint32_t i = 0; i < cnt; i++) {
			NodeIdx sub = static_cast<NodeIdx>(ns.getExtra(ex + 1 + i));
			if (!collectSwitchCases(ns, ctx, sub, armBlock, scrutTy,
			                        scrutIsEnum, einfo, out)) {
				return false;
			}
		}
		return true;
	}
	default:
		return false;
	}
}

// Compare scrut against a single pattern; the resulting i1 is fed to
// a CondBr that jumps to `armBlock` on match, `nextBlock` otherwise.
// Handles PatLit (single equality), PatRange (lo<=x<=hi), and PatOr
// (recursive disjunction). PatWildcard is handled at the dispatch
// level — the wildcard arm becomes the fallthrough default and never
// gets its own compare.
//
// `outBindings`, if non-null, receives the (name, slot) pairs for any
// payload bindings introduced by the pattern. The bindings' slot
// allocas are emitted in `outBindings` order so arm-body code can
// install them by Load via Variable.
static void astgenPatternCompare(AstGenCtx &gctx, NodeIdx patIdx, JirRef scrut,
                                 TypeIdx scrutTy, JirBlockRef armBlock,
                                 JirBlockRef nextBlock,
                                 ArmBindings *outBindings) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &p = ns.get(patIdx);
	const TypeKey &sk = gctx.ctx.getTypePool().get(scrutTy);
	bool signedCmp = sk.kind == TypeKind::Int && sk.b != 0;

	auto emitInt = [&](uint64_t v, bool isNeg) -> JirRef {
		JirInst c{};
		c.tag = JirTag::Int;
		c.a = static_cast<uint32_t>(v & 0xFFFFFFFFu);
		c.b = static_cast<uint32_t>(v >> 32);
		if (isNeg) c.flags |= 1;
		c.ty = scrutTy;
		return emit(gctx, c);
	};
	auto emitCmp = [&](JirTag tag, JirRef lhs, JirRef rhs) -> JirRef {
		JirInst c{};
		c.tag = tag;
		c.a = lhs;
		c.b = rhs;
		c.ty = BuiltinType::Bool;
		return emit(gctx, c);
	};

	switch (p.tag) {
	case AstTag::PatLit: {
		uint64_t val =
		    static_cast<uint64_t>(p.lhs) | (static_cast<uint64_t>(p.rhs) << 32);
		bool isNeg = (p.flags & 1) != 0;
		JirRef k = emitInt(val, isNeg);
		JirRef cmp = emitCmp(JirTag::ICmpEq, scrut, k);
		emitCondBr(gctx, cmp, armBlock, nextBlock);
		return;
	}
	case AstTag::PatRange: {
		// lo32 in lhs, hi32 in rhs; both positive.
		uint64_t lo = static_cast<uint64_t>(p.lhs);
		uint64_t hi = static_cast<uint64_t>(p.rhs);
		JirRef loK = emitInt(lo, false);
		JirRef hiK = emitInt(hi, false);
		JirRef geLo =
		    emitCmp(signedCmp ? JirTag::ICmpSge : JirTag::ICmpUge, scrut, loK);
		JirBlockRef checkHi = gctx.jfn.pushBlock("range.hi");
		emitCondBr(gctx, geLo, checkHi, nextBlock);
		gctx.currentBlock = checkHi;
		JirRef leHi =
		    emitCmp(signedCmp ? JirTag::ICmpSle : JirTag::ICmpUle, scrut, hiK);
		emitCondBr(gctx, leHi, armBlock, nextBlock);
		return;
	}
	case AstTag::PatOr: {
		ExtraIdx ex = static_cast<ExtraIdx>(p.lhs);
		uint32_t cnt = ns.getExtra(ex);
		for (uint32_t i = 0; i < cnt; i++) {
			NodeIdx sub = static_cast<NodeIdx>(ns.getExtra(ex + 1 + i));
			JirBlockRef tryNext =
			    (i + 1 == cnt) ? nextBlock : gctx.jfn.pushBlock("or.next");
			astgenPatternCompare(gctx, sub, scrut, scrutTy, armBlock, tryNext,
			                     outBindings);
			if (i + 1 != cnt) gctx.currentBlock = tryNext;
		}
		return;
	}
	case AstTag::PatEnumVariant: {
		// Decode the PatEnumVariant via the four-way encoding documented
		// in the legacy `decodePatEnumVariant`:
		//   flags & 1: bindings present, lhs = ExtraIdx -> [recv, variant,
		//              count, name0, name1, ...]; else lhs is the receiver
		//   flags & 2: receiver is a TypeIdx (GenericCall, resolved via
		//              lookupEnum); else it's a StringIdx
		bool hasBindings = (p.flags & 1) != 0;
		bool typeIdxReceiver = (p.flags & 2) != 0;
		bool inferReceiver = (p.flags & 4) != 0;
		uint32_t recvSlot;
		StringIdx variantNameId;
		uint32_t bindingCount = 0;
		ExtraIdx bindingsStart = 0;
		if (hasBindings) {
			ExtraIdx ex = static_cast<ExtraIdx>(p.lhs);
			recvSlot = ns.getExtra(ex);
			variantNameId = static_cast<StringIdx>(ns.getExtra(ex + 1));
			bindingCount = ns.getExtra(ex + 2);
			bindingsStart = ex + 3;
		} else {
			recvSlot = p.lhs;
			variantNameId = static_cast<StringIdx>(p.rhs);
		}

		std::string enumName;
		const auto *einfo =
		    static_cast<const decltype(gctx.ctx.getEnum(""))>(nullptr);
		if (inferReceiver) {
			// Bare variant: resolve against the scrutinee's enum type.
			if (const auto *info = gctx.ctx.lookupEnum(scrutTy)) {
				enumName = info->name;
				einfo = info;
			}
		} else if (typeIdxReceiver) {
			TypeIdx ty = static_cast<TypeIdx>(recvSlot);
			if (const auto *info = gctx.ctx.lookupEnum(ty)) {
				enumName = info->name;
				einfo = info;
			}
		} else {
			std::string recvName =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(recvSlot));
			enumName = recvName;
			einfo = gctx.ctx.getEnum(recvName);
		}
		if (einfo == nullptr) {
			// Not an enum. A bare-identifier pattern (infer-receiver, no
			// bindings) that names a module-level `const` is a CONSTANT
			// pattern: compare the scrutinee against the const's value,
			// exactly like a literal arm. Mirrors Rust, where an identifier
			// pattern that resolves to a const is the const, not a binding.
			if (inferReceiver && !hasBindings) {
				const std::string &constName =
				    gctx.ctx.getStringPool().get(variantNameId);
				if (const auto *mc = gctx.ctx.getModuleConst(constName)) {
					JirRef k = astgenExpr(gctx, mc->initExpr, scrutTy);
					JirRef cmp = emitCmp(JirTag::ICmpEq, scrut, k);
					emitCondBr(gctx, cmp, armBlock, nextBlock);
					return;
				}
			}
			failHere(gctx,
			         "astgen: pattern receiver doesn't resolve to an enum");
		}
		const std::string &variantName =
		    gctx.ctx.getStringPool().get(variantNameId);
		int vidx = gctx.ctx.getEnumVariantIndex(enumName, variantName);
		if (vidx < 0) {
			failHere(gctx, "astgen: unknown variant `" + enumName + "." +
			                   variantName + "`");
		}

		// Extract the tag as a u8. For payloaded enums the tag is
		// struct field 0; for unit-only enums the runtime form IS
		// the tag (LLVM-level i8) but the JIR type is the Named
		// enum, so emit an explicit BitCast to keep the type-check
		// happy at the verifier level.
		JirRef tagRef;
		if (einfo->hasPayloadVariant) {
			JirInst ev{};
			ev.tag = JirTag::ExtractValue;
			ev.a = scrut;
			ev.b = 0;
			ev.ty = BuiltinType::U8;
			tagRef = emit(gctx, ev);
		} else {
			JirInst bc{};
			bc.tag = JirTag::BitCast;
			bc.a = scrut;
			bc.ty = BuiltinType::U8;
			tagRef = emit(gctx, bc);
		}
		JirInst k{};
		k.tag = JirTag::Int;
		k.a = einfo->variants[vidx].discriminant;
		k.ty = BuiltinType::U8;
		JirRef kRef = emit(gctx, k);
		JirRef cmp = emitCmp(JirTag::ICmpEq, tagRef, kRef);

		// If bindings are present, we need to extract the payload after
		// the tag check succeeds. Create a small block between the
		// check and the arm body that does the extraction + binding
		// registration. Each binding extracts one payload field at its
		// computed byte offset within the payload area. The bindings
		// are stored as new locals so the arm body can read them via
		// Variable.
		if (hasBindings && bindingCount > 0) {
			JirBlockRef bindB = gctx.jfn.pushBlock("matchbind");
			emitCondBr(gctx, cmp, bindB, nextBlock);
			gctx.currentBlock = bindB;
			const auto &variant = einfo->variants[vidx];
			if (bindingCount != variant.payloadTypes.size()) {
				failHere(gctx, "astgen: pattern binds " +
				                   std::to_string(bindingCount) +
				                   " field(s), variant has " +
				                   std::to_string(variant.payloadTypes.size()));
			}

			// Spill scrut to a local alloca so EnumPayload can take its
			// address.
			JirInst alloca{};
			alloca.tag = JirTag::Alloca;
			alloca.ty = scrutTy;
			JirRef scrutSlot = emitAllocaHoisted(gctx, alloca);
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = scrutSlot;
			store.b = scrut;
			emit(gctx, store);

			uint64_t off = 0;
			for (uint32_t b = 0; b < bindingCount; b++) {
				StringIdx bindNameId =
				    static_cast<StringIdx>(ns.getExtra(bindingsStart + b));
				const std::string &bindName =
				    gctx.ctx.getStringPool().get(bindNameId);
				TypeIdx fieldTy = variant.payloadTypes[b];
				uint64_t s = gctx.ctx.typeSize(fieldTy);
				uint64_t a = gctx.ctx.typeAlign(fieldTy);
				off = (off + a - 1) / a * a;

				JirInst ep{};
				ep.tag = JirTag::EnumPayload;
				ep.a = scrutSlot;
				ep.b = static_cast<uint32_t>(off);  // byte offset
				ep.ty = fieldTy;
				JirRef payloadRef = emit(gctx, ep);

				JirInst bindAlloca{};
				bindAlloca.tag = JirTag::Alloca;
				bindAlloca.ty = fieldTy;
				JirRef bindSlot = emitAllocaHoisted(gctx, bindAlloca);
				JirInst bindStore{};
				bindStore.tag = JirTag::Store;
				bindStore.a = bindSlot;
				bindStore.b = payloadRef;
				emit(gctx, bindStore);
				if (outBindings) {
					outBindings->push_back({bindName, bindSlot, fieldTy});
				}
				off += s;
			}
			emitBr(gctx, armBlock);
			return;
		}
		emitCondBr(gctx, cmp, armBlock, nextBlock);
		return;
	}
	case AstTag::PatWildcard:
		// Wildcard inside an or-pattern: unconditionally match.
		emitBr(gctx, armBlock);
		return;
	default:
		failNode(gctx, patIdx, "this pattern form is not supported");
	}
}

static JirRef astgenMatch(AstGenCtx &gctx, const AstNode &n, TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	NodeIdx scrutIdx = static_cast<NodeIdx>(n.lhs);
	ExtraIdx armsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t armCount = ns.getExtra(armsExtra);

	// Decode arms eagerly so we can scan tails before lowering.
	struct ArmSpec {
		NodeIdx patIdx;
		std::vector<NodeIdx> body;
	};
	std::vector<ArmSpec> arms;
	arms.reserve(armCount);
	uint32_t pos = 1;
	int wildcardArmIdx = -1;
	for (uint32_t i = 0; i < armCount; i++) {
		ArmSpec a;
		a.patIdx = static_cast<NodeIdx>(ns.getExtra(armsExtra + pos));
		uint32_t bc = ns.getExtra(armsExtra + pos + 1);
		a.body.reserve(bc);
		for (uint32_t j = 0; j < bc; j++) {
			a.body.push_back(
			    static_cast<NodeIdx>(ns.getExtra(armsExtra + pos + 2 + j)));
		}
		pos += 2 + bc;
		if (ns.get(a.patIdx).tag == AstTag::PatWildcard) {
			wildcardArmIdx = static_cast<int>(arms.size());
		}
		arms.push_back(std::move(a));
	}

	// Peer-type pre-pass: when no caller-supplied expected, scan arm
	// tails and unify.
	TypeIdx peer = expected;
	if (peer == kNoType) {
		bool allInferred = true;
		for (const ArmSpec &a : arms) {
			if (a.body.empty()) continue;
			NodeIdx tail = a.body.back();
			if (stmtDiverges(gctx, tail)) continue;
			TypeIdx t = inferTailType(gctx, tail);
			if (t == kNoType) {
				allInferred = false;
				break;
			}
			peer = peerResolveType(peer, t, gctx.ctx.getTypePool());
		}
		if (!allInferred) peer = kNoType;
	}

	// Scrutinee.
	JirRef scrut = astgenExpr(gctx, scrutIdx, kNoType);
	TypeIdx scrutTy = gctx.jfn.getInst(scrut).ty;

	// MATCH-MOVE: matching a drop-bearing enum BY VALUE consumes the
	// scrutinee (Rust semantics). The match becomes the owner: binding
	// arms transfer the payload into drop-tracked bindings; arms that
	// don't bind (tag-only, wildcard, no-match fallthrough) drop the
	// residual payload at entry via the tag-dispatched glue. A bare
	// local scrutinee's own scope-exit drop is suppressed; an rvalue
	// scrutinee (`match (v.pop())`) is owned outright — which is what
	// finally gives temporary enums a drop point.
	const JamCodegenContext::EnumInfo *matchEinfo =
	    gctx.ctx.lookupEnum(scrutTy);
	// Gate must MATCH the analyzer's (typeNeedsDrop alone) — a
	// payload-less enum with its own cfn drop also consumes, or the two
	// layers disagree (analyzer says moved, codegen drops at scope
	// exit → false use-after-move rejections).
	bool matchOwns =
	    matchEinfo != nullptr && typeNeedsDrop(gctx.ctx, scrutTy);
	JirRef scrutOwned = kNoJirRef;
	if (matchOwns) {
		rejectDropBearingFieldExtract(gctx, scrutIdx, scrutTy, "consume");
		consumeMovedVariable(gctx, scrutIdx);
		// The residual-drop glue needs a stable POINTER to the owned
		// enum storage; the scrutinee here is the loaded value. One
		// spill, shared by every arm's residual path.
		JirInst sa{};
		sa.tag = JirTag::Alloca;
		sa.ty = scrutTy;
		scrutOwned = emitAllocaHoisted(gctx, sa);
		JirInst sst{};
		sst.tag = JirTag::Store;
		sst.a = scrutOwned;
		sst.b = scrut;
		emit(gctx, sst);
	}

	// Build merge + arm blocks up-front so we can hand refs around.
	JirBlockRef mergeB = gctx.jfn.pushBlock("matchend");
	std::vector<JirBlockRef> armBlocks;
	armBlocks.reserve(armCount);
	for (uint32_t i = 0; i < armCount; i++) {
		armBlocks.push_back(gctx.jfn.pushBlock("matcharm"));
	}

	// Result slot for expression-form match.
	JirRef resultSlot = kNoJirRef;
	if (peer != kNoType) {
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = peer;
		resultSlot = emitAllocaHoisted(gctx, alloca);
	}

	std::vector<ArmBindings> armBindings(armCount);
	JirBlockRef defaultB = (wildcardArmIdx >= 0)
	                           ? armBlocks[wildcardArmIdx]
	                           : gctx.jfn.pushBlock("nomatch");

	// Try the Switch lowering first. Every non-wildcard arm has to be
	// a pattern that resolves to a single integer-equality test (or a
	// PatOr of such patterns). Anything else — ranges, payload
	// bindings, mixed shapes — falls through to the CondBr chain.
	const TypeKey &sk = gctx.ctx.getTypePool().get(scrutTy);
	bool scrutIsInt = sk.kind == TypeKind::Int;
	const JamCodegenContext::EnumInfo *einfo = nullptr;
	if (sk.kind == TypeKind::Named || sk.kind == TypeKind::Enum) {
		einfo = gctx.ctx.lookupEnum(scrutTy);
	}
	bool scrutIsEnum = einfo != nullptr;
	bool tryingSwitch = (scrutIsInt || scrutIsEnum);
	std::vector<SwitchCase> switchCases;
	if (tryingSwitch) {
		const NodeStore &ns = gctx.ctx.getNodeStore();
		for (uint32_t i = 0; i < armCount; i++) {
			if (static_cast<int>(i) == wildcardArmIdx) continue;
			if (!collectSwitchCases(ns, gctx.ctx, arms[i].patIdx, armBlocks[i],
			                        scrutTy, scrutIsEnum, einfo, switchCases)) {
				tryingSwitch = false;
				break;
			}
		}
		// LLVM's SwitchInst requires unique case values; passing duplicates
		// trips an assert in some builds and throws std::bad_alloc from
		// SwitchLowering::buildJumpTable in LLVM 22 release builds. First-
		// match-wins semantics: if value V appears in arm A and arm B (A<B),
		// arm A's target must win, so we keep the earliest occurrence.
		std::unordered_set<uint64_t> seen;
		switchCases.erase(
		    std::remove_if(switchCases.begin(), switchCases.end(),
		                   [&](const SwitchCase &c) {
			                   return !seen.insert(c.value).second;
		                   }),
		    switchCases.end());
	}

	if (tryingSwitch) {
		// Build the integer scrutinee. For an enum scrut we pull out
		// the discriminant byte: ExtractValue(0) for payloaded enums,
		// BitCast otherwise (the LLVM runtime form is already i8 for
		// unit-only enums but JIR carries the Named type).
		JirRef caseScrut = scrut;
		TypeIdx caseScrutTy = scrutTy;
		if (scrutIsEnum) {
			JirInst extract{};
			if (einfo->hasPayloadVariant) {
				extract.tag = JirTag::ExtractValue;
				extract.a = scrut;
				extract.b = 0;
			} else {
				extract.tag = JirTag::BitCast;
				extract.a = scrut;
			}
			extract.ty = BuiltinType::U8;
			caseScrut = emit(gctx, extract);
			caseScrutTy = BuiltinType::U8;
		}

		// Encode extras: [defaultBlock, caseCount,
		//                 (lo, hi, signed, target) × N].
		uint32_t caseCount = static_cast<uint32_t>(switchCases.size());
		std::vector<uint32_t> packed;
		packed.reserve(2 + caseCount * 4);
		packed.push_back(static_cast<uint32_t>(defaultB));
		packed.push_back(caseCount);
		for (const SwitchCase &sc : switchCases) {
			packed.push_back(static_cast<uint32_t>(sc.value & 0xFFFFFFFFu));
			packed.push_back(static_cast<uint32_t>(sc.value >> 32));
			packed.push_back(sc.isSigned ? 1u : 0u);
			packed.push_back(static_cast<uint32_t>(sc.target));
		}
		JirExtraIdx extraIdx = gctx.jfn.pushExtra(packed.data(), packed.size());
		JirInst sw{};
		sw.tag = JirTag::Switch;
		sw.a = caseScrut;
		sw.b = extraIdx;
		(void)caseScrutTy;
		emit(gctx, sw);

		// Default block branches to merge when no wildcard arm exists,
		// matching the chained-CondBr semantics (non-exhaustive match
		// falls through; expression-form match leaves the result slot
		// at its default). An owned scrutinee still drops its payload
		// on this path.
		if (wildcardArmIdx < 0) {
			gctx.currentBlock = defaultB;
			if (matchOwns) {
				emitDropInPlace(gctx, scrutOwned, scrutTy);
			}
			emitBr(gctx, mergeB);
		}
	} else {
		// Dispatch chain. Skip the wildcard arm in the per-pattern
		// compare loop; it becomes the final fallthrough block.
		bool emittedAnyDispatch = false;
		for (uint32_t i = 0; i < armCount; i++) {
			if (static_cast<int>(i) == wildcardArmIdx) continue;
			JirBlockRef next =
			    (i + 1 < armCount && static_cast<int>(i + 1) != wildcardArmIdx)
			        ? gctx.jfn.pushBlock("matchnext")
			        : defaultB;
			astgenPatternCompare(gctx, arms[i].patIdx, scrut, scrutTy,
			                     armBlocks[i], next, &armBindings[i]);
			if (next != defaultB) gctx.currentBlock = next;
			emittedAnyDispatch = true;
		}
		// All arms were wildcards (only a `_` arm, or none). Emit an
		// unconditional Br from the entry block into the default
		// block.
		if (!emittedAnyDispatch) { emitBr(gctx, defaultB); }
		// If the last non-wildcard arm fell through, we need its
		// `next` to terminate. The compare loop above already wired
		// the final next to `defaultB`. When there's no catch-all
		// wildcard, fall through to merge — matches legacy semantics
		// where a match without `_` simply leaves any prior state
		// unchanged. (For expression-form matches the result slot
		// stays at its uninitialised value; user code that relies on
		// the value after a non-exhaustive match is at fault.)
		if (wildcardArmIdx < 0) {
			gctx.currentBlock = defaultB;
			if (matchOwns) {
				emitDropInPlace(gctx, scrutOwned, scrutTy);
			}
			emitBr(gctx, mergeB);
		}
	}

	// Arm bodies. Each arm installs its bindings into gctx.locals
	// for the duration of body lowering, then removes them so the
	// next arm sees a clean scope. Each arm runs in its own drop
	// scope so locals declared inside the body are destroyed before
	// the branch reaches the merge block.
	for (uint32_t i = 0; i < armCount; i++) {
		gctx.currentBlock = armBlocks[i];
		// Install bindings (saving any prior entries with same name).
		std::vector<std::pair<std::string, JirRef>> saved;
		std::vector<std::pair<std::string, TypeIdx>> savedTypes;
		for (const ArmBinding &bind : armBindings[i]) {
			auto it = gctx.locals.find(bind.name);
			if (it != gctx.locals.end()) {
				saved.push_back({bind.name, it->second});
				savedTypes.push_back({bind.name, gctx.localTypes[bind.name]});
			}
			gctx.locals[bind.name] = bind.slot;
			gctx.localTypes[bind.name] = bind.type;
		}
		pushDropScope(gctx);

		if (matchOwns) {
			if (armBindings[i].empty()) {
				// No bindings extracted the payload: this arm owns the
				// residual. Full drop-in-place: the enum's own cfn drop
				// (if any) plus the tag-dispatched payload glue.
				emitDropInPlace(gctx, scrutOwned, scrutTy);
			} else {
				// Rust's E0509 analog: the payload cannot move out of
				// an enum that has its OWN cfn drop — that drop expects
				// the whole value intact, so extracting a field from
				// under it would leave it running on a hollowed value.
				if (!lookupDropFnLLVMName(gctx.ctx, scrutTy).empty()) {
					failNode(gctx, arms[i].patIdx,
					         "cannot bind the payload out of an enum "
					         "that has its own `cfn drop` — the drop "
					         "needs the whole value; clone the payload "
					         "or restructure");
				}
				// Bindings took the payload — they become drop-tracked
				// owners in the arm's scope (same dual-flavor
				// registration var-decls use).
				for (const ArmBinding &bind : armBindings[i]) {
					std::string dn =
					    lookupDropFnLLVMName(gctx.ctx, bind.type);
					if (!dn.empty()) {
						gctx.dropScopes.back().push_back(
						    {bind.name, bind.slot, bind.type, dn});
					} else if (typeNeedsDrop(gctx.ctx, bind.type)) {
						gctx.dropScopes.back().push_back(
						    {bind.name, bind.slot, bind.type,
						     std::string()});
					}
				}
			}
		}

		const ArmSpec &a = arms[i];
		bool armDiverged = false;
		for (size_t s = 0; s < a.body.size(); s++) {
			NodeIdx stmt = a.body[s];
			bool isTail = (s + 1 == a.body.size());
			bool divergent = stmtDiverges(gctx, stmt);
			if (isTail && peer != kNoType && !divergent) {
				JirRef val = astgenExpr(gctx, stmt, peer);
				JirInst store{};
				store.tag = JirTag::Store;
				store.a = resultSlot;
				store.b = val;
				emit(gctx, store);
			} else {
				astgenExpr(gctx, stmt, kNoType);
			}
			if (isTail && divergent) { armDiverged = true; }
		}
		// A noreturn-call tail (like `abort()`) leaves the arm with no
		// LLVM terminator even though semantically control can't flow
		// past it. Emit `unreachable` so the verifier is happy.
		if (armDiverged &&
		    !blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock),
		                        gctx.jfn)) {
			JirInst u{};
			u.tag = JirTag::Unreachable;
			emit(gctx, u);
		}
		popDropScopeEmitting(gctx);
		if (!blockHasTerminator(gctx.jfn.getBlock(gctx.currentBlock),
		                        gctx.jfn)) {
			emitBr(gctx, mergeB);
		}

		// Pop bindings.
		for (const ArmBinding &bind : armBindings[i]) {
			gctx.locals.erase(bind.name);
			gctx.localTypes.erase(bind.name);
		}
		for (const auto &kv : saved) gctx.locals[kv.first] = kv.second;
		for (const auto &kv : savedTypes) gctx.localTypes[kv.first] = kv.second;
	}

	gctx.currentBlock = mergeB;
	if (resultSlot == kNoJirRef) return kNoJirRef;
	JirInst load{};
	load.tag = JirTag::Load;
	load.a = resultSlot;
	load.ty = peer;
	return emit(gctx, load);
}

// AstGen for `TypeMethodCall` (`Vec(i32).empty()`, `Color.Red()`).
// Resolves the receiver TypeIdx to its struct/enum name (triggering
// lazy generic instantiation as a side effect), synthesizes a regular
// Call AST node with qualified name `ReceiverName.method`, and recurses
// into astgenCall. Instantiation runs the full astgen -> jirDefineBody
// pipeline in `JamCodegenContext::instantiateStructExpr`, so the JIR
// Call here resolves cleanly by LLVM name at codegen time.
static JirRef astgenTypeMethodCall(AstGenCtx &gctx, const AstNode &n,
                                   JirRef destPtr) {
	const NodeStore &nsConst = gctx.ctx.getNodeStore();
	NodeStore &ns = const_cast<NodeStore &>(nsConst);
	TypeIdx recvTy = static_cast<TypeIdx>(n.lhs);
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	StringIdx methodNameId = static_cast<StringIdx>(ns.getExtra(extra));
	uint32_t argCount = ns.getExtra(extra + 1);
	const std::string &methodName = gctx.ctx.getStringPool().get(methodNameId);

	std::string receiverName;
	const auto *einfoForVariant =
	    static_cast<const decltype(gctx.ctx.getEnum(""))>(nullptr);
	if (const auto *sinfo = gctx.ctx.lookupStruct(recvTy)) {
		receiverName = sinfo->name;
	} else if (const auto *einfo = gctx.ctx.lookupEnum(recvTy)) {
		receiverName = einfo->name;
		einfoForVariant = einfo;
	} else {
		failHere(gctx, "astgen: TypeMethodCall receiver doesn't resolve to a "
		               "struct or enum");
	}

	// Enum-variant constructor: `Option(i32).Some(42)` — the method
	// name is a variant rather than an actual method. Build a
	// {tag, payload} aggregate. v1 supports single-payload variants
	// (the dominant case for Option/Result/either-style enums).
	if (einfoForVariant != nullptr) {
		int vidx = gctx.ctx.getEnumVariantIndex(receiverName, methodName);
		if (vidx >= 0) {
			const auto &variant = einfoForVariant->variants[vidx];
			TypeIdx enumTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::Named, 0, 0,
			            static_cast<uint32_t>(
			                gctx.ctx.getStringPool().intern(receiverName)),
			            0});
			JirInst tag{};
			tag.tag = JirTag::Int;
			tag.a = static_cast<uint32_t>(variant.discriminant);
			tag.ty = BuiltinType::U8;
			JirRef tagRef = emit(gctx, tag);
			if (!einfoForVariant->hasPayloadVariant) { return tagRef; }
			// Payloaded enum: alloca enum struct, store tag at field 0,
			// store each payload arg at the payload area.
			JirInst alloca{};
			alloca.tag = JirTag::Alloca;
			alloca.ty = enumTy;
			JirRef slot = emitAllocaHoisted(gctx, alloca);
			// Store the tag as a struct field — use FieldAccess +
			// StructLit isn't quite right since we have a heterogeneous
			// store. Use a series of EnumPayload-like stores. For v1,
			// build the struct via Load(undef) + InsertValue + Store.
			// Simpler: emit a synthesized StructLit with one tag-only
			// initializer, store that into the alloca, then store the
			// payload field separately via the payload extraction
			// inverse.
			//
			// For v1 we emit a custom JIR sequence: store tag, then
			// store payload. Since we don't have a "StorePayload" tag
			// yet, hand-roll via Load+InsertValue+Store wouldn't reach
			// the right field offsets — payload field 1 is the align-
			// driver slot, not the payload type itself.
			//
			// Pragmatic path: use the legacy-style approach — store the
			// {tag, undef} aggregate by building via StructLit with the
			// tag, then bitcast the payload pointer and store the
			// payload value at the payload-area offset. Encode the
			// payload-store via a fresh JIR tag `EnumStorePayload`
			// (deferred — for now we limit v1 to unit variants).
			// Unit-variant or single-payload-variant inside payloaded enum.
			JirInst tagFieldAddr{};
			tagFieldAddr.tag = JirTag::FieldAddr;
			tagFieldAddr.a = slot;
			tagFieldAddr.b = 0;
			// Pointer-to-i8 for tag field
			TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});
			tagFieldAddr.ty = u8PtrTy;
			JirRef tagPtr = emit(gctx, tagFieldAddr);
			JirInst tagStore{};
			tagStore.tag = JirTag::Store;
			tagStore.a = tagPtr;
			tagStore.b = tagRef;
			emit(gctx, tagStore);

			// Store payload if present. v1: single payload field at
			// the payload-area's offset 0 (which is struct field 1).
			if (!variant.payloadTypes.empty() && argCount >= 1) {
				TypeIdx fieldTy = variant.payloadTypes[0];
				NodeIdx argIdx0 = static_cast<NodeIdx>(ns.getExtra(extra + 2));
				JirRef payloadVal = astgenExpr(gctx, argIdx0, fieldTy);
				JirInst payloadFieldAddr{};
				payloadFieldAddr.tag = JirTag::FieldAddr;
				payloadFieldAddr.a = slot;
				payloadFieldAddr.b = 1;
				TypeIdx pfPtrTy = gctx.ctx.getTypePool().intern(
				    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
				payloadFieldAddr.ty = pfPtrTy;
				JirRef payPtr = emit(gctx, payloadFieldAddr);
				JirInst payStore{};
				payStore.tag = JirTag::Store;
				payStore.a = payPtr;
				payStore.b = payloadVal;
				emit(gctx, payStore);
				// Enum payload capture is a MOVE: the enum owns the
				// value now (its tag-dispatched drop glue runs it).
				rejectDropBearingFieldExtract(gctx, argIdx0, fieldTy,
				                              "capture");
				consumeMovedVariable(gctx, argIdx0);
			}

			JirInst load{};
			load.tag = JirTag::Load;
			load.a = slot;
			load.ty = enumTy;
			return emit(gctx, load);
		}
	}

	std::string qualified = receiverName + "." + methodName;
	StringIdx calleeId = gctx.ctx.getStringPool().intern(qualified);

	// Synthesize a fresh Call AST node pointing into the same arg
	// nodes already stored in the TypeMethodCall extra. The synthesized
	// node is appended to NodeStore so we get a real NodeIdx if needed
	// for diagnostics, but we don't actually need that — we lower it
	// directly via astgenCall.
	ExtraIdx callExtra = ns.reserveExtra(1 + argCount);
	ns.setExtra(callExtra, argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		ns.setExtra(callExtra + 1 + i, ns.getExtra(extra + 2 + i));
	}
	AstNode synth{AstTag::Call, 0, 0, 0, static_cast<uint32_t>(calleeId),
	              callExtra};
	// Inline the astgenCall body since we have a synthesized AstNode
	// (not a NodeIdx). For the inner-call lookup we re-derive the
	// callee's FunctionAST via the same registry path.
	const FunctionAST *fn = gctx.ctx.getFunctionAST(qualified);
	if (fn == nullptr) {
		return reportMethodMiss(gctx, qualified);
	}
	std::vector<JirRef> argRefs;
	argRefs.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(ns.getExtra(callExtra + 1 + i));
		TypeIdx expectArg = (i < fn->Args.size()) ? fn->Args[i].Type : kNoType;
		if (i < fn->Args.size()) {
			rejectAddrOfOnModeArg(gctx, argIdx, fn->Args[i]);
			if (fn->Args[i].Mode == ParamMode::Move) {
				rejectDropBearingFieldExtract(gctx, argIdx, expectArg,
				                              "move");
			}
		}
		argRefs.push_back(astgenExpr(gctx, argIdx, expectArg));
		// `move`-mode arg through the `Type.method(args)` form
		// (`Box(Counter).init(c)`): ownership transfers to the callee —
		// suppress the caller-side scope-exit drop, same as lowerArg.
		if (i < fn->Args.size() && fn->Args[i].Mode == ParamMode::Move) {
			consumeMovedVariable(gctx, argIdx);
		}
	}
	return emitCall(gctx, fn, argRefs, destPtr);
}

// Collapse a GenericCall + type-alias chain to its concrete struct
// instantiation. Walks at most `kMaxHops` levels — covers
//   const BumperI32 = Bumper(i32);     // alias → GenericCall → Named
//   const NestedBumper = Bumper(BumperI32);
// where a substituted field type lands as `Named("BumperI32")` and the
// drop-fn lookup has to chase the alias to reach `Bumper__i32` whose
// drop is registered.
static TypeIdx resolveGenericIfAny(JamCodegenContext &ctx, TypeIdx ty) {
	const auto &types = ctx.getTypePool();
	const auto &strings = ctx.getStringPool();
	constexpr int kMaxHops = 8;
	for (int i = 0; i < kMaxHops; i++) {
		const TypeKey &tk = types.get(ty);
		if (tk.kind == TypeKind::GenericCall) {
			TypeIdx r = ctx.resolveGenericCall(ty);
			if (r == kNoType || r == ty) break;
			ty = r;
			continue;
		}
		if (tk.kind == TypeKind::Named) {
			const std::string &name = strings.get(static_cast<StringIdx>(tk.a));
			TypeIdx alias = ctx.lookupTypeAlias(name);
			if (alias != kNoType && alias != ty) {
				ty = alias;
				continue;
			}
		}
		break;
	}
	return ty;
}

// True if `ty` carries any heap ownership or field that does. Used to
// decide whether a `var` binding needs scope-exit drop tracking even
// when its type doesn't itself declare `cfn drop`. Mirrors Rust's
// `needs_drop<T>()` which recurses into struct fields. Cycles are not
// possible for non-pointer/named-only descent — pointer fields are
// treated as primitives (Rust's same answer: a raw pointer doesn't
// auto-drop its pointee; if the user wants that, they own a Box/Vec
// whose own `cfn drop` handles deallocation).
bool typeNeedsDropInner(JamCodegenContext &ctx, TypeIdx ty) {
	ty = resolveGenericIfAny(ctx, ty);
	if (!lookupDropFnLLVMName(ctx, ty).empty()) return true;
	const TypeKey &tk = ctx.getTypePool().get(ty);
	// A fixed-size array owns its elements: it needs drop iff the
	// element type does. (Deferred `[expr]T` sizes resolve first so
	// field types annotated with const lengths classify correctly.)
	if (tk.kind == TypeKind::ArrayExpr) {
		return typeNeedsDrop(ctx, ctx.resolveArrayExpr(ty));
	}
	if (tk.kind == TypeKind::Array) {
		return typeNeedsDrop(ctx, static_cast<TypeIdx>(tk.a));
	}
	// A payloaded enum owns whichever variant payload is live: it needs
	// drop iff ANY variant carries a payload type that does.
	if (const auto *einfo = ctx.lookupEnum(ty)) {
		for (const auto &v : einfo->variants) {
			for (TypeIdx pt : v.payloadTypes) {
				if (typeNeedsDrop(ctx, pt)) return true;
			}
		}
		return false;
	}
	if (tk.kind != TypeKind::Struct && tk.kind != TypeKind::Named) {
		return false;
	}
	const std::string &name =
	    ctx.getStringPool().get(static_cast<StringIdx>(tk.a));
	const auto *sinfo = ctx.getStruct(name);
	if (sinfo == nullptr) return false;
	for (const auto &f : sinfo->fields) {
		if (typeNeedsDrop(ctx, f.second)) return true;
	}
	return false;
}

// True when an assignment target is an lvalue path rooted at a
// local/param where every step stays in the VALUE world: struct/union
// fields and fixed-array elements. Pointer-typed segments (`p.*`,
// `self.ptr[i]`, slice indexing) are raw memory — Vec.push writes
// uninitialized capacity slots through them, so auto-dropping the
// "old" bytes there would drop garbage (the same reason Rust assigns
// through `ptr::write` in container internals). Value-world
// destinations are definitely initialized (every jam binding
// initializes at declaration), so dropping the overwritten value is
// always sound.
static bool assignTargetIsValueWorld(AstGenCtx &gctx, NodeIdx targetIdx) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	// Collect the projection steps leaf->root, then type-check them
	// root->leaf so each Index can be classified Array-vs-pointer.
	std::vector<const AstNode *> steps;
	NodeIdx cur = targetIdx;
	const AstNode *root = nullptr;
	while (cur != kNoNode) {
		const AstNode &nd = ns.get(cur);
		if (nd.tag == AstTag::Variable) {
			root = &nd;
			break;
		}
		if (nd.tag == AstTag::MemberAccess || nd.tag == AstTag::Index) {
			steps.push_back(&nd);
			cur = static_cast<NodeIdx>(nd.lhs);
			continue;
		}
		return false;  // Deref or any other shape: pointer world
	}
	if (root == nullptr) return false;
	const std::string &rootName =
	    gctx.ctx.getStringPool().get(static_cast<StringIdx>(root->lhs));
	auto it = gctx.localTypes.find(rootName);
	if (it == gctx.localTypes.end()) return false;
	TypeIdx curTy = it->second;
	for (auto rit = steps.rbegin(); rit != steps.rend(); ++rit) {
		curTy = resolveGenericIfAny(gctx.ctx, curTy);
		const TypeKey &k = gctx.ctx.getTypePool().get(curTy);
		if ((*rit)->tag == AstTag::Index) {
			if (k.kind != TypeKind::Array) return false;
			curTy = static_cast<TypeIdx>(k.a);
			continue;
		}
		// MemberAccess: resolve the field's type through the struct
		// registry; unions and anything unresolvable bail conservative.
		if (k.kind != TypeKind::Struct && k.kind != TypeKind::Named) {
			return false;
		}
		const auto *sinfo = gctx.ctx.lookupStruct(curTy);
		if (sinfo == nullptr) return false;
		const std::string &member = gctx.ctx.getStringPool().get(
		    static_cast<StringIdx>((*rit)->rhs));
		int fidx = gctx.ctx.getFieldIndex(sinfo->name, member);
		if (fidx < 0) return false;
		curTy = sinfo->fields[static_cast<size_t>(fidx)].second;
	}
	return true;
}

// Synthesize a Rust-style `drop_in_place::<T>(*T)` call sequence at `ptrRef`
// for pointee type `pointeeTy`. Walks the type recursively:
//   1. If T has a registered `cfn drop`: emit a Call to T.drop(ptrRef).
//   2. Else if T is a struct: for each field with `needsDrop`, GEP its
//      address and recurse with the field type.
//   3. Else (primitive, no-drop type): emit nothing.
// All emission is inline — there's no synthesized helper function, no
// runtime cost beyond the explicit drop calls. The compiler-known type
// of `ptrRef` is the dispatch key.
// Walk T's struct fields and recursively emit @dropInPlace on each that
// needs drop. Does NOT fire T's own `cfn drop`; the caller is
// responsible for that (if applicable). Factored out of emitDropInPlace
// so emitDrops can re-use it to mirror Rust's "Drop::drop runs, then
// fields auto-drop" sequencing.
static void emitFieldDrops(AstGenCtx &gctx, JirRef ptrRef, TypeIdx pointeeTy) {
	pointeeTy = resolveGenericIfAny(gctx.ctx, pointeeTy);
	const TypeKey &tk = gctx.ctx.getTypePool().get(pointeeTy);
	const JamCodegenContext::StructInfo *sinfo = nullptr;
	if (tk.kind == TypeKind::Struct || tk.kind == TypeKind::Named) {
		std::string name =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(tk.a));
		sinfo = gctx.ctx.getStruct(name);
	}
	if (sinfo == nullptr) return;
	for (size_t i = 0; i < sinfo->fields.size(); i++) {
		TypeIdx fieldTy = sinfo->fields[i].second;
		if (!typeNeedsDrop(gctx.ctx, fieldTy)) continue;
		TypeIdx fieldPtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
		JirInst fieldAddr{};
		fieldAddr.tag = JirTag::FieldAddr;
		fieldAddr.a = ptrRef;
		fieldAddr.b = static_cast<uint32_t>(i);
		fieldAddr.ty = fieldPtrTy;
		JirRef fieldPtr = emit(gctx, fieldAddr);
		emitDropInPlace(gctx, fieldPtr, fieldTy);
	}
}

// If `containerTy` is a contiguous owning container — a struct exposing a
// `cfn len(self)` count accessor and exactly one many-pointer data field —
// synthesize its element destructors: loop `0..len()` and recursively
// `emitDropInPlace` each element. Reuses the same JIR a `for` loop emits
// (Alloca/Load/ICmpUlt/Add/Br/CondBr), `emitCall` for `len`, and the existing
// `emitDropInPlace` recursion. Returns true if a loop was emitted.
//
// Runs BEFORE the container's own `cfn drop` (which frees the backing) so the
// elements are destroyed while their storage is still live. POD element types
// (typeNeedsDrop == false) emit nothing — the loop is skipped.
static bool emitContainerElementDrops(AstGenCtx &gctx, JirRef ptrRef,
                                      TypeIdx containerTy) {
	const auto *sinfo = gctx.ctx.lookupStruct(containerTy);
	if (sinfo == nullptr) return false;

	// Opt-in marker: a `cfn len`. A plain `fn len` (or none) is not treated
	// as a drop-synthesizable container — this prevents element-dropping a
	// borrowing view that merely happens to expose a length + pointer.
	const FunctionAST *lenFn = gctx.ctx.getFunctionAST(sinfo->name + ".len");
	if (lenFn == nullptr || !lenFn->isCfn || lenFn->Args.empty()) return false;

	// The single many-pointer field is the data base; its pointee is the
	// element type. More than one many-pointer field is ambiguous.
	int dataIdx = -1;
	TypeIdx elemTy = kNoType;
	for (size_t i = 0; i < sinfo->fields.size(); i++) {
		const TypeKey &fk = gctx.ctx.getTypePool().get(sinfo->fields[i].second);
		if (fk.kind == TypeKind::PtrMany) {
			if (dataIdx >= 0) return false;
			dataIdx = static_cast<int>(i);
			elemTy = static_cast<TypeIdx>(fk.a);
		}
	}
	if (dataIdx < 0 || !typeNeedsDrop(gctx.ctx, elemTy)) return false;

	// count = self.len()  — receiver is the pointer-to-self we already hold.
	jam::abi::ParamABI recvAbi =
	    jam::abi::classifyParam(lenFn->Args[0].Mode, containerTy, gctx.ctx);
	JirRef recvRef;
	if (recvAbi.kind == jam::abi::ParamABI::Kind::ByPointer) {
		recvRef = ptrRef;
	} else {
		JirInst ld{};
		ld.tag = JirTag::Load;
		ld.a = ptrRef;
		ld.ty = containerTy;
		recvRef = emit(gctx, ld);
	}
	std::vector<JirRef> lenArgs;
	lenArgs.push_back(recvRef);
	JirRef countRef = emitCall(gctx, lenFn, lenArgs);
	TypeIdx countTy = gctx.jfn.getInst(countRef).ty;

	// base = load self.<dataField>  → a `*mut[] Elem` value.
	TypeIdx dataFieldTy = sinfo->fields[dataIdx].second;
	TypeIdx dataAddrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrSingle, 0, 0, dataFieldTy, 0});
	JirInst fa{};
	fa.tag = JirTag::FieldAddr;
	fa.a = ptrRef;
	fa.b = static_cast<uint32_t>(dataIdx);
	fa.ty = dataAddrTy;
	JirRef dataAddr = emit(gctx, fa);
	JirInst ldb{};
	ldb.tag = JirTag::Load;
	ldb.a = dataAddr;
	ldb.ty = dataFieldTy;
	JirRef base = emit(gctx, ldb);

	// i = 0
	JirInst alloca{};
	alloca.tag = JirTag::Alloca;
	alloca.ty = countTy;
	JirRef slot = emitAllocaHoisted(gctx, alloca);
	JirInst zero{};
	zero.tag = JirTag::Int;
	zero.a = 0;
	zero.ty = countTy;
	JirRef zeroRef = emit(gctx, zero);
	JirInst st0{};
	st0.tag = JirTag::Store;
	st0.a = slot;
	st0.b = zeroRef;
	emit(gctx, st0);

	JirBlockRef condB = gctx.jfn.pushBlock("dropcond");
	JirBlockRef bodyB = gctx.jfn.pushBlock("dropbody");
	JirBlockRef exitB = gctx.jfn.pushBlock("dropexit");
	emitBr(gctx, condB);

	// cond: i < count
	gctx.currentBlock = condB;
	JirInst li{};
	li.tag = JirTag::Load;
	li.a = slot;
	li.ty = countTy;
	JirRef iVal = emit(gctx, li);
	JirInst cmp{};
	cmp.tag = JirTag::ICmpUlt;
	cmp.a = iVal;
	cmp.b = countRef;
	cmp.ty = BuiltinType::Bool;
	JirRef cmpRef = emit(gctx, cmp);
	emitCondBr(gctx, cmpRef, bodyB, exitB);

	// body: drop element i; i = i + 1
	gctx.currentBlock = bodyB;
	JirInst lib{};
	lib.tag = JirTag::Load;
	lib.a = slot;
	lib.ty = countTy;
	JirRef iBody = emit(gctx, lib);
	TypeIdx elemPtrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrMany, 0, 0, elemTy, 0});
	JirInst ia{};
	ia.tag = JirTag::IndexAddr;
	ia.a = base;
	ia.b = iBody;
	ia.ty = elemPtrTy;
	JirRef elemPtr = emit(gctx, ia);
	emitDropInPlace(gctx, elemPtr, elemTy);
	JirInst li2{};
	li2.tag = JirTag::Load;
	li2.a = slot;
	li2.ty = countTy;
	JirRef cur = emit(gctx, li2);
	JirInst one{};
	one.tag = JirTag::Int;
	one.a = 1;
	one.ty = countTy;
	JirRef oneRef = emit(gctx, one);
	JirInst inc{};
	inc.tag = JirTag::Add;
	inc.a = cur;
	inc.b = oneRef;
	inc.ty = countTy;
	JirRef next = emit(gctx, inc);
	JirInst st1{};
	st1.tag = JirTag::Store;
	st1.a = slot;
	st1.b = next;
	emit(gctx, st1);
	emitBr(gctx, condB);

	gctx.currentBlock = exitB;
	return true;
}

// Per-element drop loop for a fixed-size array at `ptrRef`. Same loop
// shape as emitContainerElementDrops, but the count is the compile-time
// array length and the base is the array storage itself. Elements drop
// in index order (0..N).
static void emitArrayElementDrops(AstGenCtx &gctx, JirRef ptrRef,
                                  TypeIdx arrTy) {
	const TypeKey &ak = gctx.ctx.getTypePool().get(arrTy);
	TypeIdx elemTy = static_cast<TypeIdx>(ak.a);
	uint32_t count = ak.b;
	if (count == 0) return;

	TypeIdx countTy = BuiltinType::U64;
	JirInst countI{};
	countI.tag = JirTag::Int;
	countI.a = count;
	countI.ty = countTy;
	JirRef countRef = emit(gctx, countI);

	// i = 0
	JirInst alloca{};
	alloca.tag = JirTag::Alloca;
	alloca.ty = countTy;
	JirRef slot = emitAllocaHoisted(gctx, alloca);
	JirInst zero{};
	zero.tag = JirTag::Int;
	zero.a = 0;
	zero.ty = countTy;
	JirRef zeroRef = emit(gctx, zero);
	JirInst st0{};
	st0.tag = JirTag::Store;
	st0.a = slot;
	st0.b = zeroRef;
	emit(gctx, st0);

	JirBlockRef condB = gctx.jfn.pushBlock("adropcond");
	JirBlockRef bodyB = gctx.jfn.pushBlock("adropbody");
	JirBlockRef exitB = gctx.jfn.pushBlock("adropexit");
	emitBr(gctx, condB);

	gctx.currentBlock = condB;
	JirInst li{};
	li.tag = JirTag::Load;
	li.a = slot;
	li.ty = countTy;
	JirRef iVal = emit(gctx, li);
	JirInst cmp{};
	cmp.tag = JirTag::ICmpUlt;
	cmp.a = iVal;
	cmp.b = countRef;
	cmp.ty = BuiltinType::Bool;
	JirRef cmpRef = emit(gctx, cmp);
	emitCondBr(gctx, cmpRef, bodyB, exitB);

	gctx.currentBlock = bodyB;
	JirInst lib{};
	lib.tag = JirTag::Load;
	lib.a = slot;
	lib.ty = countTy;
	JirRef iBody = emit(gctx, lib);
	TypeIdx elemPtrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrSingle, 0, 0, elemTy, 0});
	JirInst ia{};
	ia.tag = JirTag::IndexAddr;
	ia.a = ptrRef;
	ia.b = iBody;
	ia.ty = elemPtrTy;
	JirRef elemPtr = emit(gctx, ia);
	emitDropInPlace(gctx, elemPtr, elemTy);
	JirInst li2{};
	li2.tag = JirTag::Load;
	li2.a = slot;
	li2.ty = countTy;
	JirRef cur = emit(gctx, li2);
	JirInst one{};
	one.tag = JirTag::Int;
	one.a = 1;
	one.ty = countTy;
	JirRef oneRef = emit(gctx, one);
	JirInst inc{};
	inc.tag = JirTag::Add;
	inc.a = cur;
	inc.b = oneRef;
	inc.ty = countTy;
	JirRef next = emit(gctx, inc);
	JirInst st1{};
	st1.tag = JirTag::Store;
	st1.a = slot;
	st1.b = next;
	emit(gctx, st1);
	emitBr(gctx, condB);

	gctx.currentBlock = exitB;
}

// Tag-dispatched payload drops for a payloaded enum at `ptrRef`. Loads
// the discriminant (struct field 0) and, for each variant whose payload
// needs dropping, branches to a block that drops each payload field at
// its byte offset within the payload area (field 1) — the SAME
// offset computation the variant constructors use (align each field to
// its own alignment, advance by its size), so construction and
// destruction can never disagree about where a payload lives.
static void emitEnumPayloadDrops(AstGenCtx &gctx, JirRef ptrRef,
                                 const JamCodegenContext::EnumInfo *einfo) {
	TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});

	// tag = load (u8*)&self.field0
	JirInst tagFA{};
	tagFA.tag = JirTag::FieldAddr;
	tagFA.a = ptrRef;
	tagFA.b = 0;
	tagFA.ty = u8PtrTy;
	JirRef tagPtr = emit(gctx, tagFA);
	JirInst tagLd{};
	tagLd.tag = JirTag::Load;
	tagLd.a = tagPtr;
	tagLd.ty = BuiltinType::U8;
	JirRef tagVal = emit(gctx, tagLd);

	for (const auto &v : einfo->variants) {
		bool anyDrop = false;
		for (TypeIdx pt : v.payloadTypes) {
			if (typeNeedsDrop(gctx.ctx, pt)) {
				anyDrop = true;
				break;
			}
		}
		if (!anyDrop) continue;

		JirBlockRef dropB = gctx.jfn.pushBlock("edrop");
		JirBlockRef contB = gctx.jfn.pushBlock("edropcont");
		JirInst disc{};
		disc.tag = JirTag::Int;
		disc.a = static_cast<uint32_t>(v.discriminant);
		disc.ty = BuiltinType::U8;
		JirRef discRef = emit(gctx, disc);
		JirInst cmp{};
		cmp.tag = JirTag::ICmpEq;
		cmp.a = tagVal;
		cmp.b = discRef;
		cmp.ty = BuiltinType::Bool;
		JirRef cmpRef = emit(gctx, cmp);
		emitCondBr(gctx, cmpRef, dropB, contB);

		gctx.currentBlock = dropB;
		JirInst payFA{};
		payFA.tag = JirTag::FieldAddr;
		payFA.a = ptrRef;
		payFA.b = 1;
		payFA.ty = u8PtrTy;
		JirRef payArea = emit(gctx, payFA);
		uint64_t off = 0;
		for (TypeIdx pt : v.payloadTypes) {
			uint64_t s = gctx.ctx.typeSize(pt);
			uint64_t a = gctx.ctx.typeAlign(pt);
			off = (off + a - 1) / a * a;
			if (typeNeedsDrop(gctx.ctx, pt)) {
				JirInst offC{};
				offC.tag = JirTag::Int;
				offC.a = static_cast<uint32_t>(off);
				offC.ty = BuiltinType::U64;
				JirRef offRef = emit(gctx, offC);
				JirInst gep{};
				gep.tag = JirTag::IndexAddr;
				gep.a = payArea;
				gep.b = offRef;
				gep.ty = u8PtrTy;  // *u8 -> byte stride
				JirRef fieldPtr = emit(gctx, gep);
				emitDropInPlace(gctx, fieldPtr, pt);
			}
			off += s;
		}
		emitBr(gctx, contB);
		gctx.currentBlock = contB;
	}
}

// Deep-copy the value at `srcPtr` into `destPtr` (see CLONE_PLAN.md).
// Three tiers, mirroring the drop tiers:
//   - plain data (no drops anywhere): whole-value load+store.
//   - type with its OWN `cfn clone`: call it, store the result.
//   - struct with drop-bearing fields but no own cfn drop: recurse per
//     field; fixed-size arrays clone element-wise.
// A type with its own `cfn drop` but NO `cfn clone` is an error — the
// compiler can clone structure but cannot guess how to duplicate a
// resource (dup the fd? malloc+memcpy?). Same conclusion as Rust
// (derive(Clone) is useless past owned resources) and Hylo (`Copyable`
// conformance with explicit .copy()).
static void emitCloneInto(AstGenCtx &gctx, JirRef srcPtr, JirRef destPtr,
                          TypeIdx ty) {
	ty = resolveGenericIfAny(gctx.ctx, ty);
	{
		const TypeKey &k0 = gctx.ctx.getTypePool().get(ty);
		if (k0.kind == TypeKind::ArrayExpr) {
			ty = gctx.ctx.resolveArrayExpr(ty);
		}
	}

	// Plain data: bitwise copy is a true value copy.
	if (!typeNeedsDrop(gctx.ctx, ty)) {
		JirInst ld{};
		ld.tag = JirTag::Load;
		ld.a = srcPtr;
		ld.ty = ty;
		JirRef v = emit(gctx, ld);
		JirInst st{};
		st.tag = JirTag::Store;
		st.a = destPtr;
		st.b = v;
		emit(gctx, st);
		return;
	}

	const TypeKey &k = gctx.ctx.getTypePool().get(ty);

	// Fixed-size array: clone element-wise (same loop shape as
	// emitArrayElementDrops, with paired src/dest element pointers).
	if (k.kind == TypeKind::Array) {
		TypeIdx elemTy = static_cast<TypeIdx>(k.a);
		TypeIdx elemPtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, elemTy, 0});
		for (uint32_t i = 0; i < k.b; i++) {
			JirInst idxI{};
			idxI.tag = JirTag::Int;
			idxI.a = i;
			idxI.ty = BuiltinType::U64;
			JirRef idxRef = emit(gctx, idxI);
			JirInst sa{};
			sa.tag = JirTag::IndexAddr;
			sa.a = srcPtr;
			sa.b = idxRef;
			sa.ty = elemPtrTy;
			JirRef se = emit(gctx, sa);
			JirInst da{};
			da.tag = JirTag::IndexAddr;
			da.a = destPtr;
			da.b = idxRef;
			da.ty = elemPtrTy;
			JirRef de = emit(gctx, da);
			emitCloneInto(gctx, se, de, elemTy);
		}
		return;
	}

	// Enums: bitwise-copy the whole value (tag + payload bytes), then
	// re-clone the live variant's droppable payload fields over the
	// raw copy — tag-dispatched at the SAME byte offsets construction
	// and the drop glue use. Non-droppable payload variants are
	// already correct from the bitwise copy.
	if (const auto *einfo = gctx.ctx.lookupEnum(ty)) {
		JirInst ld{};
		ld.tag = JirTag::Load;
		ld.a = srcPtr;
		ld.ty = ty;
		JirRef whole = emit(gctx, ld);
		JirInst st{};
		st.tag = JirTag::Store;
		st.a = destPtr;
		st.b = whole;
		emit(gctx, st);
		if (!einfo->hasPayloadVariant) return;

		TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});
		JirInst tagFA{};
		tagFA.tag = JirTag::FieldAddr;
		tagFA.a = srcPtr;
		tagFA.b = 0;
		tagFA.ty = u8PtrTy;
		JirRef tagPtr = emit(gctx, tagFA);
		JirInst tagLd{};
		tagLd.tag = JirTag::Load;
		tagLd.a = tagPtr;
		tagLd.ty = BuiltinType::U8;
		JirRef tagVal = emit(gctx, tagLd);

		for (const auto &v : einfo->variants) {
			bool anyDrop = false;
			for (TypeIdx pt : v.payloadTypes) {
				if (typeNeedsDrop(gctx.ctx, pt)) {
					anyDrop = true;
					break;
				}
			}
			if (!anyDrop) continue;

			JirBlockRef cloneB = gctx.jfn.pushBlock("eclone");
			JirBlockRef contB = gctx.jfn.pushBlock("eclonecont");
			JirInst disc{};
			disc.tag = JirTag::Int;
			disc.a = static_cast<uint32_t>(v.discriminant);
			disc.ty = BuiltinType::U8;
			JirRef discRef = emit(gctx, disc);
			JirInst cmp{};
			cmp.tag = JirTag::ICmpEq;
			cmp.a = tagVal;
			cmp.b = discRef;
			cmp.ty = BuiltinType::Bool;
			JirRef cmpRef = emit(gctx, cmp);
			emitCondBr(gctx, cmpRef, cloneB, contB);

			gctx.currentBlock = cloneB;
			JirInst sPayFA{};
			sPayFA.tag = JirTag::FieldAddr;
			sPayFA.a = srcPtr;
			sPayFA.b = 1;
			sPayFA.ty = u8PtrTy;
			JirRef sPay = emit(gctx, sPayFA);
			JirInst dPayFA{};
			dPayFA.tag = JirTag::FieldAddr;
			dPayFA.a = destPtr;
			dPayFA.b = 1;
			dPayFA.ty = u8PtrTy;
			JirRef dPay = emit(gctx, dPayFA);
			uint64_t off = 0;
			for (TypeIdx pt : v.payloadTypes) {
				uint64_t sz = gctx.ctx.typeSize(pt);
				uint64_t al = gctx.ctx.typeAlign(pt);
				off = (off + al - 1) / al * al;
				if (typeNeedsDrop(gctx.ctx, pt)) {
					JirInst offC{};
					offC.tag = JirTag::Int;
					offC.a = static_cast<uint32_t>(off);
					offC.ty = BuiltinType::U64;
					JirRef offRef = emit(gctx, offC);
					JirInst sGep{};
					sGep.tag = JirTag::IndexAddr;
					sGep.a = sPay;
					sGep.b = offRef;
					sGep.ty = u8PtrTy;
					JirRef sF = emit(gctx, sGep);
					JirInst dGep{};
					dGep.tag = JirTag::IndexAddr;
					dGep.a = dPay;
					dGep.b = offRef;
					dGep.ty = u8PtrTy;
					JirRef dF = emit(gctx, dGep);
					emitCloneInto(gctx, sF, dF, pt);
				}
				off += sz;
			}
			emitBr(gctx, contB);
			gctx.currentBlock = contB;
		}
		return;
	}

	const auto *sinfo = gctx.ctx.lookupStruct(ty);
	if (sinfo == nullptr) {
		failHere(gctx, "cannot clone this type — no structural clone "
		               "is available for it");
	}

	// Tier 3: the type's own cfn clone — in-struct form registers as
	// "Name.clone"; the top-level `cfn clone(self: T) T` form lives in
	// the CloneRegistry (mirror of how drop is found).
	const FunctionAST *cloneFn =
	    gctx.ctx.getFunctionAST(sinfo->name + ".clone");
	if (cloneFn == nullptr) {
		if (const auto *cr = gctx.ctx.getCloneRegistry()) {
			auto it = cr->find(sinfo->name);
			if (it != cr->end()) cloneFn = it->second;
		}
	}
	if (cloneFn != nullptr) {
		std::vector<JirRef> args;
		args.push_back(srcPtr);  // let-mode self on a byref aggregate
		JirRef result = emitCall(gctx, cloneFn, args);
		JirInst st{};
		st.tag = JirTag::Store;
		st.a = destPtr;
		st.b = result;
		emit(gctx, st);
		return;
	}

	// Error tier: owns a resource (its own cfn drop), no clone recipe.
	if (!lookupDropFnLLVMName(gctx.ctx, ty).empty()) {
		failHere(gctx, "`" + sinfo->name +
		                   "` owns resources (it has `cfn drop`); define "
		                   "`cfn clone(self: Self) Self` to make it "
		                   "cloneable");
	}

	// Tier 2: field-wise structural clone.
	for (size_t i = 0; i < sinfo->fields.size(); i++) {
		TypeIdx fieldTy =
		    resolveGenericIfAny(gctx.ctx, sinfo->fields[i].second);
		TypeIdx fieldPtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
		JirInst sfa{};
		sfa.tag = JirTag::FieldAddr;
		sfa.a = srcPtr;
		sfa.b = static_cast<uint32_t>(i);
		sfa.ty = fieldPtrTy;
		JirRef sf = emit(gctx, sfa);
		JirInst dfa{};
		dfa.tag = JirTag::FieldAddr;
		dfa.a = destPtr;
		dfa.b = static_cast<uint32_t>(i);
		dfa.ty = fieldPtrTy;
		JirRef df = emit(gctx, dfa);
		emitCloneInto(gctx, sf, df, fieldTy);
	}
}

// Built-in `recv.clone()` lowering for receivers WITHOUT a user cfn
// clone. Returns kNoJirRef when a user clone method exists (the caller
// proceeds with ordinary method dispatch — tier 3). Otherwise lowers
// tier 1 (plain data: the value itself) or tier 2 (structural glue into
// a fresh slot), or rejects (owns-resources / fenced types) inside
// emitCloneInto.
static JirRef tryLowerBuiltinClone(AstGenCtx &gctx, JirRef recvLvaluePtr,
                                   JirRef recvVal, TypeIdx recvTy) {
	recvTy = resolveGenericIfAny(gctx.ctx, recvTy);
	{
		const TypeKey &k0 = gctx.ctx.getTypePool().get(recvTy);
		if (k0.kind == TypeKind::ArrayExpr) {
			recvTy = gctx.ctx.resolveArrayExpr(recvTy);
		}
	}

	// An in-struct user cfn clone wins — ordinary dispatch handles it.
	// (Top-level cfn clones aren't dispatchable as "Name.clone" method
	// calls, so those route through emitCloneInto's registry lookup.)
	if (const auto *sinfo = gctx.ctx.lookupStruct(recvTy)) {
		if (gctx.ctx.getFunctionAST(sinfo->name + ".clone") != nullptr) {
			return kNoJirRef;
		}
		// Top-level form: lower as glue (emitCloneInto calls it).
		if (const auto *cr = gctx.ctx.getCloneRegistry()) {
			if (cr->find(sinfo->name) != cr->end()) {
				// fall through to the glue path below
			}
		}
	}

	// Tier 1: plain data — clone IS the value.
	if (!typeNeedsDrop(gctx.ctx, recvTy)) {
		if (recvVal != kNoJirRef) return recvVal;
		JirInst ld{};
		ld.tag = JirTag::Load;
		ld.a = recvLvaluePtr;
		ld.ty = recvTy;
		return emit(gctx, ld);
	}

	// Tier 2 / error tiers: glue into a fresh slot. Need a stable
	// source pointer; spill a value-form receiver first.
	JirRef srcPtr = recvLvaluePtr;
	if (srcPtr == kNoJirRef) {
		JirInst sp{};
		sp.tag = JirTag::Alloca;
		sp.ty = recvTy;
		srcPtr = emitAllocaHoisted(gctx, sp);
		JirInst st{};
		st.tag = JirTag::Store;
		st.a = srcPtr;
		st.b = recvVal;
		emit(gctx, st);
	}
	JirInst da{};
	da.tag = JirTag::Alloca;
	da.ty = recvTy;
	JirRef destPtr = emitAllocaHoisted(gctx, da);
	emitCloneInto(gctx, srcPtr, destPtr, recvTy);
	return destPtr;
}

static void emitDropInPlace(AstGenCtx &gctx, JirRef ptrRef, TypeIdx pointeeTy) {
	// Resolve a generic-call type to its concrete instantiation so the
	// shape check + cfn-drop lookup operate on the right TypeIdx
	// (Vec(u32) → Vec__u32 etc.). FieldAddr's lowering also needs the
	// concrete struct in `pointeeTy` so the GEP indices match the LLVM
	// layout.
	pointeeTy = resolveGenericIfAny(gctx.ctx, pointeeTy);
	// Fixed-size arrays own their elements: drop each one that needs
	// it. Arrays have no cfn drop and no fields of their own, so this
	// is the whole story for them.
	{
		const TypeKey &pk = gctx.ctx.getTypePool().get(pointeeTy);
		TypeIdx arrTy = pointeeTy;
		if (pk.kind == TypeKind::ArrayExpr) {
			arrTy = gctx.ctx.resolveArrayExpr(pointeeTy);
		}
		const TypeKey &rk = gctx.ctx.getTypePool().get(arrTy);
		if (rk.kind == TypeKind::Array) {
			if (typeNeedsDrop(gctx.ctx, static_cast<TypeIdx>(rk.a))) {
				emitArrayElementDrops(gctx, ptrRef, arrTy);
			}
			return;
		}
	}
	// Contiguous owning containers (a `cfn len` + a single data pointer)
	// synthesize their element destructors here, BEFORE the container's own
	// drop frees the backing — elements must die while their storage lives.
	emitContainerElementDrops(gctx, ptrRef, pointeeTy);
	// If the type has a custom drop fn (registered cfn drop OR an in-
	// struct `fn drop` picked up via `instantiatedDrops_`), emit the
	// call first. Then, regardless of whether the call was emitted,
	// walk droppable fields — matches Rust's `drop_in_place::<T>`
	// which always runs Drop::drop (if any) followed by field drops.
	std::string dropLLVMName = lookupDropFnLLVMName(gctx.ctx, pointeeTy);
	if (!dropLLVMName.empty()) {
		StringIdx symId = gctx.ctx.getStringPool().intern(dropLLVMName);
		JirInst drop{};
		drop.tag = JirTag::DropBinding;
		drop.a = ptrRef;
		drop.b = symId;
		drop.ty = kNoType;
		emit(gctx, drop);
	}
	emitFieldDrops(gctx, ptrRef, pointeeTy);
	// Payloaded enums own the live variant's payload: tag-dispatched
	// payload drops run after any user drop (mirroring the
	// Drop::drop-then-fields order structs get).
	if (const auto *einfo = gctx.ctx.lookupEnum(pointeeTy)) {
		if (einfo->hasPayloadVariant) {
			emitEnumPayloadDrops(gctx, ptrRef, einfo);
		}
	}
}

// AstGen for `AtCall` — comptime intrinsic (`@sizeOf(T)` / `@alignOf(T)`).
// Resolves to a constant at astgen time; the JIR never sees a call.
static JirRef astgenAtCall(AstGenCtx &gctx, const AstNode &n) {
	const std::string &name =
	    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	// Expression-arg intrinsics (`@dropInPlace(ptr)`, `@emit*(...)`) carry
	// `flags & 1` and use n.rhs as an ExtraIdx slice; the legacy type-arg
	// path uses n.rhs as a TypeIdx. Dispatch here before reading n.rhs.
	if ((n.flags & 1) != 0 && name == "dropInPlace") {
		const NodeStore &ns = gctx.ctx.getNodeStore();
		ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
		uint32_t argCount = ns.getExtra(argsExtra);
		if (argCount != 1) {
			failHere(gctx, "@dropInPlace takes exactly one pointer argument");
		}
		NodeIdx ptrIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 1));
		JirRef ptrRef = astgenExpr(gctx, ptrIdx, kNoType);
		TypeIdx ptrTy = gctx.jfn.getInst(ptrRef).ty;
		const TypeKey &pk = gctx.ctx.getTypePool().get(ptrTy);
		if (pk.kind != TypeKind::PtrSingle && pk.kind != TypeKind::PtrMany) {
			failHere(gctx, "@dropInPlace argument must be a pointer");
		}
		TypeIdx pointeeTy = static_cast<TypeIdx>(pk.a);
		emitDropInPlace(gctx, ptrRef, pointeeTy);
		return kNoJirRef;
	}
	TypeIdx tyArg = static_cast<TypeIdx>(n.rhs);
	if (name == "isDarwin" || name == "isLinux" || name == "isWindows" ||
	    name == "isUnix") {
		// Target-OS predicates, resolved at astgen time to a `Bool`
		// constant. Because the result is a `JirTag::Bool` literal,
		// `emitCondBr` folds `if (@isLinux())` to an unconditional branch
		// and jirDefineBody's reachability pass drops the dead arm — so a
		// macOS-only `_NSGetArgv` call never reaches codegen (or the
		// linker) on Linux. Sourced from the host target, which is the
		// only target today; switch to the selected target once
		// cross-compilation lands.
		// `isDarwin` follows Zig (std.Target.Os.Tag.isDarwin = the Apple
		// family). Jam's OS enum only has MacOS today, so it reduces to
		// that and widens automatically if iOS/tvOS/etc. are ever added.
		// `isUnix` is "not Windows" for the current enum.
		jam::OS os = jam::Target::getHostTarget().os;
		bool v = (name == "isDarwin")  ? (os == jam::OS::MacOS)
		         : (name == "isLinux") ? (os == jam::OS::Linux)
		         : (name == "isWindows")
		             ? (os == jam::OS::Windows)
		             : (os == jam::OS::MacOS || os == jam::OS::Linux ||
		                os == jam::OS::FreeBSD);  // isUnix
		JirInst inst{};
		inst.tag = JirTag::Bool;
		inst.a = v ? 1u : 0u;
		inst.ty = BuiltinType::Bool;
		return emit(gctx, inst);
	}
	if (name == "os") {
		// `@os()` → the OS name as a `[]u8`, mirroring Zig's
		// `@tagName(builtin.os.tag)` (e.g. "macos", "linux"). This is the
		// display form: unlike the boolean predicates it does NOT fold a
		// branch (string `==` isn't comptime-evaluated), so keep using
		// `@isDarwin()` & co. for conditional compilation / dead-code
		// elimination. Emits the same JIR a string literal would.
		jam::OS os = jam::Target::getHostTarget().os;
		const char *osName = (os == jam::OS::MacOS)     ? "macos"
		                     : (os == jam::OS::Linux)   ? "linux"
		                     : (os == jam::OS::Windows) ? "windows"
		                     : (os == jam::OS::FreeBSD) ? "freebsd"
		                                                : "unknown";
		StringIdx s = gctx.ctx.getStringPool().intern(osName);
		TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
		JirInst inst{};
		inst.tag = JirTag::Str;
		inst.a = s;
		inst.ty = sliceTy;
		return emit(gctx, inst);
	}
	if (name == "sizeOf") {
		uint64_t bytes = gctx.ctx.typeSize(tyArg);
		JirInst inst{};
		inst.tag = JirTag::Int;
		inst.a = static_cast<uint32_t>(bytes & 0xFFFFFFFFu);
		inst.b = static_cast<uint32_t>(bytes >> 32);
		inst.ty = BuiltinType::U64;
		return emit(gctx, inst);
	}
	if (name == "alignOf") {
		uint64_t a = gctx.ctx.typeAlign(tyArg);
		JirInst inst{};
		inst.tag = JirTag::Int;
		inst.a = static_cast<uint32_t>(a);
		inst.ty = BuiltinType::U8;
		return emit(gctx, inst);
	}
	return recoverHere(gctx, "unknown intrinsic `@" + name + "`", kNoType);
}

// AstGen for `Call`. Materializes each arg with the callee's parameter
// type as the expected hint so literal widths propagate; then emits a
// JirTag::Call whose `a` carries the callee StringIdx and `b` carries
// an extra-pool slice [argCount, arg0_ref, arg1_ref, ...]. The result
// type comes from the resolved FunctionAST::ReturnType.
// Lower an `assert(actual, expected)` builtin call as JIR: emit an
// ICmp followed by a CondBr to a fail block (printf + exit) and a
// pass block (continuation). Mirrors the legacy `genAssertCall`
// shape but produces only JIR — jir_codegen lowers it normally.
//
// printf and exit are registered as extern fn references the first
// time we see an assert; subsequent asserts reuse them.
static JirRef astgenAssertCall(AstGenCtx &gctx, const AstNode &n) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = ns.getExtra(argsExtra);
	if (argCount != 2) {
		failHere(gctx, "astgen: assert expects exactly 2 arguments");
	}
	NodeIdx actualIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 1));
	NodeIdx expectedIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 2));

	JirRef actualRef = astgenExpr(gctx, actualIdx, kNoType);
	TypeIdx actualTy = gctx.jfn.getInst(actualRef).ty;
	JirRef expectedRef = astgenExpr(gctx, expectedIdx, actualTy);
	// If widths differ after expected-hint lowering, widen the smaller.
	TypeIdx expectedTy = gctx.jfn.getInst(expectedRef).ty;
	if (actualTy != expectedTy) {
		const TypeKey &ak = gctx.ctx.getTypePool().get(actualTy);
		const TypeKey &ek = gctx.ctx.getTypePool().get(expectedTy);
		if (ak.kind == TypeKind::Int && ek.kind == TypeKind::Int) {
			if (ak.a > ek.a) {
				JirInst zext{};
				zext.tag = JirTag::ZExt;
				zext.a = expectedRef;
				zext.ty = actualTy;
				expectedRef = emit(gctx, zext);
			} else {
				JirInst zext{};
				zext.tag = JirTag::ZExt;
				zext.a = actualRef;
				zext.ty = expectedTy;
				actualRef = emit(gctx, zext);
				actualTy = expectedTy;
			}
		}
	}

	// Float operands need an ordered float compare, not an integer one.
	// `assert(f32, f32)` would otherwise build an ICmp on float values —
	// invalid IR (it aborts an assertions-enabled LLVM). OEQ matches the
	// `==` operator (see astgenBinaryOp) and IEEE semantics: NaN equals
	// nothing, so `assert(nan, nan)` correctly fails.
	const TypeKey &cmpTy = gctx.ctx.getTypePool().get(actualTy);
	JirInst cmp{};
	cmp.tag =
	    (cmpTy.kind == TypeKind::Float) ? JirTag::FCmpOeq : JirTag::ICmpEq;
	cmp.a = actualRef;
	cmp.b = expectedRef;
	cmp.ty = BuiltinType::Bool;
	JirRef cmpRef = emit(gctx, cmp);

	JirBlockRef failB = gctx.jfn.pushBlock("assert.fail");
	JirBlockRef passB = gctx.jfn.pushBlock("assert.pass");
	emitCondBr(gctx, cmpRef, passB, failB);

	// Fail block: call printf("Assertion failed\n") then exit(1).
	gctx.currentBlock = failB;

	StringIdx printfNameId = gctx.ctx.getStringPool().intern("printf");
	const FunctionAST *printfAST = gctx.ctx.getFunctionAST("printf");
	if (printfAST == nullptr) {
		// Register a fake FunctionAST for printf so emitCall can look
		// up parameter types. We mark it varArgs so the prototype
		// matches the C signature.
		auto fakePrintf = std::make_unique<FunctionAST>(
		    "printf",
		    std::vector<Param>{Param{"fmt", BuiltinType::U64, ParamMode::Let}},
		    BuiltinType::I32, std::vector<NodeIdx>{},
		    /*isExtern=*/true, /*isExport=*/false, /*isPub=*/false,
		    /*isTest=*/false, /*isVarArgs=*/true);
		printfAST = fakePrintf.get();
		gctx.ctx.registerFunctionAST("printf", fakePrintf.release());
		// Also declare the LLVM prototype so the call resolves.
		JamTypeRef i8PtrType = JamLLVMPointerType(gctx.ctx.getInt8Type(), 0);
		JamTypeRef paramTypes[1] = {i8PtrType};
		JamTypeRef ft = JamLLVMFunctionType(gctx.ctx.getInt32Type(), paramTypes,
		                                    1, /*isVarArgs=*/true);
		JamFunctionRef pf =
		    JamLLVMAddFunction(gctx.ctx.getModule(), "printf", ft);
		JamLLVMApplyDefaultFnAttrs(pf, /*isExtern=*/true);
	}

	// Build the format-string slice "Assertion failed\n" as a JIR Str.
	StringIdx msgId = gctx.ctx.getStringPool().intern("Assertion failed\n");
	TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
	JirInst strInst{};
	strInst.tag = JirTag::Str;
	strInst.a = msgId;
	strInst.ty = sliceTy;
	JirRef strRef = emit(gctx, strInst);
	// Extract the slice's pointer field (field 0) for printf's first arg.
	TypeIdx i8PtrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrMany, 0, 0, BuiltinType::U8, 0});
	JirInst ev{};
	ev.tag = JirTag::ExtractValue;
	ev.a = strRef;
	ev.b = 0;
	ev.ty = i8PtrTy;
	JirRef strPtrRef = emit(gctx, ev);

	// printf(strPtr)
	std::vector<uint32_t> pp = {1, strPtrRef};
	JirExtraIdx pe = gctx.jfn.pushExtra(pp.data(), pp.size());
	JirInst pcall{};
	pcall.tag = JirTag::Call;
	pcall.a = printfNameId;
	pcall.b = pe;
	pcall.ty = BuiltinType::I32;
	emit(gctx, pcall);

	// Now exit(1).
	const FunctionAST *exitAST = gctx.ctx.getFunctionAST("exit");
	if (exitAST == nullptr) {
		auto fakeExit = std::make_unique<FunctionAST>(
		    "exit",
		    std::vector<Param>{Param{"code", BuiltinType::I32, ParamMode::Let}},
		    kNoType, std::vector<NodeIdx>{},
		    /*isExtern=*/true, /*isExport=*/false, /*isPub=*/false,
		    /*isTest=*/false, /*isVarArgs=*/false);
		exitAST = fakeExit.get();
		gctx.ctx.registerFunctionAST("exit", fakeExit.release());
		JamTypeRef exitParamTypes[1] = {gctx.ctx.getInt32Type()};
		JamTypeRef et = JamLLVMFunctionType(gctx.ctx.getVoidType(),
		                                    exitParamTypes, 1, false);
		JamFunctionRef ef =
		    JamLLVMAddFunction(gctx.ctx.getModule(), "exit", et);
		JamLLVMApplyDefaultFnAttrs(ef, /*isExtern=*/true);
	}
	JirInst exitCode{};
	exitCode.tag = JirTag::Int;
	exitCode.a = 1;
	exitCode.ty = BuiltinType::I32;
	JirRef exitCodeRef = emit(gctx, exitCode);
	std::vector<uint32_t> ep = {1, exitCodeRef};
	JirExtraIdx eextra = gctx.jfn.pushExtra(ep.data(), ep.size());
	JirInst ecall{};
	ecall.tag = JirTag::Call;
	ecall.a = gctx.ctx.getStringPool().intern("exit");
	ecall.b = eextra;
	ecall.ty = kNoType;
	emit(gctx, ecall);
	JirInst unreach{};
	unreach.tag = JirTag::Unreachable;
	emit(gctx, unreach);

	gctx.currentBlock = passB;
	return kNoJirRef;
}

// Emit a Call to `fn` with the given arg refs. Handles the JIR
// instruction packing and result-type wiring.
// Translate a source-level identifier through the current generic
// substitution context. `Self` and any type-parameter name (e.g.
// `T` inside `fn Holder(T: type) ...`) resolve to the instantiated
// type's canonical struct/enum name. Plain identifiers pass through
// unchanged.
//
// Despite taking a *string*, the substitution happens at the
// TypeIdx level: `lookupCurrentSubst` returns the bound TypeIdx,
// and `lookupStruct` / `lookupEnum` internally chase `GenericCall`
// indirections via `resolveGenericCall`. We bottom-out at the
// canonical Named TypeIdx's name, which is what the rest of
// astgen wants for `<T>.method` dispatch.
//
// This is the *single* place in astgen where Self/type-param
// substitution happens. Every call-dispatch path goes through
// here. Don't add a second one.
static std::string resolvePrefix(JamCodegenContext &ctx,
                                 const std::string &prefix) {
	TypeIdx ty = ctx.lookupCurrentSubst(prefix);
	if (ty == kNoType) return prefix;
	if (const auto *sinfo = ctx.lookupStruct(ty)) return sinfo->name;
	if (const auto *einfo = ctx.lookupEnum(ty)) return einfo->name;
	return prefix;
}

// Lower one call argument honouring the callee's parameter mode. For
// `mut` / `move` params, pass the argument's address (auto-`AddrOf`
// on lvalues, pass-through `&expr` literally, spill rvalues to a
// temp alloca). For `let` / `const`, pass the value with the
// param's TypeIdx as the expected hint so literals settle at the
// right width.
static JirRef lowerArgInner(AstGenCtx &gctx, NodeIdx argIdx, const Param &p) {
	jam::abi::ParamABI pabi = jam::abi::classifyParam(p.Mode, p.Type, gctx.ctx);
	if (pabi.kind != jam::abi::ParamABI::Kind::ByPointer) {
		JirRef v = astgenExpr(gctx, argIdx, p.Type);
		// A runtime slice does not implicitly decay to a pointer parameter;
		// passing a {ptr,len} aggregate where the callee expects a bare
		// pointer silently corrupts the ABI (e.g. desyncs C varargs).
		// Require an explicit `.ptr`, matching Zig (a `[]T` slice has no
		// static NUL guarantee, so it never coerces to `[*]T`). String
		// literals decay in astgenStringLit, so they arrive here already
		// typed as a pointer and skip this check.
		const TypeKey &vk = gctx.ctx.getTypePool().get(gctx.jfn.getInst(v).ty);
		const TypeKey &pk = gctx.ctx.getTypePool().get(p.Type);
		if (vk.kind == TypeKind::Slice &&
		    (pk.kind == TypeKind::PtrMany || pk.kind == TypeKind::PtrSingle)) {
			failHere(gctx,
			         "cannot pass a slice to pointer parameter `" + p.Name +
			             "`; use `.ptr` to pass the slice's data pointer");
		}
		return v;
	}
	// ByPointer: feed an address. Lvalueable arg -> hand the existing
	// storage ptr; non-lvalue rvalue -> spill to a fresh alloca. All
	// four param modes share this shape — mut/move want the caller's
	// storage so writes are observed, let on large aggregates wants
	// it because the value is too big to pass in registers. Routing
	// let/move-by-ptr through the lvalue path saves the otherwise-
	// dead value-load + spill at every call site.
	const AstNode &argNode = gctx.ctx.getNodeStore().get(argIdx);
	const NodeStore &ns = gctx.ctx.getNodeStore();
	TypeIdx leafTy = kNoType;
	// `Color.Red` and friends look syntactically like a MemberAccess
	// but are enum-variant constructors that astgenLvalue can't
	// resolve — its `Color` lookup falls through every lvalue path.
	// Detect the shape upfront and route through the value+spill
	// fallback below. Same for any MemberAccess on a base Variable
	// whose name resolves to an enum/struct/module, not a local.
	auto memberAccessOnNonLvalue = [&](const AstNode &n) -> bool {
		if (n.tag != AstTag::MemberAccess) return false;
		NodeIdx baseIdx = static_cast<NodeIdx>(n.lhs);
		const AstNode &baseNode = ns.get(baseIdx);
		if (baseNode.tag != AstTag::Variable) return false;
		const std::string &name =
		    gctx.ctx.getStringPool().get(static_cast<StringIdx>(baseNode.lhs));
		if (gctx.locals.find(name) != gctx.locals.end()) return false;
		// Not a local — likely an enum / module / type name.
		// Treat as non-lvalueable so the spill path takes over.
		return gctx.ctx.getEnum(name) != nullptr ||
		       gctx.ctx.getStruct(name) != nullptr ||
		       gctx.ctx.getImportHandle(name) != nullptr;
	};
	switch (argNode.tag) {
	case AstTag::Variable:
	case AstTag::Index:
	case AstTag::Deref:
		return astgenLvalue(gctx, argIdx, leafTy);
	case AstTag::MemberAccess:
		if (!memberAccessOnNonLvalue(argNode)) {
			return astgenLvalue(gctx, argIdx, leafTy);
		}
		break;  // fall through to spill path
	case AstTag::AddressOf:
		return astgenExpr(gctx, argIdx, kNoType);
	default:
		break;
	}
	// Spill: hold an rvalue in a fresh stack slot so the callee can
	// receive a pointer. Try the place path first — byref producers
	// (StructLit / ArrayLit / sret Call) write directly into the
	// alloca, skipping the temp + memcpy two-hop. Fall back to value
	// compile + Store for byval rvalues.
	JirInst alloca{};
	alloca.tag = JirTag::Alloca;
	alloca.ty = p.Type;
	JirRef ptr = emitAllocaHoisted(gctx, alloca);
	if (astgenExprIntoPtr(gctx, argIdx, p.Type, ptr)) { return ptr; }
	JirRef val = astgenExpr(gctx, argIdx, p.Type);
	JirInst store{};
	store.tag = JirTag::Store;
	store.a = ptr;
	store.b = val;
	emit(gctx, store);
	return ptr;
}

// Ownership is tracked per WHOLE binding. Extracting a drop-bearing
// value out of a pure field path (`h.c`, `o.inner.c`) duplicates
// ownership: the extracted value and the aggregate's drop glue would
// both drop the same payload (double-free), or — for a moved-out field
// — the parent's glue re-drops what the callee already consumed.
// rustc models this with per-field move paths (rustc_mir_dataflow/
// move_paths); jam keeps whole-binding granularity and REJECTS,
// matching its reject-don't-flag philosophy.
//
// Scope: pure MemberAccess chains rooted at a local binding. Paths
// that index through raw pointers (`self.ptr[i]`) are the pointer
// world and stay unchecked; whole-array element extraction is covered
// by the (separate, pending) array-ownership work.
static bool isDropBearingFieldExtract(AstGenCtx &gctx, NodeIdx exprIdx,
                                      TypeIdx resultTy) {
	if (resultTy == kNoType) return false;
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &top = ns.get(exprIdx);
	if (top.tag != AstTag::MemberAccess) return false;
	NodeIdx cur = exprIdx;
	while (cur != kNoNode) {
		const AstNode &n = ns.get(cur);
		if (n.tag == AstTag::MemberAccess) {
			cur = static_cast<NodeIdx>(n.lhs);
			continue;
		}
		if (n.tag == AstTag::Variable) {
			const std::string &root =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
			// Only locals/params — `Color.Red` and module paths also
			// parse as MemberAccess on a Variable root.
			if (gctx.locals.find(root) == gctx.locals.end()) return false;
			return typeNeedsDrop(gctx.ctx, resultTy);
		}
		return false;  // Index / Deref / call in the path: pointer world
	}
	return false;
}

static void rejectDropBearingFieldExtract(AstGenCtx &gctx, NodeIdx exprIdx,
                                          TypeIdx resultTy, const char *verb) {
	if (!isDropBearingFieldExtract(gctx, exprIdx, resultTy)) return;
	failNode(gctx, exprIdx,
	         std::string("cannot ") + verb +
	             " a drop-bearing field out of its aggregate — ownership "
	             "is tracked per whole binding, so the field and the "
	             "aggregate's drop would both run; clone the field out "
	             "(`.clone()`) or move the whole value");
}

// Callsites are sigil-free for parameter modes: `mut` / `move` access
// is declared at the signature, and the argument is a plain expression
// (`scale(p, 2.0)`, `consume(c)`). There is no reference type in jam —
// `&` is address-of, producing a `*mut T` / `*const T` pointer VALUE,
// and is only meaningful where the parameter's type is a pointer
// (FFI, sink out-params). An `&` on a mode argument is rejected here
// so the pointer world and the mode world stay disjoint.
static void rejectAddrOfOnModeArg(AstGenCtx &gctx, NodeIdx argIdx,
                                  const Param &p) {
	const AstNode &argNode = gctx.ctx.getNodeStore().get(argIdx);
	if (argNode.tag != AstTag::AddressOf) return;
	const TypeKey &k = gctx.ctx.getTypePool().get(p.Type);
	if (k.kind == TypeKind::PtrSingle || k.kind == TypeKind::PtrMany) return;
	failNode(gctx, argIdx,
	         "`&` makes a pointer, but parameter `" + p.Name +
	             "` takes its argument by mode — pass it plainly (the "
	             "signature's `mut`/`move` already grants access)");
}

static JirRef lowerArg(AstGenCtx &gctx, NodeIdx argIdx, const Param &p) {
	rejectAddrOfOnModeArg(gctx, argIdx, p);
	if (p.Mode == ParamMode::Move) {
		rejectDropBearingFieldExtract(gctx, argIdx, p.Type, "move");
	}
	JirRef r = lowerArgInner(gctx, argIdx, p);
	// A `move`-mode argument transfers ownership into the callee: the
	// value's new owner runs the drop. Suppress the caller-side
	// scope-exit drop when the arg is a bare tracked drop-bearing local
	// — the same mechanism struct-literal field capture uses.
	// init_analysis guarantees the move is unconditional relative to
	// the binding's declaration and that the binding is never
	// re-initialized, so the permanent removal is sound.
	if (p.Mode == ParamMode::Move) { consumeMovedVariable(gctx, argIdx); }
	return r;
}

static JirRef emitCall(AstGenCtx &gctx, const FunctionAST *fn,
                       const std::vector<JirRef> &argRefs, JirRef destPtr) {
	std::string mangled = mangledFunctionName(*fn, gctx.ctx.getTypePool(),
	                                          gctx.ctx.getStringPool());
	StringIdx calleeId = gctx.ctx.getStringPool().intern(mangled);
	// Unified Call shape: sret-returning callees get their result slot
	// as a leading arg, the same way LLVM expects it. astgen owns the
	// slot — either the caller's destPtr (when known) or a fresh
	// alloca — so jir_codegen no longer allocates anything at call
	// time. The Call's JirRef value is the slot pointer when sret;
	// downstream `isByRef`-aware code paths treat it as the byref
	// result. When destPtr was supplied, the JirRef is kNoJirRef so
	// the caller can detect that the result was placed in-line.
	bool sretCallee = fn->ReturnType != kNoType &&
	                  jam::abi::classifyReturn(fn->ReturnType, gctx.ctx).kind ==
	                      jam::abi::ReturnABI::Kind::Indirect;
	JirRef sretSlot = kNoJirRef;
	std::vector<JirRef> allArgs;
	allArgs.reserve(argRefs.size() + (sretCallee ? 1u : 0u));
	if (sretCallee) {
		if (destPtr != kNoJirRef) {
			sretSlot = destPtr;
		} else {
			JirInst a{};
			a.tag = JirTag::Alloca;
			a.ty = resolveDeferredArrayTy(gctx, fn->ReturnType);
			sretSlot = emitAllocaHoisted(gctx, a);
		}
		allArgs.push_back(sretSlot);
	}
	for (JirRef r : argRefs) allArgs.push_back(r);
	std::vector<uint32_t> packed;
	packed.reserve(1 + allArgs.size());
	packed.push_back(static_cast<uint32_t>(allArgs.size()));
	for (JirRef r : allArgs) packed.push_back(r);
	JirExtraIdx extra = gctx.jfn.pushExtra(packed.data(), packed.size());
	JirInst call{};
	call.tag = JirTag::Call;
	call.a = calleeId;
	call.b = extra;
	call.ty = resolveDeferredArrayTy(gctx, fn->ReturnType);
	JirRef callRef = emit(gctx, call);
	// Calling a `noreturn` function diverges. Terminate the current
	// block with Unreachable so downstream code is dead and the JIR
	// is well-formed (every reachable block ends in a terminator).
	if (fn->ReturnType == BuiltinType::NoReturn) {
		JirInst u{};
		u.tag = JirTag::Unreachable;
		emit(gctx, u);
	}
	if (sretCallee) {
		// Place-call: the value was written through destPtr, so the
		// caller shouldn't bind any JirRef.
		if (destPtr != kNoJirRef) { return kNoJirRef; }
		// Otherwise the slot ptr we just allocated *is* the call
		// result — codegen returns it from the Call JirRef so byref
		// consumers see a pointer to the freshly-written aggregate.
		return callRef;
	}
	return callRef;
}

// Comp-instantiation dispatcher. When the callee fn has any comp
// params, this evaluates them at compile time, builds a mangled clone
// name, and (on cache miss) registers + declares + defines a
// monomorphised version of the fn with the comp params dropped from
// the signature. Comp param refs in the body resolve to the bound
// values via `astgenVariable`'s comp-subst path. Returns the result
// of a regular Call to the clone, with only the runtime args lowered.
//
// Mirrors the shape of struct-method instantiation in
// `JamCodegenContext::instantiateStructExpr`: two-phase isn't needed
// here because comp-instantiated functions don't have mutual recursion
// to resolve (one fn, one body).
// Ensure POSIX `dprintf(int fd, const char *fmt, ...)` is declared in
// the module so @-emit intrinsics can dispatch through it. Idempotent
// across calls — the registry check skips after the first.
static void ensureDprintfForCfn(AstGenCtx &gctx) {
	if (gctx.ctx.getFunctionAST("dprintf") != nullptr) return;
	auto fake = std::make_unique<FunctionAST>(
	    "dprintf",
	    std::vector<Param>{
	        Param{"fd", BuiltinType::I32, ParamMode::Let},
	        Param{"fmt", BuiltinType::U64, ParamMode::Let},
	    },
	    BuiltinType::I32, std::vector<NodeIdx>{}, /*isExtern=*/true,
	    /*isExport=*/false, /*isPub=*/false, /*isTest=*/false,
	    /*isVarArgs=*/true);
	gctx.ctx.registerFunctionAST("dprintf", fake.release());
	JamTypeRef i8PtrType = JamLLVMPointerType(gctx.ctx.getInt8Type(), 0);
	JamTypeRef paramTypes[2] = {gctx.ctx.getInt32Type(), i8PtrType};
	JamTypeRef ft = JamLLVMFunctionType(gctx.ctx.getInt32Type(), paramTypes, 2,
	                                    /*isVarArgs=*/true);
	JamFunctionRef pf = JamLLVMAddFunction(gctx.ctx.getModule(), "dprintf", ft);
	JamLLVMApplyDefaultFnAttrs(pf, /*isExtern=*/true);
}

// Emit `dprintf(fd, fmtSpec, arg)` using only one extra arg after the
// format string. Used by the per-type interp lowering and by
// emitPutByte for the "%c" case.
static void emitDprintfSingleArg(AstGenCtx &gctx, JirRef fdRef,
                                 const char *fmtSpec, JirRef arg) {
	TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
	TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::PtrMany, 0, 0, BuiltinType::U8, 0});
	StringIdx fmtId = gctx.ctx.getStringPool().intern(fmtSpec);
	JirInst fs{};
	fs.tag = JirTag::Str;
	fs.a = fmtId;
	fs.ty = sliceTy;
	JirRef fmtSlice = emit(gctx, fs);
	JirInst fp{};
	fp.tag = JirTag::ExtractValue;
	fp.a = fmtSlice;
	fp.b = 0;
	fp.ty = u8PtrTy;
	JirRef fmtPtr = emit(gctx, fp);
	std::vector<uint32_t> pp = {3, static_cast<uint32_t>(fdRef),
	                            static_cast<uint32_t>(fmtPtr),
	                            static_cast<uint32_t>(arg)};
	JirExtraIdx pe = gctx.jfn.pushExtra(pp.data(), pp.size());
	JirInst c{};
	c.tag = JirTag::Call;
	c.a = gctx.ctx.getStringPool().intern("dprintf");
	c.b = pe;
	c.ty = BuiltinType::I32;
	emit(gctx, c);
}

// Materialise a JirRef holding `value` as an LLVM i32.
static JirRef literalI32(AstGenCtx &gctx, uint32_t value) {
	JirInst i{};
	i.tag = JirTag::Int;
	i.a = value;
	i.ty = BuiltinType::I32;
	return emit(gctx, i);
}

// Per-type runtime print dispatch used by `@emitPrintLocalByRange`.
// `val` is the loaded local-variable JirRef; `name` is the source
// identifier (only used for the diagnostic). The dispatcher widens /
// extracts as needed and emits one dprintf call per piece.
static void emitDprintfForValue(AstGenCtx &gctx, jam::Diagnostics &diags,
                                jam::SrcLoc loc, JirRef fdRef, JirRef val,
                                const std::string &name) {
	TypeIdx ty = gctx.jfn.getInst(val).ty;
	const TypeKey &k = gctx.ctx.getTypePool().get(ty);
	switch (k.kind) {
	case TypeKind::Int: {
		bool isSigned = k.b != 0;
		uint16_t width = static_cast<uint16_t>(k.a);
		JirRef wideRef = val;
		if (width < 64) {
			JirInst ext{};
			ext.tag = isSigned ? JirTag::SExt : JirTag::ZExt;
			ext.a = val;
			ext.ty = BuiltinType::I64;
			wideRef = emit(gctx, ext);
		}
		emitDprintfSingleArg(gctx, fdRef, isSigned ? "%lld" : "%llu", wideRef);
		return;
	}
	case TypeKind::Float: {
		uint16_t width = static_cast<uint16_t>(k.a);
		JirRef wideRef = val;
		if (width < 64) {
			JirInst ext{};
			ext.tag = JirTag::FPExt;
			ext.a = val;
			ext.ty = BuiltinType::F64;
			wideRef = emit(gctx, ext);
		}
		emitDprintfSingleArg(gctx, fdRef, "%g", wideRef);
		return;
	}
	case TypeKind::Bool: {
		// "true"/"false" via CondBr to two write blocks.
		JirBlockRef trueB = gctx.jfn.pushBlock("emit.true");
		JirBlockRef falseB = gctx.jfn.pushBlock("emit.false");
		JirBlockRef joinB = gctx.jfn.pushBlock("emit.bool.end");
		emitCondBr(gctx, val, trueB, falseB);

		auto emitStrLit = [&](JirBlockRef block, const char *literal) {
			gctx.currentBlock = block;
			TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
			TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrMany, 0, 0, BuiltinType::U8, 0});
			StringIdx litId = gctx.ctx.getStringPool().intern(literal);
			JirInst si{};
			si.tag = JirTag::Str;
			si.a = litId;
			si.ty = sliceTy;
			JirRef sl = emit(gctx, si);
			JirInst pi{};
			pi.tag = JirTag::ExtractValue;
			pi.a = sl;
			pi.b = 0;
			pi.ty = u8PtrTy;
			JirRef ptr = emit(gctx, pi);
			JirInst li{};
			li.tag = JirTag::ExtractValue;
			li.a = sl;
			li.b = 1;
			li.ty = BuiltinType::U64;
			JirRef ln = emit(gctx, li);
			JirInst tr{};
			tr.tag = JirTag::Trunc;
			tr.a = ln;
			tr.ty = BuiltinType::I32;
			JirRef lnI32 = emit(gctx, tr);
			StringIdx fmtId = gctx.ctx.getStringPool().intern("%.*s");
			JirInst fs{};
			fs.tag = JirTag::Str;
			fs.a = fmtId;
			fs.ty = sliceTy;
			JirRef fmtSlice = emit(gctx, fs);
			JirInst fp{};
			fp.tag = JirTag::ExtractValue;
			fp.a = fmtSlice;
			fp.b = 0;
			fp.ty = u8PtrTy;
			JirRef fmtPtr = emit(gctx, fp);
			std::vector<uint32_t> pp = {
			    4, static_cast<uint32_t>(fdRef), static_cast<uint32_t>(fmtPtr),
			    static_cast<uint32_t>(lnI32), static_cast<uint32_t>(ptr)};
			JirExtraIdx pe = gctx.jfn.pushExtra(pp.data(), pp.size());
			JirInst c{};
			c.tag = JirTag::Call;
			c.a = gctx.ctx.getStringPool().intern("dprintf");
			c.b = pe;
			c.ty = BuiltinType::I32;
			emit(gctx, c);
			emitBr(gctx, joinB);
		};
		emitStrLit(trueB, "true");
		emitStrLit(falseB, "false");
		gctx.currentBlock = joinB;
		return;
	}
	case TypeKind::Slice: {
		TypeIdx elem = static_cast<TypeIdx>(k.a);
		if (elem != BuiltinType::U8) {
			diags.error(loc, "@emit: slice argument `" + name +
			                     "` is not a slice of u8");
			return;
		}
		TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
		TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrMany, 0, 0, BuiltinType::U8, 0});
		JirInst pi{};
		pi.tag = JirTag::ExtractValue;
		pi.a = val;
		pi.b = 0;
		pi.ty = u8PtrTy;
		JirRef ptr = emit(gctx, pi);
		JirInst li{};
		li.tag = JirTag::ExtractValue;
		li.a = val;
		li.b = 1;
		li.ty = BuiltinType::U64;
		JirRef ln = emit(gctx, li);
		JirInst tr{};
		tr.tag = JirTag::Trunc;
		tr.a = ln;
		tr.ty = BuiltinType::I32;
		JirRef lnI32 = emit(gctx, tr);
		StringIdx fmtId = gctx.ctx.getStringPool().intern("%.*s");
		JirInst fs{};
		fs.tag = JirTag::Str;
		fs.a = fmtId;
		fs.ty = sliceTy;
		JirRef fmtSlice = emit(gctx, fs);
		JirInst fp{};
		fp.tag = JirTag::ExtractValue;
		fp.a = fmtSlice;
		fp.b = 0;
		fp.ty = u8PtrTy;
		JirRef fmtPtr = emit(gctx, fp);
		std::vector<uint32_t> pp = {
		    4, static_cast<uint32_t>(fdRef), static_cast<uint32_t>(fmtPtr),
		    static_cast<uint32_t>(lnI32), static_cast<uint32_t>(ptr)};
		JirExtraIdx pe = gctx.jfn.pushExtra(pp.data(), pp.size());
		JirInst c{};
		c.tag = JirTag::Call;
		c.a = gctx.ctx.getStringPool().intern("dprintf");
		c.b = pe;
		c.ty = BuiltinType::I32;
		emit(gctx, c);
		return;
	}
	default:
		diags.error(loc, "@emit: local `" + name +
		                     "` has a type that's "
		                     "not printable (only "
		                     "int / float / bool / "
		                     "slice<u8> supported)");
		return;
	}
}

// Astgen-side emitter implementing `CompEmitter`. Owns a reference to
// the caller's `AstGenCtx` so each `@-emit` intrinsic drops JIR into
// the right function. Lives only for the duration of one cfn body
// execution.
class CfnEmitter : public jam::CompEmitter {
  public:
	explicit CfnEmitter(AstGenCtx &gctx) : gctx_(gctx) {}

	jam::ExecResult handleAtCall(const std::string &name,
	                             const std::vector<jam::ComptimeValue> &args,
	                             jam::Diagnostics &diags,
	                             jam::SrcLoc loc) override {
		if (name == "emitPutByte") { return handlePutByte(args, diags, loc); }
		if (name == "emitPrintLocalByRange") {
			return handlePrintLocalByRange(args, diags, loc);
		}
		if (name == "emitWriteBytes") {
			return handleWriteBytes(args, diags, loc);
		}
		diags.error(loc, "unknown @-emit intrinsic `@" + name + "`");
		return jam::ExecResult::Error;
	}

  private:
	AstGenCtx &gctx_;

	jam::ExecResult handlePutByte(const std::vector<jam::ComptimeValue> &args,
	                              jam::Diagnostics &diags, jam::SrcLoc loc) {
		if (args.size() != 2 || !args[0].isInt() || !args[1].isInt()) {
			diags.error(loc, "@emitPutByte expects (fd: i32, byte: u8)");
			return jam::ExecResult::Error;
		}
		ensureDprintfForCfn(gctx_);
		JirRef fdRef =
		    literalI32(gctx_, static_cast<uint32_t>(args[0].asU64()));
		JirInst byteI{};
		byteI.tag = JirTag::Int;
		byteI.a = static_cast<uint32_t>(args[1].asU64() & 0xFFu);
		byteI.ty = BuiltinType::I32;  // %c expects an int
		JirRef byteRef = emit(gctx_, byteI);
		emitDprintfSingleArg(gctx_, fdRef, "%c", byteRef);
		return jam::ExecResult::Continue;
	}

	jam::ExecResult
	handlePrintLocalByRange(const std::vector<jam::ComptimeValue> &args,
	                        jam::Diagnostics &diags, jam::SrcLoc loc) {
		if (args.size() != 4 || !args[0].isInt() || !args[1].isStr() ||
		    !args[2].isInt() || !args[3].isInt()) {
			diags.error(loc, "@emitPrintLocalByRange expects "
			                 "(fd: i32, fmt: str, start: u32, end: u32)");
			return jam::ExecResult::Error;
		}
		uint32_t fd = static_cast<uint32_t>(args[0].asU64());
		const std::string &fmtStr =
		    gctx_.ctx.getStringPool().get(args[1].strVal);
		uint32_t start = static_cast<uint32_t>(args[2].asU64());
		uint32_t end = static_cast<uint32_t>(args[3].asU64());
		if (start > end || end > fmtStr.length()) {
			diags.error(loc, "@emitPrintLocalByRange: range out of bounds "
			                 "for the format string");
			return jam::ExecResult::Error;
		}
		std::string name = fmtStr.substr(start, end - start);
		auto it = gctx_.locals.find(name);
		if (it == gctx_.locals.end()) {
			diags.error(loc, "unknown variable `" + name +
			                     "` referenced from cfn format string");
			return jam::ExecResult::Error;
		}
		ensureDprintfForCfn(gctx_);
		// Load the local.
		JirInst load{};
		load.tag = JirTag::Load;
		load.a = it->second;
		load.ty = gctx_.localTypes[name];
		JirRef val = emit(gctx_, load);
		JirRef fdRef = literalI32(gctx_, fd);
		emitDprintfForValue(gctx_, diags, loc, fdRef, val, name);
		return jam::ExecResult::Continue;
	}

	// Emit a single `dprintf(fd, "%.*s", len, ptr)` for the byte slice
	// `fmt[start..end]`. Lets a cfn batch consecutive literal bytes
	// into one runtime write instead of one dprintf per byte. The
	// substring is interned into the global StringPool at compile
	// time; the runtime code just hands its `.rodata` ptr + len to
	// dprintf.
	jam::ExecResult
	handleWriteBytes(const std::vector<jam::ComptimeValue> &args,
	                 jam::Diagnostics &diags, jam::SrcLoc loc) {
		if (args.size() != 4 || !args[0].isInt() || !args[1].isStr() ||
		    !args[2].isInt() || !args[3].isInt()) {
			diags.error(loc, "@emitWriteBytes expects "
			                 "(fd: i32, fmt: str, start: u32, end: u32)");
			return jam::ExecResult::Error;
		}
		uint32_t fd = static_cast<uint32_t>(args[0].asU64());
		const std::string &fmtStr =
		    gctx_.ctx.getStringPool().get(args[1].strVal);
		uint32_t start = static_cast<uint32_t>(args[2].asU64());
		uint32_t end = static_cast<uint32_t>(args[3].asU64());
		if (start > end || end > fmtStr.length()) {
			diags.error(loc, "@emitWriteBytes: range out of bounds "
			                 "for the format string");
			return jam::ExecResult::Error;
		}
		if (start == end) {
			// Empty span — caller-side guard against this is cheap;
			// be permissive so cfn authors don't need to branch.
			return jam::ExecResult::Continue;
		}
		ensureDprintfForCfn(gctx_);
		StringIdx litId =
		    gctx_.ctx.getStringPool().intern(fmtStr.substr(start, end - start));
		TypeIdx sliceTy = gctx_.ctx.getTypePool().intern(
		    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
		TypeIdx u8PtrTy = gctx_.ctx.getTypePool().intern(
		    TypeKey{TypeKind::PtrMany, 0, 0, BuiltinType::U8, 0});
		JirInst sl{};
		sl.tag = JirTag::Str;
		sl.a = litId;
		sl.ty = sliceTy;
		JirRef litSlice = emit(gctx_, sl);
		JirInst pi{};
		pi.tag = JirTag::ExtractValue;
		pi.a = litSlice;
		pi.b = 0;
		pi.ty = u8PtrTy;
		JirRef ptr = emit(gctx_, pi);
		JirInst li{};
		li.tag = JirTag::ExtractValue;
		li.a = litSlice;
		li.b = 1;
		li.ty = BuiltinType::U64;
		JirRef ln = emit(gctx_, li);
		JirInst tr{};
		tr.tag = JirTag::Trunc;
		tr.a = ln;
		tr.ty = BuiltinType::I32;
		JirRef lnI32 = emit(gctx_, tr);
		StringIdx fmtId = gctx_.ctx.getStringPool().intern("%.*s");
		JirInst fs{};
		fs.tag = JirTag::Str;
		fs.a = fmtId;
		fs.ty = sliceTy;
		JirRef fmtSlice = emit(gctx_, fs);
		JirInst fp{};
		fp.tag = JirTag::ExtractValue;
		fp.a = fmtSlice;
		fp.b = 0;
		fp.ty = u8PtrTy;
		JirRef fmtPtr = emit(gctx_, fp);
		JirRef fdRef = literalI32(gctx_, fd);
		std::vector<uint32_t> pp = {
		    4, static_cast<uint32_t>(fdRef), static_cast<uint32_t>(fmtPtr),
		    static_cast<uint32_t>(lnI32), static_cast<uint32_t>(ptr)};
		JirExtraIdx pe = gctx_.jfn.pushExtra(pp.data(), pp.size());
		JirInst c{};
		c.tag = JirTag::Call;
		c.a = gctx_.ctx.getStringPool().intern("dprintf");
		c.b = pe;
		c.ty = BuiltinType::I32;
		emit(gctx_, c);
		return jam::ExecResult::Continue;
	}
};

// Compile-time function call dispatcher. Runs `fn`'s body via the
// ComptimeEvaluator with the call site's args bound into a fresh comp
// scope. The body must evaluate at compile time; any @-emit intrinsic
// it contains (Phase 4) generates JIR into the *caller's* gctx — that's
// how `std.fmt.print` lowers to a sequence of write calls inside the
// user's main() rather than inside print's own LLVM function.
static JirRef astgenCompTimeFnCall(AstGenCtx &gctx, const AstNode &n,
                                   const FunctionAST *fn) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = ns.getExtra(argsExtra);

	// Arg count must match. cfn doesn't support varargs.
	if (argCount != fn->Args.size()) {
		return recoverHere(gctx,
		                   "cfn `" + fn->Name + "` expects " +
		                       std::to_string(fn->Args.size()) +
		                       " arg(s), got " + std::to_string(argCount),
		                   kNoType);
	}

	// Evaluate each arg at compile time and bind to the param name.
	jam::ComptimeEvaluator ev(ns, gctx.ctx.getStringPool(),
	                          gctx.ctx.getTypePool());
	jam::ComptimeScope outer;
	jam::Diagnostics &diags = gctx.ctx.diagnostics();
	jam::SrcLoc loc{gctx.ctx.currentFile(), 0};
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
		jam::ComptimeValue v = ev.eval(argIdx, outer);
		if (v.isNone()) {
			return recoverHere(gctx,
			                   "argument to cfn `" + fn->Name + "` (param `" +
			                       fn->Args[i].Name +
			                       "`) must be a compile-time constant",
			                   kNoType);
		}
		outer.bind(fn->Args[i].Name, std::move(v));
	}

	// Install the emitter so @-emit intrinsics inside the body drop
	// JIR into the caller's gctx. The emitter has a reference to
	// `gctx` for the duration of the body's execution; clearing it
	// after prevents accidental reuse.
	CfnEmitter emitter(gctx);
	ev.setCallContext(&emitter, &diags, loc);
	uint32_t iterCounter = 0;
	jam::ComptimeValue returned;
	std::vector<NodeIdx> bodyVec(fn->Body.begin(), fn->Body.end());
	jam::ExecResult r = ev.execBlock(
	    bodyVec.data(), bodyVec.size(), outer, iterCounter,
	    jam::ComptimeEvaluator::kDefaultIterCap, returned, diags, loc);
	ev.clearCallContext();
	if (r == jam::ExecResult::Error || r == jam::ExecResult::IterationCap) {
		// Diagnostic already pushed; return a poison so the rest of
		// the caller still gets analyzed.
		return recoverHere(gctx,
		                   "cfn `" + fn->Name +
		                       "` failed during compile-"
		                       "time evaluation",
		                   kNoType);
	}
	// cfn returns void in v1; the value (if any) is discarded for now.
	return kNoJirRef;
}

static JirRef astgenCompInstantiatedCall(AstGenCtx &gctx, const AstNode &n,
                                         const FunctionAST *fn) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = ns.getExtra(argsExtra);

	jam::ComptimeEvaluator ev(gctx.ctx.getNodeStore(), gctx.ctx.getStringPool(),
	                          gctx.ctx.getTypePool());
	jam::ComptimeScope scope;
	std::unordered_map<std::string, jam::ComptimeValue> compSubst;
	std::vector<NodeIdx> runtimeArgs;
	std::string mangleSuffix;

	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
		if (i < fn->Args.size() && fn->Args[i].isComp) {
			jam::ComptimeValue v = ev.eval(argIdx, scope);
			if (v.isNone()) {
				return recoverHere(gctx,
				                   "argument for comp param `" +
				                       fn->Args[i].Name +
				                       "` must be a compile-time constant",
				                   kNoType);
			}
			compSubst[fn->Args[i].Name] = v;
			// Mangle a unique suffix per comp value.
			mangleSuffix += "__";
			switch (v.kind) {
			case jam::ComptimeValue::Kind::Int:
				mangleSuffix +=
				    (v.intVal.isSigned ? "i" : "u") + std::to_string(v.asU64());
				break;
			case jam::ComptimeValue::Kind::Bool:
				mangleSuffix += v.boolVal ? "true" : "false";
				break;
			case jam::ComptimeValue::Kind::Str: {
				// Hash to keep the symbol stable / printable for
				// arbitrary contents.
				std::hash<std::string> hash;
				const std::string &s = gctx.ctx.getStringPool().get(v.strVal);
				mangleSuffix += "s" + std::to_string(hash(s));
				break;
			}
			case jam::ComptimeValue::Kind::Type: {
				const TypeKey &tk = gctx.ctx.getTypePool().get(v.typeVal);
				if (tk.kind == TypeKind::Int) {
					mangleSuffix += (tk.b ? "i" : "u") + std::to_string(tk.a);
				} else if (tk.kind == TypeKind::Bool) {
					mangleSuffix += "bool";
				} else {
					mangleSuffix += "t" + std::to_string(v.typeVal);
				}
				break;
			}
			default:
				mangleSuffix += "x";
				break;
			}
		} else {
			runtimeArgs.push_back(argIdx);
		}
	}

	std::string instName = fn->Name + mangleSuffix;

	// Cache hit? Skip the clone+lower and dispatch to the existing
	// instantiation.
	const FunctionAST *clone = gctx.ctx.getFunctionAST(instName);
	if (clone == nullptr) {
		// Build a clone whose signature drops the comp params. Body
		// stays as the original NodeStore indices — astgen reads the
		// comp subst when it encounters the comp param names.
		std::vector<Param> instArgs;
		instArgs.reserve(fn->Args.size());
		for (const auto &p : fn->Args) {
			if (!p.isComp) instArgs.push_back(p);
		}
		auto cloned = std::make_unique<FunctionAST>(
		    instName, std::move(instArgs), fn->ReturnType, fn->Body,
		    fn->isExtern, fn->isExport, fn->isPub, fn->isTest, fn->isVarArgs,
		    fn->isCfn);
		// Note: modulePath stays empty on the clone — stamping it
		// would alter the mangled symbol name (mangling.h:60). Body-
		// scope identifier resolution is carried separately via
		// bodyModuleStack_, pushed below around astgenBodyInto.
		clone = gctx.ctx.adoptInstantiatedFunction(std::move(cloned));
		gctx.ctx.registerFunctionAST(instName,
		                             const_cast<FunctionAST *>(clone));

		// Save the builder so we can return to the caller's insertion
		// point once the clone body's been emitted.
		JamBasicBlockRef savedBB = JamLLVMGetInsertBlock(gctx.ctx.getBuilder());

		gctx.ctx.setCurrentCompSubst(compSubst);
		JirFunction jfn = astgenMetadata(*clone, gctx.ctx);
		jfn.name = clone->Name;
		jirDeclarePrototype(jfn, gctx.ctx);
		gctx.ctx.pushBodyModule(fn->modulePath);
		astgenBodyInto(jfn, *clone, gctx.ctx);
		gctx.ctx.popBodyModule();
		jirDefineBody(jfn, gctx.ctx);
		gctx.ctx.clearCurrentCompSubst();

		if (savedBB) {
			JamLLVMPositionBuilderAtEnd(gctx.ctx.getBuilder(), savedBB);
		}
	}

	// Emit a regular Call to the cloned fn, passing only the runtime
	// args. lowerArg honours each param's mode (mut/move pass-by-ptr).
	std::vector<JirRef> argRefs;
	argRefs.reserve(runtimeArgs.size());
	for (size_t i = 0; i < runtimeArgs.size(); i++) {
		if (i < clone->Args.size()) {
			argRefs.push_back(lowerArg(gctx, runtimeArgs[i], clone->Args[i]));
		} else {
			argRefs.push_back(astgenExpr(gctx, runtimeArgs[i], kNoType));
		}
	}
	return emitCall(gctx, clone, argRefs);
}

static JirRef astgenCall(AstGenCtx &gctx, const AstNode &n, JirRef destPtr) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t argCount = ns.getExtra(argsExtra);

	// Indirect-call form (`flags & 1`): `expr.method(args)` where `expr`
	// is an arbitrary subexpression (chained calls, parenthesized, etc).
	// `n.lhs` is a NodeIdx into a MemberAccess that names the method.
	// We resolve the receiver's *type* by inspecting the MemberAccess
	// base, look up `RecvType.method` in the function registry, then
	// emit a Call passing the receiver value (or its address) as arg 0.
	if (n.flags & 1) {
		NodeIdx calleeNodeIdx = static_cast<NodeIdx>(n.lhs);
		const AstNode &cn = ns.get(calleeNodeIdx);
		if (cn.tag != AstTag::MemberAccess) {
			failHere(gctx,
			         "astgen: indirect call must be on a `.method` callee");
		}
		NodeIdx recvExprIdx = static_cast<NodeIdx>(cn.lhs);
		StringIdx methodNameId = static_cast<StringIdx>(cn.rhs);
		const std::string &methodName =
		    gctx.ctx.getStringPool().get(methodNameId);

		// Built-in array methods. `[N]T` exposes two zero-cost
		// pointer-handover methods, modeled on Rust's
		// `[T; N]::as_ptr` / `as_mut_ptr`:
		//
		//   arr.asPtr()    -> *const[] T  (FFI / read-only callees)
		//   arr.asMutPtr() -> *mut[] T    (FFI / write-needing callees)
		//
		// We peek the receiver type via astgenLvalue rather than
		// astgenExpr — for array receivers, an rvalue Load would
		// pull the entire backing storage (up to several KB) into an
		// SSA value as a dead instruction, which stalls LLVM at
		// -O1+. Taking the receiver as a pointer first keeps the
		// hot path on a GEP and lets the rvalue Load only fire on
		// the fall-through struct/enum dispatch path below.
		const AstNode &recvNode = ns.get(recvExprIdx);
		bool recvIsLvalueable = recvNode.tag == AstTag::Variable ||
		                        recvNode.tag == AstTag::MemberAccess ||
		                        recvNode.tag == AstTag::Index ||
		                        recvNode.tag == AstTag::Deref;
		if (recvIsLvalueable &&
		    (methodName == "asPtr" || methodName == "asMutPtr")) {
			JirRef basePtr =
			    astgenExpr(gctx, recvExprIdx, kNoType, ResultLoc::Pointer);
			TypeIdx leafTy = gctx.jfn.getInst(basePtr).ty;
			const TypeKey &lk0 = gctx.ctx.getTypePool().get(leafTy);
			// MemberAccess on a struct field returns a FieldAddr whose
			// type is `PtrSingle(arrayTy)` (the field's storage
			// address), not the array itself. Unwrap one pointer level
			// so the Array check fires for `self.field.asMutPtr()` the
			// same way it does for `arr.asMutPtr()` on a local. Same
			// goes for `PtrMany` from an Index lvalue.
			TypeKind lkKind = lk0.kind;
			TypeIdx unwrappedTy = leafTy;
			if (lkKind == TypeKind::PtrSingle || lkKind == TypeKind::PtrMany) {
				unwrappedTy = static_cast<TypeIdx>(lk0.a);
				lkKind = gctx.ctx.getTypePool().get(unwrappedTy).kind;
			}
			const TypeKey &lk = gctx.ctx.getTypePool().get(unwrappedTy);
			if (lkKind == TypeKind::Array) {
				TypeIdx elemTy = static_cast<TypeIdx>(lk.a);
				TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
				    TypeKey{TypeKind::PtrMany, 0, 0, elemTy, 0});
				// IndexAddr at 0 produces a `PtrMany(elem)` for the
				// array's base — same GEP shape `arr[i]` uses.
				JirInst zeroI{};
				zeroI.tag = JirTag::Int;
				zeroI.a = 0;
				zeroI.ty = BuiltinType::U64;
				JirRef zeroRef = emit(gctx, zeroI);
				JirInst ia{};
				ia.tag = JirTag::IndexAddr;
				ia.a = basePtr;
				ia.b = zeroRef;
				ia.ty = ptrTy;
				return emit(gctx, ia);
			}
			// Not an array — fall through to the rvalue dispatch
			// below. (basePtr is unused; cheap, no Load was emitted.)
		}

		// Receiver lowering. The goal is to never emit a full-
		// aggregate value-load just to peek the receiver's type when
		// `outer.field.method()` embeds a large struct.
		//   * lvalueable recv (Variable / MemberAccess / Index /
		//     Deref): astgenLvalue gives the pointer and the receiver
		//     type with no value-load. mut/move methods take the ptr
		//     straight; by-value methods Load through it.
		//   * non-lvalueable recv (Call result, etc.): value path.
		//     by-value methods consume the value directly; mut/move
		//     methods spill to a fresh alloca.
		JirRef recvVal = kNoJirRef;
		JirRef recvLvaluePtr = kNoJirRef;
		TypeIdx recvTy = kNoType;
		const AstNode &recvNodeForLvalue = ns.get(recvExprIdx);
		bool recvLvalueable = recvNodeForLvalue.tag == AstTag::Variable ||
		                      recvNodeForLvalue.tag == AstTag::MemberAccess ||
		                      recvNodeForLvalue.tag == AstTag::Index ||
		                      recvNodeForLvalue.tag == AstTag::Deref;
		if (recvLvalueable) {
			TypeIdx leafTy = kNoType;
			recvLvaluePtr = astgenLvalue(gctx, recvExprIdx, leafTy);
			recvTy = leafTy;
		} else {
			recvVal = astgenExpr(gctx, recvExprIdx, kNoType);
			recvTy = gctx.jfn.getInst(recvVal).ty;
			const TypeKey &recvKey = gctx.ctx.getTypePool().get(recvTy);
			if (recvKey.kind == TypeKind::Array) {
				// Non-lvalueable array receiver (e.g. a call result):
				// without a stable storage location we can't safely
				// hand a pointer to FFI — reject.
				failHere(gctx, "astgen: `" + methodName +
				                   "()` requires an addressable array "
				                   "(variable, field, or indexed slot)");
			}
		}

		// Built-in `recv.clone()` — explicit deep copy (CLONE_PLAN.md).
		// Works on EVERY type: plain data is the value itself, structs
		// without their own cfn clone get structural glue, arrays clone
		// element-wise. A user cfn clone returns kNoJirRef here and
		// dispatches as an ordinary method below.
		if (methodName == "clone" && argCount == 0) {
			JirRef cloned =
			    tryLowerBuiltinClone(gctx, recvLvaluePtr, recvVal, recvTy);
			if (cloned != kNoJirRef) return cloned;
		}

		// Resolve to a struct name (generic instantiation happens here
		// as a side effect of lookupStruct).
		std::string recvName;
		if (const auto *sinfo = gctx.ctx.lookupStruct(recvTy)) {
			recvName = sinfo->name;
		} else if (const auto *einfo = gctx.ctx.lookupEnum(recvTy)) {
			recvName = einfo->name;
		} else {
			failHere(gctx,
			         "astgen: indirect-call receiver is not a struct/enum");
		}
		std::string qualified = recvName + "." + methodName;
		const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
		if (method == nullptr) {
			return reportMethodMiss(gctx, qualified);
		}
		ParamMode mode =
		    method->Args.empty() ? ParamMode::Let : method->Args[0].Mode;
		// Dispatch on the full ABI, not just ParamMode: a `let
		// self: Self` where Self is byref now arrives ByPointer too,
		// so the receiver needs to land as a pointer rather than the
		// loaded aggregate value.
		jam::abi::ParamABI recvAbi =
		    jam::abi::classifyParam(mode, recvTy, gctx.ctx);
		JirRef recvArg;
		if (recvAbi.kind == jam::abi::ParamABI::Kind::ByPointer) {
			if (recvLvaluePtr != kNoJirRef) {
				recvArg = recvLvaluePtr;
			} else {
				// Non-lvalue receiver with a by-pointer method: spill
				// the already-computed value to a fresh alloca so the
				// callee gets a stable storage address. For byref-
				// typed rvalues `recvVal` is itself a pointer (from
				// the new StructLit / Call codegen), so the spill is
				// a memcpy via the Store JIR tag.
				JirInst alloca{};
				alloca.tag = JirTag::Alloca;
				alloca.ty = recvTy;
				JirRef ptr = emitAllocaHoisted(gctx, alloca);
				JirInst store{};
				store.tag = JirTag::Store;
				store.a = ptr;
				store.b = recvVal;
				emit(gctx, store);
				recvArg = ptr;
			}
		} else {
			if (recvVal != kNoJirRef) {
				recvArg = recvVal;
			} else {
				JirInst load{};
				load.tag = JirTag::Load;
				load.a = recvLvaluePtr;
				load.ty = recvTy;
				recvArg = emit(gctx, load);
			}
		}
		std::vector<JirRef> argRefs;
		argRefs.reserve(1 + argCount);
		argRefs.push_back(recvArg);
		for (uint32_t i = 0; i < argCount; i++) {
			NodeIdx argIdx =
			    static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
			if (1 + i < method->Args.size()) {
				rejectAddrOfOnModeArg(gctx, argIdx, method->Args[1 + i]);
			}
			TypeIdx expectArg = (1 + i < method->Args.size())
			                        ? method->Args[1 + i].Type
			                        : kNoType;
			if (1 + i < method->Args.size() &&
			    method->Args[1 + i].Mode == ParamMode::Move) {
				rejectDropBearingFieldExtract(gctx, argIdx, expectArg,
				                              "move");
			}
			argRefs.push_back(astgenExpr(gctx, argIdx, expectArg));
			// A `move`-mode method argument (`v.push(local)`) transfers
			// ownership to the callee — suppress the caller-side
			// scope-exit drop, same as lowerArg does for free fns.
			if (1 + i < method->Args.size() &&
			    method->Args[1 + i].Mode == ParamMode::Move) {
				consumeMovedVariable(gctx, argIdx);
			}
		}
		return emitCall(gctx, method, argRefs, destPtr);
	}

	StringIdx calleeId = static_cast<StringIdx>(n.lhs);
	std::string callee = gctx.ctx.getStringPool().get(calleeId);

	// Builtin: `assert(actual, expected)` — handled via JIR inline,
	// not as a regular function call.
	if (callee == "assert") { return astgenAssertCall(gctx, n); }

	// Multi-dot qualified call: `handle.Struct.method(args)`. Mirrors
	// Zig's `container_ty.getNamespace().lookupInNamespace(name)`
	// (Sema.zig:5295) — methods on imported structs live under the
	// importer's namespace handle, not in a flat global table. The
	// registration site in main.cpp puts these under the key
	// `handle.Struct.method`; we look them up directly here.
	if (callee.find('.') != callee.rfind('.')) {
		if (const FunctionAST *method = gctx.ctx.getFunctionAST(callee)) {
			std::vector<JirRef> argRefs;
			argRefs.reserve(argCount);
			for (uint32_t i = 0; i < argCount; i++) {
				NodeIdx argIdx =
				    static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
				if (i < method->Args.size()) {
					argRefs.push_back(lowerArg(gctx, argIdx, method->Args[i]));
				} else {
					argRefs.push_back(astgenExpr(gctx, argIdx, kNoType));
				}
			}
			return emitCall(gctx, method, argRefs, destPtr);
		}
	}

	// Single-dot qualified call: try in order
	//   1. `inst.method(args)` — instance dispatch on a local variable
	//   2. `Type.method(args)` — static method, possibly via type alias
	{
		size_t firstDot = callee.find('.');
		size_t lastDot = callee.rfind('.');
		if (firstDot != std::string::npos && firstDot == lastDot) {
			std::string prefix = callee.substr(0, firstDot);
			std::string methodName = callee.substr(firstDot + 1);

			// `Self.method(...)` inside an instantiated generic body
			// resolves through the substitution context set up by
			// instantiateStructExpr before lowering the body.
			std::string resolvedPrefix = resolvePrefix(gctx.ctx, prefix);

			// Type-alias / type-name static dispatch: if `prefix` names
			// a registered struct (or aliases one), rewrite the call to
			// the canonical-name method.
			std::string canonicalType;
			const auto *enumForVariant =
			    static_cast<const decltype(gctx.ctx.getEnum(""))>(nullptr);
			TypeIdx aliasTarget = gctx.ctx.lookupTypeAlias(resolvedPrefix);
			if (aliasTarget != kNoType) {
				if (const auto *sinfo = gctx.ctx.lookupStruct(aliasTarget)) {
					canonicalType = sinfo->name;
				} else if (const auto *einfo =
				               gctx.ctx.lookupEnum(aliasTarget)) {
					canonicalType = einfo->name;
					enumForVariant = einfo;
				}
			}
			if (canonicalType.empty()) {
				if (const auto *sinfo = gctx.ctx.getStruct(resolvedPrefix)) {
					canonicalType = sinfo->name;
				} else if (const auto *einfo =
				               gctx.ctx.getEnum(resolvedPrefix)) {
					canonicalType = einfo->name;
					enumForVariant = einfo;
				}
			}
			// Enum-variant constructor (`Result.Ok(x)`-style Call):
			// the suffix is a variant name on the resolved enum.
			if (enumForVariant != nullptr) {
				int vidx =
				    gctx.ctx.getEnumVariantIndex(canonicalType, methodName);
				if (vidx >= 0) {
					const auto &variant = enumForVariant->variants[vidx];
					TypeIdx enumTy = gctx.ctx.getTypePool().intern(TypeKey{
					    TypeKind::Named, 0, 0,
					    static_cast<uint32_t>(
					        gctx.ctx.getStringPool().intern(canonicalType)),
					    0});
					JirInst tag{};
					tag.tag = JirTag::Int;
					tag.a = static_cast<uint32_t>(variant.discriminant);
					tag.ty = BuiltinType::U8;
					JirRef tagRef = emit(gctx, tag);
					if (!enumForVariant->hasPayloadVariant) { return tagRef; }
					// Build {tag, payload-undef|val} via alloca + FieldAddr
					// stores, then load. Mirrors the TypeMethodCall path.
					JirInst alloca{};
					alloca.tag = JirTag::Alloca;
					alloca.ty = enumTy;
					JirRef slot = emitAllocaHoisted(gctx, alloca);
					TypeIdx u8PtrTy = gctx.ctx.getTypePool().intern(
					    TypeKey{TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});
					JirInst tagFA{};
					tagFA.tag = JirTag::FieldAddr;
					tagFA.a = slot;
					tagFA.b = 0;
					tagFA.ty = u8PtrTy;
					JirRef tagPtr = emit(gctx, tagFA);
					JirInst tagStore{};
					tagStore.tag = JirTag::Store;
					tagStore.a = tagPtr;
					tagStore.b = tagRef;
					emit(gctx, tagStore);
					// Store every payload field. Each one lives at its own
					// byte offset within the payload area (field 1 of the
					// enum struct is the alignment driver, which holds the
					// first payload; subsequent payloads spill into the
					// `extraBytes` array — see codegen.cpp's enum layout).
					// We address them all uniformly via a payload-area
					// pointer + byte offset GEP through i8.
					if (argCount > variant.payloadTypes.size()) {
						failHere(gctx, "astgen: too many args for variant `" +
						                   canonicalType + "." + methodName +
						                   "`");
					}
					if (!variant.payloadTypes.empty() && argCount >= 1) {
						TypeIdx payAreaPtrTy =
						    gctx.ctx.getTypePool().intern(TypeKey{
						        TypeKind::PtrSingle, 0, 0, BuiltinType::U8, 0});
						JirInst payAreaFA{};
						payAreaFA.tag = JirTag::FieldAddr;
						payAreaFA.a = slot;
						payAreaFA.b = 1;
						payAreaFA.ty = payAreaPtrTy;
						JirRef payAreaPtr = emit(gctx, payAreaFA);

						// Byte-stride GEP through `payAreaPtr` (which is *u8)
						// gives the per-field pointer. Encode the result type
						// as `*u8` so jir_codegen's IndexAddr uses i8 stride
						// (1 byte per `idx` step); opaque pointers let us
						// Store the field's actual type to the resulting ptr
						// without a separate cast.
						uint64_t off = 0;
						for (uint32_t i = 0;
						     i < variant.payloadTypes.size() && i < argCount;
						     i++) {
							TypeIdx fieldTy = variant.payloadTypes[i];
							uint64_t s = gctx.ctx.typeSize(fieldTy);
							uint64_t a = gctx.ctx.typeAlign(fieldTy);
							off = (off + a - 1) / a * a;
							NodeIdx argIdxN = static_cast<NodeIdx>(
							    ns.getExtra(argsExtra + 1 + i));
							JirRef payVal = astgenExpr(gctx, argIdxN, fieldTy);
							// Enum payload capture is a MOVE — the enum
							// owns the value (see the TypeMethodCall
							// constructor for the same rule).
							rejectDropBearingFieldExtract(
							    gctx, argIdxN, fieldTy, "capture");
							consumeMovedVariable(gctx, argIdxN);
							JirInst gepInst{};
							gepInst.tag = JirTag::IndexAddr;
							gepInst.a = payAreaPtr;
							JirInst offC{};
							offC.tag = JirTag::Int;
							offC.a = static_cast<uint32_t>(off);
							offC.ty = BuiltinType::U64;
							JirRef offRef = emit(gctx, offC);
							gepInst.b = offRef;
							gepInst.ty = payAreaPtrTy;  // *u8 -> byte stride
							JirRef fieldPtr = emit(gctx, gepInst);
							JirInst payStore{};
							payStore.tag = JirTag::Store;
							payStore.a = fieldPtr;
							payStore.b = payVal;
							emit(gctx, payStore);
							off += s;
						}
					}
					JirInst load{};
					load.tag = JirTag::Load;
					load.a = slot;
					load.ty = enumTy;
					return emit(gctx, load);
				}
			}
			if (!canonicalType.empty()) {
				std::string qualified = canonicalType + "." + methodName;
				const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
				if (method != nullptr) {
					std::vector<JirRef> argRefs;
					argRefs.reserve(argCount);
					for (uint32_t i = 0; i < argCount; i++) {
						NodeIdx argIdx = static_cast<NodeIdx>(
						    ns.getExtra(argsExtra + 1 + i));
						if (i < method->Args.size()) {
							argRefs.push_back(
							    lowerArg(gctx, argIdx, method->Args[i]));
						} else {
							// varargs tail — pass by value.
							argRefs.push_back(
							    astgenExpr(gctx, argIdx, kNoType));
						}
					}
					// `emitCall` mangles drop and test names — no need
					// to special-case here.
					return emitCall(gctx, method, argRefs, destPtr);
				}
				// Prefix resolved to a real struct / enum but the
				// method doesn't exist. The common trigger is a
				// generic body calling `T.default()` where the
				// instantiation's T (e.g. NoDefault) doesn't define
				// `default`. Naming both type and method gives the
				// user a precise pointer to the missing piece.
				failHere(gctx, "type `" + canonicalType + "` has no method `" +
				                   methodName + "`");
			}

			auto it = gctx.locals.find(prefix);
			if (it != gctx.locals.end()) {
				std::string instName = prefix;
				TypeIdx instTy = gctx.localTypes[instName];
				// Built-in `local.clone()` — explicit deep copy
				// (CLONE_PLAN.md). Handles plain data (the value
				// itself), structural glue, and element-wise arrays; a
				// user cfn clone returns kNoJirRef and dispatches as an
				// ordinary method below.
				if (methodName == "clone" && argCount == 0) {
					JirRef cloned = tryLowerBuiltinClone(
					    gctx, it->second, kNoJirRef, instTy);
					if (cloned != kNoJirRef) return cloned;
				}
				// Built-in array methods: `arr.asPtr()` / `arr.asMutPtr()`
				// return the array's storage base as a `PtrMany(elem)`.
				// Mirrors Rust's `[T; N]::as_ptr` / `as_mut_ptr`. See the
				// matching block in the indirect-call path for the
				// rationale.
				//
				// Implementation: emit an `IndexAddr` at index 0 against
				// the alloca, typing the result as `PtrMany(elem)`. This
				// reuses the same JIR shape that `arr[i]` already produces
				// (a GEP that LLVM fully understands and can optimize
				// through), instead of a direct alloca-to-pointer bitcast
				// which sent LLVM's opt passes into a loop in early
				// experiments.
				const TypeKey &instKey = gctx.ctx.getTypePool().get(instTy);
				if (instKey.kind == TypeKind::Array) {
					if (methodName == "asPtr" || methodName == "asMutPtr") {
						TypeIdx elemTy = static_cast<TypeIdx>(instKey.a);
						TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
						    TypeKey{TypeKind::PtrMany, 0, 0, elemTy, 0});
						JirInst zeroI{};
						zeroI.tag = JirTag::Int;
						zeroI.a = 0;
						zeroI.ty = BuiltinType::U64;
						JirRef zeroRef = emit(gctx, zeroI);
						JirInst ia{};
						ia.tag = JirTag::IndexAddr;
						ia.a = it->second;  // alloca slot
						ia.b = zeroRef;
						ia.ty = ptrTy;
						return emit(gctx, ia);
					}
					failHere(gctx, "astgen: array type has no method `" +
					                   methodName + "`");
				}
				const auto *sinfo = gctx.ctx.lookupStruct(instTy);
				if (sinfo != nullptr) {
					std::string qualified = sinfo->name + "." + methodName;
					const FunctionAST *method =
					    gctx.ctx.getFunctionAST(qualified);
					// A withdrawn conditional method replays its
					// reason here rather than falling through to the
					// module-handle fallback's confusing error.
					if (method == nullptr &&
					    gctx.ctx.getWithdrawnMethod(qualified) != nullptr) {
						return reportMethodMiss(gctx, qualified);
					}
					if (method != nullptr && !method->Args.empty()) {
						// Dispatch on the full ABI — a `let self:
						// Self` where Self is byref arrives ByPointer,
						// so the receiver must land as a pointer to
						// the local's storage, not a loaded value.
						ParamMode mode = method->Args[0].Mode;
						jam::abi::ParamABI recvAbi =
						    jam::abi::classifyParam(mode, instTy, gctx.ctx);
						JirRef recvRef;
						if (recvAbi.kind ==
						    jam::abi::ParamABI::Kind::ByPointer) {
							TypeIdx pointee = instTy;
							TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
							    TypeKey{TypeKind::PtrSingle, 0, 0, pointee, 0});
							JirInst ao{};
							ao.tag = JirTag::AddrOf;
							ao.a = it->second;
							ao.ty = ptrTy;
							recvRef = emit(gctx, ao);
						} else {
							JirInst ld{};
							ld.tag = JirTag::Load;
							ld.a = it->second;
							ld.ty = instTy;
							recvRef = emit(gctx, ld);
						}
						std::vector<JirRef> argRefs;
						argRefs.reserve(1 + argCount);
						argRefs.push_back(recvRef);
						for (uint32_t i = 0; i < argCount; i++) {
							NodeIdx argIdx = static_cast<NodeIdx>(
							    ns.getExtra(argsExtra + 1 + i));
							if (1 + i < method->Args.size()) {
								rejectAddrOfOnModeArg(gctx, argIdx,
								                      method->Args[1 + i]);
								if (method->Args[1 + i].Mode ==
								    ParamMode::Move) {
									rejectDropBearingFieldExtract(
									    gctx, argIdx,
									    method->Args[1 + i].Type, "move");
								}
							}
							TypeIdx expectArg = (1 + i < method->Args.size())
							                        ? method->Args[1 + i].Type
							                        : kNoType;
							argRefs.push_back(
							    astgenExpr(gctx, argIdx, expectArg));
							// `move`-mode method arg: ownership moves to
							// the callee; suppress the caller-side drop.
							if (1 + i < method->Args.size() &&
							    method->Args[1 + i].Mode == ParamMode::Move) {
								consumeMovedVariable(gctx, argIdx);
							}
						}
						return emitCall(gctx, method, argRefs, destPtr);
					}
				}
			}
		}
	}

	const FunctionAST *fn = gctx.ctx.getFunctionAST(callee);
	if (fn == nullptr && callee.find('.') != std::string::npos) {
		// Chained module access fallback: `std.fmt.print(...)` etc.
		// The flat lookup above only knows `handle.X` (single-dot)
		// entries; this walks ModuleNamespace.moduleAliases for each
		// intermediate segment and resolves the final segment as a
		// function in the leaf module's namespace.
		fn = gctx.ctx.resolveChainedFunction(callee);
	}
	// Compile-time function calls: `pub cfn` declared at the top level.
	// The cfn body executes at compile time, with each call site
	// generating a fresh execution. Args must be comp-known. The body
	// is interpreted by `ComptimeEvaluator`; @-emit intrinsics inside
	// the body drop JIR into the calling function's gctx (Phase 4 wiring).
	if (fn != nullptr && fn->isCompTimeFn) {
		return astgenCompTimeFnCall(gctx, n, fn);
	}
	// Comp-instantiated calls: fn has at least one `comp` param. Each
	// unique combination of comp args produces a fresh monomorphisation
	// (cached in the function registry under a mangled name). Runtime
	// args are passed through to the clone as normal Call args.
	if (fn != nullptr) {
		bool hasComp = false;
		for (const auto &p : fn->Args) {
			if (p.isComp) {
				hasComp = true;
				break;
			}
		}
		if (hasComp) { return astgenCompInstantiatedCall(gctx, n, fn); }
	}
	if (fn == nullptr) {
		// Before erroring, try the fn-pointer-in-local-or-field paths.
		// Two cases, both producing a Fn-typed JirRef we can call
		// indirect through:
		//   (1) zero-dot callee `f` is a local whose type is Fn.
		//       `var f: fn(...) = ...; f(args);`
		//   (2) single-dot callee `recv.field` where `field` is a Fn-
		//       typed field on recv's struct. `w.writeFn(args);`
		// Multi-dot callees (`x.y.z(args)`) take the indirect-call path
		// in the parser already; this branch is only for the cases the
		// parser emitted as direct Call (qualified-name based).
		auto buildIndirectCall = [&](JirRef calleeVal) -> JirRef {
			TypeIdx calleeTy = gctx.jfn.getInst(calleeVal).ty;
			const TypeKey &k = gctx.ctx.getTypePool().get(calleeTy);
			TypeIdx retTy = static_cast<TypeIdx>(k.a);
			const auto &paramTys = gctx.ctx.getTypePool().fnParamsAt(k.b);
			std::vector<JirRef> argRefs;
			argRefs.reserve(argCount);
			for (uint32_t i = 0; i < argCount; i++) {
				NodeIdx argIdx =
				    static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
				TypeIdx expectArg = i < paramTys.size() ? paramTys[i] : kNoType;
				argRefs.push_back(astgenExpr(gctx, argIdx, expectArg));
			}
			std::vector<uint32_t> packed;
			packed.reserve(1 + argRefs.size());
			packed.push_back(static_cast<uint32_t>(argRefs.size()));
			for (JirRef r : argRefs) packed.push_back(static_cast<uint32_t>(r));
			JirExtraIdx extraIdx =
			    gctx.jfn.pushExtra(packed.data(), packed.size());
			JirInst ic{};
			ic.tag = JirTag::CallIndirect;
			ic.a = calleeVal;
			ic.b = extraIdx;
			ic.ty = retTy;
			return emit(gctx, ic);
		};

		size_t dotPos = callee.find('.');
		if (dotPos == std::string::npos) {
			// (1) bare name — is it a Fn-typed local?
			auto it = gctx.locals.find(callee);
			if (it != gctx.locals.end()) {
				TypeIdx localTy = gctx.localTypes[callee];
				const TypeKey &k = gctx.ctx.getTypePool().get(localTy);
				if (k.kind == TypeKind::Fn) {
					JirInst load{};
					load.tag = JirTag::Load;
					load.a = it->second;
					load.ty = localTy;
					JirRef fnVal = emit(gctx, load);
					return buildIndirectCall(fnVal);
				}
			}
		} else if (callee.find('.', dotPos + 1) == std::string::npos) {
			// (2) single-dot `recv.field` — fall through only when
			// `recv` is a local AND `field` is a Fn-typed field on
			// its struct type.
			std::string recvName = callee.substr(0, dotPos);
			std::string fieldName = callee.substr(dotPos + 1);
			auto it = gctx.locals.find(recvName);
			if (it != gctx.locals.end()) {
				TypeIdx recvTy = gctx.localTypes[recvName];
				const auto *sinfo = gctx.ctx.lookupStruct(recvTy);
				if (sinfo != nullptr) {
					for (size_t i = 0; i < sinfo->fields.size(); ++i) {
						if (sinfo->fields[i].first != fieldName) continue;
						TypeIdx fieldTy = sinfo->fields[i].second;
						const TypeKey &fk = gctx.ctx.getTypePool().get(fieldTy);
						if (fk.kind != TypeKind::Fn) break;
						// Load recv as a value, ExtractValue the field.
						JirInst loadRecv{};
						loadRecv.tag = JirTag::Load;
						loadRecv.a = it->second;
						loadRecv.ty = recvTy;
						JirRef recvVal = emit(gctx, loadRecv);
						JirInst ev{};
						ev.tag = JirTag::ExtractValue;
						ev.a = recvVal;
						ev.b = static_cast<uint32_t>(i);
						ev.ty = fieldTy;
						JirRef fnVal = emit(gctx, ev);
						return buildIndirectCall(fnVal);
					}
				}
			}
		}

		// Qualified callees (`lib.priv`) get the precise pub-access
		// diagnostic via formatNamespaceLookupError.
		std::string msg =
		    (callee.find('.') != std::string::npos)
		        ? gctx.ctx.formatNamespaceLookupError("function", callee)
		        : "unknown function `" + callee + "`";
		return recoverHere(gctx, std::move(msg), kNoType);
	}

	std::vector<JirRef> argRefs;
	argRefs.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
		if (i < fn->Args.size()) {
			argRefs.push_back(lowerArg(gctx, argIdx, fn->Args[i]));
		} else {
			// varargs tail — pass by value.
			argRefs.push_back(astgenExpr(gctx, argIdx, kNoType));
		}
	}
	(void)calleeId;
	return emitCall(gctx, fn, argRefs, destPtr);
}

static JirRef astgenExpr(AstGenCtx &gctx, NodeIdx node, TypeIdx expected,
                         ResultLoc loc, TypeIdx *outLeafTy) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	const AstNode &n = ns.get(node);
	int line = ns.getLine(node);
	// Stamp the current node so error helpers without an explicit
	// NodeIdx in scope can still produce a SrcLoc anchored at the
	// expression we're working on. We never need to restore on exit:
	// the *last* entered node is the right anchor for any error
	// raised before we successfully return, and successful returns
	// either get overwritten by the next sibling's astgenExpr call
	// or fall out of scope entirely.
	gctx.currentNode = node;

	// Pointer-loc dispatch for lvalue-shaped nodes. Each branch
	// emits the address-producing JIR directly (alloca slot for
	// Variable; FieldAddr / BitCast for MemberAccess; IndexAddr for
	// Index; the operand itself for Deref) and writes the leaf
	// type into `*outLeafTy` for callers that need it. The
	// producer's job is to honor the caller's request without
	// materializing the underlying value. Non-lvalue tags fall
	// through to the value-producing switch below and the final
	// alloca-spill catches them.
	if (loc == ResultLoc::Pointer) {
		TypeIdx leafTy = kNoType;
		JirRef ptrResult = kNoJirRef;
		bool handled = true;
		switch (n.tag) {
		case AstTag::Variable: {
			const std::string &name =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
			auto it = gctx.locals.find(name);
			if (it == gctx.locals.end()) {
				failHere(gctx,
				         "astgen: unknown lvalue variable `" + name + "`");
			}
			leafTy = gctx.localTypes[name];
			ptrResult = it->second;
			break;
		}
		case AstTag::Deref: {
			// `p.* = ...` — operand IS the pointer value.
			JirRef innerPtr =
			    astgenExpr(gctx, static_cast<NodeIdx>(n.lhs), kNoType);
			TypeIdx pty = gctx.jfn.getInst(innerPtr).ty;
			const TypeKey &pk = gctx.ctx.getTypePool().get(pty);
			if (pk.kind != TypeKind::PtrSingle &&
			    pk.kind != TypeKind::PtrMany) {
				failHere(gctx, "astgen: cannot deref non-pointer");
			}
			leafTy = static_cast<TypeIdx>(pk.a);
			ptrResult = innerPtr;
			break;
		}
		case AstTag::MemberAccess: {
			TypeIdx baseTy = kNoType;
			JirRef basePtr = astgenExpr(gctx, static_cast<NodeIdx>(n.lhs),
			                            kNoType, ResultLoc::Pointer, &baseTy);
			StringIdx memberId = static_cast<StringIdx>(n.rhs);
			const std::string &memberName =
			    gctx.ctx.getStringPool().get(memberId);
			// Union field lvalue: every field shares the union's
			// address — the field pointer IS the union pointer
			// (just retyped).
			if (const auto *uinfo = gctx.ctx.lookupUnion(baseTy)) {
				TypeIdx fieldTy =
				    gctx.ctx.getUnionFieldType(uinfo->name, memberName);
				if (fieldTy == kNoType) {
					failHere(gctx, "astgen: union `" + uinfo->name +
					                   "` has no field `" + memberName + "`");
				}
				leafTy = fieldTy;
				TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
				    TypeKey{TypeKind::PtrSingle, 0, 0, fieldTy, 0});
				JirInst bc{};
				bc.tag = JirTag::BitCast;
				bc.a = basePtr;
				bc.ty = ptrTy;
				ptrResult = emit(gctx, bc);
				break;
			}
			const auto *info = gctx.ctx.lookupStruct(baseTy);
			if (info == nullptr) {
				failHere(gctx, "astgen: lvalue field access on non-struct");
			}
			int idx = gctx.ctx.getFieldIndex(info->name, memberName);
			if (idx < 0) {
				failHere(gctx, "astgen: unknown field `" + memberName +
				                   "` on `" + info->name + "`");
			}
			leafTy = resolveDeferredArrayTy(gctx, info->fields[idx].second);
			TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, leafTy, 0});
			JirInst fa{};
			fa.tag = JirTag::FieldAddr;
			fa.a = basePtr;
			fa.b = static_cast<uint32_t>(idx);
			fa.ty = ptrTy;
			ptrResult = emit(gctx, fa);
			break;
		}
		case AstTag::Index: {
			TypeIdx baseTy = kNoType;
			JirRef basePtr = astgenExpr(gctx, static_cast<NodeIdx>(n.lhs),
			                            kNoType, ResultLoc::Pointer, &baseTy);
			NodeIdx idxIdxN = static_cast<NodeIdx>(n.rhs);
			JirRef idxRef = astgenExpr(gctx, idxIdxN, BuiltinType::U64);
			const TypeKey &lk = gctx.ctx.getTypePool().get(baseTy);
			TypeIdx elemTy = kNoType;
			if (lk.kind == TypeKind::Array || lk.kind == TypeKind::Slice ||
			    lk.kind == TypeKind::PtrMany) {
				elemTy = static_cast<TypeIdx>(lk.a);
			} else {
				// A struct base usually means the `v[i]` sugar's `at`
				// was withdrawn for this instantiation — replay why.
				if (const auto *sb = gctx.ctx.lookupStruct(baseTy)) {
					if (gctx.ctx.getWithdrawnMethod(sb->name + ".at") !=
					    nullptr) {
						return reportMethodMiss(gctx, sb->name + ".at");
					}
				}
				failHere(gctx,
				         "astgen: lvalue index on non-array/slice/ptr-many");
			}
			// PtrMany base: the alloca holds the pointer value
			// itself; Load to follow it. Array base: alloca holds
			// inline storage; GEP directly. Slice base: load the
			// {ptr,len} value, ExtractValue .ptr, GEP that.
			if (lk.kind == TypeKind::PtrMany) {
				JirInst load{};
				load.tag = JirTag::Load;
				load.a = basePtr;
				load.ty = baseTy;
				basePtr = emit(gctx, load);
			} else if (lk.kind == TypeKind::Slice) {
				JirInst load{};
				load.tag = JirTag::Load;
				load.a = basePtr;
				load.ty = baseTy;
				JirRef sliceVal = emit(gctx, load);
				TypeIdx ptrManyTy = gctx.ctx.getTypePool().intern(
				    TypeKey{TypeKind::PtrMany, 0, 0, elemTy, 0});
				JirInst ev{};
				ev.tag = JirTag::ExtractValue;
				ev.a = sliceVal;
				ev.b = 0;
				ev.ty = ptrManyTy;
				basePtr = emit(gctx, ev);
			}
			leafTy = elemTy;
			TypeIdx ptrTy = gctx.ctx.getTypePool().intern(
			    TypeKey{TypeKind::PtrSingle, 0, 0, elemTy, 0});
			JirInst ia{};
			ia.tag = JirTag::IndexAddr;
			ia.a = basePtr;
			ia.b = idxRef;
			ia.ty = ptrTy;
			ptrResult = emit(gctx, ia);
			break;
		}
		default:
			handled = false;
			break;
		}
		if (handled) {
			if (outLeafTy) *outLeafTy = leafTy;
			return ptrResult;
		}
		// Fall through to value-then-spill below for non-lvalue tags.
	}

	JirRef result = kNoJirRef;
	switch (n.tag) {
	case AstTag::NumberLit:
		result = astgenNumberLit(gctx, n, expected);
		break;
	case AstTag::BoolLit:
		result = astgenBoolLit(gctx, n);
		break;
	case AstTag::StringLit:
		result = astgenStringLit(gctx, n, expected);
		break;
	case AstTag::Return:
		astgenReturn(gctx, n);
		return kNoJirRef;
	case AstTag::VarDecl:
		astgenVarDecl(gctx, n);
		return kNoJirRef;
	case AstTag::Variable:
		result = astgenVariable(gctx, n, expected);
		break;
	case AstTag::Assign:
		astgenAssign(gctx, n);
		return kNoJirRef;
	case AstTag::UnaryOp:
		result = astgenUnaryOp(gctx, n, expected);
		break;
	case AstTag::AsCast:
		result = astgenAsCast(gctx, n);
		break;
	case AstTag::StructLit:
		result = astgenStructLit(gctx, n, expected);
		break;
	case AstTag::MemberAccess:
		result = astgenMemberAccess(gctx, n);
		break;
	case AstTag::ArrayLit:
		result = astgenArrayLit(gctx, n, expected);
		break;
	case AstTag::ArrayRepeat:
		result = astgenArrayRepeat(gctx, n, expected);
		break;
	case AstTag::Index:
		result = astgenIndex(gctx, n);
		break;
	case AstTag::Slice:
		result = astgenSlice(gctx, n);
		break;
	case AstTag::AddressOf:
		result = astgenAddressOf(gctx, n);
		break;
	case AstTag::Deref:
		result = astgenDeref(gctx, n);
		break;
	case AstTag::MatchNode:
		result = astgenMatch(gctx, n, expected);
		if (result == kNoJirRef) return kNoJirRef;
		break;
	case AstTag::BinaryOp:
		result = astgenBinaryOp(gctx, n, expected);
		break;
	case AstTag::Call:
		result = astgenCall(gctx, n);
		// A void-returning call has no JirRef value to chain.
		if (result != kNoJirRef && gctx.jfn.getInst(result).ty == kNoType) {
			return kNoJirRef;
		}
		break;
	case AstTag::TypeMethodCall:
		result = astgenTypeMethodCall(gctx, n);
		if (result != kNoJirRef && gctx.jfn.getInst(result).ty == kNoType) {
			return kNoJirRef;
		}
		break;
	case AstTag::AtCall:
		result = astgenAtCall(gctx, n);
		break;
	case AstTag::IfNode:
		astgenIf(gctx, n);
		return kNoJirRef;
	case AstTag::WhileNode:
		astgenWhile(gctx, n);
		return kNoJirRef;
	case AstTag::ForNode:
		astgenFor(gctx, n);
		return kNoJirRef;
	case AstTag::Break:
		astgenBreak(gctx);
		return kNoJirRef;
	case AstTag::Continue:
		astgenContinue(gctx);
		return kNoJirRef;
	default:
		failHere(gctx, "astgen: unsupported AST node (tag = " +
		                   std::to_string(static_cast<int>(n.tag)) + ")");
	}
	if (result != kNoJirRef && line > 0) {
		gctx.jfn.getInstMut(result).srcLine = static_cast<uint32_t>(line);
	}
	// Pointer-loc spill: non-lvalue nodes (Call results, BinaryOp,
	// literals, etc.) fell through to the value-producing switch
	// above. To honor `loc == Pointer`, materialize a temporary
	// alloca, store the value into it, and return its address. The
	// pointer lives for the surrounding expression — same shape
	// `astgenAddressOf` uses for its non-lvalue branch.
	if (loc == ResultLoc::Pointer && result != kNoJirRef) {
		TypeIdx leafTy = gctx.jfn.getInst(result).ty;
		JirInst alloca{};
		alloca.tag = JirTag::Alloca;
		alloca.ty = leafTy;
		JirRef slot = emitAllocaHoisted(gctx, alloca);
		JirInst store{};
		store.tag = JirTag::Store;
		store.a = slot;
		store.b = result;
		emit(gctx, store);
		if (outLeafTy) *outLeafTy = leafTy;
		return slot;
	}
	return result;
}

}  // namespace

// Global bridge for the header-declared symbol: the implementation
// lives in this file's anonymous namespace alongside the rest of the
// drop machinery.
bool typeNeedsDrop(JamCodegenContext &ctx, TypeIdx ty) {
	return typeNeedsDropInner(ctx, ty);
}

JirFunction astgenMetadata(const FunctionAST &fn, JamCodegenContext &ctx) {
	(void)ctx;
	JirFunction jfn;
	jfn.name = fn.Name;
	jfn.returnType = fn.ReturnType;
	jfn.isExtern = fn.isExtern;
	jfn.isExport = fn.isExport;
	jfn.isPub = fn.isPub;
	jfn.isTest = fn.isTest;
	jfn.isVarArgs = fn.isVarArgs;
	for (const Param &p : fn.Args) {
		jfn.paramTypes.push_back(p.Type);
		jfn.paramModes.push_back(p.Mode);
	}
	return jfn;
}

// Append the body of `fn` to an already-populated metadata JirFunction
// (`jfn` arg). Used by generic instantiation to avoid redoing the
// param-type / by-ptr work between Pass 1 (declare-prototype) and
// Pass 2 (define-body). For a from-scratch call, see `astgenFunction`.
void astgenBodyInto(JirFunction &jfn, const FunctionAST &fn,
                    JamCodegenContext &ctx);

JirFunction astgenFunction(const FunctionAST &fn, JamCodegenContext &ctx) {
	JirFunction jfn = astgenMetadata(fn, ctx);
	if (fn.isExtern) return jfn;
	astgenBodyInto(jfn, fn, ctx);
	return jfn;
}

void astgenBodyInto(JirFunction &jfn, const FunctionAST &fn,
                    JamCodegenContext &ctx) {
	if (fn.isExtern) return;

	JirBlockRef entry = jfn.pushBlock("entry");
	AstGenCtx gctx{jfn, ctx, entry, {}, {}};
	// Scope 0 is the function body itself — every local declared at
	// the function top level drops here at function exit.
	pushDropScope(gctx);

	// A deferred `[expr]T` return annotation canonicalizes before any
	// return statement compares its value type against it. Runs before
	// jirDeclarePrototype reads jfn.returnType, so the LLVM signature
	// sees the resolved length too.
	jfn.returnType = resolveDeferredArrayTy(gctx, jfn.returnType);

	// Lower each parameter. The ABI classifier is the single source of
	// truth — both the prototype emitter and the call-site argument
	// lowering ask the same `classifyParam(mode, type)` question, so
	// the LLVM signature and the call argument types always agree.
	//   - ByValue: param arrives as a value. Alloca + store + register
	//     the alloca as the local so reads emit Load(alloca) like any
	//     ordinary local.
	//   - ByPointer: param arrives as a pointer to caller-owned
	//     storage (mut / move always; let / const for any byref
	//     aggregate). Register the param JirRef directly as
	//     the local's "alloca" so reads, writes, and field access all
	//     operate on that pointer. JIR flag bit 0 on Param tells
	//     jir_codegen + FieldAddr/IndexAddr to treat the value as a
	//     pointer-to-pointee rather than a by-value Param.
	for (size_t i = 0; i < fn.Args.size(); i++) {
		const Param &p = fn.Args[i];
		// Deferred `[expr]T` param annotations canonicalize before the
		// type is stamped on the Param inst / local binding.
		TypeIdx pTy = resolveDeferredArrayTy(gctx, p.Type);
		jam::abi::ParamABI pabi = jam::abi::classifyParam(p.Mode, pTy, ctx);
		bool byPtr = pabi.kind == jam::abi::ParamABI::Kind::ByPointer;
		JirInst paramInst{};
		paramInst.tag = JirTag::Param;
		paramInst.a = static_cast<uint32_t>(i);
		paramInst.ty = pTy;
		if (byPtr) paramInst.flags |= 1;
		JirRef paramRef = emit(gctx, paramInst);

		JirRef slotRef = paramRef;
		if (byPtr) {
			gctx.locals[p.Name] = paramRef;
			gctx.localTypes[p.Name] = pTy;
		} else {
			JirInst alloca{};
			alloca.tag = JirTag::Alloca;
			alloca.ty = pTy;
			JirRef allocaRef = emitAllocaHoisted(gctx, alloca);
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = allocaRef;
			store.b = paramRef;
			emit(gctx, store);
			gctx.locals[p.Name] = allocaRef;
			gctx.localTypes[p.Name] = pTy;
			slotRef = allocaRef;
		}

		// A `move` parameter is OWNED by the callee: it drops at
		// function exit unless the body moves it onward (a store, a
		// struct-literal capture, a further `move` call, a return) —
		// each of those sites consumes the track. `let`/`mut` params
		// stay caller-owned and never drop here. Params register
		// before any body local, so emitDrops' reverse walk drops
		// locals first, params last.
		if (p.Mode == ParamMode::Move) {
			std::string dropName = lookupDropFnLLVMName(gctx.ctx, pTy);
			if (!dropName.empty()) {
				gctx.dropScopes.back().push_back(
				    {p.Name, slotRef, pTy, dropName});
			} else if (typeNeedsDrop(gctx.ctx, pTy)) {
				gctx.dropScopes.back().push_back(
				    {p.Name, slotRef, pTy, std::string()});
			}
		}
	}

	for (NodeIdx stmt : fn.Body) { astgenExpr(gctx, stmt, kNoType); }

	// Implicit fall-through terminator. Three cases for a tail block
	// without an explicit terminator:
	//   1. Reachable from entry AND the function returns void: emit
	//      drops + Ret(void). This is the common case for a no-tail
	//      void body.
	//   2. Reachable from entry AND the function returns a value or
	//      is `noreturn`: that's a real bug (a path reaches the end
	//      without returning), so we throw.
	//   3. Unreachable (zero predecessors and not the entry block):
	//      dead post-merge / post-loop block that came up because
	//      every arm / iteration diverged. Emit `Unreachable` to
	//      give it a terminator without spuriously erroring.
	if (!blockHasTerminator(jfn.getBlock(gctx.currentBlock), jfn)) {
		bool isEntry = gctx.currentBlock == /*entry=*/1;
		bool reachable =
		    isEntry || predecessorCount(jfn, gctx.currentBlock) > 0;
		if (!reachable) {
			JirInst u{};
			u.tag = JirTag::Unreachable;
			emit(gctx, u);
		} else if (fn.ReturnType == BuiltinType::NoReturn) {
			failHere(gctx, "fn `" + fn.Name +
			                   "` is declared `noreturn` but its "
			                   "body falls through without diverging");
		} else if (fn.ReturnType != kNoType) {
			failHere(
			    gctx,
			    "fn `" + fn.Name +
			        "` has non-void return type but a "
			        "path reaches the function end without returning a value");
		} else {
			emitDropsThroughScope(gctx, 0);
			JirInst ret{};
			ret.tag = JirTag::Ret;
			emit(gctx, ret);
		}
	}
}
