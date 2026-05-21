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
#include "mangling.h"

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
	JirRef slot;             // alloca for the variable
	TypeIdx type;            // source-level type
	std::string llvmFnName;  // canonical drop fn (legacy mangles to `__drop_T`)
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
static JirRef emitCall(AstGenCtx &gctx, const FunctionAST *fn,
                       const std::vector<JirRef> &argRefs);
// `v[i]` desugar dispatch — see `emitStructCfnDispatch` for the body.
// Forward-declared so astgenAssign (in this file, above the
// definition) can call it for `v[i] = x` → setAt routing.
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
		inst.tag = JirTag::Float;
		inst.a = static_cast<uint32_t>(val & 0xFFFFFFFFu);
		inst.b = static_cast<uint32_t>(val >> 32);
		if (isNeg) inst.flags |= 1;  // sign bit applied at codegen
		inst.ty = (expected != kNoType && expected == BuiltinType::F32)
		              ? BuiltinType::F32
		              : BuiltinType::F64;
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
	// `Identity(i32)` → i32), resolve it first so the contract holds
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

// AstGen for `StringLit`. The result type is a slice-of-u8 (`[]u8`);
// the slice TypeKey is interned in the TypePool on first use so a
// stable TypeIdx flows through the IR. The actual {ptr,len} struct
// is materialised at jir-codegen time using the StringIdx in `a`.
static JirRef astgenStringLit(AstGenCtx &gctx, const AstNode &n) {
	StringIdx s = static_cast<StringIdx>(n.lhs);
	TypeIdx sliceTy = gctx.ctx.getTypePool().intern(
	    TypeKey{TypeKind::Slice, 0, 0, BuiltinType::U8, 0});
	JirInst inst{};
	inst.tag = JirTag::Str;
	inst.a = s;
	inst.ty = sliceTy;
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
static void astgenReturn(AstGenCtx &gctx, const AstNode &n) {
	JirRef valRef = kNoJirRef;
	if (n.lhs != 0) {
		valRef =
		    astgenExpr(gctx, static_cast<NodeIdx>(n.lhs), gctx.jfn.returnType);
	}
	// Drop every active scope before exiting the function.
	emitDropsThroughScope(gctx, 0);
	JirInst ret{};
	ret.tag = JirTag::Ret;
	ret.a = valRef;
	emit(gctx, ret);
}

// Resolve the canonical drop function name for a Jam type. Returns
// empty when the type has no drop fn registered. For struct/named
// types, legacy `mangledFunctionName` produces `__drop_<TypeName>`
// at the LLVM level; we mirror that here so the call resolves at
// codegen time even though the FunctionAST::Name is bare "drop".
// Falls back to the codegen context's instantiated-drops table so
// generic struct/enum instantiations (Vec(i32), Holder(i32), ...)
// fire drops too.
static std::string lookupDropFnLLVMName(AstGenCtx &gctx, TypeIdx ty) {
	const TypeKey &k = gctx.ctx.getTypePool().get(ty);
	std::string typeName;
	if (k.kind == TypeKind::Struct || k.kind == TypeKind::Named ||
	    k.kind == TypeKind::Enum) {
		typeName = gctx.ctx.getStringPool().get(static_cast<StringIdx>(k.a));
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
		const jam::drops::DropRegistry *reg = gctx.ctx.getDropRegistry();
		if (reg != nullptr) {
			auto it = reg->find(name);
			if (it != reg->end()) fn = it->second;
		}
		if (fn == nullptr) fn = gctx.ctx.lookupDropFn(name);
		if (fn == nullptr) return "";
		return mangledFunctionName(*fn, gctx.ctx.getTypePool(),
		                           gctx.ctx.getStringPool());
	};
	std::string r = resolveName(typeName);
	if (!r.empty()) return r;
	TypeIdx aliasTarget = gctx.ctx.lookupTypeAlias(typeName);
	if (aliasTarget != kNoType) {
		const TypeKey &ak0 = gctx.ctx.getTypePool().get(aliasTarget);
		if (ak0.kind == TypeKind::GenericCall) {
			TypeIdx resolved = gctx.ctx.resolveGenericCall(aliasTarget);
			if (resolved != kNoType) aliasTarget = resolved;
		}
		const TypeKey &ak = gctx.ctx.getTypePool().get(aliasTarget);
		if (ak.kind == TypeKind::Named || ak.kind == TypeKind::Struct ||
		    ak.kind == TypeKind::Enum) {
			std::string aliasName =
			    gctx.ctx.getStringPool().get(static_cast<StringIdx>(ak.a));
			r = resolveName(aliasName);
			if (!r.empty()) return r;
		}
	}
	return "";
}

// Emit drop calls for every binding in `bindings` in REVERSE order
// (LIFO — last-pushed dropped first). The argument is &binding (the
// alloca pointer itself, since `fn drop(self: mut T)` takes a *T).
static void emitDrops(AstGenCtx &gctx, const std::vector<DropTrack> &bindings) {
	// Emit explicit DropBinding JIR instructions in reverse declaration
	// order. Each carries the binding's alloca JirRef and the LLVM
	// symbol of the drop fn as a StringIdx. Codegen mechanically
	// lowers to `call void <symbol>(ptr <alloca>)` — no per-call
	// AddrOf+pack ceremony, no metadata fallback.
	for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
		const DropTrack &d = *it;
		StringIdx symId = gctx.ctx.getStringPool().intern(d.llvmFnName);
		JirInst drop{};
		drop.tag = JirTag::DropBinding;
		drop.a = d.slot;
		drop.b = symId;
		drop.ty = kNoType;
		emit(gctx, drop);
	}
}

// AstGen for `VarDecl`. Lowers to Alloca + Store(init). The variable's
// name → alloca ref binding is recorded so subsequent reads emit
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
		type = declared;
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
				const std::string &name =
				    gctx.ctx.getStringPool().get(static_cast<StringIdx>(k.a));
				TypeIdx sub = gctx.ctx.lookupCurrentSubst(name);
				if (sub != kNoType) return resolveForCmp(sub);
			}
			if (k.kind == TypeKind::GenericCall) {
				TypeIdx r = gctx.ctx.resolveGenericCall(t);
				if (r != kNoType) return resolveForCmp(r);
			}
			if (k.kind == TypeKind::Named) {
				TypeIdx a = gctx.ctx.lookupTypeAlias(
				    gctx.ctx.getStringPool().get(static_cast<StringIdx>(k.a)));
				if (a != kNoType) return resolveForCmp(a);
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
			bool aPtr =
			    ka.kind == TypeKind::PtrSingle || ka.kind == TypeKind::PtrMany;
			bool bPtr =
			    kb.kind == TypeKind::PtrSingle || kb.kind == TypeKind::PtrMany;
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
				failHere(gctx, "cannot assign integer to float-typed `" + name +
				                   "`; use a float literal (e.g. `3.0`) "
				                   "or an explicit `as` cast");
			}
			failHere(gctx, "type mismatch in `" + name +
			                   "`: declared and initialised values disagree");
		}
	}

	JirInst store{};
	store.tag = JirTag::Store;
	store.a = allocaRef;
	store.b = initRef;
	emit(gctx, store);

	// If this binding's type has a registered drop fn, track it on
	// the top of the drop-scope stack so the next scope-exit (or
	// the function-end Return) emits drop(self=&binding).
	std::string dropName = lookupDropFnLLVMName(gctx, type);
	if (!dropName.empty()) {
		if (gctx.dropScopes.empty()) pushDropScope(gctx);
		gctx.dropScopes.back().push_back({name, allocaRef, type, dropName});
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
			return recoverHere(gctx,
			                   "cannot take address of generic fn `" + name +
			                       "`",
			                   kNoType);
		}
		JirInst fnref{};
		fnref.tag = JirTag::FnRef;
		fnref.a = static_cast<JirRef>(
		    gctx.ctx.getStringPool().intern(fn->Name));
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
static JirRef astgenLvalue(AstGenCtx &gctx, NodeIdx node,
                           TypeIdx &outLeafTy) {
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

	// `v[i] = x` on a struct → `v.setAt(i, x)`.
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
		const auto *sinfo = baseTy != kNoType
		                        ? gctx.ctx.lookupStruct(baseTy)
		                        : nullptr;
		if (sinfo != nullptr) {
			const std::string qualified = sinfo->name + ".setAt";
			const FunctionAST *method = gctx.ctx.getFunctionAST(qualified);
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
				return;
			}
		}
		// Not a struct, or no `cfn setAt` defined — fall through.
	}

	TypeIdx leafTy = kNoType;
	JirRef ptrRef = astgenLvalue(gctx, targetIdx, leafTy);
	JirRef valRef = astgenExpr(gctx, valueIdx, leafTy);
	JirInst store{};
	store.tag = JirTag::Store;
	store.a = ptrRef;
	store.b = valRef;
	emit(gctx, store);
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
		TypeIdx expectedField = info->fields[idx].second;
		JirRef fieldVal = astgenExpr(gctx, exprIdx, expectedField);
		// Silent int→float widening matches the legacy struct codegen:
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
	inst.ty = info->fields[idx].second;
	return emit(gctx, inst);
}

// AstGen for `ArrayLit`. SSA array value built via a chain of
// InsertValue at codegen time. The target element type (`n.lhs`) may
// be unset; in that case we infer it from the first element.
static JirRef astgenArrayLit(AstGenCtx &gctx, const AstNode &n,
                             TypeIdx expected) {
	const NodeStore &ns = gctx.ctx.getNodeStore();
	TypeIdx elemTy = static_cast<TypeIdx>(n.lhs);
	if (elemTy == kNoType && expected != kNoType) {
		const TypeKey &ek = gctx.ctx.getTypePool().get(expected);
		if (ek.kind == TypeKind::Array) elemTy = static_cast<TypeIdx>(ek.a);
	}
	ExtraIdx elemsExtra = static_cast<ExtraIdx>(n.rhs);
	uint32_t count = ns.getExtra(elemsExtra);

	std::vector<JirRef> elems;
	elems.reserve(count);
	for (uint32_t i = 0; i < count; i++) {
		NodeIdx e = static_cast<NodeIdx>(ns.getExtra(elemsExtra + 1 + i));
		elems.push_back(astgenExpr(gctx, e, elemTy));
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
	ExtraIdx extra = static_cast<ExtraIdx>(n.rhs);
	NodeIdx valueIdx = static_cast<NodeIdx>(ns.getExtra(extra));
	NodeIdx countIdx = static_cast<NodeIdx>(ns.getExtra(extra + 1));

	const AstNode &cn = ns.get(countIdx);
	if (cn.tag != AstTag::NumberLit) {
		failHere(
		    gctx,
		    "astgen: array-repeat count must be a constant integer literal");
	}
	uint64_t count =
	    static_cast<uint64_t>(cn.lhs) | (static_cast<uint64_t>(cn.rhs) << 32);

	// Resolve element type. From the array TypeKey if we have it; else
	// from the first compile of the value.
	TypeIdx elemTy = kNoType;
	if (arrTy != kNoType) {
		const TypeKey &k = gctx.ctx.getTypePool().get(arrTy);
		if (k.kind == TypeKind::Array) elemTy = static_cast<TypeIdx>(k.a);
	}
	JirRef val = astgenExpr(gctx, valueIdx, elemTy);
	if (elemTy == kNoType) elemTy = gctx.jfn.getInst(val).ty;
	if (arrTy == kNoType) {
		arrTy = gctx.ctx.getTypePool().intern(TypeKey{
		    TypeKind::Array, 0, 0, elemTy, static_cast<uint32_t>(count)});
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

	// Array Variable fast path. Take the base as a pointer (zero-
	// cost for a Variable — just the alloca's own JirRef), emit
	// IndexAddr to the element, then Load just that element. Avoids
	// the rvalue-Load-then-spill pattern JirTag::Index uses on
	// Array SSA values, which loads the full backing storage (up to
	// several KB) per access and stalls LLVM at -O1+.
	//
	// Limited to Variable bases on purpose — astgenExpr(.., Pointer)
	// on a non-struct MemberAccess (e.g. `slice.ptr`) routes
	// through astgenLvalue's MemberAccess branch, which errors on
	// non-struct parents. The rvalue path handles slices / ptr-many
	// correctly anyway (small Load + GEP).
	{
		const AstNode &baseNode = gctx.ctx.getNodeStore().get(baseIdx);
		if (baseNode.tag == AstTag::Variable) {
			JirRef basePtr =
			    astgenExpr(gctx, baseIdx, kNoType, ResultLoc::Pointer);
			TypeIdx leafTy = gctx.jfn.getInst(basePtr).ty;
			const TypeKey &lk = gctx.ctx.getTypePool().get(leafTy);
			if (lk.kind == TypeKind::Array) {
				TypeIdx elemTy = static_cast<TypeIdx>(lk.a);
				JirRef idxRef =
				    astgenExpr(gctx, idxIdx, BuiltinType::U64);
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
			// Non-Array Variable: fall through. basePtr is the
			// existing alloca's JirRef, no fresh JIR emitted.
		}
	}

	JirRef baseRef = astgenExpr(gctx, baseIdx, kNoType);
	JirRef idxRef = astgenExpr(gctx, idxIdx, BuiltinType::U64);
	TypeIdx baseTy = gctx.jfn.getInst(baseRef).ty;
	const TypeKey &k = gctx.ctx.getTypePool().get(baseTy);

	// Struct dispatch: `v[i]` → `v.at(i)`. The `at` method is value-
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
	// Resolve GenericCall destinations (e.g. `Identity(u8)` → u8) so
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
	TypeIdx hint = (dst.kind == TypeKind::Int) ? dstTy : kNoType;
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

	// Bool-to-integer: i1 → integer of any width.
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
	// a pointer on every supported target, so restrict to that width.
	// `myPtr as u64` or `addr as *mut[] u8`. Stays out of the int↔int
	// path below because that one issues SExt/Trunc instead.
	if (isPtr(src) && dst.kind == TypeKind::Int && dst.a == 64) {
		return emitCast(JirTag::PtrToInt);
	}
	if (src.kind == TypeKind::Int && src.a == 64 && isPtr(dst)) {
		return emitCast(JirTag::IntToPtr);
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
//   `-x`  (Neg)     → 0 - x  for ints, FNeg for floats
//   `!x`  (LogNot)  → x == false (xor 1)
//   `~x`  (BitNot)  → x XOR all-ones
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

	JirRef startRef = astgenExpr(gctx, startIdx, kNoType);
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

// =====================================================================
// Match lowering (Phase 5)
// =====================================================================
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
//   * integer scrutinee → only PatLit accepted; value = literal bits.
//   * enum scrutinee   → only PatEnumVariant accepted; value = the
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
		if (!scrutIsEnum || einfo == nullptr) return false;
		// Reject pattern shapes that need a binding block before the
		// arm body — Switch can only jump straight to `armBlock`.
		bool hasBindings = (p.flags & 1) != 0;
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
		//   flags & 1: bindings present, lhs = ExtraIdx → [recv, variant,
		//              count, name0, name1, ...]; else lhs is the receiver
		//   flags & 2: receiver is a TypeIdx (GenericCall, resolved via
		//              lookupEnum); else it's a StringIdx
		bool hasBindings = (p.flags & 1) != 0;
		bool typeIdxReceiver = (p.flags & 2) != 0;
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
		if (typeIdxReceiver) {
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
		// at its default).
		if (wildcardArmIdx < 0) {
			gctx.currentBlock = defaultB;
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
// into astgenCall. Instantiation runs the full astgen → jirDefineBody
// pipeline in `JamCodegenContext::instantiateStructExpr`, so the JIR
// Call here resolves cleanly by LLVM name at codegen time.
static JirRef astgenTypeMethodCall(AstGenCtx &gctx, const AstNode &n) {
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
		return recoverHere(gctx, "unknown method `" + qualified + "`", kNoType);
	}
	std::vector<JirRef> argRefs;
	argRefs.reserve(argCount);
	for (uint32_t i = 0; i < argCount; i++) {
		NodeIdx argIdx = static_cast<NodeIdx>(ns.getExtra(callExtra + 1 + i));
		TypeIdx expectArg = (i < fn->Args.size()) ? fn->Args[i].Type : kNoType;
		argRefs.push_back(astgenExpr(gctx, argIdx, expectArg));
	}
	return emitCall(gctx, fn, argRefs);
}

// AstGen for `AtCall` — comptime intrinsic (`@sizeOf(T)` / `@alignOf(T)`).
// Resolves to a constant at astgen time; the JIR never sees a call.
static JirRef astgenAtCall(AstGenCtx &gctx, const AstNode &n) {
	const std::string &name =
	    gctx.ctx.getStringPool().get(static_cast<StringIdx>(n.lhs));
	TypeIdx tyArg = static_cast<TypeIdx>(n.rhs);
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

	JirInst cmp{};
	cmp.tag = JirTag::ICmpEq;
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
static JirRef lowerArg(AstGenCtx &gctx, NodeIdx argIdx, const Param &p) {
	jam::abi::ParamABI pabi = jam::abi::classifyParam(p.Mode, p.Type, gctx.ctx);
	if (pabi.kind != jam::abi::ParamABI::Kind::ByPointer) {
		return astgenExpr(gctx, argIdx, p.Type);
	}
	// ByPointer: feed an address. Mut / Move want the caller's
	// storage when the arg is an lvalue (so writes are observed by the
	// caller). Let / Const reach here only for large aggregates — the
	// callee can't mutate them through the pointer (no `mut`), but the
	// LLVM signature wants a pointer because the value is too big to
	// pass in registers. In both cases, when the arg isn't already an
	// lvalue we spill it to a fresh alloca.
	const AstNode &argNode = gctx.ctx.getNodeStore().get(argIdx);
	if (p.Mode == ParamMode::Mut || p.Mode == ParamMode::Move) {
		TypeIdx leafTy = kNoType;
		switch (argNode.tag) {
		case AstTag::Variable:
		case AstTag::MemberAccess:
		case AstTag::Index:
		case AstTag::Deref:
			return astgenLvalue(gctx, argIdx, leafTy);
		case AstTag::AddressOf:
			return astgenExpr(gctx, argIdx, kNoType);
		default:
			break;
		}
	}
	JirRef val = astgenExpr(gctx, argIdx, p.Type);
	TypeIdx leafTy = gctx.jfn.getInst(val).ty;
	JirInst alloca{};
	alloca.tag = JirTag::Alloca;
	alloca.ty = leafTy;
	JirRef ptr = emitAllocaHoisted(gctx, alloca);
	JirInst store{};
	store.tag = JirTag::Store;
	store.a = ptr;
	store.b = val;
	emit(gctx, store);
	return ptr;
}

static JirRef emitCall(AstGenCtx &gctx, const FunctionAST *fn,
                       const std::vector<JirRef> &argRefs) {
	std::string mangled = mangledFunctionName(*fn, gctx.ctx.getTypePool(),
	                                          gctx.ctx.getStringPool());
	StringIdx calleeId = gctx.ctx.getStringPool().intern(mangled);
	std::vector<uint32_t> packed;
	packed.reserve(1 + argRefs.size());
	packed.push_back(static_cast<uint32_t>(argRefs.size()));
	for (JirRef r : argRefs) packed.push_back(r);
	JirExtraIdx extra = gctx.jfn.pushExtra(packed.data(), packed.size());
	JirInst call{};
	call.tag = JirTag::Call;
	call.a = calleeId;
	call.b = extra;
	call.ty = fn->ReturnType;
	JirRef callRef = emit(gctx, call);
	// Calling a `noreturn` function diverges. Terminate the current
	// block with Unreachable so downstream code is dead and the JIR
	// is well-formed (every reachable block ends in a terminator).
	// Mirrors what the legacy codegen did via LLVM's `unreachable`.
	if (fn->ReturnType == BuiltinType::NoReturn) {
		JirInst u{};
		u.tag = JirTag::Unreachable;
		emit(gctx, u);
	}
	return callRef;
}

static JirRef astgenCall(AstGenCtx &gctx, const AstNode &n) {
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
		if (recvIsLvalueable && (methodName == "asPtr" ||
		                         methodName == "asMutPtr")) {
			JirRef basePtr = astgenExpr(gctx, recvExprIdx, kNoType,
			                             ResultLoc::Pointer);
			TypeIdx leafTy = gctx.jfn.getInst(basePtr).ty;
			const TypeKey &lk = gctx.ctx.getTypePool().get(leafTy);
			if (lk.kind == TypeKind::Array) {
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

		// Resolve the receiver as a value for struct / enum dispatch.
		// For mut/move dispatch we'll redo this as an lvalue below —
		// passing the original storage pointer instead of a spilled
		// copy.
		JirRef recvVal = astgenExpr(gctx, recvExprIdx, kNoType);
		TypeIdx recvTy = gctx.jfn.getInst(recvVal).ty;
		{
			const TypeKey &recvKey = gctx.ctx.getTypePool().get(recvTy);
			if (recvKey.kind == TypeKind::Array) {
				// Non-lvalueable array receiver (e.g. a call result):
				// the rvalue-Load path got us here. Without a stable
				// storage location, we can't safely hand a pointer
				// to FFI — reject.
				failHere(gctx, "astgen: `" + methodName +
				                   "()` requires an addressable array "
				                   "(variable, field, or indexed slot)");
			}
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
			return recoverHere(gctx, "unknown method `" + qualified + "`",
			                   kNoType);
		}
		ParamMode mode =
		    method->Args.empty() ? ParamMode::Let : method->Args[0].Mode;
		JirRef recvArg = recvVal;
		if (mode == ParamMode::Mut || mode == ParamMode::Move) {
			// Re-lower the receiver as an lvalue so the method sees
			// the original storage. For non-lvalue rvalues we spill
			// the already-computed value to a fresh alloca.
			const AstNode &recvNode = ns.get(recvExprIdx);
			TypeIdx leafTy = kNoType;
			JirRef ptr;
			switch (recvNode.tag) {
			case AstTag::Variable:
			case AstTag::MemberAccess:
			case AstTag::Index:
			case AstTag::Deref:
				ptr = astgenLvalue(gctx, recvExprIdx, leafTy);
				break;
			default: {
				JirInst alloca{};
				alloca.tag = JirTag::Alloca;
				alloca.ty = recvTy;
				ptr = emitAllocaHoisted(gctx, alloca);
				JirInst store{};
				store.tag = JirTag::Store;
				store.a = ptr;
				store.b = recvVal;
				emit(gctx, store);
				break;
			}
			}
			recvArg = ptr;
		}
		std::vector<JirRef> argRefs;
		argRefs.reserve(1 + argCount);
		argRefs.push_back(recvArg);
		for (uint32_t i = 0; i < argCount; i++) {
			NodeIdx argIdx =
			    static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
			TypeIdx expectArg = (1 + i < method->Args.size())
			                        ? method->Args[1 + i].Type
			                        : kNoType;
			argRefs.push_back(astgenExpr(gctx, argIdx, expectArg));
		}
		return emitCall(gctx, method, argRefs);
	}

	StringIdx calleeId = static_cast<StringIdx>(n.lhs);
	std::string callee = gctx.ctx.getStringPool().get(calleeId);

	// Builtin: `assert(actual, expected)` — handled via JIR inline,
	// not as a regular function call.
	if (callee == "assert") { return astgenAssertCall(gctx, n); }

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
							JirInst gepInst{};
							gepInst.tag = JirTag::IndexAddr;
							gepInst.a = payAreaPtr;
							JirInst offC{};
							offC.tag = JirTag::Int;
							offC.a = static_cast<uint32_t>(off);
							offC.ty = BuiltinType::U64;
							JirRef offRef = emit(gctx, offC);
							gepInst.b = offRef;
							gepInst.ty = payAreaPtrTy;  // *u8 → byte stride
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
					return emitCall(gctx, method, argRefs);
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
						TypeIdx elemTy =
						    static_cast<TypeIdx>(instKey.a);
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
					if (method != nullptr && !method->Args.empty()) {
						// Build the receiver as either &inst (mut/move)
						// or a Load of inst (let/const).
						ParamMode mode = method->Args[0].Mode;
						JirRef recvRef;
						if (mode == ParamMode::Mut || mode == ParamMode::Move) {
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
							TypeIdx expectArg = (1 + i < method->Args.size())
							                        ? method->Args[1 + i].Type
							                        : kNoType;
							argRefs.push_back(
							    astgenExpr(gctx, argIdx, expectArg));
						}
						return emitCall(gctx, method, argRefs);
					}
				}
			}
		}
	}

	const FunctionAST *fn = gctx.ctx.getFunctionAST(callee);
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
						const TypeKey &fk =
						    gctx.ctx.getTypePool().get(fieldTy);
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
	return emitCall(gctx, fn, argRefs);
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
			JirRef innerPtr = astgenExpr(
			    gctx, static_cast<NodeIdx>(n.lhs), kNoType);
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
			JirRef basePtr =
			    astgenExpr(gctx, static_cast<NodeIdx>(n.lhs), kNoType,
			               ResultLoc::Pointer, &baseTy);
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
			leafTy = info->fields[idx].second;
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
			JirRef basePtr =
			    astgenExpr(gctx, static_cast<NodeIdx>(n.lhs), kNoType,
			               ResultLoc::Pointer, &baseTy);
			NodeIdx idxIdxN = static_cast<NodeIdx>(n.rhs);
			JirRef idxRef = astgenExpr(gctx, idxIdxN, BuiltinType::U64);
			const TypeKey &lk = gctx.ctx.getTypePool().get(baseTy);
			TypeIdx elemTy = kNoType;
			if (lk.kind == TypeKind::Array ||
			    lk.kind == TypeKind::Slice ||
			    lk.kind == TypeKind::PtrMany) {
				elemTy = static_cast<TypeIdx>(lk.a);
			} else {
				failHere(
				    gctx,
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
		result = astgenStringLit(gctx, n);
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

	// Lower each parameter. The ABI classifier is the single source of
	// truth — both the prototype emitter and the call-site argument
	// lowering ask the same `classifyParam(mode, type)` question, so
	// the LLVM signature and the call argument types always agree.
	//   - ByValue: param arrives as a value. Alloca + store + register
	//     the alloca as the local so reads emit Load(alloca) like any
	//     ordinary local.
	//   - ByPointer: param arrives as a pointer to caller-owned
	//     storage (mut / move always; let / const for aggregates >
	//     kByValueMaxBytes). Register the param JirRef directly as
	//     the local's "alloca" so reads, writes, and field access all
	//     operate on that pointer. JIR flag bit 0 on Param tells
	//     jir_codegen + FieldAddr/IndexAddr to treat the value as a
	//     pointer-to-pointee rather than a by-value Param.
	for (size_t i = 0; i < fn.Args.size(); i++) {
		const Param &p = fn.Args[i];
		jam::abi::ParamABI pabi = jam::abi::classifyParam(p.Mode, p.Type, ctx);
		bool byPtr = pabi.kind == jam::abi::ParamABI::Kind::ByPointer;
		JirInst paramInst{};
		paramInst.tag = JirTag::Param;
		paramInst.a = static_cast<uint32_t>(i);
		paramInst.ty = p.Type;
		if (byPtr) paramInst.flags |= 1;
		JirRef paramRef = emit(gctx, paramInst);

		if (byPtr) {
			gctx.locals[p.Name] = paramRef;
			gctx.localTypes[p.Name] = p.Type;
		} else {
			JirInst alloca{};
			alloca.tag = JirTag::Alloca;
			alloca.ty = p.Type;
			JirRef allocaRef = emitAllocaHoisted(gctx, alloca);
			JirInst store{};
			store.tag = JirTag::Store;
			store.a = allocaRef;
			store.b = paramRef;
			emit(gctx, store);
			gctx.locals[p.Name] = allocaRef;
			gctx.localTypes[p.Name] = p.Type;
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
