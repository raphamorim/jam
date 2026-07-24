# `@callC` — plan

**Status: IMPLEMENTED** (phases 1–3; phase 4 remains future work). Shipped
with parser/astgen unit tests, `tests/unit/test_callc.jam`, and four
`tests/reject/reject_callc_*.jam` corpus files. One correction discovered
during implementation is folded into §1/§4: the aggregate guard admits
"≤ 2 full 64-bit-word leaves", not "≤ 16 bytes" — `div_t`'s `{i32,i32}`
miscompiled under the byte-size rule (LLVM splits per member; C packs per
register).

A variadic intrinsic that calls a raw C function address with a signature
synthesized from its arguments' static types:

```jam
// @callC(ReturnType, fnAddr, args...)
var d: f64    = @callC(f64,    msgSendAddr(), obj.value, sel("doubleValue").value);
var r: NSRect = @callC(NSRect, msgSendAddr(), win.value, sel("frame").value);
@callC(void, freeAddr, ptr as u64);          // void return: statement form
```

This is the compiler-side half of `../jam-objc/MSGSEND.md`: it replaces the
hand-written `Sig*` union / `rawSend*` table in `objc.jam` with one primitive.
Every new method signature then costs zero library code.

Zig 0.10.1 (`../zig-0.10.1`) is the reference implementation throughout; the
relevant lessons are folded into each section and collected in §7.

## 1. Semantics

`@callC(R, addr, a0, a1, …)`:

- `R` is a **type expression** (parsed with `parse_type`, like `@sizeOf`'s
  argument). Any sized type or `void`. Generic instantiations
  (`Option(u64)`-style) resolve the same way they do for `@sizeOf`.
- `addr` is a **`u64`** holding the function's address (`extern fn … as u64`,
  `objc.msgSendAddr()`, `dlsym` results). Not a `fn`-typed value — those
  already have call syntax; `@callC` exists for addresses whose signature the
  type system doesn't know.
- `a0…aN` are ordinary expressions. Each argument's **static type becomes the
  corresponding parameter type** of the synthesized signature — there is no
  coercion at the boundary. `-7 as i32` passes an `i32`; an `NSRect` local
  passes an `NSRect` by value, per the platform C ABI.
- The call is an indirect call with the **C calling convention** and the fixed
  synthesized signature `fn(a0T, a1T, …) R`. It is never variadic-ABI — that
  is the whole point (see MSGSEND.md).
- The expression's type is `R`; with `R = void` it produces no value.

Arity: at least 2 operands (`R`, `addr`). Zero call args is legal
(`@callC(u64, addr)` — e.g. `-init`).

**Argument-type restrictions** (compile errors, not silent miscompiles — §4):

- each arg type and non-void `R` must be sized and concrete (no `type`,
  `void`, `noreturn` args);
- aggregates (struct/union/array/payload enum) are allowed only in the shapes
  today's lowering provably gets right on AArch64 (§4): HFAs of ≤ 4 same-width
  floats, or aggregates of ≤ 2 leaves that are each a full 64-bit word
  (64-bit int / pointer). Anything else — sub-word fields like `{i32,i32}`,
  mixed float/int, > 16 bytes — is rejected with `aggregate type unsupported
  by @callC …` until phase 4 lands.

## 2. What exists today (survey results)

Reading order for an implementer; all line numbers at commit `8df0671`.

**Parsing.** All `@name(...)` intrinsics parse in one block,
`crates/jam-syntax/src/parser.rs:1260-1326`, producing a single
`AstTag::AtCall` node (`crates/jam-syntax/src/ast_flat.rs:72`):
`lhs` = name StringIdx, `rhs` + `flags` bit0 distinguish three shapes —
expr-arg list (`flags=1`, `rhs` = ExtraIdx `[count, arg0…]`, used by the
`@emit*` family and `@dropInPlace`), no-arg (`flags=0, rhs=0`), and single
type-arg (`flags=0`, `rhs` = TypeIdx, used by `@sizeOf`/`@alignOf`). The fork
is `is_expr_intrinsic` at `parser.rs:1272-1273`; list packing is
`push_count_list` (`parser.rs:1760`); type parsing is `parse_type`
(`parser.rs:1831`).

**Dispatch.** Intrinsics are not analyzed in `analyzer.rs` at all; everything
happens in `astgen_at_call`, `crates/jam-sema/src/astgen.rs:5027-5130`
(reached from `astgen.rs:1249`). `@dropInPlace` shows the expr-arg pattern
(`astgen.rs:5031-5047`); `@sizeOf` shows the type-arg pattern
(`astgen.rs:5049-5065`, with `resolve_array_expr_instantiate` +
`apply_current_subst` for generic contexts). Unknown names error at
`astgen.rs:5124-5128`.

**Function types & indirect calls.** `TypeKind::Fn` interned via
`TypePool::intern_fn(ret, params)` (`crates/jam-syntax/src/ast_flat.rs:493-506`,
params read back with `fn_params_at`). Calling a `fn`-typed value goes through
`astgen_indirect_fn_call` (`astgen.rs:4946-5023`) →
`build_indirect_call` (`astgen.rs:4905-4942`) → `JirTag::CallIndirect`
(`crates/jam-sema/src/jir.rs:156`: `a` = Fn-typed callee JirRef, `b` =
ExtraIdx `[count, args…]`, `ty` = return type). This is exactly the JIR
`@callC` emits — the union-pun in `objc.jam` today reaches the same
instruction after three lines of `Sig*` ceremony.

**Codegen.** `CallIndirect` lowers at
`crates/jam-sema/src/jir_codegen.rs:790-842`: builds the LLVM fn type from
the callee's Fn TypeKey via `get_llvm_type` (non-variadic —
`fn_type(&params, false)` at line 808), loads byref aggregates to pass by
value (815-820), calls `Builder::indirect_call`
(`crates/jam-llvm/src/ll.rs:660-681`), spills aggregate returns to an alloca
(834-840).

**The ABI gap.** The direct-call prototype builder `jir_declare_prototype`
(`jir_codegen.rs:88-141`) applies full classification
(`classify_param`/`classify_return` from `crates/jam-sema/src/abi.rs`,
sret + pointer params + `CallConv::C` + bool zeroext). The `CallIndirect`
path applies **none of it**: no sret, no byval attribute, no call-site CC.
What saves it in practice: `get_llvm_type` maps `f32`/`f64` to LLVM
`float`/`double` (FP registers land right), and LLVM's AArch64 backend lowers
small first-class aggregates the same way the C ABI does (HFAs → `v0–v3`,
≤ 16-byte structs → GP regs). That is why `examples/window.jam`'s
NSRect-by-value works. What it does **not** cover: aggregates > 16 bytes that
aren't HFAs (need byval/sret) — currently a silent miscompile if anyone tries.
Note `abi.rs::classify_param` is jam's *internal* convention (all aggregates
by pointer) — it is **not** the C ABI and must not be applied to `@callC`
args (it would break the working NSRect case).

## 3. Implementation

### Phase 1 — parser (`crates/jam-syntax`)

`@callC` is a hybrid the existing three shapes don't cover: one leading type
arg + N expr args. Add a branch in the intrinsic block keyed on
`name.as_slice() == b"callC"` (next to the `is_expr_intrinsic` fork,
`parser.rs:1272`):

1. `parse_type()` for `R`;
2. expect `,`, then the usual comma loop of `parse_logical_or()` for
   `addr, a0…` (error if empty: `@callC expects a function address after the
   return type`);
3. encode as `AtCall` with **`flags = 3`** (bit0 "has expr args" as today,
   bit1 "extra has a leading TypeIdx header") and
   `rhs` = ExtraIdx → **`[retTypeIdx, argCount, arg0…]`**. Write the header
   slot manually, then reuse the `push_count_list` layout for the tail so
   astgen's existing `[count, args…]` readers work off `extra_idx + 1`.

No lexer or AST-tag changes; `AtCall`'s doc comment in `ast_flat.rs:72` gets
the new flags bit documented.

~30 LOC.

### Phase 2 — astgen (`crates/jam-sema/src/astgen.rs`)

Add a pre-`match` branch in `astgen_at_call` alongside `@dropInPlace`
(`astgen.rs:5031`), gated on `name == "callC" && n.flags & 2 != 0`:

1. **Return type.** `TypeIdx::new(extra[0])`, concretized like `@sizeOf`
   does: `apply_current_subst` → `resolve_generic_call_instantiate` /
   `resolve_array_expr_instantiate` as applicable. Accept `void`.
2. **Address.** `astgen_expr` the first value arg; require an integer type of
   pointer width (`u64`/`usize`), else
   `@callC address must be u64 (got …)`.
3. **Args.** `astgen_expr(gctx, arg, TypeIdx::NONE)` each remaining arg;
   collect static types from `gctx.jfn.get_inst(r).ty` (the pattern
   `build_indirect_call` already uses at `astgen.rs:4912-4927`). Reject
   unsized/comptime-only types; run the aggregate-shape check (§4) on every
   aggregate arg type and on non-void `R`.
4. **Signature.** `let fn_ty = type_pool.intern_fn(ret_ty, param_tys)`.
5. **Callee value.** Emit `JirTag::IntToPtr` (`jir.rs:134`) on the address
   value with `ty = fn_ty` — `CallIndirect` codegen reads the Fn TypeKey off
   the callee's `.ty` (`jir_codegen.rs:792-798`), and `get_llvm_type` maps
   `TypeKind::Fn` to an opaque pointer (`codegen_context.rs:1943`), so the
   IntToPtr result is exactly the callable the backend expects.
6. **Call.** Pack `[count, args…]` and emit `JirTag::CallIndirect` with
   `ty = ret_ty` — either by calling `build_indirect_call`
   (`astgen.rs:4905`) with the synthesized callee, or a thin variant of it if
   its arg-lowering coupling doesn't fit (it lowers args itself; we already
   lowered them, so a variant that takes pre-lowered refs is likely cleaner
   and reusable by `astgen_indirect_fn_call` later).
7. Result: the call's JirRef (or `NO_JIR_REF` for `void`).

No changes to `jir.rs`, `jir_verify.rs` (CallIndirect's extra bounds check at
`jir_verify.rs:369` already covers this), or the LLVM crate.

~70 LOC + diagnostics.

### Phase 3 — aggregate-shape guard (part of phase 2's checks)

A small astgen-side classifier, `callc_aggregate_ok(ty) -> bool`, mirroring
zig's `arch/aarch64/abi.zig::classifyType` (`../zig-0.10.1`, lines 17-74)
but only as a *predicate*:

- walk the aggregate's leaf fields; if all are floats of one width and there
  are ≤ 4 → OK (HFA, lands in `v0–v3`);
- else if every leaf is a full 64-bit word (64-bit int / pointer) and there
  are ≤ 2 → OK (each leaf maps 1:1 to an x-register, so LLVM's per-member
  split equals the C ABI packing);
- else → reject (`aggregate type unsupported by @callC …`). Notably this
  rejects `{i32,i32}` (div_t): it is only 8 bytes, but LLVM splits it into
  two w-registers where C packs it into one x-register — the ≤-16-bytes rule
  alone would silently miscompile it.

Uses existing `type_size` (`codegen_context.rs:1951`) and struct-field
iteration; no new subsystems. This converts today's silent-miscompile class
into a compile error and is what keeps phases 1–3 at zero codegen risk.

~25-40 LOC.

Total for phases 1–3: **~125-140 LOC**, matching MSGSEND.md's estimate.

### Phase 4 (follow-up, separate change) — full C-ABI indirect calls

Lift the aggregate restriction by making `CallIndirect` lowering genuinely
C-ABI-classified, the way zig does it (`codegen/llvm.zig::airCall`, 4692-4966,
gated on the fn type's `cc == .C`):

- add a real per-target classifier (start AArch64: memory / float_array /
  integer / double_integer — port of zig's `classifyType`);
- in `CallIndirect` lowering: sret alloca prepended as arg 0 + sret attribute
  for memory-class returns, byval pointer for memory-class args;
- add an `LLVMSetInstructionCallConv` wrapper in `crates/jam-llvm/src/ll.rs`
  (none exists — grep confirms) and stamp the call site;
- decide whether jam Fn *types* grow a CC bit (zig's fn types carry `cc`;
  jam's `TypeKind::Fn` has no flag slot for it — likely a new interned field)
  so `@callC`-synthesized signatures and ordinary jam fn-pointer calls can
  coexist on one lowering path.

This also fixes a latent pre-existing bug: an indirect call to a *jam*
function with aggregate params doesn't match `jir_declare_prototype`'s
classified signature today. Out of scope here; tracked as its own plan when
needed — no objc use case requires it (NSRect is an HFA; all other Cocoa
traffic is pointer/scalar).

## 4. Why the guard boundary is where it is (AArch64 evidence)

From zig 0.10.1's AArch64 classifier and LLVM's aggregate lowering:

| shape | naive `CallIndirect` today | C ABI | verdict |
|---|---|---|---|
| ints, ptrs (as `u64`), bools | GP regs | GP regs | correct |
| `f32`/`f64` | FP regs (`get_llvm_type` → `float`/`double`) | FP regs | correct |
| HFA ≤ 4 same-width floats (NSRect, NSPoint, CGSize) | LLVM splits struct into `v0–v3` | `v0–v3` | correct |
| ≤ 2 leaves, each a 64-bit word (`{u64}`, `lldiv_t`) | split 1:1 into x-regs | x-regs | correct |
| sub-word / mixed leaves ≤ 16 B (`div_t` `{i32,i32}`, `{u64,f64}`) | split one member per register (two w-regs / x0+d0) | packed into x-regs | **wrong → reject until phase 4** |
| aggregate > 16 B non-HFA | LLVM splits into many regs/stack | pointer to caller memory (byval) / sret | **wrong → reject until phase 4** |

(The `div_t` row was confirmed empirically: the corpus test originally called
`div(7,2)` through `@callC` and got a wrong `rem` back — LLVM returned the
two `i32`s in `w0`/`w1` while libc packs them into `x0`.)

Bool zeroext: direct extern calls set `zeroext` (`jir_codegen.rs:132-146`);
the indirect path doesn't. AArch64 AAPCS requires the *caller* to have
extended — passing jam's `i1`-from-`i8` bools is fine in practice since they
lower as `i8` loads, but add a `zeroext` call-site attribute in phase 4 for
strictness; not a blocker for objc (BOOL passes as an integer arg).

## 5. Tests

### Compiler unit tests (`cargo test --workspace`)

**Parser** — `crates/jam-syntax/src/parser.rs` tests module, alongside the
`parse_expr` helper (`parser.rs:2567`):

- `callc_parses_type_then_args`: `@callC(f64, a, x, y)` → `AtCall`,
  `flags == 3`, extra decodes to `[TypeIdx(f64-ish), 3, …]`;
- `callc_zero_call_args`: `@callC(u64, a)` → argCount 1;
- `callc_struct_return_type`: `@callC(NSRect, a, r)` parses the named type;
- `callc_missing_addr_errors`: `@callC(u64)` produces the parse diagnostic.

**Astgen** — `crates/jam-sema/src/astgen.rs` tests module, using
`lower_first_fn` (`astgen.rs:8165`, runs source → JIR-verify → LLVM-verify):

- `callc_scalar_lowering`:
  `fn f(a: u64) u64 { return @callC(u64, a, 1 as u64, 2 as u64); }` —
  lowers, verifies; assert the JIR contains an `IntToPtr` whose `ty` is a
  `Fn` type and a `CallIndirect` with argCount 2 and `ty == U64`;
- `callc_f64_arg_and_return`: `fn f(a: u64, x: f64) f64 { return
  @callC(f64, a, x); }` — the synthesized Fn TypeKey's param list is `[F64]`
  (read back via `fn_params_at`), proving floats keep their static type;
- `callc_void_return`: statement form lowers to a `CallIndirect` with void
  `ty` and yields no value;
- `callc_hfa_struct_arg`: 4×`f64` struct arg accepted and LLVM-verifies;
- `callc_rejects_large_aggregate`: 24-byte `{u64,u64,u64}` arg → the phase-3
  diagnostic (add a diagnostics-capturing sibling of `lower_first_fn` if none
  exists — follow `init_analysis.rs:1346`'s `analyze_named` shape);
- `callc_rejects_non_u64_addr`: `bool` address → diagnostic.

The aggregate predicate gets direct table-driven tests (HFA f32×3 ok, f64×4
ok, f64×5 reject, mixed f32/f64 16 B ok-by-size, `{u64,u64,u64}` reject) —
in `astgen.rs`'s test module or next to the predicate if it lands in
`codegen_context.rs`.

### Corpus tests (`make test` → `jam test tests`)

`tests/unit/test_callc.jam` — modeled on `test_intrinsics.jam` /
`test_extern_call.jam`, calling real libc through addresses:

```jam
const { assert } = import("test");

pub extern fn abs(x: i32) i32;
pub extern fn labs(x: i64) i64;
pub extern fn fabs(x: f64) f64;

tfn callcInt() {
    assert(@callC(i32, abs as u64, -7 as i32) as i64, 7);
}

tfn callcFloatArgAndReturn() {
    // f64 through FP registers both directions.
    assert(@callC(f64, fabs as u64, -2.5) as i64, 2);
}

tfn callcZeroArgs() { … }        // e.g. getpid-style: nonzero result
tfn callcSmallStructReturn() {   // div(7,2) → div_t{quot,rem}, 8-byte aggregate
    …
}
```

`tests/reject/` — one file per pinned diagnostic, `// expect-error:` style:

- `reject_callc_missing_addr.jam` — `@callC(u64);`
- `reject_callc_bad_addr_type.jam` — bool address
- `reject_callc_large_aggregate.jam` — 24-byte struct arg

### End-to-end proof (manual, not in CI)

Port `../jam-objc/examples/window.jam`'s `SigInitRect`/`SigRect1` unions to
`@callC` and run it — the NSRect-by-value HFA case against live AppKit is
the acceptance test MSGSEND.md defines. Library migration itself
(deleting `Sig*`/`rawSend*` from `objc.jam`) is a jam-objc follow-up.

## 6. Order of work

1. Phase 1 parser + parser unit tests.
2. Phase 2 astgen + phase 3 guard + astgen/predicate unit tests.
3. Corpus files (`tests/unit/test_callc.jam`, `tests/reject/reject_callc_*`).
4. `docs/REFERENCE.md` intrinsics table row + a short FFI-section example.
5. Manual window.jam port to validate against AppKit; then the objc.jam
   cleanup lands in jam-objc.
6. Phase 4 (C-ABI-classified CallIndirect) only when a >16-byte non-HFA
   signature actually shows up.

## 7. Zig 0.10.1 reference notes (what we mirrored, what we didn't)

- **Variadic builtin shape**: zig marks truly variadic builtins with
  `param_count = null` (`src/BuiltinFn.zig:144`; only `@compileLog` and
  `@TypeOf`) and has them self-validate arity in AstGen
  (`src/AstGen.zig:7713` `typeOf`, 7829 `compileLog` — the multi-op
  lowering loop `@callC`'s parser branch imitates). Zig's `@call` is *not*
  variadic — it funnels arity through a tuple (`BuiltinFn.zig:279`,
  `param_count = 3`) because zig has tuples; jam doesn't, which is exactly
  why `@callC` is a genuinely variadic intrinsic instead.
- **Signature synthesis**: zig's `@Type(.Fn)` reification
  (`src/Sema.zig:18355-18448`) builds a fn type from
  `{cc, return_type, param_types, is_var_args=false}` — the analogue of our
  `intern_fn(ret, params)` step, minus a CC field jam's Fn types don't have
  yet (phase 4 decides that).
- **Where the ABI comes from**: in zig, the call site's entire C-ABI
  treatment is derived from the callee pointer's fn type and applies only
  when `cc == .C` (`src/codegen/llvm.zig:4692-4966`; classification in
  `src/arch/aarch64/abi.zig:17-74`, HFA float-count rule at 76-115). A
  synthesized type with the wrong/missing cc silently mis-lowers — that
  failure mode is why phases 1-3 *reject* the shapes the naive path can't
  prove correct instead of trusting LLVM's default lowering everywhere.
- **What we skipped**: zig's full per-target classifier + iterator
  (`ParamTypeIterator`, `llvm.zig:10508-10694`, sret via `firstParamSRet`
  10288-10311) is the phase-4 blueprint, not a phase-1 dependency.

## 8. References

- `../jam-objc/MSGSEND.md` — the problem statement; `objc.jam` (Sig*/rawSend*
  table), `examples/window.jam` (NSRect HFA case).
- Survey anchors in this repo: `parser.rs:1260` (intrinsic parsing),
  `astgen.rs:5027` (`astgen_at_call`), `astgen.rs:4905`
  (`build_indirect_call`), `jir.rs:156` (`CallIndirect`),
  `jir_codegen.rs:790` (indirect-call lowering), `jir_codegen.rs:88`
  (`jir_declare_prototype`, the classified direct path), `abi.rs:71`
  (`is_by_ref` — jam-internal, not C ABI), `ast_flat.rs:493` (`intern_fn`).
- zig 0.10.1: `src/BuiltinFn.zig`, `src/AstGen.zig` (`builtinCall`),
  `src/Sema.zig` (`analyzeCall`, `zirReify` `.Fn`), `src/codegen/llvm.zig`
  (`airCall`), `src/arch/aarch64/abi.zig` (`classifyType`).
