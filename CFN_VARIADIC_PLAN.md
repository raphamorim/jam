# Variadic `cfn` — plan

**Status: IMPLEMENTED**, including phase 0 (mixed `type` + runtime params on
fns and methods did NOT work and were made to work via the generalized
instantiation path), phase 3 (objc.jam migrated — all 22 jam-objc tests pass
and `examples/window.jam` runs against live AppKit). Implementation notes vs.
this plan: clone symbols include the owning struct
(`objc.Object.msgSend__va__…`) to keep same-named methods from colliding;
`R = void` normalizes the clone's return type to the pipeline's `NONE`
spelling and `return <void expr>;` lowers to a bare `ret`; spread must be the
last argument; re-spreading a pack into another variadic cfn's pack is
rejected (forward to `@callC` or a concrete call instead).

The ergonomics layer on top of `@callC` (`CALLC_PLAN.md`): let a `cfn` declare
a trailing argument pack and forward it, so `objc.jam` can collapse its ~25
arity/shape-suffixed wrappers into one `msgSend`:

```jam
pub const Object = struct {
    value: u64,

    pub cfn msgSend(self: Object, R: type, op: Sel, args: ...) R {
        return @callC(R, msgSendAddr(), self.value, op.value, args...);
    }
};

// call sites — any arity, any shape, one name:
const d: f64  = obj.msgSend(f64, sel("doubleValue"));
const title   = obj.msgSend(u64, sel("title"));
win.msgSend(void, sel("setFrame:display:"), r, 1 as u64);          // NSRect by value
const w: u64  = alloc.msgSend(u64, sel("initWithContentRect:styleMask:backing:defer:"),
                              r, style, backing, 0 as i64);
```

Depends on `@callC` (phases 1–3 of `CALLC_PLAN.md`) landing first. This is
the "cfn-native expansion" row of MSGSEND.md's options table, now scoped.

## 1. A corrected premise

MSGSEND.md says "jam's `cfn` already emits code into the caller, like a
macro." The survey shows `cfn` is actually **two unrelated mechanisms**,
split at `parser.rs:2521-2528` on whether the first param is `self`:

- **Top-level `cfn`** (`is_comp_time_fn`) — the macro-ish one. Its body runs
  in the `ComptimeEvaluator`; `@emit*` calls are recorded (`CfnEmitCmd`,
  `astgen.rs:3427-3444`) and replayed into the caller
  (`replay_cfn_emits`, `astgen.rs:4278-4297`). But **every argument must
  fold to a compile-time constant** — `ComptimeValue` (`comptime.rs:29-44`)
  has no variant wrapping a runtime value, and non-constant args error at
  `astgen.rs:4222-4237` / `codegen_context.rs:385-389`. Runtime data is
  reachable only by *name-string* lookup into the caller's locals
  (`replay_print_local`, `astgen.rs:4105-4125`). A `msgSend` forwarding
  runtime receivers/floats cannot be this flavor.
- **Struct `cfn` methods** (`is_cfn`) — ordinary runtime functions that
  merely opt into compiler-synthesized call sets (`drop`/`clone`/`at`/…,
  `drop_registry.rs:53-89`). They lower through `emit_call`
  (`astgen.rs:3120-3188`) like any method and receive real runtime args.

So variadic `cfn` builds on the **method flavor + monomorphization**, not on
emit/replay. The precedent is already in the language: generic methods are
"cloned per instantiation" (docs/REFERENCE.md §Generics), and `comp` value
params already produce per-callsite specialized clones
(`astgen_comp_instantiated_call`, `astgen.rs:3028-3113`). The pack is one
more axis on that existing instantiation scheme.

## 2. Semantics

### Declaration

- A `cfn` (struct method, or top-level `cfn` — see §5 note) may declare **one
  trailing named pack parameter**: `args: ...`. It must be last; at most one.
- A pack-bearing `cfn` is never lowered as-declared (it has no fixed
  signature). It joins the `is_generic()` family (`ast.rs:93-101`) and is
  skipped at decl-lowering time (`astgen.rs:1480`), exactly like type-generic
  fns today.
- Plain `fn` does not get packs. `fn` promises one real symbol with a fixed
  ABI; `cfn` already means "the compiler may specialize this." (Extern's
  bare `...` at `parser.rs:2000-2011` is unchanged and unrelated — that is a
  C-variadic *callee*; this is a compile-time *template*.)

### Call site

- Args beyond the fixed params form the pack. Each is lowered normally and
  its **static type** is captured — the same rule as `@callC` args, no
  coercion. `obj.msgSend(f64, s)` has an empty pack; that's legal.
- Instantiation key = (receiver type, explicit `type` args, `comp` values,
  **pack type tuple**). Same key → same clone (memoized, like generic
  instantiations); new key → new mangled clone, e.g.
  `Object__msgSend$f64$$u64_f64` (exact scheme follows `mangling.rs`).

### Inside the body

The pack name is **not a value**. It cannot be read, stored, indexed,
`len`'d, or returned. It has exactly one use: the spread form `args...`,
valid only inside a call-argument list (a normal call, another variadic
`cfn`, an extern C-variadic call, or an intrinsic like `@callC`), where it
expands positionally into the clone's materialized tail parameters. Any
other use of the pack name is a compile error
(`variadic pack `args` can only be forwarded with `args...``).

This forward-only restriction is what keeps the feature ~5× smaller than the
rejected general variadics (VARIADICS route in MSGSEND.md): no tuple type, no
`anytype`, no pack iteration, no pack-typed locals.

## 3. Mechanism

At a call site that resolves to a pack-bearing `cfn` (method dispatch around
`astgen.rs:4572-4640`, plain-call path `astgen.rs:3333-3418`):

1. Lower fixed args against declared params (`lower_method_args`,
   `astgen.rs:4327-4353`, already tolerates a tail via its `.get(i)`
   else-branch — the same shape as the extern varargs tail at
   `astgen.rs:3404-3418`); lower pack args with `astgen_expr(…, NONE)` and
   collect `pack_tys` from `gctx.jfn.get_inst(r).ty`.
2. Build the instantiation: clone the `FunctionAST` with the pack param
   replaced by materialized params `__va0: T0 … __vaN: TN`; substitute
   `type`/`comp` params via the existing machinery
   (`generics.rs::substitute_type`, `set_current_comp_subst` — see
   `astgen_comp_instantiated_call`'s define-once-then-call structure at
   `astgen.rs:3028-3113`, which this either extends or mirrors).
3. Record the pack binding for the clone's body:
   `current_pack: Vec<param names>` alongside the existing subst state.
4. Define the clone (once per key), `emit_call` it with fixed + pack refs.
5. When lowering the clone's body and an argument list contains `args...`
   (new AST node, §4), splice in refs to `__va0..__vaN` at that position.
   Since `@callC` reads each arg's static type, the synthesized C signature
   automatically matches the call site's — no extra plumbing.

Composability falls out: a clone is fully concrete, so `args...` passed to
another variadic `cfn` just triggers that cfn's own instantiation with
concrete types. Forwarding into an extern C-variadic (`printf(fmt, args...)`)
also works via the existing varargs-tail lowering — a free side benefit worth
a test.

## 4. Implementation

### Phase 0 — prerequisite verification (before any code)

Two things the plan assumes, to confirm on HEAD (cheap spikes, no agents
needed):

- **`@callC` landed** per CALLC_PLAN.md phases 1–3 (the body's only
  must-have consumer).
- **Mixed `type` + runtime params on a value-returning fn/method** — does
  `fn f(T: type, x: u64) u64` called as `f(f64, v)` work today?
  `is_generic()` marks it, but the call-site instantiation path
  (`astgen_comp_instantiated_call`) is documented for `comp` *value* params;
  `ComptimeValue::Type(TypeIdx)` exists (`comptime.rs:37`), so if type-args
  don't already flow through it, unifying them there ("a `type` param is a
  comp param holding a `Type` value, dropped from the clone's signature,
  substituted via `substitute_type`") is a small, separately-landable
  change — and `msgSend(R: type, …)` needs it. Budget ~30-50 LOC if missing.

### Phase 1 — parser (`crates/jam-syntax`)

- In `parse_function`'s param loop (`parser.rs:2000-2034`): accept
  `name: ...` when the decl `is_cfn`, recording `var_args_pack:
  Option<String>` (new `FunctionAST` field next to `is_cfn`/`is_var_args`,
  `ast.rs:49-59`). Keep bare `...` extern-only; keep `is_var_args = false`
  for cfns (that flag means "LLVM-variadic callee", `jir_codegen.rs:114`,
  and clones are fixed-arity — the two must not mix).
- Errors: pack on plain `fn` (`argument packs require cfn`), pack not last,
  more than one pack.
- Spread expression: in call-argument parsing, accept postfix `...` after an
  identifier → new `AstTag::Spread` node (`lhs` = the identifier). Reject
  `...` in any non-call-arg expression position at parse time.
- `is_generic()` (`ast.rs:93-101`): `|| self.var_args_pack.is_some()`.

~50 LOC.

### Phase 2 — astgen instantiation (`crates/jam-sema`)

- Call-site handling per §3 at the method-dispatch and plain-call sites;
  arity check becomes `arg_count >= fixed_params` for pack-bearing cfns
  (today's exact-arity gate for top-level cfns is `astgen.rs:4147-4159`;
  method paths have no strict gate — add one: too-few-args should be a clean
  diagnostic, not an index panic in `eval_cfn_call`-style `.args[i]` code,
  cf. `codegen_context.rs:382-391`).
- Clone construction + memoization keyed as §2; mangling extension in
  `mangling.rs` (append pack type list).
- Body lowering: `AstTag::Spread` handled inside argument-list lowering only
  (`lower_call_args` `astgen.rs:4301-4321`, `lower_method_args`, the
  `@callC` arg loop from CALLC_PLAN.md phase 2, and the extern tail path);
  resolves the pack name against `current_pack`, splices `__va*` loads.
  Anywhere else → diagnostic. Bare use of the pack identifier → diagnostic.
- Top-level `is_comp_time_fn` cfns: explicitly reject a pack for now
  (`variadic packs are not supported on comptime cfns`) — their arguments
  are comptime-folded (`astgen.rs:4222-4237`) and a pack of runtime values
  is meaningless there. This keeps the feature entirely on the
  runtime-method flavor. (If a comptime pack is ever wanted,
  `ComptimeValue::Aggregate` — constructed nowhere today, indexed only at
  `comptime.rs:613-618` — is the modeling slot; out of scope.)

~110-140 LOC.

### Phase 3 — objc.jam migration (in `../jam-objc`, after both features)

- Add `cfn msgSend(self, R: type, op: Sel, args: ...) R` (and
  `msgSendSuper`) to `Object`/`Class`; delete `Sig*` unions, `rawSend*`,
  and the suffixed wrappers (or keep one release's worth of deprecated
  aliases); port `examples/window.jam` (its two hand-rolled unions become
  two ordinary `msgSend` calls); update MSGSEND.md's status table.

Total compiler-side: **~160-190 LOC** (+ the phase-0 unification if needed) —
vs. ~370 for the rejected general variadic-generics route, because the pack
is forward-only and rides existing instantiation machinery.

## 5. Tests

### Compiler unit tests (`cargo test --workspace`)

**Parser** (`crates/jam-syntax/src/parser.rs` tests module):

- `cfn_pack_param_parses`: `cfn f(self: S, args: ...)` → `var_args_pack ==
  Some("args")`, `is_var_args == false`;
- `pack_on_fn_rejected`, `pack_not_last_rejected`, `two_packs_rejected`;
- `spread_in_call_args_parses`: `g(x, args...)` → arg list contains a
  `Spread` node wrapping the identifier;
- `spread_outside_call_rejected`: `var x = args...;` errors;
- extern bare `...` still parses and still rejects on non-extern
  (regression guard for `parser.rs:2004-2008`).

**Astgen** (`crates/jam-sema/src/astgen.rs` tests module, via
`lower_first_fn` / a diagnostics-capturing helper, and a whole-module
lowering helper since these tests need a struct + method + caller):

- `pack_monomorphizes_per_shape`: one variadic cfn called with `(u64)` and
  `(u64, f64)` tails → two clones with distinct mangled names; called twice
  with the same tail → one clone (assert on the module's function list);
- `pack_forwards_into_callc`: body `return @callC(R, addr, args...)`;
  assert the emitted `CallIndirect`'s synthesized Fn TypeKey params equal
  the call site's tail types (read back via `fn_params_at`) — this is the
  end-to-end contract of the whole feature;
- `empty_pack_ok`: zero tail args lowers and LLVM-verifies;
- `pack_forwards_into_extern_varargs`: `printf(fmt, args...)` tail lowers
  through the varargs-tail path;
- `bare_pack_use_rejected`, `pack_spread_in_non_call_rejected`,
  `too_few_fixed_args_rejected`, `pack_on_comptime_cfn_rejected` —
  diagnostics pinned;
- mangling unit test in `mangling.rs` for the pack-type suffix.

### Corpus tests (`make test`)

`tests/unit/test_cfn_variadic.jam`:

```jam
const { assert } = import("test");

pub extern fn abs(x: i32) i32;
pub extern fn fabs(x: f64) f64;

const Caller = struct {
    addr: u64,
    cfn callThrough(self: Caller, R: type, args: ...) R {
        return @callC(R, self.addr, args...);
    }
};

tfn packInt() {
    const c: Caller = { addr: abs as u64 };
    assert(c.callThrough(i32, -7 as i32) as i64, 7);
}
tfn packFloat() {          // different shape, same cfn
    const c: Caller = { addr: fabs as u64 };
    assert(c.callThrough(f64, -2.5) as i64, 2);
}
tfn packEmptyAndMulti() { … }   // zero-arg tail + a 3-arg tail through one cfn
```

`tests/reject/`:

- `reject_pack_on_fn.jam` — `fn f(args: ...)`;
- `reject_pack_bare_use.jam` — body reads `args` without `...`;
- `reject_pack_spread_outside_call.jam`;
- `reject_pack_on_comptime_cfn.jam` — top-level cfn with a pack.

### Acceptance (manual)

The phase-3 objc migration: `examples/window.jam` running against AppKit
with every send going through the single `msgSend`, including the NSRect
HFA shapes.

## 6. Design notes & rejected alternatives

- **Why not the emit/replay macro flavor**: runtime values can't cross into
  the comptime evaluator (§1); extending `CfnEmitCmd` with a `Call` variant
  plus name-string argument capture would be a stringly-typed macro system —
  strictly worse than monomorphized clones for type safety and for reusing
  the generics infrastructure.
- **Why clones, not call-site inlining**: clones are memoized per shape
  (code-size win across many call sites — objc apps repeat shapes heavily),
  reuse the define-once-then-call structure that already exists for `comp`
  instantiation, and keep MVS/drop analysis working on an ordinary function
  body instead of inventing hygiene rules for spliced ASTs.
- **Why forward-only packs**: every capability beyond `args...`-in-arg-lists
  (indexing, len, iteration, storing) drags in tuple types — the exact
  subsystem MSGSEND.md's variadics evaluation rejected. Forward-only covers
  the entire motivating use case.
- **`is_var_args` stays extern-only**: a pack-bearing cfn's clones are
  fixed-arity; conflating the flags would make `jir_codegen.rs:114` emit an
  LLVM-variadic function type for a clone. Separate field, asserted disjoint.
- MSGSEND.md's options table should be updated when this lands: "cfn-native
  expansion" becomes the accepted route for the ergonomics layer, with
  `@callC` as its foundation.

## 7. References

- `CALLC_PLAN.md` — the ABI primitive this forwards into.
- `../jam-objc/MSGSEND.md` — problem statement; `objc.jam:341-583` — the
  wrapper surface phase 3 deletes; `examples/window.jam` — acceptance case.
- Survey anchors (commit `8df0671`): cfn split `parser.rs:2521-2528`; param
  parsing `parser.rs:2000-2034`; AST flags `ast.rs:54-59`, `is_generic()`
  `ast.rs:93-101`; comp instantiation `astgen.rs:3028-3113`; varargs tail
  `astgen.rs:3404-3418`; top-level cfn expansion `astgen.rs:4141-4275`
  (arity gate 4147-4159, const-fold gate 4222-4237); arg lowerers
  `astgen.rs:4301-4386`; `emit_call` `astgen.rs:3120-3188`; comptime values
  `comptime.rs:29-44`; generics `generics.rs:24-113`; synthesis registry
  `drop_registry.rs:53-89`.
