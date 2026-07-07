/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! AstGen — eager lowering of a parsed `FunctionAST` into a typed
//! [`JirFunction`]. Ported (incrementally) from `src/astgen.cpp`.
//!
//! The output is typed from the start: literals lower at their natural width or
//! at an `expected`-type hint threaded from the parent. Each visited node
//! pushes zero or more `JirInst`s into the function's instruction array.
//!
//! ## Scope of this increment (the end-to-end seed)
//!
//! The slice that closes source→JIR→LLVM-IR for arithmetic functions:
//! `astgen_metadata` (signature), `astgen_body_into` (params + body walk +
//! fall-through terminator), and `astgen_expr` covering NumberLit, BoolLit,
//! Variable (local/param load), BinaryOp (arith + comparison with int
//! widening), and the Return statement. The remaining node shapes
//! (VarDecl/Assign/If/While/For/Match, struct/array/index/member, calls,
//! comp/drop machinery, the integer smallest-fit fallback, generic
//! instantiation) return a clean "not yet ported" error and land next.

use std::collections::HashMap;

use float128::quad_to_target_as_double;
use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};
use jam_syntax::ast::{FunctionAST, Param};
use jam_syntax::ast_flat::{AstNode, AstTag, TypeKind, TypePool, builtin};

use crate::abi::{ParamAbiKind, ReturnAbiKind, classify_param, classify_return};
use crate::codegen_context::CodegenContext;
use crate::comptime::ComptimeValue;
use crate::jir::{JirBlockRef, JirFunction, JirInst, JirRef, JirTag, NO_JIR_BLOCK, NO_JIR_REF};
use crate::jir_codegen::{jir_declare_prototype, jir_define_body};
use crate::mangling::mangled_function_name;

/// A drop-bearing binding tracked for scope-exit destruction (the C++
/// `DropTrack`). The mangled drop symbol is re-resolved at emission time by
/// `emit_drop_in_place` (matching the C++ emit path), so it isn't cached here.
#[derive(Clone)]
struct DropTrack {
    var_name: String,
    slot: JirRef,
    ty: TypeIdx,
}

/// An active loop's targets + the `drop_scopes` index of its body scope, so
/// `break`/`continue` drop everything down through the loop body before jumping.
struct LoopFrame {
    continue_block: JirBlockRef,
    break_block: JirBlockRef,
    body_scope_idx: usize,
}

/// Per-function lowering state. The comp-binding scope chain, `localScopes`
/// (redeclaration namespaces), and `runtimeCondDepth` the C++ `AstGenCtx`
/// carries are deferred with their features; this slice carries the locals map,
/// the loop stack, and the drop-scope stack.
/// Metadata for an explicit `comp const` / `comp var` binding (the C++
/// `AstGenCtx::CompBindingInfo`).
#[derive(Clone, Copy)]
struct CompBindingInfo {
    /// Runtime-conditional depth at declaration — assignments from deeper
    /// runtime control flow are rejected (a comp value cannot depend on a
    /// runtime branch).
    decl_depth: u32,
    /// `comp const` rejects reassignment.
    is_const: bool,
}

struct AstGenCtx<'a, 'ctx> {
    jfn: &'a mut JirFunction,
    ctx: &'a CodegenContext<'ctx>,
    current_block: JirBlockRef,
    locals: HashMap<String, JirRef>,
    local_types: HashMap<String, TypeIdx>,
    current_node: NodeIdx,
    loop_stack: Vec<LoopFrame>,
    /// Stack of lexical drop scopes (frame 0 = function body), each holding the
    /// drop-bearing bindings declared at that level in declaration order.
    drop_scopes: Vec<Vec<DropTrack>>,
    /// Names DECLARED (via var/const decls) per lexical scope, parallel to
    /// `drop_scopes` — push/pop happen together. The redeclaration check in
    /// `astgen_var_decl` only consults the top frame, so sibling blocks
    /// (`if { const op = ...; }` twice at the same level) don't trip it.
    local_scopes: Vec<std::collections::HashSet<String>>,
    /// Function-local `comp const` / `comp var` bindings — compile-time values
    /// inlined at use sites, never given a runtime slot.
    comp_scope: crate::comptime::ComptimeScope,
    /// Per-frame metadata for the explicit comp bindings declared in that
    /// frame (pushed/popped with the drop scopes). Seeded names (module
    /// consts, comp params) are absent, so they never match the
    /// comp-assignment path — same contract as the C++ `compBindInfo`.
    comp_bind_info: Vec<HashMap<String, CompBindingInfo>>,
    /// Depth of runtime conditional control flow (if / while / for / match arm
    /// bodies). `comp if` arms do NOT count: their statements lower inline and
    /// execute unconditionally.
    runtime_cond_depth: u32,
    /// Recoverable diagnostics collected while still walking the body (the C++
    /// `recoverHere` path: append the error + emit a `Poison` placeholder and
    /// keep going, so independent statements each report their own miss). These
    /// are flushed — in source order — when the body finishes lowering; a
    /// non-empty list makes the body's overall result an `Err`.
    recovered: Vec<String>,
}

/// Append a recoverable diagnostic (already prefixed `file:line: error:` via
/// [`fail_node`]) anchored at `gctx.current_node` and hand back a `Poison`
/// placeholder so the walk continues. Mirrors the C++ `recoverHere`
/// (astgen.cpp:243): the driver short-circuits before codegen whenever any
/// error was recorded, so the Poison never reaches a real backend.
fn recover_here(gctx: &mut AstGenCtx, message: String, ty: TypeIdx) -> JirRef {
    let prefixed = fail_node(gctx, gctx.current_node, &message);
    gctx.recovered.push(prefixed);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Poison,
            ty,
            ..Default::default()
        },
    )
}

/// Method-miss reporting that understands CONDITIONAL methods (the C++
/// `reportMethodMiss`, astgen.cpp:194): an instantiated generic method
/// withdrawn for these type arguments replays the recorded reason instead of a
/// bare "unknown method".
fn report_method_miss(gctx: &mut AstGenCtx, qualified: &str) -> Result<JirRef, String> {
    if let Some(why) = gctx.ctx.get_withdrawn_method(qualified) {
        return Err(fail_node(
            gctx,
            gctx.current_node,
            &format!("method `{qualified}` is not available for this instantiation — {why}"),
        ));
    }
    Ok(recover_here(
        gctx,
        format!("unknown method `{qualified}`"),
        TypeIdx::NONE,
    ))
}

/// Stamp a propagated astgen error with the `file:line: error:` prefix anchored
/// at `gctx.current_node` — UNLESS it already carries one. Most raw `Err(String)`
/// sites deep in the lowering tree return a bare body (the C++ `failHere` adds
/// the prefix there); this is the single boundary that reproduces that prefix.
/// Move-safety errors (init_analysis) and the field-extract rejections already
/// format their own `file:line: error:`, so an error that already contains
/// `": error: "` is passed through untouched to avoid double-prefixing.
fn prefix_hard_error(gctx: &AstGenCtx, e: String) -> String {
    if e.contains(": error: ") || e.starts_with("error: ") {
        e
    } else {
        fail_node(gctx, gctx.current_node, &e)
    }
}

/// Append `inst` to the function's instruction array AND the current block.
fn emit(gctx: &mut AstGenCtx, mut inst: JirInst) -> JirRef {
    inst.src_line = gctx.ctx.node_store.get_line(gctx.current_node);
    let r = gctx.jfn.push_inst(inst);
    gctx.jfn.get_block_mut(gctx.current_block).insts.push(r);
    r
}

/// Append an Alloca to the function's ENTRY block (block 1) regardless of the
/// current block — stack slots are function-scoped, and `jir_verify` requires
/// allocas to live in the entry block. The alloca is inserted *before* any
/// trailing terminator(s) of the entry block, so an alloca hoisted from inside
/// a loop body (e.g. a nested loop's induction variable) lands among the other
/// entry allocas rather than after the entry block's `Br`.
fn emit_alloca_hoisted(gctx: &mut AstGenCtx, mut inst: JirInst) -> JirRef {
    inst.src_line = gctx.ctx.node_store.get_line(gctx.current_node);
    let r = gctx.jfn.push_inst(inst);
    let mut insert_at = gctx.jfn.get_block(1).insts.len();
    while insert_at > 0 {
        let prev = gctx.jfn.get_block(1).insts[insert_at - 1];
        if !gctx.jfn.get_inst(prev).tag.is_terminator() {
            break;
        }
        insert_at -= 1;
    }
    gctx.jfn.get_block_mut(1).insts.insert(insert_at, r);
    r
}

fn str_at(gctx: &AstGenCtx, id: u32) -> String {
    String::from_utf8_lossy(&gctx.ctx.string_pool.get(StringIdx::new(id))).into_owned()
}

// ---- drop / move tracking ----

