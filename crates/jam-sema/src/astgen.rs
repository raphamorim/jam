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

/// Append a recoverable diagnostic WITHOUT emitting a Poison (the C++
/// `appendErrorHere`) — for skip-and-continue loops like struct-literal fields,
/// where the offending item is simply dropped.
fn append_error_here(gctx: &mut AstGenCtx, message: String) {
    let prefixed = fail_node(gctx, gctx.current_node, &message);
    gctx.recovered.push(prefixed);
}

/// Append a recoverable diagnostic (already prefixed `file:line: error:` via
/// [`fail_node`]) anchored at `gctx.current_node` and hand back a `Poison`
/// placeholder so the walk continues. Mirrors the C++ `recoverHere`
/// (astgen.cpp:243): the driver short-circuits before codegen whenever any
/// error was recorded, so the Poison never reaches a real backend.
fn recover_here(gctx: &mut AstGenCtx, message: String, ty: TypeIdx) -> JirRef {
    append_error_here(gctx, message);
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

/// Enter a structured body's lexical scope (new drop frame).
fn push_drop_scope(gctx: &mut AstGenCtx) {
    gctx.drop_scopes.push(Vec::new());
    gctx.local_scopes.push(std::collections::HashSet::new());
    // Comp bindings are lexically scoped too: an inner block's `comp const x`
    // shadows an outer one, and the outer value is restored at block exit.
    gctx.comp_scope.push();
    gctx.comp_bind_info.push(HashMap::new());
}

/// Drop a scope's bindings in reverse declaration order.
fn emit_drops(gctx: &mut AstGenCtx, bindings: &[DropTrack]) {
    for d in bindings.iter().rev() {
        emit_drop_in_place(gctx, d.slot, d.ty);
    }
}

/// Emit drops for every scope from the top down to (and including) `target_idx`,
/// WITHOUT popping — the AST walker still owns the scope stack (break / continue
/// / return; the structured bodies pop on their own).
fn emit_drops_through_scope(gctx: &mut AstGenCtx, target_idx: usize) {
    let mut i = gctx.drop_scopes.len();
    while i > target_idx {
        let scope = gctx.drop_scopes[i - 1].clone();
        emit_drops(gctx, &scope);
        i -= 1;
    }
}

/// Like [`emit_drops_through_scope`] but skips the first binding named
/// `moved_var` (return-as-move: ownership transferred to the caller). Rebuilds
/// filtered copies; does not mutate the scope stack, so other return paths still
/// drop the local.
fn emit_drops_through_scope_moved_out(gctx: &mut AstGenCtx, target_idx: usize, moved_var: &str) {
    let mut i = gctx.drop_scopes.len();
    while i > target_idx {
        let orig = gctx.drop_scopes[i - 1].clone();
        let mut filtered = Vec::with_capacity(orig.len());
        let mut removed = false;
        for d in orig {
            if !removed && d.var_name == moved_var {
                removed = true;
                continue;
            }
            filtered.push(d);
        }
        emit_drops(gctx, &filtered);
        i -= 1;
    }
}

/// Pop the top drop scope, emitting its drops first unless the current block
/// already terminated (a divergent return/break/continue dropped them).
fn pop_drop_scope_emitting(gctx: &mut AstGenCtx) {
    if gctx.drop_scopes.is_empty() {
        return;
    }
    if !block_has_terminator(gctx) {
        let scope = gctx.drop_scopes.last().unwrap().clone();
        emit_drops(gctx, &scope);
    }
    gctx.drop_scopes.pop();
    gctx.local_scopes.pop();
    gctx.comp_scope.pop();
    gctx.comp_bind_info.pop();
}

/// Find the binding-info record for an explicit comp binding, innermost frame
/// first (the C++ `lookupCompBindingInfo`). `None` when `name` is not an
/// explicit comp binding — seeded module consts / comp params don't match.
fn lookup_comp_binding_info(gctx: &AstGenCtx, name: &str) -> Option<CompBindingInfo> {
    gctx.comp_bind_info
        .iter()
        .rev()
        .find_map(|frame| frame.get(name).copied())
}

/// If `expr_idx` is a bare `Variable` naming a tracked drop local, remove it
/// from its drop scope — ownership has moved to the surrounding owner (a struct
/// field, an array element, a `move` arg), which now runs the drop.
fn consume_moved_variable(gctx: &mut AstGenCtx, expr_idx: NodeIdx) {
    let n = *gctx.ctx.node_store.get(expr_idx);
    if n.tag != AstTag::Variable {
        return;
    }
    let name = str_at(gctx, n.lhs);
    for scope in gctx.drop_scopes.iter_mut() {
        if let Some(pos) = scope.iter().position(|d| d.var_name == name) {
            scope.remove(pos);
            return;
        }
    }
}

/// Format an astgen failure anchored at `node` as `file:line: error: message`,
/// matching the C++ `failNode` -> `Diagnostics::emit` byte output. The file is
/// the context's current display file; the line is `node`'s parse-time line.
fn fail_node(gctx: &AstGenCtx, node: NodeIdx, message: &str) -> String {
    let file = gctx.ctx.current_file();
    let line = gctx.ctx.node_store.get_line(node);
    if file.is_empty() {
        if line > 0 {
            format!("{line}: error: {message}")
        } else {
            format!("error: {message}")
        }
    } else if line > 0 {
        format!("{file}:{line}: error: {message}")
    } else {
        format!("{file}: error: {message}")
    }
}

/// Ownership is tracked per WHOLE binding. Extracting a drop-bearing value out
/// of a pure field path (`h.c`, `o.inner.c`) duplicates ownership: the extracted
/// value and the aggregate's drop glue would both drop the same payload. True
/// when `expr_idx` is a `MemberAccess` chain rooted at a LOCAL binding whose
/// `result_ty` needs drop. Paths that index through raw pointers (`self.ptr[i]`)
/// are the pointer world and stay unchecked. Ported from the C++
/// `isDropBearingFieldExtract`, astgen.cpp:5874-5898.
fn is_drop_bearing_field_extract(gctx: &AstGenCtx, expr_idx: NodeIdx, result_ty: TypeIdx) -> bool {
    if result_ty.is_none() {
        return false;
    }
    let top = *gctx.ctx.node_store.get(expr_idx);
    if top.tag != AstTag::MemberAccess {
        return false;
    }
    let mut cur = expr_idx;
    while !cur.is_none() {
        let n = *gctx.ctx.node_store.get(cur);
        if n.tag == AstTag::MemberAccess {
            cur = NodeIdx::new(n.lhs);
            continue;
        }
        if n.tag == AstTag::Variable {
            let root = str_at(gctx, n.lhs);
            // Only locals/params — `Color.Red` and module paths also parse as
            // MemberAccess on a Variable root.
            if !gctx.locals.contains_key(&root) {
                return false;
            }
            return gctx.ctx.type_needs_drop(result_ty);
        }
        // Index / Deref / call in the path: pointer world.
        return false;
    }
    false
}

/// Reject moving/extracting a drop-bearing field out of its owned aggregate (else
/// double-free). The C++ `rejectDropBearingFieldExtract`, astgen.cpp:5900-5909.
fn reject_drop_bearing_field_extract(
    gctx: &AstGenCtx,
    expr_idx: NodeIdx,
    result_ty: TypeIdx,
    verb: &str,
) -> Result<(), String> {
    if !is_drop_bearing_field_extract(gctx, expr_idx, result_ty) {
        return Ok(());
    }
    Err(fail_node(
        gctx,
        expr_idx,
        &format!(
            "cannot {verb} a drop-bearing field out of its aggregate — ownership is tracked per \
             whole binding, so the field and the aggregate's drop would both run; clone the field \
             out (`.clone()`) or move the whole value"
        ),
    ))
}

/// If the return expr is a bare `Variable` naming a tracked drop local, return
/// its name — its scope-exit drop is suppressed in this return path (the value
/// is moved to the caller).
fn detect_return_move(gctx: &AstGenCtx, val_idx: NodeIdx) -> Option<String> {
    let n = *gctx.ctx.node_store.get(val_idx);
    if n.tag != AstTag::Variable {
        return None;
    }
    let name = str_at(gctx, n.lhs);
    if gctx
        .drop_scopes
        .iter()
        .any(|s| s.iter().any(|d| d.var_name == name))
    {
        Some(name)
    } else {
        None
    }
}

/// Drop every element of a fixed-size array `*ptr` via a runtime `0..count`
/// loop (IndexAddr + recursive drop). Mirrors the C++ `emitArrayElementDrops`.
fn emit_array_element_drops(gctx: &mut AstGenCtx, ptr: JirRef, arr_ty: TypeIdx) {
    let (elem_ty, count) = {
        let ak = gctx.ctx.type_pool.get(arr_ty);
        (TypeIdx::new(ak.a), ak.b)
    };
    if count == 0 {
        return;
    }
    let count_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: count,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let slot = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let zero_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 0,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: zero_ref,
            ..Default::default()
        },
    );

    let cond_b = gctx.jfn.push_block("adropcond");
    let body_b = gctx.jfn.push_block("adropbody");
    let exit_b = gctx.jfn.push_block("adropexit");
    emit_br(gctx, cond_b);

    gctx.current_block = cond_b;
    let i_val = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let cmp_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::ICmpUlt,
            a: i_val,
            b: count_ref,
            ty: builtin::BOOL,
            ..Default::default()
        },
    );
    emit_cond_br(gctx, cmp_ref, body_b, exit_b);

    gctx.current_block = body_b;
    let i_body = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let elem_ptr_ty = gctx.ctx.type_pool.intern_ptr_single(elem_ty);
    let elem_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::IndexAddr,
            a: ptr,
            b: i_body,
            ty: elem_ptr_ty,
            ..Default::default()
        },
    );
    emit_drop_in_place(gctx, elem_ptr, elem_ty);
    let cur = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let one_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 1,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let next = emit(
        gctx,
        JirInst {
            tag: JirTag::Add,
            a: cur,
            b: one_ref,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: next,
            ..Default::default()
        },
    );
    emit_br(gctx, cond_b);

    gctx.current_block = exit_b;
}

/// Recursively destroy `*ptr` of type `ty`: a fixed array drops its elements; a
/// struct/named type emits its `cfn drop` call (if any) then drops droppable
/// fields. Container (Vec etc.) and payloaded-enum element drops are deferred.
fn emit_drop_in_place(gctx: &mut AstGenCtx, ptr: JirRef, ty: TypeIdx) {
    // Fixed-size array: own its elements (no cfn drop / fields of its own).
    if gctx.ctx.type_pool.get(ty).kind == TypeKind::Array {
        let elem = TypeIdx::new(gctx.ctx.type_pool.get(ty).a);
        if gctx.ctx.type_needs_drop(elem) {
            emit_array_element_drops(gctx, ptr, ty);
        }
        return;
    }
    // Contiguous owning container (a `cfn len` + one `*mut[] Elem` data field):
    // synthesize the element-destructor loop BEFORE the container's own drop
    // frees the backing — elements must die while their storage lives.
    emit_container_element_drops(gctx, ptr, ty);
    if let Some(name) = gctx.ctx.lookup_drop_fn_name(ty) {
        let sym = gctx.ctx.string_pool.intern_str(&name).raw();
        emit(
            gctx,
            JirInst {
                tag: JirTag::DropBinding,
                a: ptr,
                b: sym,
                ..Default::default()
            },
        );
    }
    emit_field_drops(gctx, ptr, ty);
    // Payloaded enums own the live variant's payload: tag-dispatched payload drops
    // run after any user drop (the C++ emitEnumPayloadDrops, astgen.cpp:5476).
    emit_enum_payload_drops(gctx, ptr, ty);
}

/// Tag-dispatched payload drops for a payloaded enum at `ptr`: load the
/// discriminant (field 0) and, for each variant whose payload needs dropping,
/// branch to an `edrop` block that drops each payload field at its byte offset
/// within the payload area (field 1) — the same offset math the constructors use.
/// A no-op for non-enums / enums with no drop-bearing payload.
fn emit_enum_payload_drops(gctx: &mut AstGenCtx, ptr: JirRef, ty: TypeIdx) {
    let Some(name) = gctx.ctx.enum_name_of(ty) else {
        return;
    };
    let Some(variants) = gctx.ctx.enum_variants_by_name(&name) else {
        return;
    };
    let u8_ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
    let tag_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::FieldAddr,
            a: ptr,
            b: 0,
            ty: u8_ptr,
            ..Default::default()
        },
    );
    let tag_val = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: tag_ptr,
            ty: builtin::U8,
            ..Default::default()
        },
    );
    for v in &variants {
        if !v
            .payload_types
            .iter()
            .any(|pt| gctx.ctx.type_needs_drop(*pt))
        {
            continue;
        }
        let drop_b = gctx.jfn.push_block("edrop");
        let cont_b = gctx.jfn.push_block("edropcont");
        let disc = emit(
            gctx,
            JirInst {
                tag: JirTag::Int,
                a: v.discriminant,
                ty: builtin::U8,
                ..Default::default()
            },
        );
        let cmp = emit(
            gctx,
            JirInst {
                tag: JirTag::ICmpEq,
                a: tag_val,
                b: disc,
                ty: builtin::BOOL,
                ..Default::default()
            },
        );
        emit_cond_br(gctx, cmp, drop_b, cont_b);
        gctx.current_block = drop_b;
        let pay_area = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: ptr,
                b: 1,
                ty: u8_ptr,
                ..Default::default()
            },
        );
        let mut off: u64 = 0;
        for pt in &v.payload_types {
            let s = gctx.ctx.type_size(*pt).unwrap_or(0);
            let a = gctx.ctx.type_align(*pt).unwrap_or(1).max(1);
            off = off.div_ceil(a) * a;
            if gctx.ctx.type_needs_drop(*pt) {
                let off_ref = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Int,
                        a: off as u32,
                        ty: builtin::U64,
                        ..Default::default()
                    },
                );
                let field_ptr = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::IndexAddr,
                        a: pay_area,
                        b: off_ref,
                        ty: u8_ptr,
                        ..Default::default()
                    },
                );
                emit_drop_in_place(gctx, field_ptr, *pt);
            }
            off += s;
        }
        emit_br(gctx, cont_b);
        gctx.current_block = cont_b;
    }
}

/// If `container_ty` is a contiguous owning container — a struct with a `cfn len`
/// and exactly one `*mut[] Elem` data field whose `Elem` needs drop — emit the
/// `0..self.len()` loop that drops each live element (the C++
/// `emitContainerElementDrops`). Returns whether it applied.
fn emit_container_element_drops(gctx: &mut AstGenCtx, ptr: JirRef, container_ty: TypeIdx) -> bool {
    let Some(sname) = gctx.ctx.struct_name_of(container_ty) else {
        return false;
    };
    // Opt-in marker: a `cfn len` (a plain `fn len` is a borrowing view, not an
    // owning container).
    let Some(len_fn) = gctx.ctx.get_function_ast(&format!("{sname}.len")) else {
        return false;
    };
    if !len_fn.is_cfn || len_fn.args.is_empty() {
        return false;
    }
    let Some(fields) = gctx.ctx.struct_fields(container_ty) else {
        return false;
    };
    let mut data_idx: Option<usize> = None;
    let mut elem_ty = TypeIdx::NONE;
    for (i, (_, fty)) in fields.iter().enumerate() {
        let k = gctx.ctx.type_pool.get(*fty);
        if k.kind == TypeKind::PtrMany {
            if data_idx.is_some() {
                return false; // ambiguous
            }
            data_idx = Some(i);
            elem_ty = TypeIdx::new(k.a);
        }
    }
    let Some(data_idx) = data_idx else {
        return false;
    };
    if !gctx.ctx.type_needs_drop(elem_ty) {
        return false;
    }
    // count = self.len()  (receiver is the pointer-to-self we hold).
    let Ok(recv_abi) = classify_param(len_fn.args[0].mode, container_ty, gctx.ctx) else {
        return false;
    };
    let recv = if recv_abi.kind == ParamAbiKind::ByPointer {
        ptr
    } else {
        emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: ptr,
                ty: container_ty,
                ..Default::default()
            },
        )
    };
    let Ok(count) = emit_call(gctx, &len_fn, &[recv], NO_JIR_REF) else {
        return false;
    };
    let count_ty = gctx.jfn.get_inst(count).ty;
    // base = load self.<data> -> a `*mut[] Elem` value.
    let data_field_ty = fields[data_idx].1;
    let data_addr_ty = gctx.ctx.type_pool.intern_ptr_single(data_field_ty);
    let data_addr = emit(
        gctx,
        JirInst {
            tag: JirTag::FieldAddr,
            a: ptr,
            b: data_idx as u32,
            ty: data_addr_ty,
            ..Default::default()
        },
    );
    let base = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: data_addr,
            ty: data_field_ty,
            ..Default::default()
        },
    );
    // i = 0
    let slot = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty: count_ty,
            ..Default::default()
        },
    );
    let zero = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 0,
            ty: count_ty,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: zero,
            ..Default::default()
        },
    );
    let cond_b = gctx.jfn.push_block("dropcond");
    let body_b = gctx.jfn.push_block("dropbody");
    let exit_b = gctx.jfn.push_block("dropexit");
    emit_br(gctx, cond_b);
    // cond: i < count
    gctx.current_block = cond_b;
    let i_val = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: count_ty,
            ..Default::default()
        },
    );
    let cmp = emit(
        gctx,
        JirInst {
            tag: JirTag::ICmpUlt,
            a: i_val,
            b: count,
            ty: builtin::BOOL,
            ..Default::default()
        },
    );
    emit_cond_br(gctx, cmp, body_b, exit_b);
    // body: drop element i; i += 1
    gctx.current_block = body_b;
    let i_body = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: count_ty,
            ..Default::default()
        },
    );
    let elem_ptr_ty = gctx.ctx.type_pool.intern_ptr_many(elem_ty);
    let elem_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::IndexAddr,
            a: base,
            b: i_body,
            ty: elem_ptr_ty,
            ..Default::default()
        },
    );
    emit_drop_in_place(gctx, elem_ptr, elem_ty);
    let cur = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: count_ty,
            ..Default::default()
        },
    );
    let one = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 1,
            ty: count_ty,
            ..Default::default()
        },
    );
    let next = emit(
        gctx,
        JirInst {
            tag: JirTag::Add,
            a: cur,
            b: one,
            ty: count_ty,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: next,
            ..Default::default()
        },
    );
    emit_br(gctx, cond_b);
    gctx.current_block = exit_b;
    true
}

/// Deep-clone `*src` of type `ty` into `*dest` (the C++ `emitCloneInto`): plain
/// data bitwise-copies; arrays / enum payloads / struct fields recurse; a type
/// with its own `cfn clone` calls it; a drop-bearing struct with NO clone recipe
/// is an error (it owns resources). Returns Err on the error tier (the caller
/// withdraws the conditional method).
fn emit_clone_into(
    gctx: &mut AstGenCtx,
    src: JirRef,
    dest: JirRef,
    ty: TypeIdx,
) -> Result<(), String> {
    let mut ty = gctx.ctx.apply_current_subst(ty);
    if gctx.ctx.type_pool.get(ty).kind == TypeKind::GenericCall {
        ty = gctx.ctx.resolve_generic_call_instantiate(ty)?;
    }
    if gctx.ctx.type_pool.get(ty).kind == TypeKind::ArrayExpr {
        ty = gctx.ctx.resolve_array_expr_instantiate(ty)?;
    }
    // Tier 1: plain data — a bitwise Load+Store is a true value copy.
    if !gctx.ctx.type_needs_drop(ty) {
        let v = emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: src,
                ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: dest,
                b: v,
                ..Default::default()
            },
        );
        return Ok(());
    }
    let k = gctx.ctx.type_pool.get(ty);
    // Fixed-size array: clone element-wise.
    if k.kind == TypeKind::Array {
        let (elem, n) = (TypeIdx::new(k.a), k.b);
        let elem_ptr = gctx.ctx.type_pool.intern_ptr_single(elem);
        for i in 0..n {
            let idx = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: i,
                    ty: builtin::U64,
                    ..Default::default()
                },
            );
            let se = emit(
                gctx,
                JirInst {
                    tag: JirTag::IndexAddr,
                    a: src,
                    b: idx,
                    ty: elem_ptr,
                    ..Default::default()
                },
            );
            let de = emit(
                gctx,
                JirInst {
                    tag: JirTag::IndexAddr,
                    a: dest,
                    b: idx,
                    ty: elem_ptr,
                    ..Default::default()
                },
            );
            emit_clone_into(gctx, se, de, elem)?;
        }
        return Ok(());
    }
    // Enum: bitwise-copy the whole value, then re-clone each live variant's
    // droppable payload over the raw copy (tag-dispatched at byte offsets).
    if let Some(en) = gctx.ctx.enum_name_of(ty) {
        let whole = emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: src,
                ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: dest,
                b: whole,
                ..Default::default()
            },
        );
        if gctx.ctx.enum_has_payload(ty) != Some(true) {
            return Ok(());
        }
        let u8_ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
        let tag_ptr = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: src,
                b: 0,
                ty: u8_ptr,
                ..Default::default()
            },
        );
        let tag_val = emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: tag_ptr,
                ty: builtin::U8,
                ..Default::default()
            },
        );
        let variants = gctx.ctx.enum_variants_by_name(&en).unwrap_or_default();
        for v in &variants {
            if !v
                .payload_types
                .iter()
                .any(|&pt| gctx.ctx.type_needs_drop(pt))
            {
                continue;
            }
            let clone_b = gctx.jfn.push_block("eclone");
            let cont_b = gctx.jfn.push_block("eclonecont");
            let disc = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: v.discriminant,
                    ty: builtin::U8,
                    ..Default::default()
                },
            );
            let cmp = emit(
                gctx,
                JirInst {
                    tag: JirTag::ICmpEq,
                    a: tag_val,
                    b: disc,
                    ty: builtin::BOOL,
                    ..Default::default()
                },
            );
            emit_cond_br(gctx, cmp, clone_b, cont_b);
            gctx.current_block = clone_b;
            let s_pay = emit(
                gctx,
                JirInst {
                    tag: JirTag::FieldAddr,
                    a: src,
                    b: 1,
                    ty: u8_ptr,
                    ..Default::default()
                },
            );
            let d_pay = emit(
                gctx,
                JirInst {
                    tag: JirTag::FieldAddr,
                    a: dest,
                    b: 1,
                    ty: u8_ptr,
                    ..Default::default()
                },
            );
            let mut off: u64 = 0;
            for &pt in &v.payload_types {
                let sz = gctx.ctx.type_size(pt)?;
                let al = gctx.ctx.type_align(pt)?;
                off = off.div_ceil(al) * al;
                if gctx.ctx.type_needs_drop(pt) {
                    let offr = emit(
                        gctx,
                        JirInst {
                            tag: JirTag::Int,
                            a: off as u32,
                            ty: builtin::U64,
                            ..Default::default()
                        },
                    );
                    let sf = emit(
                        gctx,
                        JirInst {
                            tag: JirTag::IndexAddr,
                            a: s_pay,
                            b: offr,
                            ty: u8_ptr,
                            ..Default::default()
                        },
                    );
                    let df = emit(
                        gctx,
                        JirInst {
                            tag: JirTag::IndexAddr,
                            a: d_pay,
                            b: offr,
                            ty: u8_ptr,
                            ..Default::default()
                        },
                    );
                    emit_clone_into(gctx, sf, df, pt)?;
                }
                off += sz;
            }
            emit_br(gctx, cont_b);
            gctx.current_block = cont_b;
        }
        return Ok(());
    }
    // Struct.
    let Some(sname) = gctx.ctx.struct_name_of(ty) else {
        return Err("astgen: cannot clone this type — no structural clone is available".into());
    };
    // Tier 3: the type's own `cfn clone` — an in-struct `Name.clone` method, or the
    // top-level `cfn clone(self: T) T` from the clone registry.
    if let Some(clone_fn) = gctx
        .ctx
        .get_function_ast(&format!("{sname}.clone"))
        .or_else(|| gctx.ctx.lookup_clone_fn(&sname))
    {
        let result = emit_call(gctx, &clone_fn, &[src], NO_JIR_REF)?;
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: dest,
                b: result,
                ..Default::default()
            },
        );
        return Ok(());
    }
    // Error tier: owns a resource (its own drop), no clone recipe.
    if gctx.ctx.lookup_drop_fn_name(ty).is_some() {
        return Err(format!(
            "`{sname}` owns resources (it has `cfn drop`); define `cfn clone(self: Self) Self` to make it cloneable"
        ));
    }
    // Tier 2: field-wise structural clone.
    let fields = gctx.ctx.struct_fields(ty).unwrap_or_default();
    for (i, (_, fty)) in fields.iter().enumerate() {
        let fty = *fty;
        let fptr = gctx.ctx.type_pool.intern_ptr_single(fty);
        let sf = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: src,
                b: i as u32,
                ty: fptr,
                ..Default::default()
            },
        );
        let df = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: dest,
                b: i as u32,
                ty: fptr,
                ..Default::default()
            },
        );
        emit_clone_into(gctx, sf, df, fty)?;
    }
    Ok(())
}

/// Built-in `recv.clone()` for receivers WITHOUT a user `cfn clone` (the C++
/// `tryLowerBuiltinClone`). Returns `Some(NO_JIR_REF-or-value)`; `Ok(None)` when
/// a user clone exists (caller does ordinary dispatch). Tier 1 returns the value
/// directly; tier 2 glues into a fresh slot and returns the dest pointer.
fn try_lower_builtin_clone(
    gctx: &mut AstGenCtx,
    recv_ptr: JirRef,
    recv_val: JirRef,
    recv_ty: TypeIdx,
) -> Result<Option<JirRef>, String> {
    let mut ty = gctx.ctx.apply_current_subst(recv_ty);
    if gctx.ctx.type_pool.get(ty).kind == TypeKind::GenericCall {
        ty = gctx.ctx.resolve_generic_call_instantiate(ty)?;
    }
    // A user in-struct `cfn clone` wins — let ordinary dispatch handle it.
    if let Some(sname) = gctx.ctx.struct_name_of(ty)
        && gctx
            .ctx
            .get_function_ast(&format!("{sname}.clone"))
            .is_some()
    {
        return Ok(None);
    }
    // Tier 1: plain data — the clone IS the value.
    if !gctx.ctx.type_needs_drop(ty) {
        if recv_val != NO_JIR_REF {
            return Ok(Some(recv_val));
        }
        return Ok(Some(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: recv_ptr,
                ty,
                ..Default::default()
            },
        )));
    }
    // Tier 2: glue into a fresh slot (spill a value-form receiver first).
    let src = if recv_ptr != NO_JIR_REF {
        recv_ptr
    } else {
        let s = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: s,
                b: recv_val,
                ..Default::default()
            },
        );
        s
    };
    let dest = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty,
            ..Default::default()
        },
    );
    emit_clone_into(gctx, src, dest, ty)?;
    Ok(Some(dest))
}

/// Drop each droppable field of struct `*ptr` (FieldAddr + recursion).
fn emit_field_drops(gctx: &mut AstGenCtx, ptr: JirRef, ty: TypeIdx) {
    let Some(fields) = gctx.ctx.struct_fields(ty) else {
        return;
    };
    for (i, (_, fty)) in fields.iter().enumerate() {
        if !gctx.ctx.type_needs_drop(*fty) {
            continue;
        }
        let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(*fty);
        let field_ptr = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: ptr,
                b: i as u32,
                ty: ptr_ty,
                ..Default::default()
            },
        );
        emit_drop_in_place(gctx, field_ptr, *fty);
    }
}

/// Try to fold an expression node to a JIR ref producing its value.
fn astgen_expr(gctx: &mut AstGenCtx, node: NodeIdx, expected: TypeIdx) -> Result<JirRef, String> {
    if node.is_none() {
        return Err("astgen: null expression node".into());
    }
    let n = *gctx.ctx.node_store.get(node);
    gctx.current_node = node;
    match n.tag {
        AstTag::NumberLit => astgen_number_lit(gctx, &n, expected),
        AstTag::BoolLit => {
            let inst = JirInst {
                tag: JirTag::Bool,
                a: if n.lhs != 0 { 1 } else { 0 },
                ty: builtin::BOOL,
                ..Default::default()
            };
            Ok(emit(gctx, inst))
        }
        AstTag::StringLit => astgen_string_lit(gctx, &n, expected),
        AstTag::Variable => astgen_variable(gctx, &n, expected),
        AstTag::BinaryOp => astgen_binary_op(gctx, &n, expected),
        AstTag::UnaryOp => astgen_unary_op(gctx, &n, expected),
        AstTag::AsCast => astgen_as_cast(gctx, &n),
        AstTag::Call => astgen_call(gctx, &n, NO_JIR_REF),
        AstTag::TypeMethodCall => astgen_type_method_call(gctx, &n, NO_JIR_REF),
        AstTag::AtCall => astgen_at_call(gctx, &n),
        AstTag::StructLit => astgen_struct_lit(gctx, &n, expected),
        AstTag::ArrayLit => astgen_array_lit(gctx, &n, expected),
        AstTag::ArrayRepeat => astgen_array_repeat(gctx, &n, expected),
        AstTag::Index => astgen_index(gctx, &n),
        AstTag::MatchNode => astgen_match(gctx, &n, expected),
        AstTag::MemberAccess => astgen_member_access(gctx, &n),
        AstTag::Deref => astgen_deref(gctx, &n),
        AstTag::AddressOf => astgen_address_of(gctx, &n),
        // `astgen_slice` is implemented but NOT wired: it unblocks the large std
        // files (test_os_intrinsics/print/fs/...) past the slice node into deeper
        // string-pool intern-order divergences. They must stay fully unported
        // until those deeper features land, so leave Slice deferred to keep the
        // gate green; flip this on together with the std-file intern-order work.
        AstTag::Slice => astgen_slice(gctx, &n),
        AstTag::Return => {
            astgen_return(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::VarDecl => {
            astgen_var_decl(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::Assign => {
            astgen_assign(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::IfNode => {
            astgen_if(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::WhileNode => {
            astgen_while(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::ForNode => {
            astgen_for(gctx, &n)?;
            Ok(NO_JIR_REF)
        }
        AstTag::Break => {
            astgen_break(gctx)?;
            Ok(NO_JIR_REF)
        }
        AstTag::Continue => {
            astgen_continue(gctx)?;
            Ok(NO_JIR_REF)
        }
        other => Err(format!("astgen: node {other:?} not yet ported")),
    }
}

fn astgen_number_lit(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let val = (n.lhs as u64) | ((n.rhs as u64) << 32);
    let is_neg = n.flags & 1 != 0;
    let is_float = n.flags & 2 != 0;

    if is_float {
        // Coercion point: round once to the target f32/f64 (no double-round).
        let to_f32 = expected == builtin::F32;
        let d = if n.flags & 4 != 0 {
            let ei = n.lhs;
            let quad = [
                gctx.ctx.node_store.get_extra(ExtraIdx::new(ei)),
                gctx.ctx.node_store.get_extra(ExtraIdx::new(ei + 1)),
                gctx.ctx.node_store.get_extra(ExtraIdx::new(ei + 2)),
                gctx.ctx.node_store.get_extra(ExtraIdx::new(ei + 3)),
            ];
            quad_to_target_as_double(&quad, to_f32)
        } else {
            f64::from_bits(val)
        };
        let bits = d.to_bits();
        let inst = JirInst {
            tag: JirTag::Float,
            a: (bits & 0xFFFF_FFFF) as u32,
            b: (bits >> 32) as u32,
            flags: if is_neg { 1 } else { 0 },
            ty: if to_f32 { builtin::F32 } else { builtin::F64 },
            ..Default::default()
        };
        return Ok(emit(gctx, inst));
    }

    let mut inst = JirInst {
        tag: JirTag::Int,
        a: (val & 0xFFFF_FFFF) as u32,
        b: (val >> 32) as u32,
        flags: if is_neg { 1 } else { 0 },
        ..Default::default()
    };
    // With an integer `expected` we honour it directly (peer-type propagation).
    // GenericCall destinations and type-alias chains resolve first so the
    // contract holds transitively; these route through context stubs that are
    // no-ops today, so a builtin int resolves immediately.
    if !expected.is_none() {
        let mut resolved = expected;
        if gctx.ctx.type_pool.get(resolved).kind == TypeKind::GenericCall {
            let r = gctx.ctx.resolve_generic_call(resolved);
            if !r.is_none() {
                resolved = r;
            }
        }
        // A Named expected may be a type-alias chain ending at an integer
        // (`const Flag = u64; var f: Flag = 100000;`). Chase it, requalifying
        // per hop so the alias resolves in the right module.
        for _ in 0..8 {
            let module = gctx.ctx.current_body_module();
            let rq = gctx.ctx.requalify_type(resolved, &module);
            let (kn_kind, kn_a) = {
                let kn = gctx.ctx.type_pool.get(rq);
                (kn.kind, kn.a)
            };
            if kn_kind != TypeKind::Named {
                resolved = rq;
                break;
            }
            let name = str_at(gctx, kn_a);
            let target = gctx.ctx.lookup_type_alias(&name);
            if target.is_none() || target == rq {
                resolved = rq;
                break;
            }
            resolved = target;
        }
        if gctx.ctx.type_pool.get(resolved).kind == TypeKind::Int {
            inst.ty = resolved;
            return Ok(emit(gctx, inst));
        }
        // Int literal into a float destination stays permissive — falls through
        // to smallest-fit, and the consumer emits the SIToFP / UIToFP.
    }
    // Smallest-fit width: the narrowest builtin that holds the magnitude.
    inst.ty = if is_neg {
        if val <= 128 {
            builtin::I8
        } else if val <= 32768 {
            builtin::I16
        } else if val <= 2_147_483_648 {
            builtin::I32
        } else {
            builtin::I64
        }
    } else if val <= 255 {
        builtin::U8
    } else if val <= 65535 {
        builtin::U16
    } else if val <= 4_294_967_295 {
        builtin::U32
    } else {
        builtin::U64
    };
    Ok(emit(gctx, inst))
}

/// Lower a `StringLit` to a `Str` inst typed as `[]u8` (a `{ptr,len}` slice).
/// Pointer decay: when the use site expects a `*const[] u8` / `*mut[] u8` /
/// `*const u8`, the literal lowers to the bare global pointer instead of a fat
/// slice (the C FFI hand-off), keyed on the expected type.
fn astgen_string_lit(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let mut result_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
    if !expected.is_none() {
        let ek = gctx.ctx.type_pool.get(expected);
        if (ek.kind == TypeKind::PtrMany || ek.kind == TypeKind::PtrSingle)
            && TypeIdx::new(ek.a) == builtin::U8
        {
            result_ty = expected;
        }
    }
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: n.lhs,
            ty: result_ty,
            ..Default::default()
        },
    ))
}

fn astgen_variable(gctx: &mut AstGenCtx, n: &AstNode, expected: TypeIdx) -> Result<JirRef, String> {
    let name = str_at(gctx, n.lhs);
    if let Some(&slot) = gctx.locals.get(&name) {
        let ty = gctx.local_types[&name];
        let inst = JirInst {
            tag: JirTag::Load,
            a: slot,
            ty,
            ..Default::default()
        };
        return Ok(emit(gctx, inst));
    }
    // Function-local comp binding (shadows module consts): materialize the value.
    if let Some(cv) = gctx.comp_scope.lookup(&name).cloned() {
        return materialize_comptime_value(gctx, &cv, expected, &format!("comp binding `{name}`"));
    }
    // Module-scope const: inline it. Resolve the CURRENT module's const first
    // (two modules may define the same name). A `comp` const folds to a single
    // value and materializes (narrowed to its declared type); a plain const
    // re-lowers its initializer (recursively, for sibling references).
    let bm = gctx.ctx.current_body_module();
    let mc = (!bm.is_empty())
        .then(|| gctx.ctx.get_module_const(&format!("{bm}.{name}")))
        .flatten()
        .or_else(|| gctx.ctx.get_module_const(&name));
    if let Some(mc) = mc {
        if mc.is_comp {
            let v = gctx.ctx.fold_comptime_expr(mc.init_expr);
            if !v.is_none() {
                // A `comp const` (width u64 for an untyped int literal)
                // materializes narrowed to the USE SITE's expected type first,
                // falling back to the const's own declared type — the C++
                // `want = (expected != kNoType) ? expected : declaredType`
                // (astgen.cpp:1424). Without this, `var n: u32 = N` stores an
                // i64 into the i32 slot.
                let want = if expected.is_none() {
                    mc.declared_type
                } else {
                    expected
                };
                return materialize_comptime_value(gctx, &v, want, &format!("comp const `{name}`"));
            }
        }
        return astgen_expr(gctx, mc.init_expr, mc.declared_type);
    }
    // Fn-name as a value: a typed function pointer when the use site expects a
    // `Fn`, else the legacy raw u64 address. (Generic fns have no body to point
    // at here.)
    if let Some(f) = gctx.ctx.get_function_ast(&name) {
        if f.is_generic() {
            // Recoverable (the C++ astgen.cpp:1444).
            return Ok(recover_here(
                gctx,
                format!("cannot take address of generic fn `{name}`"),
                TypeIdx::NONE,
            ));
        }
        let sid = gctx.ctx.string_pool.intern(name.as_bytes());
        let expect_fn =
            !expected.is_none() && gctx.ctx.type_pool.get(expected).kind == TypeKind::Fn;
        let ty = if expect_fn { expected } else { builtin::U64 };
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::FnRef,
                a: sid.raw(),
                ty,
                ..Default::default()
            },
        ));
    }
    // Recoverable (the C++ astgen.cpp:1466): emit a Poison so the rest of the
    // function still gets analyzed and additional errors report in one pass.
    Ok(recover_here(
        gctx,
        format!("unknown variable `{name}`"),
        TypeIdx::NONE,
    ))
}

/// Emit a JIR constant for a folded compile-time value, narrowed to `expected`
/// when it is a matching scalar (the C++ `materializeComptimeValue`).
/// Does comp int `bits` (interpreted per `is_signed`) fit a `width`-bit integer
/// of signedness `target_signed`? Mirrors the C++ `compIntFits` (astgen.cpp).
fn comp_int_fits(bits: u64, is_signed: bool, width: u16, target_signed: bool) -> bool {
    if width >= 64 {
        if target_signed {
            return is_signed || bits <= i64::MAX as u64;
        }
        return !is_signed || (bits as i64) >= 0;
    }
    if is_signed {
        let x = bits as i64;
        if target_signed {
            let min = -(1i64 << (width - 1));
            let max = (1i64 << (width - 1)) - 1;
            return x >= min && x <= max;
        }
        if x < 0 {
            return false;
        }
        return (x as u64) <= ((1u64 << width) - 1);
    }
    if target_signed {
        return bits <= ((1u64 << (width - 1)) - 1);
    }
    bits <= ((1u64 << width) - 1)
}

/// Decimal spelling of a comp int for diagnostics (C++ `compIntToString`).
fn comp_int_to_string(bits: u64, is_signed: bool) -> String {
    if is_signed {
        (bits as i64).to_string()
    } else {
        bits.to_string()
    }
}

/// Chase a declared annotation through generic-call / substitution / type-alias
/// links to its underlying scalar (the C++ `resolveScalarExpected`).
fn resolve_scalar_expected(ctx: &CodegenContext, mut t: TypeIdx) -> TypeIdx {
    let mut guard = 0;
    while guard < 8 && !t.is_none() {
        guard += 1;
        let k = ctx.type_pool.get(t);
        if k.kind == TypeKind::GenericCall {
            let r = ctx.resolve_generic_call(t);
            if r.is_none() || r == t {
                return t;
            }
            t = r;
            continue;
        }
        if k.kind == TypeKind::Named {
            let nm =
                String::from_utf8_lossy(&ctx.string_pool.get(StringIdx::new(k.a))).into_owned();
            let sub = ctx.lookup_current_subst(&nm);
            if !sub.is_none() && sub != t {
                t = sub;
                continue;
            }
            let alias = ctx.lookup_type_alias(&nm);
            if !alias.is_none() && alias != t {
                t = alias;
                continue;
            }
            return t;
        }
        return t;
    }
    t
}

/// Coerce a freshly evaluated comp value to a declared annotation
/// (`comp const N: u8 = 200;`) — scalar kinds only, with hard-error misfits.
/// Ports the C++ `coerceCompToDeclared` (astgen.cpp:909-963). The returned value
/// carries the declared width/signedness for ints; the error strings match the
/// oracle byte-for-byte.
fn coerce_comp_to_declared(
    gctx: &mut AstGenCtx,
    v: ComptimeValue,
    declared: TypeIdx,
    name: &str,
) -> Result<ComptimeValue, String> {
    let resolved = resolve_scalar_expected(gctx.ctx, declared);
    let k = gctx.ctx.type_pool.get(resolved);
    match k.kind {
        TypeKind::Int => {
            let ComptimeValue::Int {
                bits, is_signed, ..
            } = v
            else {
                return Err(format!(
                    "comp binding `{name}` declared as an integer type but its initializer is not an integer"
                ));
            };
            let ew = k.a as u16;
            let es = k.b != 0;
            if !comp_int_fits(bits, is_signed, ew, es) {
                return Err(format!(
                    "comp binding `{name}` value {} does not fit {}{}",
                    comp_int_to_string(bits, is_signed),
                    if es { "i" } else { "u" },
                    ew
                ));
            }
            Ok(ComptimeValue::Int {
                bits,
                width: ew,
                is_signed: es,
            })
        }
        TypeKind::Float => {
            let ComptimeValue::Float { value, .. } = v else {
                return Err(format!(
                    "comp binding `{name}` declared as a float type; use a float literal (e.g. `3.0`)"
                ));
            };
            Ok(ComptimeValue::Float {
                value,
                width: k.a as u16,
            })
        }
        TypeKind::Bool => {
            if !matches!(v, ComptimeValue::Bool(_)) {
                return Err(format!(
                    "comp binding `{name}` declared as bool but its initializer is not a bool"
                ));
            }
            Ok(v)
        }
        TypeKind::Slice => {
            if k.a == builtin::U8.index() as u32 && matches!(v, ComptimeValue::Str(_)) {
                return Ok(v);
            }
            Err(format!(
                "comp binding `{name}` declared as a slice type; only `[]u8` (str) comp values are supported"
            ))
        }
        _ => Err(format!(
            "comp binding `{name}` has an unsupported declared type — comp bindings hold int / float / bool / str values"
        )),
    }
}

fn materialize_comptime_value(
    gctx: &mut AstGenCtx,
    cv: &ComptimeValue,
    expected: TypeIdx,
    what: &str,
) -> Result<JirRef, String> {
    match cv {
        ComptimeValue::Int {
            bits,
            width,
            is_signed,
        } => {
            let (mut w, mut s) = (*width, *is_signed);
            if !expected.is_none() {
                let resolved = resolve_scalar_expected(gctx.ctx, expected);
                let ek = gctx.ctx.type_pool.get(resolved);
                if ek.kind == TypeKind::Int {
                    let ew = ek.a as u16;
                    let es = ek.b != 0;
                    // A comp value that can't represent the expected width is a
                    // hard error, not a silent truncation (the C++
                    // materializeComptimeValue fit-check, astgen.cpp:785+).
                    if !comp_int_fits(*bits, *is_signed, ew, es) {
                        // Recoverable (the C++ astgen.cpp:800).
                        return Ok(recover_here(
                            gctx,
                            format!(
                                "{what} has value {}, which does not fit the expected {}{}",
                                comp_int_to_string(*bits, *is_signed),
                                if es { "i" } else { "u" },
                                ew
                            ),
                            resolved,
                        ));
                    }
                    w = ew;
                    s = es;
                }
            }
            let ty = gctx.ctx.type_pool.intern_int(w, s);
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: (*bits & 0xFFFF_FFFF) as u32,
                    b: (*bits >> 32) as u32,
                    ty,
                    ..Default::default()
                },
            ))
        }
        ComptimeValue::Bool(v) => Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Bool,
                a: u32::from(*v),
                ty: builtin::BOOL,
                ..Default::default()
            },
        )),
        ComptimeValue::Float { value, width } => {
            let mut w = *width;
            if !expected.is_none() {
                let resolved = resolve_scalar_expected(gctx.ctx, expected);
                let ek = gctx.ctx.type_pool.get(resolved);
                if ek.kind == TypeKind::Float {
                    w = ek.a as u16;
                }
            }
            // f32 narrowing rounds once through the target precision.
            let d = if w == 32 {
                *value as f32 as f64
            } else {
                *value
            };
            let bits = d.to_bits();
            let ty = if w == 32 { builtin::F32 } else { builtin::F64 };
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Float,
                    a: (bits & 0xFFFF_FFFF) as u32,
                    b: (bits >> 32) as u32,
                    ty,
                    ..Default::default()
                },
            ))
        }
        ComptimeValue::Str(sid) => {
            // Slice-of-u8 by default; decay to a bare u8 pointer when expected.
            let mut result_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
            if !expected.is_none() {
                let ek = gctx.ctx.type_pool.get(expected);
                if (ek.kind == TypeKind::PtrMany || ek.kind == TypeKind::PtrSingle)
                    && TypeIdx::new(ek.a) == builtin::U8
                {
                    result_ty = expected;
                }
            }
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Str,
                    a: sid.raw(),
                    ty: result_ty,
                    ..Default::default()
                },
            ))
        }
        // Type values have no runtime representation — a value position
        // consuming one is an error, not a lowering (the C++ astgen.cpp:860).
        ComptimeValue::Type(_) => Ok(recover_here(
            gctx,
            format!("{what} is of type `type` and has no runtime representation"),
            TypeIdx::NONE,
        )),
        _ => Ok(recover_here(
            gctx,
            format!("{what} has no runtime lowering yet"),
            TypeIdx::NONE,
        )),
    }
}

fn astgen_binary_op(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let mut lhs_ref = astgen_expr(gctx, NodeIdx::new(n.lhs), expected)?;
    // Peer-type hint: lower the RHS at the LHS's resolved width.
    let mut lhs_type = gctx.jfn.get_inst(lhs_ref).ty;
    let mut rhs_ref = astgen_expr(gctx, NodeIdx::new(n.rhs), lhs_type)?;

    // Width reconciliation: floats must match exactly; integers widen the
    // narrower side (sext if signed, else zext).
    let rhs_type = gctx.jfn.get_inst(rhs_ref).ty;
    let lk = gctx.ctx.type_pool.get(lhs_type);
    let rk = gctx.ctx.type_pool.get(rhs_type);
    if lk.kind == TypeKind::Float && rk.kind == TypeKind::Float && lk.a != rk.a {
        return Err("astgen: mismatched float widths; use an explicit `as` cast".into());
    }
    if lk.kind == TypeKind::Int && rk.kind == TypeKind::Int && lk.a != rk.a {
        if lk.a < rk.a {
            let tag = if lk.b != 0 {
                JirTag::SExt
            } else {
                JirTag::ZExt
            };
            lhs_ref = emit(
                gctx,
                JirInst {
                    tag,
                    a: lhs_ref,
                    ty: rhs_type,
                    ..Default::default()
                },
            );
            lhs_type = rhs_type;
        } else {
            let tag = if rk.b != 0 {
                JirTag::SExt
            } else {
                JirTag::ZExt
            };
            rhs_ref = emit(
                gctx,
                JirInst {
                    tag,
                    a: rhs_ref,
                    ty: lhs_type,
                    ..Default::default()
                },
            );
        }
    }

    // Short-circuit LogAnd (11) / LogOr (12) lower as an if-expression over a
    // bool result slot. Both operands were eagerly emitted above (and are i1,
    // so the reconciliation was a no-op); the RHS value is *stored* only on the
    // branch that needs it — `LogAnd: lhs ? rhs : false`, `LogOr: lhs ? true :
    // rhs`. (jir_codegen folds the slot to a phi at -O2.)
    if n.op == 11 || n.op == 12 {
        let is_and = n.op == 11;
        let res_slot = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: builtin::BOOL,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: res_slot,
                b: lhs_ref,
                ..Default::default()
            },
        );
        let rhs_b = gctx
            .jfn
            .push_block(if is_and { "and.rhs" } else { "or.rhs" });
        let end_b = gctx
            .jfn
            .push_block(if is_and { "and.end" } else { "or.end" });
        if is_and {
            emit_cond_br(gctx, lhs_ref, rhs_b, end_b);
        } else {
            emit_cond_br(gctx, lhs_ref, end_b, rhs_b);
        }
        gctx.current_block = rhs_b;
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: res_slot,
                b: rhs_ref,
                ..Default::default()
            },
        );
        emit_br(gctx, end_b);
        gctx.current_block = end_b;
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: res_slot,
                ty: builtin::BOOL,
                ..Default::default()
            },
        ));
    }

    let k = gctx.ctx.type_pool.get(lhs_type);
    let is_float = k.kind == TypeKind::Float;
    let is_signed = k.kind == TypeKind::Int && k.b != 0;

    // `n.op` is the BinOp discriminant (1=Add..18=Ge; see ast_flat BinOp).
    use JirTag::*;
    let (tag, is_cmp): (JirTag, bool) = match n.op {
        1 => (if is_float { FAdd } else { Add }, false),
        2 => (if is_float { FSub } else { Sub }, false),
        3 => (if is_float { FMul } else { Mul }, false),
        4 => (
            if is_float {
                FDiv
            } else if is_signed {
                SDiv
            } else {
                UDiv
            },
            false,
        ),
        5 => (
            if is_float {
                FRem
            } else if is_signed {
                SRem
            } else {
                URem
            },
            false,
        ),
        6 => (BitAnd, false),
        7 => (BitOr, false),
        8 => (BitXor, false),
        9 => (Shl, false),
        10 => (if is_signed { AShr } else { LShr }, false),
        13 => (if is_float { FCmpOeq } else { ICmpEq }, true),
        14 => (if is_float { FCmpOne } else { ICmpNe }, true),
        15 => (
            if is_float {
                FCmpOlt
            } else if is_signed {
                ICmpSlt
            } else {
                ICmpUlt
            },
            true,
        ),
        16 => (
            if is_float {
                FCmpOle
            } else if is_signed {
                ICmpSle
            } else {
                ICmpUle
            },
            true,
        ),
        17 => (
            if is_float {
                FCmpOgt
            } else if is_signed {
                ICmpSgt
            } else {
                ICmpUgt
            },
            true,
        ),
        18 => (
            if is_float {
                FCmpOge
            } else if is_signed {
                ICmpSge
            } else {
                ICmpUge
            },
            true,
        ),
        11 | 12 => unreachable!("LogAnd/LogOr handled before the operator table"),
        other => return Err(format!("astgen: unsupported binary operator {other}")),
    };
    let result_ty = if is_cmp { builtin::BOOL } else { lhs_type };
    Ok(emit(
        gctx,
        JirInst {
            tag,
            a: lhs_ref,
            b: rhs_ref,
            ty: result_ty,
            ..Default::default()
        },
    ))
}

/// Resolve a declared/initializer type through generic-call, type-alias,
/// generic-substitution, and module re-export chains for the var-decl
/// declared-vs-initializer comparison (the C++ `resolveForCmp` lambda,
/// astgen.cpp:1252-1289). Both sides are run through this so `var a:
/// Identity(i32) = 42;` compares `i32 == i32` rather than `GenericCall != Int`.
fn resolve_for_cmp(ctx: &CodegenContext, t: TypeIdx) -> TypeIdx {
    if t.is_none() {
        return t;
    }
    // Qualify bare user-type references first so a declared bare `Color` and an
    // initializer typed with the qualified `mod.Color` compare equal. No-op for
    // already-qualified, substitution, and primitive types.
    let t = ctx.requalify_type(t, &ctx.current_body_module());
    let k = ctx.type_pool.get(t);
    if k.kind == TypeKind::Named {
        let name = String::from_utf8_lossy(&ctx.string_pool.get(StringIdx::new(k.a))).into_owned();
        // Generic substitution wins (inside an instantiated method body, `T`
        // resolves to whatever the instantiation supplied).
        let sub = ctx.lookup_current_subst(&name);
        if !sub.is_none() {
            return resolve_for_cmp(ctx, sub);
        }
    }
    if k.kind == TypeKind::GenericCall {
        let r = ctx.resolve_generic_call(t);
        if !r.is_none() {
            return resolve_for_cmp(ctx, r);
        }
    }
    if k.kind == TypeKind::Named {
        let name = String::from_utf8_lossy(&ctx.string_pool.get(StringIdx::new(k.a))).into_owned();
        let a = ctx.lookup_type_alias(&name);
        if !a.is_none() {
            return resolve_for_cmp(ctx, a);
        }
        // 3+ segment chain through module re-exports — collapse `w.leaf.Point`
        // to the canonical `Point`.
        if name.contains('.') {
            let c = ctx.resolve_chained_type(&name);
            if !c.is_none() {
                return resolve_for_cmp(ctx, c);
            }
        }
    }
    t
}

/// Lower a local `var`/`const` declaration (extra layout `[nameId, declared,
/// initIdx]`). For a declared type the slot is registered *before* the
/// initializer is lowered (so a self-referential init can find it); inferred
/// types register after, once the init's result type is known. An alloca plus
/// a value Store is emitted — `jir_codegen` lowers it correctly for both
/// scalars and (via memcpy) aggregates.
fn astgen_var_decl(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let extra = n.lhs;
    let name_id = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra));
    // The declared type stays LITERAL inside an instantiated body (the C++
    // astgenVarDecl never substitutes it): `var out: Vec(T)` keeps `Vec(T)`,
    // so the slot's AddrOf receiver interns the literal `*Vec(T)` GenericCall in
    // the oracle's TypePool order. Downstream registry lookups requalify/resolve
    // at use, so the bare literal slot type is still correct.
    let declared = TypeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1)));
    let init_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 2)));
    let name = str_at(gctx, name_id);

    // Reject re-declaration within the same lexical scope only. We inspect the
    // innermost `local_scopes` frame, so `const a = X; var a = Y;` at the same
    // level errors, but
    //     fn f() { var x = 1; if (c) { var x = 2; } }
    // still compiles — intentional inner-block shadowing is allowed (the C++
    // astgen.cpp:1149).
    if let Some(frame) = gctx.local_scopes.last()
        && frame.contains(&name)
    {
        return Err(fail_node(
            gctx,
            gctx.current_node,
            &format!("redeclaration of `{name}` in the same scope"),
        ));
    }

    // `comp const X = E;` / `comp var X = E;` (rhs bit 1) — fold the initializer
    // and bind it in the comp scope (no runtime slot); uses inline the value.
    // Mirrors the C++ astgen.cpp:1157-1172: non-foldable inits are rejected
    // (anchored at the initializer), and a declared annotation coerces +
    // range-checks the value.
    if n.rhs & 2 != 0 {
        let is_const_binding = n.rhs & 1 != 0;
        let mut v = gctx.ctx.fold_comptime_expr_in(init_idx, &gctx.comp_scope);
        if v.is_none() {
            return Err(fail_node(
                gctx,
                init_idx,
                &format!("comp initializer of `{name}` must be a compile-time-known value"),
            ));
        }
        if !declared.is_none() {
            v = coerce_comp_to_declared(gctx, v, declared, &name)?;
        }
        gctx.comp_scope.bind(name.clone(), v);
        gctx.comp_bind_info.last_mut().unwrap().insert(
            name.clone(),
            CompBindingInfo {
                decl_depth: gctx.runtime_cond_depth,
                is_const: is_const_binding,
            },
        );
        if let Some(frame) = gctx.local_scopes.last_mut() {
            frame.insert(name);
        }
        return Ok(());
    }

    let (alloca_ref, ty) = if declared.is_none() {
        // Inferred: lower the init first so we have a concrete type to allocate,
        // then store the value (the place path needs a pre-existing slot).
        let init_ref = astgen_expr(gctx, init_idx, TypeIdx::NONE)?;
        let ty = gctx.jfn.get_inst(init_ref).ty;
        if ty.is_none() {
            return Err(format!(
                "could not infer type of `{name}`; add an explicit `: T` annotation"
            ));
        }
        let alloca_ref = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty,
                ..Default::default()
            },
        );
        gctx.locals.insert(name.clone(), alloca_ref);
        gctx.local_types.insert(name.clone(), ty);
        if let Some(frame) = gctx.local_scopes.last_mut() {
            frame.insert(name.clone());
        }
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: alloca_ref,
                b: init_ref,
                ..Default::default()
            },
        );
        (alloca_ref, ty)
    } else {
        // Declared: register the slot before lowering the init (self-ref). Try
        // the place-into-destination path first (StructLit / sret Call write
        // directly into the slot); fall back to value-compile + Store. The
        // declared type stays bare (the oracle keeps local-slot types bare;
        // registry lookups requalify at use). An `[expr]T` array type resolves
        // its comptime length so the slot is a concrete `[n]T`.
        let ty = if gctx.ctx.type_pool.get(declared).kind == TypeKind::ArrayExpr {
            gctx.ctx.resolve_array_expr_instantiate(declared)?
        } else {
            declared
        };
        let alloca_ref = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty,
                ..Default::default()
            },
        );
        gctx.locals.insert(name.clone(), alloca_ref);
        gctx.local_types.insert(name.clone(), ty);
        if let Some(frame) = gctx.local_scopes.last_mut() {
            frame.insert(name.clone());
        }
        if !astgen_expr_into_ptr(gctx, init_idx, ty, alloca_ref)? {
            let init_ref = astgen_expr(gctx, init_idx, ty)?;
            // Type-check the init against the declared type (the C++
            // astgen.cpp:1290-1327). The astgen_number_lit path already narrows
            // integer literals to the declared int width when `expected` is an
            // Int, so `var x: i32 = 5;` lands as I32. Anything else that doesn't
            // match is a real mismatch — `var x: f32 = 3;` (int into float),
            // `var x: i32 = 3.5;` (float into int), `const y: u32 = x`(i8), etc.
            // Both sides are resolved through generic-call / type-alias chains
            // first; PtrSingle(T)/PtrMany(T) share a representation so they're
            // leniently compatible (a zero-cost retag).
            let init_ty = gctx.jfn.get_inst(init_ref).ty;
            let decl_res = resolve_for_cmp(gctx.ctx, ty);
            let init_res = resolve_for_cmp(gctx.ctx, init_ty);
            let pointer_compatible = |a: TypeIdx, b: TypeIdx| -> bool {
                if a.is_none() || b.is_none() {
                    return false;
                }
                let ka = gctx.ctx.type_pool.get(a);
                let kb = gctx.ctx.type_pool.get(b);
                let a_ptr = ka.kind == TypeKind::PtrSingle || ka.kind == TypeKind::PtrMany;
                let b_ptr = kb.kind == TypeKind::PtrSingle || kb.kind == TypeKind::PtrMany;
                a_ptr && b_ptr && ka.a == kb.a
            };
            let types_match = decl_res == init_res || pointer_compatible(decl_res, init_res);
            if !init_ty.is_none() && !types_match {
                let dk = gctx.ctx.type_pool.get(decl_res);
                let ik = gctx.ctx.type_pool.get(init_res);
                if dk.kind == TypeKind::Float && ik.kind == TypeKind::Int {
                    return Err(format!(
                        "cannot assign integer to float-typed `{name}`; use a float literal (e.g. `3.0`) or an explicit `as` cast"
                    ));
                }
                return Err(format!(
                    "type mismatch in `{name}`: declared and initialised values disagree"
                ));
            }
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: alloca_ref,
                    b: init_ref,
                    ..Default::default()
                },
            );
        }
        (alloca_ref, ty)
    };

    // `var owned = c;` with a bare drop-bearing `c` MOVES it: the new binding
    // owns the value, so the source's scope-exit drop is suppressed. `var x = h.c`
    // (field extraction) has no per-field suppression — reject it instead (the C++
    // rejectDropBearingFieldExtract, astgen.cpp:1344).
    reject_drop_bearing_field_extract(gctx, init_idx, ty, "copy")?;
    consume_moved_variable(gctx, init_idx);
    // If this binding's type owns heap, track it for scope-exit cleanup (a
    // registered `cfn drop`, or field-recursive when the name is empty).
    if gctx.ctx.type_needs_drop(ty) {
        gctx.drop_scopes
            .last_mut()
            .expect("a drop scope is always active during a body")
            .push(DropTrack {
                var_name: name,
                slot: alloca_ref,
                ty,
            });
    }
    Ok(())
}

/// Resolve an assignable expression to its (pointer JirRef, leaf type) — the
/// `ResultLoc::Pointer` lowering of an lvalue:
///   * `Variable` — the local's alloca slot (no Load).
///   * `Deref` — the operand pointer value itself.
///   * `MemberAccess` — `FieldAddr` into a struct base, or a `BitCast` of a
///     union base (every union field shares the union's address).
///
/// `Index` lvalues (array/slice element addresses) land with array support.
fn astgen_lvalue(gctx: &mut AstGenCtx, node: NodeIdx) -> Result<(JirRef, TypeIdx), String> {
    let n = *gctx.ctx.node_store.get(node);
    match n.tag {
        AstTag::Variable => {
            let name = str_at(gctx, n.lhs);
            match gctx.locals.get(&name) {
                Some(&slot) => Ok((slot, gctx.local_types[&name])),
                None => Err(format!("astgen: unknown lvalue variable `{name}`")),
            }
        }
        AstTag::Deref => {
            let inner = astgen_expr(gctx, NodeIdx::new(n.lhs), TypeIdx::NONE)?;
            let pk = gctx.ctx.type_pool.get(gctx.jfn.get_inst(inner).ty);
            if pk.kind != TypeKind::PtrSingle && pk.kind != TypeKind::PtrMany {
                return Err("astgen: cannot deref non-pointer".into());
            }
            Ok((inner, TypeIdx::new(pk.a)))
        }
        AstTag::MemberAccess => {
            let (base_ptr, base_ty) = astgen_lvalue(gctx, NodeIdx::new(n.lhs))?;
            let member = str_at(gctx, n.rhs);
            // Union field lvalue: every field shares the union's address — the
            // field pointer IS the union pointer, just retyped.
            if gctx.ctx.is_union_registered(base_ty) {
                let field_ty = gctx
                    .ctx
                    .union_fields(base_ty)
                    .and_then(|fs| fs.into_iter().find(|(nm, _)| *nm == member).map(|(_, t)| t))
                    .ok_or_else(|| format!("astgen: union has no field `{member}`"))?;
                let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(field_ty);
                let p = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::BitCast,
                        a: base_ptr,
                        ty: ptr_ty,
                        ..Default::default()
                    },
                );
                return Ok((p, field_ty));
            }
            let fields = gctx
                .ctx
                .struct_fields(base_ty)
                .ok_or_else(|| "astgen: lvalue field access on non-struct".to_string())?;
            let Some(idx) = fields.iter().position(|(nm, _)| *nm == member) else {
                // Recoverable (the C++ astgen.cpp:2172).
                let sname = gctx.ctx.struct_name_of(base_ty).unwrap_or_default();
                let p = recover_here(
                    gctx,
                    format!("unknown field `{member}` on `{sname}`"),
                    TypeIdx::NONE,
                );
                return Ok((p, TypeIdx::NONE));
            };
            let mut field_ty = fields[idx].1;
            if gctx.ctx.type_pool.get(field_ty).kind == TypeKind::ArrayExpr {
                let r = gctx.ctx.resolve_array_expr(field_ty);
                if !r.is_none() {
                    field_ty = r;
                }
            }
            let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(field_ty);
            let fa = emit(
                gctx,
                JirInst {
                    tag: JirTag::FieldAddr,
                    a: base_ptr,
                    b: idx as u32,
                    ty: ptr_ty,
                    ..Default::default()
                },
            );
            Ok((fa, field_ty))
        }
        AstTag::Index => {
            let (mut base_ptr, base_ty) = astgen_lvalue(gctx, NodeIdx::new(n.lhs))?;
            let idx_ref = astgen_expr(gctx, NodeIdx::new(n.rhs), builtin::U64)?;
            let (kind, a) = {
                let k = gctx.ctx.type_pool.get(base_ty);
                (k.kind, k.a)
            };
            let elem_ty = match kind {
                TypeKind::Array | TypeKind::Slice | TypeKind::PtrMany => TypeIdx::new(a),
                _ => {
                    // A struct base usually means the `v[i]` sugar's `at` was
                    // withdrawn for this instantiation — replay why (the C++
                    // astgen.cpp:7473).
                    if let Some(sb) = gctx.ctx.struct_name_of(base_ty) {
                        let qualified = format!("{sb}.at");
                        if gctx.ctx.get_withdrawn_method(&qualified).is_some() {
                            report_method_miss(gctx, &qualified)?;
                        }
                    }
                    return Err("astgen: lvalue index on non-array/slice/ptr-many".into());
                }
            };
            // PtrMany base: the slot holds the pointer value — Load to follow it.
            // Slice base: Load the {ptr,len}, ExtractValue the data pointer.
            // Array base: the slot IS the inline storage — GEP it directly.
            if kind == TypeKind::PtrMany {
                base_ptr = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Load,
                        a: base_ptr,
                        ty: base_ty,
                        ..Default::default()
                    },
                );
            } else if kind == TypeKind::Slice {
                let slice_val = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Load,
                        a: base_ptr,
                        ty: base_ty,
                        ..Default::default()
                    },
                );
                let ptr_many_ty = gctx.ctx.type_pool.intern_ptr_many(elem_ty);
                base_ptr = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::ExtractValue,
                        a: slice_val,
                        b: 0,
                        ty: ptr_many_ty,
                        ..Default::default()
                    },
                );
            }
            let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(elem_ty);
            let ia = emit(
                gctx,
                JirInst {
                    tag: JirTag::IndexAddr,
                    a: base_ptr,
                    b: idx_ref,
                    ty: ptr_ty,
                    ..Default::default()
                },
            );
            Ok((ia, elem_ty))
        }
        other => Err(format!("astgen: lvalue {other:?} not yet ported")),
    }
}

/// Is the assignment target a *value-world* destination (a whole local, a
/// struct field, or a fixed-array element) rather than pointer-world (through a
/// `Deref` / slice / many-pointer)? Only value-world destinations get the
/// overwrite-drop of their old value. Walks the projection root→leaf, requiring
/// every `Index` step to land on a fixed `Array`.
fn assign_target_is_value_world(gctx: &AstGenCtx, target_idx: NodeIdx) -> bool {
    let mut steps: Vec<AstNode> = Vec::new();
    let mut cur = target_idx;
    let mut root: Option<AstNode> = None;
    while !cur.is_none() {
        let nd = *gctx.ctx.node_store.get(cur);
        if nd.tag == AstTag::Variable {
            root = Some(nd);
            break;
        }
        if nd.tag == AstTag::MemberAccess || nd.tag == AstTag::Index {
            steps.push(nd);
            cur = NodeIdx::new(nd.lhs);
            continue;
        }
        return false; // Deref or any other shape: pointer world
    }
    let Some(root) = root else { return false };
    let root_name = str_at(gctx, root.lhs);
    let mut cur_ty = match gctx.local_types.get(&root_name) {
        Some(&t) => t,
        None => return false,
    };
    for step in steps.iter().rev() {
        let k = gctx.ctx.type_pool.get(cur_ty);
        if step.tag == AstTag::Index {
            if k.kind != TypeKind::Array {
                return false;
            }
            cur_ty = TypeIdx::new(k.a);
            continue;
        }
        // MemberAccess: advance to the field's type.
        if k.kind != TypeKind::Struct && k.kind != TypeKind::Named {
            return false;
        }
        let member = str_at(gctx, step.rhs);
        match gctx
            .ctx
            .struct_fields(cur_ty)
            .and_then(|fs| fs.into_iter().find(|(nm, _)| *nm == member).map(|(_, t)| t))
        {
            Some(t) => cur_ty = t,
            None => return false,
        }
    }
    true
}

/// Lower an assignment `target = value`: resolve the target to a pointer, lower
/// the value at the slot's leaf type, drop the old value first (overwrite-drop
/// of a live drop-bearing value-world destination — RHS evaluates first, old
/// drops, new stores), then Store. A bare drop-bearing RHS local is MOVED.
fn astgen_assign(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let target_idx = NodeIdx::new(n.lhs);
    let value_idx = NodeIdx::new(n.rhs);
    // Assignment to a `comp var` (an explicit comp binding, not a runtime
    // local) mutates the comp scope — no JIR. Runtime locals shadow comp
    // bindings; seeded names (module consts, comp params) don't match and fall
    // through to the runtime lvalue path. Ports the C++ astgenAssign comp path
    // (astgen.cpp:1510-1565) rule-for-rule.
    let tnode = *gctx.ctx.node_store.get(target_idx);
    if tnode.tag == AstTag::Variable {
        let tname = str_at(gctx, tnode.lhs);
        if !gctx.locals.contains_key(&tname)
            && let Some(info) = lookup_comp_binding_info(gctx, &tname)
        {
            if info.is_const {
                return Err(fail_node(
                    gctx,
                    target_idx,
                    &format!("cannot assign to comp const `{tname}`"),
                ));
            }
            if gctx.runtime_cond_depth != info.decl_depth {
                return Err(fail_node(
                    gctx,
                    target_idx,
                    &format!(
                        "cannot assign to comp binding `{tname}` from inside runtime \
                         conditional control flow — a comp value cannot depend on a \
                         runtime branch"
                    ),
                ));
            }
            let mut v = gctx.ctx.fold_comptime_expr_in(value_idx, &gctx.comp_scope);
            if v.is_none() {
                return Err(fail_node(
                    gctx,
                    value_idx,
                    &format!("comp assignment to `{tname}` must be a compile-time-known value"),
                ));
            }
            // Keep the binding's shape stable: same kind, and for ints the
            // established width/signedness (with a fit check) so reads keep
            // lowering consistently.
            if let Some(prev) = gctx.comp_scope.lookup(&tname).cloned() {
                if std::mem::discriminant(&prev) != std::mem::discriminant(&v) {
                    return Err(fail_node(
                        gctx,
                        value_idx,
                        &format!("comp assignment changes the kind of `{tname}` (e.g. int -> str)"),
                    ));
                }
                if let ComptimeValue::Int {
                    width: prev_width,
                    is_signed: prev_signed,
                    ..
                } = prev
                    && let ComptimeValue::Int {
                        bits,
                        width,
                        is_signed,
                    } = &mut v
                {
                    if !comp_int_fits(*bits, *is_signed, prev_width, prev_signed) {
                        return Err(fail_node(
                            gctx,
                            value_idx,
                            &format!(
                                "comp assignment value {} does not fit `{tname}` ({}{})",
                                comp_int_to_string(*bits, *is_signed),
                                if prev_signed { "i" } else { "u" },
                                prev_width
                            ),
                        ));
                    }
                    *width = prev_width;
                    *is_signed = prev_signed;
                }
            }
            gctx.comp_scope.set(&tname, v);
            return Ok(());
        }
    }
    // `v[i] = x` on a container -> `v.setAt(i, x)` cfn dispatch (the cfn frees the
    // overwritten element + takes ownership of the moved value).
    if tnode.tag == AstTag::Index {
        let base_idx = NodeIdx::new(tnode.lhs);
        let idx_idx = NodeIdx::new(tnode.rhs);
        if let Some((recv, method)) = container_index_recv(gctx, base_idx, "setAt")? {
            // The oracle's emitStructCfnDispatch order: index at u64, THEN the value
            // by value (astgen_expr, not lower_arg — a by-value StructLit arg, no
            // spill), THEN narrow the index, then the call.
            // A bare drop-bearing local stored into a container element MOVES;
            // field extraction on the RHS has no per-field suppression — reject
            // (the C++ rejectDropBearingFieldExtract, astgen.cpp:1624).
            let val_param_ty = method.args[2].ty;
            reject_drop_bearing_field_extract(gctx, value_idx, val_param_ty, "copy")?;
            let idx_u64 = astgen_expr(gctx, idx_idx, builtin::U64)?;
            let val_ref = astgen_expr(gctx, value_idx, val_param_ty)?;
            let idx_ref = narrow_index(gctx, idx_u64, method.args[1].ty);
            emit_call(gctx, &method, &[recv, idx_ref, val_ref], NO_JIR_REF)?;
            consume_moved_variable(gctx, value_idx);
            return Ok(());
        }
    }
    let (ptr_ref, leaf_ty) = astgen_lvalue(gctx, target_idx)?;
    // Field extraction on the RHS (`x = h.c`) has no per-field suppression —
    // reject (the C++ rejectDropBearingFieldExtract, astgen.cpp:1660).
    reject_drop_bearing_field_extract(gctx, value_idx, leaf_ty, "copy")?;
    let val_ref = astgen_expr(gctx, value_idx, leaf_ty)?;
    if !leaf_ty.is_none()
        && gctx.ctx.type_needs_drop(leaf_ty)
        && assign_target_is_value_world(gctx, target_idx)
    {
        emit_drop_in_place(gctx, ptr_ref, leaf_ty);
    }
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: ptr_ref,
            b: val_ref,
            ..Default::default()
        },
    );
    consume_moved_variable(gctx, value_idx);
    Ok(())
}

/// Lower a `UnaryOp` (`n.op`: 1=Neg, 2=LogNot, 3=BitNot).
///   `-x` — `FNeg` for floats, `0 - x` for ints (reuses Sub's signedness).
///   `!x` — `LogNot` (i1).
///   `~x` — `BitNot`.
fn astgen_unary_op(gctx: &mut AstGenCtx, n: &AstNode, expected: TypeIdx) -> Result<JirRef, String> {
    let operand = astgen_expr(gctx, NodeIdx::new(n.lhs), expected)?;
    let ty = gctx.jfn.get_inst(operand).ty;
    match n.op {
        1 => {
            if gctx.ctx.type_pool.get(ty).kind == TypeKind::Float {
                return Ok(emit(
                    gctx,
                    JirInst {
                        tag: JirTag::FNeg,
                        a: operand,
                        ty,
                        ..Default::default()
                    },
                ));
            }
            // Integer negate as `0 - operand` so Sub's signed/unsigned
            // semantics fall out for free.
            let zero = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: 0,
                    b: 0,
                    ty,
                    ..Default::default()
                },
            );
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Sub,
                    a: zero,
                    b: operand,
                    ty,
                    ..Default::default()
                },
            ))
        }
        2 => Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::LogNot,
                a: operand,
                ty: builtin::BOOL,
                ..Default::default()
            },
        )),
        3 => Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::BitNot,
                a: operand,
                ty,
                ..Default::default()
            },
        )),
        other => Err(format!("astgen: unknown UnaryOp {other}")),
    }
}

/// Lower an `as` cast (`expr as T`, `lhs`=operand, `rhs`=target type). The
/// destination is passed as the operand's expected hint for numeric targets so
/// a literal settles at the target width in one step (essential for floats — a
/// double-then-FPTrunc would double-round). Scalar conversions are covered:
/// bool→int, ptr↔ptr, ptr↔int, int↔int, int↔float, float↔float. The
/// enum↔int conversions are deferred with enum codegen; GenericCall
/// destinations resolve through the (currently no-op) context stub.
fn astgen_as_cast(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let operand_idx = NodeIdx::new(n.lhs);
    let mut dst_ty = TypeIdx::new(n.rhs);
    if gctx.ctx.type_pool.get(dst_ty).kind == TypeKind::GenericCall {
        let r = gctx.ctx.resolve_generic_call(dst_ty);
        if !r.is_none() {
            dst_ty = r;
        }
    }
    let (dst_kind, dst_width, dst_signed) = {
        let d = gctx.ctx.type_pool.get(dst_ty);
        (d.kind, d.a, d.b != 0)
    };
    let hint = if dst_kind == TypeKind::Int || dst_kind == TypeKind::Float {
        dst_ty
    } else {
        TypeIdx::NONE
    };
    let val = astgen_expr(gctx, operand_idx, hint)?;
    let mut src_ty = gctx.jfn.get_inst(val).ty;
    if src_ty == dst_ty {
        return Ok(val);
    }
    if gctx.ctx.type_pool.get(src_ty).kind == TypeKind::GenericCall {
        let r = gctx.ctx.resolve_generic_call(src_ty);
        if !r.is_none() {
            src_ty = r;
        }
    }
    if src_ty == dst_ty {
        return Ok(val);
    }
    let (src_kind, src_width, src_signed) = {
        let s = gctx.ctx.type_pool.get(src_ty);
        (s.kind, s.a, s.b != 0)
    };

    let is_ptr = |k: TypeKind| k == TypeKind::PtrSingle || k == TypeKind::PtrMany;
    let cast = |gctx: &mut AstGenCtx, tag: JirTag| {
        emit(
            gctx,
            JirInst {
                tag,
                a: val,
                ty: dst_ty,
                ..Default::default()
            },
        )
    };

    // Bool → integer.
    if src_kind == TypeKind::Bool && dst_kind == TypeKind::Int {
        return Ok(cast(gctx, JirTag::ZExt));
    }
    // Pointer ↔ pointer: identical runtime representation under opaque
    // pointers, so a zero-cost retag.
    if is_ptr(src_kind) && is_ptr(dst_kind) {
        return Ok(cast(gctx, JirTag::BitCast));
    }
    // Pointer → integer: only u64 is wide enough to round-trip on every target.
    if is_ptr(src_kind) && dst_kind == TypeKind::Int && dst_width == 64 {
        return Ok(cast(gctx, JirTag::PtrToInt));
    }
    // Integer → thin pointer: widen/truncate to pointer width first.
    if src_kind == TypeKind::Int && is_ptr(dst_kind) {
        let widened = if src_width < 64 {
            let tag = if src_signed {
                JirTag::SExt
            } else {
                JirTag::ZExt
            };
            emit(
                gctx,
                JirInst {
                    tag,
                    a: val,
                    ty: builtin::U64,
                    ..Default::default()
                },
            )
        } else if src_width > 64 {
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Trunc,
                    a: val,
                    ty: builtin::U64,
                    ..Default::default()
                },
            )
        } else {
            val
        };
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::IntToPtr,
                a: widened,
                ty: dst_ty,
                ..Default::default()
            },
        ));
    }
    if src_kind == TypeKind::Int && dst_kind == TypeKind::Int {
        if src_width < dst_width {
            return Ok(cast(
                gctx,
                if src_signed {
                    JirTag::SExt
                } else {
                    JirTag::ZExt
                },
            ));
        }
        if src_width > dst_width {
            return Ok(cast(gctx, JirTag::Trunc));
        }
        // Same width, different signedness — retag only.
        return Ok(cast(gctx, JirTag::BitCast));
    }
    if src_kind == TypeKind::Int && dst_kind == TypeKind::Float {
        return Ok(cast(
            gctx,
            if src_signed {
                JirTag::SIToFP
            } else {
                JirTag::UIToFP
            },
        ));
    }
    if src_kind == TypeKind::Float && dst_kind == TypeKind::Int {
        return Ok(cast(
            gctx,
            if dst_signed {
                JirTag::FPToSI
            } else {
                JirTag::FPToUI
            },
        ));
    }
    if src_kind == TypeKind::Float && dst_kind == TypeKind::Float {
        if src_width < dst_width {
            return Ok(cast(gctx, JirTag::FPExt));
        }
        if src_width > dst_width {
            return Ok(cast(gctx, JirTag::FPTrunc));
        }
        return Ok(val);
    }
    // Integer → enum: narrow the source to the u8 tag. A unit-only enum IS that
    // tag; a payloaded enum becomes a `{tag, undef-payload}` aggregate.
    if src_kind == TypeKind::Int
        && (dst_kind == TypeKind::Named || dst_kind == TypeKind::Enum)
        && gctx.ctx.enum_name_of(dst_ty).is_some()
    {
        let tag_ref = if src_width != 8 {
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Trunc,
                    a: val,
                    ty: builtin::U8,
                    ..Default::default()
                },
            )
        } else if src_signed {
            emit(
                gctx,
                JirInst {
                    tag: JirTag::BitCast,
                    a: val,
                    ty: builtin::U8,
                    ..Default::default()
                },
            )
        } else {
            val
        };
        if !gctx.ctx.enum_has_payload(dst_ty).unwrap_or(false) {
            return Ok(tag_ref);
        }
        let slot = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: dst_ty,
                ..Default::default()
            },
        );
        let u8_ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
        let tag_ptr = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: slot,
                b: 0,
                ty: u8_ptr,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: tag_ptr,
                b: tag_ref,
                ..Default::default()
            },
        );
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: slot,
                ty: dst_ty,
                ..Default::default()
            },
        ));
    }
    // Enum → integer: extract the discriminant byte (payloaded enums are a
    // `{tag, payload}` aggregate; unit-only enums are already an i8 tag), then
    // width-adjust the u8 tag to the destination integer width.
    if (src_kind == TypeKind::Named || src_kind == TypeKind::Enum)
        && dst_kind == TypeKind::Int
        && gctx.ctx.enum_name_of(src_ty).is_some()
    {
        let tag_ref = if gctx.ctx.enum_has_payload(src_ty).unwrap_or(false) {
            emit(
                gctx,
                JirInst {
                    tag: JirTag::ExtractValue,
                    a: val,
                    b: 0,
                    ty: builtin::U8,
                    ..Default::default()
                },
            )
        } else {
            val
        };
        if dst_ty == builtin::U8 {
            return Ok(tag_ref);
        }
        let tag_inst = |gctx: &mut AstGenCtx, tag: JirTag| {
            emit(
                gctx,
                JirInst {
                    tag,
                    a: tag_ref,
                    ty: dst_ty,
                    ..Default::default()
                },
            )
        };
        return Ok(if dst_width > 8 {
            tag_inst(gctx, JirTag::ZExt)
        } else if dst_width < 8 {
            tag_inst(gctx, JirTag::Trunc)
        } else {
            tag_inst(gctx, JirTag::BitCast)
        });
    }
    Err("astgen: unsupported `as` cast between these types".into())
}

/// Lower a single call argument against parameter `p` (the C++ `lowerArgInner`).
/// ByValue params take the value (with the slice→pointer rejection); ByPointer
/// params (`mut`, or any aggregate) take an address — the caller's storage for
/// an lvalue arg, or a fresh spill slot for an rvalue.
///
/// A `move`-mode arg transfers ownership to the callee — the source's drop is
/// consumed. Deferred: the `AddressOf`-on-mode validation and the
/// place-into-destination spill optimization (`astgenExprIntoPtr`).
/// True when `n` is a `MemberAccess` whose base is a non-local Variable — i.e. a
/// `Type.Variant` / `handle.X` constructor expression rather than an addressable
/// lvalue (a local's `s.field` has its base in `locals`). lower_arg spills these.
fn member_access_is_ctor(gctx: &AstGenCtx, n: &AstNode) -> bool {
    if n.tag != AstTag::MemberAccess {
        return false;
    }
    let base = *gctx.ctx.node_store.get(NodeIdx::new(n.lhs));
    base.tag == AstTag::Variable && !gctx.locals.contains_key(&str_at(gctx, base.lhs))
}

fn lower_arg(gctx: &mut AstGenCtx, arg_idx: NodeIdx, p: &Param) -> Result<JirRef, String> {
    // A `move`-mode arg that extracts a drop-bearing field out of an owned
    // aggregate is rejected — the callee and the parent's glue would both drop it
    // (the C++ lowerArg's rejectDropBearingFieldExtract, astgen.cpp:5933).
    if p.mode == jam_core::param_mode::ParamMode::Move {
        reject_drop_bearing_field_extract(gctx, arg_idx, p.ty, "move")?;
    }
    let pabi = classify_param(p.mode, p.ty, gctx.ctx)?;
    let r = if pabi.kind != ParamAbiKind::ByPointer {
        let v = astgen_expr(gctx, arg_idx, p.ty)?;
        // A runtime slice does not implicitly decay to a pointer parameter;
        // passing a {ptr,len} aggregate where a bare pointer is expected
        // silently corrupts the ABI — require an explicit `.ptr`.
        let vk = gctx.ctx.type_pool.get(gctx.jfn.get_inst(v).ty).kind;
        let pk = gctx.ctx.type_pool.get(p.ty).kind;
        if vk == TypeKind::Slice && (pk == TypeKind::PtrMany || pk == TypeKind::PtrSingle) {
            return Err(format!(
                "cannot pass a slice to pointer parameter `{}`; use `.ptr` to pass \
                 the slice's data pointer",
                p.name
            ));
        }
        v
    } else {
        // ByPointer: feed an address.
        let arg_node = *gctx.ctx.node_store.get(arg_idx);
        match arg_node.tag {
            // Lvalueable args hand over their existing storage pointer. (Index is
            // accepted by astgen_lvalue only once array support lands.)
            AstTag::Variable | AstTag::Index | AstTag::Deref => astgen_lvalue(gctx, arg_idx)?.0,
            // A `Type.Variant` / `handle.X` member access whose base is a non-local
            // Variable is a CONSTRUCTOR (e.g. a payloaded enum's unit variant
            // `Msg.Quit`), not an lvalue — astgen_lvalue can't resolve the base. Spill
            // it like any rvalue (the `_` arm) — the C++ `memberAccessOnNonLvalue`
            // guard, astgen.cpp:5808. A member access on a LOCAL (`s.field`) stays an
            // lvalue.
            AstTag::MemberAccess if !member_access_is_ctor(gctx, &arg_node) => {
                astgen_lvalue(gctx, arg_idx)?.0
            }
            AstTag::AddressOf => astgen_expr(gctx, arg_idx, TypeIdx::NONE)?,
            // Spill an rvalue into a fresh slot so the callee gets a pointer. The
            // place-into-destination fast path is deferred — value-compile + Store.
            _ => {
                let ptr = emit_alloca_hoisted(
                    gctx,
                    JirInst {
                        tag: JirTag::Alloca,
                        ty: p.ty,
                        ..Default::default()
                    },
                );
                let val = astgen_expr(gctx, arg_idx, p.ty)?;
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Store,
                        a: ptr,
                        b: val,
                        ..Default::default()
                    },
                );
                ptr
            }
        }
    };
    if p.mode == jam_core::param_mode::ParamMode::Move {
        consume_moved_variable(gctx, arg_idx);
    }
    Ok(r)
}

/// Emit a `Call` to `fn_ast` with already-lowered `arg_refs`. sret-returning
/// callees get their result slot as a leading arg (the caller's `dest_ptr` when
/// supplied, else a fresh alloca that becomes the Call's result). The arg list
/// The per-value mangle token for a `comp` argument (`scale__u2`), mirroring the
/// C++ switch (astgen.cpp:6539-6566): an int folds to `i`/`u` + its u64 value, a
/// bool to `true`/`false`, a string to `s` + its libc++ `std::hash` (stable +
/// printable over arbitrary contents), a type to its width spelling / `bool` /
/// `t` + TypeIdx. A bare `_ => "x"` previously collapsed DISTINCT string and type
/// instantiations onto one cache key + LLVM symbol (the linker then merged them).
fn comp_value_spelling(v: &ComptimeValue, gctx: &AstGenCtx) -> String {
    match v {
        ComptimeValue::Int { is_signed, .. } => {
            format!("{}{}", if *is_signed { "i" } else { "u" }, v.as_u64())
        }
        ComptimeValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        ComptimeValue::Str(sid) => {
            format!(
                "s{}",
                crate::libcxx_order::libcxx_string_hash(&gctx.ctx.string_pool.get(*sid))
            )
        }
        ComptimeValue::Type(tid) => {
            let tk = gctx.ctx.type_pool.get(*tid);
            match tk.kind {
                TypeKind::Int => format!("{}{}", if tk.b != 0 { "i" } else { "u" }, tk.a),
                TypeKind::Bool => "bool".to_string(),
                _ => format!("t{}", tid.raw()),
            }
        }
        _ => "x".to_string(),
    }
}

/// A call to a fn with `comp` value params (`fn scale(comp k: u32, x: u32)`):
/// fold each comp arg to a value, bake it into a per-instantiation symbol
/// (`scale__u2`), and dispatch to that monomorphization passing only the
/// runtime args. The instantiated body is emitted lazily by the backend and is
/// not part of the JIR dump, so only the call site is lowered here.
fn astgen_comp_instantiated_call(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    fn_ast: &FunctionAST,
    dest_ptr: JirRef,
) -> Result<JirRef, String> {
    let args_extra = n.rhs;
    let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra));
    let mut suffix = String::new();
    let mut runtime_idx: Vec<NodeIdx> = Vec::new();
    let mut runtime_params: Vec<jam_syntax::ast::Param> = Vec::new();
    // The comp-param substitution baked into the clone's body (the C++
    // `compSubst`): a body reference to `k` folds to this call-site value.
    let mut comp_subst: HashMap<String, ComptimeValue> = HashMap::new();
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        match fn_ast.args.get(i as usize) {
            Some(p) if p.is_comp => {
                let v = gctx.ctx.fold_comptime_expr_in(arg_idx, &gctx.comp_scope);
                if matches!(v, ComptimeValue::None) {
                    // Recoverable (the C++ astgen.cpp:6530).
                    return Ok(recover_here(
                        gctx,
                        format!(
                            "argument for comp param `{}` must be a compile-time constant",
                            p.name
                        ),
                        TypeIdx::NONE,
                    ));
                }
                suffix.push_str("__");
                suffix.push_str(&comp_value_spelling(&v, gctx));
                comp_subst.insert(p.name.clone(), v);
            }
            Some(p) => {
                runtime_idx.push(arg_idx);
                runtime_params.push(p.clone());
            }
            None => runtime_idx.push(arg_idx),
        }
    }
    let base = if fn_ast.module_path.is_empty() {
        fn_ast.name.clone()
    } else {
        format!("{}.{}", fn_ast.module_path, fn_ast.name)
    };
    // A clone whose signature drops the comp params; module_path empty so the
    // mangler passes the instantiation name through unchanged.
    let mut clone = fn_ast.clone();
    clone.name = format!("{base}{suffix}");
    clone.module_path = String::new();
    clone.parent_struct = String::new();
    clone.args = runtime_params;

    // EAGERLY declare + define the clone the first time this instantiation is
    // seen (the C++ `astgenCompInstantiatedCall`): the clone body is never
    // emitted by the regular pipeline, so without this the call dangles
    // ("unknown callee"). The clone's name doubles as the cache key AND its
    // LLVM symbol. Cache on `get_function_ast` so a repeat call reuses it.
    if gctx.ctx.get_function_ast(&clone.name).is_none() {
        gctx.ctx
            .register_function_ast(clone.name.clone(), clone.clone());
        // Metadata + LLVM prototype, then the body — with the comp subst
        // active so body refs to comp params fold to the baked constants, and
        // the defining module pushed so bare type/name refs resolve there.
        let mut jfn = astgen_metadata(&clone, gctx.ctx);
        jfn.name = clone.name.clone();
        let _ = jir_declare_prototype(&jfn, gctx.ctx);
        gctx.ctx.push_body_module(fn_ast.module_path.clone());
        gctx.ctx.set_current_comp_subst(comp_subst);
        let body_res = astgen_body_into(&mut jfn, &clone, gctx.ctx);
        gctx.ctx.clear_current_comp_subst();
        gctx.ctx.pop_body_module();
        body_res?;
        jir_define_body(&jfn, gctx.ctx)?;
    }

    let mut arg_refs: Vec<JirRef> = Vec::with_capacity(runtime_idx.len());
    for (j, arg_idx) in runtime_idx.iter().enumerate() {
        arg_refs.push(lower_arg(gctx, *arg_idx, &clone.args[j])?);
    }
    emit_call(gctx, &clone, &arg_refs, dest_ptr)
}

/// is packed `[count, args..]` into the function's extra array. A `noreturn`
/// callee terminates the current block with `Unreachable`.
fn emit_call(
    gctx: &mut AstGenCtx,
    fn_ast: &FunctionAST,
    arg_refs: &[JirRef],
    dest_ptr: JirRef,
) -> Result<JirRef, String> {
    let mangled = mangled_function_name(fn_ast, &gctx.ctx.type_pool, &gctx.ctx.string_pool);
    let callee_id = gctx.ctx.string_pool.intern_str(&mangled).raw();

    // The callee's return type spelled in ITS module (a callee returning its own
    // module's `Token` shows `mod.Token` at the call site, like the oracle).
    let rq = gctx
        .ctx
        .requalify_type(fn_ast.return_type, &fn_ast.module_path);
    let ret_ty = gctx.ctx.qualify_generic_callee(rq, &fn_ast.module_path);

    let sret_callee =
        !ret_ty.is_none() && classify_return(ret_ty, gctx.ctx)?.kind == ReturnAbiKind::Indirect;

    let mut all_args: Vec<u32> = Vec::with_capacity(arg_refs.len() + usize::from(sret_callee));
    if sret_callee {
        // The caller's destination when supplied, else a fresh slot that
        // becomes the Call's result (a pointer to the freshly-written value).
        let slot = if dest_ptr != NO_JIR_REF {
            dest_ptr
        } else {
            emit_alloca_hoisted(
                gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: ret_ty,
                    ..Default::default()
                },
            )
        };
        all_args.push(slot);
    }
    all_args.extend_from_slice(arg_refs);

    let mut packed: Vec<u32> = Vec::with_capacity(1 + all_args.len());
    packed.push(all_args.len() as u32);
    packed.extend_from_slice(&all_args);
    let extra = gctx.jfn.push_extra(&packed);

    let call_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: callee_id,
            b: extra,
            ty: ret_ty,
            ..Default::default()
        },
    );
    if fn_ast.return_type == builtin::NORETURN {
        emit(
            gctx,
            JirInst {
                tag: JirTag::Unreachable,
                ..Default::default()
            },
        );
    }
    // Place-call: the value was written through `dest_ptr`, so bind no JirRef.
    if sret_callee && dest_ptr != NO_JIR_REF {
        return Ok(NO_JIR_REF);
    }
    Ok(call_ref)
}

/// Lower a direct free-function `Call` (`lhs`=callee StringIdx, `rhs`=ExtraIdx
/// → `[argCount, args..]`). The indirect/method form (flags bit 0), comptime-fn
/// and comp-instantiated calls, and the fn-pointer-in-local fallback are
/// deferred — they error cleanly until ported.
/// Construct an enum variant value `Enum.Variant(args)`: the discriminant byte
/// (unit-only enums return it directly), else a `{tag, payload}` aggregate built
/// via an alloca — `FieldAddr(0)` = tag, `FieldAddr(1)` = payload area, each
/// payload field stored at its computed byte offset through an i8-stride GEP.
/// Returns `None` when the variant name is unknown (caller falls through).
fn astgen_enum_variant_ctor(
    gctx: &mut AstGenCtx,
    canonical_type: &str,
    variant_name: &str,
    args_extra: u32,
    arg_count: u32,
) -> Result<Option<JirRef>, String> {
    let vidx = gctx.ctx.enum_variant_index(canonical_type, variant_name);
    if vidx < 0 {
        return Ok(None);
    }
    let variants = gctx
        .ctx
        .enum_variants_by_name(canonical_type)
        .unwrap_or_default();
    let disc = variants[vidx as usize].discriminant;
    let payload_types = variants[vidx as usize].payload_types.clone();
    let has_payload = gctx
        .ctx
        .enum_has_payload_by_name(canonical_type)
        .unwrap_or(false);
    let esid = gctx.ctx.string_pool.intern(canonical_type.as_bytes());
    let enum_ty = gctx.ctx.type_pool.intern_named(esid);

    let tag_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: disc,
            ty: builtin::U8,
            ..Default::default()
        },
    );
    if !has_payload {
        return Ok(Some(tag_ref));
    }
    let slot = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty: enum_ty,
            ..Default::default()
        },
    );
    let u8_ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
    let tag_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::FieldAddr,
            a: slot,
            b: 0,
            ty: u8_ptr,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: tag_ptr,
            b: tag_ref,
            ..Default::default()
        },
    );

    if arg_count as usize > payload_types.len() {
        return Err(format!(
            "astgen: too many args for variant `{canonical_type}.{variant_name}`"
        ));
    }
    if !payload_types.is_empty() && arg_count >= 1 {
        let pay_area = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: slot,
                b: 1,
                ty: u8_ptr,
                ..Default::default()
            },
        );
        let mut off: u64 = 0;
        for (i, &field_ty) in payload_types.iter().take(arg_count as usize).enumerate() {
            let s = gctx.ctx.type_size(field_ty)?;
            let a = gctx.ctx.type_align(field_ty)?;
            off = off.div_ceil(a) * a;
            let arg_idx = NodeIdx::new(
                gctx.ctx
                    .node_store
                    .get_extra(ExtraIdx::new(args_extra + 1 + i as u32)),
            );
            // Enum payload capture is a MOVE — extracting a drop-bearing field out
            // of an aggregate to capture it is rejected (the C++
            // rejectDropBearingFieldExtract, astgen.cpp:4561).
            reject_drop_bearing_field_extract(gctx, arg_idx, field_ty, "capture")?;
            let pay_val = astgen_expr(gctx, arg_idx, field_ty)?;
            consume_moved_variable(gctx, arg_idx);
            let off_ref = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: off as u32,
                    ty: builtin::U64,
                    ..Default::default()
                },
            );
            let field_ptr = emit(
                gctx,
                JirInst {
                    tag: JirTag::IndexAddr,
                    a: pay_area,
                    b: off_ref,
                    ty: u8_ptr,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: field_ptr,
                    b: pay_val,
                    ..Default::default()
                },
            );
            off += s;
        }
    }
    Ok(Some(emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: enum_ty,
            ..Default::default()
        },
    )))
}

fn astgen_call(gctx: &mut AstGenCtx, n: &AstNode, dest_ptr: JirRef) -> Result<JirRef, String> {
    if n.flags & 1 != 0 {
        return astgen_indirect_call(gctx, n, dest_ptr);
    }
    let callee = str_at(gctx, n.lhs);
    // Inside an instantiated body, rewrite `Self.method`/`T.method` to the
    // concrete `Inst.method` (the C++ resolvePrefix). No-op otherwise.
    let callee = gctx.ctx.resolve_subst_prefix(&callee);
    let args_extra = n.rhs;
    let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra));

    // `assert(actual, expected)` is a compiler builtin (from `import("test")`),
    // not a real function — lower it directly.
    if callee == "assert" {
        return astgen_assert_call(gctx, n);
    }

    // Resolve a bare callee against the BODY module first (the C++ namespace
    // lookup): two modules may each define `scale`, and a body must reach its
    // own, not whichever registered last under the bare key.
    let body_qualified = {
        let bm = gctx.ctx.current_body_module();
        if !bm.is_empty() && !callee.contains('.') {
            gctx.ctx.get_function_ast(&format!("{bm}.{callee}"))
        } else {
            None
        }
    };
    let fn_ast = match body_qualified.or_else(|| gctx.ctx.get_function_ast(&callee)) {
        Some(f) => f,
        None => {
            // A dotted callee `prefix.suffix`: an enum-variant constructor, a
            // `Type.method` static call, or a method on a local instance.
            if let Some(r) = astgen_dotted_call(gctx, &callee, args_extra, arg_count, dest_ptr)? {
                return Ok(r);
            }
            // A Fn-typed local (`f(args)`) or Fn-typed struct field
            // (`recv.field(args)`) — call indirect through the pointer.
            if let Some(r) = astgen_indirect_fn_call(gctx, &callee, args_extra, arg_count)? {
                return Ok(r);
            }
            // Qualified callees (`std.fmt.println`, `lib.priv`) get the precise
            // namespace diagnostic — "symbol `X` does not exist in module `M`"
            // — anchored at the import handle; bare callees get "unknown
            // function" (the C++ astgen.cpp:7336-7341). The C++ `recoverHere`s
            // here (not `failHere`): each independent call site reports its own
            // miss, so a body with several bad calls emits one error per call.
            let msg = if callee.contains('.') {
                gctx.ctx.format_namespace_lookup_error("function", &callee)
            } else {
                format!("unknown function `{callee}`")
            };
            return Ok(recover_here(gctx, msg, TypeIdx::NONE));
        }
    };
    if fn_ast.is_comp_time_fn {
        return astgen_comptime_fn_call(gctx, &fn_ast, args_extra, arg_count);
    }
    if fn_ast.args.iter().any(|p| p.is_comp) {
        return astgen_comp_instantiated_call(gctx, n, &fn_ast, dest_ptr);
    }

    let mut arg_refs: Vec<JirRef> = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        if (i as usize) < fn_ast.args.len() {
            arg_refs.push(lower_arg(gctx, arg_idx, &fn_ast.args[i as usize])?);
        } else {
            // varargs tail — pass by value.
            arg_refs.push(astgen_expr(gctx, arg_idx, TypeIdx::NONE)?);
        }
    }
    emit_call(gctx, &fn_ast, &arg_refs, dest_ptr)
}

// ---- cfn-call expansion (the C++ astgenCompTimeFnCall + CfnEmitter) ----

/// A `@emit*` intrinsic recorded from a cfn body. The C++ aliases the AstGenCtx
/// mutably during execBlock; Rust's borrow checker forbids that (the evaluator
/// holds immutable pool borrows for all of exec_block), so we RECORD the
/// intrinsics as pure data during execution and REPLAY them into the caller's
/// JIR after the evaluator's borrows release.
enum CfnEmitCmd {
    WriteBytes {
        fd: u32,
        fmt: jam_core::index::StringIdx,
        start: u32,
        end: u32,
    },
    PrintLocal {
        fd: u32,
        fmt: jam_core::index::StringIdx,
        start: u32,
        end: u32,
    },
    PutByte { fd: u32, byte: u8 },
}

#[derive(Default)]
struct RecordingCfnEmitter {
    cmds: Vec<CfnEmitCmd>,
}

impl crate::comptime::CompEmitter for RecordingCfnEmitter {
    fn handle_at_call(
        &mut self,
        name: &str,
        args: &[ComptimeValue],
        diags: &mut jam_core::diag::Diagnostics,
        loc: &jam_core::diag::SrcLoc,
    ) -> crate::comptime::ExecResult {
        use crate::comptime::ExecResult;
        match name {
            "emitWriteBytes" | "emitPrintLocalByRange" => {
                if args.len() != 4
                    || !args[0].is_int()
                    || !args[1].is_str()
                    || !args[2].is_int()
                    || !args[3].is_int()
                {
                    diags.error(
                        loc.clone(),
                        format!("@{name} expects (fd: i32, fmt: str, start: u32, end: u32)"),
                    );
                    return ExecResult::Error;
                }
                let ComptimeValue::Str(fmt) = args[1] else {
                    unreachable!()
                };
                let fd = args[0].as_u64() as u32;
                let start = args[2].as_u64() as u32;
                let end = args[3].as_u64() as u32;
                self.cmds.push(if name == "emitWriteBytes" {
                    CfnEmitCmd::WriteBytes {
                        fd,
                        fmt,
                        start,
                        end,
                    }
                } else {
                    CfnEmitCmd::PrintLocal {
                        fd,
                        fmt,
                        start,
                        end,
                    }
                });
                ExecResult::Continue
            }
            "emitPutByte" => {
                if args.len() != 2 || !args[0].is_int() || !args[1].is_int() {
                    diags.error(
                        loc.clone(),
                        "@emitPutByte expects (fd: i32, byte: u8)".to_string(),
                    );
                    return ExecResult::Error;
                }
                self.cmds.push(CfnEmitCmd::PutByte {
                    fd: args[0].as_u64() as u32,
                    byte: (args[1].as_u64() & 0xff) as u8,
                });
                ExecResult::Continue
            }
            _ => {
                diags.error(loc.clone(), format!("unknown @-emit intrinsic `@{name}`"));
                ExecResult::Error
            }
        }
    }
}

/// Register + declare a fake `printf` / `exit` extern prototype, once (the C++
/// astgenAssertCall registers these lazily). Mirrors ensure_dprintf.
fn ensure_printf(gctx: &mut AstGenCtx) {
    if gctx.ctx.get_function_ast("printf").is_some() {
        return;
    }
    let u8ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
    let mut f = FunctionAST::new(
        "printf",
        vec![Param::new("fmt", u8ptr)],
        builtin::I32,
        vec![],
    );
    f.is_extern = true;
    f.is_var_args = true;
    let jfn = astgen_metadata(&f, gctx.ctx);
    gctx.ctx.register_function_ast("printf", f);
    let _ = jir_declare_prototype(&jfn, gctx.ctx);
}

fn ensure_exit(gctx: &mut AstGenCtx) {
    if gctx.ctx.get_function_ast("exit").is_some() {
        return;
    }
    let mut f = FunctionAST::new(
        "exit",
        vec![Param::new("code", builtin::I32)],
        TypeIdx::NONE,
        vec![],
    );
    f.is_extern = true;
    let jfn = astgen_metadata(&f, gctx.ctx);
    gctx.ctx.register_function_ast("exit", f);
    let _ = jir_declare_prototype(&jfn, gctx.ctx);
}

/// Lower `assert(actual, expected)`: ICmpEq (or FCmpOeq for floats) + CondBr to
/// an `assert.fail` block (printf "Assertion failed\n" + exit(1) + Unreachable)
/// and an `assert.pass` continuation (the C++ astgenAssertCall).
fn astgen_assert_call(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let args_extra = n.rhs;
    let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra));
    if arg_count != 2 {
        return Err("astgen: assert expects exactly 2 arguments".into());
    }
    let actual_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra + 1)));
    let expected_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra + 2)));
    let mut actual_ref = astgen_expr(gctx, actual_idx, TypeIdx::NONE)?;
    let mut actual_ty = gctx.jfn.get_inst(actual_ref).ty;
    let mut expected_ref = astgen_expr(gctx, expected_idx, actual_ty)?;
    let expected_ty = gctx.jfn.get_inst(expected_ref).ty;
    if actual_ty != expected_ty {
        let ak = gctx.ctx.type_pool.get(actual_ty);
        let ek = gctx.ctx.type_pool.get(expected_ty);
        if ak.kind == TypeKind::Int && ek.kind == TypeKind::Int {
            if ak.a > ek.a {
                expected_ref = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::ZExt,
                        a: expected_ref,
                        ty: actual_ty,
                        ..Default::default()
                    },
                );
            } else {
                actual_ref = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::ZExt,
                        a: actual_ref,
                        ty: expected_ty,
                        ..Default::default()
                    },
                );
                actual_ty = expected_ty;
            }
        }
    }
    let cmp_tag = if gctx.ctx.type_pool.get(actual_ty).kind == TypeKind::Float {
        JirTag::FCmpOeq
    } else {
        JirTag::ICmpEq
    };
    let cmp_ref = emit(
        gctx,
        JirInst {
            tag: cmp_tag,
            a: actual_ref,
            b: expected_ref,
            ty: builtin::BOOL,
            ..Default::default()
        },
    );
    let fail_b = gctx.jfn.push_block("assert.fail");
    let pass_b = gctx.jfn.push_block("assert.pass");
    emit_cond_br(gctx, cmp_ref, pass_b, fail_b);

    gctx.current_block = fail_b;
    // Intern "printf" FIRST (the C++ printfNameId at astgen.cpp:5646, before the
    // message string) to keep the string-pool order byte-exact.
    let pid = gctx.ctx.string_pool.intern(b"printf");
    ensure_printf(gctx);
    let slice_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
    let u8ptr = gctx.ctx.type_pool.intern_ptr_many(builtin::U8);
    let msg = gctx.ctx.string_pool.intern(b"Assertion failed\n");
    let str_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: msg.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let str_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: str_ref,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let pe = gctx.jfn.push_extra(&[1, str_ptr]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: pid.raw(),
            b: pe,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    ensure_exit(gctx);
    let code = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 1,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    let eid = gctx.ctx.string_pool.intern(b"exit");
    let ee = gctx.jfn.push_extra(&[1, code]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: eid.raw(),
            b: ee,
            ty: TypeIdx::NONE,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Unreachable,
            ..Default::default()
        },
    );

    gctx.current_block = pass_b;
    Ok(NO_JIR_REF)
}

/// Register + declare the libc `dprintf` extern prototype, once. cfn output
/// (print/eprint) lowers to `dprintf(fd, "%.*s", len, ptr)` calls — the C++
/// `ensureDprintfForCfn`. The LLVM proto is `(i32, i8*, ...)` varargs; the
/// `fmt` param is typed `*const u8` so jir_declare_prototype lowers it to `i8*`.
fn ensure_dprintf(gctx: &mut AstGenCtx) {
    if gctx.ctx.get_function_ast("dprintf").is_some() {
        return;
    }
    let u8ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
    let mut f = FunctionAST::new(
        "dprintf",
        vec![Param::new("fd", builtin::I32), Param::new("fmt", u8ptr)],
        builtin::I32,
        vec![],
    );
    f.is_extern = true;
    f.is_var_args = true;
    let jfn = astgen_metadata(&f, gctx.ctx);
    gctx.ctx.register_function_ast("dprintf", f);
    let _ = jir_declare_prototype(&jfn, gctx.ctx);
}

/// Replay `@emitWriteBytes(fd, fmt, start, end)` — emit `dprintf(fd, "%.*s",
/// len, ptr)` for the literal byte span `fmt[start..end]` (the C++ handleWrite
/// Bytes). The substring is interned to `.rodata`; runtime hands its ptr+len to
/// dprintf.
fn replay_write_bytes(
    gctx: &mut AstGenCtx,
    fd: u32,
    fmt: jam_core::index::StringIdx,
    start: u32,
    end: u32,
) -> Result<(), String> {
    if start == end {
        return Ok(());
    }
    let fmt_bytes = gctx.ctx.string_pool.get(fmt).to_vec();
    if start > end || end as usize > fmt_bytes.len() {
        return Err("astgen: cfn @emitWriteBytes range out of bounds".to_string());
    }
    ensure_dprintf(gctx);
    let lit = gctx
        .ctx
        .string_pool
        .intern(&fmt_bytes[start as usize..end as usize]);
    let slice_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
    let u8ptr = gctx.ctx.type_pool.intern_ptr_many(builtin::U8);
    let lit_slice = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: lit.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: lit_slice,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let ln = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: lit_slice,
            b: 1,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let ln_i32 = emit(
        gctx,
        JirInst {
            tag: JirTag::Trunc,
            a: ln,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    let fmt_id = gctx.ctx.string_pool.intern(b"%.*s");
    let fmt_slice = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: fmt_id.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let fmt_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: fmt_slice,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let fd_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: fd,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    let dp = gctx.ctx.string_pool.intern(b"dprintf");
    let pe = gctx.jfn.push_extra(&[4, fd_ref, fmt_ptr, ln_i32, ptr]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: dp.raw(),
            b: pe,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    Ok(())
}

/// Emit `dprintf(fd, "%.*s", len, ptr)` for a compile-time literal byte string
/// in the current block (the C++ Bool-case `emitStrLit`). `fd_ref` is reused, not
/// re-emitted.
fn emit_write_literal(gctx: &mut AstGenCtx, fd_ref: JirRef, literal: &[u8]) {
    let slice_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
    let u8ptr = gctx.ctx.type_pool.intern_ptr_many(builtin::U8);
    let lit = gctx.ctx.string_pool.intern(literal);
    let lit_slice = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: lit.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: lit_slice,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let ln = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: lit_slice,
            b: 1,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let ln_i32 = emit(
        gctx,
        JirInst {
            tag: JirTag::Trunc,
            a: ln,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    let fmt_id = gctx.ctx.string_pool.intern(b"%.*s");
    let fmt_slice = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: fmt_id.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let fmt_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: fmt_slice,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let dp = gctx.ctx.string_pool.intern(b"dprintf");
    let pe = gctx.jfn.push_extra(&[4, fd_ref, fmt_ptr, ln_i32, ptr]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: dp.raw(),
            b: pe,
            ty: builtin::I32,
            ..Default::default()
        },
    );
}

/// Replay `@emitPutByte(fd, byte)` — emit `dprintf(fd, "%c", byte)` with the
/// byte widened to i32 (`%c` expects an int) — the C++ handlePutByte
/// (astgen.cpp:6276).
fn replay_put_byte(gctx: &mut AstGenCtx, fd: u32, byte: u8) {
    ensure_dprintf(gctx);
    let fd_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: fd,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    let byte_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: byte as u32,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    emit_dprintf_single_arg(gctx, fd_ref, b"%c", byte_ref);
}

/// Emit `dprintf(fd, fmtSpec, arg)` — one extra runtime arg after the spec (the
/// C++ emitDprintfSingleArg).
fn emit_dprintf_single_arg(gctx: &mut AstGenCtx, fd_ref: JirRef, spec: &[u8], arg: JirRef) {
    let slice_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
    let u8ptr = gctx.ctx.type_pool.intern_ptr_many(builtin::U8);
    let fmt_id = gctx.ctx.string_pool.intern(spec);
    let fmt_slice = emit(
        gctx,
        JirInst {
            tag: JirTag::Str,
            a: fmt_id.raw(),
            ty: slice_ty,
            ..Default::default()
        },
    );
    let fmt_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::ExtractValue,
            a: fmt_slice,
            b: 0,
            ty: u8ptr,
            ..Default::default()
        },
    );
    let dp = gctx.ctx.string_pool.intern(b"dprintf");
    let pe = gctx.jfn.push_extra(&[3, fd_ref, fmt_ptr, arg]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::Call,
            a: dp.raw(),
            b: pe,
            ty: builtin::I32,
            ..Default::default()
        },
    );
}

/// Per-type runtime print dispatch for `@emitPrintLocalByRange` (the C++
/// emitDprintfForValue): widen ints to i64 (`%lld`/`%llu`), floats to f64
/// (`%g`), and write `str` slices as `%.*s`. Bool stays a follow-up.
fn emit_dprintf_for_value(
    gctx: &mut AstGenCtx,
    fd_ref: JirRef,
    val: JirRef,
    ty: TypeIdx,
) -> Result<(), String> {
    let k = gctx.ctx.type_pool.get(ty);
    match k.kind {
        TypeKind::Int => {
            let signed = k.b != 0;
            let wide = if k.a < 64 {
                let tag = if signed { JirTag::SExt } else { JirTag::ZExt };
                emit(
                    gctx,
                    JirInst {
                        tag,
                        a: val,
                        ty: builtin::I64,
                        ..Default::default()
                    },
                )
            } else {
                val
            };
            let spec: &[u8] = if signed { b"%lld" } else { b"%llu" };
            emit_dprintf_single_arg(gctx, fd_ref, spec, wide);
            Ok(())
        }
        TypeKind::Float => {
            let wide = if k.a < 64 {
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::FPExt,
                        a: val,
                        ty: builtin::F64,
                        ..Default::default()
                    },
                )
            } else {
                val
            };
            emit_dprintf_single_arg(gctx, fd_ref, b"%g", wide);
            Ok(())
        }
        TypeKind::Slice if k.a == builtin::U8.raw() => {
            let slice_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
            let u8ptr = gctx.ctx.type_pool.intern_ptr_many(builtin::U8);
            let ptr = emit(
                gctx,
                JirInst {
                    tag: JirTag::ExtractValue,
                    a: val,
                    b: 0,
                    ty: u8ptr,
                    ..Default::default()
                },
            );
            let ln = emit(
                gctx,
                JirInst {
                    tag: JirTag::ExtractValue,
                    a: val,
                    b: 1,
                    ty: builtin::U64,
                    ..Default::default()
                },
            );
            let ln_i32 = emit(
                gctx,
                JirInst {
                    tag: JirTag::Trunc,
                    a: ln,
                    ty: builtin::I32,
                    ..Default::default()
                },
            );
            let fmt_id = gctx.ctx.string_pool.intern(b"%.*s");
            let fmt_slice = emit(
                gctx,
                JirInst {
                    tag: JirTag::Str,
                    a: fmt_id.raw(),
                    ty: slice_ty,
                    ..Default::default()
                },
            );
            let fmt_ptr = emit(
                gctx,
                JirInst {
                    tag: JirTag::ExtractValue,
                    a: fmt_slice,
                    b: 0,
                    ty: u8ptr,
                    ..Default::default()
                },
            );
            let dp = gctx.ctx.string_pool.intern(b"dprintf");
            let pe = gctx.jfn.push_extra(&[4, fd_ref, fmt_ptr, ln_i32, ptr]);
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Call,
                    a: dp.raw(),
                    b: pe,
                    ty: builtin::I32,
                    ..Default::default()
                },
            );
            Ok(())
        }
        TypeKind::Bool => {
            // Branch on the value, writing the "true"/"false" literal in each arm
            // (the C++ Bool case). Block labels match the oracle for byte-exact IR.
            let true_b = gctx.jfn.push_block("emit.true");
            let false_b = gctx.jfn.push_block("emit.false");
            let join_b = gctx.jfn.push_block("emit.bool.end");
            emit_cond_br(gctx, val, true_b, false_b);
            gctx.current_block = true_b;
            emit_write_literal(gctx, fd_ref, b"true");
            emit_br(gctx, join_b);
            gctx.current_block = false_b;
            emit_write_literal(gctx, fd_ref, b"false");
            emit_br(gctx, join_b);
            gctx.current_block = join_b;
            Ok(())
        }
        _ => Err("astgen: cfn format placeholder has an unsupported value type".to_string()),
    }
}

/// Replay `@emitPrintLocalByRange(fd, fmt, start, end)` — look up the caller
/// local named `fmt[start..end]`, load it, and emit the type-dispatched write.
fn replay_print_local(
    gctx: &mut AstGenCtx,
    fd: u32,
    fmt: jam_core::index::StringIdx,
    start: u32,
    end: u32,
) -> Result<(), String> {
    let fmt_bytes = gctx.ctx.string_pool.get(fmt).to_vec();
    if start > end || end as usize > fmt_bytes.len() {
        return Err("astgen: cfn @emitPrintLocalByRange range out of bounds".to_string());
    }
    let name = String::from_utf8_lossy(&fmt_bytes[start as usize..end as usize]).into_owned();
    let Some(&slot) = gctx.locals.get(&name) else {
        return Err(format!(
            "unknown variable `{name}` referenced from cfn format string"
        ));
    };
    let local_ty = gctx
        .local_types
        .get(&name)
        .copied()
        .unwrap_or(TypeIdx::NONE);
    ensure_dprintf(gctx);
    let val = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: local_ty,
            ..Default::default()
        },
    );
    let fd_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: fd,
            ty: builtin::I32,
            ..Default::default()
        },
    );
    emit_dprintf_for_value(gctx, fd_ref, val, local_ty)
}

/// Run a `cfn` body at the call site, replaying its `@emit*` output into the
/// caller's JIR (the C++ astgenCompTimeFnCall). The args fold in the caller's
/// comp scope; the body then runs in a fresh scope holding only the params.
fn astgen_comptime_fn_call(
    gctx: &mut AstGenCtx,
    fn_ast: &FunctionAST,
    args_extra: u32,
    arg_count: u32,
) -> Result<JirRef, String> {
    if arg_count as usize != fn_ast.args.len() {
        // Recoverable (the C++ astgen.cpp:6432) — cfn doesn't support varargs.
        return Ok(recover_here(
            gctx,
            format!(
                "cfn `{}` expects {} arg(s), got {}",
                fn_ast.name,
                fn_ast.args.len(),
                arg_count
            ),
            TypeIdx::NONE,
        ));
    }
    let line = gctx.ctx.node_store.get_line(gctx.current_node);

    // A VALUE-returning cfn (declared return type) folds its body at compile time
    // — with a CfnResolver active so cfn->cfn calls and recursion resolve — and
    // materializes the `return` value as a JIR constant, narrowed to the declared
    // return type (the C++ astgen.cpp:6499-6509). Value cfns carry no @-emit
    // output, so the recording path below is only for VOID @-emit cfns.
    if !fn_ast.return_type.is_none() && fn_ast.return_type != builtin::VOID {
        let arg_exprs: Vec<NodeIdx> = (0..arg_count)
            .map(|i| {
                NodeIdx::new(
                    gctx.ctx
                        .node_store
                        .get_extra(ExtraIdx::new(args_extra + 1 + i)),
                )
            })
            .collect();
        let returned = gctx
            .ctx
            .eval_cfn_call(fn_ast, &arg_exprs, &gctx.comp_scope, line)?;
        if returned.is_none() {
            // Recoverable (the C++ astgen.cpp:6501).
            return Ok(recover_here(
                gctx,
                format!(
                    "cfn `{}` declares a return type but its body did not `return` a \
                     compile-time value on every path",
                    fn_ast.name
                ),
                fn_ast.return_type,
            ));
        }
        return materialize_comptime_value(
            gctx,
            &returned,
            fn_ast.return_type,
            &format!("cfn `{}` result", fn_ast.name),
        );
    }

    // A VOID cfn (no declared return type) produces no value — its effect is the
    // @-emit JIR it drops into the caller. Run the body RECORDING @-emit
    // intrinsics (no JIR mutation while the evaluator holds its pool borrows),
    // folding args in the caller's comp scope.
    use crate::comptime::{
        CompCtx, ComptimeEvaluator, ComptimeScope, DEFAULT_ITER_CAP, ExecResult,
    };
    let host_os = crate::target::Target::from_triple_str(&jam_llvm::default_target_triple()).os;
    let mut outer = ComptimeScope::new();
    {
        let ev = ComptimeEvaluator::new(
            &gctx.ctx.node_store,
            &gctx.ctx.string_pool,
            &gctx.ctx.type_pool,
        );
        let mut diags = jam_core::diag::Diagnostics::new();
        let mut ctx = CompCtx {
            resolver: None,
            emitter: None,
            diags: Some(&mut diags),
            loc: jam_core::diag::SrcLoc::new("", line),
            host_os,
        };
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(
                gctx.ctx
                    .node_store
                    .get_extra(ExtraIdx::new(args_extra + 1 + i)),
            );
            let v = ev.eval(arg_idx, &gctx.comp_scope, &mut ctx);
            if v.is_none() {
                // Recoverable (the C++ astgen.cpp:6462).
                let msg = format!(
                    "argument to cfn `{}` (param `{}`) must be a compile-time constant",
                    fn_ast.name, fn_ast.args[i as usize].name
                );
                return Ok(recover_here(gctx, msg, TypeIdx::NONE));
            }
            outer.bind(fn_ast.args[i as usize].name.clone(), v);
        }
    }
    let mut emitter = RecordingCfnEmitter::default();
    {
        let ev = ComptimeEvaluator::new(
            &gctx.ctx.node_store,
            &gctx.ctx.string_pool,
            &gctx.ctx.type_pool,
        );
        let mut diags = jam_core::diag::Diagnostics::new();
        let mut ctx = CompCtx {
            resolver: None,
            emitter: Some(&mut emitter),
            diags: Some(&mut diags),
            loc: jam_core::diag::SrcLoc::new("", line),
            host_os,
        };
        let mut iter = 0u32;
        let mut ret = ComptimeValue::None;
        let r = ev.exec_block(
            &fn_ast.body,
            &mut outer,
            &mut iter,
            DEFAULT_ITER_CAP,
            &mut ret,
            &mut ctx,
        );
        if matches!(r, ExecResult::Error | ExecResult::IterationCap) {
            // Recoverable (the C++ astgen.cpp:6488) — return a poison so the
            // rest of the caller still gets analyzed.
            let msg = format!("cfn `{}` failed during compile-time evaluation", fn_ast.name);
            return Ok(recover_here(gctx, msg, TypeIdx::NONE));
        }
    }
    replay_cfn_emits(gctx, &emitter)?;
    Ok(NO_JIR_REF)
}

/// Replay a cfn body's recorded `@emit*` commands into the caller's JIR.
fn replay_cfn_emits(gctx: &mut AstGenCtx, emitter: &RecordingCfnEmitter) -> Result<(), String> {
    for cmd in &emitter.cmds {
        match *cmd {
            CfnEmitCmd::WriteBytes {
                fd,
                fmt,
                start,
                end,
            } => replay_write_bytes(gctx, fd, fmt, start, end)?,
            CfnEmitCmd::PrintLocal {
                fd,
                fmt,
                start,
                end,
            } => replay_print_local(gctx, fd, fmt, start, end)?,
            CfnEmitCmd::PutByte { fd, byte } => replay_put_byte(gctx, fd, byte),
        }
    }
    Ok(())
}

/// Lower a call's positional args against `params` (param `i` is `params[i]`),
/// passing a varargs tail by value. Used by static-method + free-fn-shaped emits.
fn lower_call_args(
    gctx: &mut AstGenCtx,
    params: &[Param],
    args_extra: u32,
    arg_count: u32,
) -> Result<Vec<JirRef>, String> {
    let mut arg_refs: Vec<JirRef> = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        if (i as usize) < params.len() {
            arg_refs.push(lower_arg(gctx, arg_idx, &params[i as usize])?);
        } else {
            arg_refs.push(astgen_expr(gctx, arg_idx, TypeIdx::NONE)?);
        }
    }
    Ok(arg_refs)
}

/// Lower a method call's non-receiver args. Method args are lowered by VALUE
/// (`astgen_expr`, not `lower_arg`) — the oracle's method-dispatch loads a
/// `move`/byref arg's value rather than passing its address — with the move's
/// caller-side drop consumed. Param `1+i` skips the implicit `self`.
fn lower_method_args(
    gctx: &mut AstGenCtx,
    method: &FunctionAST,
    args_extra: u32,
    arg_count: u32,
) -> Result<Vec<JirRef>, String> {
    let mut out: Vec<JirRef> = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        let param = method.args.get(i as usize + 1);
        let expect = param.map(|p| p.ty).unwrap_or(TypeIdx::NONE);
        // A `move`-mode method arg extracting a drop-bearing field out of an
        // aggregate is rejected (the C++ rejectDropBearingFieldExtract,
        // astgen.cpp:6949).
        if param.map(|p| p.mode) == Some(jam_core::param_mode::ParamMode::Move) {
            reject_drop_bearing_field_extract(gctx, arg_idx, expect, "move")?;
        }
        out.push(astgen_expr(gctx, arg_idx, expect)?);
        if param.map(|p| p.mode) == Some(jam_core::param_mode::ParamMode::Move) {
            consume_moved_variable(gctx, arg_idx);
        }
    }
    Ok(out)
}

/// Lower a `Type.method(args)` (TypeMethodCall) call's args BY VALUE — the oracle's
/// `astgenTypeMethodCall` lowers each arg with `astgenExpr` regardless of the param
/// ABI (no spill, no address-of even for a `move`/byref struct), consuming a move
/// arg's caller-side drop. Like `lower_method_args` but params index from 0 (the
/// `Type.method` form has no implicit `self`).
fn lower_type_method_args(
    gctx: &mut AstGenCtx,
    method: &FunctionAST,
    args_extra: u32,
    arg_count: u32,
) -> Result<Vec<JirRef>, String> {
    let mut out: Vec<JirRef> = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        let param = method.args.get(i as usize);
        let expect = param.map(|p| p.ty).unwrap_or(TypeIdx::NONE);
        // A `move`-mode arg extracting a drop-bearing field out of an aggregate is
        // rejected (the C++ rejectDropBearingFieldExtract, astgen.cpp:7192).
        if param.map(|p| p.mode) == Some(jam_core::param_mode::ParamMode::Move) {
            reject_drop_bearing_field_extract(gctx, arg_idx, expect, "move")?;
        }
        out.push(astgen_expr(gctx, arg_idx, expect)?);
        if param.map(|p| p.mode) == Some(jam_core::param_mode::ParamMode::Move) {
            consume_moved_variable(gctx, arg_idx);
        }
    }
    Ok(out)
}

/// Materialize a method receiver as call-arg 0. The pass-by-pointer-vs-value
/// decision is the self param's full ABI (`classify_param`), NOT its bare mode:
/// a `let self: Self` on a byref aggregate is still ByPointer. `recv_ptr` is a
/// storage pointer (lvalue / local slot) when available, else `recv_val` holds
/// the receiver value; exactly one is set.
fn materialize_receiver(
    gctx: &mut AstGenCtx,
    method: &FunctionAST,
    recv_ty: TypeIdx,
    recv_ptr: JirRef,
    recv_val: JirRef,
) -> Result<JirRef, String> {
    let self_mode = method
        .args
        .first()
        .map(|p| p.mode)
        .unwrap_or(jam_core::param_mode::ParamMode::Let);
    let by_ptr = classify_param(self_mode, recv_ty, gctx.ctx)?.kind == ParamAbiKind::ByPointer;
    if by_ptr {
        if recv_ptr != NO_JIR_REF {
            // The lvalue path already yields a pointer — pass it directly.
            Ok(recv_ptr)
        } else {
            // Non-lvalue receiver: spill to a fresh slot so the callee has a
            // stable address (byref values are pointers, so this is a memcpy).
            let slot = emit_alloca_hoisted(
                gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: recv_ty,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: slot,
                    b: recv_val,
                    ..Default::default()
                },
            );
            Ok(slot)
        }
    } else if recv_val != NO_JIR_REF {
        Ok(recv_val)
    } else {
        Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: recv_ptr,
                ty: recv_ty,
                ..Default::default()
            },
        ))
    }
}

/// The `n.flags & 1` indirect/method form `expr.method(args)`: the callee is a
/// `MemberAccess` node (receiver subexpr + method name); the receiver is lowered
/// as arg 0 and `RecvType.method` resolved in the function registry.
fn astgen_indirect_call(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    dest_ptr: JirRef,
) -> Result<JirRef, String> {
    let cn = *gctx.ctx.node_store.get(NodeIdx::new(n.lhs));
    if cn.tag != AstTag::MemberAccess {
        return Err("astgen: indirect-call callee is not a member access".into());
    }
    let recv_expr_idx = NodeIdx::new(cn.lhs);
    let method_name = str_at(gctx, cn.rhs);
    let args_extra = n.rhs;
    let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra));

    // Lvalueable receivers yield a pointer + leaf type without a value Load.
    let recv_tag = gctx.ctx.node_store.get(recv_expr_idx).tag;
    let lvalueable = matches!(
        recv_tag,
        AstTag::Variable | AstTag::MemberAccess | AstTag::Index | AstTag::Deref
    );
    let (recv_ptr, recv_val, recv_ty) = if lvalueable {
        let (ptr, leaf) = astgen_lvalue(gctx, recv_expr_idx)?;
        (ptr, NO_JIR_REF, leaf)
    } else {
        let v = astgen_expr(gctx, recv_expr_idx, TypeIdx::NONE)?;
        let ty = gctx.jfn.get_inst(v).ty;
        (NO_JIR_REF, v, ty)
    };

    // Instantiate a generic receiver, then resolve its struct/enum name.
    let _ = gctx.ctx.resolve_generic_call_instantiate(recv_ty)?;
    // Built-in `recv.clone()`: tier 1 (plain data -> the value), tier 2 (struct
    // glue), or an error tier (owns-resources). A user `cfn clone` returns None
    // here -> ordinary method dispatch below.
    if method_name == "clone"
        && let Some(r) = try_lower_builtin_clone(gctx, recv_ptr, recv_val, recv_ty)?
    {
        return Ok(r);
    }
    // Array builtin `recv.asPtr()` / `recv.asMutPtr()` on a field / index / deref
    // array lvalue (`self.buf.asMutPtr()`): the array's base address — an
    // `IndexAddr` at 0 -> `PtrMany(elem)`. Mirrors the C++ handling of any
    // lvalueable receiver (astgen.cpp:6788-6827); the local-Variable form is
    // handled in the prefix dispatch.
    if (method_name == "asPtr" || method_name == "asMutPtr")
        && recv_ptr != NO_JIR_REF
        && gctx.ctx.type_pool.get(recv_ty).kind == TypeKind::Array
    {
        let elem = TypeIdx::new(gctx.ctx.type_pool.get(recv_ty).a);
        let pm = gctx.ctx.type_pool.intern_ptr_many(elem);
        let zero = emit(
            gctx,
            JirInst {
                tag: JirTag::Int,
                a: 0,
                ty: builtin::U64,
                ..Default::default()
            },
        );
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::IndexAddr,
                a: recv_ptr,
                b: zero,
                ty: pm,
                ..Default::default()
            },
        ));
    }
    let recv_name = match gctx
        .ctx
        .struct_name_of(recv_ty)
        .or_else(|| gctx.ctx.enum_name_of(recv_ty))
    {
        Some(n) => n,
        None => {
            return Err(format!(
                "astgen: receiver of `.{method_name}` is not a struct/enum"
            ));
        }
    };
    let qualified = format!("{recv_name}.{method_name}");
    let Some(method) = gctx.ctx.get_function_ast(&qualified) else {
        return report_method_miss(gctx, &qualified);
    };

    let recv_arg = materialize_receiver(gctx, &method, recv_ty, recv_ptr, recv_val)?;
    let mut arg_refs = vec![recv_arg];
    arg_refs.extend(lower_method_args(gctx, &method, args_extra, arg_count)?);
    emit_call(gctx, &method, &arg_refs, dest_ptr)
}

/// A dotted direct-call callee `prefix.suffix`: enum-variant constructor, a
/// `Type.method` static call (prefix is a type/alias), or a method on a local
/// instance (prefix is a variable). Returns `None` when nothing matches (caller
/// reports the unknown-function error).
fn astgen_dotted_call(
    gctx: &mut AstGenCtx,
    callee: &str,
    args_extra: u32,
    arg_count: u32,
    dest_ptr: JirRef,
) -> Result<Option<JirRef>, String> {
    let Some((prefix, suffix)) = callee.rsplit_once('.') else {
        return Ok(None);
    };
    // Multi-dot handle chains (`w.leaf.makePoint`): resolve through the module
    // namespace re-export chain to the leaf module's free function.
    if prefix.contains('.') {
        // The oracle's multi-dot pre-check probes the recv prefix as a Named
        // (a handle-qualified enum-variant ctor); interning it here keeps the
        // string pool byte-aligned even when the prefix isn't an enum.
        let sid = gctx.ctx.string_pool.intern(prefix.as_bytes());
        let _ = gctx.ctx.type_pool.intern_named(sid);
        if let Some(method) = gctx.ctx.resolve_chained_function(callee) {
            // A chained `cfn` (`std.fmt.print(...)`) expands at the call site.
            if method.is_comp_time_fn {
                return Ok(Some(astgen_comptime_fn_call(
                    gctx, &method, args_extra, arg_count,
                )?));
            }
            let arg_refs = lower_call_args(gctx, &method.args, args_extra, arg_count)?;
            return Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?));
        }
        // `a.Counter.init(args)` / `a.Status.Ok(args)`: a handle-qualified struct
        // method/ctor or enum-variant ctor. Resolve the `handle.Type` prefix to its
        // module, then dispatch (the C++ multi-dot branch, astgen.cpp:6981).
        if let Some((handle, type_name)) = prefix.split_once('.')
            && let Some(module) = gctx.ctx.import_handle_module(handle)
        {
            let qual = format!("{module}.{type_name}");
            if let Some(method) = gctx.ctx.get_function_ast(&format!("{qual}.{suffix}")) {
                let arg_refs = lower_call_args(gctx, &method.args, args_extra, arg_count)?;
                return Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?));
            }
            let esid = gctx.ctx.string_pool.intern(qual.as_bytes());
            let enum_named = gctx.ctx.type_pool.intern_named(esid);
            if let Some(en) = gctx.ctx.enum_name_of(enum_named)
                && let Some(r) = astgen_enum_variant_ctor(gctx, &en, suffix, args_extra, arg_count)?
            {
                return Ok(Some(r));
            }
        }
        return Ok(None);
    }

    // 1. Prefix as a TYPE: enum-variant ctor / enum static method / struct static.
    let sid = gctx.ctx.string_pool.intern(prefix.as_bytes());
    let named = gctx.ctx.type_pool.intern_named(sid);
    let bm = gctx.ctx.current_body_module();
    let named = gctx.ctx.requalify_type(named, &bm);
    if let Some(en) = gctx.ctx.enum_name_of(named) {
        if let Some(r) = astgen_enum_variant_ctor(gctx, &en, suffix, args_extra, arg_count)? {
            return Ok(Some(r));
        }
        let qualified = format!("{en}.{suffix}");
        if let Some(method) = gctx.ctx.get_function_ast(&qualified) {
            let arg_refs = lower_call_args(gctx, &method.args, args_extra, arg_count)?;
            return Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?));
        }
    }
    if let Some(sn) = gctx.ctx.struct_name_of(named) {
        let qualified = format!("{sn}.{suffix}");
        let method = gctx
            .ctx
            .get_function_ast(&qualified)
            .ok_or_else(|| format!("astgen: type `{sn}` has no method `{suffix}`"))?;
        let arg_refs = lower_call_args(gctx, &method.args, args_extra, arg_count)?;
        return Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?));
    }

    // 2. Prefix as an IMPORT HANDLE (`const lib = import("m"); lib.fn()`) -> the
    // free function `m.fn`.
    if let Some(module) = gctx.ctx.import_handle_module(prefix)
        && let Some(method) = gctx.ctx.get_function_ast(&format!("{module}.{suffix}"))
    {
        // Privacy: a non-`pub` function reached through a module handle is not
        // exported (the C++ `formatNamespaceLookupError` privateNames branch,
        // codegen.cpp:831). `extern`/`export` libc bare names stay accessible.
        if !method.is_pub && !method.is_extern && !method.is_export {
            return Err(format!(
                "symbol `{suffix}` is not exported from module `{module}`"
            ));
        }
        // An import-handle `cfn` (`fmt.print(...)`) expands at the call site.
        if method.is_comp_time_fn {
            return Ok(Some(astgen_comptime_fn_call(
                gctx, &method, args_extra, arg_count,
            )?));
        }
        let arg_refs = lower_call_args(gctx, &method.args, args_extra, arg_count)?;
        return Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?));
    }

    // 3. Prefix as a LOCAL instance: dispatch `RecvType.method` with a receiver.
    let Some(&slot) = gctx.locals.get(prefix) else {
        return Ok(None);
    };
    let inst_ty = gctx
        .local_types
        .get(prefix)
        .copied()
        .unwrap_or(TypeIdx::NONE);
    // Array builtin `arr.asPtr()` / `arr.asMutPtr()`: the array's base address,
    // an i8-stride `IndexAddr` at constant 0 -> `PtrMany(elem)`.
    if (suffix == "asPtr" || suffix == "asMutPtr")
        && gctx.ctx.type_pool.get(inst_ty).kind == TypeKind::Array
    {
        let elem = TypeIdx::new(gctx.ctx.type_pool.get(inst_ty).a);
        let zero = emit(
            gctx,
            JirInst {
                tag: JirTag::Int,
                a: 0,
                ty: builtin::U64,
                ..Default::default()
            },
        );
        let pm = gctx.ctx.type_pool.intern_ptr_many(elem);
        return Ok(Some(emit(
            gctx,
            JirInst {
                tag: JirTag::IndexAddr,
                a: slot,
                b: zero,
                ty: pm,
                ..Default::default()
            },
        )));
    }
    let _ = gctx.ctx.resolve_generic_call_instantiate(inst_ty)?;
    // `t.clone()` on a local without a user `cfn clone` -> built-in clone
    // synthesis (covers structs, arrays, and primitives, which aren't structs);
    // a user clone returns None here and falls through to ordinary dispatch.
    if suffix == "clone"
        && let Some(r) = try_lower_builtin_clone(gctx, slot, NO_JIR_REF, inst_ty)?
    {
        return Ok(Some(r));
    }
    let Some(recv_name) = gctx.ctx.struct_name_of(inst_ty) else {
        return Ok(None);
    };
    let qualified = format!("{recv_name}.{suffix}");
    // Not a method — fall through (e.g. a Fn-typed field call via the
    // fn-pointer fallback). A withdrawn conditional method replays its reason
    // here rather than falling through to the module-handle fallback's
    // confusing error (the C++ astgen.cpp:7152).
    let Some(method) = gctx.ctx.get_function_ast(&qualified) else {
        if gctx.ctx.get_withdrawn_method(&qualified).is_some() {
            report_method_miss(gctx, &qualified)?;
        }
        return Ok(None);
    };

    // Receiver from the local slot: ByPointer wraps the slot in an explicit
    // AddrOf (the oracle's `AddrOf ty=*Recv`); ByValue loads it.
    let self_mode = method
        .args
        .first()
        .map(|p| p.mode)
        .unwrap_or(jam_core::param_mode::ParamMode::Let);
    let recv_arg = if classify_param(self_mode, inst_ty, gctx.ctx)?.kind == ParamAbiKind::ByPointer
    {
        let pty = gctx.ctx.type_pool.intern_ptr_single(inst_ty);
        emit(
            gctx,
            JirInst {
                tag: JirTag::AddrOf,
                a: slot,
                ty: pty,
                ..Default::default()
            },
        )
    } else {
        emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: slot,
                ty: inst_ty,
                ..Default::default()
            },
        )
    };
    let mut arg_refs = vec![recv_arg];
    arg_refs.extend(lower_method_args(gctx, &method, args_extra, arg_count)?);
    Ok(Some(emit_call(gctx, &method, &arg_refs, dest_ptr)?))
}

/// The `Type.method()` enum-variant constructor (`Option(i32).Some(42)`): the
/// SINGLE-payload form the oracle's `astgenTypeMethodCall` uses — `FieldAddr(1)`
/// typed `*PayloadType` + a direct store (distinct from the shared byte-stride
/// `astgen_enum_variant_ctor` the dotted-Call path uses). `None` when the suffix
/// isn't a variant. `extra` is the TypeMethodCall extra (`[methodId, argc, args]`).
fn astgen_type_method_enum_ctor(
    gctx: &mut AstGenCtx,
    enum_name: &str,
    variant_name: &str,
    extra: u32,
    arg_count: u32,
) -> Result<Option<JirRef>, String> {
    let vidx = gctx.ctx.enum_variant_index(enum_name, variant_name);
    if vidx < 0 {
        return Ok(None);
    }
    let variants = gctx
        .ctx
        .enum_variants_by_name(enum_name)
        .unwrap_or_default();
    let disc = variants[vidx as usize].discriminant;
    let payload_types = variants[vidx as usize].payload_types.clone();
    let has_payload = gctx
        .ctx
        .enum_has_payload_by_name(enum_name)
        .unwrap_or(false);
    let esid = gctx.ctx.string_pool.intern(enum_name.as_bytes());
    let enum_ty = gctx.ctx.type_pool.intern_named(esid);

    let tag_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: disc,
            ty: builtin::U8,
            ..Default::default()
        },
    );
    if !has_payload {
        return Ok(Some(tag_ref));
    }
    let slot = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty: enum_ty,
            ..Default::default()
        },
    );
    let u8_ptr = gctx.ctx.type_pool.intern_ptr_single(builtin::U8);
    let tag_ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::FieldAddr,
            a: slot,
            b: 0,
            ty: u8_ptr,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: tag_ptr,
            b: tag_ref,
            ..Default::default()
        },
    );
    if !payload_types.is_empty() && arg_count >= 1 {
        let field_ty = payload_types[0];
        let arg_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 2)));
        // Enum payload capture is a MOVE — a drop-bearing field extracted out of an
        // aggregate is rejected (the C++ rejectDropBearingFieldExtract, the
        // TypeMethodCall enum-construct path).
        reject_drop_bearing_field_extract(gctx, arg_idx, field_ty, "capture")?;
        let payload_val = astgen_expr(gctx, arg_idx, field_ty)?;
        let pf_ptr = gctx.ctx.type_pool.intern_ptr_single(field_ty);
        let pay_ptr = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: slot,
                b: 1,
                ty: pf_ptr,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: pay_ptr,
                b: payload_val,
                ..Default::default()
            },
        );
        consume_moved_variable(gctx, arg_idx);
    }
    Ok(Some(emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: enum_ty,
            ..Default::default()
        },
    )))
}

/// The `Type.method(args)` static-call node (`Counter.init()`, `Vec(i32).empty()`,
/// `Option(i32).Some(x)`). `n.lhs` is the receiver TYPE (instantiated on use for
/// a generic), `n.rhs` an extra `[methodNameId, argCount, args...]`. Resolves to
/// an enum-variant constructor or a `RecvType.method` static call (no receiver).
fn astgen_type_method_call(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    dest_ptr: JirRef,
) -> Result<JirRef, String> {
    // The receiver type stays LITERAL (the C++ astgenTypeMethodCall reads
    // `n.lhs` directly and never substitutes it): a generic receiver inside an
    // instantiated body (`Vec(T).empty()`) keeps `Vec(T)`, so the AddrOf
    // receiver interns the literal `*Vec(T)` GenericCall in the oracle's
    // TypePool order. Resolution applies the active subst internally
    // (resolve_generic_call reads current_subst), so `Vec(T)` still resolves to
    // the right monomorph.
    let recv_ty = TypeIdx::new(n.lhs);
    let extra = n.rhs;
    let method_name_id = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra));
    let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1));
    let method_name = str_at(gctx, method_name_id);
    // TypeMethodCall args live at extra+2+i; the Call-shaped helpers read
    // args_extra+1+i, so offset the base by one.
    let args_extra = extra + 1;

    // Instantiate a generic receiver (`Vec(i32)`) so its name + methods resolve.
    let _ = gctx.ctx.resolve_generic_call_instantiate(recv_ty)?;

    if let Some(en) = gctx.ctx.enum_name_of(recv_ty) {
        if let Some(r) = astgen_type_method_enum_ctor(gctx, &en, &method_name, extra, arg_count)? {
            return Ok(r);
        }
        let qualified = format!("{en}.{method_name}");
        if let Some(method) = gctx.ctx.get_function_ast(&qualified) {
            let arg_refs = lower_type_method_args(gctx, &method, args_extra, arg_count)?;
            return emit_call(gctx, &method, &arg_refs, dest_ptr);
        }
    }
    let recv_name = gctx
        .ctx
        .struct_name_of(recv_ty)
        .ok_or_else(|| format!("astgen: `{method_name}` receiver is not a struct/enum"))?;
    let qualified = format!("{recv_name}.{method_name}");
    // Intern the qualified callee name BEFORE lowering args (the C++
    // astgenTypeMethodCall interns it at the head, ahead of the arg loop). When
    // an arg fails to lower — e.g. `Box(Counter).init(self.ptr[0].clone())` in
    // Box(T).clone, where `Counter` has no `clone` — the callee name has already
    // landed in the string pool, matching the oracle's eager nested-arg order.
    gctx.ctx.string_pool.intern_str(&qualified);
    let Some(method) = gctx.ctx.get_function_ast(&qualified) else {
        return report_method_miss(gctx, &qualified);
    };
    let arg_refs = lower_type_method_args(gctx, &method, args_extra, arg_count)?;
    emit_call(gctx, &method, &arg_refs, dest_ptr)
}

/// Emit a `CallIndirect` through an already-lowered Fn-typed value: lower the
/// args against the Fn type's parameter types, pack `[count, args..]`, return
/// the call (typed by the Fn's return type).
fn build_indirect_call(
    gctx: &mut AstGenCtx,
    callee_val: JirRef,
    args_extra: u32,
    arg_count: u32,
) -> Result<JirRef, String> {
    let (ret_ty, param_tys) = {
        let k = gctx.ctx.type_pool.get(gctx.jfn.get_inst(callee_val).ty);
        (
            TypeIdx::new(k.a),
            gctx.ctx.type_pool.fn_params_at(k.b).to_vec(),
        )
    };
    let mut arg_refs: Vec<u32> = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(args_extra + 1 + i)),
        );
        let expect = param_tys.get(i as usize).copied().unwrap_or(TypeIdx::NONE);
        arg_refs.push(astgen_expr(gctx, arg_idx, expect)?);
    }
    let mut packed: Vec<u32> = Vec::with_capacity(1 + arg_refs.len());
    packed.push(arg_refs.len() as u32);
    packed.extend_from_slice(&arg_refs);
    let extra = gctx.jfn.push_extra(&packed);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::CallIndirect,
            a: callee_val,
            b: extra,
            ty: ret_ty,
            ..Default::default()
        },
    ))
}

/// Call through a Fn-typed local (`f(args)`) or a Fn-typed struct field
/// (`recv.field(args)`). Returns `None` when the callee isn't one of those.
fn astgen_indirect_fn_call(
    gctx: &mut AstGenCtx,
    callee: &str,
    args_extra: u32,
    arg_count: u32,
) -> Result<Option<JirRef>, String> {
    let fn_val = match callee.split_once('.') {
        None => {
            // Zero-dot bare name — a Fn-typed local?
            let Some(&slot) = gctx.locals.get(callee) else {
                return Ok(None);
            };
            let local_ty = gctx
                .local_types
                .get(callee)
                .copied()
                .unwrap_or(TypeIdx::NONE);
            if gctx.ctx.type_pool.get(local_ty).kind != TypeKind::Fn {
                return Ok(None);
            }
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Load,
                    a: slot,
                    ty: local_ty,
                    ..Default::default()
                },
            )
        }
        Some((recv_name, field_name)) => {
            // Single-dot `recv.field` where `field` is a Fn-typed field.
            if field_name.contains('.') {
                return Ok(None);
            }
            let Some(&slot) = gctx.locals.get(recv_name) else {
                return Ok(None);
            };
            let recv_ty = gctx
                .local_types
                .get(recv_name)
                .copied()
                .unwrap_or(TypeIdx::NONE);
            let Some(fields) = gctx.ctx.struct_fields(recv_ty) else {
                return Ok(None);
            };
            let Some(idx) = fields.iter().position(|(n, _)| n == field_name) else {
                return Ok(None);
            };
            let field_ty = fields[idx].1;
            if gctx.ctx.type_pool.get(field_ty).kind != TypeKind::Fn {
                return Ok(None);
            }
            let recv_val = emit(
                gctx,
                JirInst {
                    tag: JirTag::Load,
                    a: slot,
                    ty: recv_ty,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::ExtractValue,
                    a: recv_val,
                    b: idx as u32,
                    ty: field_ty,
                    ..Default::default()
                },
            )
        }
    };
    Ok(Some(build_indirect_call(
        gctx, fn_val, args_extra, arg_count,
    )?))
}

/// `@`-intrinsics. `@sizeOf(T)` / `@alignOf(T)` fold the type's layout to an
/// integer constant (an `[expr]T` argument resolves its comptime length first).
fn astgen_at_call(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let name = str_at(gctx, n.lhs);
    // `@dropInPlace(ptr)` (flags&1: n.rhs is an args ExtraIdx, not a type arg):
    // synthesize T's drop sequence at the pointee. Rust's `drop_in_place::<T>`.
    if n.flags & 1 != 0 && name == "dropInPlace" {
        let args_extra = n.rhs;
        let arg_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra));
        if arg_count != 1 {
            return Err("astgen: @dropInPlace takes exactly one pointer argument".into());
        }
        let ptr_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(args_extra + 1)));
        let ptr_ref = astgen_expr(gctx, ptr_idx, TypeIdx::NONE)?;
        let pk = gctx.ctx.type_pool.get(gctx.jfn.get_inst(ptr_ref).ty);
        if pk.kind != TypeKind::PtrSingle && pk.kind != TypeKind::PtrMany {
            return Err("astgen: @dropInPlace argument must be a pointer".into());
        }
        let pointee = gctx.ctx.apply_current_subst(TypeIdx::new(pk.a));
        let _ = gctx.ctx.resolve_generic_call_instantiate(pointee)?;
        emit_drop_in_place(gctx, ptr_ref, pointee);
        return Ok(NO_JIR_REF);
    }
    match name.as_str() {
        "sizeOf" | "alignOf" => {
            let mut ty_arg = TypeIdx::new(n.rhs);
            if gctx.ctx.type_pool.get(ty_arg).kind == TypeKind::ArrayExpr {
                ty_arg = gctx.ctx.resolve_array_expr_instantiate(ty_arg)?;
            }
            if name == "sizeOf" {
                let bytes = gctx.ctx.type_size(ty_arg)?;
                Ok(emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Int,
                        a: (bytes & 0xFFFF_FFFF) as u32,
                        b: (bytes >> 32) as u32,
                        ty: builtin::U64,
                        ..Default::default()
                    },
                ))
            } else {
                let align = gctx.ctx.type_align(ty_arg)?;
                Ok(emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Int,
                        a: align as u32,
                        ty: builtin::U8,
                        ..Default::default()
                    },
                ))
            }
        }
        "isDarwin" | "isLinux" | "isWindows" | "isUnix" => {
            // Target-OS predicates fold to a `Bool` literal from the host OS;
            // emit_cond_br then drops the dead arm. Host = the default triple.
            use crate::target::{Os, Target};
            let os = Target::from_triple_str(&jam_llvm::default_target_triple()).os;
            let v = match name.as_str() {
                "isDarwin" => os == Os::MacOs,
                "isLinux" => os == Os::Linux,
                "isWindows" => os == Os::Windows,
                _ => matches!(os, Os::MacOs | Os::Linux | Os::FreeBsd), // isUnix
            };
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Bool,
                    a: u32::from(v),
                    ty: builtin::BOOL,
                    ..Default::default()
                },
            ))
        }
        "os" => {
            // `@os()` -> the host OS tag name as a `[]u8`, the same JIR a string
            // literal would emit (an interned StringIdx in a Str inst).
            use crate::target::{Os, Target};
            let os = Target::from_triple_str(&jam_llvm::default_target_triple()).os;
            let os_name = match os {
                Os::MacOs => "macos",
                Os::Linux => "linux",
                Os::Windows => "windows",
                Os::FreeBsd => "freebsd",
                Os::Unknown => "unknown",
            };
            let sid = gctx.ctx.string_pool.intern_str(os_name).raw();
            let result_ty = gctx.ctx.type_pool.intern_slice(builtin::U8);
            Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Str,
                    a: sid,
                    ty: result_ty,
                    ..Default::default()
                },
            ))
        }
        // Recoverable (the C++ astgen.cpp:5574).
        other => Ok(recover_here(
            gctx,
            format!("unknown intrinsic `@{other}`"),
            TypeIdx::NONE,
        )),
    }
}

/// One payload binding introduced by an enum pattern: the source name, the
/// alloca holding the extracted field, and the field's type.
struct ArmBinding {
    name: String,
    slot: JirRef,
    ty: TypeIdx,
}

/// A Switch case row: the discriminant value (zero-extended to 64 bits; sign
/// recorded separately so codegen picks the right const helper) and its target.
struct SwitchCase {
    value: u64,
    is_signed: bool,
    target: JirBlockRef,
}

/// Peer-resolve two arm-tail types: widen the narrower int / float, or `NONE`
/// when they can't unify (the C++ `peerResolveType`).
fn peer_resolve_type(a: TypeIdx, b: TypeIdx, tp: &TypePool) -> TypeIdx {
    if a.is_none() {
        return b;
    }
    if b.is_none() {
        return a;
    }
    if a == b {
        return a;
    }
    let (ka_kind, ka_w, ka_s) = {
        let k = tp.get(a);
        (k.kind, k.a, k.b != 0)
    };
    let (kb_kind, kb_w, kb_s) = {
        let k = tp.get(b);
        (k.kind, k.a, k.b != 0)
    };
    if ka_kind == TypeKind::Int && kb_kind == TypeKind::Int {
        return tp.intern_int(ka_w.max(kb_w) as u16, ka_s || kb_s);
    }
    if ka_kind == TypeKind::Float && kb_kind == TypeKind::Float {
        return tp.intern_float(ka_w.max(kb_w) as u16);
    }
    TypeIdx::NONE
}

/// Lightweight source-level type inference for an arm tail (only NumberLit /
/// BoolLit / Variable are precise; else `NONE` and the caller falls back).
fn infer_tail_type(gctx: &AstGenCtx, idx: NodeIdx) -> TypeIdx {
    let n = *gctx.ctx.node_store.get(idx);
    match n.tag {
        AstTag::NumberLit => {
            let val = (n.lhs as u64) | ((n.rhs as u64) << 32);
            if n.flags & 2 != 0 {
                return builtin::F64;
            }
            if n.flags & 1 != 0 {
                if val <= 128 {
                    builtin::I8
                } else if val <= 32768 {
                    builtin::I16
                } else if val <= 2_147_483_648 {
                    builtin::I32
                } else {
                    builtin::I64
                }
            } else if val <= 255 {
                builtin::U8
            } else if val <= 65535 {
                builtin::U16
            } else if val <= 4_294_967_295 {
                builtin::U32
            } else {
                builtin::U64
            }
        }
        AstTag::BoolLit => builtin::BOOL,
        AstTag::Variable => gctx
            .local_types
            .get(&str_at(gctx, n.lhs))
            .copied()
            .unwrap_or(TypeIdx::NONE),
        _ => TypeIdx::NONE,
    }
}

/// Whether `node` is a divergent statement (return / break / continue, or a
/// call to a `noreturn` fn), so the arm contributes no tail value.
fn stmt_diverges(gctx: &AstGenCtx, node: NodeIdx) -> bool {
    let n = *gctx.ctx.node_store.get(node);
    if matches!(n.tag, AstTag::Return | AstTag::Break | AstTag::Continue) {
        return true;
    }
    if n.tag == AstTag::Call && n.flags & 1 == 0 {
        let callee = str_at(gctx, n.lhs);
        if let Some(f) = gctx.ctx.get_function_ast(&callee) {
            return f.return_type == builtin::NORETURN;
        }
    }
    false
}

/// Try to express a pattern as Switch case(s) targeting `arm_block`. Returns
/// `false` for any shape that needs a binding/compare block (ranges, payload
/// bindings, module-const patterns) — the caller then falls to the CondBr chain.
fn collect_switch_cases(
    gctx: &mut AstGenCtx,
    pat_idx: NodeIdx,
    arm_block: JirBlockRef,
    scrut_ty: TypeIdx,
    scrut_is_enum: bool,
    enum_name: Option<&str>,
    out: &mut Vec<SwitchCase>,
) -> bool {
    let p = *gctx.ctx.node_store.get(pat_idx);
    match p.tag {
        AstTag::PatLit => {
            if scrut_is_enum {
                return false;
            }
            let mut val = (p.lhs as u64) | ((p.rhs as u64) << 32);
            if p.flags & 1 != 0 {
                val = (val as i64).wrapping_neg() as u64;
            }
            let signed = {
                let sk = gctx.ctx.type_pool.get(scrut_ty);
                sk.kind == TypeKind::Int && sk.b != 0
            };
            out.push(SwitchCase {
                value: val,
                is_signed: signed,
                target: arm_block,
            });
            true
        }
        AstTag::PatEnumVariant => {
            let has_bindings = p.flags & 1 != 0;
            let infer_receiver = p.flags & 4 != 0;
            // A bare-identifier const pattern (`A` naming `const A = 10`) folds
            // the const to a Switch case value (the C++ collectSwitchCases).
            if !scrut_is_enum && infer_receiver && !has_bindings {
                let cname = str_at(gctx, p.rhs);
                let bm = gctx.ctx.current_body_module();
                let mc = (!bm.is_empty())
                    .then(|| gctx.ctx.get_module_const(&format!("{bm}.{cname}")))
                    .flatten()
                    .or_else(|| gctx.ctx.get_module_const(&cname));
                if let Some(mc) = mc {
                    let v = gctx.ctx.fold_comptime_expr(mc.init_expr);
                    if v.is_int() {
                        let sk = gctx.ctx.type_pool.get(scrut_ty);
                        let signed = sk.kind == TypeKind::Int && sk.b != 0;
                        out.push(SwitchCase {
                            value: v.as_u64(),
                            is_signed: signed,
                            target: arm_block,
                        });
                        return true;
                    }
                }
                return false;
            }
            if !scrut_is_enum || enum_name.is_none() || has_bindings {
                return false;
            }
            let en = enum_name.unwrap();
            let variant = str_at(gctx, p.rhs);
            let vidx = gctx.ctx.enum_variant_index(en, &variant);
            if vidx < 0 {
                return false;
            }
            match gctx
                .ctx
                .enum_variants_by_name(en)
                .and_then(|vs| vs.get(vidx as usize).map(|v| v.discriminant))
            {
                Some(disc) => {
                    out.push(SwitchCase {
                        value: disc as u64,
                        is_signed: false,
                        target: arm_block,
                    });
                    true
                }
                None => false,
            }
        }
        AstTag::PatOr => {
            let ex = p.lhs;
            let cnt = gctx.ctx.node_store.get_extra(ExtraIdx::new(ex));
            for i in 0..cnt {
                let sub = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(ex + 1 + i)));
                if !collect_switch_cases(
                    gctx,
                    sub,
                    arm_block,
                    scrut_ty,
                    scrut_is_enum,
                    enum_name,
                    out,
                ) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// Compare `scrut` against one pattern, branching to `arm_block` on match and
/// `next_block` otherwise (the C++ `astgenPatternCompare`). Handles PatLit,
/// PatRange, PatOr, PatEnumVariant (with payload-binding extraction into
/// `out_bindings`), and PatWildcard. Module-const patterns are deferred.
fn astgen_pattern_compare(
    gctx: &mut AstGenCtx,
    pat_idx: NodeIdx,
    scrut: JirRef,
    scrut_ty: TypeIdx,
    arm_block: JirBlockRef,
    next_block: JirBlockRef,
    out_bindings: &mut Vec<ArmBinding>,
) -> Result<(), String> {
    let p = *gctx.ctx.node_store.get(pat_idx);
    let signed_cmp = {
        let sk = gctx.ctx.type_pool.get(scrut_ty);
        sk.kind == TypeKind::Int && sk.b != 0
    };
    match p.tag {
        AstTag::PatLit => {
            let val = (p.lhs as u64) | ((p.rhs as u64) << 32);
            let k = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: (val & 0xFFFF_FFFF) as u32,
                    b: (val >> 32) as u32,
                    flags: if p.flags & 1 != 0 { 1 } else { 0 },
                    ty: scrut_ty,
                    ..Default::default()
                },
            );
            let cmp = emit(
                gctx,
                JirInst {
                    tag: JirTag::ICmpEq,
                    a: scrut,
                    b: k,
                    ty: builtin::BOOL,
                    ..Default::default()
                },
            );
            emit_cond_br(gctx, cmp, arm_block, next_block);
            Ok(())
        }
        AstTag::PatRange => {
            let lo_k = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: p.lhs,
                    ty: scrut_ty,
                    ..Default::default()
                },
            );
            let hi_k = emit(
                gctx,
                JirInst {
                    tag: JirTag::Int,
                    a: p.rhs,
                    ty: scrut_ty,
                    ..Default::default()
                },
            );
            let ge = emit(
                gctx,
                JirInst {
                    tag: if signed_cmp {
                        JirTag::ICmpSge
                    } else {
                        JirTag::ICmpUge
                    },
                    a: scrut,
                    b: lo_k,
                    ty: builtin::BOOL,
                    ..Default::default()
                },
            );
            let check_hi = gctx.jfn.push_block("range.hi");
            emit_cond_br(gctx, ge, check_hi, next_block);
            gctx.current_block = check_hi;
            let le = emit(
                gctx,
                JirInst {
                    tag: if signed_cmp {
                        JirTag::ICmpSle
                    } else {
                        JirTag::ICmpUle
                    },
                    a: scrut,
                    b: hi_k,
                    ty: builtin::BOOL,
                    ..Default::default()
                },
            );
            emit_cond_br(gctx, le, arm_block, next_block);
            Ok(())
        }
        AstTag::PatOr => {
            let ex = p.lhs;
            let cnt = gctx.ctx.node_store.get_extra(ExtraIdx::new(ex));
            for i in 0..cnt {
                let sub = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(ex + 1 + i)));
                let try_next = if i + 1 == cnt {
                    next_block
                } else {
                    gctx.jfn.push_block("or.next")
                };
                astgen_pattern_compare(
                    gctx,
                    sub,
                    scrut,
                    scrut_ty,
                    arm_block,
                    try_next,
                    out_bindings,
                )?;
                if i + 1 != cnt {
                    gctx.current_block = try_next;
                }
            }
            Ok(())
        }
        AstTag::PatEnumVariant => astgen_pattern_compare_enum(
            gctx,
            &p,
            scrut,
            scrut_ty,
            arm_block,
            next_block,
            out_bindings,
        ),
        AstTag::PatWildcard => {
            emit_br(gctx, arm_block);
            Ok(())
        }
        other => Err(format!("astgen: unsupported pattern form {other:?}")),
    }
}

/// The `PatEnumVariant` arm of [`astgen_pattern_compare`] (tag check + optional
/// payload-binding extraction). Split out to keep the dispatch readable.
fn astgen_pattern_compare_enum(
    gctx: &mut AstGenCtx,
    p: &AstNode,
    scrut: JirRef,
    scrut_ty: TypeIdx,
    arm_block: JirBlockRef,
    next_block: JirBlockRef,
    out_bindings: &mut Vec<ArmBinding>,
) -> Result<(), String> {
    let has_bindings = p.flags & 1 != 0;
    let type_idx_receiver = p.flags & 2 != 0;
    let infer_receiver = p.flags & 4 != 0;
    let (recv_slot, variant_name_id, binding_count, bindings_start) = if has_bindings {
        let ex = p.lhs;
        (
            gctx.ctx.node_store.get_extra(ExtraIdx::new(ex)),
            gctx.ctx.node_store.get_extra(ExtraIdx::new(ex + 1)),
            gctx.ctx.node_store.get_extra(ExtraIdx::new(ex + 2)),
            ex + 3,
        )
    } else {
        (p.lhs, p.rhs, 0, 0)
    };

    let enum_name: Option<String> = if infer_receiver {
        gctx.ctx.enum_name_of(scrut_ty)
    } else if type_idx_receiver {
        gctx.ctx.enum_name_of(TypeIdx::new(recv_slot))
    } else {
        let recv_name = str_at(gctx, recv_slot);
        let sid = gctx.ctx.string_pool.intern(recv_name.as_bytes());
        let recv_ty = gctx.ctx.type_pool.intern_named(sid);
        let bm = gctx.ctx.current_body_module();
        let recv_ty = gctx.ctx.requalify_type(recv_ty, &bm);
        gctx.ctx.enum_name_of(recv_ty).or(Some(recv_name))
    };
    let en = match enum_name {
        Some(ref n) if gctx.ctx.is_enum_name_registered(n) => n.clone(),
        _ => {
            // Const pattern (bare ident naming a module const): compare the
            // scrutinee against the const's value (the C++ astgenPatternCompare).
            if infer_receiver && !has_bindings {
                let cname = str_at(gctx, variant_name_id);
                let bm = gctx.ctx.current_body_module();
                let mc = (!bm.is_empty())
                    .then(|| gctx.ctx.get_module_const(&format!("{bm}.{cname}")))
                    .flatten()
                    .or_else(|| gctx.ctx.get_module_const(&cname));
                if let Some(mc) = mc {
                    let k = astgen_expr(gctx, mc.init_expr, scrut_ty)?;
                    let cmp = emit(
                        gctx,
                        JirInst {
                            tag: JirTag::ICmpEq,
                            a: scrut,
                            b: k,
                            ty: builtin::BOOL,
                            ..Default::default()
                        },
                    );
                    emit_cond_br(gctx, cmp, arm_block, next_block);
                    return Ok(());
                }
            }
            return Err(
                "astgen: pattern receiver doesn't resolve to an enum (const pattern deferred)"
                    .into(),
            );
        }
    };

    let variant = str_at(gctx, variant_name_id);
    let vidx = gctx.ctx.enum_variant_index(&en, &variant);
    if vidx < 0 {
        return Err(format!("astgen: unknown variant `{en}.{variant}`"));
    }
    let variants = gctx.ctx.enum_variants_by_name(&en).unwrap_or_default();
    let disc = variants[vidx as usize].discriminant;
    let has_payload = gctx.ctx.enum_has_payload_by_name(&en).unwrap_or(false);

    // Tag = ExtractValue(0) for payloaded enums, BitCast for unit-only.
    let tag_ref = emit(
        gctx,
        JirInst {
            tag: if has_payload {
                JirTag::ExtractValue
            } else {
                JirTag::BitCast
            },
            a: scrut,
            b: 0,
            ty: builtin::U8,
            ..Default::default()
        },
    );
    let k_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: disc,
            ty: builtin::U8,
            ..Default::default()
        },
    );
    let cmp = emit(
        gctx,
        JirInst {
            tag: JirTag::ICmpEq,
            a: tag_ref,
            b: k_ref,
            ty: builtin::BOOL,
            ..Default::default()
        },
    );

    if has_bindings && binding_count > 0 {
        let bind_b = gctx.jfn.push_block("matchbind");
        emit_cond_br(gctx, cmp, bind_b, next_block);
        gctx.current_block = bind_b;
        let payload_types = variants[vidx as usize].payload_types.clone();
        if binding_count as usize != payload_types.len() {
            return Err(format!(
                "astgen: pattern binds {binding_count} field(s), variant has {}",
                payload_types.len()
            ));
        }
        // Spill scrut so EnumPayload can take its address.
        let scrut_slot = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: scrut_ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: scrut_slot,
                b: scrut,
                ..Default::default()
            },
        );
        let mut off: u64 = 0;
        for b in 0..binding_count {
            let bind_name_id = gctx
                .ctx
                .node_store
                .get_extra(ExtraIdx::new(bindings_start + b));
            let bind_name = str_at(gctx, bind_name_id);
            let field_ty = payload_types[b as usize];
            let s = gctx.ctx.type_size(field_ty)?;
            let a = gctx.ctx.type_align(field_ty)?;
            off = off.div_ceil(a) * a;
            let payload_ref = emit(
                gctx,
                JirInst {
                    tag: JirTag::EnumPayload,
                    a: scrut_slot,
                    b: off as u32,
                    ty: field_ty,
                    ..Default::default()
                },
            );
            let bind_slot = emit_alloca_hoisted(
                gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: field_ty,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: bind_slot,
                    b: payload_ref,
                    ..Default::default()
                },
            );
            out_bindings.push(ArmBinding {
                name: bind_name,
                slot: bind_slot,
                ty: field_ty,
            });
            off += s;
        }
        emit_br(gctx, arm_block);
        return Ok(());
    }
    emit_cond_br(gctx, cmp, arm_block, next_block);
    Ok(())
}

/// Lower a `match` (`lhs`=scrutinee, `rhs`=ExtraIdx → `[armCount, (pat,
/// bodyCount, body..)..]`). Peer-types the arm tails into a result slot for
/// expression position; lowers via a `Switch` when every non-wildcard arm is a
/// single integer/variant equality, else a `CondBr` chain (ranges, payload
/// bindings). A drop-bearing enum scrutinee is consumed and its residual
/// payload dropped on non-binding arms (match-move).
fn astgen_match(gctx: &mut AstGenCtx, n: &AstNode, expected: TypeIdx) -> Result<JirRef, String> {
    let scrut_idx = NodeIdx::new(n.lhs);
    let arms_extra = n.rhs;
    let arm_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(arms_extra));

    // Decode arms (pattern + body statements); note the wildcard arm.
    let mut arms: Vec<(NodeIdx, Vec<NodeIdx>)> = Vec::with_capacity(arm_count as usize);
    let mut wildcard_arm_idx: i64 = -1;
    let mut pos = 1u32;
    for _ in 0..arm_count {
        let pat_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(arms_extra + pos)),
        );
        let bc = gctx
            .ctx
            .node_store
            .get_extra(ExtraIdx::new(arms_extra + pos + 1));
        let mut body = Vec::with_capacity(bc as usize);
        for j in 0..bc {
            body.push(NodeIdx::new(
                gctx.ctx
                    .node_store
                    .get_extra(ExtraIdx::new(arms_extra + pos + 2 + j)),
            ));
        }
        pos += 2 + bc;
        if gctx.ctx.node_store.get(pat_idx).tag == AstTag::PatWildcard {
            wildcard_arm_idx = arms.len() as i64;
        }
        arms.push((pat_idx, body));
    }

    // Peer-type pre-pass (statement-form matches with non-value arms stay None).
    let mut peer = expected;
    if peer.is_none() {
        let mut all_inferred = true;
        for (_, body) in &arms {
            if body.is_empty() {
                continue;
            }
            let tail = *body.last().unwrap();
            if stmt_diverges(gctx, tail) {
                continue;
            }
            let t = infer_tail_type(gctx, tail);
            if t.is_none() {
                all_inferred = false;
                break;
            }
            peer = peer_resolve_type(peer, t, &gctx.ctx.type_pool);
        }
        if !all_inferred {
            peer = TypeIdx::NONE;
        }
    }

    let scrut = astgen_expr(gctx, scrut_idx, TypeIdx::NONE)?;
    let scrut_ty = gctx.jfn.get_inst(scrut).ty;

    // Match-move: a drop-bearing enum scrutinee is owned by the match.
    let match_owns =
        gctx.ctx.enum_name_of(scrut_ty).is_some() && gctx.ctx.type_needs_drop(scrut_ty);
    let mut scrut_owned = NO_JIR_REF;
    if match_owns {
        // Matching a drop-bearing field extracted out of an aggregate would leave
        // the aggregate's glue to re-drop it — reject (the C++
        // rejectDropBearingFieldExtract, astgen.cpp:4175).
        reject_drop_bearing_field_extract(gctx, scrut_idx, scrut_ty, "consume")?;
        consume_moved_variable(gctx, scrut_idx);
        scrut_owned = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: scrut_ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: scrut_owned,
                b: scrut,
                ..Default::default()
            },
        );
    }

    let merge_b = gctx.jfn.push_block("matchend");
    let mut arm_blocks: Vec<JirBlockRef> = Vec::with_capacity(arm_count as usize);
    for _ in 0..arm_count {
        arm_blocks.push(gctx.jfn.push_block("matcharm"));
    }
    let result_slot = if peer.is_none() {
        NO_JIR_REF
    } else {
        emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: peer,
                ..Default::default()
            },
        )
    };
    let mut arm_bindings: Vec<Vec<ArmBinding>> = (0..arm_count).map(|_| Vec::new()).collect();
    let default_b = if wildcard_arm_idx >= 0 {
        arm_blocks[wildcard_arm_idx as usize]
    } else {
        gctx.jfn.push_block("nomatch")
    };

    let scrut_is_int = gctx.ctx.type_pool.get(scrut_ty).kind == TypeKind::Int;
    let enum_name = gctx.ctx.enum_name_of(scrut_ty);
    let scrut_is_enum = enum_name.is_some();
    let mut trying_switch = scrut_is_int || scrut_is_enum;
    let mut switch_cases: Vec<SwitchCase> = Vec::new();
    if trying_switch {
        for i in 0..arm_count as usize {
            if i as i64 == wildcard_arm_idx {
                continue;
            }
            if !collect_switch_cases(
                gctx,
                arms[i].0,
                arm_blocks[i],
                scrut_ty,
                scrut_is_enum,
                enum_name.as_deref(),
                &mut switch_cases,
            ) {
                trying_switch = false;
                break;
            }
        }
        // LLVM SwitchInst needs unique case values; keep the earliest
        // (first-match-wins).
        let mut seen = std::collections::HashSet::new();
        switch_cases.retain(|c| seen.insert(c.value));
    }

    if trying_switch {
        // Integer scrutinee for the Switch — an enum yields its discriminant byte.
        let case_scrut = if scrut_is_enum {
            let has_payload = gctx
                .ctx
                .enum_has_payload_by_name(enum_name.as_deref().unwrap())
                .unwrap_or(false);
            emit(
                gctx,
                JirInst {
                    tag: if has_payload {
                        JirTag::ExtractValue
                    } else {
                        JirTag::BitCast
                    },
                    a: scrut,
                    b: 0,
                    ty: builtin::U8,
                    ..Default::default()
                },
            )
        } else {
            scrut
        };
        let mut packed: Vec<u32> = Vec::with_capacity(2 + switch_cases.len() * 4);
        packed.push(default_b);
        packed.push(switch_cases.len() as u32);
        for sc in &switch_cases {
            packed.push((sc.value & 0xFFFF_FFFF) as u32);
            packed.push((sc.value >> 32) as u32);
            packed.push(if sc.is_signed { 1 } else { 0 });
            packed.push(sc.target);
        }
        let extra = gctx.jfn.push_extra(&packed);
        emit(
            gctx,
            JirInst {
                tag: JirTag::Switch,
                a: case_scrut,
                b: extra,
                ..Default::default()
            },
        );
        if wildcard_arm_idx < 0 {
            gctx.current_block = default_b;
            if match_owns {
                emit_drop_in_place(gctx, scrut_owned, scrut_ty);
            }
            emit_br(gctx, merge_b);
        }
    } else {
        let mut emitted_any = false;
        for i in 0..arm_count as usize {
            if i as i64 == wildcard_arm_idx {
                continue;
            }
            let next = if i + 1 < arm_count as usize && (i + 1) as i64 != wildcard_arm_idx {
                gctx.jfn.push_block("matchnext")
            } else {
                default_b
            };
            astgen_pattern_compare(
                gctx,
                arms[i].0,
                scrut,
                scrut_ty,
                arm_blocks[i],
                next,
                &mut arm_bindings[i],
            )?;
            if next != default_b {
                gctx.current_block = next;
            }
            emitted_any = true;
        }
        if !emitted_any {
            emit_br(gctx, default_b);
        }
        if wildcard_arm_idx < 0 {
            gctx.current_block = default_b;
            if match_owns {
                emit_drop_in_place(gctx, scrut_owned, scrut_ty);
            }
            emit_br(gctx, merge_b);
        }
    }

    // Arm bodies — install bindings, run statements (tail stores to the result
    // slot in expression form), restore the prior locals. Arm bodies run at +1
    // runtime-conditional depth (comp bindings declared outside may not be
    // mutated inside an arm).
    gctx.runtime_cond_depth += 1;
    for i in 0..arm_count as usize {
        gctx.current_block = arm_blocks[i];
        let bindings = std::mem::take(&mut arm_bindings[i]);
        let mut saved: Vec<(String, Option<(JirRef, TypeIdx)>)> = Vec::new();
        for bind in &bindings {
            let prior = gctx
                .locals
                .get(&bind.name)
                .map(|&s| (s, gctx.local_types[&bind.name]));
            saved.push((bind.name.clone(), prior));
            gctx.locals.insert(bind.name.clone(), bind.slot);
            gctx.local_types.insert(bind.name.clone(), bind.ty);
        }
        push_drop_scope(gctx);

        if match_owns {
            if bindings.is_empty() {
                emit_drop_in_place(gctx, scrut_owned, scrut_ty);
            } else if gctx.ctx.lookup_drop_fn_name(scrut_ty).is_some() {
                return Err(
                    "astgen: cannot bind the payload out of an enum that has its own `cfn drop`"
                        .into(),
                );
            } else {
                for bind in &bindings {
                    if gctx.ctx.type_needs_drop(bind.ty) {
                        gctx.drop_scopes.last_mut().unwrap().push(DropTrack {
                            var_name: bind.name.clone(),
                            slot: bind.slot,
                            ty: bind.ty,
                        });
                    }
                }
            }
        }

        let body = arms[i].1.clone();
        let mut arm_diverged = false;
        for (s, &stmt) in body.iter().enumerate() {
            let is_tail = s + 1 == body.len();
            let divergent = stmt_diverges(gctx, stmt);
            if is_tail && !peer.is_none() && !divergent {
                let val = astgen_expr(gctx, stmt, peer)?;
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Store,
                        a: result_slot,
                        b: val,
                        ..Default::default()
                    },
                );
            } else {
                astgen_expr(gctx, stmt, TypeIdx::NONE)?;
            }
            if is_tail && divergent {
                arm_diverged = true;
            }
        }
        if arm_diverged && !block_has_terminator(gctx) {
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Unreachable,
                    ..Default::default()
                },
            );
        }
        pop_drop_scope_emitting(gctx);
        if !block_has_terminator(gctx) {
            emit_br(gctx, merge_b);
        }
        for (name, prior) in saved {
            match prior {
                Some((slot, ty)) => {
                    gctx.locals.insert(name.clone(), slot);
                    gctx.local_types.insert(name, ty);
                }
                None => {
                    gctx.locals.remove(&name);
                    gctx.local_types.remove(&name);
                }
            }
        }
    }
    gctx.runtime_cond_depth -= 1;

    gctx.current_block = merge_b;
    if result_slot == NO_JIR_REF {
        Ok(NO_JIR_REF)
    } else {
        Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: result_slot,
                ty: peer,
                ..Default::default()
            },
        ))
    }
}

/// Peek (no JIR emitted) whether `node` resolves to a fixed `Array` through an
/// addressable chain — a local `Variable`, or a `MemberAccess` on a local
/// struct. Returns the array type for the index fast path, else `None`.
fn peek_addressable_array_leaf_type(gctx: &AstGenCtx, node: NodeIdx) -> Option<TypeIdx> {
    let n = *gctx.ctx.node_store.get(node);
    let ty = match n.tag {
        AstTag::Variable => *gctx.local_types.get(&str_at(gctx, n.lhs))?,
        AstTag::MemberAccess => {
            let parent = *gctx.ctx.node_store.get(NodeIdx::new(n.lhs));
            if parent.tag != AstTag::Variable {
                return None;
            }
            let parent_ty = *gctx.local_types.get(&str_at(gctx, parent.lhs))?;
            let field = str_at(gctx, n.rhs);
            gctx.ctx
                .struct_fields(parent_ty)?
                .into_iter()
                .find(|(nm, _)| *nm == field)
                .map(|(_, t)| t)?
        }
        _ => return None,
    };
    if gctx.ctx.type_pool.get(ty).kind == TypeKind::Array {
        Some(ty)
    } else {
        None
    }
}

/// Lower an `ArrayLit` (`lhs`=elem TypeIdx or 0, `rhs`=ExtraIdx → `[count,
/// elems..]`) to a `[N]elem` SSA aggregate (jir_codegen emits the InsertValue
/// chain). Each bare drop-bearing local element is MOVED into the array.
fn astgen_array_lit(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let mut elem_ty = TypeIdx::new(n.lhs);
    if elem_ty.is_none() && !expected.is_none() {
        let ek = gctx.ctx.type_pool.get(expected);
        if ek.kind == TypeKind::Array {
            elem_ty = TypeIdx::new(ek.a);
        }
    }
    let elems_extra = n.rhs;
    let count = gctx.ctx.node_store.get_extra(ExtraIdx::new(elems_extra));
    if !expected.is_none() {
        let ek = gctx.ctx.type_pool.get(expected);
        if ek.kind == TypeKind::Array && ek.b != count {
            return Err(format!(
                "array literal has {count} element(s) but the array type expects {}",
                ek.b
            ));
        }
    }
    let mut elems: Vec<JirRef> = Vec::with_capacity(count as usize);
    for i in 0..count {
        let e = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(elems_extra + 1 + i)),
        );
        // Capturing a drop-bearing field extracted out of an aggregate into an
        // array element is rejected (the C++ rejectDropBearingFieldExtract,
        // astgen.cpp:1989).
        reject_drop_bearing_field_extract(gctx, e, elem_ty, "capture")?;
        let r = astgen_expr(gctx, e, elem_ty)?;
        consume_moved_variable(gctx, e);
        elems.push(r);
    }
    if elem_ty.is_none() && !elems.is_empty() {
        elem_ty = gctx.jfn.get_inst(elems[0]).ty;
    }
    if elem_ty.is_none() {
        return Err("astgen: array literal element type could not be inferred".into());
    }
    let arr_ty = gctx.ctx.type_pool.intern_array(elem_ty, count);
    let mut packed: Vec<u32> = Vec::with_capacity(1 + elems.len());
    packed.push(count);
    packed.extend_from_slice(&elems);
    let extra = gctx.jfn.push_extra(&packed);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::ArrayLit,
            b: extra,
            ty: arr_ty,
            ..Default::default()
        },
    ))
}

/// Lower an `ArrayRepeat` (`[expr; N]`). A constant zero / byte fill lowers to a
/// single `MemSet`; everything else expands to an N-copy `ArrayLit`. Drop-
/// bearing element types are rejected (N owners of one value). Non-literal
/// counts (which need the comptime folder) are deferred.
fn astgen_array_repeat(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let mut arr_ty = TypeIdx::new(n.lhs);
    if arr_ty.is_none() {
        arr_ty = expected;
    }
    if !arr_ty.is_none() && gctx.ctx.type_pool.get(arr_ty).kind == TypeKind::ArrayExpr {
        arr_ty = gctx.ctx.resolve_array_expr_instantiate(arr_ty)?;
    }
    let extra = n.rhs;
    let value_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra)));
    let count_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1)));
    let cn = *gctx.ctx.node_store.get(count_idx);
    let count: u64 = if cn.tag == AstTag::NumberLit {
        (cn.lhs as u64) | ((cn.rhs as u64) << 32)
    } else {
        // Comptime-fold a non-literal count (a `comp const` / const expr).
        let v = gctx.ctx.fold_comptime_expr(count_idx);
        if !v.is_int() {
            return Err("astgen: array-repeat count is not comptime-foldable".into());
        }
        v.as_u64()
    };
    if count > u32::MAX as u64 {
        return Err(format!("array repeat count {count} exceeds u32 range"));
    }

    let mut elem_ty = TypeIdx::NONE;
    if !arr_ty.is_none() {
        let k = gctx.ctx.type_pool.get(arr_ty);
        if k.kind == TypeKind::Array {
            elem_ty = TypeIdx::new(k.a);
            if k.b as u64 != count {
                return Err(format!(
                    "array repeat count {count} does not match array type length {}",
                    k.b
                ));
            }
        }
    }
    let val = astgen_expr(gctx, value_idx, elem_ty)?;
    if elem_ty.is_none() {
        elem_ty = gctx.jfn.get_inst(val).ty;
    }
    if arr_ty.is_none() {
        arr_ty = gctx.ctx.type_pool.intern_array(elem_ty, count as u32);
    }
    if gctx.ctx.type_needs_drop(elem_ty) {
        return Err(format!(
            "repeat literal would create {count} owners of one drop-bearing value; \
             initialize each element explicitly"
        ));
    }

    // Constant zero / byte fill → one MemSet.
    let vinst = *gctx.jfn.get_inst(val);
    if vinst.tag == JirTag::Int {
        let fv = (vinst.a as u64) | ((vinst.b as u64) << 32);
        let elem_sz = gctx.ctx.type_size(elem_ty)?;
        if fv == 0 || (elem_sz == 1 && fv <= 255) {
            let slot = emit_alloca_hoisted(
                gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: arr_ty,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::MemSet,
                    a: slot,
                    b: (count * elem_sz) as u32,
                    flags: (fv & 0xFF) as u16,
                    ..Default::default()
                },
            );
            return Ok(slot);
        }
    }

    let mut packed: Vec<u32> = Vec::with_capacity(1 + count as usize);
    packed.push(count as u32);
    for _ in 0..count {
        packed.push(val);
    }
    let extra2 = gctx.jfn.push_extra(&packed);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::ArrayLit,
            b: extra2,
            ty: arr_ty,
            ..Default::default()
        },
    ))
}

/// The receiver `JirRef` for a `cfn` index method on a `Variable` container base
/// (`v[i]` / `v[i] = x`): `AddrOf`/`Load` of the local slot per the `self` ABI,
/// plus the resolved method AST + its instance type. `None` when the base isn't a
/// local struct with the named `cfn` method (caller falls through / errors).
fn container_index_recv(
    gctx: &mut AstGenCtx,
    base_idx: NodeIdx,
    method_suffix: &str,
) -> Result<Option<(JirRef, FunctionAST)>, String> {
    let bnode = *gctx.ctx.node_store.get(base_idx);
    if bnode.tag != AstTag::Variable {
        return Ok(None);
    }
    let name = str_at(gctx, bnode.lhs);
    if !gctx.locals.contains_key(&name) {
        return Ok(None);
    }
    let inst_ty = gctx
        .local_types
        .get(&name)
        .copied()
        .unwrap_or(TypeIdx::NONE);
    let Some(recv_name) = gctx.ctx.struct_name_of(inst_ty) else {
        return Ok(None);
    };
    let qualified = format!("{recv_name}.{method_suffix}");
    let Some(method) = gctx.ctx.get_function_ast(&qualified) else {
        // `at`/`setAt` may be a WITHDRAWN conditional method for this
        // instantiation (e.g. Vec(T) where T isn't cloneable): replay the
        // reason instead of the generic index error (the C++ astgen.cpp:1598,
        // 2699).
        if gctx.ctx.get_withdrawn_method(&qualified).is_some() {
            report_method_miss(gctx, &qualified)?;
        }
        return Ok(None);
    };
    if !method.is_cfn {
        return Ok(None);
    }
    let self_mode = method
        .args
        .first()
        .map(|p| p.mode)
        .unwrap_or(jam_core::param_mode::ParamMode::Let);
    // The receiver via the lvalue path (the C++ `astgenLvalue`): a Variable base's
    // storage pointer is just its alloca ref — no extra `AddrOf`. ByPointer self
    // takes that pointer; ByValue self loads through it.
    let (slot_ptr, _) = astgen_lvalue(gctx, base_idx)?;
    let recv = if classify_param(self_mode, inst_ty, gctx.ctx)?.kind == ParamAbiKind::ByPointer {
        slot_ptr
    } else {
        emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: slot_ptr,
                ty: inst_ty,
                ..Default::default()
            },
        )
    };
    Ok(Some((recv, method)))
}

/// Lower a container index expression the way the oracle's index-method dispatch
/// does: at its natural `u64` width, then `Trunc` to the method's index-param type
/// (typically `u32`). Matches the `Int(u64)`+`Trunc` the oracle emits.
fn lower_index_arg(
    gctx: &mut AstGenCtx,
    idx_idx: NodeIdx,
    param_ty: TypeIdx,
) -> Result<JirRef, String> {
    let idx_u64 = astgen_expr(gctx, idx_idx, builtin::U64)?;
    Ok(narrow_index(gctx, idx_u64, param_ty))
}

/// Narrow an already-lowered `u64` index to the index-method's param type (a
/// `Trunc` to the typical `u32`; a no-op at `u64`). Kept separate from
/// `lower_index_arg` so the `setAt` dispatch can lower the value BETWEEN the index
/// and this narrowing, matching the oracle's emitStructCfnDispatch order.
fn narrow_index(gctx: &mut AstGenCtx, idx_u64: JirRef, param_ty: TypeIdx) -> JirRef {
    if param_ty == builtin::U64 {
        return idx_u64;
    }
    emit(
        gctx,
        JirInst {
            tag: JirTag::Trunc,
            a: idx_u64,
            ty: param_ty,
            ..Default::default()
        },
    )
}

/// Lower an `Index` (`base[idx]`) in value position. An addressable `Array` base
/// reads through `IndexAddr` + `Load` (avoids loading the whole backing
/// storage); a value base (slice / many-pointer / array SSA) lowers to `Index`.
/// A struct container `v[i]` dispatches its `cfn at(self, i)`.
fn astgen_index(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let base_idx = NodeIdx::new(n.lhs);
    let idx_idx = NodeIdx::new(n.rhs);

    // Container `v[i]` -> `v.at(i)` cfn dispatch (Variable base).
    if let Some((recv, method)) = container_index_recv(gctx, base_idx, "at")? {
        let idx_ref = lower_index_arg(gctx, idx_idx, method.args[1].ty)?;
        return emit_call(gctx, &method, &[recv, idx_ref], NO_JIR_REF);
    }

    if peek_addressable_array_leaf_type(gctx, base_idx).is_some() {
        let (base_ptr, leaf_ty) = astgen_lvalue(gctx, base_idx)?;
        let (kind, a) = {
            let k = gctx.ctx.type_pool.get(leaf_ty);
            (k.kind, k.a)
        };
        if kind == TypeKind::Array {
            let elem_ty = TypeIdx::new(a);
            let idx_ref = astgen_expr(gctx, idx_idx, builtin::U64)?;
            let elem_ptr_ty = gctx.ctx.type_pool.intern_ptr_single(elem_ty);
            let elem_ptr = emit(
                gctx,
                JirInst {
                    tag: JirTag::IndexAddr,
                    a: base_ptr,
                    b: idx_ref,
                    ty: elem_ptr_ty,
                    ..Default::default()
                },
            );
            return Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Load,
                    a: elem_ptr,
                    ty: elem_ty,
                    ..Default::default()
                },
            ));
        }
    }

    let base_ref = astgen_expr(gctx, base_idx, TypeIdx::NONE)?;
    let idx_ref = astgen_expr(gctx, idx_idx, builtin::U64)?;
    let (kind, a) = {
        let k = gctx.ctx.type_pool.get(gctx.jfn.get_inst(base_ref).ty);
        (k.kind, k.a)
    };
    let elem_ty = match kind {
        TypeKind::Array | TypeKind::Slice | TypeKind::PtrMany => TypeIdx::new(a),
        _ => return Err("astgen: cannot index value of this type".into()),
    };
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::Index,
            a: base_ref,
            b: idx_ref,
            ty: elem_ty,
            ..Default::default()
        },
    ))
}

/// Lower a `MemberAccess` (`lhs`=base, `rhs`=member StringIdx) in value
/// position. An addressable base reads through a `FieldAddr`/`BitCast` + `Load`
/// (no whole-aggregate value load); a value base uses `ExtractValue` (slice
/// `.ptr`/`.len`), a spill+`Load` (union), or `FieldAccess` (struct).
///
/// Deferred: enum-variant constructor references (`Color.Red`) — they fall
/// through to the lvalue/value path and surface as an unknown-variable error.
/// Lower a slice expression `base[a..b]` (the C++ `astgenSlice`): the base is a
/// many-item pointer; emit `IndexAddr(base, a)` for the data pointer and `b - a`
/// (u64) for the length, then a `MakeSlice` `{ptr, len}` aggregate. Implemented
/// wired into astgen_expr dispatch.
fn astgen_slice(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let base_idx = NodeIdx::new(n.lhs);
    let ex = n.rhs;
    let start_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(ex)));
    let end_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(ex + 1)));

    let base = astgen_expr(gctx, base_idx, TypeIdx::NONE)?;
    let bk = gctx.ctx.type_pool.get(gctx.jfn.get_inst(base).ty);
    if bk.kind != TypeKind::PtrMany && bk.kind != TypeKind::PtrSingle {
        // Recoverable (the C++ astgen.cpp:2742).
        return Ok(recover_here(
            gctx,
            "slice expression `base[a..b]` needs a many-item pointer base".to_string(),
            TypeIdx::NONE,
        ));
    }
    let elem = TypeIdx::new(bk.a);

    let mut start = astgen_expr(gctx, start_idx, builtin::U64)?;
    let mut end = astgen_expr(gctx, end_idx, builtin::U64)?;
    if gctx.jfn.get_inst(start).ty != builtin::U64 {
        start = emit(
            gctx,
            JirInst {
                tag: JirTag::ZExt,
                a: start,
                ty: builtin::U64,
                ..Default::default()
            },
        );
    }
    if gctx.jfn.get_inst(end).ty != builtin::U64 {
        end = emit(
            gctx,
            JirInst {
                tag: JirTag::ZExt,
                a: end,
                ty: builtin::U64,
                ..Default::default()
            },
        );
    }

    let ptr_ty = gctx.ctx.type_pool.intern_ptr_many(elem);
    let ptr = emit(
        gctx,
        JirInst {
            tag: JirTag::IndexAddr,
            a: base,
            b: start,
            ty: ptr_ty,
            ..Default::default()
        },
    );
    let len = emit(
        gctx,
        JirInst {
            tag: JirTag::Sub,
            a: end,
            b: start,
            ty: builtin::U64,
            ..Default::default()
        },
    );
    let slice_ty = gctx.ctx.type_pool.intern_slice(elem);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::MakeSlice,
            a: ptr,
            b: len,
            ty: slice_ty,
            ..Default::default()
        },
    ))
}

/// Lower `Deref` (`p.*`): the operand evaluates to a pointer; emit a `Deref`
/// (a Load through it) typed at the pointee.
fn astgen_deref(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let ptr_ref = astgen_expr(gctx, NodeIdx::new(n.lhs), TypeIdx::NONE)?;
    let (kind, pointee) = {
        let k = gctx.ctx.type_pool.get(gctx.jfn.get_inst(ptr_ref).ty);
        (k.kind, k.a)
    };
    if kind != TypeKind::PtrSingle && kind != TypeKind::PtrMany {
        return Err("astgen: cannot dereference non-pointer".into());
    }
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::Deref,
            a: ptr_ref,
            ty: TypeIdx::new(pointee),
            ..Default::default()
        },
    ))
}

/// Lower `&op` (`AddressOf`): an lvalueable operand hands over its storage
/// pointer; an rvalue is spilled to a fresh slot. Result is `*const leafTy`.
fn astgen_address_of(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let op_idx = NodeIdx::new(n.lhs);
    let op_tag = gctx.ctx.node_store.get(op_idx).tag;
    let (ptr_ref, leaf_ty) = match op_tag {
        AstTag::Variable | AstTag::MemberAccess | AstTag::Index | AstTag::Deref => {
            astgen_lvalue(gctx, op_idx)?
        }
        _ => {
            let val = astgen_expr(gctx, op_idx, TypeIdx::NONE)?;
            let leaf = gctx.jfn.get_inst(val).ty;
            let slot = emit_alloca_hoisted(
                gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: leaf,
                    ..Default::default()
                },
            );
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: slot,
                    b: val,
                    ..Default::default()
                },
            );
            (slot, leaf)
        }
    };
    let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(leaf_ty);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::AddrOf,
            a: ptr_ref,
            ty: ptr_ty,
            ..Default::default()
        },
    ))
}

fn astgen_member_access(gctx: &mut AstGenCtx, n: &AstNode) -> Result<JirRef, String> {
    let base_idx = NodeIdx::new(n.lhs);
    let member = str_at(gctx, n.rhs);
    let base_node = *gctx.ctx.node_store.get(base_idx);
    let base_tag = base_node.tag;

    // Enum-variant constructor reference (`Color.Red`): a `Variable` base whose
    // name is a registered enum (and isn't a runtime local). Unit-only enums
    // lower to the tag i8 retagged to the enum type; a payloaded enum's unit
    // variant builds a `{tag, undef}` StructLit. (Handle-qualified chains —
    // `a.Status.Ok` — land with module resolution.)
    if base_tag == AstTag::Variable {
        let base_name = str_at(gctx, base_node.lhs);
        // Mirror the C++ `astgenMemberAccess` (codegen.cpp:2053-2055, 2094): a
        // pure-Variable base ALWAYS interns its name as a `Named` enum probe,
        // even when the name is a runtime local — `self.field` interns
        // `Named("self")` before the field projection. The earlier `!locals`
        // guard skipped this, leaving the type pool one `Named` short and
        // shifting every later GenericCall's TypeIdx (the `Vec__T<idx>`
        // monomorph spelling read 4 too low).
        {
            let sid = gctx.ctx.string_pool.intern(base_name.as_bytes());
            let named = gctx.ctx.type_pool.intern_named(sid);
            let bm = gctx.ctx.current_body_module();
            let named = gctx.ctx.requalify_type(named, &bm);
            if !gctx.locals.contains_key(&base_name)
                && let Some(en) = gctx.ctx.enum_name_of(named)
            {
                let vidx = gctx.ctx.enum_variant_index(&en, &member);
                if vidx < 0 {
                    return Err(format!("astgen: enum `{en}` has no variant `{member}`"));
                }
                let disc = gctx.ctx.enum_variants_by_name(&en).unwrap()[vidx as usize].discriminant;
                let has_payload = gctx.ctx.enum_has_payload_by_name(&en).unwrap_or(false);
                let esid = gctx.ctx.string_pool.intern(en.as_bytes());
                let enum_ty = gctx.ctx.type_pool.intern_named(esid);
                let tag = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Int,
                        a: disc,
                        ty: builtin::U8,
                        ..Default::default()
                    },
                );
                if !has_payload {
                    return Ok(emit(
                        gctx,
                        JirInst {
                            tag: JirTag::BitCast,
                            a: tag,
                            ty: enum_ty,
                            ..Default::default()
                        },
                    ));
                }
                let extra = gctx.jfn.push_extra(&[1, tag]);
                return Ok(emit(
                    gctx,
                    JirInst {
                        tag: JirTag::StructLit,
                        b: extra,
                        ty: enum_ty,
                        ..Default::default()
                    },
                ));
            }
        }
    }

    // Handle-qualified enum-variant reference (`a.Status.Ok`): the base is itself
    // a `MemberAccess` whose own base is a non-local `Variable` (an import handle).
    // Build the dotted `handle.Type` spelling, intern it as a `Named`, and if it
    // resolves to a registered enum (via the A2 `handle.Type` alias), emit the
    // unit-variant reference — exactly like the single-dot `Color.Red` path above.
    // The multi-dot CALL form (`a.Status.Bad(5)`) is handled in astgen_dotted_call.
    if base_tag == AstTag::MemberAccess {
        let inner_base = *gctx.ctx.node_store.get(NodeIdx::new(base_node.lhs));
        if inner_base.tag == AstTag::Variable {
            let handle = str_at(gctx, inner_base.lhs);
            let type_name = str_at(gctx, base_node.rhs);
            if !gctx.locals.contains_key(&handle) {
                let dotted = format!("{handle}.{type_name}");
                let sid = gctx.ctx.string_pool.intern(dotted.as_bytes());
                let named = gctx.ctx.type_pool.intern_named(sid);
                if let Some(en) = gctx.ctx.enum_name_of(named) {
                    let vidx = gctx.ctx.enum_variant_index(&en, &member);
                    if vidx < 0 {
                        return Err(format!("astgen: enum `{en}` has no variant `{member}`"));
                    }
                    let disc =
                        gctx.ctx.enum_variants_by_name(&en).unwrap()[vidx as usize].discriminant;
                    let has_payload = gctx.ctx.enum_has_payload_by_name(&en).unwrap_or(false);
                    let esid = gctx.ctx.string_pool.intern(en.as_bytes());
                    let enum_ty = gctx.ctx.type_pool.intern_named(esid);
                    let tag = emit(
                        gctx,
                        JirInst {
                            tag: JirTag::Int,
                            a: disc,
                            ty: builtin::U8,
                            ..Default::default()
                        },
                    );
                    if !has_payload {
                        return Ok(emit(
                            gctx,
                            JirInst {
                                tag: JirTag::BitCast,
                                a: tag,
                                ty: enum_ty,
                                ..Default::default()
                            },
                        ));
                    }
                    let extra = gctx.jfn.push_extra(&[1, tag]);
                    return Ok(emit(
                        gctx,
                        JirInst {
                            tag: JirTag::StructLit,
                            b: extra,
                            ty: enum_ty,
                            ..Default::default()
                        },
                    ));
                }
            }
        }
    }

    // Addressable base → field pointer + Load.
    if matches!(
        base_tag,
        AstTag::Variable | AstTag::MemberAccess | AstTag::Index | AstTag::Deref
    ) {
        let (base_ptr, base_leaf) = astgen_lvalue(gctx, base_idx)?;
        if gctx.ctx.is_union_registered(base_leaf) {
            let field_ty = gctx
                .ctx
                .union_fields(base_leaf)
                .and_then(|fs| fs.into_iter().find(|(nm, _)| *nm == member).map(|(_, t)| t))
                .ok_or_else(|| format!("astgen: union has no field `{member}`"))?;
            let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(field_ty);
            let fp = emit(
                gctx,
                JirInst {
                    tag: JirTag::BitCast,
                    a: base_ptr,
                    ty: ptr_ty,
                    ..Default::default()
                },
            );
            return Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Load,
                    a: fp,
                    ty: field_ty,
                    ..Default::default()
                },
            ));
        }
        if let Some(fields) = gctx.ctx.struct_fields(base_leaf) {
            let sname = gctx.ctx.struct_name_of(base_leaf).unwrap_or_default();
            let Some(idx) = fields.iter().position(|(nm, _)| *nm == member) else {
                // Recoverable (the C++ astgen.cpp:2172).
                return Ok(recover_here(
                    gctx,
                    format!("unknown field `{member}` on `{sname}`"),
                    TypeIdx::NONE,
                ));
            };
            let mut field_ty = fields[idx].1;
            // An `[expr]T` field resolves to its concrete `[n]T` (cached at
            // registration) for the FieldAddr/Load types.
            if gctx.ctx.type_pool.get(field_ty).kind == TypeKind::ArrayExpr {
                let r = gctx.ctx.resolve_array_expr(field_ty);
                if !r.is_none() {
                    field_ty = r;
                }
            }
            let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(field_ty);
            let fp = emit(
                gctx,
                JirInst {
                    tag: JirTag::FieldAddr,
                    a: base_ptr,
                    b: idx as u32,
                    ty: ptr_ty,
                    ..Default::default()
                },
            );
            return Ok(emit(
                gctx,
                JirInst {
                    tag: JirTag::Load,
                    a: fp,
                    ty: field_ty,
                    ..Default::default()
                },
            ));
        }
        // Slice / other base: fall through to the value path.
    }

    let base_ref = astgen_expr(gctx, base_idx, TypeIdx::NONE)?;
    let base_ty = gctx.jfn.get_inst(base_ref).ty;
    let (bk_kind, bk_a) = {
        let bk = gctx.ctx.type_pool.get(base_ty);
        (bk.kind, bk.a)
    };

    // Slice `.ptr` (field 0, `[*]elem`) / `.len` (field 1, u64).
    if bk_kind == TypeKind::Slice {
        let (field_idx, field_ty) = match member.as_str() {
            "ptr" => (0u32, gctx.ctx.type_pool.intern_ptr_many(TypeIdx::new(bk_a))),
            "len" => (1u32, builtin::U64),
            _ => return Err(format!("astgen: slice has no field `{member}`")),
        };
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::ExtractValue,
                a: base_ref,
                b: field_idx,
                ty: field_ty,
                ..Default::default()
            },
        ));
    }

    // Union value: spill to a slot, reload at the field's type (opaque pointers
    // let any field-typed value come back from the same storage).
    if gctx.ctx.is_union_registered(base_ty) {
        let field_ty = gctx
            .ctx
            .union_fields(base_ty)
            .and_then(|fs| fs.into_iter().find(|(nm, _)| *nm == member).map(|(_, t)| t))
            .ok_or_else(|| format!("astgen: union has no field `{member}`"))?;
        let slot = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty: base_ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: slot,
                b: base_ref,
                ..Default::default()
            },
        );
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: slot,
                ty: field_ty,
                ..Default::default()
            },
        ));
    }

    // Struct value: ExtractValue at the field's positional index.
    let fields = gctx
        .ctx
        .struct_fields(base_ty)
        .ok_or_else(|| "astgen: cannot access field of non-struct type".to_string())?;
    let sname = gctx.ctx.struct_name_of(base_ty).unwrap_or_default();
    let Some(idx) = fields.iter().position(|(nm, _)| *nm == member) else {
        // Recoverable (the C++ astgen.cpp:2258).
        return Ok(recover_here(
            gctx,
            format!("unknown field `{member}` on `{sname}`"),
            TypeIdx::NONE,
        ));
    };
    let field_ty = fields[idx].1;
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::FieldAccess,
            a: base_ref,
            b: idx as u32,
            ty: field_ty,
            ..Default::default()
        },
    ))
}

/// Lower a `StructLit` (`lhs`=type TypeIdx or 0, `rhs`=ExtraIdx →
/// `[fieldCount, (nameId, exprIdx)..]`). Struct literals build an SSA aggregate
/// by permuting the named field values into positional order (jir_codegen emits
/// the InsertValue chain). A union literal lists exactly one field: alloca the
/// union, store the field value into the slot, load the whole union back.
///
/// Deferred: the place-into-destination form (`astgenStructLitInto`) and
/// drop/move tracking of captured fields.
fn astgen_struct_lit(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected: TypeIdx,
) -> Result<JirRef, String> {
    let mut ty = TypeIdx::new(n.lhs);
    if ty.is_none() {
        ty = expected;
    }
    if ty.is_none() {
        return Err("astgen: struct literal without target type".into());
    }
    // A generic instance (`Pair(u64)`) must be instantiated so its fields
    // resolve; the inst type keeps its `GenericCall` spelling in the JIR.
    gctx.ctx.resolve_alias_generic_instantiate(ty)?;
    let fields_extra = n.rhs;
    let field_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(fields_extra));

    // Union literal: exactly one field, stored into a union-sized slot.
    if gctx.ctx.is_union_registered(ty) {
        if field_count != 1 {
            return Err("astgen: union literal must list exactly one field".into());
        }
        let name_id = gctx
            .ctx
            .node_store
            .get_extra(ExtraIdx::new(fields_extra + 1));
        let expr_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(fields_extra + 2)),
        );
        let field_name = str_at(gctx, name_id);
        let field_ty = gctx
            .ctx
            .union_fields(ty)
            .and_then(|fs| {
                fs.into_iter()
                    .find(|(nm, _)| *nm == field_name)
                    .map(|(_, t)| t)
            })
            .ok_or_else(|| format!("astgen: union has no field `{field_name}`"))?;
        let field_val = astgen_expr(gctx, expr_idx, field_ty)?;
        let slot = emit_alloca_hoisted(
            gctx,
            JirInst {
                tag: JirTag::Alloca,
                ty,
                ..Default::default()
            },
        );
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: slot,
                b: field_val,
                ..Default::default()
            },
        );
        return Ok(emit(
            gctx,
            JirInst {
                tag: JirTag::Load,
                a: slot,
                ty,
                ..Default::default()
            },
        ));
    }

    let fields = gctx
        .ctx
        .struct_fields(ty)
        .ok_or_else(|| "astgen: struct literal type is not a known struct".to_string())?;
    let mut ordered = vec![NO_JIR_REF; fields.len()];
    for i in 0..field_count {
        let name_id = gctx
            .ctx
            .node_store
            .get_extra(ExtraIdx::new(fields_extra + 1 + i * 2));
        let expr_idx = NodeIdx::new(
            gctx.ctx
                .node_store
                .get_extra(ExtraIdx::new(fields_extra + 2 + i * 2)),
        );
        let field_name = str_at(gctx, name_id);
        let idx = match fields.iter().position(|(nm, _)| *nm == field_name) {
            Some(p) => p,
            None => {
                // Recoverable (the C++ astgen.cpp:1749): record the bad field
                // name and skip it so other malformed fields in the same
                // literal still get reported in this pass.
                append_error_here(gctx, format!("unknown struct field `{field_name}`"));
                continue;
            }
        };
        let expected_field = fields[idx].1;
        // Capturing a drop-bearing field extracted out of an aggregate would leave
        // the aggregate's glue to re-drop it — reject (the C++
        // rejectDropBearingFieldExtract, astgen.cpp:1754).
        reject_drop_bearing_field_extract(gctx, expr_idx, expected_field, "capture")?;
        let mut field_val = astgen_expr(gctx, expr_idx, expected_field)?;
        // Struct-literal field capture MOVES a bare drop-bearing local: the
        // field now owns it, so its scope-exit drop is suppressed.
        consume_moved_variable(gctx, expr_idx);
        // Silent int→float widening (matches the legacy struct codegen): an
        // integer literal in a float field settles via SIToFP / UIToFP rather
        // than feeding an int into the float slot of the InsertValue chain.
        let vt = gctx.jfn.get_inst(field_val).ty;
        if vt != expected_field && !vt.is_none() {
            let fk_kind = gctx.ctx.type_pool.get(expected_field).kind;
            let (vk_kind, vk_signed) = {
                let vk = gctx.ctx.type_pool.get(vt);
                (vk.kind, vk.b != 0)
            };
            if fk_kind == TypeKind::Float && vk_kind == TypeKind::Int {
                let tag = if vk_signed {
                    JirTag::SIToFP
                } else {
                    JirTag::UIToFP
                };
                field_val = emit(
                    gctx,
                    JirInst {
                        tag,
                        a: field_val,
                        ty: expected_field,
                        ..Default::default()
                    },
                );
            }
        }
        ordered[idx] = field_val;
    }
    for (i, &r) in ordered.iter().enumerate() {
        if r == NO_JIR_REF {
            return Err(format!(
                "astgen: struct literal missing field `{}`",
                fields[i].0
            ));
        }
    }

    let mut packed: Vec<u32> = Vec::with_capacity(1 + ordered.len());
    packed.push(ordered.len() as u32);
    packed.extend_from_slice(&ordered);
    let extra = gctx.jfn.push_extra(&packed);
    Ok(emit(
        gctx,
        JirInst {
            tag: JirTag::StructLit,
            b: extra,
            ty,
            ..Default::default()
        },
    ))
}

/// Emit a `SretArg` — a pointer (`*const retTy`) to the caller-provided return
/// slot for an indirect (sret) return.
fn emit_sret_arg(gctx: &mut AstGenCtx, ret_ty: TypeIdx) -> JirRef {
    let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(ret_ty);
    emit(
        gctx,
        JirInst {
            tag: JirTag::SretArg,
            ty: ptr_ty,
            ..Default::default()
        },
    )
}

/// Try to lower `expr_idx` directly into `dest_ptr`, returning `true` when the
/// destination was written (no SSA value remains for the caller to bind).
/// Byref producers — `StructLit` (per-field stores) and sret `Call` (forwarded
/// sret slot) — place in-line; everything else returns `false` so the caller
/// takes the value-compile + Store path. (ArrayLit/TypeMethodCall place paths
/// land with those handlers.)
fn astgen_expr_into_ptr(
    gctx: &mut AstGenCtx,
    expr_idx: NodeIdx,
    expected_ty: TypeIdx,
    dest_ptr: JirRef,
) -> Result<bool, String> {
    let n = *gctx.ctx.node_store.get(expr_idx);
    match n.tag {
        AstTag::StructLit => {
            astgen_struct_lit_into(gctx, &n, expected_ty, dest_ptr)?;
            Ok(true)
        }
        AstTag::Call => {
            let r = astgen_call(gctx, &n, dest_ptr)?;
            // A real ref means a Direct (ByValue) return — finish by storing it.
            // `NO_JIR_REF` means the call used `dest_ptr` as its sret slot.
            if r != NO_JIR_REF {
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Store,
                        a: dest_ptr,
                        b: r,
                        ..Default::default()
                    },
                );
            }
            Ok(true)
        }
        AstTag::TypeMethodCall => {
            let r = astgen_type_method_call(gctx, &n, dest_ptr)?;
            if r != NO_JIR_REF {
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Store,
                        a: dest_ptr,
                        b: r,
                        ..Default::default()
                    },
                );
            }
            Ok(true)
        }
        AstTag::ArrayLit => {
            // Per-element write into dest_ptr (no SSA array). ArrayRepeat
            // deliberately falls through to the value+Store path (keeps JIR
            // compact; jir_codegen's byref ArrayLit emits the per-element stores).
            let mut e_ty = TypeIdx::new(n.lhs);
            if e_ty.is_none() && !expected_ty.is_none() {
                let ek = gctx.ctx.type_pool.get(expected_ty);
                if ek.kind == TypeKind::Array {
                    e_ty = TypeIdx::new(ek.a);
                }
            }
            if e_ty.is_none() {
                return Ok(false);
            }
            let elems_extra = n.rhs;
            let count = gctx.ctx.node_store.get_extra(ExtraIdx::new(elems_extra));
            if !expected_ty.is_none() {
                let ek = gctx.ctx.type_pool.get(expected_ty);
                if ek.kind == TypeKind::Array && ek.b != count {
                    return Err(format!(
                        "array literal has {count} element(s) but the array type expects {}",
                        ek.b
                    ));
                }
            }
            let elem_ptr_ty = gctx.ctx.type_pool.intern_ptr_single(e_ty);
            for i in 0..count {
                let e_idx = NodeIdx::new(
                    gctx.ctx
                        .node_store
                        .get_extra(ExtraIdx::new(elems_extra + 1 + i)),
                );
                let idx_ref = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Int,
                        a: i,
                        ty: builtin::U64,
                        ..Default::default()
                    },
                );
                let ep = emit(
                    gctx,
                    JirInst {
                        tag: JirTag::IndexAddr,
                        a: dest_ptr,
                        b: idx_ref,
                        ty: elem_ptr_ty,
                        ..Default::default()
                    },
                );
                if !astgen_expr_into_ptr(gctx, e_idx, e_ty, ep)? {
                    let ev = astgen_expr(gctx, e_idx, e_ty)?;
                    emit(
                        gctx,
                        JirInst {
                            tag: JirTag::Store,
                            a: ep,
                            b: ev,
                            ..Default::default()
                        },
                    );
                }
                consume_moved_variable(gctx, e_idx);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Write a `StructLit` directly into `dest_ptr` (per-field `FieldAddr` + place /
/// store) instead of building an SSA aggregate. Unions / non-structs fall back
/// to the value-form `StructLit` + `Store`.
fn astgen_struct_lit_into(
    gctx: &mut AstGenCtx,
    n: &AstNode,
    expected_ty: TypeIdx,
    dest_ptr: JirRef,
) -> Result<(), String> {
    let mut ty = TypeIdx::new(n.lhs);
    if ty.is_none() {
        ty = expected_ty;
    }
    if ty.is_none() {
        return Err("astgen: struct literal without target type".into());
    }
    gctx.ctx.resolve_alias_generic_instantiate(ty)?;
    if !gctx.ctx.is_struct_registered(ty) || gctx.ctx.is_union_registered(ty) {
        let val = astgen_struct_lit(gctx, n, expected_ty)?;
        emit(
            gctx,
            JirInst {
                tag: JirTag::Store,
                a: dest_ptr,
                b: val,
                ..Default::default()
            },
        );
        return Ok(());
    }

    let fields = gctx.ctx.struct_fields(ty).unwrap();
    let fields_extra = n.rhs;
    let field_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(fields_extra));

    // Map declared-field index -> initializer NodeIdx (0 = unset).
    let mut expr_by_idx = vec![0u32; fields.len()];
    let mut has_field = vec![false; fields.len()];
    for i in 0..field_count {
        let name_id = gctx
            .ctx
            .node_store
            .get_extra(ExtraIdx::new(fields_extra + 1 + i * 2));
        let expr_idx = gctx
            .ctx
            .node_store
            .get_extra(ExtraIdx::new(fields_extra + 2 + i * 2));
        let field_name = str_at(gctx, name_id);
        match fields.iter().position(|(nm, _)| *nm == field_name) {
            Some(p) => {
                expr_by_idx[p] = expr_idx;
                has_field[p] = true;
            }
            None => {
                // Recoverable (the C++ astgen.cpp:1854).
                append_error_here(gctx, format!("unknown struct field `{field_name}`"));
            }
        }
    }

    for i in 0..fields.len() {
        if !has_field[i] {
            return Err(format!(
                "astgen: struct literal missing field `{}`",
                fields[i].0
            ));
        }
        let expected_field = fields[i].1;
        let expr_idx = NodeIdx::new(expr_by_idx[i]);
        // Field capture (place path) extracting a drop-bearing field out of an
        // aggregate is rejected (the C++ rejectDropBearingFieldExtract,
        // astgen.cpp:1868).
        reject_drop_bearing_field_extract(gctx, expr_idx, expected_field, "capture")?;
        let ptr_ty = gctx.ctx.type_pool.intern_ptr_single(expected_field);
        let field_ptr = emit(
            gctx,
            JirInst {
                tag: JirTag::FieldAddr,
                a: dest_ptr,
                b: i as u32,
                ty: ptr_ty,
                ..Default::default()
            },
        );
        if !astgen_expr_into_ptr(gctx, expr_idx, expected_field, field_ptr)? {
            let mut val = astgen_expr(gctx, expr_idx, expected_field)?;
            let vt = gctx.jfn.get_inst(val).ty;
            if vt != expected_field && !vt.is_none() {
                let fk_kind = gctx.ctx.type_pool.get(expected_field).kind;
                let (vk_kind, vk_signed) = {
                    let vk = gctx.ctx.type_pool.get(vt);
                    (vk.kind, vk.b != 0)
                };
                if fk_kind == TypeKind::Float && vk_kind == TypeKind::Int {
                    let tag = if vk_signed {
                        JirTag::SIToFP
                    } else {
                        JirTag::UIToFP
                    };
                    val = emit(
                        gctx,
                        JirInst {
                            tag,
                            a: val,
                            ty: expected_field,
                            ..Default::default()
                        },
                    );
                }
            }
            emit(
                gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: field_ptr,
                    b: val,
                    ..Default::default()
                },
            );
        }
        // Field capture MOVES a bare drop-bearing local into the struct field.
        consume_moved_variable(gctx, expr_idx);
    }
    Ok(())
}

fn astgen_return(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let mut val_ref = NO_JIR_REF;
    // A bare `return localOwned;` MOVES the value to the caller — its drop must
    // be suppressed on THIS return path (computed before lowering, which only
    // reads it). Detected up front so both the sret and value paths skip it.
    let moved_var = if n.lhs != 0 {
        detect_return_move(gctx, NodeIdx::new(n.lhs))
    } else {
        None
    };
    if n.lhs != 0 {
        let val_idx = NodeIdx::new(n.lhs);
        let ret_ty = gctx.jfn.return_type;
        // `return h.c` extracting a drop-bearing field out of an owned aggregate
        // is rejected — the caller's copy and the aggregate's glue would both drop
        // the payload (the C++ rejectDropBearingFieldExtract, astgen.cpp:1030).
        reject_drop_bearing_field_extract(gctx, val_idx, ret_ty, "return")?;
        // sret (indirect) returns place the value directly into the return slot:
        // byref producers (StructLit / sret Call) write through `SretArg`, and
        // the function returns void. The `SretArg` is emitted eagerly (matching
        // the C++ argument evaluation) even when the place path declines.
        let sret_fn =
            !ret_ty.is_none() && classify_return(ret_ty, gctx.ctx)?.kind == ReturnAbiKind::Indirect;
        if sret_fn {
            let sret = emit_sret_arg(gctx, ret_ty);
            if astgen_expr_into_ptr(gctx, val_idx, ret_ty, sret)? {
                emit_return_drops(gctx, moved_var.as_deref());
                emit(
                    gctx,
                    JirInst {
                        tag: JirTag::Ret,
                        ..Default::default()
                    },
                );
                return Ok(());
            }
        }
        val_ref = astgen_expr(gctx, val_idx, ret_ty)?;
        // Reconcile the returned value's type with the declared return type
        // (resolve-then-compare, like the var-decl declared-vs-init check).
        // The C++ oracle has NO check here and ships LLVM-invalid IR for
        // `fn f() i32 { var x = 1; return x; }` (the literal infers at
        // smallest fit, the load returns at the slot's width, and codegen
        // emits `ret i8` from an i32 function). We deliberately diverge:
        // a LOSSLESS narrower int widens (ZExt/SExt by the value's own
        // signedness — the semantics the invalid narrow `ret` accidentally
        // approximated), and anything lossy (narrowing, signed→unsigned) is
        // a hard error.
        let val_ty = gctx.jfn.get_inst(val_ref).ty;
        if !ret_ty.is_none() && !val_ty.is_none() {
            let want = resolve_for_cmp(gctx.ctx, ret_ty);
            let got = resolve_for_cmp(gctx.ctx, val_ty);
            let pointer_compatible = {
                let ka = gctx.ctx.type_pool.get(want);
                let kb = gctx.ctx.type_pool.get(got);
                let a_ptr = ka.kind == TypeKind::PtrSingle || ka.kind == TypeKind::PtrMany;
                let b_ptr = kb.kind == TypeKind::PtrSingle || kb.kind == TypeKind::PtrMany;
                a_ptr && b_ptr && ka.a == kb.a
            };
            // A unit-only enum lowers to its bare u8 tag, and the enum→int
            // cast returns that tag WITHOUT retagging (both here and in the
            // C++), so `return p as u8;` carries the enum TypeIdx at u8's
            // representation. Treat the pair as identical.
            let unit_enum_as_u8 = |t: TypeIdx, other: TypeIdx| -> bool {
                other == builtin::U8
                    && gctx.ctx.enum_name_of(t).is_some()
                    && !gctx.ctx.enum_has_payload(t).unwrap_or(true)
            };
            if want != got
                && !pointer_compatible
                && !unit_enum_as_u8(got, want)
                && !unit_enum_as_u8(want, got)
            {
                let wk = gctx.ctx.type_pool.get(want);
                let gk = gctx.ctx.type_pool.get(got);
                let lossless_widen = wk.kind == TypeKind::Int
                    && gk.kind == TypeKind::Int
                    && gk.a < wk.a
                    // Unsigned zero-extends into any wider int; signed
                    // sign-extends only into a wider SIGNED int.
                    && (gk.b == 0 || wk.b != 0);
                if lossless_widen {
                    val_ref = emit(
                        gctx,
                        JirInst {
                            tag: if gk.b != 0 { JirTag::SExt } else { JirTag::ZExt },
                            a: val_ref,
                            ty: want,
                            ..Default::default()
                        },
                    );
                } else {
                    return Err(format!(
                        "return value type does not match the declared return type of `{}`; \
                         add an explicit `as` cast or adjust the declaration",
                        gctx.jfn.name
                    ));
                }
            }
        }
    }
    emit_return_drops(gctx, moved_var.as_deref());
    emit(
        gctx,
        JirInst {
            tag: JirTag::Ret,
            a: val_ref,
            ..Default::default()
        },
    );
    Ok(())
}

/// Drop every active scope (down to the function root) before returning,
/// suppressing the drop of a moved-out return value when present.
fn emit_return_drops(gctx: &mut AstGenCtx, moved_var: Option<&str>) {
    match moved_var {
        Some(v) => emit_drops_through_scope_moved_out(gctx, 0, v),
        None => emit_drops_through_scope(gctx, 0),
    }
}

/// Emit an unconditional `Br` to `target`.
fn emit_br(gctx: &mut AstGenCtx, target: JirBlockRef) {
    emit(
        gctx,
        JirInst {
            tag: JirTag::Br,
            a: target,
            ..Default::default()
        },
    );
}

/// Emit a `CondBr` on `cond` to `then_b`/`else_b`. A constant `Bool` condition
/// folds to a plain `Br` to the live branch — this keeps the dead branch from
/// looking reachable, which matters for the function-end fall-through check on
/// `while (true)` patterns. The successor block refs are packed into the
/// function's extra array (`b` is the extra index).
fn emit_cond_br(gctx: &mut AstGenCtx, cond: JirRef, then_b: JirBlockRef, else_b: JirBlockRef) {
    let c = *gctx.jfn.get_inst(cond);
    if c.tag == JirTag::Bool {
        emit_br(gctx, if c.a != 0 { then_b } else { else_b });
        return;
    }
    let extra = gctx.jfn.push_extra(&[then_b, else_b]);
    emit(
        gctx,
        JirInst {
            tag: JirTag::CondBr,
            a: cond,
            b: extra,
            ..Default::default()
        },
    );
}

/// Lower an `if`/`else` statement. Builds then/else/merge blocks; `else` is
/// optional. After both arms, the merge block becomes the insertion point —
/// dead (zero predecessors) if both arms diverged, which the verifier tolerates.
///
/// Deferred: the `comp if` (flags bit 0) conditional-compilation path, per-arm
/// drop scopes, and the `runtimeCondDepth` bookkeeping (only observable through
/// comp-binding mutation rules, which error today).
fn astgen_if(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let cond_idx = NodeIdx::new(n.lhs);
    let extra = n.rhs;
    let then_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra));
    let else_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1));

    // `comp if`: fold the condition at compile time and lower ONLY the taken
    // branch inline (the dead branch is never analyzed — it may reference
    // symbols that don't exist on this target). Code after the `comp if` goes in
    // a fresh `compifend` block, matching the oracle.
    if n.flags & 1 != 0 {
        let taken = match gctx.ctx.fold_comptime_expr_in(cond_idx, &gctx.comp_scope) {
            ComptimeValue::Bool(b) => b,
            _ => return Err("astgen: comp-if condition is not a comptime bool".into()),
        };
        let (start, count) = if taken {
            (extra + 2, then_count)
        } else {
            (extra + 2 + then_count, else_count)
        };
        // The taken arm runs at the surrounding runtime depth but keeps its own
        // lexical scope, so its locals (and their drops) stay arm-local.
        push_drop_scope(gctx);
        for i in 0..count {
            let s = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(start + i)));
            astgen_expr(gctx, s, TypeIdx::NONE)?;
        }
        pop_drop_scope_emitting(gctx);
        // Only open a continuation block if the arm DIVERGED (return/break/
        // continue): statements after the comp-if would otherwise append after the
        // terminator. A non-terminating arm just continues in the current block —
        // no `compifend`, no branch (matches the C++ astgen.cpp:3382-3385).
        if block_has_terminator(gctx) {
            gctx.current_block = gctx.jfn.push_block("compifend");
        }
        return Ok(());
    }

    let cond_ref = astgen_expr(gctx, cond_idx, builtin::BOOL)?;
    let then_b = gctx.jfn.push_block("then");
    let else_b = if else_count > 0 {
        gctx.jfn.push_block("else")
    } else {
        NO_JIR_BLOCK
    };
    let merge_b = gctx.jfn.push_block("ifend");

    emit_cond_br(
        gctx,
        cond_ref,
        then_b,
        if else_count > 0 { else_b } else { merge_b },
    );

    // Then arm — its own drop scope so locals declared inside drop at branch
    // exit (popDropScopeEmitting drops them before the fall-through Br).
    // Arm bodies run at +1 runtime-conditional depth (comp bindings declared
    // outside may not be mutated inside).
    gctx.runtime_cond_depth += 1;
    gctx.current_block = then_b;
    push_drop_scope(gctx);
    for i in 0..then_count {
        let s = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 2 + i)));
        astgen_expr(gctx, s, TypeIdx::NONE)?;
    }
    pop_drop_scope_emitting(gctx);
    if !block_has_terminator(gctx) {
        emit_br(gctx, merge_b);
    }

    // Else arm (if present).
    if else_count > 0 {
        gctx.current_block = else_b;
        push_drop_scope(gctx);
        for i in 0..else_count {
            let s = NodeIdx::new(
                gctx.ctx
                    .node_store
                    .get_extra(ExtraIdx::new(extra + 2 + then_count + i)),
            );
            astgen_expr(gctx, s, TypeIdx::NONE)?;
        }
        pop_drop_scope_emitting(gctx);
        if !block_has_terminator(gctx) {
            emit_br(gctx, merge_b);
        }
    }
    gctx.runtime_cond_depth -= 1;

    gctx.current_block = merge_b;
    Ok(())
}

/// Lower a `while (cond) { body }` loop: cond/body/exit blocks, branch into
/// cond, conditional branch into body or exit, fall the body back to cond.
fn astgen_while(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let cond_idx = NodeIdx::new(n.lhs);
    let extra = n.rhs;
    let body_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra));

    let cond_b = gctx.jfn.push_block("loopcond");
    let body_b = gctx.jfn.push_block("loopbody");
    let exit_b = gctx.jfn.push_block("loopexit");

    emit_br(gctx, cond_b);
    gctx.current_block = cond_b;
    let cond_ref = astgen_expr(gctx, cond_idx, builtin::BOOL)?;
    emit_cond_br(gctx, cond_ref, body_b, exit_b);

    gctx.current_block = body_b;
    push_drop_scope(gctx);
    let body_scope_idx = gctx.drop_scopes.len() - 1;
    gctx.loop_stack.push(LoopFrame {
        continue_block: cond_b,
        break_block: exit_b,
        body_scope_idx,
    });
    gctx.runtime_cond_depth += 1;
    for i in 0..body_count {
        let s = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1 + i)));
        astgen_expr(gctx, s, TypeIdx::NONE)?;
    }
    gctx.runtime_cond_depth -= 1;
    pop_drop_scope_emitting(gctx);
    if !block_has_terminator(gctx) {
        emit_br(gctx, cond_b);
    }
    gctx.loop_stack.pop();

    gctx.current_block = exit_b;
    Ok(())
}

/// Lower `for x in start..end { body }`. Desugars to an explicit induction
/// variable looped while `x < end`, incremented in a dedicated step block (so
/// `continue` re-runs the step before re-testing). Extra layout `[varName,
/// start, end, bodyCount, body..]`. A bare-`Variable` end bound lends its
/// declared type as the start's expected hint so the induction var adopts the
/// bound's exact width and signedness; otherwise the narrower side of the
/// comparison is widened (sext/zext).
fn astgen_for(gctx: &mut AstGenCtx, n: &AstNode) -> Result<(), String> {
    let extra = n.lhs;
    let var_name_id = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra));
    let start_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 1)));
    let end_idx = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 2)));
    let body_count = gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 3));
    let var_name = str_at(gctx, var_name_id);

    let mut start_hint = TypeIdx::NONE;
    {
        let en = *gctx.ctx.node_store.get(end_idx);
        if en.tag == AstTag::Variable {
            let end_name = str_at(gctx, en.lhs);
            if let Some(&t) = gctx.local_types.get(&end_name) {
                start_hint = t;
            }
        }
    }
    let start_ref = astgen_expr(gctx, start_idx, start_hint)?;
    let idx_ty = gctx.jfn.get_inst(start_ref).ty;

    let slot = emit_alloca_hoisted(
        gctx,
        JirInst {
            tag: JirTag::Alloca,
            ty: idx_ty,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: start_ref,
            ..Default::default()
        },
    );
    gctx.locals.insert(var_name.clone(), slot);
    gctx.local_types.insert(var_name.clone(), idx_ty);

    let cond_b = gctx.jfn.push_block("forcond");
    let body_b = gctx.jfn.push_block("forbody");
    let step_b = gctx.jfn.push_block("forstep");
    let exit_b = gctx.jfn.push_block("forexit");
    emit_br(gctx, cond_b);

    gctx.current_block = cond_b;
    let mut load_idx = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: idx_ty,
            ..Default::default()
        },
    );
    let mut end_ref = astgen_expr(gctx, end_idx, idx_ty)?;
    let (ik_kind, ik_width, ik_signed) = {
        let ik = gctx.ctx.type_pool.get(idx_ty);
        (ik.kind, ik.a, ik.kind == TypeKind::Int && ik.b != 0)
    };
    let end_ty = gctx.jfn.get_inst(end_ref).ty;
    if end_ty != idx_ty {
        let (ek_kind, ek_width, ek_signed_bit) = {
            let ek = gctx.ctx.type_pool.get(end_ty);
            (ek.kind, ek.a, ek.b)
        };
        // Same width-class (signedness bit) but different width: widen the
        // narrower side. Otherwise the bounds are genuinely incompatible.
        let ik_signed_bit = if ik_signed { 1 } else { 0 };
        if ik_kind == TypeKind::Int && ek_kind == TypeKind::Int && ik_signed_bit == ek_signed_bit {
            let tag = if ik_signed {
                JirTag::SExt
            } else {
                JirTag::ZExt
            };
            if ik_width > ek_width {
                end_ref = emit(
                    gctx,
                    JirInst {
                        tag,
                        a: end_ref,
                        ty: idx_ty,
                        ..Default::default()
                    },
                );
            } else {
                load_idx = emit(
                    gctx,
                    JirInst {
                        tag,
                        a: load_idx,
                        ty: end_ty,
                        ..Default::default()
                    },
                );
            }
        } else {
            return Err("for-range bounds have mismatched types; cast one side \
                        with `as` so both bounds agree"
                .into());
        }
    }
    let cmp_tag = if ik_signed {
        JirTag::ICmpSlt
    } else {
        JirTag::ICmpUlt
    };
    let cmp_ref = emit(
        gctx,
        JirInst {
            tag: cmp_tag,
            a: load_idx,
            b: end_ref,
            ty: builtin::BOOL,
            ..Default::default()
        },
    );
    emit_cond_br(gctx, cmp_ref, body_b, exit_b);

    // Body — `continue` jumps to the step block, `break` to exit.
    gctx.current_block = body_b;
    push_drop_scope(gctx);
    let body_scope_idx = gctx.drop_scopes.len() - 1;
    gctx.loop_stack.push(LoopFrame {
        continue_block: step_b,
        break_block: exit_b,
        body_scope_idx,
    });
    gctx.runtime_cond_depth += 1;
    for i in 0..body_count {
        let s = NodeIdx::new(gctx.ctx.node_store.get_extra(ExtraIdx::new(extra + 4 + i)));
        astgen_expr(gctx, s, TypeIdx::NONE)?;
    }
    gctx.runtime_cond_depth -= 1;
    pop_drop_scope_emitting(gctx);
    if !block_has_terminator(gctx) {
        emit_br(gctx, step_b);
    }
    gctx.loop_stack.pop();

    // Step: x = x + 1; br cond.
    gctx.current_block = step_b;
    let cur = emit(
        gctx,
        JirInst {
            tag: JirTag::Load,
            a: slot,
            ty: idx_ty,
            ..Default::default()
        },
    );
    let one_ref = emit(
        gctx,
        JirInst {
            tag: JirTag::Int,
            a: 1,
            ty: idx_ty,
            ..Default::default()
        },
    );
    let next = emit(
        gctx,
        JirInst {
            tag: JirTag::Add,
            a: cur,
            b: one_ref,
            ty: idx_ty,
            ..Default::default()
        },
    );
    emit(
        gctx,
        JirInst {
            tag: JirTag::Store,
            a: slot,
            b: next,
            ..Default::default()
        },
    );
    emit_br(gctx, cond_b);

    gctx.current_block = exit_b;
    gctx.locals.remove(&var_name);
    gctx.local_types.remove(&var_name);
    Ok(())
}

/// Lower `break` — drop everything down through the loop body's scope, then
/// branch to the loop's exit block.
fn astgen_break(gctx: &mut AstGenCtx) -> Result<(), String> {
    let frame = gctx
        .loop_stack
        .last()
        .ok_or_else(|| "astgen: `break` outside of loop".to_string())?;
    let (target, body_scope_idx) = (frame.break_block, frame.body_scope_idx);
    emit_drops_through_scope(gctx, body_scope_idx);
    emit_br(gctx, target);
    Ok(())
}

/// Lower `continue` — drop through the loop body's scope, then branch to the
/// loop's continue (cond/step) block.
fn astgen_continue(gctx: &mut AstGenCtx) -> Result<(), String> {
    let frame = gctx
        .loop_stack
        .last()
        .ok_or_else(|| "astgen: `continue` outside of loop".to_string())?;
    let (target, body_scope_idx) = (frame.continue_block, frame.body_scope_idx);
    emit_drops_through_scope(gctx, body_scope_idx);
    emit_br(gctx, target);
    Ok(())
}

/// Count the predecessors of `target`: blocks (other than `target`) whose
/// terminator branches to it. Used by the function-end fall-through check — a
/// non-entry block with zero predecessors is unreachable, so the "non-void
/// function falls through" diagnostic must not fire for it.
fn predecessor_count(jfn: &JirFunction, target: JirBlockRef) -> usize {
    let mut count = 0;
    for b in 1..jfn.blocks.len() as JirBlockRef {
        if b == target {
            continue;
        }
        let Some(&last_ref) = jfn.blocks[b as usize].insts.last() else {
            continue;
        };
        let last = *jfn.get_inst(last_ref);
        match last.tag {
            JirTag::Br => {
                if last.a == target {
                    count += 1;
                }
            }
            JirTag::CondBr => {
                let ex = last.b;
                if (ex as usize) + 2 <= jfn.extra.len() {
                    if jfn.get_extra(ex) == target {
                        count += 1;
                    }
                    if jfn.get_extra(ex + 1) == target {
                        count += 1;
                    }
                }
            }
            JirTag::Switch => {
                let ex = last.b;
                if (ex as usize) + 2 > jfn.extra.len() {
                    continue;
                }
                if jfn.get_extra(ex) == target {
                    count += 1;
                }
                let case_count = jfn.get_extra(ex + 1);
                for i in 0..case_count {
                    let case_slot = ex + 2 + i * 4 + 3;
                    if (case_slot as usize) < jfn.extra.len() && jfn.get_extra(case_slot) == target
                    {
                        count += 1;
                    }
                }
            }
            _ => {}
        }
    }
    count
}

fn block_has_terminator(gctx: &AstGenCtx) -> bool {
    gctx.jfn
        .get_block(gctx.current_block)
        .insts
        .last()
        .map(|&r| gctx.jfn.get_inst(r).tag.is_terminator())
        .unwrap_or(false)
}

/// Build a signature-only `JirFunction` (name, return + param types/modes,
/// extern/test/pub bookkeeping) without walking the body.
pub fn astgen_metadata(fn_ast: &FunctionAST, ctx: &CodegenContext) -> JirFunction {
    let mut jfn = JirFunction::new();
    jfn.name = fn_ast.name.clone();
    jfn.module_path = fn_ast.module_path.clone();
    // Signature types qualify a bare imported type AND a generic-call callee/args
    // to their modules; a signature generic is instantiated eagerly so layout /
    // sret classification can resolve it (body/local-slot types stay bare).
    let rt = ctx.requalify_type(fn_ast.return_type, &fn_ast.module_path);
    let rt = ctx.qualify_generic_callee(rt, &fn_ast.module_path);
    let _ = ctx.resolve_generic_call_instantiate(rt);
    jfn.return_type = rt;
    jfn.is_extern = fn_ast.is_extern;
    jfn.is_export = fn_ast.is_export;
    jfn.is_pub = fn_ast.is_pub;
    jfn.is_test = fn_ast.is_test;
    jfn.is_var_args = fn_ast.is_var_args;
    for p in &fn_ast.args {
        let pt = ctx.requalify_type(p.ty, &fn_ast.module_path);
        let pt = ctx.qualify_generic_callee(pt, &fn_ast.module_path);
        let _ = ctx.resolve_generic_call_instantiate(pt);
        jfn.param_types.push(pt);
        jfn.param_modes.push(p.mode);
    }
    jfn
}

/// Append the body of `fn_ast` to a pre-populated metadata `JirFunction`.
pub fn astgen_body_into(
    jfn: &mut JirFunction,
    fn_ast: &FunctionAST,
    ctx: &CodegenContext,
) -> Result<(), String> {
    if fn_ast.is_extern {
        return Ok(());
    }
    let entry = jfn.push_block("entry");
    let mut gctx = AstGenCtx {
        jfn,
        ctx,
        current_block: entry,
        locals: HashMap::new(),
        local_types: HashMap::new(),
        current_node: NodeIdx::NONE,
        loop_stack: Vec::new(),
        drop_scopes: Vec::new(),
        local_scopes: Vec::new(),
        comp_scope: crate::comptime::ComptimeScope::new(),
        comp_bind_info: Vec::new(),
        runtime_cond_depth: 0,
        recovered: Vec::new(),
    };
    // Seed the root comp frame with any active comp-param substitutions, so a
    // body reference to a comp param (`k`) folds to the call-site constant (the
    // C++ `seedComptimeScope` tail). No-op for ordinary (non-comp) bodies.
    ctx.seed_comp_subst_into(&mut gctx.comp_scope);
    // The function-body drop scope (frame 0). Params register into it so the
    // reverse drop walk fires body locals first, params last.
    push_drop_scope(&mut gctx);

    // Lower each parameter. ByValue: alloca + store + register the alloca as
    // the local (reads Load it). ByPointer: register the param JirRef directly.
    for (i, p) in fn_ast.args.iter().enumerate() {
        // Requalify a bare imported-module type (`Counter`→`mod_x.Counter`) so
        // the ABI classifier + the Param inst's type match the registry.
        let bm = gctx.ctx.current_body_module();
        // requalify_type now qualifies a GenericCall callee+args itself, so the
        // Param/Alloca/Load instruction types match the oracle's qualified spelling.
        let mut pty = gctx.ctx.requalify_type(p.ty, &bm);
        // An `[expr]T` param resolves its comptime length for the Param inst +
        // ABI (the signature header keeps the unresolved ArrayExpr spelling).
        if gctx.ctx.type_pool.get(pty).kind == TypeKind::ArrayExpr {
            match gctx.ctx.resolve_array_expr_instantiate(pty) {
                Ok(t) => pty = t,
                Err(e) => return finish(&gctx, Some(e)),
            }
        }
        let pabi = match classify_param(p.mode, pty, gctx.ctx) {
            Ok(p) => p,
            Err(e) => return finish(&gctx, Some(e)),
        };
        let by_ptr = pabi.kind == ParamAbiKind::ByPointer;
        let param_ref = emit(
            &mut gctx,
            JirInst {
                tag: JirTag::Param,
                a: i as u32,
                ty: pty,
                flags: if by_ptr { 1 } else { 0 },
                ..Default::default()
            },
        );
        let slot_ref = if by_ptr {
            gctx.locals.insert(p.name.clone(), param_ref);
            param_ref
        } else {
            let alloca_ref = emit_alloca_hoisted(
                &mut gctx,
                JirInst {
                    tag: JirTag::Alloca,
                    ty: pty,
                    ..Default::default()
                },
            );
            emit(
                &mut gctx,
                JirInst {
                    tag: JirTag::Store,
                    a: alloca_ref,
                    b: param_ref,
                    ..Default::default()
                },
            );
            gctx.locals.insert(p.name.clone(), alloca_ref);
            alloca_ref
        };
        gctx.local_types.insert(p.name.clone(), pty);
        // A `move` parameter is callee-OWNED: it drops at function exit unless
        // the body moves it onward (which consumes the track). `let`/`mut` stay
        // caller-owned and never drop here.
        if p.mode == jam_core::param_mode::ParamMode::Move && gctx.ctx.type_needs_drop(pty) {
            gctx.drop_scopes.last_mut().unwrap().push(DropTrack {
                var_name: p.name.clone(),
                slot: slot_ref,
                ty: pty,
            });
        }
    }

    // Walk the body statements (statements dispatch through astgen_expr too).
    // A hard error (`failHere`/`failNode` analogue) propagates out, but any
    // recoverable diagnostics already collected must surface first — the C++
    // pushes both to the shared Diagnostics, so the hard one trails the
    // recovered ones in source order. `finish` folds the two streams together.
    fn finish(gctx: &AstGenCtx, hard: Option<String>) -> Result<(), String> {
        let mut all: Vec<String> = gctx.recovered.clone();
        if let Some(h) = hard {
            all.push(prefix_hard_error(gctx, h));
        }
        if all.is_empty() {
            Ok(())
        } else {
            Err(all.join("\n"))
        }
    }
    let body = fn_ast.body.clone();
    for stmt in body {
        if let Err(e) = astgen_expr(&mut gctx, stmt, TypeIdx::NONE) {
            return finish(&gctx, Some(e));
        }
    }

    // Implicit fall-through terminator. Three cases for a tail block without an
    // explicit terminator:
    //   1. Unreachable (zero predecessors and not the entry block): a dead
    //      post-merge / post-loop block left behind because every arm /
    //      iteration diverged — give it an `Unreachable` terminator so the JIR
    //      is well-formed, without spuriously erroring.
    //   2. Reachable and the function returns a value (or is `noreturn`): a
    //      real bug — a path reaches the end without returning.
    //   3. Reachable and the function returns void: drop every active scope,
    //      then emit `Ret`.
    if !block_has_terminator(&gctx) {
        let is_entry = gctx.current_block == 1;
        let reachable = is_entry || predecessor_count(gctx.jfn, gctx.current_block) > 0;
        if !reachable {
            emit(
                &mut gctx,
                JirInst {
                    tag: JirTag::Unreachable,
                    ..Default::default()
                },
            );
        } else if fn_ast.return_type == builtin::NORETURN {
            // The C++ `failHere`s (anchored at the last-entered node), so the
            // diagnostic carries the `file:line: error:` prefix.
            let msg = format!(
                "fn `{}` is declared `noreturn` but its body falls through without diverging",
                fn_ast.name
            );
            let hard = fail_node(&gctx, gctx.current_node, &msg);
            return finish(&gctx, Some(hard));
        } else if !fn_ast.return_type.is_none() {
            // C++ body: "has non-void return type" (no "a"), `failHere`-anchored.
            let msg = format!(
                "fn `{}` has non-void return type but a path reaches the \
                 function end without returning a value",
                fn_ast.name
            );
            let hard = fail_node(&gctx, gctx.current_node, &msg);
            return finish(&gctx, Some(hard));
        } else {
            emit_drops_through_scope(&mut gctx, 0);
            emit(
                &mut gctx,
                JirInst {
                    tag: JirTag::Ret,
                    ..Default::default()
                },
            );
        }
    }
    finish(&gctx, None)
}

/// Lower a function from scratch: metadata + body.
pub fn astgen_function(
    fn_ast: &FunctionAST,
    ctx: &mut CodegenContext,
) -> Result<JirFunction, String> {
    let mut jfn = astgen_metadata(fn_ast, ctx);
    if fn_ast.is_extern {
        return Ok(jfn);
    }
    astgen_body_into(&mut jfn, fn_ast, ctx)?;
    Ok(jfn)
}

/// Pass-2 of generic instantiation (wired into `CodegenContext` via
/// `set_method_instantiator`): lower each instantiated method's body under the
/// active substitution so call-site method names + recursive sub-instantiations
/// intern in the oracle's order. Bodies are NOT dumped — they run purely for the
/// string-pool side effects. A conditional method (everything but `drop`) whose
/// body fails to compile for these type args is WITHDRAWN (the C++ `withdraw`);
/// strings its partial lowering already interned survive, matching the oracle.
pub fn instantiate_methods(
    ctx: &CodegenContext,
    clones: &[FunctionAST],
    body_subst: &std::collections::HashMap<String, TypeIdx>,
) -> Result<(), String> {
    ctx.push_subst(body_subst.clone());
    // Pass-1: JIR metadata + LLVM prototype for every method (the C++ two-pass —
    // all prototypes declared before any body so cross-method LLVM calls in
    // Pass-2 resolve). jir_declare_prototype does not touch the string pool, so
    // the --emit-jir dump is unaffected.
    let mut built: Vec<(JirFunction, usize)> = Vec::with_capacity(clones.len());
    for (i, clone) in clones.iter().enumerate() {
        if clone.is_extern {
            continue;
        }
        let jfn = astgen_metadata(clone, ctx);
        let _ = jir_declare_prototype(&jfn, ctx);
        built.push((jfn, i));
    }
    // Pass-2: JIR body + LLVM body. A non-`drop` method whose body fails to lower
    // is the C++ conditional WITHDRAWAL; until clone-synthesis + nested-generic
    // intern order are byte-exact, DEFER the whole instantiation on the first
    // such method rather than ship a divergence. The instantiated LLVM body
    // (jir_define_body) is what makes `--emit-ir` carry the method definitions;
    // it's a no-op on the --emit-jir text dump.
    // Withdraw-and-continue (the faithful C++ behavior): a non-`drop` method whose
    // body fails to lower is a CONDITIONAL WITHDRAWAL — drop it from the registry
    // and keep going; the rest of the instantiation still succeeds.
    for (jfn, i) in &mut built {
        let clone = &clones[*i];
        let is_drop = clone.name.ends_with(".drop");
        // The recorded reason is the bare diagnostic message (the C++ takes
        // `diagnostics()[diagMark].message`), so strip the `file:line: error: `
        // anchor a propagated astgen error carries.
        let bare_reason = |e: &str| -> String {
            match e.split_once(": error: ") {
                Some((_, msg)) => msg.to_string(),
                None => e.to_string(),
            }
        };
        // (1) Body that fails to lower for these type args is a CONDITIONAL
        // WITHDRAWAL (the C++ `withdraw`): drop it from the registry, record
        // the reason for call-site replay, skip its body, keep going.
        if let Err(e) = astgen_body_into(jfn, clone, ctx)
            && !is_drop
        {
            ctx.unregister_function_ast(&clone.name);
            ctx.record_withdrawn_method(&clone.name, bare_reason(&e));
            continue;
        }
        // (2) Move/ownership WITHDRAWAL: even when the body lowers, a non-`drop`
        // method whose move-analysis reports a diagnostic under these
        // (substituted) type args is withdrawn to a bare `declare` — exactly the
        // oracle's `init_analysis` -> `withdraw` for e.g. `Vec(T).filled`, which
        // moves a by-value drop-bearing parameter in a loop. The subst pushed
        // above is active, so a body-local `Self`/`Vec(T)` type resolves to its
        // drop-bearing monomorph here.
        if !is_drop {
            let adiags = crate::init_analysis::analyze(clone, ctx);
            if let Some(first) = adiags.first() {
                ctx.unregister_function_ast(&clone.name);
                ctx.record_withdrawn_method(&clone.name, first.message.clone());
                continue;
            }
        }
        // (3) Structural JIR verification (the C++ verifyJirFunction at
        // instantiation, codegen.cpp:1846-1864): a malformed instantiated body
        // (bad dispatch, missing terminator, OOB ref, cross-block use-before-def)
        // is withdrawn for non-`drop` methods, the same as a move-analysis
        // failure. The resolver resolves the clone's GenericCall / ArrayExpr type
        // refs against the active substitution.
        let resolver = |t: TypeIdx| -> TypeIdx {
            let k = ctx.type_pool.get(t);
            if k.kind == TypeKind::GenericCall {
                let r = ctx.resolve_generic_call(t);
                if !r.is_none() {
                    return r;
                }
            }
            if k.kind == TypeKind::ArrayExpr {
                return ctx.resolve_array_expr(t);
            }
            t
        };
        let vdiags = crate::jir_verify::verify_jir_function(
            jfn,
            Some(&ctx.type_pool),
            Some(&ctx.string_pool),
            Some(&resolver),
        );
        if let Some(first) = vdiags.first()
            && !is_drop
        {
            ctx.unregister_function_ast(&clone.name);
            ctx.record_withdrawn_method(&clone.name, first.message.clone());
            continue;
        }
        let _ = jir_define_body(jfn, ctx);
    }
    ctx.pop_subst();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jir_codegen::{jir_declare_prototype, jir_define_body};
    use crate::jir_verify::verify_jir_function;
    use jam_llvm::Context;
    use jam_syntax::lexer::Lexer;
    use jam_syntax::parser::Parser;

    /// Parse `src`, astgen its first function, run jir_verify, then lower it to
    /// LLVM IR through jir_codegen and confirm the function verifies — the full
    /// source → tokens → AST → JIR → LLVM-IR pipeline end to end.
    fn lower_first_fn(src: &str) {
        let owner = Context::new();
        let mut cg = CodegenContext::new(&owner, "m");

        // Parse into the context's pools so the function's body NodeIdxs index
        // the same NodeStore astgen reads.
        let module = {
            let mut lexer = Lexer::new(src.as_bytes().to_vec());
            lexer.scan_tokens().expect("lex");
            let tokens = lexer.tokens().to_vec();
            let mut diags = jam_core::diag::Diagnostics::new();
            let mut parser = Parser::new(
                tokens,
                src.as_bytes().to_vec(),
                &mut cg.type_pool,
                &mut cg.string_pool,
                &mut cg.node_store,
                &mut diags,
                "test.jam",
            );
            let m = parser.parse().expect("parse");
            assert!(!diags.has_errors(), "parse errors");
            m
        };
        assert!(!module.functions.is_empty(), "no functions parsed");

        // astgen the first function, then verify + lower.
        let jfn = {
            let func = &module.functions[0];
            astgen_function(func, &mut cg).expect("astgen")
        };
        let diags = verify_jir_function(&jfn, Some(&cg.type_pool), Some(&cg.string_pool), None);
        assert!(diags.is_empty(), "jir_verify: {diags:?}");

        jir_declare_prototype(&jfn, &cg).expect("prototype");
        jir_define_body(&jfn, &cg).expect("define body");
        let f = cg.module().get_function(&jfn.name).unwrap();
        assert!(f.verify(), "LLVM verify failed for `{}`", jfn.name);
    }

    #[test]
    fn end_to_end_constant_arithmetic() {
        lower_first_fn("fn five() i32 { return 2 + 3; }");
    }

    #[test]
    fn end_to_end_params_and_binop() {
        lower_first_fn("fn add(a: i32, b: i32) i32 { return a + b; }");
    }

    #[test]
    fn end_to_end_comparison_returns_bool() {
        lower_first_fn("fn lt(a: i32, b: i32) bool { return a < b; }");
    }

    #[test]
    fn end_to_end_var_decl_declared() {
        lower_first_fn("fn f() i32 { var x: i32 = 5; return x + 1; }");
    }

    #[test]
    fn end_to_end_var_decl_inferred() {
        lower_first_fn("fn g(a: i32) i32 { var y = a + 3; return y; }");
    }

    #[test]
    fn end_to_end_assign_to_local() {
        lower_first_fn("fn h() i32 { var x: i32 = 1; x = x + 2; return x; }");
    }

    #[test]
    fn end_to_end_assign_to_mut_param() {
        lower_first_fn("fn k(a: mut i32) i32 { a = a + 5; return a; }");
    }

    #[test]
    fn end_to_end_if_else() {
        lower_first_fn("fn mx(a: i32, b: i32) i32 { if (a > b) { return a; } else { return b; } }");
    }

    #[test]
    fn end_to_end_if_no_else_falls_through() {
        lower_first_fn("fn clampLow(a: mut i32) i32 { if (a < 0) { a = 0; } return a; }");
    }

    #[test]
    fn end_to_end_while_loop() {
        lower_first_fn(
            "fn sumTo(n: i32) i32 { var i: i32 = 0; var s: i32 = 0; \
             while (i < n) { s = s + i; i = i + 1; } return s; }",
        );
    }

    #[test]
    fn end_to_end_while_with_break_continue() {
        lower_first_fn(
            "fn f(n: i32) i32 { var i: i32 = 0; \
             while (i < n) { i = i + 1; if (i > 5) { break; } continue; } return i; }",
        );
    }

    #[test]
    fn end_to_end_for_range() {
        lower_first_fn(
            "fn sumRange(n: i32) i32 { var s: i32 = 0; for i in 0:n { s = s + i; } return s; }",
        );
    }

    #[test]
    fn end_to_end_for_with_break() {
        lower_first_fn(
            "fn firstHit(n: i32) i32 { var r: i32 = 0; \
             for i in 0:n { if (i > 3) { r = i; break; } } return r; }",
        );
    }

    #[test]
    fn end_to_end_unary_neg() {
        lower_first_fn("fn neg(a: i32) i32 { return -a; }");
    }

    #[test]
    fn end_to_end_unary_not() {
        lower_first_fn("fn no(a: bool) bool { return !a; }");
    }

    #[test]
    fn end_to_end_unary_bitnot() {
        lower_first_fn("fn flip(a: u32) u32 { return ~a; }");
    }

    #[test]
    fn end_to_end_cast_int_widen() {
        lower_first_fn("fn widen(a: i32) i64 { return a as i64; }");
    }

    #[test]
    fn end_to_end_cast_int_to_float() {
        lower_first_fn("fn toF(a: i32) f64 { return a as f64; }");
    }

    #[test]
    fn end_to_end_cast_float_trunc() {
        lower_first_fn("fn narrow(a: f64) f32 { return a as f32; }");
    }

    /// Parse `src` and astgen its first function (no LLVM lowering) — for
    /// asserting astgen-level diagnostics.
    fn astgen_first_fn(src: &str) -> Result<(), String> {
        let owner = Context::new();
        let mut cg = CodegenContext::new(&owner, "m");
        let module = {
            let mut lexer = Lexer::new(src.as_bytes().to_vec());
            lexer.scan_tokens().expect("lex");
            let tokens = lexer.tokens().to_vec();
            let mut diags = jam_core::diag::Diagnostics::new();
            let mut parser = Parser::new(
                tokens,
                src.as_bytes().to_vec(),
                &mut cg.type_pool,
                &mut cg.string_pool,
                &mut cg.node_store,
                &mut diags,
                "test.jam",
            );
            let m = parser.parse().expect("parse");
            assert!(!diags.has_errors(), "parse errors");
            m
        };
        astgen_function(&module.functions[0], &mut cg).map(|_| ())
    }

    /// Returned-value vs declared-return-type reconciliation (deliberate
    /// divergence from the C++ oracle, which has no check and emits
    /// LLVM-invalid `ret i8` from an i32 function for the inferred-literal
    /// case): lossless narrower ints widen, lossy mismatches error.
    #[test]
    fn return_type_mismatch_widens_or_rejects() {
        // `var x = 1` infers u8 (smallest fit); returning it at i32 now
        // ZExt-widens and produces verifiable IR (the original bug shape).
        lower_first_fn("fn f() i32 { var x = 1; return x; }");
        // Signed widens into wider signed.
        lower_first_fn("fn f2() i64 { var x: i32 = 1; return x; }");
        // Declared at the return width / hinted literal / explicit cast: fine.
        lower_first_fn("fn f3() i32 { var x: i32 = 1; return x; }");
        lower_first_fn("fn g() u8 { return 200; }");
        lower_first_fn("fn h() i64 { var x: i32 = 1; return x as i64; }");

        // Narrowing is lossy: hard error.
        let err = astgen_first_fn("fn n() u8 { var x: i32 = 300; return x; }").unwrap_err();
        assert!(
            err.contains("return value type does not match the declared return type of `n`"),
            "unexpected error: {err}"
        );
        // Signed into wider UNSIGNED is lossy (negatives): hard error.
        let err = astgen_first_fn("fn s() u32 { var x: i8 = 1; return x; }").unwrap_err();
        assert!(
            err.contains("return value type does not match the declared return type of `s`"),
            "unexpected error: {err}"
        );
    }

    /// Same-scope redeclaration is rejected; inner-block shadowing, sibling
    /// blocks reusing a name, and param shadowing stay legal (the C++
    /// `localScopes` check, astgen.cpp:1149).
    #[test]
    fn same_scope_redeclaration_rejected() {
        let err = astgen_first_fn("fn f() { const a = 1; var a = 2; }").unwrap_err();
        assert!(
            err.contains("redeclaration of `a` in the same scope"),
            "unexpected error: {err}"
        );

        // A comp binding occupies the scope's name too.
        let err = astgen_first_fn("fn f() { comp const N = 1; var N = 2; }").unwrap_err();
        assert!(
            err.contains("redeclaration of `N` in the same scope"),
            "unexpected error: {err}"
        );

        // Inner-block shadowing is intentional and allowed.
        lower_first_fn(
            "fn f(c: bool) i32 { var x: i32 = 1; if (c) { var x: i32 = 2; x = x + 1; } return x; }",
        );

        // Sibling blocks at the same level each get a fresh frame.
        lower_first_fn(
            "fn g(c: bool) i32 { if (c) { const op: i32 = 1; } if (c) { const op: i32 = 2; } return 0; }",
        );

        // Params are not in the decl frame — shadowing one is allowed.
        lower_first_fn("fn h(a: i32) i32 { var a: i32 = 2; return a; }");
    }

    /// The comp-binding assignment rules (the C++ astgenAssign comp path,
    /// astgen.cpp:1510-1565): const-ness, runtime-conditional depth, kind
    /// stability, and int width/signedness fit.
    #[test]
    fn comp_binding_assignment_rules() {
        // Legal reassignment at the declaration depth lowers end to end.
        lower_first_fn("fn f() i32 { comp var N = 1; N = N + 2; return N; }");

        let err = astgen_first_fn("fn f() { comp const N = 1; N = 2; }").unwrap_err();
        assert!(
            err.contains("cannot assign to comp const `N`"),
            "unexpected error: {err}"
        );

        let err = astgen_first_fn("fn f(c: bool) { comp var N = 1; if (c) { N = 2; } }")
            .unwrap_err();
        assert!(
            err.contains(
                "cannot assign to comp binding `N` from inside runtime conditional \
                 control flow — a comp value cannot depend on a runtime branch"
            ),
            "unexpected error: {err}"
        );

        let err = astgen_first_fn("fn f() { comp var N = 1; N = \"s\"; }").unwrap_err();
        assert!(
            err.contains("comp assignment changes the kind of `N` (e.g. int -> str)"),
            "unexpected error: {err}"
        );

        let err = astgen_first_fn("fn f() { comp var N: u8 = 1; N = 300; }").unwrap_err();
        assert!(
            err.contains("comp assignment value 300 does not fit `N` (u8)"),
            "unexpected error: {err}"
        );
    }

    /// `materialize_comptime_value` chases the expected type through alias /
    /// generic links (the C++ resolveScalarExpected at astgen.cpp:794): a comp
    /// value flowing into an alias-typed slot gets the alias target's width and
    /// fit-check, not a pass-through at its natural width.
    #[test]
    fn comp_value_into_alias_typed_slot_resolves_and_fit_checks() {
        let astgen_with_byte_alias = |src: &str| {
            let owner = Context::new();
            let mut cg = CodegenContext::new(&owner, "m");
            let module = {
                let mut lexer = Lexer::new(src.as_bytes().to_vec());
                lexer.scan_tokens().expect("lex");
                let tokens = lexer.tokens().to_vec();
                let mut diags = jam_core::diag::Diagnostics::new();
                let mut parser = Parser::new(
                    tokens,
                    src.as_bytes().to_vec(),
                    &mut cg.type_pool,
                    &mut cg.string_pool,
                    &mut cg.node_store,
                    &mut diags,
                    "test.jam",
                );
                let m = parser.parse().expect("parse");
                assert!(!diags.has_errors(), "parse errors");
                m
            };
            let byte = cg.type_pool.intern_int(8, false);
            cg.register_type_alias("Byte", byte);
            astgen_function(&module.functions[0], &mut cg).map(|_| ())
        };

        // 5 fits u8: materializes at the alias target's width, so the
        // declared-type check (which resolves both sides) passes.
        astgen_with_byte_alias("fn f() { comp const N = 5; var x: Byte = N; }")
            .expect("fitting comp value through alias");

        // 300 does not fit u8: the fit-check must fire against the resolved
        // alias target, not silently materialize at the natural width.
        let err = astgen_with_byte_alias("fn g() { comp const N = 300; var x: Byte = N; }")
            .expect_err("misfit comp value through alias");
        assert!(
            err.contains("does not fit the expected u8"),
            "unexpected error: {err}"
        );
    }

    /// Parse a multi-function module, register every function (so calls
    /// resolve), then astgen + verify + declare every prototype, and finally
    /// define + LLVM-verify every body (prototypes must all exist before any
    /// body so cross-function `Call`s find their callee).
    fn lower_all_fns(src: &str) {
        let owner = Context::new();
        let mut cg = CodegenContext::new(&owner, "m");
        let module = {
            let mut lexer = Lexer::new(src.as_bytes().to_vec());
            lexer.scan_tokens().expect("lex");
            let tokens = lexer.tokens().to_vec();
            let mut diags = jam_core::diag::Diagnostics::new();
            let mut parser = Parser::new(
                tokens,
                src.as_bytes().to_vec(),
                &mut cg.type_pool,
                &mut cg.string_pool,
                &mut cg.node_store,
                &mut diags,
                "test.jam",
            );
            let m = parser.parse().expect("parse");
            assert!(!diags.has_errors(), "parse errors");
            m
        };
        for f in &module.functions {
            cg.register_function_ast(f.name.clone(), f.clone());
        }
        let mut jfns = Vec::new();
        for func in &module.functions {
            let mut jfn = astgen_function(func, &mut cg).expect("astgen");
            jfn.name = mangled_function_name(func, &cg.type_pool, &cg.string_pool);
            let diags = verify_jir_function(&jfn, Some(&cg.type_pool), Some(&cg.string_pool), None);
            assert!(diags.is_empty(), "jir_verify `{}`: {diags:?}", jfn.name);
            jir_declare_prototype(&jfn, &cg).expect("prototype");
            jfns.push(jfn);
        }
        for jfn in &jfns {
            jir_define_body(jfn, &cg).expect("define body");
            let f = cg.module().get_function(&jfn.name).unwrap();
            assert!(f.verify(), "LLVM verify failed for `{}`", jfn.name);
        }
    }

    #[test]
    fn end_to_end_direct_call() {
        lower_all_fns(
            "fn add(a: i32, b: i32) i32 { return a + b; } \
             fn double(x: i32) i32 { return add(x, x); }",
        );
    }

    #[test]
    fn end_to_end_nested_calls() {
        lower_all_fns(
            "fn add(a: i32, b: i32) i32 { return a + b; } \
             fn chain(a: i32, b: i32, c: i32) i32 { return add(add(a, b), c); }",
        );
    }

    #[test]
    fn end_to_end_call_with_mut_ptr_arg() {
        lower_all_fns(
            "fn add(a: i32, b: i32) i32 { return a + b; } \
             fn bump(n: mut i32) { n = add(n, 1); }",
        );
    }

    #[test]
    fn end_to_end_short_circuit_and() {
        lower_first_fn("fn band(a: bool, b: bool) bool { return a && b; }");
    }

    #[test]
    fn end_to_end_short_circuit_or_mixed() {
        lower_first_fn("fn either(x: i32) bool { return x < 0 || x > 100; }");
    }

    #[test]
    fn end_to_end_string_literal_ptr_decay() {
        lower_first_fn("fn greeting() *const u8 { return \"hi\"; }");
    }

    #[test]
    fn end_to_end_array_literal_return() {
        lower_first_fn("fn mk() [3]i32 { return [10, 20, 30]; }");
    }

    #[test]
    fn end_to_end_array_index() {
        lower_first_fn("fn idx(a: [4]i32, i: u64) i32 { return a[i]; }");
    }

    #[test]
    fn end_to_end_array_repeat_memset() {
        lower_first_fn("fn zeros() [8]u8 { return [0; 8]; }");
    }

    #[test]
    fn end_to_end_array_local_and_assign() {
        lower_first_fn("fn f() i32 { var a: [3]i32 = [1, 2, 3]; a[0] = 9; return a[0] + a[2]; }");
    }

    #[test]
    fn end_to_end_match_int_switch() {
        lower_first_fn(
            "fn classify(x: i32) u8 { match (x) { 0 { return 1; } 1 { return 2; } _ { return 0; } } }",
        );
    }

    #[test]
    fn end_to_end_match_range_and_or() {
        lower_first_fn(
            "fn band(x: i32) u8 { match (x) { 0 | 1 { return 9; } 2..=10 { return 5; } _ { return 0; } } }",
        );
    }

    /// A `move` parameter of a drop-bearing type drops at function exit — the
    /// body must contain a `DropBinding` to its registered `cfn drop` symbol.
    #[test]
    fn drop_binding_emitted_for_move_param() {
        let owner = Context::new();
        let mut cg = CodegenContext::new(&owner, "m");
        let src = "const T = struct { x: u64 }; fn consume(t: move T) { }";
        let module = {
            let mut lexer = Lexer::new(src.as_bytes().to_vec());
            lexer.scan_tokens().expect("lex");
            let tokens = lexer.tokens().to_vec();
            let mut diags = jam_core::diag::Diagnostics::new();
            let mut parser = Parser::new(
                tokens,
                src.as_bytes().to_vec(),
                &mut cg.type_pool,
                &mut cg.string_pool,
                &mut cg.node_store,
                &mut diags,
                "test.jam",
            );
            let m = parser.parse().expect("parse");
            assert!(!diags.has_errors(), "parse errors");
            m
        };
        // Register T (named LLVM struct + body) and a (pretend) `cfn drop`.
        let st = &module.structs[0];
        let named = cg.context().named_struct(&st.name);
        cg.register_struct(st.name.clone(), named, st.fields.clone());
        named.set_body(&[cg.context().i64_type()], false);
        cg.register_drop_fn(st.name.clone(), "T.drop");

        let consume = &module.functions[0];
        let jfn = astgen_function(consume, &mut cg).expect("astgen");
        let drops: Vec<&JirInst> = jfn
            .insts
            .iter()
            .filter(|i| i.tag == JirTag::DropBinding)
            .collect();
        assert_eq!(
            drops.len(),
            1,
            "expected exactly one DropBinding for the move param"
        );
        // Its `b` operand is the StringIdx of the mangled drop symbol "T.drop".
        let sym_bytes = cg.string_pool.get(StringIdx::new(drops[0].b));
        let sym = String::from_utf8_lossy(&sym_bytes);
        assert_eq!(sym, "T.drop");
    }

    /// `@emitPutByte` in a cfn replays as `dprintf(fd, "%c", byte)` at the call
    /// site (the C++ handlePutByte, astgen.cpp:6276).
    #[test]
    fn cfn_emit_put_byte_replays_dprintf() {
        let owner = Context::new();
        let mut cg = CodegenContext::new(&owner, "m");
        let src = "cfn putc(fd: i32, b: u8) { @emitPutByte(fd, b); }\n\
                   fn f() { putc(1, 65); }";
        let module = {
            let mut lexer = Lexer::new(src.as_bytes().to_vec());
            lexer.scan_tokens().expect("lex");
            let tokens = lexer.tokens().to_vec();
            let mut diags = jam_core::diag::Diagnostics::new();
            let mut parser = Parser::new(
                tokens,
                src.as_bytes().to_vec(),
                &mut cg.type_pool,
                &mut cg.string_pool,
                &mut cg.node_store,
                &mut diags,
                "test.jam",
            );
            let m = parser.parse().expect("parse");
            assert!(!diags.has_errors(), "parse errors");
            m
        };
        for func in &module.functions {
            cg.register_function_ast(func.name.clone(), func.clone());
        }
        let f = module.functions.iter().find(|f| f.name == "f").unwrap();
        let jfn = astgen_function(f, &mut cg).expect("astgen");

        // The replay lands: Int 65 at i32 (`%c` expects an int), the "%c"
        // format literal, and a Call to the lazily-declared dprintf.
        let dp = cg.string_pool.intern(b"dprintf");
        assert!(
            jfn.insts
                .iter()
                .any(|i| i.tag == JirTag::Call && i.a == dp.raw()),
            "expected a dprintf call"
        );
        let pc = cg.string_pool.intern(b"%c");
        assert!(
            jfn.insts
                .iter()
                .any(|i| i.tag == JirTag::Str && i.a == pc.raw()),
            "expected the \"%c\" format literal"
        );
        assert!(
            jfn.insts
                .iter()
                .any(|i| i.tag == JirTag::Int && i.a == 65 && i.ty == builtin::I32),
            "expected the byte literal widened to i32"
        );

        // And the body lowers through JIR verify + LLVM.
        let diags = verify_jir_function(&jfn, Some(&cg.type_pool), Some(&cg.string_pool), None);
        assert!(diags.is_empty(), "jir_verify: {diags:?}");
        jir_declare_prototype(&jfn, &cg).expect("prototype");
        jir_define_body(&jfn, &cg).expect("define body");
        let lf = cg.module().get_function(&jfn.name).unwrap();
        assert!(lf.verify(), "LLVM verify failed");
    }

    /// A conditional method withdrawn for an instantiation replays the recorded
    /// reason at the call site (the C++ `reportMethodMiss`), both for a dotted
    /// method call and for the `v[i]` -> `at` index sugar.
    #[test]
    fn withdrawn_method_replays_reason_at_call_site() {
        let run = |src: &str, withdrawn: &str| -> Result<(), String> {
            let owner = Context::new();
            let mut cg = CodegenContext::new(&owner, "m");
            let module = {
                let mut lexer = Lexer::new(src.as_bytes().to_vec());
                lexer.scan_tokens().expect("lex");
                let tokens = lexer.tokens().to_vec();
                let mut diags = jam_core::diag::Diagnostics::new();
                let mut parser = Parser::new(
                    tokens,
                    src.as_bytes().to_vec(),
                    &mut cg.type_pool,
                    &mut cg.string_pool,
                    &mut cg.node_store,
                    &mut diags,
                    "test.jam",
                );
                let m = parser.parse().expect("parse");
                assert!(!diags.has_errors(), "parse errors");
                m
            };
            let st = &module.structs[0];
            let named = cg.context().named_struct(&st.name);
            cg.register_struct(st.name.clone(), named, st.fields.clone());
            named.set_body(&[cg.context().i32_type()], false);
            cg.record_withdrawn_method(withdrawn, "does not compile for these type arguments");
            astgen_function(&module.functions[0], &mut cg).map(|_| ())
        };

        // Dotted method call on a local instance.
        let err = run(
            "const T = struct { x: i32 }; fn f(t: T) { t.foo(); }",
            "T.foo",
        )
        .unwrap_err();
        assert!(
            err.contains(
                "method `T.foo` is not available for this instantiation — \
                 does not compile for these type arguments"
            ),
            "unexpected error: {err}"
        );

        // `v[i]` sugar whose `at` was withdrawn.
        let err = run(
            "const T = struct { x: i32 }; fn f(v: T) i32 { return v[0]; }",
            "T.at",
        )
        .unwrap_err();
        assert!(
            err.contains("method `T.at` is not available for this instantiation"),
            "unexpected error: {err}"
        );
    }
}
