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
            return Err(format!(
                "astgen: cannot take address of generic fn `{name}`"
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
    Err(format!("unknown variable `{name}`"))
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
                        return Err(format!(
                            "{what} has value {}, which does not fit the expected {}{}",
                            comp_int_to_string(*bits, *is_signed),
                            if es { "i" } else { "u" },
                            ew
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
        _ => Err("astgen: comptime value kind not yet materializable".into()),
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
            let idx = fields
                .iter()
                .position(|(nm, _)| *nm == member)
                .ok_or_else(|| format!("astgen: unknown field `{member}`"))?;
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

