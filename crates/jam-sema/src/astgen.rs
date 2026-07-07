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

