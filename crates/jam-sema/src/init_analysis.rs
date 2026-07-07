/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Definite-initialization / move-ownership analysis — ported from
//! `src/init_analysis.{h,cpp}`.
//!
//! A tree-walking dataflow over a function body produces, at each program
//! point, a map from binding name to a 2-bit lattice state {Unknown, Init,
//! Uninit, MaybeInit}. The lattice merges by bitwise OR, exactly as Clang's
//! `UninitializedValues` pass does. Jam programs are structured (no goto), so a
//! recursive analyzer over the AST suffices — there is no separate CFG pass.
//!
//! ## Scope of this port
//!
//! Unlike the C++ standalone analyzer (which threaded raw callback function
//! pointers through an `AnalysisHooks` struct), this port lives *inside*
//! `jam_sema` and borrows the [`CodegenContext`] directly for the questions it
//! cannot answer from its own tables: `type_needs_drop`, `requalify_type`, and
//! the enum-variant-constructor registry. The pools (`NodeStore`, `StringPool`,
//! `TypePool`) are read off the context too.
//!
//! ## Why it exists
//!
//! The single consumer is generic method-instantiation withdrawal
//! ([`crate::astgen::instantiate_methods`]): when a generic method's body is
//! lowered under drop-bearing type args, the oracle runs this analysis and, if
//! it reports any move/ownership diagnostic, WITHDRAWS the method to a bare
//! `declare` (no body). The canonical case is `Vec(T).filled`, which moves a
//! by-value (`let`-mode) drop-bearing parameter into a slot inside a `while`
//! loop — a use-of-moved-binding on loop re-entry, and a move out of a borrowed
//! parameter besides.
//!
//! This port only needs to reproduce the oracle's *accept/reject* verdict, so
//! diagnostic message strings are intentionally terse — what matters is whether
//! [`analyze`] returns a non-empty diagnostics list.

use std::collections::{HashMap, HashSet};

use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};
use jam_core::param_mode::ParamMode;
use jam_syntax::ast::FunctionAST;
use jam_syntax::ast_flat::{AstTag, TypeKind};

use crate::codegen_context::CodegenContext;

/// Per-binding initialization state, a 2-bit lattice. Merge is bitwise OR.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
enum InitState {
    Unknown = 0b00,
    Init = 0b01,
    Uninit = 0b10,
    MaybeInit = 0b11,
}

fn merge_state(a: InitState, b: InitState) -> InitState {
    match (a as u8) | (b as u8) {
        0b00 => InitState::Unknown,
        0b01 => InitState::Init,
        0b10 => InitState::Uninit,
        _ => InitState::MaybeInit,
    }
}

/// One reportable move/ownership problem found during analysis.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub message: String,
    pub var_name: String,
    /// 1-based source line of the anchor node (`0` when none). Parse-time
    /// stamped, so it is correct for imported modules too. Mirrors the C++
    /// `Diagnostic::line` (`lineOf(anchor)`).
    pub line: u32,
}

type NameMap = HashMap<String, InitState>;

/// One step of a borrow path under a call arg: a field access, an index, or a
/// dereference. Used by the exclusivity check (MVS P5) to compare two arg
/// expressions for overlapping access. Mirrors the C++ `Analyzer::PathStep`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PathStep {
    /// `.field` — carries the field-name `StringIdx`.
    Field(u32),
    /// `[idx]` — `Some(n)` for a constant index, `None` for a dynamic one.
    Index(Option<u64>),
    /// `.*` dereference.
    Deref,
}

/// A borrow path rooted at a base binding. `base == None` when the arg isn't a
/// simple lvalue chain (the exclusivity check skips it). Mirrors the C++
/// `Analyzer::BorrowPath`; `steps` is in root->leaf order after extraction.
struct BorrowPath {
    base: Option<u32>,
    steps: Vec<PathStep>,
}

/// Result of analyzing a statement or expression.
struct StmtResult {
    state: NameMap,
    /// Control flow does not reach the next statement (return/break/continue,
    /// or such an exit on every path).
    terminated: bool,
}

/// Run the move/ownership analysis on `fn_ast`'s body, borrowing `ctx` for the
/// type/enum queries. Returns an empty vector on success; otherwise every
/// detected move/ownership problem. The instantiation-withdrawal caller only
/// inspects `is_empty()`.
pub fn analyze(fn_ast: &FunctionAST, ctx: &CodegenContext) -> Vec<Diagnostic> {
    let mut a = Analyzer::new(ctx);
    a.run(fn_ast)
}

struct Analyzer<'a, 'ctx> {
    ctx: &'a CodegenContext<'ctx>,
    diagnostics: Vec<Diagnostic>,

    /// Current function's parameter modes, prebuilt in `run`.
    param_modes: HashMap<String, ParamMode>,
    /// Current function's parameter list (name + mode), for the borrowed-move
    /// rejection message.
    params: Vec<(String, ParamMode)>,

    /// Names declared `comp const` / `comp var` — no runtime slot, skipped.
    comp_names: HashSet<String>,
    /// Static type per binding name (params + var decls).
    var_types: HashMap<String, TypeIdx>,
    /// Conditional nesting depth: 0 in the body, +1 per if-arm/loop/match-arm.
    cond_depth: i32,
    /// Conditional depth at each `var` local's declaration (params at depth 0
    /// only when `move`-mode).
    decl_depth: HashMap<String, i32>,
    /// Every binding moved anywhere (any type) — phrases read-errors as moves.
    moved_bindings: HashSet<String>,
    /// Drop-bearing bindings that were definitely moved.
    moved_drop_bearing: HashSet<String>,
}

impl<'a, 'ctx> Analyzer<'a, 'ctx> {
    fn new(ctx: &'a CodegenContext<'ctx>) -> Self {
        Analyzer {
            ctx,
            diagnostics: Vec::new(),
            param_modes: HashMap::new(),
            params: Vec::new(),
            comp_names: HashSet::new(),
            var_types: HashMap::new(),
            cond_depth: 0,
            decl_depth: HashMap::new(),
            moved_bindings: HashSet::new(),
            moved_drop_bearing: HashSet::new(),
        }
    }

    fn run(&mut self, fn_ast: &FunctionAST) -> Vec<Diagnostic> {
        self.param_modes.clear();
        self.params.clear();
        self.var_types.clear();
        self.decl_depth.clear();
        self.moved_bindings.clear();
        self.moved_drop_bearing.clear();
        self.comp_names.clear();
        self.cond_depth = 0;

        let mut state: NameMap = HashMap::new();
        for p in &fn_ast.args {
            self.param_modes.insert(p.name.clone(), p.mode);
            self.params.push((p.name.clone(), p.mode));
            state.insert(p.name.clone(), InitState::Init);
            self.var_types.insert(p.name.clone(), p.ty);
            // A `move` parameter is OWNED by the callee — it participates in the
            // move/drop rules like a depth-0 local. `let`/`mut` params are
            // borrowed: moving a drop-bearing value out of them is rejected.
            if p.mode == ParamMode::Move {
                self.decl_depth.insert(p.name.clone(), 0);
            }
        }

        let mut r = StmtResult {
            state,
            terminated: false,
        };
        for &stmt in &fn_ast.body {
            if r.terminated {
                break;
            }
            r = self.analyze_node(stmt, r.state);
        }

        // If control reaches the end of the body without an explicit return,
        // every in-scope drop-bearing local must be definitely initialized —
        // codegen synthesizes a drop at fall-off-end. (Functions that always
        // return skip this; the per-return checks cover them.) The C++
        // run()'s `if (!r.terminated)` guard, init_analysis.cpp:298.
        if !r.terminated {
            self.check_drop_bearing_locals_init(&r.state, NodeIdx::NONE);
        }

        std::mem::take(&mut self.diagnostics)
    }

    // ---- hooks into the codegen context ----

    /// Resolve a `StringIdx` to its (lossy-UTF-8) name.
    fn str_name(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.ctx.string_pool.get(StringIdx::new(id))).into_owned()
    }
    fn node_tag(&self, idx: NodeIdx) -> AstTag {
        self.ctx.node_store.get(idx).tag
    }
    fn extra(&self, i: u32) -> u32 {
        self.ctx.node_store.get_extra(ExtraIdx::new(i))
    }

    /// MATCH-MOVE oracle: does this type carry ownership (so matching it by
    /// value consumes the scrutinee)? The C++ `typeNeedsDrop` hook.
    fn type_owns_drops(&self, ty: TypeIdx) -> bool {
        self.ctx.type_needs_drop(ty)
    }

    /// True when `enum_name.variant_name` names a known enum-variant
    /// constructor (concrete enums + generic factories register by name). Tries
    /// the bare name first, then the body-module-qualified spelling so a
    /// module's own enum reference resolves to its registry key.
    fn is_enum_variant_ctor(&self, enum_name: &str, variant_name: &str) -> bool {
        let probe = |key: &str| {
            self.ctx
                .enum_variants_by_name(key)
                .map(|vs| vs.iter().any(|v| v.name == variant_name))
                .unwrap_or(false)
        };
        if probe(enum_name) {
            return true;
        }
        let bm = self.ctx.current_body_module();
        !bm.is_empty() && probe(&format!("{bm}.{enum_name}"))
    }

    /// Move-gate oracle: is `binding_name`'s type drop-bearing? Consults the
    /// context's `type_needs_drop` (which covers registered drops, structs whose
    /// fields drop, payloaded enums, arrays, and generic instantiations).
    fn binding_is_drop_bearing(&self, binding_name: &str) -> bool {
        match self.var_types.get(binding_name) {
            Some(&ty) => self.type_owns_drops(ty),
            None => false,
        }
    }

    // ---- tag dispatch ----

    fn analyze_node(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        if idx.is_none() {
            return StmtResult {
                state,
                terminated: false,
            };
        }
        let n = *self.ctx.node_store.get(idx);
        match n.tag {
            AstTag::VarDecl => self.analyze_var_decl(idx, state),
            AstTag::Assign => self.analyze_assign(idx, state),
            AstTag::IfNode => self.analyze_if(idx, state),
            AstTag::WhileNode => self.analyze_while(idx, state),
            AstTag::ForNode => self.analyze_for(idx, state),
            AstTag::MatchNode => self.analyze_match(idx, state),
            AstTag::Return => self.analyze_return(idx, state),
            AstTag::Break | AstTag::Continue => StmtResult {
                state,
                terminated: true,
            },

            AstTag::Variable => {
                self.check_variable_read(idx, &state);
                StmtResult {
                    state,
                    terminated: false,
                }
            }
            AstTag::Call => self.analyze_call(idx, state),
            AstTag::TypeMethodCall => self.analyze_type_method_call(idx, &n, state),
            AstTag::BinaryOp => {
                let r = self.analyze_node(NodeIdx::new(n.lhs), state);
                if r.terminated {
                    return r;
                }
                self.analyze_node(NodeIdx::new(n.rhs), r.state)
            }
            AstTag::UnaryOp => self.analyze_node(NodeIdx::new(n.lhs), state),
            AstTag::MemberAccess => self.analyze_node(NodeIdx::new(n.lhs), state),
            AstTag::Index => {
                let r = self.analyze_node(NodeIdx::new(n.lhs), state);
                if r.terminated {
                    return r;
                }
                self.analyze_node(NodeIdx::new(n.rhs), r.state)
            }
            AstTag::Slice => {
                let r = self.analyze_node(NodeIdx::new(n.lhs), state);
                if r.terminated {
                    return r;
                }
                let start = NodeIdx::new(self.extra(n.rhs));
                let end = NodeIdx::new(self.extra(n.rhs + 1));
                let r = self.analyze_node(start, r.state);
                if r.terminated {
                    return r;
                }
                self.analyze_node(end, r.state)
            }
            AstTag::Deref => self.analyze_node(NodeIdx::new(n.lhs), state),
            // `&x` — taking an address is not a read.
            AstTag::AddressOf => StmtResult {
                state,
                terminated: false,
            },
            AstTag::AsCast => self.analyze_node(NodeIdx::new(n.lhs), state),
            AstTag::StructLit => self.analyze_struct_lit(idx, &n, state),
            AstTag::ArrayLit => {
                let count = self.extra(n.rhs);
                let mut r = StmtResult {
                    state,
                    terminated: false,
                };
                for i in 0..count {
                    let elem = NodeIdx::new(self.extra(n.rhs + 1 + i));
                    r = self.analyze_node(elem, r.state);
                    if r.terminated {
                        return r;
                    }
                    if self.node_tag(elem) == AstTag::Variable {
                        let src = self.str_name(self.ctx.node_store.get(elem).lhs);
                        if self.binding_is_drop_bearing(&src) {
                            self.apply_move_to_binding(&src, elem, &mut r.state);
                        }
                    }
                }
                r
            }
            AstTag::ArrayRepeat => {
                let value = NodeIdx::new(self.extra(n.rhs));
                let count = NodeIdx::new(self.extra(n.rhs + 1));
                let r = self.analyze_node(value, state);
                if r.terminated {
                    return r;
                }
                self.analyze_node(count, r.state)
            }

            // Literals + comptime intrinsics + type expressions — no effect.
            AstTag::NumberLit
            | AstTag::BoolLit
            | AstTag::StringLit
            | AstTag::ImportLit
            | AstTag::AtCall
            | AstTag::StructExpr
            | AstTag::EnumExpr
            | AstTag::PatLit
            | AstTag::PatRange
            | AstTag::PatWildcard
            | AstTag::PatOr
            | AstTag::PatEnumVariant
            | AstTag::Invalid
            | AstTag::Count => StmtResult {
                state,
                terminated: false,
            },
        }
    }

    fn analyze_var_decl(&mut self, idx: NodeIdx, mut state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        let extra = n.lhs;
        let name_idx = self.extra(extra);
        let type_idx = TypeIdx::new(self.extra(extra + 1));
        let init_idx = NodeIdx::new(self.extra(extra + 2));
        let name = self.str_name(name_idx);

        // `comp const` / `comp var` (rhs bit 1): no runtime slot, nothing to walk.
        if n.rhs & 2 != 0 {
            self.comp_names.insert(name);
            return StmtResult {
                state,
                terminated: false,
            };
        }
        self.comp_names.remove(&name);

        self.var_types.insert(name.clone(), type_idx);
        self.decl_depth.insert(name.clone(), self.cond_depth);
        // Re-declaration (inner-block shadow) starts a fresh binding.
        self.moved_bindings.remove(&name);
        self.moved_drop_bearing.remove(&name);

        if init_idx.is_none() {
            state.insert(name, InitState::Init);
            return StmtResult {
                state,
                terminated: false,
            };
        }

        let mut r = self.analyze_node(init_idx, state);
        if r.terminated {
            return r;
        }

        // `var owned = c;` with a bare drop-bearing `c` MOVES it.
        if self.node_tag(init_idx) == AstTag::Variable {
            let src = self.str_name(self.ctx.node_store.get(init_idx).lhs);
            if self.binding_is_drop_bearing(&src) {
                self.apply_move_to_binding(&src, init_idx, &mut r.state);
            }
        }

        r.state.insert(name, InitState::Init);
        r
    }

    fn analyze_assign(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        // Assignment to a comp binding mutates astgen's comp scope — skip.
        {
            let target = *self.ctx.node_store.get(NodeIdx::new(n.lhs));
            if target.tag == AstTag::Variable {
                let tname = self.str_name(target.lhs);
                if self.comp_names.contains(&tname) && !self.var_types.contains_key(&tname) {
                    return StmtResult {
                        state,
                        terminated: false,
                    };
                }
            }
        }
        // RHS evaluated first (drop old, then store new).
        let mut r = self.analyze_node(NodeIdx::new(n.rhs), state);
        if r.terminated {
            return r;
        }

        // A bare drop-bearing RHS is MOVED into the store destination
        // (`self.ptr[i] = value`, `s.field = c`).
        if self.node_tag(NodeIdx::new(n.rhs)) == AstTag::Variable {
            let src = self.str_name(self.ctx.node_store.get(NodeIdx::new(n.rhs)).lhs);
            if self.binding_is_drop_bearing(&src) {
                self.apply_move_to_binding(&src, NodeIdx::new(n.rhs), &mut r.state);
            }
        }

        self.analyze_assign_target(NodeIdx::new(n.lhs), r.state)
    }

    fn analyze_assign_target(&mut self, idx: NodeIdx, mut state: NameMap) -> StmtResult {
        if idx.is_none() {
            return StmtResult {
                state,
                terminated: false,
            };
        }
        let n = *self.ctx.node_store.get(idx);
        match n.tag {
            AstTag::Variable => {
                let name = self.str_name(n.lhs);
                // A moved-out drop-bearing binding cannot be re-initialized.
                if self.moved_drop_bearing.contains(&name) {
                    self.emit_error(
                        format!(
                            "cannot assign to `{name}` after it was moved — its scope-exit drop \
                             was suppressed by the move; bind a new name instead"
                        ),
                        idx,
                        name.clone(),
                    );
                }
                state.insert(name, InitState::Init);
                StmtResult {
                    state,
                    terminated: false,
                }
            }
            AstTag::MemberAccess => self.analyze_assign_target(NodeIdx::new(n.lhs), state),
            AstTag::Index => {
                let r = self.analyze_node(NodeIdx::new(n.rhs), state);
                if r.terminated {
                    return r;
                }
                self.analyze_assign_target(NodeIdx::new(n.lhs), r.state)
            }
            AstTag::Deref => self.analyze_node(NodeIdx::new(n.lhs), state),
            _ => self.analyze_node(idx, state),
        }
    }

    fn analyze_if(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);

        // `comp if` (flags bit 0): astgen folds the condition and lowers ONLY
        // the taken arm, inline at the surrounding depth. Mirror that — walk
        // just the taken arm, no depth bump, no merge. If the condition can't be
        // folded (no comp scope here), fall through to the conservative
        // both-arm analysis below, which never under-reports a move.
        if n.flags & 1 != 0
            && let Some(verdict) = self.comp_if_verdict(NodeIdx::new(n.lhs))
        {
            let c_extra = n.rhs;
            let c_then = self.extra(c_extra);
            let c_else = self.extra(c_extra + 1);
            let (base, count) = if verdict {
                (2, c_then)
            } else {
                (2 + c_then, c_else)
            };
            let mut arm = StmtResult {
                state,
                terminated: false,
            };
            for i in 0..count {
                if arm.terminated {
                    break;
                }
                let s = NodeIdx::new(self.extra(c_extra + base + i));
                arm = self.analyze_node(s, arm.state);
            }
            return arm;
        }

        let r = self.analyze_node(NodeIdx::new(n.lhs), state);
        if r.terminated {
            return r;
        }

        let extra = n.rhs;
        let then_count = self.extra(extra);
        let else_count = self.extra(extra + 1);

        let state_before_branch = r.state.clone();

        self.cond_depth += 1;
        let mut then_r = StmtResult {
            state: r.state,
            terminated: false,
        };
        for i in 0..then_count {
            if then_r.terminated {
                break;
            }
            let s = NodeIdx::new(self.extra(extra + 2 + i));
            then_r = self.analyze_node(s, then_r.state);
        }

        let mut else_r = StmtResult {
            state: state_before_branch,
            terminated: false,
        };
        for i in 0..else_count {
            if else_r.terminated {
                break;
            }
            let s = NodeIdx::new(self.extra(extra + 2 + then_count + i));
            else_r = self.analyze_node(s, else_r.state);
        }
        self.cond_depth -= 1;

        if then_r.terminated && else_r.terminated {
            return StmtResult {
                state: HashMap::new(),
                terminated: true,
            };
        }
        if then_r.terminated {
            return else_r;
        }
        if else_r.terminated {
            return then_r;
        }
        StmtResult {
            state: merge_maps(&then_r.state, &else_r.state),
            terminated: false,
        }
    }

    fn analyze_while(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        let r = self.analyze_node(NodeIdx::new(n.lhs), state);
        if r.terminated {
            return r;
        }

        let state_before = r.state.clone();
        let extra = n.rhs;
        let body_count = self.extra(extra);

        self.cond_depth += 1;
        let mut body_r = StmtResult {
            state: r.state,
            terminated: false,
        };
        for i in 0..body_count {
            if body_r.terminated {
                break;
            }
            let s = NodeIdx::new(self.extra(extra + 1 + i));
            body_r = self.analyze_node(s, body_r.state);
        }
        self.cond_depth -= 1;

        if body_r.terminated {
            return StmtResult {
                state: state_before,
                terminated: false,
            };
        }
        StmtResult {
            state: merge_maps(&state_before, &body_r.state),
            terminated: false,
        }
    }

    fn analyze_for(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        let extra = n.lhs;
        let var_idx = self.extra(extra);
        let start_idx = NodeIdx::new(self.extra(extra + 1));
        let end_idx = NodeIdx::new(self.extra(extra + 2));
        let body_count = self.extra(extra + 3);

        let r = self.analyze_node(start_idx, state);
        if r.terminated {
            return r;
        }
        let mut r = self.analyze_node(end_idx, r.state);
        if r.terminated {
            return r;
        }

        let state_before = r.state.clone();
        let var_name = self.str_name(var_idx);
        r.state.insert(var_name, InitState::Init);

        self.cond_depth += 1;
        let mut body_r = StmtResult {
            state: r.state,
            terminated: false,
        };
        for i in 0..body_count {
            if body_r.terminated {
                break;
            }
            let s = NodeIdx::new(self.extra(extra + 4 + i));
            body_r = self.analyze_node(s, body_r.state);
        }
        self.cond_depth -= 1;

        // For-over-range assumes the body runs at least once (matches existing
        // Jam idioms). On `break`, fall back to the pre-loop state.
        if body_r.terminated {
            return StmtResult {
                state: state_before,
                terminated: false,
            };
        }
        body_r
    }

    fn analyze_match(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        let mut r = self.analyze_node(NodeIdx::new(n.lhs), state);
        if r.terminated {
            return r;
        }

        // MATCH-MOVE: matching a drop-bearing enum by value CONSUMES the
        // scrutinee, before the arms fork (so it is unconditional).
        {
            let scrut = *self.ctx.node_store.get(NodeIdx::new(n.lhs));
            if scrut.tag == AstTag::Variable {
                let sname = self.str_name(scrut.lhs);
                if let Some(&ty) = self.var_types.get(&sname)
                    && self.type_owns_drops(ty)
                {
                    self.apply_move_to_binding(&sname, NodeIdx::new(n.lhs), &mut r.state);
                }
            }
        }

        let extra = n.rhs;
        let arm_count = self.extra(extra);
        let state_before = r.state.clone();

        let mut merged = StmtResult {
            state: HashMap::new(),
            terminated: true,
        };
        let mut cursor = 1u32;
        for _ in 0..arm_count {
            cursor += 1; // patIdx
            let arm_body_count = self.extra(extra + cursor);
            cursor += 1;

            self.cond_depth += 1;
            let mut arm = StmtResult {
                state: state_before.clone(),
                terminated: false,
            };
            for i in 0..arm_body_count {
                if arm.terminated {
                    break;
                }
                let s = NodeIdx::new(self.extra(extra + cursor + i));
                arm = self.analyze_node(s, arm.state);
            }
            self.cond_depth -= 1;
            cursor += arm_body_count;

            if merged.terminated && arm.terminated {
                // both terminated; stay terminated
            } else if merged.terminated {
                merged = arm;
            } else if arm.terminated {
                // keep merged
            } else {
                merged.state = merge_maps(&merged.state, &arm.state);
            }
        }

        merged
    }

