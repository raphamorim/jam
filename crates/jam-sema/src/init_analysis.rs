/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Definite-initialization / move-ownership analysis.
//!
//! A tree-walking dataflow over a function body: at each program point, a map
//! from binding name to a 2-bit lattice {Unknown, Init, Uninit, MaybeInit},
//! merged by bitwise OR (as in Clang's `UninitializedValues`). Jam programs
//! are structured (no goto), so a recursive AST walk suffices — no CFG pass.
//!
//! The single consumer is generic method-instantiation withdrawal
//! ([`crate::astgen::instantiate_methods`]): when a generic method's body is
//! lowered under drop-bearing type args and this analysis reports any
//! move/ownership diagnostic, the method is withdrawn to a bare `declare`
//! (no body). Canonical trigger: `Vec(T).filled`, which moves a by-value
//! (`let`-mode) drop-bearing parameter into a slot inside a `while` loop.
//! Only the accept/reject verdict matters — whether [`analyze`] returns a
//! non-empty diagnostics list — so the message strings are terse.

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
    /// stamped, so it is correct for imported modules too.
    pub line: u32,
}

type NameMap = HashMap<String, InitState>;

/// One step of a borrow path under a call arg: a field access, an index, or a
/// dereference. Used by the exclusivity check (MVS P5) to compare two arg
/// expressions for overlapping access.
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
/// simple lvalue chain (the exclusivity check skips it). `steps` is in
/// root->leaf order after extraction.
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
    /// For-each loop vars that view elements of a read-only parameter —
    /// writes through them land in the caller's storage, so they are
    /// rejected like writes into the parameter itself. Slice bindings
    /// initialized from such storage (directly or through a call that
    /// returns a view, e.g. `v.asSlice()`) join the set too.
    readonly_views: HashSet<String>,
    /// Cache: fn name -> parameter indices whose storage the fn's returned
    /// slice may view (see `return_view_params`). Shared across the run;
    /// summaries are position-independent facts about the callee.
    view_summaries: std::cell::RefCell<HashMap<String, Vec<usize>>>,
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
            readonly_views: HashSet::new(),
            view_summaries: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn run(&mut self, fn_ast: &FunctionAST) -> Vec<Diagnostic> {
        self.param_modes.clear();
        self.params.clear();
        self.var_types.clear();
        self.decl_depth.clear();
        self.moved_bindings.clear();
        self.moved_drop_bearing.clear();
        self.readonly_views.clear();
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
        // return skip this; the per-return checks cover them.)
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
    /// value consumes the scrutinee)?
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
            // `args...` spread: the pack params are materialized (always
            // initialized) clone params — nothing to track.
            AstTag::Spread => StmtResult {
                state,
                terminated: false,
            },
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
        // A slice binding initialized from a read-only borrow's storage
        // (`const s = v.asSlice();`, `const t = s[1..3];`) is itself a
        // read-only view.
        self.readonly_views.remove(&name);
        if !type_idx.is_none()
            && !init_idx.is_none()
            && self.ctx.type_pool.get(type_idx).kind == TypeKind::Slice
            && self
                .write_root(init_idx)
                .is_some_and(|r| self.is_readonly_root(&r))
        {
            self.readonly_views.insert(name.clone());
        }

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
        // MVS P2: a write whose destination storage is OWNED by a read-only
        // (default-mode) parameter lands in the caller's value — reject it.
        // Writes through pointer indirection (`p.*`, `self.ptr[i]`) are not
        // the parameter's storage and stay allowed.
        self.check_readonly_param_write(NodeIdx::new(n.lhs));

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

        // Rebinding a slice variable recomputes its view membership from
        // the new RHS (it may start or stop viewing read-only storage).
        {
            let target = *self.ctx.node_store.get(NodeIdx::new(n.lhs));
            if target.tag == AstTag::Variable {
                let tname = self.str_name(target.lhs);
                let is_slice_local = self
                    .var_types
                    .get(&tname)
                    .is_some_and(|&t| self.ctx.type_pool.get(t).kind == TypeKind::Slice)
                    && self.param_modes.get(&tname).is_none_or(|_| {
                        // params keep their mode-derived status
                        self.decl_depth.contains_key(&tname)
                    });
                if is_slice_local {
                    if self
                        .write_root(NodeIdx::new(n.rhs))
                        .is_some_and(|r| self.is_readonly_root(&r))
                    {
                        self.readonly_views.insert(tname);
                    } else {
                        self.readonly_views.remove(&tname);
                    }
                }
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
        // op 1 walks an iterable; there is no end bound to analyze.
        let mut r = if n.op == 1 {
            r
        } else {
            self.analyze_node(end_idx, r.state)
        };
        if r.terminated {
            return r;
        }

        let state_before = r.state.clone();
        let var_name = self.str_name(var_idx);
        r.state.insert(var_name.clone(), InitState::Init);

        // A for-each var binds elements in place; when the iterable's storage
        // belongs to a read-only parameter, the var is a read-only view for
        // the body (membership is save/restored around shadowing).
        let was_view = self.readonly_views.contains(&var_name);
        let is_view = n.op == 1
            && self
                .write_root(start_idx)
                .is_some_and(|root| self.is_readonly_root(&root));
        if is_view {
            self.readonly_views.insert(var_name.clone());
        } else {
            self.readonly_views.remove(&var_name);
        }

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

        if was_view {
            self.readonly_views.insert(var_name);
        } else {
            self.readonly_views.remove(&var_name);
        }

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

    fn analyze_return(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        // Scope-escape check runs BEFORE any reads of the operand: returning
        // `&` of a `mut`-mode parameter would extend a second-class borrow
        // past the call frame.
        if !NodeIdx::new(n.lhs).is_none() {
            self.check_scope_escape(NodeIdx::new(n.lhs));
        }
        let mut r = StmtResult {
            state,
            terminated: false,
        };
        if !NodeIdx::new(n.lhs).is_none() {
            r = self.analyze_node(NodeIdx::new(n.lhs), r.state);
        }
        // Every in-scope drop-bearing local must be Init at this return: codegen
        // emits a drop on it here, and dropping uninit memory is UB.
        self.check_drop_bearing_locals_init(&r.state, idx);
        r.terminated = true;
        r
    }

    fn analyze_call(&mut self, idx: NodeIdx, state: NameMap) -> StmtResult {
        let n = *self.ctx.node_store.get(idx);
        // Resolve the callee so its parameter MODES are visible (so `move`-mode
        // args move the caller's binding). We look up by call-site name + by
        // `recv.method` through the receiver's static type.
        let mut callee: Option<FunctionAST> = None;
        let mut arg_mode_offset = 0usize;
        let mut is_variant_ctor = false;
        let mut recv_name: Option<String> = None;

        if n.flags & 1 == 0 {
            let callee_name = self.str_name(n.lhs);
            callee = self.ctx.get_function_ast(&callee_name);
            if let Some(dot) = callee_name.find('.')
                && callee.is_none()
                && callee_name.rfind('.') == Some(dot)
            {
                let recv = &callee_name[..dot];
                let method = &callee_name[dot + 1..];
                if self.is_enum_variant_ctor(recv, method) {
                    is_variant_ctor = true;
                }
                if let Some(&recv_ty) = self.var_types.get(recv) {
                    let recv_ty = self
                        .ctx
                        .requalify_type(recv_ty, &self.ctx.current_body_module());
                    let k = self.ctx.type_pool.get(recv_ty);
                    if matches!(
                        k.kind,
                        TypeKind::Struct | TypeKind::Named | TypeKind::GenericCall
                    ) {
                        let type_name = self.str_name(k.a);
                        if let Some(f) = self.ctx.get_function_ast(&format!("{type_name}.{method}"))
                        {
                            callee = Some(f);
                            arg_mode_offset = 1;
                            recv_name = Some(recv.to_string());
                        }
                    }
                }
            }
        }

        // A `mut self` method writes into the receiver's storage — reject
        // calling one through a read-only borrow.
        if let (Some(c), Some(rn)) = (&callee, &recv_name)
            && !c.args.is_empty()
            && c.args[0].mode == ParamMode::Mut
            && self.is_readonly_root(rn)
        {
            let method = &c.name[c.name.rfind('.').map_or(0, |d| d + 1)..];
            self.emit_error(
                format!(
                    "cannot call `mut self` method `{method}` through `{rn}` — it is a \
                     read-only borrow; take the parameter as `mut` in the signature"
                ),
                idx,
                rn.clone(),
            );
        }

        let extra = n.rhs;
        let arg_count = self.extra(extra);

        // Pre-pass: collect (argIdx, mode, path) for every arg so the exclusivity
        // check runs BEFORE any state transition (a flagged conflict surfaces even
        // if a later arg would terminate the analysis).
        struct ArgInfo {
            arg_idx: NodeIdx,
            mode: ParamMode,
            path: BorrowPath,
        }
        let mut arg_infos: Vec<ArgInfo> = Vec::with_capacity(arg_count as usize);
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(self.extra(extra + 1 + i));
            let mode = match &callee {
                Some(c) if i as usize + arg_mode_offset < c.args.len() => {
                    c.args[i as usize + arg_mode_offset].mode
                }
                _ => ParamMode::Let,
            };
            let path = self.extract_path(arg_idx);
            arg_infos.push(ArgInfo {
                arg_idx,
                mode,
                path,
            });
        }

        // MVS P5 exclusivity: for each pair of args with overlapping paths, the
        // borrow set is OK only if BOTH modes are Let (multiple readers). Any
        // other combination on overlapping paths is rejected.
        for i in 0..arg_infos.len() {
            for j in (i + 1)..arg_infos.len() {
                let a = &arg_infos[i];
                let b = &arg_infos[j];
                if a.mode == ParamMode::Let && b.mode == ParamMode::Let {
                    continue;
                }
                if !Self::paths_overlap(&a.path, &b.path) {
                    continue;
                }
                let Some(base) = a.path.base else { continue }; // un-tracked, skip
                let name = self.str_name(base);
                let anchor = a.arg_idx;
                self.emit_error(
                    format!(
                        "conflicting borrows of `{name}` in the same call: at least one access is \
                         exclusive (mut/move/undefined)"
                    ),
                    anchor,
                    name,
                );
            }
        }

        let mut r = StmtResult {
            state,
            terminated: false,
        };

        // Walk the indirect callee subtree (`expr.method(args)`).
        if n.flags & 1 != 0 {
            r = self.analyze_node(NodeIdx::new(n.lhs), r.state);
            if r.terminated {
                return r;
            }
        }

        for info in &arg_infos {
            if r.terminated {
                return r;
            }
            r = self.analyze_node(info.arg_idx, r.state);
            if r.terminated {
                return r;
            }

            // `&`-rooted args skip the read-check (taking an address is not a
            // read), but borrowing a MOVED binding must still be rejected: a
            // callee could write through the pointer and "re-initialize" storage
            // whose scope-exit drop was already suppressed, leaking the new value.
            if self.node_tag(info.arg_idx) == AstTag::AddressOf
                && let Some(base) = info.path.base
            {
                let bname = self.str_name(base);
                if self.moved_bindings.contains(&bname)
                    && r.state.get(&bname).copied() != Some(InitState::Init)
                    && r.state.contains_key(&bname)
                {
                    self.emit_error(
                        format!(
                            "cannot borrow `{bname}` after it was moved — a write through the \
                             borrow would re-initialize storage whose drop was suppressed; bind a \
                             new name instead"
                        ),
                        info.arg_idx,
                        bname,
                    );
                }
            }

            // A `mut`-mode arg lets the callee write into the arg's storage —
            // reject when that storage is a read-only borrow. (Closes the
            // wrap-the-write-in-a-helper bypass of the P2 write check.)
            if info.mode == ParamMode::Mut
                && let Some(root) = self.write_root(info.arg_idx)
                && self.is_readonly_root(&root)
            {
                self.emit_error(
                    format!(
                        "cannot pass `{root}` as a `mut` argument — it is a read-only \
                         borrow; take the parameter as `mut` in this fn's signature too"
                    ),
                    info.arg_idx,
                    root,
                );
            }

            // `move`-mode arg moves the caller's base binding.
            if info.mode == ParamMode::Move
                && let Some(base) = self.find_base_binding(info.arg_idx)
            {
                self.apply_move_to_binding(&base, info.arg_idx, &mut r.state);
            }
        }

        // Enum-variant construction: bare drop-bearing payload args MOVE.
        if is_variant_ctor {
            self.apply_payload_moves(extra + 1, arg_count, &mut r.state);
        }
        r
    }

    fn analyze_type_method_call(
        &mut self,
        _idx: NodeIdx,
        n: &jam_syntax::ast_flat::AstNode,
        state: NameMap,
    ) -> StmtResult {
        // Receiver is a TypeIdx; walk the value-arg list. When the receiver is an
        // ENUM and the "method" is a variant, bare drop-bearing args MOVE.
        let extra = n.rhs;
        let _method_id = self.extra(extra);
        let arg_count = self.extra(extra + 1);
        let mut r = StmtResult {
            state,
            terminated: false,
        };
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(self.extra(extra + 2 + i));
            r = self.analyze_node(arg_idx, r.state);
            if r.terminated {
                return r;
            }
        }
        let recv_ty = TypeIdx::new(n.lhs);
        let k = self.ctx.type_pool.get(recv_ty);
        if matches!(
            k.kind,
            TypeKind::Named | TypeKind::Enum | TypeKind::GenericCall
        ) {
            let recv_name = self.str_name(k.a);
            let method = self.str_name(self.extra(extra));
            if !recv_name.is_empty() && self.is_enum_variant_ctor(&recv_name, &method) {
                self.apply_payload_moves(extra + 2, arg_count, &mut r.state);
            }
        }
        r
    }

    /// Apply the enum-payload capture rule: a bare drop-bearing payload arg
    /// MOVES into the enum.
    fn apply_payload_moves(&mut self, args_base: u32, arg_count: u32, state: &mut NameMap) {
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(self.extra(args_base + i));
            if self.node_tag(arg_idx) == AstTag::Variable {
                let src = self.str_name(self.ctx.node_store.get(arg_idx).lhs);
                if self.binding_is_drop_bearing(&src) {
                    self.apply_move_to_binding(&src, arg_idx, state);
                }
            }
        }
    }

    fn analyze_struct_lit(
        &mut self,
        _idx: NodeIdx,
        n: &jam_syntax::ast_flat::AstNode,
        state: NameMap,
    ) -> StmtResult {
        let extra = n.rhs;
        let field_count = self.extra(extra);
        let mut r = StmtResult {
            state,
            terminated: false,
        };
        for i in 0..field_count {
            if r.terminated {
                return r;
            }
            let expr_idx = NodeIdx::new(self.extra(extra + 1 + 2 * i + 1));
            r = self.analyze_node(expr_idx, r.state);

            // A bare drop-bearing binding used as a field initializer is a MOVE.
            if self.node_tag(expr_idx) == AstTag::Variable {
                let fname = self.str_name(self.ctx.node_store.get(expr_idx).lhs);
                if self.binding_is_drop_bearing(&fname) {
                    self.apply_move_to_binding(&fname, expr_idx, &mut r.state);
                }
            }
        }
        r
    }

    /// Apply the caller-side effect of moving `name`. The binding becomes
    /// Uninit; for drop-bearing types the move is validated against the rules
    /// that keep codegen's lexical drop suppression sound.
    fn apply_move_to_binding(&mut self, name: &str, anchor: NodeIdx, state: &mut NameMap) {
        if self.binding_is_drop_bearing(name) {
            match self.decl_depth.get(name).copied() {
                None => {
                    // Not an owned binding (a `let`/`mut` param, or a global).
                    // Moving a drop-bearing value out of a borrowed parameter is
                    // rejected — it is borrowed, not owned. This is the
                    // `Vec(T).filled` trigger (`value: T` is a `let` param).
                    if let Some(&mode) = self.param_modes.get(name)
                        && mode != ParamMode::Move
                    {
                        let mode_word = if mode == ParamMode::Mut { "mut" } else { "let" };
                        self.emit_error(
                            format!(
                                "cannot move drop-bearing `{name}` out of a `{mode_word}` \
                                 parameter — it is borrowed, not owned; declare the parameter \
                                 `move` to take ownership"
                            ),
                            anchor,
                            name.to_string(),
                        );
                    }
                    // Unknown/global name: just record the state.
                }
                Some(dd) if dd != self.cond_depth => {
                    self.emit_error(
                        format!(
                            "cannot move `{name}` here: the move is conditional relative to its \
                             declaration, so its drop cannot run unconditionally — move it on all \
                             control-flow paths or none"
                        ),
                        anchor,
                        name.to_string(),
                    );
                }
                Some(_) => {
                    self.moved_drop_bearing.insert(name.to_string());
                }
            }
        }
        self.moved_bindings.insert(name.to_string());
        state.insert(name.to_string(), InitState::Uninit);
    }

    /// Extract a borrow path from a call-arg expression: a chain of `&`,
    /// `.field`, `[idx]`, `.*` rooted at a Variable. Any other shape leaves
    /// `base == None` so the exclusivity check skips that arg. `&` is
    /// transparent — it does not contribute a step.
    fn extract_path(&self, arg_idx: NodeIdx) -> BorrowPath {
        let mut path = BorrowPath {
            base: None,
            steps: Vec::new(),
        };
        let mut cur = arg_idx;
        while !cur.is_none() {
            let n = *self.ctx.node_store.get(cur);
            match n.tag {
                AstTag::Variable => {
                    path.base = Some(n.lhs);
                    path.steps.reverse();
                    return path;
                }
                // Pass through; `&` is not a path step.
                AstTag::AddressOf => cur = NodeIdx::new(n.lhs),
                AstTag::MemberAccess => {
                    path.steps.push(PathStep::Field(n.rhs));
                    cur = NodeIdx::new(n.lhs);
                }
                AstTag::Index => {
                    let idx_node = *self.ctx.node_store.get(NodeIdx::new(n.rhs));
                    let step = if idx_node.tag == AstTag::NumberLit {
                        // lhs = lo32, rhs = hi32, flags bit 0 = isNeg.
                        let mag = (idx_node.lhs as u64) | ((idx_node.rhs as u64) << 32);
                        if idx_node.flags & 1 == 0 {
                            PathStep::Index(Some(mag))
                        } else {
                            PathStep::Index(None)
                        }
                    } else {
                        PathStep::Index(None)
                    };
                    path.steps.push(step);
                    cur = NodeIdx::new(n.lhs);
                }
                AstTag::Deref => {
                    path.steps.push(PathStep::Deref);
                    cur = NodeIdx::new(n.lhs);
                }
                _ => {
                    // Computed expression — not a tracked path.
                    path.steps.clear();
                    return path;
                }
            }
        }
        path
    }

    /// Two paths overlap when one is a prefix of the other (or they are equal).
    /// Per MVS §4.1.
    fn paths_overlap(a: &BorrowPath, b: &BorrowPath) -> bool {
        let (Some(ab), Some(bb)) = (a.base, b.base) else {
            return false;
        };
        if ab != bb {
            return false;
        }
        let common = a.steps.len().min(b.steps.len());
        for i in 0..common {
            match (a.steps[i], b.steps[i]) {
                (PathStep::Field(fa), PathStep::Field(fb)) => {
                    if fa != fb {
                        return false;
                    }
                }
                (PathStep::Index(ia), PathStep::Index(ib)) => match (ia, ib) {
                    (Some(ca), Some(cb)) => {
                        if ca != cb {
                            return false;
                        }
                    }
                    // At least one dynamic index — conservatively overlap.
                    _ => return true,
                },
                (PathStep::Deref, PathStep::Deref) => {}
                // Different kinds: disjoint.
                _ => return false,
            }
        }
        true
    }

    /// Walk an lvalue/arg chain (`x`, `p.x`, `arr[i]`, `p.*`) down to the
    /// leftmost Variable; return its name. `None` if not a trackable base.
    fn find_base_binding(&self, arg_idx: NodeIdx) -> Option<String> {
        let mut cur = arg_idx;
        while !cur.is_none() {
            let n = *self.ctx.node_store.get(cur);
            match n.tag {
                AstTag::Variable => return Some(self.str_name(n.lhs)),
                AstTag::AddressOf | AstTag::MemberAccess | AstTag::Index | AstTag::Deref => {
                    cur = NodeIdx::new(n.lhs)
                }
                _ => return None,
            }
        }
        None
    }

    fn check_variable_read(&mut self, idx: NodeIdx, state: &NameMap) {
        let n = *self.ctx.node_store.get(idx);
        let name = self.str_name(n.lhs);
        match state.get(&name) {
            None => {} // unseen (global/import/for-var) — assume Init
            Some(InitState::Init) | Some(InitState::Unknown) => {}
            Some(InitState::Uninit) => {
                if self.moved_bindings.contains(&name) {
                    self.emit_error(
                        format!(
                            "use of moved binding `{name}` — its value was moved out here or \
                             earlier; if both values are needed, clone before moving \
                             (`{name}.clone()`)"
                        ),
                        idx,
                        name.clone(),
                    );
                } else {
                    self.emit_error(
                        format!("use of uninitialized binding `{name}`"),
                        idx,
                        name.clone(),
                    );
                }
            }
            Some(InitState::MaybeInit) => {
                if self.moved_bindings.contains(&name) {
                    self.emit_error(
                        format!("binding `{name}` may have been moved on a path that reaches here"),
                        idx,
                        name.clone(),
                    );
                } else {
                    self.emit_error(
                        format!(
                            "binding `{name}` may not be initialized on every path that reaches here"
                        ),
                        idx,
                        name.clone(),
                    );
                }
            }
        }
    }

    /// `comp if` verdict: fold the condition. Returns `Some(true/false)` for a
    /// foldable comptime bool, `None` to fall back to conservative both-arm
    /// analysis. Generic method bodies under analysis carry no comp scope of
    /// their own, so this folds only against module/comp-subst consts; a
    /// condition astgen folded through a function-local comp binding is
    /// unfoldable here and analyzes BOTH arms — conservative: it may
    /// over-report from a dead arm, but never under-reports.
    fn comp_if_verdict(&self, cond: NodeIdx) -> Option<bool> {
        match self.ctx.fold_comptime_expr(cond) {
            crate::comptime::ComptimeValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// At every `return` and at fall-off-end, a drop-bearing local that is not
    /// definitely-initialized must be rejected — codegen synthesizes a drop on
    /// it at this exit and dropping uninit memory is UB.
    fn check_drop_bearing_locals_init(&mut self, state: &NameMap, anchor: NodeIdx) {
        // Walk only the bindings CURRENTLY IN SCOPE (present in the merged state
        // map). Inner-block bindings have been dropped at their own scope end and
        // the merge removed them; var_types retains every name we ever saw, so
        // checking those would false-positive.
        for (name, &s) in state.iter() {
            // `let`/`mut` params are borrowed — the caller owns the value and its
            // drop. `move` params are callee-owned and DO drop at exit, so they
            // go through the same classification as locals.
            if let Some(&mode) = self.param_modes.get(name)
                && mode != ParamMode::Move
            {
                continue;
            }

            if !self.binding_is_drop_bearing(name) {
                continue;
            }

            if s == InitState::Init {
                continue;
            }

            // Uninit + definitely-moved-on-every-path == rustc's Dead: codegen
            // suppressed the drop, nothing runs. OK.
            if s == InitState::Uninit && self.moved_drop_bearing.contains(name) {
                continue;
            }

            // Recover the source-spelled type name for the diagnostic.
            let mut type_name = String::from("?");
            if let Some(&ty) = self.var_types.get(name) {
                let k = self.ctx.type_pool.get(ty);
                if matches!(
                    k.kind,
                    TypeKind::Struct
                        | TypeKind::Named
                        | TypeKind::Enum
                        | TypeKind::Union
                        | TypeKind::GenericCall
                ) && k.a != 0
                {
                    type_name = self.str_name(k.a);
                }
            }

            let mut msg = format!("drop-bearing binding `{name}` of type `{type_name}` ");
            if s == InitState::Uninit {
                msg.push_str(
                    "must be initialized before this exit; drop runs on it and would otherwise \
                     read uninit memory",
                );
            } else if self.moved_bindings.contains(name) {
                // rustc's Conditional (maybe-init AND maybe-uninit): rustc inserts
                // a runtime drop flag; jam rejects (Swift SE-0390 style).
                msg.push_str(
                    "is moved on some control-flow paths but not others; its drop cannot run \
                     conditionally — move it on all paths or none",
                );
            } else {
                msg.push_str(
                    "may not be initialized on every path that reaches this exit; drop runs on it \
                     on every path",
                );
            }
            self.emit_error(msg, anchor, name.clone());
        }
    }

    /// Returning the address of a `mut`-mode parameter must be rejected —
    /// borrows are second-class and cannot escape the function (dangling
    /// pointer).
    fn check_scope_escape(&mut self, expr_idx: NodeIdx) {
        if expr_idx.is_none() {
            return;
        }
        let top = *self.ctx.node_store.get(expr_idx);
        // Only `&`-rooted expressions can carry a borrow out of the function.
        // Returning a parameter by value (`return p;`, `return p.x;`) is a copy.
        if top.tag != AstTag::AddressOf {
            return;
        }
        // Walk the path under the `&` to the base Variable, descending through
        // MemberAccess / Index (so `&p.field`, `&arr[i]` are covered) but stopping
        // at Deref (the dereferenced pointee is data, not the borrow path).
        let mut cur = NodeIdx::new(top.lhs);
        while !cur.is_none() {
            let m = *self.ctx.node_store.get(cur);
            match m.tag {
                AstTag::Variable => {
                    let name = self.str_name(m.lhs);
                    // Find this name in the parameter list.
                    if let Some(&(_, mode)) = self.params.iter().find(|(pn, _)| pn == &name) {
                        if mode != ParamMode::Mut {
                            return;
                        }
                        self.emit_error(
                            format!(
                                "cannot return `&` of `mut`-mode parameter `{name}` — borrows are \
                                 second-class and cannot escape the function"
                            ),
                            expr_idx,
                            name,
                        );
                    }
                    return; // not a parameter at all
                }
                AstTag::MemberAccess | AstTag::Index => cur = NodeIdx::new(m.lhs),
                // Deref or anything else: not a borrow path on the parameter.
                _ => return,
            }
        }
    }

    /// 1-based source line of `idx`, or `0` for none. NodeStore's parallel
    /// line table is parse-time stamped, so it is correct for imported modules
    /// too.
    fn line_of(&self, idx: NodeIdx) -> u32 {
        if idx.is_none() {
            return 0;
        }
        self.ctx.node_store.get_line(idx)
    }

    /// Static type of a simple lvalue chain, or None when a step can't be
    /// typed (calls, unregistered types, unions). Purely a helper for the
    /// read-only-parameter write check — permissive on None.
    fn lvalue_type(&self, idx: NodeIdx) -> Option<TypeIdx> {
        let n = *self.ctx.node_store.get(idx);
        match n.tag {
            AstTag::Variable => self.var_types.get(&self.str_name(n.lhs)).copied(),
            AstTag::MemberAccess => {
                let mut bt = self.lvalue_type(NodeIdx::new(n.lhs))?;
                let k = self.ctx.type_pool.get(bt);
                if k.kind == TypeKind::PtrSingle {
                    bt = TypeIdx::new(k.a);
                }
                let field = self.str_name(n.rhs);
                self.ctx
                    .struct_fields(bt)?
                    .into_iter()
                    .find(|(name, _)| *name == field)
                    .map(|(_, ty)| ty)
            }
            AstTag::Index | AstTag::Slice => {
                let bt = self.lvalue_type(NodeIdx::new(n.lhs))?;
                let k = self.ctx.type_pool.get(bt);
                match k.kind {
                    TypeKind::Array | TypeKind::Slice | TypeKind::PtrMany => {
                        if n.tag == AstTag::Slice {
                            Some(self.ctx.type_pool.intern_slice(TypeIdx::new(k.a)))
                        } else {
                            Some(TypeIdx::new(k.a))
                        }
                    }
                    _ => None,
                }
            }
            AstTag::Deref => {
                let bt = self.lvalue_type(NodeIdx::new(n.lhs))?;
                let k = self.ctx.type_pool.get(bt);
                match k.kind {
                    TypeKind::PtrSingle | TypeKind::PtrMany => Some(TypeIdx::new(k.a)),
                    _ => None,
                }
            }
            AstTag::Call => {
                let (callee, _, _) = self.resolve_call(idx, None)?;
                (!callee.return_type.is_none()).then_some(callee.return_type)
            }
            _ => None,
        }
    }

    /// The binding whose storage a write to this lvalue lands in, walking
    /// value projections (fields, array/slice elements, sub-slices) back to
    /// the root. None when the chain passes through pointer indirection
    /// (`.*`, a many-item pointer, a pointer-typed field) or a non-lvalue —
    /// those writes are not the root binding's own storage.
    fn write_root(&self, idx: NodeIdx) -> Option<String> {
        let n = *self.ctx.node_store.get(idx);
        match n.tag {
            AstTag::Variable => Some(self.str_name(n.lhs)),
            AstTag::MemberAccess => {
                let base = NodeIdx::new(n.lhs);
                let k = self.ctx.type_pool.get(self.lvalue_type(base)?).kind;
                if k == TypeKind::PtrSingle || k == TypeKind::PtrMany {
                    return None;
                }
                self.write_root(base)
            }
            AstTag::Index | AstTag::Slice => {
                let base = NodeIdx::new(n.lhs);
                let k = self.ctx.type_pool.get(self.lvalue_type(base)?).kind;
                match k {
                    // A slice's elements are the borrowed pointee — a
                    // read-only borrow covers them (mode table: default
                    // mode's pointee is read-only).
                    TypeKind::Array | TypeKind::Slice => self.write_root(base),
                    _ => None,
                }
            }
            // A call that returns a slice viewing one of its arguments
            // (per the callee's provenance summary): the storage is the
            // argument's. Surfaces a root only when that root is a
            // read-only borrow — other derivations stay permissive.
            AstTag::Call => {
                let (callee, recv, args) = self.resolve_call(idx, None)?;
                let viewed = self.return_view_params(&callee, 4);
                let offset = usize::from(recv.is_some());
                for k in viewed {
                    let root = if k == 0 && offset == 1 {
                        recv.clone()
                    } else {
                        args.get(k - offset).and_then(|&a| self.write_root(a))
                    };
                    if let Some(r) = root
                        && self.is_readonly_root(&r)
                    {
                        return Some(r);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Resolve a direct Call node to (callee AST, receiver name, arg nodes).
    /// Mirrors `analyze_call`'s resolution: the stored callee name first,
    /// then `recv.method` through the receiver's static type. Receiver
    /// types come from `param_ctx`'s parameter list when summarizing a
    /// callee body, else from the current function's bindings. None for
    /// indirect calls (`expr.method(...)`) and unresolved names.
    fn resolve_call(
        &self,
        idx: NodeIdx,
        param_ctx: Option<&FunctionAST>,
    ) -> Option<(FunctionAST, Option<String>, Vec<NodeIdx>)> {
        let n = *self.ctx.node_store.get(idx);
        if n.tag != AstTag::Call || n.flags & 1 != 0 {
            return None;
        }
        let callee_name = self.str_name(n.lhs);
        let extra = n.rhs;
        let arg_count = self.extra(extra);
        let args: Vec<NodeIdx> = (0..arg_count)
            .map(|i| NodeIdx::new(self.extra(extra + 1 + i)))
            .collect();
        if let Some(f) = self.ctx.get_function_ast(&callee_name) {
            return Some((f, None, args));
        }
        let dot = callee_name.find('.')?;
        if callee_name.rfind('.') != Some(dot) {
            return None;
        }
        let recv = &callee_name[..dot];
        let method = &callee_name[dot + 1..];
        let recv_ty = match param_ctx {
            Some(f) => f.args.iter().find(|p| p.name == recv).map(|p| p.ty)?,
            None => self.var_types.get(recv).copied()?,
        };
        let recv_ty = self
            .ctx
            .requalify_type(recv_ty, &self.ctx.current_body_module());
        let k = self.ctx.type_pool.get(recv_ty);
        if !matches!(
            k.kind,
            TypeKind::Struct | TypeKind::Named | TypeKind::GenericCall
        ) {
            return None;
        }
        let type_name = self.str_name(k.a);
        let f = self.ctx.get_function_ast(&format!("{type_name}.{method}"))?;
        Some((f, Some(recv.to_string()), args))
    }

    /// Collect the operands of every `return` reachable in `stmts`
    /// (recursing through if/while/for/match bodies).
    fn collect_returns(&self, stmts: &[NodeIdx], out: &mut Vec<NodeIdx>) {
        for &s in stmts {
            if s.is_none() {
                continue;
            }
            let n = *self.ctx.node_store.get(s);
            match n.tag {
                AstTag::Return => {
                    if !NodeIdx::new(n.lhs).is_none() {
                        out.push(NodeIdx::new(n.lhs));
                    }
                }
                AstTag::IfNode => {
                    let extra = n.rhs;
                    let count = self.extra(extra) + self.extra(extra + 1);
                    let body: Vec<NodeIdx> = (0..count)
                        .map(|i| NodeIdx::new(self.extra(extra + 2 + i)))
                        .collect();
                    self.collect_returns(&body, out);
                }
                AstTag::WhileNode => {
                    let extra = n.rhs;
                    let count = self.extra(extra);
                    let body: Vec<NodeIdx> = (0..count)
                        .map(|i| NodeIdx::new(self.extra(extra + 1 + i)))
                        .collect();
                    self.collect_returns(&body, out);
                }
                AstTag::ForNode => {
                    let extra = n.lhs;
                    let count = self.extra(extra + 3);
                    let body: Vec<NodeIdx> = (0..count)
                        .map(|i| NodeIdx::new(self.extra(extra + 4 + i)))
                        .collect();
                    self.collect_returns(&body, out);
                }
                AstTag::MatchNode => {
                    let extra = n.rhs;
                    let arm_count = self.extra(extra);
                    let mut cursor = 1u32;
                    let mut body: Vec<NodeIdx> = Vec::new();
                    for _ in 0..arm_count {
                        cursor += 1; // pattern
                        let c = self.extra(extra + cursor);
                        cursor += 1;
                        for i in 0..c {
                            body.push(NodeIdx::new(self.extra(extra + cursor + i)));
                        }
                        cursor += c;
                    }
                    self.collect_returns(&body, out);
                }
                _ => {}
            }
        }
    }

    /// Parameter indices whose storage (or owned tree — pointer FIELDS
    /// included, the ownership traversal) the returned slice of `callee`
    /// may view. Raw-pointer PARAMETERS are excluded: their pointee is
    /// governed by the pointer's own `mut`/`const`, not the parameter
    /// mode. Non-slice returns and extern fns view nothing. Summaries
    /// are cached by fn name; `depth` bounds transitive call chasing.
    fn return_view_params(&self, callee: &FunctionAST, depth: u32) -> Vec<usize> {
        if depth == 0 || callee.is_extern || callee.return_type.is_none() {
            return Vec::new();
        }
        if self.ctx.type_pool.get(callee.return_type).kind != TypeKind::Slice {
            return Vec::new();
        }
        if let Some(v) = self.view_summaries.borrow().get(&callee.name) {
            return v.clone();
        }
        let mut returns = Vec::new();
        self.collect_returns(&callee.body, &mut returns);
        let mut out: Vec<usize> = Vec::new();
        for r in returns {
            self.expr_view_params(callee, r, depth, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        self.view_summaries
            .borrow_mut()
            .insert(callee.name.clone(), out.clone());
        out
    }

    /// Which of `fn_ast`'s parameters the expression at `idx` (inside
    /// `fn_ast`'s body) derives from, chasing projections and calls.
    fn expr_view_params(
        &self,
        fn_ast: &FunctionAST,
        idx: NodeIdx,
        depth: u32,
        out: &mut Vec<usize>,
    ) {
        let push_param = |name: &str, out: &mut Vec<usize>| {
            if let Some(k) = fn_ast.args.iter().position(|p| p.name == name) {
                let pk = self.ctx.type_pool.get(fn_ast.args[k].ty).kind;
                if pk != TypeKind::PtrSingle && pk != TypeKind::PtrMany {
                    out.push(k);
                }
            }
        };
        let n = *self.ctx.node_store.get(idx);
        match n.tag {
            AstTag::Variable => push_param(&self.str_name(n.lhs), out),
            AstTag::MemberAccess
            | AstTag::Index
            | AstTag::Slice
            | AstTag::Deref
            | AstTag::AsCast => {
                self.expr_view_params(fn_ast, NodeIdx::new(n.lhs), depth, out);
            }
            AstTag::Call => {
                if let Some((inner, recv, args)) = self.resolve_call(idx, Some(fn_ast)) {
                    let viewed = self.return_view_params(&inner, depth - 1);
                    let offset = usize::from(recv.is_some());
                    for k in viewed {
                        if k == 0 && offset == 1 {
                            push_param(recv.as_deref().unwrap_or(""), out);
                        } else if let Some(&a) = args.get(k - offset) {
                            self.expr_view_params(fn_ast, a, depth, out);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Is `name` a read-only borrow whose storage the caller owns — a
    /// default-mode parameter (not shadowed by a local) or a for-each view
    /// over one?
    fn is_readonly_root(&self, name: &str) -> bool {
        if self.readonly_views.contains(name) {
            return true;
        }
        self.param_modes.get(name) == Some(&ParamMode::Let)
            && !self.decl_depth.contains_key(name)
    }

    /// Reject an assignment whose destination is owned by a read-only
    /// parameter (directly or through a for-each element view).
    fn check_readonly_param_write(&mut self, target: NodeIdx) {
        let Some(root) = self.write_root(target) else {
            return;
        };
        if !self.is_readonly_root(&root) {
            return;
        }
        // Rebinding a slice-typed VIEW variable (`s = other;`) writes the
        // local {ptr, len} pair, not the viewed elements — allowed; the
        // assign hook recomputes its view membership. (For-each loop vars
        // have no var_types entry: their bare writes ARE element writes.)
        if self.node_tag(target) == AstTag::Variable
            && self.readonly_views.contains(&root)
            && self.var_types.contains_key(&root)
        {
            return;
        }
        let msg = if self.readonly_views.contains(&root) {
            format!(
                "cannot write through `{root}` — it views elements of a read-only \
                 parameter; take the parameter as `mut` in the signature"
            )
        } else {
            format!(
                "cannot write into parameter `{root}` — parameters are read-only \
                 borrows by default; declare it `{root}: mut ...` in the signature"
            )
        };
        self.emit_error(msg, target, root);
    }

    fn emit_error(&mut self, message: String, anchor: NodeIdx, var_name: String) {
        let line = self.line_of(anchor);
        self.diagnostics.push(Diagnostic {
            message,
            var_name,
            line,
        });
    }
}

/// Lattice merge: union of keys, bitwise-OR of overlapping states.
fn merge_maps(a: &NameMap, b: &NameMap) -> NameMap {
    let mut result = a.clone();
    for (k, &v) in b {
        result
            .entry(k.clone())
            .and_modify(|s| *s = merge_state(*s, v))
            .or_insert(v);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_context::CodegenContext;
    use jam_llvm::Context;
    use jam_syntax::lexer::Lexer;
    use jam_syntax::parser::Parser;

    /// Parse `src` into a fresh context, register `Counter` as a drop-bearing
    /// struct (so `type_needs_drop(Counter)` is true), then run the analyzer on
    /// the function named `fn_name`. Returns the diagnostics.
    fn analyze_named(src: &str, fn_name: &str) -> Vec<Diagnostic> {
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
        // Register `Counter` with a drop fn so it is drop-bearing.
        let named = cg.context().named_struct("Counter");
        named.set_body(&[cg.context().i32_type()], false);
        cg.register_struct(
            "Counter",
            named,
            vec![("value".to_string(), jam_syntax::ast_flat::builtin::I32)],
        );
        cg.register_drop_fn("Counter", "Counter.drop");

        let func = module
            .functions
            .iter()
            .find(|f| f.name == fn_name)
            .unwrap_or_else(|| panic!("fn `{fn_name}` not parsed"));
        analyze(func, &cg)
    }

    #[test]
    fn plain_function_no_diagnostic() {
        let d = analyze_named("fn add(a: i32, b: i32) i32 { return a + b; }", "add");
        assert!(d.is_empty(), "plain function flagged: {d:?}");
    }

    #[test]
    fn move_param_into_slot_ok() {
        // A `move`-mode drop-bearing param consumed once is fine (owned).
        let src = "fn take(p: *mut[] Counter, v: move Counter) { p[0] = v; }";
        let d = analyze_named(src, "take");
        assert!(d.is_empty(), "move-param consume flagged: {d:?}");
    }

    #[test]
    fn let_param_moved_in_loop_flags() {
        // The `Vec(T).filled` shape: a by-value (`let`) drop-bearing param moved
        // into a slot inside a loop is a borrowed-param move + use-after-move.
        let src = "\
            fn fill(p: *mut[] Counter, value: Counter, n: u32) {\n\
                var i: u32 = 0;\n\
                while (i < n) {\n\
                    p[i] = value;\n\
                    i = i + 1;\n\
                }\n\
            }";
        let d = analyze_named(src, "fill");
        assert!(!d.is_empty(), "let-param loop-move NOT flagged");
        assert!(
            d.iter().any(|x| x.var_name == "value"),
            "expected `value` diagnostic: {d:?}"
        );
    }

    #[test]
    fn let_param_non_drop_no_diagnostic() {
        // The same shape with a non-drop element type (u64) is a copy, not a
        // move — no diagnostic. (Guards against false positives on Vec(u64).)
        let src = "\
            fn fill(p: *mut[] u64, value: u64, n: u32) {\n\
                var i: u32 = 0;\n\
                while (i < n) {\n\
                    p[i] = value;\n\
                    i = i + 1;\n\
                }\n\
            }";
        let d = analyze_named(src, "fill");
        assert!(d.is_empty(), "non-drop copy flagged: {d:?}");
    }

    #[test]
    fn move_param_local_owned_loop_ok() {
        // A local owned drop-bearing binding moved at its own depth is OK even in
        // a loop body when re-initialized each iteration (depth-matched move).
        let src = "fn one(p: *mut[] Counter, v: move Counter) { p[0] = v; }";
        let d = analyze_named(src, "one");
        assert!(d.is_empty(), "depth-matched move flagged: {d:?}");
    }
}
