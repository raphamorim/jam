/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Compile-time evaluation (`comp` / `cfn`). Folds AST expression nodes to
//! [`ComptimeValue`]s and interprets `comp` statement bodies. Ported from
//! `src/comptime.{h,cpp}`.
//!
//! Failure modes (depends on a runtime value, unsupported operator, type
//! mismatch) all surface as [`ComptimeValue::None`] — the evaluator never
//! panics. Callers that *require* a fold call [`ComptimeEvaluator::eval_required`],
//! which pushes a diagnostic on failure.
//!
//! Two divergences from the C++, both behaviour-preserving:
//!   * The C++ stores the call context (emitter / resolver / diags / loc) as
//!     `mutable` fields on a `const` evaluator. Rust threads an explicit
//!     [`CompCtx`] through the eval/exec methods instead — same data, no interior
//!     mutability. The C++'s separate `Diagnostics&` statement param is unified
//!     into `CompCtx::diags`.
//!   * `@isDarwin()` & friends query the host OS. The C++ calls
//!     `Target::getHostTarget()` (an LLVM call) inline; here the host OS is
//!     injected via [`CompCtx::host_os`] so the evaluator stays LLVM-free and
//!     unit-testable. The driver supplies it from `jam_llvm::default_target_triple`.
//!
//! All integer arithmetic uses wrapping ops to match C++ unsigned wraparound /
//! 2's-complement semantics (and to avoid Rust's debug overflow panics).

use std::collections::HashMap;

use float128::quad_to_target_as_double;
use jam_core::diag::{Diagnostics, SrcLoc};
use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};
use jam_syntax::ast_flat::{AstNode, AstTag, BinOp, NodeStore, StringPool, TypePool, UnaryOp};

use crate::target::Os;

/// A value known at compile time. `None` is the failure / "not foldable" state.
#[derive(Clone, Debug, PartialEq)]
pub enum ComptimeValue {
    None,
    Int {
        bits: u64,
        width: u16,
        is_signed: bool,
    },
    Float {
        value: f64,
        width: u16,
    },
    Bool(bool),
    Str(StringIdx),
    Type(TypeIdx),
    Aggregate(Vec<ComptimeValue>),
}

impl ComptimeValue {
    pub fn is_none(&self) -> bool {
        matches!(self, ComptimeValue::None)
    }
    pub fn is_int(&self) -> bool {
        matches!(self, ComptimeValue::Int { .. })
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, ComptimeValue::Bool(_))
    }
    pub fn is_str(&self) -> bool {
        matches!(self, ComptimeValue::Str(_))
    }
    pub fn is_type(&self) -> bool {
        matches!(self, ComptimeValue::Type(_))
    }

    /// Sign-extend an Int payload to `i64` from its bit-width so `i8(-1)`
    /// reads back as -1 (not 255). Non-Int returns 0.
    pub fn as_i64(&self) -> i64 {
        match *self {
            ComptimeValue::Int {
                bits,
                width,
                is_signed,
            } => {
                let mut b = bits;
                if is_signed && width < 64 {
                    let sign_bit = 1u64 << (width - 1);
                    if b & sign_bit != 0 {
                        let mask = !((1u64 << width) - 1);
                        b |= mask;
                    }
                }
                b as i64
            }
            _ => 0,
        }
    }

    /// Mask an Int payload to its bit-width so over-wide patterns can't leak.
    /// `i8(-1)` and `u8(255)` both return 0xFF. Non-Int returns 0.
    pub fn as_u64(&self) -> u64 {
        match *self {
            ComptimeValue::Int { bits, width, .. } => {
                if width >= 64 {
                    bits
                } else {
                    bits & ((1u64 << width) - 1)
                }
            }
            _ => 0,
        }
    }
}

/// A lexically-scoped map of name -> [`ComptimeValue`]. The C++ chains scope
/// objects by parent pointer; here a stack of frames models the same nesting
/// (push a frame on block entry, pop on exit), which avoids self-referential
/// borrows while preserving the bind/set/lookup semantics exactly.
#[derive(Clone, Debug, Default)]
pub struct ComptimeScope {
    frames: Vec<HashMap<String, ComptimeValue>>,
}

impl ComptimeScope {
    pub fn new() -> ComptimeScope {
        ComptimeScope {
            frames: vec![HashMap::new()],
        }
    }

    /// Enter a nested scope (mirrors `ComptimeScope inner(&scope)`).
    pub fn push(&mut self) {
        self.frames.push(HashMap::new());
    }

    /// Leave the innermost scope.
    pub fn pop(&mut self) {
        self.frames.pop();
    }

    /// Declare a new local in the innermost frame; shadows any parent binding.
    pub fn bind(&mut self, name: impl Into<String>, value: ComptimeValue) {
        self.frames
            .last_mut()
            .expect("scope has at least one frame")
            .insert(name.into(), value);
    }

    /// Walk inner->outer; mutate the frame where `name` was first bound. Returns
    /// false if `name` isn't bound anywhere.
    pub fn set(&mut self, name: &str, value: ComptimeValue) -> bool {
        for frame in self.frames.iter_mut().rev() {
            if let Some(slot) = frame.get_mut(name) {
                *slot = value;
                return true;
            }
        }
        false
    }

    /// Read; walks inner->outer. `None` if unbound.
    pub fn lookup(&self, name: &str) -> Option<&ComptimeValue> {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get(name) {
                return Some(v);
            }
        }
        None
    }

    /// Copy every visible binding (outer->inner, inner shadows) into `dst`'s
    /// innermost frame. Lets a per-module seed cache replay a computed-const
    /// fixpoint into a fresh scope without re-evaluating initializers.
    pub fn copy_bindings_into(&self, dst: &mut ComptimeScope) {
        for frame in &self.frames {
            for (k, v) in frame {
                dst.bind(k.clone(), v.clone());
            }
        }
    }
}

/// Outcome of executing a statement or block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecResult {
    /// Executed normally; keep going.
    Continue,
    /// A `return` was hit; caller propagates upward.
    Returned,
    /// A loop / recursive call exceeded the iteration cap.
    IterationCap,
    /// Unrecoverable failure (diagnostic already pushed).
    Error,
}

/// Sink for `@`-emit intrinsics inside a `cfn` body.
pub trait CompEmitter {
    fn handle_at_call(
        &mut self,
        name: &str,
        args: &[ComptimeValue],
        diags: &mut Diagnostics,
        loc: &SrcLoc,
    ) -> ExecResult;
}

/// Resolves a comptime call `name(args)` to a value (how one `cfn` calls
/// another, and how a `cfn` appears in a `[N]u8` length position). Returns
/// `None` when `name` isn't comptime-evaluable or the body fails to fold.
pub trait CompCallResolver {
    fn resolve_call(
        &mut self,
        name: &str,
        args: &[ComptimeValue],
        diags: &mut Diagnostics,
        loc: &SrcLoc,
    ) -> ComptimeValue;
}

/// The call context threaded through evaluation: optional emit/resolve hooks,
/// the diagnostics sink, the call-site location, and the host OS for target
/// predicates. Replaces the C++ evaluator's `mutable` context fields.
pub struct CompCtx<'c> {
    pub resolver: Option<&'c mut dyn CompCallResolver>,
    pub emitter: Option<&'c mut dyn CompEmitter>,
    pub diags: Option<&'c mut Diagnostics>,
    pub loc: SrcLoc,
    pub host_os: Os,
}

impl<'c> CompCtx<'c> {
    /// A context with no hooks and no diagnostics sink (peephole folding).
    pub fn bare(host_os: Os) -> CompCtx<'c> {
        CompCtx {
            resolver: None,
            emitter: None,
            diags: None,
            loc: SrcLoc::new("", 0),
            host_os,
        }
    }

    fn push_err(&mut self, msg: impl Into<String>) {
        let loc = self.loc.clone();
        if let Some(d) = self.diags.as_deref_mut() {
            d.error(loc, msg);
        }
    }
}

/// Default iteration cap for `comp while` / recursive `cfn`.
pub const DEFAULT_ITER_CAP: u32 = 10_000;

/// Folds AST expression nodes to compile-time values and interprets `comp`
/// statement bodies. Captures the shared pools at construction; reads bindings
/// from a caller-supplied scope per invocation.
pub struct ComptimeEvaluator<'a> {
    nodes: &'a NodeStore,
    strings: &'a StringPool,
    #[allow(dead_code)] // reserved for type-aware folding (parity with C++)
    types: &'a TypePool,
}

impl<'a> ComptimeEvaluator<'a> {
    pub fn new(
        nodes: &'a NodeStore,
        strings: &'a StringPool,
        types: &'a TypePool,
    ) -> ComptimeEvaluator<'a> {
        ComptimeEvaluator {
            nodes,
            strings,
            types,
        }
    }

    fn str_text(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.strings.get(StringIdx::new(id))).into_owned()
    }

    /// Try to fold `expr` to a value. Returns `None` on any failure.
    pub fn eval(&self, expr: NodeIdx, scope: &ComptimeScope, ctx: &mut CompCtx) -> ComptimeValue {
        if expr.is_none() {
            return ComptimeValue::None;
        }
        let n = *self.nodes.get(expr);
        match n.tag {
            AstTag::NumberLit => self.eval_number_lit(&n),
            AstTag::BoolLit => ComptimeValue::Bool(n.lhs != 0),
            AstTag::StringLit => ComptimeValue::Str(StringIdx::new(n.lhs)),
            AstTag::Variable => self.eval_variable(&n, scope),
            AstTag::UnaryOp => self.eval_unary_op(&n, scope, ctx),
            AstTag::BinaryOp => self.eval_binary_op(&n, scope, ctx),
            AstTag::Index => self.eval_index(&n, scope, ctx),
            AstTag::MemberAccess => self.eval_member_access(&n, scope, ctx),
            AstTag::AtCall => self.eval_at_call(&n, scope, ctx),
            AstTag::Call => self.eval_call(&n, scope, ctx),
            // Operator/construct we don't fold yet — None keeps optional-fold
            // callers silent; eval_required turns it into a diagnostic.
            _ => ComptimeValue::None,
        }
    }

    /// Same, but pushes a diagnostic + returns `None` when the expression can't
    /// be folded.
    pub fn eval_required(
        &self,
        expr: NodeIdx,
        scope: &ComptimeScope,
        ctx: &mut CompCtx,
    ) -> ComptimeValue {
        let v = self.eval(expr, scope, ctx);
        if v.is_none() {
            ctx.push_err("expression cannot be evaluated at compile time");
        }
        v
    }

    fn eval_number_lit(&self, n: &AstNode) -> ComptimeValue {
        let is_neg = n.flags & 1 != 0;
        let is_float = n.flags & 2 != 0;
        if is_float {
            let mut v = if n.flags & 4 != 0 {
                let ei = n.lhs;
                let quad = [
                    self.nodes.get_extra(ExtraIdx::new(ei)),
                    self.nodes.get_extra(ExtraIdx::new(ei + 1)),
                    self.nodes.get_extra(ExtraIdx::new(ei + 2)),
                    self.nodes.get_extra(ExtraIdx::new(ei + 3)),
                ];
                quad_to_target_as_double(&quad, false)
            } else {
                let bits = (n.lhs as u64) | ((n.rhs as u64) << 32);
                f64::from_bits(bits)
            };
            if is_neg {
                v = -v;
            }
            return ComptimeValue::Float {
                value: v,
                width: 64,
            };
        }
        let bits = (n.lhs as u64) | ((n.rhs as u64) << 32);
        // Default integer width: u64 (or i64 if negative); kept full-width to
        // preserve precision during folding.
        if is_neg {
            let signed_bits = (bits as i64).wrapping_neg() as u64;
            ComptimeValue::Int {
                bits: signed_bits,
                width: 64,
                is_signed: true,
            }
        } else {
            ComptimeValue::Int {
                bits,
                width: 64,
                is_signed: false,
            }
        }
    }

    fn eval_variable(&self, n: &AstNode, scope: &ComptimeScope) -> ComptimeValue {
        let name = self.str_text(n.lhs);
        scope.lookup(&name).cloned().unwrap_or(ComptimeValue::None)
    }

    fn eval_unary_op(
        &self,
        n: &AstNode,
        scope: &ComptimeScope,
        ctx: &mut CompCtx,
    ) -> ComptimeValue {
        let v = self.eval(NodeIdx::new(n.lhs), scope, ctx);
        if v.is_none() {
            return v;
        }
        let op = unary_op_from_u8(n.op);
        match op {
            UnaryOp::Neg => match v {
                ComptimeValue::Int { width, .. } => {
                    let neg = (v.as_u64() as i64).wrapping_neg() as u64;
                    ComptimeValue::Int {
                        bits: neg,
                        width,
                        is_signed: true,
                    }
                }
                ComptimeValue::Float { value, width } => ComptimeValue::Float {
                    value: -value,
                    width,
                },
                _ => ComptimeValue::None,
            },
            UnaryOp::LogNot => match v {
                ComptimeValue::Bool(b) => ComptimeValue::Bool(!b),
                _ => ComptimeValue::None,
            },
            UnaryOp::BitNot => match v {
                ComptimeValue::Int {
                    bits,
                    width,
                    is_signed,
                } => ComptimeValue::Int {
                    bits: !bits,
                    width,
                    is_signed,
                },
                _ => ComptimeValue::None,
            },
            UnaryOp::Invalid => ComptimeValue::None,
        }
    }

    fn eval_binary_op(
        &self,
        n: &AstNode,
        scope: &ComptimeScope,
        ctx: &mut CompCtx,
    ) -> ComptimeValue {
        let op = bin_op_from_u8(n.op);
        let lhs = NodeIdx::new(n.lhs);
        let rhs = NodeIdx::new(n.rhs);

        // Short-circuit LogAnd / LogOr.
        if op == BinOp::LogAnd || op == BinOp::LogOr {
            let l = self.eval(lhs, scope, ctx);
            let lb = match l {
                ComptimeValue::Bool(b) => b,
                _ => return ComptimeValue::None,
            };
            if op == BinOp::LogAnd && !lb {
                return ComptimeValue::Bool(false);
            }
            if op == BinOp::LogOr && lb {
                return ComptimeValue::Bool(true);
            }
            let r = self.eval(rhs, scope, ctx);
            return match r {
                ComptimeValue::Bool(b) => ComptimeValue::Bool(b),
                _ => ComptimeValue::None,
            };
        }

        let l = self.eval(lhs, scope, ctx);
        if l.is_none() {
            return l;
        }
        let r = self.eval(rhs, scope, ctx);
        if r.is_none() {
            return r;
        }

        // Integer arithmetic / bitwise / comparison.
        if let (
            ComptimeValue::Int {
                bits: a,
                width: lw0,
                is_signed: ls,
            },
            ComptimeValue::Int {
                bits: b,
                width: rw0,
                is_signed: rs,
            },
        ) = (&l, &r)
        {
            let (a, b) = (*a, *b);
            let (mut lw, mut rw) = (*lw0, *rw0);
            let (ls, rs) = (*ls, *rs);
            let is_comparison = matches!(
                op,
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            );
            if lw != rw {
                if !is_comparison || ls != rs {
                    return ComptimeValue::None;
                }
                // Widen the narrower operand (comparisons only).
                if lw < rw {
                    lw = rw;
                } else {
                    rw = lw;
                }
            }
            if ls != rs {
                return ComptimeValue::None;
            }
            let _ = rw;
            let w = lw;
            let sgn = ls;
            let mk = |bits: u64| ComptimeValue::Int {
                bits,
                width: w,
                is_signed: sgn,
            };
            return match op {
                BinOp::Add => mk(a.wrapping_add(b)),
                BinOp::Sub => mk(a.wrapping_sub(b)),
                BinOp::Mul => mk(a.wrapping_mul(b)),
                BinOp::Div => {
                    if b == 0 {
                        ComptimeValue::None
                    } else if sgn {
                        mk((a as i64).wrapping_div(b as i64) as u64)
                    } else {
                        mk(a.wrapping_div(b))
                    }
                }
                BinOp::Mod => {
                    if b == 0 {
                        ComptimeValue::None
                    } else if sgn {
                        mk((a as i64).wrapping_rem(b as i64) as u64)
                    } else {
                        mk(a.wrapping_rem(b))
                    }
                }
                BinOp::BitAnd => mk(a & b),
                BinOp::BitOr => mk(a | b),
                BinOp::BitXor => mk(a ^ b),
                BinOp::Shl => mk(a.wrapping_shl(b as u32)),
                BinOp::Shr => {
                    if sgn {
                        mk((a as i64).wrapping_shr(b as u32) as u64)
                    } else {
                        mk(a.wrapping_shr(b as u32))
                    }
                }
                BinOp::Eq => ComptimeValue::Bool(a == b),
                BinOp::Ne => ComptimeValue::Bool(a != b),
                BinOp::Lt => ComptimeValue::Bool(if sgn { (a as i64) < (b as i64) } else { a < b }),
                BinOp::Le => ComptimeValue::Bool(if sgn {
                    (a as i64) <= (b as i64)
                } else {
                    a <= b
                }),
                BinOp::Gt => ComptimeValue::Bool(if sgn { (a as i64) > (b as i64) } else { a > b }),
                BinOp::Ge => ComptimeValue::Bool(if sgn {
                    (a as i64) >= (b as i64)
                } else {
                    a >= b
                }),
                _ => ComptimeValue::None,
            };
        }

        // String equality (format-string field-name matching).
        if let (ComptimeValue::Str(ls), ComptimeValue::Str(rs)) = (&l, &r) {
            let eq = ls == rs;
            return match op {
                BinOp::Eq => ComptimeValue::Bool(eq),
                BinOp::Ne => ComptimeValue::Bool(!eq),
                _ => ComptimeValue::None,
            };
        }

        // Bool equality.
        if let (ComptimeValue::Bool(lb), ComptimeValue::Bool(rb)) = (&l, &r) {
            return match op {
                BinOp::Eq => ComptimeValue::Bool(lb == rb),
                BinOp::Ne => ComptimeValue::Bool(lb != rb),
                _ => ComptimeValue::None,
            };
        }

        // Type equality — the linchpin of `comp if (@TypeOf(arg) == i32)`.
        if let (ComptimeValue::Type(lt), ComptimeValue::Type(rt)) = (&l, &r) {
            let eq = lt == rt;
            return match op {
                BinOp::Eq => ComptimeValue::Bool(eq),
                BinOp::Ne => ComptimeValue::Bool(!eq),
                _ => ComptimeValue::None,
            };
        }

        ComptimeValue::None
    }

    fn eval_member_access(
        &self,
        n: &AstNode,
        scope: &ComptimeScope,
        ctx: &mut CompCtx,
    ) -> ComptimeValue {
        let member = self.str_text(n.rhs);
        let base = self.eval(NodeIdx::new(n.lhs), scope, ctx);
        if base.is_none() {
            return base;
        }
        if let ComptimeValue::Str(s) = base
            && member == "length"
        {
            let len = self.strings.get(s).len() as u64;
            return ComptimeValue::Int {
                bits: len,
                width: 64,
                is_signed: false,
            };
        }
        ComptimeValue::None
    }

    fn eval_index(&self, n: &AstNode, scope: &ComptimeScope, ctx: &mut CompCtx) -> ComptimeValue {
        let base = self.eval(NodeIdx::new(n.lhs), scope, ctx);
        if base.is_none() {
            return base;
        }
        let idx = self.eval(NodeIdx::new(n.rhs), scope, ctx);
        if !idx.is_int() {
            return ComptimeValue::None;
        }
        let i = idx.as_u64();
        match base {
            ComptimeValue::Str(s) => {
                let bytes = self.strings.get(s);
                if i as usize >= bytes.len() {
                    return ComptimeValue::None;
                }
                ComptimeValue::Int {
                    bits: bytes[i as usize] as u64,
                    width: 8,
                    is_signed: false,
                }
            }
            ComptimeValue::Aggregate(fields) => {
                if i as usize >= fields.len() {
                    return ComptimeValue::None;
                }
                fields[i as usize].clone()
            }
            _ => ComptimeValue::None,
        }
    }

    fn eval_call(&self, n: &AstNode, scope: &ComptimeScope, ctx: &mut CompCtx) -> ComptimeValue {
        // Only direct calls fold (flags bit 0 = indirect method on a runtime
        // value). Direct form: lhs = callee StringIdx, rhs = ExtraIdx ->
        // [argCount, arg0, ...].
        if n.flags & 1 != 0 {
            return ComptimeValue::None;
        }
        if ctx.resolver.is_none() {
            return ComptimeValue::None;
        }
        let name = self.str_text(n.lhs);
        let extra = n.rhs;
        let arg_count = self.nodes.get_extra(ExtraIdx::new(extra));
        let mut arg_vals = Vec::with_capacity(arg_count as usize);
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(self.nodes.get_extra(ExtraIdx::new(extra + 1 + i)));
            let v = self.eval(arg_idx, scope, ctx);
            if v.is_none() {
                ctx.push_err(format!(
                    "argument #{i} to comptime call `{name}` is not a compile-time constant"
                ));
                return ComptimeValue::None;
            }
            arg_vals.push(v);
        }
        // Now invoke the resolver with disjoint borrows of resolver + diags.
        let loc = ctx.loc.clone();
        let mut dummy = Diagnostics::new();
        let resolver = ctx.resolver.as_deref_mut().expect("resolver present");
        let diags = ctx.diags.as_deref_mut().unwrap_or(&mut dummy);
        resolver.resolve_call(&name, &arg_vals, diags, &loc)
    }

    fn eval_at_call(&self, n: &AstNode, scope: &ComptimeScope, ctx: &mut CompCtx) -> ComptimeValue {
        // No-arg / type-arg intrinsics (flags bit 0 clear). The target-OS
        // predicates fold to a comptime bool so `comp if (@isDarwin())` works.
        if n.flags & 1 == 0 {
            let iname = self.str_text(n.lhs);
            if iname == "isDarwin"
                || iname == "isLinux"
                || iname == "isWindows"
                || iname == "isUnix"
            {
                let os = ctx.host_os;
                let v = match iname.as_str() {
                    "isDarwin" => os == Os::MacOs,
                    "isLinux" => os == Os::Linux,
                    "isWindows" => os == Os::Windows,
                    // isUnix
                    _ => os == Os::MacOs || os == Os::Linux || os == Os::FreeBsd,
                };
                return ComptimeValue::Bool(v);
            }
            return ComptimeValue::None;
        }

        // Expr-arg multi-form: rhs = ExtraIdx -> [argCount, arg0, ...].
        let extra = n.rhs;
        let arg_count = self.nodes.get_extra(ExtraIdx::new(extra));
        let mut arg_vals = Vec::with_capacity(arg_count as usize);
        for i in 0..arg_count {
            let arg_idx = NodeIdx::new(self.nodes.get_extra(ExtraIdx::new(extra + 1 + i)));
            let v = self.eval(arg_idx, scope, ctx);
            if v.is_none() {
                ctx.push_err(format!(
                    "@-emit argument must be a compile-time constant (arg #{i})"
                ));
                return ComptimeValue::None;
            }
            arg_vals.push(v);
        }

        if ctx.emitter.is_none() {
            // No emitter installed — running outside a cfn dispatcher context.
            return ComptimeValue::None;
        }
        let name = self.str_text(n.lhs);
        let loc = ctx.loc.clone();
        let mut dummy = Diagnostics::new();
        let emitter = ctx.emitter.as_deref_mut().expect("emitter present");
        let diags = ctx.diags.as_deref_mut().unwrap_or(&mut dummy);
        // Emit-style intrinsics return Continue on success; errors surface via
        // pushed diagnostics. @-emit produces side effects, not values.
        let _ = emitter.handle_at_call(&name, &arg_vals, diags, &loc);
        ComptimeValue::None
    }

    /// Execute a statement, mutating `scope` for var-decls/assignments and
    /// recursing for control flow. `iter_counter` is incremented per loop
    /// iteration; exceeding `iter_cap` returns [`ExecResult::IterationCap`].
    /// `out_return` receives a `return EXPR;` value if one fires.
    #[allow(clippy::too_many_arguments)]
    pub fn exec_stmt(
        &self,
        stmt: NodeIdx,
        scope: &mut ComptimeScope,
        iter_counter: &mut u32,
        iter_cap: u32,
        out_return: &mut ComptimeValue,
        ctx: &mut CompCtx,
    ) -> ExecResult {
        if stmt.is_none() {
            return ExecResult::Continue;
        }
        let n = *self.nodes.get(stmt);
        match n.tag {
            AstTag::VarDecl => {
                // extra: [name StringIdx, type TypeIdx, init NodeIdx]
                let extra = n.lhs;
                let name_id = self.nodes.get_extra(ExtraIdx::new(extra));
                let init_idx = NodeIdx::new(self.nodes.get_extra(ExtraIdx::new(extra + 2)));
                let v = self.eval(init_idx, scope, ctx);
                if v.is_none() {
                    ctx.push_err("comp var-decl initializer must be a compile-time-known value");
                    return ExecResult::Error;
                }
                scope.bind(self.str_text(name_id), v);
                ExecResult::Continue
            }
            AstTag::Assign => {
                let target = *self.nodes.get(NodeIdx::new(n.lhs));
                if target.tag != AstTag::Variable {
                    ctx.push_err(
                        "comp assignment target must be a bare variable (member-access / index \
                         targets are not supported in v1)",
                    );
                    return ExecResult::Error;
                }
                let name = self.str_text(target.lhs);
                let v = self.eval(NodeIdx::new(n.rhs), scope, ctx);
                if v.is_none() {
                    ctx.push_err("comp assignment value must be a compile-time-known value");
                    return ExecResult::Error;
                }
                if !scope.set(&name, v) {
                    ctx.push_err(format!(
                        "assignment to undeclared variable `{name}` (declare with `var` first)"
                    ));
                    return ExecResult::Error;
                }
                ExecResult::Continue
            }
            AstTag::IfNode => {
                let extra = n.rhs;
                let then_count = self.nodes.get_extra(ExtraIdx::new(extra));
                let else_count = self.nodes.get_extra(ExtraIdx::new(extra + 1));
                let c = self.eval(NodeIdx::new(n.lhs), scope, ctx);
                let cb = match c {
                    ComptimeValue::Bool(b) => b,
                    _ => {
                        ctx.push_err("comp `if` condition must fold to bool");
                        return ExecResult::Error;
                    }
                };
                let stmts: Vec<NodeIdx> = if cb {
                    (0..then_count)
                        .map(|i| NodeIdx::new(self.nodes.get_extra(ExtraIdx::new(extra + 2 + i))))
                        .collect()
                } else {
                    (0..else_count)
                        .map(|i| {
                            NodeIdx::new(
                                self.nodes
                                    .get_extra(ExtraIdx::new(extra + 2 + then_count + i)),
                            )
                        })
                        .collect()
                };
                scope.push();
                let r = self.exec_block(&stmts, scope, iter_counter, iter_cap, out_return, ctx);
                scope.pop();
                r
            }
            AstTag::WhileNode => {
                let extra = n.rhs;
                let body_count = self.nodes.get_extra(ExtraIdx::new(extra));
                let body: Vec<NodeIdx> = (0..body_count)
                    .map(|i| NodeIdx::new(self.nodes.get_extra(ExtraIdx::new(extra + 1 + i))))
                    .collect();
                loop {
                    let c = self.eval(NodeIdx::new(n.lhs), scope, ctx);
                    let cb = match c {
                        ComptimeValue::Bool(b) => b,
                        _ => {
                            ctx.push_err("comp `while` condition must fold to bool");
                            return ExecResult::Error;
                        }
                    };
                    if !cb {
                        break;
                    }
                    *iter_counter += 1;
                    if *iter_counter > iter_cap {
                        ctx.push_err(format!(
                            "comp evaluation iteration cap ({iter_cap}) exceeded — possible \
                             infinite loop"
                        ));
                        return ExecResult::IterationCap;
                    }
                    scope.push();
                    let r = self.exec_block(&body, scope, iter_counter, iter_cap, out_return, ctx);
                    scope.pop();
                    if r != ExecResult::Continue {
                        return r;
                    }
                }
                ExecResult::Continue
            }
            AstTag::Return => {
                let val_idx = NodeIdx::new(n.lhs);
                if val_idx.is_none() {
                    *out_return = ComptimeValue::None;
                } else {
                    *out_return = self.eval(val_idx, scope, ctx);
                    if out_return.is_none() {
                        ctx.push_err("comp `return` expression must fold to a value");
                        return ExecResult::Error;
                    }
                }
                ExecResult::Returned
            }
            _ => {
                // Expression statement: evaluate and discard.
                let _ = self.eval(stmt, scope, ctx);
                ExecResult::Continue
            }
        }
    }

    /// Execute a sequence of statements; returns the first non-`Continue` result.
    #[allow(clippy::too_many_arguments)]
    pub fn exec_block(
        &self,
        stmts: &[NodeIdx],
        scope: &mut ComptimeScope,
        iter_counter: &mut u32,
        iter_cap: u32,
        out_return: &mut ComptimeValue,
        ctx: &mut CompCtx,
    ) -> ExecResult {
        for &s in stmts {
            let r = self.exec_stmt(s, scope, iter_counter, iter_cap, out_return, ctx);
            if r != ExecResult::Continue {
                return r;
            }
        }
        ExecResult::Continue
    }
}

fn bin_op_from_u8(op: u8) -> BinOp {
    match op {
        1 => BinOp::Add,
        2 => BinOp::Sub,
        3 => BinOp::Mul,
        4 => BinOp::Div,
        5 => BinOp::Mod,
        6 => BinOp::BitAnd,
        7 => BinOp::BitOr,
        8 => BinOp::BitXor,
        9 => BinOp::Shl,
        10 => BinOp::Shr,
        11 => BinOp::LogAnd,
        12 => BinOp::LogOr,
        13 => BinOp::Eq,
        14 => BinOp::Ne,
        15 => BinOp::Lt,
        16 => BinOp::Le,
        17 => BinOp::Gt,
        18 => BinOp::Ge,
        _ => BinOp::Invalid,
    }
}

fn unary_op_from_u8(op: u8) -> UnaryOp {
    match op {
        1 => UnaryOp::Neg,
        2 => UnaryOp::LogNot,
        3 => UnaryOp::BitNot,
        _ => UnaryOp::Invalid,
    }
}

