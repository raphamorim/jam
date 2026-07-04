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

