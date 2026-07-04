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

