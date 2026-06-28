/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam_core` — leaf types every other compiler crate depends on.
//!
//!   * [`diag`] — `SrcLoc`, `Diagnostic`, the `Diagnostics` accumulator and its
//!     byte-faithful renderer (ported from the C++ `diagnostics.{h,cpp}`).
//!   * [`index`] — the frozen `#[repr(transparent)]` arena-index newtypes
//!     (`NodeIdx`, `TypeIdx`, `DeclIndex`, `FunctionId`, …) shared by the AST
//!     arenas, decl table, modules, ownership, and the backend.
//!   * [`param_mode`] — the `ParamMode` enum (`Let`/`Mut`/`Move`), a shared leaf
//!     used by ast, decl, ownership, and jir.
//!
//! Pure std, no compiler-internal dependencies.

pub mod diag;
pub mod index;
pub mod param_mode;

pub use diag::{Diagnostic, Diagnostics, RefTraceGuard, Severity, SrcLoc, Trace, TraceKind};
pub use index::{
    DeclIndex, ExtraIdx, FunctionId, JirBlockRef, JirRef, ModuleId, NodeIdx, StringIdx, TypeIdx,
};
pub use param_mode::ParamMode;
