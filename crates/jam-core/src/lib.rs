/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam_core` — leaf types shared by every compiler crate: diagnostics
//! ([`diag`]), arena-index newtypes ([`index`]), and [`param_mode`].
//! Pure std, no compiler-internal dependencies.

pub mod diag;
pub mod index;
pub mod param_mode;

pub use diag::{Diagnostic, Diagnostics, RefTraceGuard, Severity, SrcLoc, Trace, TraceKind};
pub use index::{
    DeclIndex, ExtraIdx, FunctionId, JirBlockRef, JirRef, ModuleId, NodeIdx, StringIdx, TypeIdx,
};
pub use param_mode::ParamMode;
