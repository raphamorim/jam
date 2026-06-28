/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Diagnostics — ported 1:1 from `src/diagnostics.{h,cpp}`.
//!
//! The renderer ([`Diagnostics::emit`]) is a **byte-for-byte** reproduction of
//! the C++ `emitOne`: the whole test suite asserts on stderr substrings, so the
//! 4-space-per-note-level indent, the ` at file:line` elision when `line == 0`,
//! the `error → note` severity demotion for attached notes, and the
//! `stable_sort` ordering are all load-bearing and must not drift.

use std::io;

/// Where in source a diagnostic is anchored. `(file, line)` only today; `line
/// == 0` and an empty `file` are sentinels the formatter branches on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SrcLoc {
    /// Display filename — an already-formatted relative path.
    pub file: String,
    /// 1-based source line; `0` means "no line" (suppresses the `:line` suffix).
    pub line: u32,
}

impl SrcLoc {
    pub fn new(file: impl Into<String>, line: u32) -> SrcLoc {
        SrcLoc {
            file: file.into(),
            line,
        }
    }
    /// A location with a file but no line.
    pub fn file_only(file: impl Into<String>) -> SrcLoc {
        SrcLoc {
            file: file.into(),
            line: 0,
        }
    }
    /// The empty location (no file, no line).
    pub fn none() -> SrcLoc {
        SrcLoc::default()
    }
}

/// Severity. The underlying `u8` ordering (`Error < Warning < Note`) IS the
/// sort key — do not reorder.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Severity {
    Error = 0,
    Warning = 1,
    Note = 2,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

/// The verb a trace frame renders with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TraceKind {
    /// "in instantiation of `…`" — a generic monomorphisation frame.
    #[default]
    Instantiation,
    /// "referenced by `…`" — a decl-dependency frame.
    Reference,
}

/// One frame of a reference / instantiation trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub loc: SrcLoc,
    /// e.g. `Vec(NoDefault).default`.
    pub decl: String,
    pub kind: TraceKind,
    /// Count of frames elided when the chain exceeded a limit (0 today).
    pub hidden: u32,
}

impl Trace {
    pub fn new(loc: SrcLoc, decl: impl Into<String>, kind: TraceKind) -> Trace {
        Trace {
            loc,
            decl: decl.into(),
            kind,
            hidden: 0,
        }
    }
}

/// One diagnostic emitted by any phase. `notes` carry secondary messages
/// (possibly in another file); `reference_trace` records the
/// generic-instantiation chain that led to an astgen failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub loc: SrcLoc,
    pub severity: Severity,
    pub message: String,
    pub notes: Vec<Diagnostic>,
    pub reference_trace: Vec<Trace>,
}

impl Diagnostic {
    pub fn new(severity: Severity, loc: SrcLoc, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            loc,
            severity,
            message: message.into(),
            notes: Vec::new(),
            reference_trace: Vec::new(),
        }
    }
}

