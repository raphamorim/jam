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

/// RAII guard for a reference-trace stack: push on construct, pop on drop.
/// Faithful to the C++ `RefTraceFrame`, but the stack is owned elsewhere
/// (codegen) and borrowed `&mut` for the guard's lifetime — Rust makes the
/// push/pop balance and the borrow exclusivity checkable, which the C++ raw
/// reference could not.
pub struct RefTraceGuard<'a> {
    stack: &'a mut Vec<Trace>,
}

impl<'a> RefTraceGuard<'a> {
    pub fn push(stack: &'a mut Vec<Trace>, frame: Trace) -> RefTraceGuard<'a> {
        stack.push(frame);
        RefTraceGuard { stack }
    }
}

impl Drop for RefTraceGuard<'_> {
    fn drop(&mut self) {
        self.stack.pop();
    }
}

/// Per-compilation accumulator. Phases push; the driver emits at the end.
/// Nothing aborts on first error — passes push and continue so the user sees
/// the whole batch.
#[derive(Default, Debug)]
pub struct Diagnostics {
    diags: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new() -> Diagnostics {
        Diagnostics::default()
    }

    pub fn error(&mut self, loc: SrcLoc, message: impl Into<String>) {
        self.diags
            .push(Diagnostic::new(Severity::Error, loc, message));
    }

    pub fn warning(&mut self, loc: SrcLoc, message: impl Into<String>) {
        self.diags
            .push(Diagnostic::new(Severity::Warning, loc, message));
    }

    pub fn error_with_notes(
        &mut self,
        loc: SrcLoc,
        message: impl Into<String>,
        notes: Vec<Diagnostic>,
    ) {
        let mut d = Diagnostic::new(Severity::Error, loc, message);
        d.notes = notes;
        self.diags.push(d);
    }

    pub fn error_with_trace(
        &mut self,
        loc: SrcLoc,
        message: impl Into<String>,
        reference_trace: Vec<Trace>,
    ) {
        let mut d = Diagnostic::new(Severity::Error, loc, message);
        d.reference_trace = reference_trace;
        self.diags.push(d);
    }

    /// Push a fully-built diagnostic (notes/trace already assembled).
    pub fn push(&mut self, diag: Diagnostic) {
        self.diags.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diags.iter().any(|d| d.severity == Severity::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diags
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    pub fn all(&self) -> &[Diagnostic] {
        &self.diags
    }

    pub fn len(&self) -> usize {
        self.diags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.diags.is_empty()
    }

    /// Roll back to a previously observed size — used by conditional
    /// instantiation to withdraw a failed attempt's diagnostics. Mirrors the
    /// C++ `truncateTo` (`resize` down only).
    pub fn truncate_to(&mut self, n: usize) {
        if n < self.diags.len() {
            self.diags.truncate(n);
        }
    }

    /// Render the report to `out`. Stable-sorted by `(file, line, severity)`;
    /// errors before warnings at the same location.
    pub fn emit<W: io::Write>(&self, out: &mut W) -> io::Result<()> {
        let mut sorted: Vec<&Diagnostic> = self.diags.iter().collect();
        // Stable sort — same-(file,line,severity) diagnostics keep insertion
        // order (the C++ uses std::stable_sort; Rust's sort_by is stable).
        sorted.sort_by(|a, b| {
            a.loc
                .file
                .cmp(&b.loc.file)
                .then(a.loc.line.cmp(&b.loc.line))
                .then(a.severity.cmp(&b.severity))
        });
        for d in sorted {
            emit_one(out, d, 0)?;
        }
        Ok(())
    }

    /// Convenience: render to a `String` (for tests and substring checks).
    pub fn render_to_string(&self) -> String {
        let mut buf = Vec::new();
        self.emit(&mut buf)
            .expect("writing diagnostics to a Vec cannot fail");
        String::from_utf8(buf).expect("diagnostic text is valid UTF-8")
    }
}

