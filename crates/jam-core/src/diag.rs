/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Diagnostics: `SrcLoc`, `Diagnostic`, the `Diagnostics` accumulator and its
//! renderer. The output format is load-bearing — the test suite asserts on
//! stderr substrings — so the 4-space note indent, `:line` elision, note
//! severity demotion, and stable sort order must not drift.

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
    /// instantiation to withdraw a failed attempt's diagnostics. Shrinks only.
    pub fn truncate_to(&mut self, n: usize) {
        if n < self.diags.len() {
            self.diags.truncate(n);
        }
    }

    /// Render the report to `out`. Stable-sorted by `(file, line, severity)`;
    /// errors before warnings at the same location.
    pub fn emit<W: io::Write>(&self, out: &mut W) -> io::Result<()> {
        let mut sorted: Vec<&Diagnostic> = self.diags.iter().collect();
        // sort_by is stable: same-(file,line,severity) diagnostics keep
        // insertion order.
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

fn emit_one<W: io::Write>(out: &mut W, d: &Diagnostic, indent: usize) -> io::Result<()> {
    let pad = " ".repeat(indent);
    write!(out, "{pad}")?;
    if !d.loc.file.is_empty() {
        write!(out, "{}:", d.loc.file)?;
        if d.loc.line > 0 {
            write!(out, "{}:", d.loc.line)?;
        }
        write!(out, " ")?;
    }
    writeln!(out, "{}: {}", d.severity.label(), d.message)?;

    for note in &d.notes {
        // Attached notes inherit Note severity when the caller left them Error,
        // so an attached note renders as `note:` not `error:`.
        let mut nd = note.clone();
        if nd.severity == Severity::Error {
            nd.severity = Severity::Note;
        }
        emit_one(out, &nd, indent + 4)?;
    }

    for frame in &d.reference_trace {
        let verb = match frame.kind {
            TraceKind::Reference => "referenced by",
            TraceKind::Instantiation => "in instantiation of",
        };
        write!(out, "{pad}    note: {verb} `{}`", frame.decl)?;
        if !frame.loc.file.is_empty() && frame.loc.line > 0 {
            write!(out, " at {}:{}", frame.loc.file, frame.loc.line)?;
        }
        if frame.hidden > 0 {
            write!(out, " ({} more references hidden)", frame.hidden)?;
        }
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_error() {
        let mut d = Diagnostics::new();
        d.error(SrcLoc::new("a.jam", 5), "bad thing");
        assert_eq!(d.render_to_string(), "a.jam:5: error: bad thing\n");
    }

    #[test]
    fn error_with_demoted_note() {
        let mut d = Diagnostics::new();
        // Note built as Error (no file) must render as `note:`, indented 4.
        let note = Diagnostic::new(Severity::Error, SrcLoc::none(), "declared here");
        d.error_with_notes(SrcLoc::new("a.jam", 5), "bad thing", vec![note]);
        assert_eq!(
            d.render_to_string(),
            "a.jam:5: error: bad thing\n    note: declared here\n"
        );
    }

    #[test]
    fn instantiation_trace_frame() {
        let mut d = Diagnostics::new();
        let frame = Trace::new(
            SrcLoc::new("v.jam", 10),
            "Vec(i32).default",
            TraceKind::Instantiation,
        );
        d.error_with_trace(SrcLoc::new("a.jam", 5), "bad thing", vec![frame]);
        assert_eq!(
            d.render_to_string(),
            "a.jam:5: error: bad thing\n    note: in instantiation of `Vec(i32).default` at v.jam:10\n"
        );
    }

    #[test]
    fn reference_trace_with_hidden() {
        let mut d = Diagnostics::new();
        let mut frame = Trace::new(SrcLoc::none(), "B", TraceKind::Reference);
        frame.hidden = 3;
        d.error_with_trace(SrcLoc::new("a.jam", 1), "x", vec![frame]);
        assert_eq!(
            d.render_to_string(),
            "a.jam:1: error: x\n    note: referenced by `B` (3 more references hidden)\n"
        );
    }

    #[test]
    fn file_without_line() {
        let mut d = Diagnostics::new();
        d.error(SrcLoc::file_only("x.jam"), "msg");
        assert_eq!(d.render_to_string(), "x.jam: error: msg\n");
    }

    #[test]
    fn no_location() {
        let mut d = Diagnostics::new();
        d.error(SrcLoc::none(), "msg");
        assert_eq!(d.render_to_string(), "error: msg\n");
    }

    #[test]
    fn stable_sort_by_line_then_severity() {
        let mut d = Diagnostics::new();
        // Pushed out of order; warning at line 5 before error at line 5.
        d.warning(SrcLoc::new("a.jam", 5), "w5");
        d.error(SrcLoc::new("a.jam", 5), "e5");
        d.error(SrcLoc::new("a.jam", 3), "e3");
        assert_eq!(
            d.render_to_string(),
            "a.jam:3: error: e3\n\
             a.jam:5: error: e5\n\
             a.jam:5: warning: w5\n"
        );
    }

    #[test]
    fn truncate_and_counts() {
        let mut d = Diagnostics::new();
        d.error(SrcLoc::new("a", 1), "1");
        let mark = d.len();
        d.error(SrcLoc::new("a", 2), "2");
        d.warning(SrcLoc::new("a", 3), "3");
        assert_eq!(d.len(), 3);
        assert_eq!(d.error_count(), 2);
        assert!(d.has_errors());
        d.truncate_to(mark);
        assert_eq!(d.len(), 1);
        assert_eq!(d.error_count(), 1);
        // truncate_to never grows.
        d.truncate_to(99);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn ref_trace_guard_push_pop() {
        let mut stack: Vec<Trace> = Vec::new();
        {
            let _g = RefTraceGuard::push(
                &mut stack,
                Trace::new(SrcLoc::none(), "A", TraceKind::Instantiation),
            );
            // can't inspect stack here (borrowed), so re-check after drop
        }
        assert!(stack.is_empty(), "guard must pop on drop");
        let g = RefTraceGuard::push(
            &mut stack,
            Trace::new(SrcLoc::none(), "B", TraceKind::Reference),
        );
        drop(g);
        assert!(stack.is_empty());
    }
}
