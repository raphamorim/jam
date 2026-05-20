/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "diagnostics.h"

#include <algorithm>

namespace jam {

void Diagnostics::error(SrcLoc loc, std::string message) {
	Diagnostic d;
	d.loc = std::move(loc);
	d.severity = Diagnostic::Severity::Error;
	d.message = std::move(message);
	diags_.push_back(std::move(d));
}

void Diagnostics::warning(SrcLoc loc, std::string message) {
	Diagnostic d;
	d.loc = std::move(loc);
	d.severity = Diagnostic::Severity::Warning;
	d.message = std::move(message);
	diags_.push_back(std::move(d));
}

void Diagnostics::errorWithNotes(SrcLoc loc, std::string message,
                                 std::vector<Diagnostic> notes) {
	Diagnostic d;
	d.loc = std::move(loc);
	d.severity = Diagnostic::Severity::Error;
	d.message = std::move(message);
	d.notes = std::move(notes);
	diags_.push_back(std::move(d));
}

void Diagnostics::errorWithTrace(SrcLoc loc, std::string message,
                                 std::vector<Diagnostic::Trace> trace) {
	Diagnostic d;
	d.loc = std::move(loc);
	d.severity = Diagnostic::Severity::Error;
	d.message = std::move(message);
	d.referenceTrace = std::move(trace);
	diags_.push_back(std::move(d));
}

void Diagnostics::push(Diagnostic d) { diags_.push_back(std::move(d)); }

bool Diagnostics::hasErrors() const {
	for (const auto &d : diags_) {
		if (d.severity == Diagnostic::Severity::Error) return true;
	}
	return false;
}

std::size_t Diagnostics::errorCount() const {
	std::size_t n = 0;
	for (const auto &d : diags_) {
		if (d.severity == Diagnostic::Severity::Error) ++n;
	}
	return n;
}

namespace {

const char *severityLabel(Diagnostic::Severity s) {
	switch (s) {
	case Diagnostic::Severity::Error:
		return "error";
	case Diagnostic::Severity::Warning:
		return "warning";
	case Diagnostic::Severity::Note:
		return "note";
	}
	return "error";
}

// Order diagnostics by source position so the report is stable
// regardless of which pass produced them in which order. The test
// suite asserts on stderr substrings and needs a deterministic line
// ordering to stay green, and a user reading multi-error output
// benefits from top-down reads. Errors come before warnings at the
// same location, both before notes.
bool less(const Diagnostic &a, const Diagnostic &b) {
	if (a.loc.file != b.loc.file) return a.loc.file < b.loc.file;
	if (a.loc.line != b.loc.line) return a.loc.line < b.loc.line;
	return static_cast<uint8_t>(a.severity) < static_cast<uint8_t>(b.severity);
}

void emitOne(std::ostream &out, const Diagnostic &d, int indent = 0) {
	std::string pad(static_cast<std::size_t>(indent), ' ');
	out << pad;
	if (!d.loc.file.empty()) {
		out << d.loc.file << ":";
		if (d.loc.line > 0) out << d.loc.line << ":";
		out << " ";
	}
	out << severityLabel(d.severity) << ": " << d.message << "\n";
	for (const auto &note : d.notes) {
		Diagnostic nd = note;
		// Notes inherit Note severity when the caller didn't pick
		// one — keeps the diff between an attached note and a
		// top-level error visible in the output.
		if (nd.severity == Diagnostic::Severity::Error) {
			nd.severity = Diagnostic::Severity::Note;
		}
		emitOne(out, nd, indent + 4);
	}
	for (const auto &frame : d.referenceTrace) {
		const char *verb = frame.kind == Diagnostic::Trace::Kind::Reference
		                       ? "referenced by"
		                       : "in instantiation of";
		out << pad << "    note: " << verb << " `" << frame.decl << "`";
		if (!frame.loc.file.empty() && frame.loc.line > 0) {
			out << " at " << frame.loc.file << ":" << frame.loc.line;
		}
		if (frame.hidden > 0) {
			out << " (" << frame.hidden << " more references hidden)";
		}
		out << "\n";
	}
}

}  // namespace

void Diagnostics::emit(std::ostream &out) const {
	std::vector<Diagnostic> sorted = diags_;
	std::stable_sort(sorted.begin(), sorted.end(), less);
	for (const auto &d : sorted) emitOne(out, d);
}

}  // namespace jam
