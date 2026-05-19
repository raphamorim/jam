/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef JAM_DIAGNOSTICS_H
#define JAM_DIAGNOSTICS_H

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace jam {

// Where in source a diagnostic is anchored.
//
// Today we resolve to `(file, line)` only. Column / span / byte
// offset will be tacked on as additional fields when the parser
// records token ranges per node — none of the call sites change
// when that happens because they only handle SrcLoc by value.
struct SrcLoc {
	// Display filename — already-formatted relative path the user
	// recognises. We keep it as a plain std::string for now; a
	// future StringPool migration is cheap because every site
	// constructs SrcLoc by value.
	std::string file;
	int line = 0;
};

// One diagnostic emitted by any phase of the compiler (parser,
// astgen, init_analysis, jir_verify, …).
//   * `notes` carry secondary messages that may live in a different
//     file/line — used for "X was declared here" / "did you mean Y?".
//   * `referenceTrace` records the generic-instantiation chain that
//     led to an astgen failure inside a monomorphisation.
struct Diagnostic {
	enum class Severity : uint8_t { Error, Warning, Note };

	struct Trace {
		SrcLoc loc;
		std::string decl;  // e.g. "Vec(NoDefault).default"
		// `hidden` counts trace frames that were elided when the
		// chain exceeded a (future) `--reference-trace=N` limit.
		// Kept at 0 today: the full live stack is always
		// materialised at error time — no truncation yet.
		uint32_t hidden = 0;
	};

	SrcLoc loc;
	Severity severity = Severity::Error;
	std::string message;
	std::vector<Diagnostic> notes;
	std::vector<Trace> referenceTrace;
};

// RAII guard for the reference-trace stack — push on construct, pop
// on destruct. Generic-instantiation entry points wrap their body
// emission in one of these so any AstGenAnalysisFail raised deeper
// reaches the diagnostic emitter with the chain attached.
//
// Scope: this is a stack of *currently-active* instantiation frames.
// It captures nested instantiation (A→B→C while all three are still
// in flight) but does NOT capture references from already-completed
// decls. The stack covers the practical case where instantiation
// errors fire mid-stack; cross-decl historical reference chains are
// a follow-up if/when a real reference-graph pass exists.
class RefTraceFrame {
  public:
	RefTraceFrame(std::vector<Diagnostic::Trace> &stack, Diagnostic::Trace f)
	    : stack_(stack) {
		stack_.push_back(std::move(f));
	}
	~RefTraceFrame() { stack_.pop_back(); }
	RefTraceFrame(const RefTraceFrame &) = delete;
	RefTraceFrame &operator=(const RefTraceFrame &) = delete;

  private:
	std::vector<Diagnostic::Trace> &stack_;
};

// Per-compilation accumulator. Phases push; the driver emits at the
// end. Nothing in the compiler aborts on the first error — every
// pass is expected to push and continue so the user sees the whole
// batch.
class Diagnostics {
  public:
	void error(SrcLoc loc, std::string message);
	void warning(SrcLoc loc, std::string message);
	void errorWithNotes(SrcLoc loc, std::string message,
	                    std::vector<Diagnostic> notes);
	void errorWithTrace(SrcLoc loc, std::string message,
	                    std::vector<Diagnostic::Trace> referenceTrace);

	// Push a fully-built diagnostic (used by helpers that already
	// have notes / trace assembled).
	void push(Diagnostic diag);

	bool hasErrors() const;
	std::size_t errorCount() const;
	const std::vector<Diagnostic> &all() const { return diags_; }

	// Format the report:
	//     file:line: error: message
	//         note: ...
	//         in instantiation of `decl` at file:line
	// Stable order: stable sort by (file, line, severity). Errors
	// emitted before warnings at the same location.
	void emit(std::ostream &out) const;

  private:
	std::vector<Diagnostic> diags_;
};

}  // namespace jam

#endif  // JAM_DIAGNOSTICS_H
