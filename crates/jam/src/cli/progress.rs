/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

// 0=remove
// 1=normal+percent
// 2=error
// 3=indeterminate
// 4=warning

use std::io::{self, IsTerminal, Write};

pub struct ProgressGuard {
    active: bool,
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        self.clear();
    }
}

// The OSC 9;4 progress escapes go to STDERR, only when stderr is a TTY —
// matching the C++ ProgressGuard (`isatty(STDERR_FILENO)`, main.cpp:55). They
// must NEVER touch stdout: stdout carries `--emit-ir` output, which is the
// differential harness's primary gate, and stray escape bytes there poison
// every IR diff. Write failures are ignored (a missing progress hint must not
// crash the compiler).
impl ProgressGuard {
    /// `enabled=false` (test mode) constructs an inert guard — the C++
    /// `ProgressGuard progress(!testMode)`.
    pub fn new(enabled: bool) -> Self {
        if !enabled || !io::stderr().is_terminal() {
            return ProgressGuard { active: false };
        }
        Self::emit("\x1b]9;4;3;\x1b\\");
        ProgressGuard { active: true }
    }

    /// Flag the indicator as errored (state 2) and stop reporting.
    pub fn error(&mut self) {
        if self.active {
            Self::emit("\x1b]9;4;2;\x1b\\");
            self.active = false;
        }
    }

    /// Stop reporting without flagging an error (same exit as success).
    pub fn stop(&mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        if self.active {
            Self::emit("\x1b]9;4;0;\x1b\\");
            self.active = false;
        }
    }

    fn emit(seq: &str) {
        let mut err = io::stderr();
        let _ = err.write_all(seq.as_bytes());
        let _ = err.flush();
    }
}
