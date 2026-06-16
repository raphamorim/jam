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

use std::io;
use std::io::Write;

pub struct ProgressGuard {
    active: bool,
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        self.clear();
    }
}

impl ProgressGuard {
    pub fn new() -> Self {
        let mut out = io::stdout();
        write!(out, "\x1b]9;4;3;\x1b\\").unwrap();
        out.flush().unwrap();
        ProgressGuard {
            active: true
        }
    }

    pub fn error(&mut self) {
        if self.active {
            let mut out = io::stdout();
            write!(out, "\x1b]9;4;2;\x1b\\").unwrap();
            out.flush().unwrap();
            self.active = false;
        }
    }

    fn clear(&mut self) {
        if self.active {
            let mut out = io::stdout();
            write!(out, "\x1b]9;4;0;\x1b\\").unwrap();
            out.flush().unwrap();
            self.active = false;
        }
    }
}
