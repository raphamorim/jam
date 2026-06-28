/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Lexer — ported 1:1 from `src/lexer.{h,cpp}`.
//!
//! Operates on raw bytes (`&[u8]`) so byte offsets match the C++ exactly (the
//! C++ indexes `source[current]` as `char`). Keywords/operators are ASCII;
//! string contents are scanned byte-by-byte and may carry arbitrary bytes via
//! `\x` escapes. The C++ throws on lexical errors; here those become
//! `Err(LexError)`. An "unexpected character" is reported and skipped (no
//! token), matching the C++ `cerr` + continue.

use crate::token::{Token, TokenType};

/// A lexical error, anchored to a source line (mirrors the C++ `runtime_error`
/// messages, which all carry `at line N`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError {
    pub line: u32,
    pub message: String,
}

pub struct Lexer {
    source: Vec<u8>,
    tokens: Vec<Token>,
    current: usize,
    line: u32,
    /// Byte offset where the current token began (snapshotted per scan-loop
    /// iteration before the first `advance`).
    token_start: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NumState {
    Int,
    IntPeriod,
    Float,
    FloatExp,
}

impl Lexer {
    pub fn new(source: impl Into<Vec<u8>>) -> Lexer {
        Lexer {
            source: source.into(),
            tokens: Vec::new(),
            current: 0,
            line: 1,
            token_start: 0,
        }
    }

    /// The source buffer the lexer ran over — pass to [`Token::text`].
    pub fn source(&self) -> &[u8] {
        &self.source
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn advance(&mut self) -> u8 {
        let c = self.source[self.current];
        self.current += 1;
        c
    }

    fn peek(&self) -> u8 {
        if self.is_at_end() {
            0
        } else {
            self.source[self.current]
        }
    }

    fn peek_next(&self) -> u8 {
        if self.current + 1 >= self.source.len() {
            0
        } else {
            self.source[self.current + 1]
        }
    }

    fn match_(&mut self, expected: u8) -> bool {
        if self.is_at_end() || self.source[self.current] != expected {
            return false;
        }
        self.current += 1;
        true
    }

    fn err(&self, message: impl Into<String>) -> LexError {
        LexError {
            line: self.line,
            message: message.into(),
        }
    }

