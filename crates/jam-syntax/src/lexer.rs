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

    fn skip_whitespace(&mut self) -> Result<(), LexError> {
        loop {
            let c = self.peek();
            match c {
                b' ' | b'\r' | b'\t' => {
                    self.advance();
                }
                b'\n' => {
                    self.line += 1;
                    self.advance();
                }
                b'/' => {
                    if self.peek_next() == b'/' {
                        // `//` line comment — to end of line.
                        while self.peek() != b'\n' && !self.is_at_end() {
                            self.advance();
                        }
                        // continue the loop (handle following ws/comments)
                    } else if self.peek_next() == b'*' {
                        // jam has no block comments (deliberate).
                        return Err(self.err(format!(
                            "jam has no block comments; use `//` line comments (line {})",
                            self.line
                        )));
                    } else {
                        // bare `/` is the divide operator — leave it.
                        return Ok(());
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn is_digit(c: u8) -> bool {
        c.is_ascii_digit()
    }
    fn is_alpha(c: u8) -> bool {
        c.is_ascii_alphabetic() || c == b'_'
    }
    fn is_alpha_numeric(c: u8) -> bool {
        Self::is_alpha(c) || Self::is_digit(c)
    }
    fn is_alpha_digit(c: u8) -> bool {
        c.is_ascii_alphanumeric()
    }

    fn add_token(&mut self, ttype: TokenType) {
        self.add_token_lex(ttype, Vec::new());
    }

    fn add_token_lex(&mut self, ttype: TokenType, lexeme: Vec<u8>) {
        let length = self.current as u32 - self.token_start;
        self.tokens.push(Token {
            ttype,
            lexeme,
            line: self.line,
            byte_offset: self.token_start,
            length,
        });
    }

