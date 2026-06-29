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

    fn identifier(&mut self) {
        let start = self.current - 1; // first char already consumed
        while Self::is_alpha_numeric(self.peek()) {
            self.advance();
        }
        let text = &self.source[start..self.current];
        let ttype = match text {
            b"fn" => TokenType::Fn,
            b"cfn" => TokenType::Cfn,
            b"return" => TokenType::Return,
            b"const" => TokenType::Const,
            b"var" => TokenType::Var,
            b"mut" => TokenType::Mut,
            b"if" => TokenType::If,
            b"else" => TokenType::Else,
            b"match" => TokenType::Match,
            b"while" => TokenType::While,
            b"loop" => TokenType::Loop,
            b"for" => TokenType::For,
            b"break" => TokenType::Break,
            b"continue" => TokenType::Continue,
            b"in" => TokenType::In,
            b"true" => TokenType::True,
            b"false" => TokenType::False,
            b"extern" => TokenType::Extern,
            b"export" => TokenType::Export,
            b"pub" => TokenType::Pub,
            b"import" => TokenType::Import,
            b"tfn" => TokenType::Tfn,
            b"struct" => TokenType::Struct,
            b"union" => TokenType::Union,
            b"enum" => TokenType::Enum,
            b"as" => TokenType::As,
            b"move" => TokenType::Move,
            b"comp" => TokenType::Comp,
            b"inline" => TokenType::Inline,
            // The scalar built-ins and the meta-type `type` all lex as TYPE.
            b"u1" | b"u8" | b"u16" | b"u32" | b"u64" | b"i8" | b"i16" | b"i32" | b"i64"
            | b"f32" | b"f64" | b"bool" | b"str" | b"type" | b"noreturn" => TokenType::Type,
            _ => TokenType::Identifier,
        };
        self.add_token(ttype);
    }

    fn number(&mut self) {
        self.scan_number_body();
    }

    fn negative_number(&mut self) {
        self.advance(); // consume the first digit after the leading `-`
        self.scan_number_body();
    }

    fn scan_number_body(&mut self) {
        let mut state = NumState::Int;
        loop {
            let c = self.peek();
            let mut stop = false;
            match state {
                NumState::Int => {
                    if c == b'.' {
                        let n = self.peek_next();
                        if n == b'.' || n == b'=' {
                            stop = true;
                        } else {
                            self.advance();
                            state = NumState::IntPeriod;
                        }
                    } else if c == b'_' || Self::is_alpha_digit(c) {
                        if c == b'e' || c == b'E' || c == b'p' || c == b'P' {
                            self.advance();
                            state = NumState::FloatExp;
                        } else {
                            self.advance();
                        }
                    } else {
                        stop = true;
                    }
                }
                NumState::IntPeriod => {
                    if c == b'_' || Self::is_alpha_digit(c) {
                        if c == b'e' || c == b'E' || c == b'p' || c == b'P' {
                            self.advance();
                            state = NumState::FloatExp;
                        } else {
                            self.advance();
                            state = NumState::Float;
                        }
                    } else {
                        // `1.` then non-numeric: rewind the `.` to tokenize it
                        // separately (member access / operator).
                        self.current -= 1;
                        stop = true;
                    }
                }
                NumState::Float => {
                    if c == b'_' || Self::is_alpha_digit(c) {
                        if c == b'e' || c == b'E' || c == b'p' || c == b'P' {
                            self.advance();
                            state = NumState::FloatExp;
                        } else {
                            self.advance();
                        }
                    } else {
                        stop = true;
                    }
                }
                NumState::FloatExp => {
                    if c == b'+' || c == b'-' {
                        self.advance();
                        state = NumState::Float;
                    } else {
                        // sign was optional; bounce to Float and retry.
                        state = NumState::Float;
                    }
                }
            }
            if stop {
                break;
            }
        }
        self.add_token(TokenType::Number);
    }

