/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The lexer. Operates on raw bytes (`&[u8]`): keywords and operators are
//! ASCII; string contents are scanned byte-by-byte and may carry arbitrary
//! bytes via `\x` escapes. Lexical errors surface as `Err(LexError)`; an
//! "unexpected character" is reported to stderr and skipped (no token).

use crate::token::{Token, TokenType};

/// A lexical error, anchored to a source line.
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

    fn parse_hex_byte(&mut self) -> Result<u8, LexError> {
        let mut value: u8 = 0;
        for _ in 0..2 {
            let c = self.peek();
            if c.is_ascii_digit() {
                value = value.wrapping_mul(16).wrapping_add(c - b'0');
            } else if (b'a'..=b'f').contains(&c) {
                value = value.wrapping_mul(16).wrapping_add(c - b'a' + 10);
            } else if (b'A'..=b'F').contains(&c) {
                value = value.wrapping_mul(16).wrapping_add(c - b'A' + 10);
            } else {
                return Err(self.err(format!("Expected hex digit at line {}", self.line)));
            }
            self.advance();
        }
        Ok(value)
    }

    fn parse_unicode_escape(&mut self) -> Result<Vec<u8>, LexError> {
        if self.peek() != b'{' {
            return Err(self.err(format!("Expected '{{' after \\u at line {}", self.line)));
        }
        self.advance(); // '{'

        let mut codepoint: u32 = 0;
        let mut digit_count = 0;
        while self.peek() != b'}' && !self.is_at_end() {
            let c = self.peek();
            if c.is_ascii_digit() {
                codepoint = codepoint * 16 + (c - b'0') as u32;
            } else if (b'a'..=b'f').contains(&c) {
                codepoint = codepoint * 16 + (c - b'a' + 10) as u32;
            } else if (b'A'..=b'F').contains(&c) {
                codepoint = codepoint * 16 + (c - b'A' + 10) as u32;
            } else {
                return Err(self.err(format!(
                    "Expected hex digit in unicode escape at line {}",
                    self.line
                )));
            }
            self.advance();
            digit_count += 1;
            if digit_count > 6 {
                return Err(self.err(format!(
                    "Unicode escape too long (max 6 hex digits) at line {}",
                    self.line
                )));
            }
        }
        if self.is_at_end() || self.peek() != b'}' {
            return Err(self.err(format!(
                "Expected '}}' to close unicode escape at line {}",
                self.line
            )));
        }
        self.advance(); // '}'
        if digit_count == 0 {
            return Err(self.err(format!("Empty unicode escape at line {}", self.line)));
        }
        if codepoint > 0x10FFFF {
            return Err(self.err(format!(
                "Unicode codepoint out of range (max 0x10FFFF) at line {}",
                self.line
            )));
        }
        // Encode the codepoint as UTF-8.
        let mut out = Vec::new();
        if codepoint <= 0x7F {
            out.push(codepoint as u8);
        } else if codepoint <= 0x7FF {
            out.push(0xC0 | (codepoint >> 6) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        } else if codepoint <= 0xFFFF {
            out.push(0xE0 | (codepoint >> 12) as u8);
            out.push(0x80 | ((codepoint >> 6) & 0x3F) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        } else {
            out.push(0xF0 | (codepoint >> 18) as u8);
            out.push(0x80 | ((codepoint >> 12) & 0x3F) as u8);
            out.push(0x80 | ((codepoint >> 6) & 0x3F) as u8);
            out.push(0x80 | (codepoint & 0x3F) as u8);
        }
        Ok(out)
    }

    fn string_literal(&mut self) -> Result<(), LexError> {
        let mut value: Vec<u8> = Vec::new();
        while self.peek() != b'"' && !self.is_at_end() {
            if self.peek() == b'\n' {
                return Err(self.err(format!("Unterminated string at line {}", self.line)));
            }
            if self.peek() == b'\\' {
                self.advance(); // backslash
                if self.is_at_end() {
                    return Err(self.err(format!(
                        "Unterminated escape sequence at line {}",
                        self.line
                    )));
                }
                let escaped = self.advance();
                match escaped {
                    b'n' => value.push(b'\n'),
                    b'r' => value.push(b'\r'),
                    b't' => value.push(b'\t'),
                    b'\\' => value.push(b'\\'),
                    b'"' => value.push(b'"'),
                    b'\'' => value.push(b'\''),
                    b'0' => value.push(b'\0'),
                    b'x' => value.push(self.parse_hex_byte()?),
                    b'u' => value.extend_from_slice(&self.parse_unicode_escape()?),
                    other => {
                        return Err(self.err(format!(
                            "Invalid escape sequence '\\{}' at line {}",
                            other as char, self.line
                        )));
                    }
                }
            } else {
                let b = self.advance();
                value.push(b);
            }
        }
        if self.is_at_end() {
            return Err(self.err(format!("Unterminated string at line {}", self.line)));
        }
        self.advance(); // closing "
        self.add_token_lex(TokenType::StringLiteral, value);
        Ok(())
    }

    /// Scan the whole source into tokens (plus a trailing `Eof`).
    pub fn scan_tokens(&mut self) -> Result<(), LexError> {
        while !self.is_at_end() {
            self.skip_whitespace()?;
            if self.is_at_end() {
                break;
            }
            self.token_start = self.current as u32;
            let c = self.advance();
            match c {
                b'(' => self.add_token(TokenType::OpenParen),
                b')' => self.add_token(TokenType::CloseParen),
                b'{' => self.add_token(TokenType::OpenBrace),
                b'}' => self.add_token(TokenType::CloseBrace),
                b'[' => self.add_token(TokenType::OpenBracket),
                b']' => self.add_token(TokenType::CloseBracket),
                b',' => self.add_token(TokenType::Comma),
                b';' => self.add_token(TokenType::Semi),
                b':' => self.add_token(TokenType::Colon),
                b'+' => self.add_token(TokenType::Plus),
                b'.' => {
                    if self.peek() == b'.' && self.peek_next() == b'.' {
                        self.advance();
                        self.advance();
                        self.add_token(TokenType::Ellipsis);
                    } else if self.peek() == b'.' && self.peek_next() == b'=' {
                        self.advance();
                        self.advance();
                        self.add_token(TokenType::DotDotEq);
                    } else if self.peek() == b'.' {
                        self.advance();
                        self.add_token(TokenType::DotDot);
                    } else {
                        self.add_token(TokenType::Dot);
                    }
                }
                b'"' => self.string_literal()?,
                b'=' => {
                    if self.match_(b'=') {
                        self.add_token(TokenType::EqualEqual);
                    } else {
                        self.add_token(TokenType::Equal);
                    }
                }
                b'!' => {
                    if self.match_(b'=') {
                        self.add_token(TokenType::NotEqual);
                    } else {
                        self.add_token(TokenType::Not);
                    }
                }
                b'@' => self.add_token(TokenType::At),
                b'<' => {
                    if self.match_(b'=') {
                        self.add_token(TokenType::LessEqual);
                    } else if self.match_(b'<') {
                        self.add_token(TokenType::Lshift);
                    } else {
                        self.add_token(TokenType::Less);
                    }
                }
                b'>' => {
                    if self.match_(b'=') {
                        self.add_token(TokenType::GreaterEqual);
                    } else if self.match_(b'>') {
                        self.add_token(TokenType::Rshift);
                    } else {
                        self.add_token(TokenType::Greater);
                    }
                }
                b'*' => self.add_token(TokenType::Star),
                b'/' => self.add_token(TokenType::Slash),
                b'%' => self.add_token(TokenType::Percent),
                b'&' => {
                    if self.match_(b'&') {
                        self.add_token(TokenType::And);
                    } else {
                        self.add_token(TokenType::Amp);
                    }
                }
                b'|' => {
                    if self.match_(b'|') {
                        self.add_token(TokenType::Or);
                    } else {
                        self.add_token(TokenType::Pipe);
                    }
                }
                b'^' => self.add_token(TokenType::Caret),
                b'~' => self.add_token(TokenType::Tilde),
                b'-' => {
                    if Self::is_digit(self.peek()) {
                        self.negative_number();
                    } else {
                        self.add_token(TokenType::Minus);
                    }
                }
                _ => {
                    if Self::is_digit(c) {
                        self.number();
                    } else if Self::is_alpha(c) {
                        self.identifier();
                    } else {
                        // Unexpected character: report and skip.
                        eprintln!("Unexpected character at line {}: {}", self.line, c as char);
                    }
                }
            }
        }
        self.tokens.push(Token {
            ttype: TokenType::Eof,
            lexeme: Vec::new(),
            line: self.line,
            byte_offset: 0,
            length: 0,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenType> {
        let mut lx = Lexer::new(src.as_bytes().to_vec());
        lx.scan_tokens().unwrap();
        lx.tokens().iter().map(|t| t.ttype).collect()
    }

    #[test]
    fn keywords_and_ops() {
        use TokenType::*;
        assert_eq!(
            kinds("fn add ( ) { return 1 + 2 ; }"),
            vec![
                Fn, Identifier, OpenParen, CloseParen, OpenBrace, Return, Number, Plus, Number,
                Semi, CloseBrace, Eof
            ]
        );
    }

    #[test]
    fn range_vs_float() {
        use TokenType::*;
        // `0..=9` must split, not swallow the dot into a float.
        assert_eq!(kinds("0..=9"), vec![Number, DotDotEq, Number, Eof]);
        assert_eq!(kinds("1.5"), vec![Number, Eof]);
        // The number scanner is greedy (letters are potential suffixes;
        // validation is deferred to the parser), so `1.x` is one NUMBER token.
        assert_eq!(kinds("1.x"), vec![Number, Eof]);
    }

    #[test]
    fn negative_number_is_one_token() {
        let mut lx = Lexer::new(b"-5".to_vec());
        lx.scan_tokens().unwrap();
        assert_eq!(lx.tokens()[0].ttype, TokenType::Number);
        assert_eq!(lx.tokens()[0].text(lx.source()), b"-5");
    }

    #[test]
    fn line_comment_and_lines() {
        let mut lx = Lexer::new("// hi\nfn".as_bytes().to_vec());
        lx.scan_tokens().unwrap();
        assert_eq!(lx.tokens()[0].ttype, TokenType::Fn);
        assert_eq!(lx.tokens()[0].line, 2);
    }

    #[test]
    fn string_escapes_decode() {
        let mut lx = Lexer::new(b"\"a\\nb\"".to_vec());
        lx.scan_tokens().unwrap();
        assert_eq!(lx.tokens()[0].ttype, TokenType::StringLiteral);
        assert_eq!(lx.tokens()[0].lexeme, b"a\nb");
    }

    #[test]
    fn block_comment_errors() {
        let mut lx = Lexer::new(b"/* x */".to_vec());
        assert!(lx.scan_tokens().is_err());
    }

    #[test]
    fn type_keywords() {
        use TokenType::*;
        assert_eq!(
            kinds("i32 bool type Foo"),
            vec![Type, Type, Type, Identifier, Eof]
        );
    }
}
