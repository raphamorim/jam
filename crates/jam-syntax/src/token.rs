/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Tokens — ported 1:1 from `src/token.h`.
//!
//! Storage model matches the C++: `byte_offset` + `length` mark the token's raw
//! span in the source buffer; [`Token::text`] slices it (the right call for
//! every kind except `StringLiteral`). `lexeme` carries the *decoded* value and
//! is populated only for `StringLiteral` (escape sequences resolved to the
//! bytes the runtime expects — which the raw source slice doesn't contain). It
//! is `Vec<u8>`, not `String`, because `\xFF`-style escapes can produce
//! non-UTF-8 bytes.

/// Token kind. Variant order matches the C++ `enum TokenType` exactly, so the
/// discriminants line up (`Eof == 0`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TokenType {
    Eof,
    Fn,
    Cfn,
    Identifier,
    Colon,
    OpenBrace,
    CloseBrace,
    OpenParen,
    CloseParen,
    Comma,
    Return,
    Plus,
    Minus,
    Semi,
    Number,
    Const,
    Var,
    Mut,
    Equal,
    Type,
    If,
    Else,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    True,
    False,
    OpenBracket,
    CloseBracket,
    StringLiteral,
    While,
    Loop,
    For,
    Break,
    Continue,
    In,
    Extern,
    Export,
    Pub,
    Import,
    Dot,
    And,
    Or,
    Not,
    Tfn,
    Struct,
    Union,
    Enum,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Lshift,
    Rshift,
    Move,
    Ellipsis,
    Match,
    DotDot,
    DotDotEq,
    As,
    At,
    Comp,
    Inline,
}

impl TokenType {
    /// Stable dump name — the `--emit-tokens` contract (the C++ `TOK_` prefix
    /// dropped). Must stay byte-identical to the oracle's `tokenName`.
    pub fn name(self) -> &'static str {
        use TokenType::*;
        match self {
            Eof => "EOF",
            Fn => "FN",
            Cfn => "CFN",
            Identifier => "IDENTIFIER",
            Colon => "COLON",
            OpenBrace => "OPEN_BRACE",
            CloseBrace => "CLOSE_BRACE",
            OpenParen => "OPEN_PAREN",
            CloseParen => "CLOSE_PAREN",
            Comma => "COMMA",
            Return => "RETURN",
            Plus => "PLUS",
            Minus => "MINUS",
            Semi => "SEMI",
            Number => "NUMBER",
            Const => "CONST",
            Var => "VAR",
            Mut => "MUT",
            Equal => "EQUAL",
            Type => "TYPE",
            If => "IF",
            Else => "ELSE",
            EqualEqual => "EQUAL_EQUAL",
            NotEqual => "NOT_EQUAL",
            Less => "LESS",
            LessEqual => "LESS_EQUAL",
            Greater => "GREATER",
            GreaterEqual => "GREATER_EQUAL",
            True => "TRUE",
            False => "FALSE",
            OpenBracket => "OPEN_BRACKET",
            CloseBracket => "CLOSE_BRACKET",
            StringLiteral => "STRING_LITERAL",
            While => "WHILE",
            Loop => "LOOP",
            For => "FOR",
            Break => "BREAK",
            Continue => "CONTINUE",
            In => "IN",
            Extern => "EXTERN",
            Export => "EXPORT",
            Pub => "PUB",
            Import => "IMPORT",
            Dot => "DOT",
            And => "AND",
            Or => "OR",
            Not => "NOT",
            Tfn => "TFN",
            Struct => "STRUCT",
            Union => "UNION",
            Enum => "ENUM",
            Star => "STAR",
            Slash => "SLASH",
            Percent => "PERCENT",
            Amp => "AMP",
            Pipe => "PIPE",
            Caret => "CARET",
            Tilde => "TILDE",
            Lshift => "LSHIFT",
            Rshift => "RSHIFT",
            Move => "MOVE",
            Ellipsis => "ELLIPSIS",
            Match => "MATCH",
            DotDot => "DOTDOT",
            DotDotEq => "DOTDOT_EQ",
            As => "AS",
            At => "AT",
            Comp => "COMP",
            Inline => "INLINE",
        }
    }
}

/// One lexed token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub ttype: TokenType,
    /// Decoded value — only populated for `StringLiteral`; empty otherwise.
    pub lexeme: Vec<u8>,
    /// 1-based source line.
    pub line: u32,
    pub byte_offset: u32,
    pub length: u32,
}

impl Token {
    /// Raw source span. The returned slice is valid as long as `source` is the
    /// same buffer the lexer ran over. For `StringLiteral` this is the raw
    /// `"..."`-bracketed bytes — use [`Token::lexeme`] for the decoded value.
    pub fn text<'a>(&self, source: &'a [u8]) -> &'a [u8] {
        let start = self.byte_offset as usize;
        let end = start + self.length as usize;
        &source[start..end]
    }
}
