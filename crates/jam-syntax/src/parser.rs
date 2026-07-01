/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Recursive-descent parser — ported from `src/parser.cpp`.
//!
//! The complete port, validated against the C++ `--emit-ast` oracle: cursor/
//! error scaffolding, the binary-precedence ladder, `parse_unary`,
//! `parse_type`, and `parse_primary` (literals incl. the f64/f128 number
//! encoding, struct literals, generic type-method calls, `match` + patterns,
//! struct/enum expressions, and the postfix member/index/slice/call chain with
//! the exact direct-vs-indirect call decision), plus statements
//! (`parse_expression`), declarations, and the public `parse()` module entry.
//!
//! `ParserAbort` becomes `Err(ParseError)`; the diagnostic is pushed before the
//! error is returned (matching the C++ `parseError` sink).

use std::collections::HashSet;

use jam_core::diag::{Diagnostics, SrcLoc};
use jam_core::index::{ExtraIdx, NodeIdx, StringIdx, TypeIdx};

use crate::ast::{EnumDeclAST, FunctionAST, StructDeclAST, UnionDeclAST};
use crate::ast_flat::{AstNode, AstTag, BinOp, NodeStore, StringPool, TypePool, UnaryOp, builtin};
use crate::number_literal::{NumberResultKind, number_error_message, parse_number_literal};
use crate::token::{Token, TokenType};

/// The parser aborted after pushing a diagnostic. No recovery (the grammar has
/// no synchronization points) — one bad token ends the parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseError;

type PResult<T> = Result<T, ParseError>;

/// What a `name(...)` followed by `)` then `{` or `.id(` turns out to be.
#[derive(PartialEq, Eq)]
enum TcShape {
    None,
    GenericStructLit,
    TypeMethodCall,
}

pub struct Parser<'a> {
    tokens: Vec<Token>,
    source: Vec<u8>,
    current: usize,
    type_pool: &'a mut TypePool,
    string_pool: &'a mut StringPool,
    nodes: &'a mut NodeStore,
    diagnostics: &'a mut Diagnostics,
    filename: String,

    allow_struct_lit: bool,
    #[allow(dead_code)]
    anon_structs: Vec<StructDeclAST>,
    /// Global base for anon-struct NAMING (shared across module parses), so
    /// `__anon_struct_N` is numbered across all modules like the C++ shared
    /// registry. The per-module `anon_structs` index stays local.
    anon_base: u32,
    #[allow(dead_code)]
    anon_enums: Vec<EnumDeclAST>,
    #[allow(dead_code)]
    struct_context_stack: Vec<String>,
    import_handles: HashSet<String>,
}

impl<'a> Parser<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: Vec<Token>,
        source: Vec<u8>,
        type_pool: &'a mut TypePool,
        string_pool: &'a mut StringPool,
        nodes: &'a mut NodeStore,
        diagnostics: &'a mut Diagnostics,
        filename: impl Into<String>,
    ) -> Parser<'a> {
        Parser {
            tokens,
            source,
            current: 0,
            type_pool,
            string_pool,
            nodes,
            diagnostics,
            filename: filename.into(),
            allow_struct_lit: true,
            anon_structs: Vec::new(),
            anon_base: 0,
            anon_enums: Vec::new(),
            struct_context_stack: Vec::new(),
            import_handles: HashSet::new(),
        }
    }

    /// Set the global anon-struct naming base (the count of anon structs created
    /// by earlier module parses), so `__anon_struct_N` is numbered globally.
    pub fn set_anon_base(&mut self, base: u32) {
        self.anon_base = base;
    }

    // ---- cursor primitives (index-based; no token clones) ----

    fn peek_type(&self) -> TokenType {
        self.tokens[self.current].ttype
    }
    fn is_at_end(&self) -> bool {
        self.peek_type() == TokenType::Eof
    }
    fn check(&self, t: TokenType) -> bool {
        !self.is_at_end() && self.peek_type() == t
    }
    /// Advance one token (never past EOF); the consumed token is `previous`.
    fn advance(&mut self) {
        if !self.is_at_end() {
            self.current += 1;
        }
    }
    fn match_(&mut self, t: TokenType) -> bool {
        if self.check(t) {
            self.advance();
            true
        } else {
            false
        }
    }
    fn consume(&mut self, t: TokenType, msg: &str) -> PResult<()> {
        if self.check(t) {
            self.advance();
            Ok(())
        } else {
            Err(self.parse_error(msg))
        }
    }

    /// Raw source bytes of the most-recently-consumed token.
    fn prev_text(&self) -> &[u8] {
        self.tokens[self.current - 1].text(&self.source)
    }
    /// Decoded lexeme of the most-recently-consumed token (string literals).
    fn prev_lexeme(&self) -> &[u8] {
        &self.tokens[self.current - 1].lexeme
    }

    fn current_loc(&self) -> SrcLoc {
        let line = if self.tokens.is_empty() {
            0
        } else {
            let idx = if self.current < self.tokens.len() {
                self.current
            } else {
                self.current - 1
            };
            self.tokens[idx].line
        };
        SrcLoc::new(self.filename.clone(), line)
    }

    fn parse_error(&mut self, message: impl Into<String>) -> ParseError {
        let loc = self.current_loc();
        self.diagnostics.error(loc, message);
        ParseError
    }

    /// Finalize a node: default `main_token` to the most-recently-consumed
    /// token, record its source line, and append to the store.
    fn emit(&mut self, mut n: AstNode) -> NodeIdx {
        if n.main_token == 0 && self.current > 0 {
            n.main_token = (self.current - 1) as u32;
        }
        let line = self
            .tokens
            .get(n.main_token as usize)
            .map(|t| t.line)
            .unwrap_or(0);
        self.nodes.add_node_at(n, line)
    }

    fn intern_str(&mut self, bytes: &[u8]) -> StringIdx {
        self.string_pool.intern(bytes)
    }

    // ---- number literal decode (ports parseNumLexeme) ----

