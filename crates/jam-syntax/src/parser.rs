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

    /// Returns `(magnitude, is_neg, is_float)`. Errors (with the same messages)
    /// on BigInt / malformed literals. The float bit pattern it would compute
    /// is vestigial — `parse_primary` re-derives floats via `float128`.
    fn parse_num_lexeme(&mut self, s: &[u8]) -> PResult<(u64, bool, bool)> {
        let neg = !s.is_empty() && s[0] == b'-';
        let abs = if neg { &s[1..] } else { s };
        let r = parse_number_literal(abs);
        match r.kind {
            NumberResultKind::Int => Ok((r.int_value, neg, false)),
            NumberResultKind::Float => Ok((r.float_value.to_bits(), neg, true)),
            NumberResultKind::BigInt => {
                let msg = format!(
                    "integer literal `{}` exceeds u64 range",
                    String::from_utf8_lossy(abs)
                );
                Err(self.parse_error(msg))
            }
            NumberResultKind::Failure => {
                let kind = r.failure.expect("failure kind").kind;
                let msg = format!(
                    "invalid numeric literal `{}`: {}",
                    String::from_utf8_lossy(abs),
                    number_error_message(kind)
                );
                Err(self.parse_error(msg))
            }
        }
    }

    // ---- qualified-name chain helpers ----

    fn qualified_name(&mut self, chain_root: NodeIdx) -> PResult<String> {
        let n = *self.nodes.get(chain_root);
        match n.tag {
            AstTag::Variable => Ok(String::from_utf8_lossy(
                &self.string_pool.get(StringIdx::new(n.lhs)),
            )
            .into_owned()),
            AstTag::MemberAccess => {
                let base = self.qualified_name(NodeIdx::new(n.lhs))?;
                let member = String::from_utf8_lossy(&self.string_pool.get(StringIdx::new(n.rhs)))
                    .into_owned();
                Ok(format!("{base}.{member}"))
            }
            _ => Err(self.parse_error("Invalid member access chain")),
        }
    }

    fn is_qualified_name_chain(&self, chain_root: NodeIdx) -> bool {
        let n = self.nodes.get(chain_root);
        match n.tag {
            AstTag::Variable => true,
            AstTag::MemberAccess => self.is_qualified_name_chain(NodeIdx::new(n.lhs)),
            _ => false,
        }
    }

    fn chain_root_name(&self, chain_root: NodeIdx) -> String {
        let n = self.nodes.get(chain_root);
        match n.tag {
            AstTag::Variable => {
                String::from_utf8_lossy(&self.string_pool.get(StringIdx::new(n.lhs))).into_owned()
            }
            AstTag::MemberAccess => self.chain_root_name(NodeIdx::new(n.lhs)),
            _ => String::new(),
        }
    }

    fn chain_dot_count(&self, chain_root: NodeIdx) -> i32 {
        let n = self.nodes.get(chain_root);
        if n.tag != AstTag::MemberAccess {
            0
        } else {
            1 + self.chain_dot_count(NodeIdx::new(n.lhs))
        }
    }

    // ===================================================================
    // Binary-precedence ladder (lowest -> highest). Each level is a
    // left-associative fold emitting BinaryOp nodes (lhs/rhs = NodeIdx).
    // ===================================================================

    fn parse_logical_or(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_logical_and()?;
        while self.match_(TokenType::Or) {
            let rhs = self.parse_logical_and()?;
            lhs = self.emit_binop(BinOp::LogOr, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_comparison()?;
        while self.match_(TokenType::And) {
            let rhs = self.parse_comparison()?;
            lhs = self.emit_binop(BinOp::LogAnd, lhs, rhs);
        }
        Ok(lhs)
    }

    /// Comparisons are NON-associative: exactly one (`if`, not `while`).
    fn parse_comparison(&mut self) -> PResult<NodeIdx> {
        let lhs = self.parse_bitwise()?;
        let op = match self.peek_type() {
            TokenType::EqualEqual => Some(BinOp::Eq),
            TokenType::NotEqual => Some(BinOp::Ne),
            TokenType::Less => Some(BinOp::Lt),
            TokenType::LessEqual => Some(BinOp::Le),
            TokenType::Greater => Some(BinOp::Gt),
            TokenType::GreaterEqual => Some(BinOp::Ge),
            _ => None,
        };
        if let Some(k) = op {
            self.advance();
            let rhs = self.parse_bitwise()?;
            Ok(self.emit_binop(k, lhs, rhs))
        } else {
            Ok(lhs)
        }
    }

    fn parse_bitwise(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_shift()?;
        loop {
            let k = if self.match_(TokenType::Amp) {
                BinOp::BitAnd
            } else if self.match_(TokenType::Pipe) {
                BinOp::BitOr
            } else if self.match_(TokenType::Caret) {
                BinOp::BitXor
            } else {
                break;
            };
            let rhs = self.parse_shift()?;
            lhs = self.emit_binop(k, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_addition()?;
        loop {
            let k = if self.match_(TokenType::Lshift) {
                BinOp::Shl
            } else if self.match_(TokenType::Rshift) {
                BinOp::Shr
            } else {
                break;
            };
            let rhs = self.parse_addition()?;
            lhs = self.emit_binop(k, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_addition(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_multiplication()?;
        loop {
            let k = if self.match_(TokenType::Plus) {
                BinOp::Add
            } else if self.match_(TokenType::Minus) {
                BinOp::Sub
            } else {
                break;
            };
            let rhs = self.parse_multiplication()?;
            lhs = self.emit_binop(k, lhs, rhs);
        }
        Ok(lhs)
    }

    fn parse_multiplication(&mut self) -> PResult<NodeIdx> {
        let mut lhs = self.parse_unary()?;
        loop {
            let k = if self.match_(TokenType::Star) {
                BinOp::Mul
            } else if self.match_(TokenType::Slash) {
                BinOp::Div
            } else if self.match_(TokenType::Percent) {
                BinOp::Mod
            } else {
                break;
            };
            let rhs = self.parse_unary()?;
            lhs = self.emit_binop(k, lhs, rhs);
        }
        Ok(lhs)
    }

    fn emit_binop(&mut self, op: BinOp, lhs: NodeIdx, rhs: NodeIdx) -> NodeIdx {
        self.emit(AstNode {
            tag: AstTag::BinaryOp,
            op: op as u8,
            flags: 0,
            main_token: 0,
            lhs: lhs.raw(),
            rhs: rhs.raw(),
        })
    }

    // ===================================================================
    // Statements (parse_expression is the STATEMENT entry, per the C++ name).
    // Entry-level subtleties are load-bearing: init/cond use parse_logical_or;
    // for-bounds and assign-target use parse_comparison.
    // ===================================================================

    /// Parse statements until `}` (not consumed) or EOF.
    fn parse_stmts_until_brace(&mut self) -> PResult<Vec<NodeIdx>> {
        let mut body = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            body.push(self.parse_expression()?);
        }
        Ok(body)
    }

    /// `[thenCount, elseCount, then..., else...]` extra-pool block (If).
    fn push_if_extra(&mut self, then_body: &[NodeIdx], else_body: &[NodeIdx]) -> ExtraIdx {
        let e = self
            .nodes
            .reserve_extra(2 + then_body.len() + else_body.len());
        self.nodes.set_extra(e, then_body.len() as u32);
        self.nodes
            .set_extra(ExtraIdx::new(e.raw() + 1), else_body.len() as u32);
        let mut p = e.raw() + 2;
        for b in then_body {
            self.nodes.set_extra(ExtraIdx::new(p), b.raw());
            p += 1;
        }
        for b in else_body {
            self.nodes.set_extra(ExtraIdx::new(p), b.raw());
            p += 1;
        }
        e
    }

    /// `[bodyCount, body...]` extra-pool block (While).
    fn push_body_extra(&mut self, body: &[NodeIdx]) -> ExtraIdx {
        let e = self.nodes.reserve_extra(1 + body.len());
        self.nodes.set_extra(e, body.len() as u32);
        for (i, b) in body.iter().enumerate() {
            self.nodes
                .set_extra(ExtraIdx::new(e.raw() + 1 + i as u32), b.raw());
        }
        e
    }

    /// Parse one `name [: type] = init ;` var binding into a VarDecl whose rhs
    /// flags are `flags` (bit0 = isConst, bit1 = comp).
    fn parse_var_decl(&mut self, flags: u32) -> PResult<NodeIdx> {
        self.consume(TokenType::Identifier, "Expected variable name")?;
        let name = self.prev_text().to_vec();
        let name_id = self.intern_str(&name);
        let ty = if self.match_(TokenType::Colon) {
            self.parse_type()?
        } else {
            TypeIdx::NONE
        };
        self.consume(
            TokenType::Equal,
            "Expected '=' (every variable must be initialized at declaration)",
        )?;
        let init = self.parse_logical_or()?;
        self.consume(TokenType::Semi, "Expected ';' after variable declaration")?;
        let e = self.nodes.reserve_extra(3);
        self.nodes.set_extra(e, name_id.raw());
        self.nodes.set_extra(ExtraIdx::new(e.raw() + 1), ty.raw());
        self.nodes.set_extra(ExtraIdx::new(e.raw() + 2), init.raw());
        Ok(self.emit(AstNode {
            tag: AstTag::VarDecl,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: e.raw(),
            rhs: flags,
        }))
    }

    /// Parse a parenthesized `( cond )` head + `{ body }`, returning
    /// `(cond, body)`. Used by if/while (heads are unambiguous via parens, so
    /// `allow_struct_lit` is left as-is). `kw_quoted` is the keyword as quoted
    /// in the open-paren message (the C++ quotes `'while'` but
    /// `` `inline while` ``); `kw` is the form used in the remaining messages.
    fn parse_paren_cond_block(
        &mut self,
        kw_quoted: &str,
        kw: &str,
    ) -> PResult<(NodeIdx, Vec<NodeIdx>)> {
        self.consume(
            TokenType::OpenParen,
            &format!("Expected '(' after {kw_quoted}"),
        )?;
        let cond = self.parse_logical_or()?;
        self.consume(
            TokenType::CloseParen,
            &format!("Expected ')' after {kw} condition"),
        )?;
        self.consume(
            TokenType::OpenBrace,
            &format!("Expected '{{' after {kw} condition"),
        )?;
        let body = self.parse_stmts_until_brace()?;
        self.consume(
            TokenType::CloseBrace,
            &format!("Expected '}}' after {kw} body"),
        )?;
        Ok((cond, body))
    }

    fn parse_expression(&mut self) -> PResult<NodeIdx> {
        // `comp` prefix: comp const/var decl or comp if.
        if self.match_(TokenType::Comp) {
            if self.match_(TokenType::Const) {
                return self.parse_var_decl(1 | 2);
            }
            if self.match_(TokenType::Var) {
                return self.parse_var_decl(2);
            }
            if self.check(TokenType::If) {
                self.advance(); // consume `if`
                let (cond, then_body, else_body) = self.parse_if_tail(true)?;
                let e = self.push_if_extra(&then_body, &else_body);
                return Ok(self.emit(AstNode {
                    tag: AstTag::IfNode,
                    op: 0,
                    flags: 1, // comp
                    main_token: 0,
                    lhs: cond.raw(),
                    rhs: e.raw(),
                }));
            }
            return Err(self.parse_error("`comp` must be followed by `const`, `var`, or `if`"));
        }

        // `inline while`.
        if self.match_(TokenType::Inline) {
            self.consume(TokenType::While, "Expected `while` after `inline`")?;
            let (cond, body) = self.parse_paren_cond_block("`inline while`", "`inline while`")?;
            let e = self.push_body_extra(&body);
            return Ok(self.emit(AstNode {
                tag: AstTag::WhileNode,
                op: 0,
                flags: 1, // inline
                main_token: 0,
                lhs: cond.raw(),
                rhs: e.raw(),
            }));
        }

        if self.match_(TokenType::Return) {
            if self.match_(TokenType::Semi) {
                return Ok(self.emit(AstNode {
                    tag: AstTag::Return,
                    op: 0,
                    flags: 0,
                    main_token: 0,
                    lhs: NodeIdx::NONE.raw(),
                    rhs: 0,
                }));
            }
            let expr = self.parse_logical_or()?;
            self.consume(TokenType::Semi, "Expected ';' after return statement")?;
            return Ok(self.emit(AstNode {
                tag: AstTag::Return,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: expr.raw(),
                rhs: 0,
            }));
        }

        if self.match_(TokenType::Const) {
            return self.parse_var_decl(1);
        }
        if self.match_(TokenType::Var) {
            return self.parse_var_decl(0);
        }

        // `match` — delegate (leaves the token for parse_match). Not yet ported.
        if self.check(TokenType::Match) {
            return self.parse_match();
        }

        if self.match_(TokenType::If) {
            let (cond, then_body, else_body) = self.parse_if_tail(false)?;
            let e = self.push_if_extra(&then_body, &else_body);
            return Ok(self.emit(AstNode {
                tag: AstTag::IfNode,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: cond.raw(),
                rhs: e.raw(),
            }));
        }

        if self.match_(TokenType::While) {
            let (cond, body) = self.parse_paren_cond_block("'while'", "while")?;
            let e = self.push_body_extra(&body);
            return Ok(self.emit(AstNode {
                tag: AstTag::WhileNode,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: cond.raw(),
                rhs: e.raw(),
            }));
        }

        if self.match_(TokenType::Loop) {
            // Desugars to while(true). Emit the BoolLit cond FIRST (node-id
            // ordering must match the C++).
            let cond = self.emit(AstNode {
                tag: AstTag::BoolLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 1,
                rhs: 0,
            });
            self.consume(TokenType::OpenBrace, "Expected '{' after 'loop'")?;
            let body = self.parse_stmts_until_brace()?;
            self.consume(TokenType::CloseBrace, "Expected '}' after loop body")?;
            let e = self.push_body_extra(&body);
            return Ok(self.emit(AstNode {
                tag: AstTag::WhileNode,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: cond.raw(),
                rhs: e.raw(),
            }));
        }

        if self.match_(TokenType::For) {
            self.consume(TokenType::Identifier, "Expected variable name after 'for'")?;
            let var = self.prev_text().to_vec();
            let var_id = self.intern_str(&var);
            self.consume(TokenType::In, "Expected 'in' after for variable")?;
            // Bounds use parse_comparison (no ||/&&) with struct-lit disabled.
            let prev_allow = self.allow_struct_lit;
            self.allow_struct_lit = false;
            let start = self.parse_comparison()?;
            self.consume(TokenType::Colon, "Expected ':' in for range")?;
            let end = self.parse_comparison()?;
            self.allow_struct_lit = prev_allow;
            self.consume(TokenType::OpenBrace, "Expected '{' after for range")?;
            let body = self.parse_stmts_until_brace()?;
            self.consume(TokenType::CloseBrace, "Expected '}' after for body")?;
            let e = self.nodes.reserve_extra(4 + body.len());
            self.nodes.set_extra(e, var_id.raw());
            self.nodes
                .set_extra(ExtraIdx::new(e.raw() + 1), start.raw());
            self.nodes.set_extra(ExtraIdx::new(e.raw() + 2), end.raw());
            self.nodes
                .set_extra(ExtraIdx::new(e.raw() + 3), body.len() as u32);
            for (i, b) in body.iter().enumerate() {
                self.nodes
                    .set_extra(ExtraIdx::new(e.raw() + 4 + i as u32), b.raw());
            }
            return Ok(self.emit(AstNode {
                tag: AstTag::ForNode,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: e.raw(),
                rhs: 0,
            }));
        }

        if self.match_(TokenType::Break) {
            self.consume(TokenType::Semi, "Expected ';' after break")?;
            return Ok(self.emit(AstNode {
                tag: AstTag::Break,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0,
                rhs: 0,
            }));
        }
        if self.match_(TokenType::Continue) {
            self.consume(TokenType::Semi, "Expected ';' after continue")?;
            return Ok(self.emit(AstNode {
                tag: AstTag::Continue,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0,
                rhs: 0,
            }));
        }

        // Leading identifier: assignment, call-statement, or bare expr value.
        if self.check(TokenType::Identifier) {
            let target = self.parse_comparison()?;
            if self.match_(TokenType::Equal) {
                let value = self.parse_logical_or()?;
                self.consume(TokenType::Semi, "Expected ';' after assignment")?;
                return Ok(self.emit(AstNode {
                    tag: AstTag::Assign,
                    op: 0,
                    flags: 0,
                    main_token: 0,
                    lhs: target.raw(),
                    rhs: value.raw(),
                }));
            }
            // Call-statement consumes `;`; a bare Variable/MemberAccess value
            // (e.g. a block's final expression) does not.
            if self.nodes.get(target).tag == AstTag::Call {
                self.consume(TokenType::Semi, "Expected ';' after function call")?;
            }
            return Ok(target);
        }

        // Bare non-identifier expression (e.g. `@call(...)`). `@`-call statements
        // consume `;`.
        let expr = self.parse_logical_or()?;
        if self.nodes.get(expr).tag == AstTag::AtCall {
            self.consume(TokenType::Semi, "Expected ';' after `@`-call statement")?;
        }
        Ok(expr)
    }

    /// Shared if/comp-if tail: parses `( cond ) { then } [else ...]`, returning
    /// `(cond, then_body, else_body)`. `is_comp` controls the else-if guard
    /// (`else comp if` chains too) and the error messages.
    fn parse_if_tail(&mut self, is_comp: bool) -> PResult<(NodeIdx, Vec<NodeIdx>, Vec<NodeIdx>)> {
        // The C++ writes the comp-if keyword as `` `comp if` `` but the plain
        // keyword as `'if'`/bare `if`; match each message byte-for-byte.
        let (kw_quoted, kw) = if is_comp {
            ("`comp if`", "`comp if`")
        } else {
            ("'if'", "if")
        };
        self.consume(
            TokenType::OpenParen,
            &format!("Expected '(' after {kw_quoted}"),
        )?;
        let cond = self.parse_logical_or()?;
        self.consume(
            TokenType::CloseParen,
            &format!("Expected ')' after {kw} condition"),
        )?;
        self.consume(
            TokenType::OpenBrace,
            &format!("Expected '{{' after {kw} condition"),
        )?;
        let then_body = self.parse_stmts_until_brace()?;
        self.consume(
            TokenType::CloseBrace,
            &format!("Expected '}}' after {kw} body"),
        )?;
        let mut else_body = Vec::new();
        if self.match_(TokenType::Else) {
            let chains = self.check(TokenType::If) || (is_comp && self.check(TokenType::Comp));
            if chains {
                // `else if` / `else comp if`: the chained construct is one stmt.
                else_body.push(self.parse_expression()?);
            } else {
                let msg = if is_comp {
                    "Expected '{' or `if` / `comp if` after 'else'"
                } else {
                    "Expected '{' or 'if' after 'else'"
                };
                self.consume(TokenType::OpenBrace, msg)?;
                else_body = self.parse_stmts_until_brace()?;
                let close_msg = if is_comp {
                    "Expected '}' after `comp if` else body"
                } else {
                    "Expected '}' after else body"
                };
                self.consume(TokenType::CloseBrace, close_msg)?;
            }
        }
        Ok((cond, then_body, else_body))
    }

    fn parse_match(&mut self) -> PResult<NodeIdx> {
        self.consume(TokenType::Match, "Expected `match`")?;
        self.consume(TokenType::OpenParen, "Expected `(` after `match`")?;
        let scrutinee = self.parse_logical_or()?;
        self.consume(TokenType::CloseParen, "Expected `)` after match scrutinee")?;
        self.consume(TokenType::OpenBrace, "Expected `{` to begin match body")?;

        // Each arm: (pattern NodeIdx, body NodeIdx list).
        let mut arms: Vec<(NodeIdx, Vec<NodeIdx>)> = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            let pat = self.parse_pattern()?;
            self.consume(TokenType::OpenBrace, "Expected `{` to begin arm body")?;
            let mut body = Vec::new();
            while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
                body.push(self.parse_expression()?);
            }
            self.consume(TokenType::CloseBrace, "Expected `}` to close arm body")?;
            arms.push((pat, body));
        }
        self.consume(TokenType::CloseBrace, "Expected `}` to close match body")?;

        // Pack arms: [armCount, (patIdx, bodyCount, body...)...]. The total
        // precompute and fill loop must walk an identical slot count.
        let mut total = 1usize; // armCount
        for (_, body) in &arms {
            total += 2 + body.len(); // patIdx + bodyCount + body...
        }
        let extra = self.nodes.reserve_extra(total);
        let mut pos = 0u32;
        self.nodes.set_extra(extra, arms.len() as u32);
        pos += 1;
        for (pat, body) in &arms {
            self.nodes
                .set_extra(ExtraIdx::new(extra.raw() + pos), pat.raw());
            pos += 1;
            self.nodes
                .set_extra(ExtraIdx::new(extra.raw() + pos), body.len() as u32);
            pos += 1;
            for s in body {
                self.nodes
                    .set_extra(ExtraIdx::new(extra.raw() + pos), s.raw());
                pos += 1;
            }
        }

        Ok(self.emit(AstNode {
            tag: AstTag::MatchNode,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: scrutinee.raw(),
            rhs: extra.raw(),
        }))
    }

    /// `atom (| atom)*` — a lone atom is returned unwrapped (no `PatOr`).
    fn parse_pattern(&mut self) -> PResult<NodeIdx> {
        let first = self.parse_pattern_atom()?;
        if !self.check(TokenType::Pipe) {
            return Ok(first);
        }
        let mut alternatives = vec![first];
        while self.match_(TokenType::Pipe) {
            alternatives.push(self.parse_pattern_atom()?);
        }
        let extra = self.nodes.reserve_extra(1 + alternatives.len());
        self.nodes.set_extra(extra, alternatives.len() as u32);
        for (i, a) in alternatives.iter().enumerate() {
            self.nodes
                .set_extra(ExtraIdx::new(extra.raw() + 1 + i as u32), a.raw());
        }
        Ok(self.emit(AstNode {
            tag: AstTag::PatOr,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: extra.raw(),
            rhs: 0,
        }))
    }

    /// Parse a binding list `(name, name, ...)` after a consumed `(`. Empty
    /// `()` is permitted. Returns the interned binding-name StringIdxs.
    fn parse_pattern_bindings(&mut self) -> PResult<Vec<StringIdx>> {
        let mut bindings = Vec::new();
        if !self.check(TokenType::CloseParen) {
            loop {
                self.consume(
                    TokenType::Identifier,
                    "Expected binding name in variant payload",
                )?;
                let b = self.prev_text().to_vec();
                bindings.push(self.intern_str(&b));
                if !self.match_(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(
            TokenType::CloseParen,
            "Expected `)` to close payload bindings",
        )?;
        Ok(bindings)
    }

