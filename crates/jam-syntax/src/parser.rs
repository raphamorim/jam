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

    fn parse_pattern_atom(&mut self) -> PResult<NodeIdx> {
        // (A) Generic-typed enum variant `Ident(typeArgs).Variant` —
        // detected by a balanced-paren scan to `).Ident`.
        if self.check(TokenType::Identifier) && self.prev_is_not_wildcard_peek() {
            let peek_idx = self.current + 1;
            if self.tokens.get(peek_idx).map(|t| t.ttype) == Some(TokenType::OpenParen) {
                let mut depth = 1;
                let mut scan = peek_idx + 1;
                while scan < self.tokens.len() && depth > 0 {
                    match self.tokens[scan].ttype {
                        TokenType::OpenParen => depth += 1,
                        TokenType::CloseParen => depth -= 1,
                        _ => {}
                    }
                    if depth == 0 {
                        break;
                    }
                    scan += 1;
                }
                if depth == 0
                    && scan + 2 < self.tokens.len()
                    && self.tokens[scan].ttype == TokenType::CloseParen
                    && self.tokens[scan + 1].ttype == TokenType::Dot
                    && self.tokens[scan + 2].ttype == TokenType::Identifier
                {
                    let type_name = self.tokens[self.current].text(&self.source).to_vec();
                    self.advance(); // IDENT
                    self.advance(); // `(`
                    let mut type_args = Vec::new();
                    if !self.check(TokenType::CloseParen) {
                        type_args.push(self.parse_type()?);
                        while self.match_(TokenType::Comma) {
                            type_args.push(self.parse_type()?);
                        }
                    }
                    self.consume(
                        TokenType::CloseParen,
                        "Expected ')' after generic type arguments",
                    )?;
                    self.consume(TokenType::Dot, "Expected '.' after generic type")?;
                    self.consume(TokenType::Identifier, "Expected variant name after `.`")?;
                    let vn = self.prev_text().to_vec();
                    let variant_name_id = self.intern_str(&vn);
                    let tn = self.intern_str(&type_name);
                    let receiver_ty = self.type_pool.intern_generic_call(tn, type_args);

                    if self.match_(TokenType::OpenParen) {
                        let bindings = self.parse_pattern_bindings()?;
                        // flags = 3 (bit0 bindings | bit1 TypeIdx recv);
                        // extra = [recvTy, variant, count, b...]
                        let extra = self.nodes.reserve_extra(3 + bindings.len());
                        self.nodes.set_extra(extra, receiver_ty.raw());
                        self.nodes
                            .set_extra(ExtraIdx::new(extra.raw() + 1), variant_name_id.raw());
                        self.nodes
                            .set_extra(ExtraIdx::new(extra.raw() + 2), bindings.len() as u32);
                        for (i, b) in bindings.iter().enumerate() {
                            self.nodes
                                .set_extra(ExtraIdx::new(extra.raw() + 3 + i as u32), b.raw());
                        }
                        return Ok(self.emit(AstNode {
                            tag: AstTag::PatEnumVariant,
                            op: 0,
                            flags: 3,
                            main_token: 0,
                            lhs: extra.raw(),
                            rhs: 0,
                        }));
                    }
                    // flags = 2 (TypeIdx recv, no bindings)
                    return Ok(self.emit(AstNode {
                        tag: AstTag::PatEnumVariant,
                        op: 0,
                        flags: 2,
                        main_token: 0,
                        lhs: receiver_ty.raw(),
                        rhs: variant_name_id.raw(),
                    }));
                }
            }
        }

        // (B) Qualified/bare enum variant.
        if self.check(TokenType::Identifier) && self.prev_is_not_wildcard_peek() {
            let saved = self.current;
            self.advance(); // enum/variant name
            if self.match_(TokenType::Dot) {
                self.consume(TokenType::Identifier, "Expected variant name after `.`")?;
                // `a.Status.Ok` → receiver `a.Status` + variant `Ok`.
                let mut recv_name = self.tokens[saved].text(&self.source).to_vec();
                let mut last_seg = self.prev_text().to_vec();
                while self.match_(TokenType::Dot) {
                    self.consume(TokenType::Identifier, "Expected variant name after `.`")?;
                    recv_name.push(b'.');
                    recv_name.extend_from_slice(&last_seg);
                    last_seg = self.prev_text().to_vec();
                }
                let enum_name_id = self.intern_str(&recv_name);
                let variant_name_id = self.intern_str(&last_seg);

                if self.match_(TokenType::OpenParen) {
                    let bindings = self.parse_pattern_bindings()?;
                    // flags = 1; extra = [enum, variant, count, b...]
                    let extra = self.nodes.reserve_extra(3 + bindings.len());
                    self.nodes.set_extra(extra, enum_name_id.raw());
                    self.nodes
                        .set_extra(ExtraIdx::new(extra.raw() + 1), variant_name_id.raw());
                    self.nodes
                        .set_extra(ExtraIdx::new(extra.raw() + 2), bindings.len() as u32);
                    for (i, b) in bindings.iter().enumerate() {
                        self.nodes
                            .set_extra(ExtraIdx::new(extra.raw() + 3 + i as u32), b.raw());
                    }
                    return Ok(self.emit(AstNode {
                        tag: AstTag::PatEnumVariant,
                        op: 0,
                        flags: 1,
                        main_token: 0,
                        lhs: extra.raw(),
                        rhs: 0,
                    }));
                }
                // flags = 0; lhs = enum, rhs = variant
                return Ok(self.emit(AstNode {
                    tag: AstTag::PatEnumVariant,
                    op: 0,
                    flags: 0,
                    main_token: 0,
                    lhs: enum_name_id.raw(),
                    rhs: variant_name_id.raw(),
                }));
            }
            // Bare variant `Variant` / `Variant(bindings)` — receiver inferred.
            let bare = self.tokens[saved].text(&self.source).to_vec();
            let bare_variant_id = self.intern_str(&bare);
            if self.match_(TokenType::OpenParen) {
                let bindings = self.parse_pattern_bindings()?;
                // flags = 5 (bit0 bindings | bit2 infer-recv);
                // extra = [0, variant, count, b...]
                let extra = self.nodes.reserve_extra(3 + bindings.len());
                self.nodes.set_extra(extra, 0);
                self.nodes
                    .set_extra(ExtraIdx::new(extra.raw() + 1), bare_variant_id.raw());
                self.nodes
                    .set_extra(ExtraIdx::new(extra.raw() + 2), bindings.len() as u32);
                for (i, b) in bindings.iter().enumerate() {
                    self.nodes
                        .set_extra(ExtraIdx::new(extra.raw() + 3 + i as u32), b.raw());
                }
                return Ok(self.emit(AstNode {
                    tag: AstTag::PatEnumVariant,
                    op: 0,
                    flags: 5,
                    main_token: 0,
                    lhs: extra.raw(),
                    rhs: 0,
                }));
            }
            // flags = 4 (bit2 infer-recv, no bindings)
            return Ok(self.emit(AstNode {
                tag: AstTag::PatEnumVariant,
                op: 0,
                flags: 4,
                main_token: 0,
                lhs: 0,
                rhs: bare_variant_id.raw(),
            }));
        }

        // (C) Integer literal or inclusive range `lo..=hi`.
        if self.match_(TokenType::Number) {
            let lex = self.prev_text().to_vec();
            let (lo, is_negative, is_float) = self.parse_num_lexeme(&lex)?;
            if is_float {
                return Err(self.parse_error(
                    "Float literals are not allowed in `match` patterns \
                     (use an integer literal or a `..=` range)",
                ));
            }
            if self.match_(TokenType::DotDotEq) {
                self.consume(TokenType::Number, "Expected upper bound after `..=`")?;
                let hilex = self.prev_text().to_vec();
                let (hi, _hineg, hifloat) = self.parse_num_lexeme(&hilex)?;
                if hifloat {
                    return Err(
                        self.parse_error("Float literals are not allowed in `match` patterns")
                    );
                }
                // bounds truncate to u32
                return Ok(self.emit(AstNode {
                    tag: AstTag::PatRange,
                    op: 0,
                    flags: 0,
                    main_token: 0,
                    lhs: (lo & 0xFFFF_FFFF) as u32,
                    rhs: (hi & 0xFFFF_FFFF) as u32,
                }));
            }
            let flags: u16 = if is_negative { 1 } else { 0 };
            return Ok(self.emit(AstNode {
                tag: AstTag::PatLit,
                op: 0,
                flags,
                main_token: 0,
                lhs: (lo & 0xFFFF_FFFF) as u32,
                rhs: (lo >> 32) as u32,
            }));
        }

        // (D) Char literal — not yet supported.
        if self.match_(TokenType::StringLiteral) {
            return Err(self.parse_error("Char literals in patterns are not yet supported"));
        }

        // (E) Wildcard `_`.
        if self.check(TokenType::Identifier) && self.tokens[self.current].text(&self.source) == b"_"
        {
            self.advance();
            return Ok(self.emit(AstNode {
                tag: AstTag::PatWildcard,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0,
                rhs: 0,
            }));
        }

        Err(self.parse_error("Expected pattern (integer literal, range, or `_`)"))
    }

    /// True when the current token (an identifier) is not the `_` wildcard.
    fn prev_is_not_wildcard_peek(&self) -> bool {
        self.tokens[self.current].text(&self.source) != b"_"
    }

    // ---- unary prefixes + postfix `as` casts ----

    fn parse_unary(&mut self) -> PResult<NodeIdx> {
        // Prefix unary ops recurse; `as` casts are applied postfix via wrap_as.
        if self.match_(TokenType::Not) {
            let operand = self.parse_unary()?;
            let node = self.emit_unop(UnaryOp::LogNot, operand);
            return self.wrap_as(node);
        }
        if self.match_(TokenType::Tilde) {
            let operand = self.parse_unary()?;
            let node = self.emit_unop(UnaryOp::BitNot, operand);
            return self.wrap_as(node);
        }
        if self.match_(TokenType::Amp) {
            let operand = self.parse_unary()?;
            let node = self.emit(AstNode {
                tag: AstTag::AddressOf,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: operand.raw(),
                rhs: 0,
            });
            return self.wrap_as(node);
        }
        if self.match_(TokenType::Minus) {
            let operand = self.parse_unary()?;
            // fold a leading "-" into the literal's sign so the natural type
            // is signed. Otherwise "-16 as i64" parses as "Neg(AsCast(16,
            // i64))" where the inner 16 is a u8.
            let inner_lit = {
                let mut cur = operand;
                loop {
                    let n = self.nodes.get(cur);
                    match n.tag {
                        AstTag::NumberLit => break Some(cur),
                        AstTag::AsCast => cur = NodeIdx::new(n.lhs),
                        _ => break None,
                    }
                }
            };
            if let Some(lit_idx) = inner_lit {
                let mut lit = *self.nodes.get(lit_idx);
                lit.flags ^= 1;
                let mut new_lit = self.emit(lit);
                // if the original operand was a cast, rebuild the cast chain
                // around the sign-flipped literal so "-16 as i64" becomes
                // "AsCast(NumberLit(-16), i64)".
                if operand != lit_idx {
                    let mut casts = Vec::new();
                    let mut cur = operand;
                    while cur != lit_idx {
                        let n = self.nodes.get(cur);
                        casts.push(n.rhs);
                        cur = NodeIdx::new(n.lhs);
                    }
                    for ty in casts.into_iter().rev() {
                        new_lit = self.emit(AstNode {
                            tag: AstTag::AsCast,
                            op: 0,
                            flags: 0,
                            main_token: 0,
                            lhs: new_lit.raw(),
                            rhs: ty,
                        });
                    }
                }
                return self.wrap_as(new_lit);
            }
            let node = self.emit_unop(UnaryOp::Neg, operand);
            return self.wrap_as(node);
        }
        let p = self.parse_primary()?;
        self.wrap_as(p)
    }

    fn wrap_as(&mut self, mut node: NodeIdx) -> PResult<NodeIdx> {
        while self.match_(TokenType::As) {
            let ty = self.parse_type()?;
            node = self.emit(AstNode {
                tag: AstTag::AsCast,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: node.raw(),
                rhs: ty.raw(),
            });
        }
        Ok(node)
    }

    fn emit_unop(&mut self, op: UnaryOp, operand: NodeIdx) -> NodeIdx {
        self.emit(AstNode {
            tag: AstTag::UnaryOp,
            op: op as u8,
            flags: 0,
            main_token: 0,
            lhs: operand.raw(),
            rhs: 0,
        })
    }

    // ---- primary + postfix ----

    /// Parse a NUMBER token already consumed. Reproduces the C++
    /// f64-inline-vs-f128-quad encoding exactly. A preceding unary `-` is
    /// folded into the literal's sign flag by `parse_unary`, not here.
    fn parse_number(&mut self) -> PResult<NodeIdx> {
        let raw = self.prev_text().to_vec();
        let (mag, lex_neg, is_float) = self.parse_num_lexeme(&raw)?;
        let is_neg = lex_neg;
        let mut flags: u16 = 0;
        if is_neg {
            flags |= 1;
        }
        let (lhs, rhs) = if is_float {
            flags |= 2;
            // Re-derive the float via the f128 path (NOT parse_num_lexeme's
            // bits): strip one leading '-' and all '_', then parse.
            let mut clean: Vec<u8> = Vec::with_capacity(raw.len());
            let body = if !raw.is_empty() && raw[0] == b'-' {
                &raw[1..]
            } else {
                &raw[..]
            };
            for &c in body {
                if c != b'_' {
                    clean.push(c);
                }
            }
            let s = std::str::from_utf8(&clean).unwrap_or("");
            match float128::parse_decimal_float(s) {
                float128::ParsedFloat::Quad(q) => {
                    flags |= 4;
                    let e: ExtraIdx = self.nodes.push_extra_span(&q);
                    (e.raw(), 0)
                }
                float128::ParsedFloat::Fits64(bits) => {
                    ((bits & 0xFFFF_FFFF) as u32, (bits >> 32) as u32)
                }
            }
        } else {
            let _ = mag; // (force_neg path still uses mag for the value)
            ((mag & 0xFFFF_FFFF) as u32, (mag >> 32) as u32)
        };
        Ok(self.emit(AstNode {
            tag: AstTag::NumberLit,
            op: 0,
            flags,
            main_token: 0,
            lhs,
            rhs,
        }))
    }

    fn parse_primary(&mut self) -> PResult<NodeIdx> {
        // `match` used as an expression.
        if self.check(TokenType::Match) {
            return self.parse_match();
        }

        // `@name(...)` compiler intrinsic.
        if self.match_(TokenType::At) {
            self.consume(TokenType::Identifier, "Expected intrinsic name after '@'")?;
            let name = self.prev_text().to_vec();
            let name_id = self.intern_str(&name);
            self.consume(TokenType::OpenParen, "Expected '(' after '@name'")?;
            // Names starting with "emit" are the @-emit family; they take
            // expression args, as does `@dropInPlace(ptr)`. Everything else
            // (sizeOf, alignOf) stays on the legacy type-arg path. The test
            // order matches the C++: the emit-family/expr-intrinsic decision
            // comes FIRST, so a zero-arg emit intrinsic still encodes as the
            // expr-arg multi-form (flags=1, extra=[argCount=0]).
            let is_emit_family = name.starts_with(b"emit");
            let is_expr_intrinsic = is_emit_family || name.as_slice() == b"dropInPlace";
            if is_expr_intrinsic {
                let mut args = Vec::new();
                if !self.check(TokenType::CloseParen) {
                    args.push(self.parse_logical_or()?);
                    while self.match_(TokenType::Comma) {
                        args.push(self.parse_logical_or()?);
                    }
                }
                self.consume(
                    TokenType::CloseParen,
                    "Expected ')' after '@' intrinsic arguments",
                )?;
                let e = self.push_count_list(&args);
                let node = self.emit(AstNode {
                    tag: AstTag::AtCall,
                    op: 0,
                    flags: 1,
                    main_token: 0,
                    lhs: name_id.raw(),
                    rhs: e.raw(),
                });
                return Ok(node);
            }
            // No-arg form (`@isMacOS()`, `@isLinux()`, ...): empty parens.
            // rhs is unused — astgen dispatches purely by name.
            if self.check(TokenType::CloseParen) {
                self.consume(
                    TokenType::CloseParen,
                    "Expected ')' after '@' intrinsic name",
                )?;
                return Ok(self.emit(AstNode {
                    tag: AstTag::AtCall,
                    op: 0,
                    flags: 0,
                    main_token: 0,
                    lhs: name_id.raw(),
                    rhs: 0,
                }));
            }
            let ty = self.parse_type()?;
            self.consume(
                TokenType::CloseParen,
                "Expected ')' after '@' intrinsic argument",
            )?;
            return Ok(self.emit(AstNode {
                tag: AstTag::AtCall,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: name_id.raw(),
                rhs: ty.raw(),
            }));
        }

        if self.match_(TokenType::Number) {
            return self.parse_number();
        }
        if self.match_(TokenType::True) {
            return Ok(self.emit(AstNode {
                tag: AstTag::BoolLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 1,
                rhs: 0,
            }));
        }
        if self.match_(TokenType::False) {
            return Ok(self.emit(AstNode {
                tag: AstTag::BoolLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0,
                rhs: 0,
            }));
        }
        if self.match_(TokenType::StringLiteral) {
            let lex = self.prev_lexeme().to_vec();
            let s = self.intern_str(&lex);
            return Ok(self.emit(AstNode {
                tag: AstTag::StringLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: s.raw(),
                rhs: 0,
            }));
        }
        if self.match_(TokenType::Import) {
            self.consume(TokenType::OpenParen, "Expected '(' after 'import'")?;
            self.consume(
                TokenType::StringLiteral,
                "Expected string literal for import path",
            )?;
            let lex = self.prev_lexeme().to_vec();
            let path = self.intern_str(&lex);
            self.consume(TokenType::CloseParen, "Expected ')' after import path")?;
            return Ok(self.emit(AstNode {
                tag: AstTag::ImportLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: path.raw(),
                rhs: 0,
            }));
        }
        if self.match_(TokenType::OpenParen) {
            let inner = self.parse_logical_or()?;
            self.consume(TokenType::CloseParen, "Expected ')' after expression")?;
            return Ok(inner); // parens are transparent
        }
        if self.check(TokenType::OpenBrace) {
            return Err(self.parse_error(
                "Struct literals must name their type: write `TypeName { ... }` \
                 (or `Self { ... }` inside a struct body)",
            ));
        }
        if self.match_(TokenType::OpenBracket) {
            return self.parse_array_literal();
        }

        // Anonymous `struct { ... }` in expression position (the generic-type
        // factory pattern `fn Box(T: type) type { return struct {...}; }`).
        if self.match_(TokenType::Struct) {
            return self.parse_struct_expression();
        }
        // Same shape for tagged sum types: `enum { V1, V2(T), ... }`.
        if self.match_(TokenType::Enum) {
            return self.parse_enum_expression();
        }

        // Identifier / builtin-type-name: a Variable, optionally followed by a
        // postfix chain of `.member` / `[index]` / `(call)`.
        if self.match_(TokenType::Identifier) || self.match_(TokenType::Type) {
            let name = self.prev_text().to_vec();
            let name_id = self.intern_str(&name);
            let name_str = String::from_utf8_lossy(&name).into_owned();
            // Typed struct literal: `Foo { ... }` (or `Self { ... }`).
            if self.allow_struct_lit && self.check(TokenType::OpenBrace) {
                self.advance(); // '{'
                let lit = self.parse_struct_literal()?;
                let type_name_id = if name.as_slice() == b"Self" {
                    match self.struct_context_stack.last() {
                        Some(ctx) => {
                            let c = ctx.clone();
                            self.intern_str(c.as_bytes())
                        }
                        None => {
                            return Err(
                                self.parse_error("`Self` is only valid inside a struct body")
                            );
                        }
                    }
                } else {
                    name_id
                };
                let named_ty = self.type_pool.intern_named(type_name_id);
                self.nodes.get_mut(lit).lhs = named_ty.raw();
                return Ok(lit);
            }
            let mut expr = self.emit(AstNode {
                tag: AstTag::Variable,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: name_id.raw(),
                rhs: 0,
            });
            let mut chain_started = false;
            loop {
                if self.match_(TokenType::Dot) {
                    chain_started = true;
                    if self.match_(TokenType::Star) {
                        expr = self.emit(AstNode {
                            tag: AstTag::Deref,
                            op: 0,
                            flags: 0,
                            main_token: 0,
                            lhs: expr.raw(),
                            rhs: 0,
                        });
                    } else {
                        self.consume(TokenType::Identifier, "Expected member name after '.'")?;
                        let m = self.prev_text().to_vec();
                        let mem = self.intern_str(&m);
                        expr = self.emit(AstNode {
                            tag: AstTag::MemberAccess,
                            op: 0,
                            flags: 0,
                            main_token: 0,
                            lhs: expr.raw(),
                            rhs: mem.raw(),
                        });
                    }
                } else if self.match_(TokenType::OpenBracket) {
                    chain_started = true;
                    let start = self.parse_logical_or()?;
                    if self.match_(TokenType::DotDot) {
                        let end = self.parse_logical_or()?;
                        self.consume(TokenType::CloseBracket, "Expected ']' after slice range")?;
                        let sx = self.nodes.reserve_extra(2);
                        self.nodes.set_extra(sx, start.raw());
                        self.nodes.set_extra(ExtraIdx::new(sx.raw() + 1), end.raw());
                        expr = self.emit(AstNode {
                            tag: AstTag::Slice,
                            op: 0,
                            flags: 0,
                            main_token: 0,
                            lhs: expr.raw(),
                            rhs: sx.raw(),
                        });
                    } else {
                        self.consume(TokenType::CloseBracket, "Expected ']' after index")?;
                        expr = self.emit(AstNode {
                            tag: AstTag::Index,
                            op: 0,
                            flags: 0,
                            main_token: 0,
                            lhs: expr.raw(),
                            rhs: start.raw(),
                        });
                    }
                } else if self.check(TokenType::OpenParen) {
                    let is_namespaced_typecall =
                        chain_started && self.import_handles.contains(&name_str);
                    if !chain_started || is_namespaced_typecall {
                        match self.typecall_lookahead() {
                            TcShape::GenericStructLit => {
                                // `Vec(i32) { ... }` — a generic struct literal.
                                self.advance(); // '('
                                let mut type_args = Vec::new();
                                if !self.check(TokenType::CloseParen) {
                                    type_args.push(self.parse_type()?);
                                    while self.match_(TokenType::Comma) {
                                        type_args.push(self.parse_type()?);
                                    }
                                }
                                self.consume(
                                    TokenType::CloseParen,
                                    "Expected ')' after type arguments",
                                )?;
                                self.consume(
                                    TokenType::OpenBrace,
                                    "Expected '{' after generic type arguments",
                                )?;
                                let receiver = if is_namespaced_typecall {
                                    self.qualified_name(expr)?
                                } else {
                                    name_str.clone()
                                };
                                let rid = self.intern_str(receiver.as_bytes());
                                let generic_ty = self.type_pool.intern_generic_call(rid, type_args);
                                let lit = self.parse_struct_literal()?;
                                self.nodes.get_mut(lit).lhs = generic_ty.raw();
                                return Ok(lit);
                            }
                            TcShape::TypeMethodCall => {
                                // `Vec(i32).empty(args)` — a static method call
                                // on a generic-call receiver type.
                                self.advance(); // '('
                                let mut type_args = Vec::new();
                                if !self.check(TokenType::CloseParen) {
                                    type_args.push(self.parse_type()?);
                                    while self.match_(TokenType::Comma) {
                                        type_args.push(self.parse_type()?);
                                    }
                                }
                                self.consume(
                                    TokenType::CloseParen,
                                    "Expected ')' after type arguments",
                                )?;
                                self.consume(TokenType::Dot, "Expected '.' after generic call")?;
                                self.consume(
                                    TokenType::Identifier,
                                    "Expected method name after '.'",
                                )?;
                                let mname = self.prev_text().to_vec();
                                let mid = self.intern_str(&mname);
                                self.consume(
                                    TokenType::OpenParen,
                                    "Expected '(' after method name",
                                )?;
                                let mut margs = Vec::new();
                                if !self.check(TokenType::CloseParen) {
                                    margs.push(self.parse_comparison()?);
                                    while self.match_(TokenType::Comma) {
                                        margs.push(self.parse_comparison()?);
                                    }
                                }
                                self.consume(
                                    TokenType::CloseParen,
                                    "Expected ')' after method arguments",
                                )?;
                                let receiver = if is_namespaced_typecall {
                                    self.qualified_name(expr)?
                                } else {
                                    name_str.clone()
                                };
                                let rid = self.intern_str(receiver.as_bytes());
                                let receiver_ty =
                                    self.type_pool.intern_generic_call(rid, type_args);
                                let extra = self.nodes.reserve_extra(2 + margs.len());
                                self.nodes.set_extra(extra, mid.raw());
                                self.nodes
                                    .set_extra(ExtraIdx::new(extra.raw() + 1), margs.len() as u32);
                                for (i, a) in margs.iter().enumerate() {
                                    self.nodes.set_extra(
                                        ExtraIdx::new(extra.raw() + 2 + i as u32),
                                        a.raw(),
                                    );
                                }
                                expr = self.emit(AstNode {
                                    tag: AstTag::TypeMethodCall,
                                    op: 0,
                                    flags: 0,
                                    main_token: 0,
                                    lhs: receiver_ty.raw(),
                                    rhs: extra.raw(),
                                });
                                chain_started = true;
                                continue;
                            }
                            TcShape::None => {}
                        }
                    }
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    if !self.check(TokenType::CloseParen) {
                        args.push(self.parse_comparison()?);
                        while self.match_(TokenType::Comma) {
                            args.push(self.parse_comparison()?);
                        }
                    }
                    self.consume(
                        TokenType::CloseParen,
                        "Expected ')' after function arguments",
                    )?;
                    let e = self.push_count_list(&args);
                    expr = self.emit_call(expr, &name, e);
                    chain_started = true;
                } else {
                    break;
                }
            }
            // Qualified-name struct literal: `a.b { ... }` (only for a
            // MemberAccess chain).
            if self.allow_struct_lit
                && self.check(TokenType::OpenBrace)
                && self.is_qualified_name_chain(expr)
                && self.nodes.get(expr).tag == AstTag::MemberAccess
            {
                self.advance(); // '{'
                let qname = self.qualified_name(expr)?;
                let nid = self.intern_str(qname.as_bytes());
                let named_ty = self.type_pool.intern_named(nid);
                let lit = self.parse_struct_literal()?;
                self.nodes.get_mut(lit).lhs = named_ty.raw();
                return Ok(lit);
            }
            return Ok(expr);
        }

        Err(self.parse_error("Expected primary expression"))
    }

    /// Balanced-paren lookahead from a `(` at `current`: classify `name(...)` as
    /// a generic struct literal (`)` then `{`), a generic type-method-call (`)`
    /// then `.id(`), or neither (a plain call).
    fn typecall_lookahead(&self) -> TcShape {
        let mut peek_idx = self.current + 1;
        let mut depth = 1;
        while peek_idx < self.tokens.len() && depth > 0 {
            match self.tokens[peek_idx].ttype {
                TokenType::OpenParen => depth += 1,
                TokenType::CloseParen => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                break;
            }
            peek_idx += 1;
        }
        if depth != 0 {
            return TcShape::None;
        }
        let after = self.tokens.get(peek_idx + 1).map(|t| t.ttype);
        if self.allow_struct_lit && after == Some(TokenType::OpenBrace) {
            return TcShape::GenericStructLit;
        }
        if after == Some(TokenType::Dot)
            && self.tokens.get(peek_idx + 2).map(|t| t.ttype) == Some(TokenType::Identifier)
            && self.tokens.get(peek_idx + 3).map(|t| t.ttype) == Some(TokenType::OpenParen)
        {
            TcShape::TypeMethodCall
        } else {
            TcShape::None
        }
    }

    /// Struct-literal body: `{ ident : expr (, ident : expr)* ,? }` with the
    /// opening `{` already consumed. Emits `StructLit` with `lhs = kNoType`
    /// (the caller patches it to the receiver type).
    fn parse_struct_literal(&mut self) -> PResult<NodeIdx> {
        let mut fields: Vec<(StringIdx, NodeIdx)> = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            self.consume(
                TokenType::Identifier,
                "Expected field name in struct literal",
            )?;
            let fname = self.prev_text().to_vec();
            let fid = self.intern_str(&fname);
            self.consume(TokenType::Colon, "Expected ':' after field name")?;
            let value = self.parse_logical_or()?;
            fields.push((fid, value));
            if !self.match_(TokenType::Comma) {
                break;
            }
        }
        self.consume(
            TokenType::CloseBrace,
            "Expected '}' to close struct literal",
        )?;
        let extra = self.nodes.reserve_extra(1 + fields.len() * 2);
        self.nodes.set_extra(extra, fields.len() as u32);
        for (i, (fid, val)) in fields.iter().enumerate() {
            self.nodes
                .set_extra(ExtraIdx::new(extra.raw() + 1 + (i * 2) as u32), fid.raw());
            self.nodes
                .set_extra(ExtraIdx::new(extra.raw() + 2 + (i * 2) as u32), val.raw());
        }
        Ok(self.emit(AstNode {
            tag: AstTag::StructLit,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: 0, // kNoType — patched by the caller
            rhs: extra.raw(),
        }))
    }

    /// Emit a `Call` node, choosing the direct (StringIdx callee) vs indirect
    /// (NodeIdx receiver, flags bit 0) encoding exactly as the C++ does.
    fn emit_call(&mut self, expr: NodeIdx, bare_name: &[u8], args_extra: ExtraIdx) -> NodeIdx {
        if self.is_qualified_name_chain(expr) {
            let dot_count = self.chain_dot_count(expr);
            let root = self.chain_root_name(expr);
            let root_is_import = self.import_handles.contains(&root);
            let is_multi_dot_local = dot_count >= 2 && !root_is_import;
            if is_multi_dot_local {
                return self.emit(AstNode {
                    tag: AstTag::Call,
                    op: 0,
                    flags: 1,
                    main_token: 0,
                    lhs: expr.raw(),
                    rhs: args_extra.raw(),
                });
            }
            let callee = if self.nodes.get(expr).tag == AstTag::MemberAccess {
                self.qualified_name(expr).unwrap_or_default()
            } else {
                String::from_utf8_lossy(bare_name).into_owned()
            };
            let callee_id = self.intern_str(callee.as_bytes());
            self.emit(AstNode {
                tag: AstTag::Call,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: callee_id.raw(),
                rhs: args_extra.raw(),
            })
        } else {
            self.emit(AstNode {
                tag: AstTag::Call,
                op: 0,
                flags: 1,
                main_token: 0,
                lhs: expr.raw(),
                rhs: args_extra.raw(),
            })
        }
    }

    /// `[count, item0, item1, ...]` in the extra pool; returns the start index.
    fn push_count_list(&mut self, items: &[NodeIdx]) -> ExtraIdx {
        let e = self.nodes.reserve_extra(1 + items.len());
        self.nodes.set_extra(e, items.len() as u32);
        for (i, it) in items.iter().enumerate() {
            self.nodes
                .set_extra(ExtraIdx::new(e.raw() + 1 + i as u32), it.raw());
        }
        e
    }

    fn parse_array_literal(&mut self) -> PResult<NodeIdx> {
        // `[` already consumed. Empty: `]`.
        if self.match_(TokenType::CloseBracket) {
            let e = self.nodes.reserve_extra(1);
            self.nodes.set_extra(e, 0);
            return Ok(self.emit(AstNode {
                tag: AstTag::ArrayLit,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0, // kNoType
                rhs: e.raw(),
            }));
        }
        let first = self.parse_logical_or()?;
        if self.match_(TokenType::Semi) {
            // `[expr; N]` repeat.
            let count = self.parse_logical_or()?;
            self.consume(
                TokenType::CloseBracket,
                "Expected `]` after array repeat count",
            )?;
            let e = self.nodes.reserve_extra(2);
            self.nodes.set_extra(e, first.raw());
            self.nodes
                .set_extra(ExtraIdx::new(e.raw() + 1), count.raw());
            return Ok(self.emit(AstNode {
                tag: AstTag::ArrayRepeat,
                op: 0,
                flags: 0,
                main_token: 0,
                lhs: 0,
                rhs: e.raw(),
            }));
        }
        let mut elems = vec![first];
        while self.match_(TokenType::Comma) {
            if self.check(TokenType::CloseBracket) {
                break; // trailing comma
            }
            elems.push(self.parse_logical_or()?);
        }
        self.consume(
            TokenType::CloseBracket,
            "Expected `]` or `,` in array literal",
        )?;
        let e = self.push_count_list(&elems);
        Ok(self.emit(AstNode {
            tag: AstTag::ArrayLit,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: 0,
            rhs: e.raw(),
        }))
    }

    // ===================================================================
    // Types (core forms; full parse_type lands with installment 2)
    // ===================================================================

    fn parse_type(&mut self) -> PResult<TypeIdx> {
        // (1) `fn(T1, T2, ...) Ret` function type.
        if self.match_(TokenType::Fn) {
            self.consume(TokenType::OpenParen, "Expected `(` after `fn` in type")?;
            let mut params = Vec::new();
            if !self.check(TokenType::CloseParen) {
                params.push(self.parse_type()?);
                while self.match_(TokenType::Comma) {
                    params.push(self.parse_type()?);
                }
            }
            self.consume(
                TokenType::CloseParen,
                "Expected `)` after fn-type parameters",
            )?;
            let ret = self.parse_type()?;
            return Ok(self.type_pool.intern_fn(ret, params));
        }
        // (2) `*const|*mut [optional []] T` pointer. Mutability is not encoded
        // (*const and *mut intern to the same TypeIdx).
        if self.match_(TokenType::Star) {
            if !self.match_(TokenType::Const) {
                let _ = self.match_(TokenType::Mut);
            }
            // Many-item only if `[` is immediately followed by `]`; otherwise
            // the `[` belongs to a pointer-to-array element type.
            let mut is_many = false;
            if self.check(TokenType::OpenBracket) {
                let saved = self.current;
                self.advance(); // '['
                if self.match_(TokenType::CloseBracket) {
                    is_many = true;
                } else {
                    self.current = saved;
                }
            }
            let elem = self.parse_type()?;
            return Ok(if is_many {
                self.type_pool.intern_ptr_many(elem)
            } else {
                self.type_pool.intern_ptr_single(elem)
            });
        }
        // (3) `[]T` slice, `[N]T` literal array, or `[expr]T` deferred array.
        if self.match_(TokenType::OpenBracket) {
            if self.match_(TokenType::CloseBracket) {
                let elem = self.parse_type()?;
                return Ok(self.type_pool.intern_slice(elem));
            }
            // Literal fast-path: a plain `Number` immediately followed by `]`.
            if self.check(TokenType::Number)
                && self.tokens.get(self.current + 1).map(|t| t.ttype)
                    == Some(TokenType::CloseBracket)
            {
                self.advance(); // number
                let raw = self.prev_text().to_vec();
                let (mag, _neg, _is_float) = self.parse_num_lexeme(&raw)?;
                self.consume(TokenType::CloseBracket, "Expected `]` after array size")?;
                let elem = self.parse_type()?;
                return Ok(self.type_pool.intern_array(elem, mag as u32));
            }
            // Expression length -> ArrayExpr (folded later in codegen). Uses
            // parse_logical_or (not parse_expression, which would want a `;`).
            let len_expr = self.parse_logical_or()?;
            self.consume(TokenType::CloseBracket, "Expected `]` after array size")?;
            let elem = self.parse_type()?;
            return Ok(self.type_pool.intern_array_expr(elem, len_expr));
        }
        // (4) builtin keyword. `str` is sugar for `[]u8`; everything else maps
        // to a fixed BuiltinType index.
        if self.match_(TokenType::Type) {
            let name = self.prev_text().to_vec();
            if name.as_slice() == b"str" {
                return Ok(self.type_pool.intern_slice(builtin::U8));
            }
            return Ok(self.builtin_type(&name));
        }
        // (5) identifier: `Self`, a dotted qualified name, or a generic call.
        if self.match_(TokenType::Identifier) {
            let first = self.prev_text().to_vec();
            // Build the (possibly dotted) name; `Self` resolves to the
            // enclosing struct.
            let mut dotted = if first.as_slice() == b"Self" {
                match self.struct_context_stack.last() {
                    Some(c) => c.clone(),
                    None => {
                        return Err(self.parse_error("`Self` is only valid inside a struct body"));
                    }
                }
            } else {
                String::from_utf8_lossy(&first).into_owned()
            };
            while self.match_(TokenType::Dot) {
                self.consume(TokenType::Identifier, "Expected type name after `.`")?;
                dotted.push('.');
                dotted.push_str(&String::from_utf8_lossy(self.prev_text()));
            }
            // `Name(args)` generic-call type.
            if self.check(TokenType::OpenParen) {
                self.advance(); // '('
                let mut args = Vec::new();
                if !self.check(TokenType::CloseParen) {
                    args.push(self.parse_type()?);
                    while self.match_(TokenType::Comma) {
                        args.push(self.parse_type()?);
                    }
                }
                self.consume(
                    TokenType::CloseParen,
                    "Expected ')' after generic type arguments",
                )?;
                let nid = self.intern_str(dotted.as_bytes());
                return Ok(self.type_pool.intern_generic_call(nid, args));
            }
            let nid = self.intern_str(dotted.as_bytes());
            return Ok(self.type_pool.intern_named(nid));
        }
        Err(self.parse_error("Expected type"))
    }

    // ===================================================================
    // Declarations + top-level parse()
    // ===================================================================

    fn parse_function(&mut self) -> PResult<FunctionAST> {
        // Modifiers in any order, each at most once.
        let mut is_extern = false;
        let mut is_export = false;
        let mut is_pub = false;
        let mut is_test = false;
        loop {
            if !is_extern && self.match_(TokenType::Extern) {
                is_extern = true;
            } else if !is_export && self.match_(TokenType::Export) {
                is_export = true;
            } else if !is_pub && self.match_(TokenType::Pub) {
                is_pub = true;
            } else if !is_test && self.match_(TokenType::Tfn) {
                is_test = true;
            } else {
                break;
            }
        }
        let mut is_cfn = false;
        if !is_test {
            // `tfn` already consumed the keyword; otherwise expect fn/cfn.
            if self.match_(TokenType::Cfn) {
                is_cfn = true;
            } else {
                self.consume(TokenType::Fn, "Expected 'fn' or 'cfn' keyword")?;
            }
        }
        self.consume(TokenType::Identifier, "Expected function name")?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        self.consume(TokenType::OpenParen, "Expected '(' after function name")?;

        let mut args: Vec<crate::ast::Param> = Vec::new();
        let mut is_var_args = false;
        if !self.check(TokenType::CloseParen) {
            loop {
                if self.check(TokenType::Ellipsis) {
                    self.advance();
                    if !is_extern {
                        return Err(
                            self.parse_error("`...` is only allowed in extern fn declarations")
                        );
                    }
                    is_var_args = true;
                    break;
                }
                let is_comp = self.match_(TokenType::Comp);
                self.consume(TokenType::Identifier, "Expected parameter name")?;
                let pname = String::from_utf8_lossy(self.prev_text()).into_owned();
                self.consume(TokenType::Colon, "Expected ':' after parameter name")?;
                let mode = if self.match_(TokenType::Mut) {
                    jam_core::ParamMode::Mut
                } else if self.match_(TokenType::Move) {
                    jam_core::ParamMode::Move
                } else {
                    jam_core::ParamMode::Let
                };
                let pty = self.parse_type()?;
                args.push(crate::ast::Param {
                    name: pname,
                    ty: pty,
                    mode,
                    is_comp,
                });
                if !self.match_(TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(TokenType::CloseParen, "Expected ')' after parameters")?;

        // Return type only when the next token can start one.
        let return_type = if matches!(
            self.peek_type(),
            TokenType::Type
                | TokenType::OpenBracket
                | TokenType::Const
                | TokenType::Identifier
                | TokenType::Star
        ) {
            self.parse_type()?
        } else {
            TypeIdx::NONE
        };

        if is_extern {
            self.consume(
                TokenType::Semi,
                "Expected ';' after extern function declaration",
            )?;
            let mut f = FunctionAST::new(name, args, return_type, Vec::new());
            f.is_extern = true;
            f.is_export = is_export;
            f.is_pub = is_pub;
            f.is_test = false; // forced (matches C++)
            f.is_var_args = is_var_args;
            f.is_cfn = is_cfn;
            return Ok(f);
        }
        self.consume(TokenType::OpenBrace, "Expected '{' before function body")?;
        let mut body = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            body.push(self.parse_expression()?);
        }
        self.consume(TokenType::CloseBrace, "Expected '}' after function body")?;
        let mut f = FunctionAST::new(name, args, return_type, body);
        f.is_export = is_export;
        f.is_pub = is_pub;
        f.is_test = is_test;
        f.is_var_args = false; // forced (matches C++)
        f.is_cfn = is_cfn;
        Ok(f)
    }

    #[allow(clippy::type_complexity)]
    fn parse_struct_body(&mut self) -> PResult<(Vec<(String, TypeIdx)>, Vec<FunctionAST>)> {
        let mut fields = Vec::new();
        let mut methods = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            if matches!(
                self.peek_type(),
                TokenType::Fn
                    | TokenType::Cfn
                    | TokenType::Pub
                    | TokenType::Extern
                    | TokenType::Export
                    | TokenType::Tfn
            ) {
                let mut m = self.parse_function()?;
                if let Some(ctx) = self.struct_context_stack.last() {
                    m.parent_struct = ctx.clone();
                }
                methods.push(m);
                self.match_(TokenType::Comma); // optional trailing comma
            } else {
                self.consume(TokenType::Identifier, "Expected field name or 'fn'")?;
                let fname = String::from_utf8_lossy(self.prev_text()).into_owned();
                self.consume(TokenType::Colon, "Expected ':' after field name")?;
                let fty = self.parse_type()?;
                fields.push((fname, fty));
                if !self.check(TokenType::CloseBrace) {
                    self.consume(TokenType::Comma, "Expected ',' or '}' after struct field")?;
                }
            }
        }
        self.consume(
            TokenType::CloseBrace,
            "Expected '}' to close struct definition",
        )?;
        Ok((fields, methods))
    }

    fn parse_struct_decl(&mut self) -> PResult<StructDeclAST> {
        self.consume(TokenType::Const, "Expected 'const' for struct declaration")?;
        self.consume(TokenType::Identifier, "Expected struct name")?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        self.consume(TokenType::Equal, "Expected '=' after struct name")?;
        self.consume(TokenType::Struct, "Expected 'struct' keyword")?;
        self.consume(TokenType::OpenBrace, "Expected '{' after 'struct'")?;
        self.struct_context_stack.push(name.clone());
        let (fields, methods) = self.parse_struct_body()?;
        self.struct_context_stack.pop();
        self.consume(TokenType::Semi, "Expected ';' after struct declaration")?;
        let mut s = StructDeclAST::new(name, fields);
        s.methods = methods;
        Ok(s)
    }

    fn parse_enum_decl(&mut self) -> PResult<EnumDeclAST> {
        use crate::ast::EnumVariantAST;
        self.consume(TokenType::Const, "Expected 'const' for enum declaration")?;
        self.consume(TokenType::Identifier, "Expected enum name")?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        self.consume(TokenType::Equal, "Expected '=' after enum name")?;
        self.consume(TokenType::Enum, "Expected 'enum' keyword")?;
        self.consume(TokenType::OpenBrace, "Expected '{' after 'enum'")?;
        let mut variants = Vec::new();
        let mut next_discrim: u32 = 0;
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            self.consume(TokenType::Identifier, "Expected enum variant name")?;
            let vname = String::from_utf8_lossy(self.prev_text()).into_owned();
            let mut payload_types = Vec::new();
            if self.match_(TokenType::OpenParen) {
                if !self.check(TokenType::CloseParen) {
                    payload_types.push(self.parse_type()?);
                    while self.match_(TokenType::Comma) {
                        payload_types.push(self.parse_type()?);
                    }
                }
                self.consume(
                    TokenType::CloseParen,
                    "Expected `)` to close variant payload list",
                )?;
            }
            let discriminant = if self.match_(TokenType::Equal) {
                self.consume(
                    TokenType::Number,
                    "Expected integer literal after `=` in enum variant",
                )?;
                let raw = self.prev_text().to_vec();
                // Use the validator directly: a negative literal must fail
                // (not have its sign stripped).
                let r = parse_number_literal(&raw);
                if r.kind != NumberResultKind::Int {
                    return Err(
                        self.parse_error("Enum discriminant must be a non-negative integer")
                    );
                }
                let mag = r.int_value;
                if mag > 255 {
                    return Err(self.parse_error(format!(
                        "Enum discriminant {mag} is out of range; M2 enums are u8-tagged"
                    )));
                }
                let d = mag as u32;
                next_discrim = d + 1;
                d
            } else {
                let d = next_discrim;
                next_discrim += 1;
                d
            };
            variants.push(EnumVariantAST {
                name: vname,
                payload_types,
                discriminant,
            });
            if !self.match_(TokenType::Comma) {
                break;
            }
        }
        self.consume(
            TokenType::CloseBrace,
            "Expected '}' to close enum definition",
        )?;
        self.consume(TokenType::Semi, "Expected ';' after enum declaration")?;
        if variants.is_empty() {
            return Err(
                self.parse_error(format!("Enum `{name}` must declare at least one variant"))
            );
        }
        if variants.len() > 256 {
            return Err(self.parse_error(format!(
                "Enum `{name}` has more than 256 variants; M2 enums are u8-tagged"
            )));
        }
        Ok(EnumDeclAST::new(name, variants))
    }

    /// Anonymous `struct { ... }` in expression position (caller consumed
    /// `struct`). Pushes a synthetic `__anon_struct_N` decl and emits a
    /// `StructExpr` node whose `lhs` is the side-table index. The index is
    /// taken *before* parsing the body (mirrors the C++ ordering exactly).
    fn parse_struct_expression(&mut self) -> PResult<NodeIdx> {
        self.consume(TokenType::OpenBrace, "Expected '{' after 'struct'")?;
        // `lhs` (the side-table index) stays LOCAL to this module; the synthetic
        // NAME is GLOBAL across all module parses (shared-pool numbering) so the
        // interned `__anon_struct_N` strings match the C++ oracle's shared
        // anon-struct registry.
        let idx = self.anon_structs.len() as u32;
        let name = format!("__anon_struct_{}", self.anon_base + idx);
        self.struct_context_stack.push(name.clone());
        let (fields, methods) = self.parse_struct_body()?;
        self.struct_context_stack.pop();
        let mut s = StructDeclAST::new(name, fields);
        s.methods = methods;
        self.anon_structs.push(s);
        Ok(self.emit(AstNode {
            tag: AstTag::StructExpr,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: idx,
            rhs: 0,
        }))
    }

    /// Anonymous `enum { V1, V2(T), ... }` in expression position (caller
    /// consumed `enum`). No discriminant `=` and no methods — sequential
    /// discriminants from 0; emits an `EnumExpr` carrying the table index.
    fn parse_enum_expression(&mut self) -> PResult<NodeIdx> {
        use crate::ast::EnumVariantAST;
        self.consume(TokenType::OpenBrace, "Expected '{' after 'enum'")?;
        let idx = self.anon_enums.len() as u32;
        let name = format!("__anon_enum_{idx}");
        let mut variants = Vec::new();
        let mut next_discrim: u32 = 0;
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            self.consume(TokenType::Identifier, "Expected enum variant name")?;
            let vname = String::from_utf8_lossy(self.prev_text()).into_owned();
            let mut payload_types = Vec::new();
            if self.match_(TokenType::OpenParen) {
                if !self.check(TokenType::CloseParen) {
                    payload_types.push(self.parse_type()?);
                    while self.match_(TokenType::Comma) {
                        payload_types.push(self.parse_type()?);
                    }
                }
                self.consume(
                    TokenType::CloseParen,
                    "Expected `)` to close variant payload list",
                )?;
            }
            let discriminant = next_discrim;
            next_discrim += 1;
            variants.push(EnumVariantAST {
                name: vname,
                payload_types,
                discriminant,
            });
            if !self.match_(TokenType::Comma) {
                break;
            }
        }
        self.consume(TokenType::CloseBrace, "Expected '}' to close enum body")?;
        self.anon_enums.push(EnumDeclAST::new(name, variants));
        Ok(self.emit(AstNode {
            tag: AstTag::EnumExpr,
            op: 0,
            flags: 0,
            main_token: 0,
            lhs: idx,
            rhs: 0,
        }))
    }

    fn parse_union_decl(&mut self) -> PResult<UnionDeclAST> {
        self.consume(TokenType::Const, "Expected 'const' for union declaration")?;
        self.consume(TokenType::Identifier, "Expected union name")?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        self.consume(TokenType::Equal, "Expected '=' after union name")?;
        self.consume(TokenType::Union, "Expected 'union' keyword")?;
        self.consume(TokenType::OpenBrace, "Expected '{' after 'union'")?;
        let mut fields = Vec::new();
        while !self.check(TokenType::CloseBrace) && !self.is_at_end() {
            self.consume(TokenType::Identifier, "Expected union field name")?;
            let fname = String::from_utf8_lossy(self.prev_text()).into_owned();
            self.consume(TokenType::Colon, "Expected ':' after field name")?;
            let fty = self.parse_type()?;
            fields.push((fname, fty));
            if !self.match_(TokenType::Comma) {
                break;
            }
        }
        self.consume(
            TokenType::CloseBrace,
            "Expected '}' to close union definition",
        )?;
        self.consume(TokenType::Semi, "Expected ';' after union declaration")?;
        Ok(UnionDeclAST::new(name, fields))
    }

    fn parse_const_decl(&mut self) -> PResult<crate::ast::ConstDeclAST> {
        self.consume(
            TokenType::Const,
            "Expected 'const' for module-scope constant",
        )?;
        self.consume(
            TokenType::Identifier,
            "Expected identifier for constant name",
        )?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        let declared = if self.match_(TokenType::Colon) {
            self.parse_type()?
        } else {
            TypeIdx::NONE
        };
        self.consume(
            TokenType::Equal,
            "Expected '=' in module-scope const declaration",
        )?;
        let init = self.parse_logical_or()?;
        self.consume(
            TokenType::Semi,
            "Expected ';' after module-scope const declaration",
        )?;
        Ok(crate::ast::ConstDeclAST::new(name, declared, init))
    }

    fn parse_import_decl(&mut self) -> PResult<crate::ast::ImportDeclAST> {
        self.consume(TokenType::Const, "Expected 'const' for import declaration")?;
        self.consume(TokenType::Identifier, "Expected identifier for import name")?;
        let name = String::from_utf8_lossy(self.prev_text()).into_owned();
        self.consume(TokenType::Equal, "Expected '=' after import name")?;
        self.consume(TokenType::Import, "Expected 'import' keyword")?;
        self.consume(TokenType::OpenParen, "Expected '(' after 'import'")?;
        self.consume(
            TokenType::StringLiteral,
            "Expected string literal for import path",
        )?;
        let path = String::from_utf8_lossy(self.prev_lexeme()).into_owned();
        self.consume(TokenType::CloseParen, "Expected ')' after import path")?;
        let mut decl = crate::ast::ImportDeclAST::new(name.clone(), path);
        while self.match_(TokenType::Dot) {
            self.consume(
                TokenType::Identifier,
                "Expected identifier after `.` in import chain",
            )?;
            decl.chain
                .push(String::from_utf8_lossy(self.prev_text()).into_owned());
        }
        self.consume(TokenType::Semi, "Expected ';' after import declaration")?;
        self.import_handles.insert(name);
        Ok(decl)
    }

    fn parse_destructuring_import(&mut self) -> PResult<crate::ast::DestructuringImportDeclAST> {
        self.consume(
            TokenType::Const,
            "Expected 'const' for destructuring import",
        )?;
        self.consume(
            TokenType::OpenBrace,
            "Expected '{' for destructuring import",
        )?;
        let mut names = Vec::new();
        loop {
            self.consume(
                TokenType::Identifier,
                "Expected identifier in destructuring import",
            )?;
            names.push(String::from_utf8_lossy(self.prev_text()).into_owned());
            if !self.match_(TokenType::Comma) {
                break;
            }
        }
        self.consume(
            TokenType::CloseBrace,
            "Expected '}' after destructuring names",
        )?;
        self.consume(TokenType::Equal, "Expected '=' after destructuring")?;
        self.consume(TokenType::Import, "Expected 'import' keyword")?;
        self.consume(TokenType::OpenParen, "Expected '(' after 'import'")?;
        self.consume(
            TokenType::StringLiteral,
            "Expected string literal for import path",
        )?;
        let path = String::from_utf8_lossy(self.prev_lexeme()).into_owned();
        self.consume(TokenType::CloseParen, "Expected ')' after import path")?;
        let mut decl = crate::ast::DestructuringImportDeclAST::new(names, path);
        while self.match_(TokenType::Dot) {
            self.consume(
                TokenType::Identifier,
                "Expected identifier after `.` in import chain",
            )?;
            decl.chain
                .push(String::from_utf8_lossy(self.prev_text()).into_owned());
        }
        self.consume(TokenType::Semi, "Expected ';' after import declaration")?;
        for n in &decl.names {
            self.import_handles.insert(n.clone());
        }
        Ok(decl)
    }

    /// Parse a whole module.
    pub fn parse(&mut self) -> PResult<crate::ast::ModuleAST> {
        let mut module = crate::ast::ModuleAST::new();
        while !self.is_at_end() {
            let mut is_pub = false;

            // `pub` absorb: only consume it here if it precedes const/comp;
            // otherwise leave it for parse_function.
            if self.check(TokenType::Pub) {
                let pub_saved = self.current;
                self.advance();
                let next_is_const = self.check(TokenType::Const) || self.check(TokenType::Comp);
                self.current = pub_saved;
                if next_is_const {
                    self.advance(); // consume `pub`
                    is_pub = true;
                    // `pub const { ... }` (destructuring import) is rejected.
                    let peek = self.current + 1;
                    if peek < self.tokens.len() && self.tokens[peek].ttype == TokenType::OpenBrace {
                        return Err(
                            self.parse_error("`pub` is not allowed on destructuring imports")
                        );
                    }
                }
            }

            // `comp const`.
            if self.check(TokenType::Comp) {
                self.advance();
                if !self.check(TokenType::Const) {
                    return Err(
                        self.parse_error("`comp` at module scope must be followed by `const`")
                    );
                }
                let mut c = self.parse_const_decl()?;
                c.is_pub = is_pub;
                c.is_comp = true;
                module.consts.push(c);
                continue;
            }

            // `const` classification (import / destructuring / struct / union /
            // enum / typed-or-untyped const) via cursor save/restore.
            if self.check(TokenType::Const) {
                let saved = self.current;
                self.advance(); // past `const`
                if self.check(TokenType::OpenBrace) {
                    self.current = saved;
                    module
                        .destructuring_imports
                        .push(self.parse_destructuring_import()?);
                    continue;
                }
                if self.check(TokenType::Identifier) {
                    self.advance(); // past ident
                    if self.check(TokenType::Colon) {
                        self.current = saved;
                        let mut c = self.parse_const_decl()?;
                        c.is_pub = is_pub;
                        module.consts.push(c);
                        continue;
                    }
                    if self.check(TokenType::Equal) {
                        self.advance(); // past `=`
                        let rhs = self.peek_type();
                        self.current = saved;
                        match rhs {
                            TokenType::Import => {
                                let mut d = self.parse_import_decl()?;
                                d.is_pub = is_pub;
                                module.imports.push(d);
                            }
                            TokenType::Struct => {
                                let mut d = self.parse_struct_decl()?;
                                d.is_pub = is_pub;
                                module.structs.push(d);
                            }
                            TokenType::Union => {
                                let mut d = self.parse_union_decl()?;
                                d.is_pub = is_pub;
                                module.unions.push(d);
                            }
                            TokenType::Enum => {
                                let mut d = self.parse_enum_decl()?;
                                d.is_pub = is_pub;
                                module.enums.push(d);
                            }
                            _ => {
                                let mut c = self.parse_const_decl()?;
                                c.is_pub = is_pub;
                                module.consts.push(c);
                            }
                        }
                        continue;
                    }
                }
                // const + ident but neither `:` nor `=`: rewind, fall to fn.
                self.current = saved;
            }

            // Function declaration (handles its own pub/extern/export/tfn/cfn).
            let mut f = self.parse_function()?;
            if f.is_cfn {
                // A top-level `cfn` whose first param isn't `self` is a
                // compile-time function, not a drop-method.
                let first_is_self = f.args.first().map(|p| p.name == "self").unwrap_or(false);
                if !first_is_self {
                    f.is_cfn = false;
                    f.is_comp_time_fn = true;
                }
            }
            module.functions.push(f);
        }
        // Move parser-collected anonymous decls into the module (StructExpr /
        // EnumExpr nodes index into these).
        module.anon_structs = std::mem::take(&mut self.anon_structs);
        module.anon_enums = std::mem::take(&mut self.anon_enums);
        Ok(module)
    }

    fn builtin_type(&self, name: &[u8]) -> TypeIdx {
        match name {
            b"void" => builtin::VOID,
            b"bool" | b"u1" => builtin::BOOL,
            b"u8" => builtin::U8,
            b"i8" => builtin::I8,
            b"u16" => builtin::U16,
            b"i16" => builtin::I16,
            b"u32" => builtin::U32,
            b"i32" => builtin::I32,
            b"u64" => builtin::U64,
            b"i64" => builtin::I64,
            b"f32" => builtin::F32,
            b"f64" => builtin::F64,
            b"type" => builtin::TYPE,
            b"noreturn" => builtin::NORETURN,
            // `str` and others resolve later; treat as named for now.
            _ => builtin::VOID,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    /// Lex `src` and run the expression ladder; return (NodeStore, root).
    fn parse_expr(src: &str) -> (NodeStore, NodeIdx) {
        let mut lx = Lexer::new(src.as_bytes().to_vec());
        lx.scan_tokens().unwrap();
        let tokens = lx.tokens().to_vec();
        let source = lx.source().to_vec();
        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let mut diags = Diagnostics::new();
        let root = {
            let mut p = Parser::new(
                tokens, source, &mut tp, &mut sp, &mut ns, &mut diags, "t.jam",
            );
            p.parse_logical_or().expect("parse")
        };
        (ns, root)
    }

    #[test]
    fn precedence_mul_binds_tighter_than_add() {
        // 1 + 2 * 3  =>  Add(1, Mul(2,3))
        let (ns, root) = parse_expr("1 + 2 * 3");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::BinaryOp);
        assert_eq!(n.op, BinOp::Add as u8);
        let rhs = ns.get(NodeIdx::new(n.rhs));
        assert_eq!(rhs.tag, AstTag::BinaryOp);
        assert_eq!(rhs.op, BinOp::Mul as u8);
    }

    #[test]
    fn logical_below_comparison() {
        // a && b == c  =>  And(a, Eq(b,c))
        let (ns, root) = parse_expr("a && b == c");
        let n = ns.get(root);
        assert_eq!(n.op, BinOp::LogAnd as u8);
        assert_eq!(ns.get(NodeIdx::new(n.rhs)).op, BinOp::Eq as u8);
    }

    #[test]
    fn number_int_and_neg() {
        let (ns, root) = parse_expr("42");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::NumberLit);
        assert_eq!(n.lhs, 42);
        assert_eq!(n.flags & 1, 0);
        // -7 (lexed as one token) -> negative flag.
        let (ns2, r2) = parse_expr("-7");
        let n2 = ns2.get(r2);
        assert_eq!(n2.tag, AstTag::NumberLit);
        assert_eq!(n2.lhs, 7);
        assert_eq!(n2.flags & 1, 1);
    }

    #[test]
    fn unary_minus_folds_into_literal() {
        // `- 7` (whitespace keeps the lexer from gluing) folds the sign into
        // the literal, consuming both tokens.
        let (ns, root) = parse_expr("- 7");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::NumberLit);
        assert_eq!(n.lhs, 7);
        assert_eq!(n.flags & 1, 1);

        // The number token is consumed: the rest of the expression parses.
        let (ns2, r2) = parse_expr("a + - 7");
        let n2 = ns2.get(r2);
        assert_eq!(n2.tag, AstTag::BinaryOp);
        assert_eq!(n2.op, BinOp::Add as u8);
        let rhs = ns2.get(NodeIdx::new(n2.rhs));
        assert_eq!(rhs.tag, AstTag::NumberLit);
        assert_eq!(rhs.lhs, 7);
        assert_eq!(rhs.flags & 1, 1);

        // Double negation cancels (sign XOR, matching the C++ fold).
        let (ns3, r3) = parse_expr("- -7");
        let n3 = ns3.get(r3);
        assert_eq!(n3.tag, AstTag::NumberLit);
        assert_eq!(n3.lhs, 7);
        assert_eq!(n3.flags & 1, 0);

        // Non-literal operand still gets a Neg unary op.
        let (ns4, r4) = parse_expr("- x");
        assert_eq!(ns4.get(r4).tag, AstTag::UnaryOp);
        assert_eq!(ns4.get(r4).op, UnaryOp::Neg as u8);
    }

    #[test]
    fn unary_minus_folds_through_as_cast() {
        // `- 16 as i64` => AsCast(NumberLit(-16), i64), not Neg(AsCast(16)).
        let (ns, root) = parse_expr("- 16 as i64");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::AsCast);
        let lit = ns.get(NodeIdx::new(n.lhs));
        assert_eq!(lit.tag, AstTag::NumberLit);
        assert_eq!(lit.lhs, 16);
        assert_eq!(lit.flags & 1, 1);
    }

    #[test]
    fn float_fits_f64() {
        let (ns, root) = parse_expr("0.5");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::NumberLit);
        assert_eq!(n.flags & 2, 2); // float
        assert_eq!(n.flags & 4, 0); // fits f64
        let bits = (n.lhs as u64) | ((n.rhs as u64) << 32);
        assert_eq!(f64::from_bits(bits), 0.5);
    }

    #[test]
    fn call_and_member_and_index() {
        // foo(a, b) -> Call, direct (foo), 2 args
        let (ns, root) = parse_expr("foo(a, b)");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::Call);
        assert_eq!(n.flags & 1, 0); // direct
        assert_eq!(ns.get_extra(ExtraIdx::new(n.rhs)), 2); // arg count

        // a.b -> MemberAccess
        let (ns2, r2) = parse_expr("a.b");
        assert_eq!(ns2.get(r2).tag, AstTag::MemberAccess);

        // a[0] -> Index
        let (ns3, r3) = parse_expr("a[0]");
        assert_eq!(ns3.get(r3).tag, AstTag::Index);
    }

    /// Lex `src` and run one statement via parse_expression.
    fn parse_stmt(src: &str) -> (NodeStore, NodeIdx) {
        let mut lx = Lexer::new(src.as_bytes().to_vec());
        lx.scan_tokens().unwrap();
        let tokens = lx.tokens().to_vec();
        let source = lx.source().to_vec();
        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let mut diags = Diagnostics::new();
        let root = {
            let mut p = Parser::new(
                tokens, source, &mut tp, &mut sp, &mut ns, &mut diags, "t.jam",
            );
            p.parse_expression().expect("parse stmt")
        };
        (ns, root)
    }

    #[test]
    fn stmt_return_expr() {
        let (ns, root) = parse_stmt("return a + b;");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::Return);
        assert_eq!(ns.get(NodeIdx::new(n.lhs)).tag, AstTag::BinaryOp);
    }

    #[test]
    fn stmt_var_and_const_decl() {
        // var x: i32 = 5;  -> VarDecl, not const, type=i32, init=NumberLit
        let (ns, root) = parse_stmt("var x: i32 = 5;");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::VarDecl);
        assert_eq!(n.rhs & 1, 0); // not const
        let e = ExtraIdx::new(n.lhs);
        assert_eq!(ns.get_extra(ExtraIdx::new(e.raw() + 1)), builtin::I32.raw()); // type
        let init = NodeIdx::new(ns.get_extra(ExtraIdx::new(e.raw() + 2)));
        assert_eq!(ns.get(init).tag, AstTag::NumberLit);

        // const y = 7;  -> VarDecl, const, type=kNoType
        let (ns2, r2) = parse_stmt("const y = 7;");
        let n2 = ns2.get(r2);
        assert_eq!(n2.tag, AstTag::VarDecl);
        assert_eq!(n2.rhs & 1, 1); // const
        assert_eq!(ns2.get_extra(ExtraIdx::new(n2.lhs + 1)), 0); // kNoType
    }

    #[test]
    fn stmt_if_else() {
        // if (a) { return 1; } else { return 2; }
        let (ns, root) = parse_stmt("if (a) { return 1; } else { return 2; }");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::IfNode);
        assert_eq!(n.flags, 0); // plain (not comp)
        assert_eq!(ns.get(NodeIdx::new(n.lhs)).tag, AstTag::Variable); // cond
        let e = ExtraIdx::new(n.rhs);
        assert_eq!(ns.get_extra(e), 1); // thenCount
        assert_eq!(ns.get_extra(ExtraIdx::new(e.raw() + 1)), 1); // elseCount
    }

    #[test]
    fn stmt_while_and_assign() {
        let (ns, root) = parse_stmt("while (a) { x = 1; }");
        assert_eq!(ns.get(root).tag, AstTag::WhileNode);
        assert_eq!(ns.get(root).flags, 0);

        let (ns2, r2) = parse_stmt("x = 1;");
        assert_eq!(ns2.get(r2).tag, AstTag::Assign);
    }

    #[test]
    fn stmt_loop_desugars_to_while_true() {
        // loop { break; } -> WhileNode with a BoolLit(true) cond emitted first.
        let (ns, root) = parse_stmt("loop { break; }");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::WhileNode);
        assert_eq!(n.flags, 0); // not inline
        let cond = ns.get(NodeIdx::new(n.lhs));
        assert_eq!(cond.tag, AstTag::BoolLit);
        assert_eq!(cond.lhs, 1); // true
    }

    fn parse_module(src: &str) -> crate::ast::ModuleAST {
        let mut lx = Lexer::new(src.as_bytes().to_vec());
        lx.scan_tokens().unwrap();
        let tokens = lx.tokens().to_vec();
        let source = lx.source().to_vec();
        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let mut diags = Diagnostics::new();
        let mut p = Parser::new(
            tokens, source, &mut tp, &mut sp, &mut ns, &mut diags, "t.jam",
        );
        p.parse().expect("parse module")
    }

    #[test]
    fn module_imports_struct_const_fn() {
        let m = parse_module(
            "const std = import(\"std\");\n\
             pub const Point = struct { x: i32, y: i32 };\n\
             const N = 10;\n\
             pub fn main() i32 { return N; }\n",
        );
        assert_eq!(m.imports.len(), 1);
        assert_eq!(m.imports[0].name, "std");
        assert_eq!(m.imports[0].path, "std");
        assert_eq!(m.structs.len(), 1);
        assert_eq!(m.structs[0].name, "Point");
        assert!(m.structs[0].is_pub);
        assert_eq!(m.structs[0].fields.len(), 2);
        assert_eq!(m.structs[0].fields[0], ("x".to_string(), builtin::I32));
        assert_eq!(m.consts.len(), 1);
        assert_eq!(m.consts[0].name, "N");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "main");
        assert!(m.functions[0].is_pub);
        assert_eq!(m.functions[0].return_type, builtin::I32);
        assert_eq!(m.functions[0].body.len(), 1);
    }

    #[test]
    fn module_enum_discriminants() {
        let m = parse_module("const Dir = enum { Up, Down(i32), Right = 5, Last };");
        assert_eq!(m.enums.len(), 1);
        let e = &m.enums[0];
        assert_eq!(e.variants.len(), 4);
        assert_eq!(e.variants[0].discriminant, 0); // Up
        assert_eq!(e.variants[1].discriminant, 1); // Down(i32)
        assert_eq!(e.variants[1].payload_types, vec![builtin::I32]);
        assert_eq!(e.variants[2].discriminant, 5); // Right = 5
        assert_eq!(e.variants[3].discriminant, 6); // Last resumes at 6
    }

    #[test]
    fn module_enum_over_256_variants_rejected() {
        // 257 variants exceeds the u8 tag; 256 is the last accepted count.
        let variants = |n: usize| {
            (0..n).map(|i| format!("V{i}")).collect::<Vec<_>>().join(", ")
        };
        let ok = format!("const Big = enum {{ {} }};", variants(256));
        assert_eq!(parse_module(&ok).enums[0].variants.len(), 256);

        let src = format!("const Big = enum {{ {} }};", variants(257));
        let mut lx = Lexer::new(src.into_bytes());
        lx.scan_tokens().unwrap();
        let tokens = lx.tokens().to_vec();
        let source = lx.source().to_vec();
        let mut tp = TypePool::new();
        let mut sp = StringPool::new();
        let mut ns = NodeStore::new();
        let mut diags = Diagnostics::new();
        let mut p = Parser::new(
            tokens, source, &mut tp, &mut sp, &mut ns, &mut diags, "t.jam",
        );
        assert!(p.parse().is_err());
        assert!(diags.all().iter().any(|d| {
            d.message
                .contains("has more than 256 variants; M2 enums are u8-tagged")
        }));
    }

    #[test]
    fn module_fn_params_and_modes() {
        // Mode keyword comes after the `:` (`name: mut Type`). `void` is not a
        // lexer keyword, so an omitted return type (kNoType) is the void case.
        let m = parse_module("fn f(a: i32, b: mut i64, c: move u8) i32 { return a; }");
        let f = &m.functions[0];
        assert_eq!(f.args.len(), 3);
        assert_eq!(f.args[0].mode, jam_core::ParamMode::Let);
        assert_eq!(f.args[1].mode, jam_core::ParamMode::Mut);
        assert_eq!(f.args[2].mode, jam_core::ParamMode::Move);
        assert_eq!(f.return_type, builtin::I32);
    }

    #[test]
    fn stmt_for_range() {
        // for i in 0:n { x = i; } -> ForNode, extra=[var,start,end,bodyCount,...]
        let (ns, root) = parse_stmt("for i in 0:n { x = i; }");
        let n = ns.get(root);
        assert_eq!(n.tag, AstTag::ForNode);
        let e = ExtraIdx::new(n.lhs);
        // start is NumberLit 0, end is Variable n
        assert_eq!(
            ns.get(NodeIdx::new(ns.get_extra(ExtraIdx::new(e.raw() + 1))))
                .tag,
            AstTag::NumberLit
        );
        assert_eq!(
            ns.get(NodeIdx::new(ns.get_extra(ExtraIdx::new(e.raw() + 2))))
                .tag,
            AstTag::Variable
        );
        assert_eq!(ns.get_extra(ExtraIdx::new(e.raw() + 3)), 1); // bodyCount
    }
}
