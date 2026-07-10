/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `--emit-*` dumps used by the differential harness. Today: `--emit-tokens`,
//! whose output is byte-for-byte identical to the C++ oracle's
//! `emitTokensForFile` (`KIND line escaped-lexeme` per token, incl. trailing
//! EOF). Escaping and the string-vs-raw lexeme choice match the oracle exactly.

use std::io::{self, Write};

use jam_core::diag::Diagnostics;
use jam_core::index::{DeclIndex, ExtraIdx, NodeIdx, StringIdx, TypeIdx};
use jam_llvm::{CallConv, Context, Linkage, Lto, OptLevel, TargetMachine, default_target_triple};
use jam_sema::analyzer::Analyzer;
use jam_sema::astgen::{astgen_function, astgen_metadata};
use jam_sema::codegen_context::{CodegenContext, EnumVariantInfo, ModuleConstInfo};
use jam_sema::decl::{DeclKind, DeclTable};
use jam_sema::drop_registry::{CloneRegistry, add_clone_candidates, build_drop_registry};
use jam_sema::jir::{JirFunction, JirTag};
use jam_sema::jir_codegen::{jir_declare_prototype, jir_define_body};
use jam_sema::mangling::mangled_function_name;
use jam_sema::module_resolver::ModuleResolver;
use jam_sema::target::{Abi, Os, Target};
use jam_syntax::ast::{FunctionAST, ModuleAST};
use jam_syntax::ast_flat::{AstTag, BinOp, NodeStore, StringPool, TypeKind, TypePool, UnaryOp};
use jam_syntax::parser::Parser;
use jam_syntax::{Lexer, TokenType};

use crate::cli::progress::ProgressGuard;

/// Lex `path` and print the token stream. Returns a process exit code.
pub fn emit_tokens(path: &str) -> i32 {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Could not open file: {path}");
            return 1;
        }
    };
    let mut lexer = Lexer::new(bytes);
    if let Err(e) = lexer.scan_tokens() {
        eprintln!("lex error in {path}: {} (line {})", e.message, e.line);
        return 1;
    }
    let source = lexer.source();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for t in lexer.tokens() {
        // String literals show their decoded value; everything else shows the
        // raw source slice (matches the oracle).
        let lex: &[u8] = if t.ttype == TokenType::StringLiteral {
            &t.lexeme
        } else {
            t.text(source)
        };
        let _ = write!(out, "{} {} ", t.ttype.name(), t.line);
        write_escaped(&mut out, lex);
        let _ = out.write_all(b"\n");
    }
    let _ = out.flush();
    0
}

/// Escape a lexeme so each token stays on one line — identical to the oracle's
/// `escapeForDump` (backslash, `\n`/`\t`/`\r`, and `\xNN` for other controls;
/// every other byte, including UTF-8 continuation bytes, passes through raw).
fn write_escaped<W: Write>(out: &mut W, s: &[u8]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &c in s {
        match c {
            b'\\' => {
                let _ = out.write_all(b"\\\\");
            }
            b'\n' => {
                let _ = out.write_all(b"\\n");
            }
            b'\t' => {
                let _ = out.write_all(b"\\t");
            }
            b'\r' => {
                let _ = out.write_all(b"\\r");
            }
            _ if c < 0x20 || c == 0x7f => {
                let _ = out.write_all(&[
                    b'\\',
                    b'x',
                    HEX[(c >> 4) as usize & 0xf],
                    HEX[c as usize & 0xf],
                ]);
            }
            _ => {
                let _ = out.write_all(&[c]);
            }
        }
    }
}

// ---- --emit-ast: byte-exact mirror of the C++ emitAstForFile dumper --------

