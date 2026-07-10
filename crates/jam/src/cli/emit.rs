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

fn pad(o: &mut Vec<u8>, n: usize) {
    for _ in 0..n {
        o.push(b' ');
    }
}

fn num(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(v.to_string().as_bytes());
}

fn spell_type(o: &mut Vec<u8>, tp: &TypePool, sp: &StringPool, ty: TypeIdx) {
    let k = tp.get(ty);
    match k.kind {
        TypeKind::Invalid => o.extend_from_slice(b"<invalid>"),
        TypeKind::Void => o.extend_from_slice(b"void"),
        TypeKind::NoReturn => o.extend_from_slice(b"noreturn"),
        TypeKind::Bool => o.extend_from_slice(b"bool"),
        TypeKind::Int => {
            o.push(if k.b != 0 { b'i' } else { b'u' });
            num(o, k.a as u64);
        }
        TypeKind::Float => {
            o.push(b'f');
            num(o, k.a as u64);
        }
        TypeKind::PtrSingle => {
            o.push(b'*');
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
        TypeKind::PtrMany => {
            o.extend_from_slice(b"[*]");
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
        TypeKind::Slice => {
            o.extend_from_slice(b"[]");
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
        TypeKind::Array => {
            o.push(b'[');
            num(o, k.b as u64);
            o.push(b']');
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
        TypeKind::ArrayExpr => {
            o.extend_from_slice(b"[expr#");
            num(o, k.b as u64);
            o.push(b']');
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
        TypeKind::Struct | TypeKind::Enum | TypeKind::Union | TypeKind::Named => {
            o.extend_from_slice(&sp.get(StringIdx::new(k.a)));
        }
        TypeKind::Type => o.extend_from_slice(b"type"),
        TypeKind::GenericCall => {
            o.extend_from_slice(&sp.get(StringIdx::new(k.a)));
            o.push(b'(');
            for (i, a) in tp.generic_args_at(k.b).iter().enumerate() {
                if i > 0 {
                    o.extend_from_slice(b", ");
                }
                spell_type(o, tp, sp, *a);
            }
            o.push(b')');
        }
        TypeKind::Module => {
            o.extend_from_slice(b"module(");
            o.extend_from_slice(&sp.get(StringIdx::new(k.a)));
            o.push(b')');
        }
        TypeKind::Fn => {
            o.extend_from_slice(b"fn(");
            for (i, p) in tp.fn_params_at(k.b).iter().enumerate() {
                if i > 0 {
                    o.extend_from_slice(b", ");
                }
                spell_type(o, tp, sp, *p);
            }
            o.extend_from_slice(b") ");
            spell_type(o, tp, sp, TypeIdx::new(k.a));
        }
    }
}

fn bin_op_name(op: u8) -> &'static str {
    match op {
        x if x == BinOp::Add as u8 => "Add",
        x if x == BinOp::Sub as u8 => "Sub",
        x if x == BinOp::Mul as u8 => "Mul",
        x if x == BinOp::Div as u8 => "Div",
        x if x == BinOp::Mod as u8 => "Mod",
        x if x == BinOp::BitAnd as u8 => "BitAnd",
        x if x == BinOp::BitOr as u8 => "BitOr",
        x if x == BinOp::BitXor as u8 => "BitXor",
        x if x == BinOp::Shl as u8 => "Shl",
        x if x == BinOp::Shr as u8 => "Shr",
        x if x == BinOp::LogAnd as u8 => "LogAnd",
        x if x == BinOp::LogOr as u8 => "LogOr",
        x if x == BinOp::Eq as u8 => "Eq",
        x if x == BinOp::Ne as u8 => "Ne",
        x if x == BinOp::Lt as u8 => "Lt",
        x if x == BinOp::Le as u8 => "Le",
        x if x == BinOp::Gt as u8 => "Gt",
        x if x == BinOp::Ge as u8 => "Ge",
        _ => "Invalid",
    }
}

fn un_op_name(op: u8) -> &'static str {
    match op {
        x if x == UnaryOp::Neg as u8 => "Neg",
        x if x == UnaryOp::LogNot as u8 => "LogNot",
        x if x == UnaryOp::BitNot as u8 => "BitNot",
        _ => "Invalid",
    }
}

fn param_mode_name(m: jam_core::ParamMode) -> &'static str {
    match m {
        jam_core::ParamMode::Let => "let",
        jam_core::ParamMode::Mut => "mut",
        jam_core::ParamMode::Move => "move",
    }
}

fn ex(ns: &NodeStore, base: u32, off: u32) -> u32 {
    ns.get_extra(ExtraIdx::new(base + off))
}

fn dump_child(
    o: &mut Vec<u8>,
    ns: &NodeStore,
    tp: &TypePool,
    sp: &StringPool,
    id: u32,
    indent: usize,
) {
    if id == 0 {
        return; // kNoNode prints nothing (matches the C++)
    }
    dump_node(o, ns, tp, sp, id, indent);
}

fn dump_node(
    o: &mut Vec<u8>,
    ns: &NodeStore,
    tp: &TypePool,
    sp: &StringPool,
    id: u32,
    indent: usize,
) {
    let n = *ns.get(NodeIdx::new(id));
    pad(o, indent);
    let ci = indent + 2;
    match n.tag {
        AstTag::NumberLit => {
            let v = (n.lhs as u64) | ((n.rhs as u64) << 32);
            o.extend_from_slice(b"NumberLit ");
            num(o, v);
            o.extend_from_slice(if n.flags & 1 != 0 { b" neg\n" } else { b"\n" });
        }
        AstTag::BoolLit => {
            o.extend_from_slice(b"BoolLit ");
            num(o, n.lhs as u64);
            o.push(b'\n');
        }
        AstTag::StringLit => {
            o.extend_from_slice(b"StringLit \"");
            write_escaped(o, &sp.get(StringIdx::new(n.lhs)));
            o.extend_from_slice(b"\"\n");
        }
        AstTag::Variable => {
            o.extend_from_slice(b"Variable ");
            o.extend_from_slice(&sp.get(StringIdx::new(n.lhs)));
            o.push(b'\n');
        }
        AstTag::MemberAccess => {
            o.extend_from_slice(b"MemberAccess .");
            o.extend_from_slice(&sp.get(StringIdx::new(n.rhs)));
            o.push(b'\n');
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::Index => {
            o.extend_from_slice(b"Index\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            dump_child(o, ns, tp, sp, n.rhs, ci);
        }
        AstTag::Slice => {
            o.extend_from_slice(b"Slice\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            dump_child(o, ns, tp, sp, ex(ns, n.rhs, 0), ci);
            dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1), ci);
        }
        AstTag::Deref => {
            o.extend_from_slice(b"Deref\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::AddressOf => {
            o.extend_from_slice(b"AddressOf\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::UnaryOp => {
            o.extend_from_slice(b"UnaryOp ");
            o.extend_from_slice(un_op_name(n.op).as_bytes());
            o.push(b'\n');
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::BinaryOp => {
            o.extend_from_slice(b"BinaryOp ");
            o.extend_from_slice(bin_op_name(n.op).as_bytes());
            o.push(b'\n');
            dump_child(o, ns, tp, sp, n.lhs, ci);
            dump_child(o, ns, tp, sp, n.rhs, ci);
        }
        AstTag::Call => {
            o.extend_from_slice(b"Call ");
            o.extend_from_slice(&sp.get(StringIdx::new(n.lhs)));
            o.push(b'\n');
            let count = ex(ns, n.rhs, 0);
            for i in 0..count {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1 + i), ci);
            }
        }
        AstTag::Return => {
            o.extend_from_slice(b"Return\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::Assign => {
            o.extend_from_slice(b"Assign\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            dump_child(o, ns, tp, sp, n.rhs, ci);
        }
        AstTag::VarDecl => {
            let name = ex(ns, n.lhs, 0);
            let ty = ex(ns, n.lhs, 1);
            let init = ex(ns, n.lhs, 2);
            o.extend_from_slice(b"VarDecl ");
            o.extend_from_slice(&sp.get(StringIdx::new(name)));
            o.extend_from_slice(b": ");
            spell_type(o, tp, sp, TypeIdx::new(ty));
            o.extend_from_slice(if n.rhs & 1 != 0 { b" const\n" } else { b"\n" });
            dump_child(o, ns, tp, sp, init, ci);
        }
        AstTag::IfNode => {
            o.extend_from_slice(b"If\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            let then_n = ex(ns, n.rhs, 0);
            let else_n = ex(ns, n.rhs, 1);
            pad(o, ci);
            o.extend_from_slice(b"then\n");
            for i in 0..then_n {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 2 + i), ci + 2);
            }
            pad(o, ci);
            o.extend_from_slice(b"else\n");
            for i in 0..else_n {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 2 + then_n + i), ci + 2);
            }
        }
        AstTag::WhileNode => {
            o.extend_from_slice(b"While\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            let body_n = ex(ns, n.rhs, 0);
            for i in 0..body_n {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1 + i), ci);
            }
        }
        AstTag::ForNode => {
            let var = ex(ns, n.lhs, 0);
            let start = ex(ns, n.lhs, 1);
            let end = ex(ns, n.lhs, 2);
            let body_n = ex(ns, n.lhs, 3);
            o.extend_from_slice(b"For ");
            o.extend_from_slice(&sp.get(StringIdx::new(var)));
            o.push(b'\n');
            dump_child(o, ns, tp, sp, start, ci);
            dump_child(o, ns, tp, sp, end, ci);
            for i in 0..body_n {
                dump_child(o, ns, tp, sp, ex(ns, n.lhs, 4 + i), ci);
            }
        }
        AstTag::Break => o.extend_from_slice(b"Break\n"),
        AstTag::Continue => o.extend_from_slice(b"Continue\n"),
        AstTag::ImportLit => {
            o.extend_from_slice(b"ImportLit ");
            o.extend_from_slice(&sp.get(StringIdx::new(n.lhs)));
            o.push(b'\n');
        }
        AstTag::StructLit => {
            o.extend_from_slice(b"StructLit ");
            spell_type(o, tp, sp, TypeIdx::new(n.lhs));
            o.push(b'\n');
            let count = ex(ns, n.rhs, 0);
            for i in 0..count {
                let fname = ex(ns, n.rhs, 1 + i * 2);
                let fexpr = ex(ns, n.rhs, 2 + i * 2);
                pad(o, ci);
                o.extend_from_slice(b"field ");
                o.extend_from_slice(&sp.get(StringIdx::new(fname)));
                o.push(b'\n');
                dump_child(o, ns, tp, sp, fexpr, ci + 2);
            }
        }
        AstTag::ArrayLit => {
            o.extend_from_slice(b"ArrayLit ");
            spell_type(o, tp, sp, TypeIdx::new(n.lhs));
            o.push(b'\n');
            let count = ex(ns, n.rhs, 0);
            for i in 0..count {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1 + i), ci);
            }
        }
        AstTag::ArrayRepeat => {
            o.extend_from_slice(b"ArrayRepeat ");
            spell_type(o, tp, sp, TypeIdx::new(n.lhs));
            o.push(b'\n');
            dump_child(o, ns, tp, sp, ex(ns, n.rhs, 0), ci);
            dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1), ci);
        }
        AstTag::StructExpr => {
            o.extend_from_slice(b"StructExpr #");
            num(o, n.lhs as u64);
            o.push(b'\n');
        }
        AstTag::EnumExpr => {
            o.extend_from_slice(b"EnumExpr #");
            num(o, n.lhs as u64);
            o.push(b'\n');
        }
        AstTag::MatchNode => {
            o.extend_from_slice(b"Match\n");
            dump_child(o, ns, tp, sp, n.lhs, ci);
            let arm_n = ex(ns, n.rhs, 0);
            let mut p = n.rhs + 1;
            for _ in 0..arm_n {
                let pat = ns.get_extra(ExtraIdx::new(p));
                p += 1;
                let body_n = ns.get_extra(ExtraIdx::new(p));
                p += 1;
                pad(o, ci);
                o.extend_from_slice(b"arm\n");
                dump_child(o, ns, tp, sp, pat, ci + 2);
                for _ in 0..body_n {
                    let b = ns.get_extra(ExtraIdx::new(p));
                    p += 1;
                    dump_child(o, ns, tp, sp, b, ci + 2);
                }
            }
        }
        AstTag::AsCast => {
            o.extend_from_slice(b"AsCast ");
            spell_type(o, tp, sp, TypeIdx::new(n.rhs));
            o.push(b'\n');
            dump_child(o, ns, tp, sp, n.lhs, ci);
        }
        AstTag::AtCall => {
            o.extend_from_slice(b"AtCall @");
            o.extend_from_slice(&sp.get(StringIdx::new(n.lhs)));
            if n.flags & 1 != 0 {
                o.push(b'\n');
                let count = ex(ns, n.rhs, 0);
                for i in 0..count {
                    dump_child(o, ns, tp, sp, ex(ns, n.rhs, 1 + i), ci);
                }
            } else {
                o.push(b' ');
                spell_type(o, tp, sp, TypeIdx::new(n.rhs));
                o.push(b'\n');
            }
        }
        AstTag::TypeMethodCall => {
            o.extend_from_slice(b"TypeMethodCall ");
            spell_type(o, tp, sp, TypeIdx::new(n.lhs));
            o.push(b'.');
            o.extend_from_slice(&sp.get(StringIdx::new(ex(ns, n.rhs, 0))));
            o.push(b'\n');
            let count = ex(ns, n.rhs, 1);
            for i in 0..count {
                dump_child(o, ns, tp, sp, ex(ns, n.rhs, 2 + i), ci);
            }
        }
        AstTag::PatLit => {
            let v = (n.lhs as u64) | ((n.rhs as u64) << 32);
            o.extend_from_slice(b"PatLit ");
            num(o, v);
            o.extend_from_slice(if n.flags & 1 != 0 { b" neg\n" } else { b"\n" });
        }
        AstTag::PatRange => {
            o.extend_from_slice(b"PatRange ");
            num(o, n.lhs as u64);
            o.extend_from_slice(b"..");
            num(o, n.rhs as u64);
            o.push(b'\n');
        }
        AstTag::PatWildcard => o.extend_from_slice(b"PatWildcard\n"),
        AstTag::PatOr => {
            o.extend_from_slice(b"PatOr\n");
            let count = ex(ns, n.lhs, 0);
            for i in 0..count {
                dump_child(o, ns, tp, sp, ex(ns, n.lhs, 1 + i), ci);
            }
        }
        AstTag::PatEnumVariant => {
            o.extend_from_slice(b"PatEnumVariant ");
            o.extend_from_slice(&sp.get(StringIdx::new(n.lhs)));
            o.push(b'.');
            o.extend_from_slice(&sp.get(StringIdx::new(n.rhs)));
            o.push(b'\n');
        }
        AstTag::Invalid => o.extend_from_slice(b"Invalid\n"),
        AstTag::Count => o.extend_from_slice(b"Count\n"),
    }
}

fn dump_function(
    o: &mut Vec<u8>,
    ns: &NodeStore,
    tp: &TypePool,
    sp: &StringPool,
    f: &FunctionAST,
    indent: usize,
) {
    pad(o, indent);
    o.extend_from_slice(b"fn ");
    o.extend_from_slice(f.name.as_bytes());
    o.push(b'(');
    for (i, p) in f.args.iter().enumerate() {
        if i > 0 {
            o.extend_from_slice(b", ");
        }
        if p.is_comp {
            o.extend_from_slice(b"comp ");
        }
        o.extend_from_slice(param_mode_name(p.mode).as_bytes());
        o.push(b' ');
        o.extend_from_slice(p.name.as_bytes());
        o.extend_from_slice(b": ");
        spell_type(o, tp, sp, p.ty);
    }
    o.extend_from_slice(b") -> ");
    spell_type(o, tp, sp, f.return_type);
    if f.is_extern {
        o.extend_from_slice(b" extern");
    }
    if f.is_export {
        o.extend_from_slice(b" export");
    }
    if f.is_pub {
        o.extend_from_slice(b" pub");
    }
    if f.is_test {
        o.extend_from_slice(b" test");
    }
    if f.is_var_args {
        o.extend_from_slice(b" varargs");
    }
    if f.is_cfn {
        o.extend_from_slice(b" cfn");
    }
    if f.is_comp_time_fn {
        o.extend_from_slice(b" comptimefn");
    }
    o.push(b'\n');
    for b in &f.body {
        dump_child(o, ns, tp, sp, b.raw(), indent + 2);
    }
}

/// Lex + parse `path` and print the AST (byte-identical to the C++ oracle).
pub fn emit_ast(path: &str) -> i32 {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Could not open file: {path}");
            return 1;
        }
    };
    let mut lexer = Lexer::new(bytes);
    if lexer.scan_tokens().is_err() {
        eprintln!("lex error in {path}");
        return 1;
    }
    let tokens = lexer.tokens().to_vec();
    let source = lexer.source().to_vec();

    let mut tp = TypePool::new();
    let mut sp = StringPool::new();
    let mut ns = NodeStore::new();
    let mut diags = Diagnostics::new();
    let module: ModuleAST = {
        let mut p = Parser::new(tokens, source, &mut tp, &mut sp, &mut ns, &mut diags, path);
        match p.parse() {
            Ok(m) => m,
            Err(_) => {
                eprint!("{}", diags.render_to_string());
                return 1;
            }
        }
    };

    let mut o: Vec<u8> = Vec::new();
    o.extend_from_slice(b"module\n");
    for imp in &module.imports {
        o.extend_from_slice(b"  import ");
        o.extend_from_slice(imp.name.as_bytes());
        o.extend_from_slice(b" = \"");
        o.extend_from_slice(imp.path.as_bytes());
        o.push(b'"');
        for c in &imp.chain {
            o.push(b'.');
            o.extend_from_slice(c.as_bytes());
        }
        if imp.is_pub {
            o.extend_from_slice(b" pub");
        }
        o.push(b'\n');
    }
    for di in &module.destructuring_imports {
        o.extend_from_slice(b"  destructuring-import {");
        for (i, name) in di.names.iter().enumerate() {
            if i > 0 {
                o.extend_from_slice(b", ");
            }
            o.extend_from_slice(name.as_bytes());
        }
        o.extend_from_slice(b"} = \"");
        o.extend_from_slice(di.path.as_bytes());
        o.push(b'"');
        for c in &di.chain {
            o.push(b'.');
            o.extend_from_slice(c.as_bytes());
        }
        o.push(b'\n');
    }
    for s in &module.structs {
        o.extend_from_slice(b"  struct ");
        o.extend_from_slice(s.name.as_bytes());
        if s.is_pub {
            o.extend_from_slice(b" pub");
        }
        o.push(b'\n');
        for (fname, fty) in &s.fields {
            o.extend_from_slice(b"    field ");
            o.extend_from_slice(fname.as_bytes());
            o.extend_from_slice(b": ");
            spell_type(&mut o, &tp, &sp, *fty);
            o.push(b'\n');
        }
        for m in &s.methods {
            dump_function(&mut o, &ns, &tp, &sp, m, 4);
        }
    }
    for u in &module.unions {
        o.extend_from_slice(b"  union ");
        o.extend_from_slice(u.name.as_bytes());
        if u.is_pub {
            o.extend_from_slice(b" pub");
        }
        o.push(b'\n');
        for (fname, fty) in &u.fields {
            o.extend_from_slice(b"    field ");
            o.extend_from_slice(fname.as_bytes());
            o.extend_from_slice(b": ");
            spell_type(&mut o, &tp, &sp, *fty);
            o.push(b'\n');
        }
    }
    for e in &module.enums {
        o.extend_from_slice(b"  enum ");
        o.extend_from_slice(e.name.as_bytes());
        if e.is_pub {
            o.extend_from_slice(b" pub");
        }
        o.push(b'\n');
        for v in &e.variants {
            o.extend_from_slice(b"    variant ");
            o.extend_from_slice(v.name.as_bytes());
            if !v.payload_types.is_empty() {
                o.push(b'(');
                for (i, t) in v.payload_types.iter().enumerate() {
                    if i > 0 {
                        o.extend_from_slice(b", ");
                    }
                    spell_type(&mut o, &tp, &sp, *t);
                }
                o.push(b')');
            }
            o.extend_from_slice(b" = ");
            num(&mut o, v.discriminant as u64);
            o.push(b'\n');
        }
    }
    for c in &module.consts {
        o.extend_from_slice(b"  const ");
        o.extend_from_slice(c.name.as_bytes());
        o.extend_from_slice(b": ");
        spell_type(&mut o, &tp, &sp, c.declared_type);
        if c.is_pub {
            o.extend_from_slice(b" pub");
        }
        if c.is_comp {
            o.extend_from_slice(b" comp");
        }
        if c.aliased_type != TypeIdx::NONE {
            o.extend_from_slice(b" alias=");
            spell_type(&mut o, &tp, &sp, c.aliased_type);
        }
        o.push(b'\n');
        dump_child(&mut o, &ns, &tp, &sp, c.init_expr.raw(), 4);
    }
    for f in &module.functions {
        dump_function(&mut o, &ns, &tp, &sp, f, 2);
    }

    let stdout = io::stdout();
    let mut w = stdout.lock();
    let _ = w.write_all(&o);
    let _ = w.flush();
    0
}

// ---- --emit-jir: byte-exact mirror of the C++ emitJirFunctions dumper -------

