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

/// Print one `JirFunction` in the oracle's `emitJirFunctions` format.
fn dump_jir_function(o: &mut Vec<u8>, tp: &TypePool, sp: &StringPool, f: &JirFunction) {
    o.extend_from_slice(b"function ");
    o.extend_from_slice(f.name.as_bytes());
    o.extend_from_slice(b" pub=");
    num(o, f.is_pub as u64);
    o.extend_from_slice(b" export=");
    num(o, f.is_export as u64);
    o.extend_from_slice(b" extern=");
    num(o, f.is_extern as u64);
    o.extend_from_slice(b" test=");
    num(o, f.is_test as u64);
    o.extend_from_slice(b" varargs=");
    num(o, f.is_var_args as u64);
    o.extend_from_slice(b" module=\"");
    o.extend_from_slice(f.module_path.as_bytes());
    o.extend_from_slice(b"\"\n");

    o.extend_from_slice(b"  return: ");
    spell_type(o, tp, sp, f.return_type);
    o.push(b'\n');

    for (i, pty) in f.param_types.iter().enumerate() {
        let mode = if i < f.param_modes.len() {
            param_mode_name(f.param_modes[i])
        } else {
            "?"
        };
        o.extend_from_slice(b"  param ");
        num(o, i as u64);
        o.extend_from_slice(b": ");
        spell_type(o, tp, sp, *pty);
        o.push(b' ');
        o.extend_from_slice(mode.as_bytes());
        o.push(b'\n');
    }

    o.extend_from_slice(b"  extra:");
    for v in &f.extra {
        o.push(b' ');
        num(o, *v as u64);
    }
    o.push(b'\n');

    for (bi, blk) in f.blocks.iter().enumerate() {
        o.extend_from_slice(b"  block ");
        num(o, bi as u64);
        o.extend_from_slice(b" \"");
        o.extend_from_slice(blk.name.as_bytes());
        o.extend_from_slice(b"\":\n");
        for &r in &blk.insts {
            let inst = &f.insts[r as usize];
            o.extend_from_slice(b"    ");
            num(o, r as u64);
            o.extend_from_slice(b": ");
            o.extend_from_slice(inst.tag.name().as_bytes());
            o.extend_from_slice(b" a=");
            num(o, inst.a as u64);
            o.extend_from_slice(b" b=");
            num(o, inst.b as u64);
            o.extend_from_slice(b" ty=");
            spell_type(o, tp, sp, inst.ty);
            o.extend_from_slice(b" flags=");
            num(o, inst.flags as u64);
            // Name-carrying instructions append the referenced string (OOB-safe,
            // unescaped — matches the oracle's `dumpStr`).
            match inst.tag {
                JirTag::Call | JirTag::FnRef | JirTag::Str => {
                    o.extend_from_slice(b" str=\"");
                    o.extend_from_slice(&sp.get(StringIdx::new(inst.a)));
                    o.push(b'"');
                }
                JirTag::DropBinding => {
                    o.extend_from_slice(b" str=\"");
                    o.extend_from_slice(&sp.get(StringIdx::new(inst.b)));
                    o.push(b'"');
                }
                _ => {}
            }
            o.push(b'\n');
        }
    }
}

/// Qualify a type/method name with its owning module (the C++
/// `qualifyTypeName`): bare for the entry module, `module.name` otherwise.
fn qualify_type_name(module_path: &str, name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        format!("{module_path}.{name}")
    }
}

/// Register a value const under both its qualified key (for imports) and its
/// bare name. The bare name is the per-module body's spelling, so when the ENTRY
/// and an imported module both define `const N` the entry's value must win the
/// bare slot (the entry body's `current_body_module()==""` reads the bare key).
/// The C++ avoids this by keying every const under its qualified identity (the
/// entry's modulePath is its non-empty filename) and resolving the bare spelling
/// through a per-module owner map (main.cpp:1847). We mirror the OUTCOME: the
/// entry const (empty module_path) is authoritative for the bare key — it always
/// writes it — while an imported const claims the bare key only if no const is
/// registered there yet, so it never clobbers an entry const regardless of the
/// registration order across the two passes (entry-first vs entry-last).
fn register_const_keyed(
    cg: &CodegenContext,
    c: &jam_syntax::ast::ConstDeclAST,
    info: ModuleConstInfo,
) {
    if c.module_path.is_empty() {
        // Entry const: own the bare key unconditionally.
        cg.register_module_const(c.name.clone(), info);
    } else {
        cg.register_module_const(format!("{}.{}", c.module_path, c.name), info.clone());
        if !cg.has_module_const(&c.name) {
            cg.register_module_const(c.name.clone(), info);
        }
    }
}

/// Register the module's struct/union/enum types into the context (creating the
/// named LLVM struct handles the analyzer fills) and drive the analyzer over
/// every type decl, materialising the LLVM layouts that AstGen's `Named` field /
/// param / return types resolve through. Mirrors the C++ register pass
/// (`declareStructs`/`declareUnions`/`declareEnums`) + the analyzer loop.
///
/// `modules: &'ctx` — the decl table borrows the module ASTs for the context's
/// (invariant) lifetime; the caller guarantees they outlive the context. Each
/// module's types register under their module-qualified name (bare for the
/// entry module, `module.Name` for imports), and imported bare names get a
/// pre-interned `requalify` entry so a module body's bare type references
/// resolve to their qualified identity.
fn register_and_analyze<'ctx>(cg: &mut CodegenContext<'ctx>, modules: &[&'ctx ModuleAST]) {
    // -1. Register value consts FIRST so a const-sized array field
    //     (`cells: [RAM_END]u8`) folds during the layout pass below.
    for m in modules {
        for c in &m.consts {
            // A type-alias const (`const CounterI32 = Counter(i32)`): register it
            // in the type-alias table (qualified to its own module), so
            // `name_for_kinds`' chase resolves `CounterI32` -> `Counter(i32)` ->
            // the monomorphized struct. The C++ registerConsts (main.cpp:891-915).
            if !c.aliased_type.is_none() {
                let target = cg.requalify_type(c.aliased_type, &c.module_path);
                cg.register_type_alias(c.name.clone(), target);
                if !c.module_path.is_empty() {
                    cg.register_type_alias(format!("{}.{}", c.module_path, c.name), target);
                }
                continue;
            }
            if c.init_expr.is_none() {
                continue;
            }
            let info = ModuleConstInfo {
                init_expr: c.init_expr,
                declared_type: c.declared_type,
                is_comp: c.is_comp,
                bare_name: c.name.clone(),
                module_path: c.module_path.clone(),
            };
            register_const_keyed(cg, c, info);
        }
    }
    // 0. Pre-intern the qualified `Named` TypeIdx for every imported type +
    //    record it in the requalify map. Done FIRST (before fields) so field
    //    requalification below resolves forward references.
    for m in modules {
        let stamp = |cg: &mut CodegenContext, module_path: &str, name: &str| {
            if !module_path.is_empty() {
                let q = qualify_type_name(module_path, name);
                let sid = cg.string_pool.intern(q.as_bytes());
                let qty = cg.type_pool.intern_named(sid);
                cg.register_requalify(module_path, name, qty);
            }
        };
        for s in &m.structs {
            stamp(cg, &s.module_path, &s.name);
        }
        for u in &m.unions {
            stamp(cg, &u.module_path, &u.name);
        }
        for e in &m.enums {
            stamp(cg, &e.module_path, &e.name);
        }
    }

    // 1. Register every module's types under their qualified name, with field /
    //    payload types REQUALIFIED (so a struct field of an imported sibling
    //    type carries its qualified identity — the C++ `requalFields`).
    // NOTE: struct/union/enum field types are requalified but NOT array-length-
    // resolved here — a struct-field array `cells: [N]u8` is left as the deferred
    // ArrayExpr (so const-sized struct fields stay cleanly unported) because
    // indexing such a field (`self.cells[i]`) needs a Load-array-then-Index
    // lowering in astgen_index that isn't ported yet.
    for m in modules {
        for s in &m.structs {
            let q = qualify_type_name(&s.module_path, &s.name);
            let fields = s
                .fields
                .iter()
                .map(|(n, t)| (n.clone(), cg.requalify_type(*t, &s.module_path)))
                .collect();
            let named = cg.context().named_struct(&q);
            cg.register_struct(q, named, fields);
        }
        for u in &m.unions {
            let q = qualify_type_name(&u.module_path, &u.name);
            let fields = u
                .fields
                .iter()
                .map(|(n, t)| (n.clone(), cg.requalify_type(*t, &u.module_path)))
                .collect();
            let named = cg.context().named_struct(&q);
            cg.register_union(q, named, fields);
        }
        for e in &m.enums {
            let q = qualify_type_name(&e.module_path, &e.name);
            let variants = e
                .variants
                .iter()
                .map(|v| EnumVariantInfo {
                    name: v.name.clone(),
                    payload_types: v
                        .payload_types
                        .iter()
                        .map(|t| cg.requalify_type(*t, &e.module_path))
                        .collect(),
                    discriminant: v.discriminant,
                })
                .collect();
            cg.register_enum(q, variants);
        }
    }

    // 1b. POPULATE the array-expr resolution cache for `[expr]T` field/payload
    // types (without changing the registered ArrayExpr — `peek` must still see
    // ArrayExpr so a field-array index takes the value path, like the oracle).
    // get_llvm_type/type_size/type_align then resolve via the cache.
    let cache_array_fields = |cg: &mut CodegenContext, module_path: &str, tys: Vec<TypeIdx>| {
        cg.push_body_module(module_path.to_string());
        for t in tys {
            if cg.type_pool.get(t).kind == TypeKind::ArrayExpr {
                let _ = cg.resolve_array_expr_instantiate(t);
            }
        }
        cg.pop_body_module();
    };
    for m in modules {
        for s in &m.structs {
            cache_array_fields(
                cg,
                &s.module_path,
                s.fields.iter().map(|(_, t)| *t).collect(),
            );
        }
        for u in &m.unions {
            cache_array_fields(
                cg,
                &u.module_path,
                u.fields.iter().map(|(_, t)| *t).collect(),
            );
        }
        for e in &m.enums {
            let tys = e
                .variants
                .iter()
                .flat_map(|v| v.payload_types.iter().copied())
                .collect();
            cache_array_fields(cg, &e.module_path, tys);
        }
    }
}

/// Materialise every type's LLVM body (the C++ demand-driven body-fill pass,
/// main.cpp:1877-1884). Split out of `register_and_analyze` and run LATER —
/// AFTER import handles + module namespaces are registered — so a struct/enum
/// whose field/payload is a CHAINED type (`w.leaf.Point`) can resolve through
/// the re-export chain when its body lays out. The C++ runs this pass after the
/// namespace registration too; running it eagerly inside `register_and_analyze`
/// (before namespaces existed) left chained-typed fields unresolved (stub body).
fn fill_type_bodies<'ctx>(cg: &mut CodegenContext<'ctx>, modules: &[&'ctx ModuleAST]) {
    // One decl table over all modules' type decls (keyed by qualified name).
    let mut decls = DeclTable::new();
    let mut type_idxs: Vec<DeclIndex> = Vec::new();
    for m in modules {
        for s in &m.structs {
            let idx = decls.create(DeclKind::Struct, qualify_type_name(&s.module_path, &s.name));
            decls.get_mut(idx).struct_ast = Some(s);
            type_idxs.push(idx);
        }
        for e in &m.enums {
            let idx = decls.create(DeclKind::Enum, qualify_type_name(&e.module_path, &e.name));
            decls.get_mut(idx).enum_ast = Some(e);
            type_idxs.push(idx);
        }
        for u in &m.unions {
            let idx = decls.create(DeclKind::Union, qualify_type_name(&u.module_path, &u.name));
            decls.get_mut(idx).union_ast = Some(u);
            type_idxs.push(idx);
        }
    }

    // Force analysis of every type body (re-entrant via field deps).
    let mut az = Analyzer::new(cg, &mut decls);
    for idx in type_idxs {
        az.ensure_decl_analyzed(idx);
    }
}

