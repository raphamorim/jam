/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `--emit-*` text dumps (tokens / ast / jir / ir). The formats are frozen —
//! they match the original C++ compiler's output byte-for-byte (kept at the
//! `cpp-final` tag), so don't reorder interning or printing casually.

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

/// Escape a lexeme so each token stays on one line — matches the oracle's
/// escaping exactly (backslash, `\n`/`\t`/`\r`, `\xNN` for other controls;
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

// ---- --emit-ast: byte-exact mirror of the oracle's AST dumper --------------

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
        return; // "no node" prints nothing
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

/// Lex + parse `path` and print the AST (byte-identical to the oracle).
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

// ---- --emit-jir: byte-exact mirror of the oracle's JIR dumper ---------------

/// Print one `JirFunction` in the oracle's `--emit-jir` format.
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
            // Name-carrying instructions append the referenced string
            // (OOB-safe, unescaped — matches the oracle).
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

/// Qualify a type/method name with its owning module: bare for the entry
/// module, `module.name` otherwise.
fn qualify_type_name(module_path: &str, name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        format!("{module_path}.{name}")
    }
}

/// Register a value const under both its qualified key (for imports) and its
/// bare name. The entry const (empty module_path) is authoritative for the
/// bare key — it always writes it — while an imported const claims the bare
/// key only if nothing is registered there yet, so it never clobbers an entry
/// const regardless of registration order across the two passes.
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

/// Register every module's struct/union/enum types (under module-qualified
/// names) and drive the analyzer over each type decl, materialising the LLVM
/// layouts that `Named` field/param/return types resolve through. Imported
/// bare names get a pre-interned `requalify` entry.
///
/// `modules: &'ctx` — the decl table borrows the module ASTs for the context's
/// (invariant) lifetime; the caller guarantees they outlive the context.
fn register_and_analyze<'ctx>(cg: &mut CodegenContext<'ctx>, modules: &[&'ctx ModuleAST]) {
    // -1. Register value consts FIRST so a const-sized array field
    //     (`cells: [RAM_END]u8`) folds during the layout pass below.
    for m in modules {
        for c in &m.consts {
            // A type-alias const (`const CounterI32 = Counter(i32)`): register it
            // in the type-alias table (qualified to its own module), so
            // `name_for_kinds`' chase resolves `CounterI32` -> `Counter(i32)` ->
            // the monomorphized struct.
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
    //    type carries its qualified identity).
    // NOTE: field types are requalified but NOT array-length-resolved here — a
    // struct-field array `cells: [N]u8` stays a deferred ArrayExpr because
    // indexing such a field (`self.cells[i]`) needs a Load-array-then-Index
    // lowering in astgen_index that doesn't exist yet.
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

/// Materialise every type's LLVM body. Split out of `register_and_analyze` and
/// run LATER — AFTER import handles + module namespaces are registered — so a
/// struct/enum whose field/payload is a CHAINED type (`w.leaf.Point`) can
/// resolve through the re-export chain when its body lays out; filling eagerly
/// (before namespaces exist) leaves such fields as stub bodies.
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

/// Resolve a const init expression AS A TYPE: a `Variable` naming a builtin
/// scalar or a user type, or a direct `Call` to a `type`-returning generic fn
/// -> a `GenericCall`. NONE otherwise. Drives type-alias-const registration
/// (`const CounterI32 = Counter(i32)`).
fn resolve_expr_as_type(cg: &mut CodegenContext, expr: NodeIdx) -> TypeIdx {
    use jam_syntax::ast_flat::builtin;
    if expr.is_none() {
        return TypeIdx::NONE;
    }
    let n = *cg.node_store.get(expr);
    match n.tag {
        AstTag::Variable => {
            let name =
                String::from_utf8_lossy(&cg.string_pool.get(StringIdx::new(n.lhs))).into_owned();
            match name.as_str() {
                "void" => builtin::VOID,
                "bool" | "u1" => builtin::BOOL,
                "u8" => builtin::U8,
                "i8" => builtin::I8,
                "u16" => builtin::U16,
                "i16" => builtin::I16,
                "u32" => builtin::U32,
                "i32" => builtin::I32,
                "u64" => builtin::U64,
                "i64" => builtin::I64,
                "f32" => builtin::F32,
                "f64" => builtin::F64,
                "type" => builtin::TYPE,
                "noreturn" => builtin::NORETURN,
                _ => {
                    let sid = cg.string_pool.intern(name.as_bytes());
                    cg.type_pool.intern_named(sid)
                }
            }
        }
        AstTag::Call => {
            if n.flags & 1 != 0 {
                return TypeIdx::NONE;
            }
            let callee =
                String::from_utf8_lossy(&cg.string_pool.get(StringIdx::new(n.lhs))).into_owned();
            match cg.get_function_ast(&callee) {
                Some(f) if f.is_generic() && f.return_type == builtin::TYPE => {}
                _ => return TypeIdx::NONE,
            }
            let args_extra = n.rhs;
            let arg_count = cg.node_store.get_extra(ExtraIdx::new(args_extra));
            let mut arg_types: Vec<TypeIdx> = Vec::with_capacity(arg_count as usize);
            for i in 0..arg_count {
                let arg_idx =
                    NodeIdx::new(cg.node_store.get_extra(ExtraIdx::new(args_extra + 1 + i)));
                let at = resolve_expr_as_type(cg, arg_idx);
                if at.is_none() {
                    return TypeIdx::NONE;
                }
                arg_types.push(at);
            }
            let sid = cg.string_pool.intern(callee.as_bytes());
            cg.type_pool.intern_generic_call(sid, arg_types)
        }
        _ => TypeIdx::NONE,
    }
}

/// What the driver emits after lowering: the JIR text dump (`--emit-jir`), the
/// LLVM IR (`--emit-ir`), a linked native executable (`build`), a compile +
/// run of the program (`jam run`), or a compile + run of the file's test
/// harness (`jam test <file>`). `Run`/`Test` carry the linker libraries
/// (`-l<name>`) to pass through to the link step.
pub enum EmitMode {
    Jir,
    /// `--emit-ir`: print the LLVM IR and exit before the object/link step.
    /// With `test` set the dump includes the synthesized harness `main`.
    Ir {
        test: bool,
    },
    Binary {
        output: String,
        libs: Vec<String>,
    },
    Run {
        output: String,
        libs: Vec<String>,
    },
    Test {
        output: String,
        libs: Vec<String>,
    },
}

/// `-C strip=MODE`: how much to strip from the linked binary. Applied as
/// linker flags at link time; no effect on the IR.
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub enum Strip {
    /// Keep everything (`none`/`off`/`false`/`no`).
    #[default]
    None,
    /// Strip DWARF / debug sections only (`debuginfo`).
    DebugInfo,
    /// Strip debug + local symbols (`symbols`).
    Symbols,
}

impl EmitMode {
    /// `true` for the test-harness build (`jam test <file>`, with or without
    /// `--emit-ir`): skip the user's `fn main`, lower `tfn` test functions, and
    /// synthesize a harness `main`.
    fn is_test(&self) -> bool {
        matches!(self, EmitMode::Test { .. } | EmitMode::Ir { test: true })
    }
}

/// Requalify + instantiate every decl-level GenericCall type (function/method
/// return+params, struct/union fields, enum payloads) so the monomorph TYPE
/// interns EARLY — at the oracle's string-pool position — while the deferral
/// flag keeps each generic's method bodies stashed for the later method
/// pre-pass. Each decl resolves in its OWN module's scope.
fn bind_decl_module(cg: &CodegenContext, m: &ModuleAST, with_fns: bool) {
    let inst = |ty: TypeIdx, mpath: &str| {
        cg.push_body_module(mpath.to_string());
        let q = cg.requalify_type(ty, mpath);
        let _ = cg.resolve_generic_call_instantiate(q);
        cg.pop_body_module();
    };
    // Function/method SIGNATURE types only run for imported modules: the entry's
    // own functions resolve lazily during body lowering (see the caller). But the
    // entry's aggregate TYPES (struct fields, enum payloads, union fields) ARE
    // laid out here, so e.g. an entry enum's `Many(Vec(u64))` payload requalifies
    // to `std/collections.Vec(u64)` rather than staying bare.
    if with_fns {
        for f in &m.functions {
            inst(f.return_type, &f.module_path);
            for p in &f.args {
                inst(p.ty, &f.module_path);
            }
        }
    }
    for s in &m.structs {
        for (_, fty) in &s.fields {
            inst(*fty, &s.module_path);
        }
        if with_fns {
            for meth in &s.methods {
                inst(meth.return_type, &meth.module_path);
                for p in &meth.args {
                    inst(p.ty, &meth.module_path);
                }
            }
        }
    }
    for e in &m.enums {
        for v in &e.variants {
            for pt in &v.payload_types {
                inst(*pt, &e.module_path);
            }
        }
    }
    for u in &m.unions {
        for (_, fty) in &u.fields {
            inst(*fty, &u.module_path);
        }
    }
}

/// `--target-info`: print the host target's properties. Standalone — prints;
/// the caller exits 0.
pub fn print_target_info() {
    let t = Target::from_triple_str(&default_target_triple());
    println!("Target Information:");
    println!("  Name: {}", t.name());
    println!("  Triple: {}", t.to_llvm_triple());
    println!("  Pointer size: {} bytes", t.pointer_size());
    println!("  Libc: {}", t.libc_name());
    println!(
        "  Requires PIC: {}",
        if t.requires_pic() { "yes" } else { "no" }
    );
    println!(
        "  Requires PIE: {}",
        if t.requires_pie() { "yes" } else { "no" }
    );
    println!("  Uses C ABI: {}", if t.uses_cabi() { "yes" } else { "no" });
    println!();
}

pub fn emit_jir(path: &str, mode: EmitMode, opt: OptLevel, lto: Lto, strip: Strip) -> i32 {
    // OSC 9;4 terminal progress for the whole compile: indeterminate while
    // compiling, error state on a diagnostic bail, cleared before handing the
    // terminal to the child (run/test) or printing the success line.
    let mut progress = ProgressGuard::new(!mode.is_test());
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => {
            progress.error();
            eprintln!("Could not open file: {path}");
            return 1;
        }
    };
    let mut lexer = Lexer::new(bytes);
    if lexer.scan_tokens().is_err() {
        progress.error();
        eprintln!("lex error in {path}");
        return 1;
    }
    let tokens = lexer.tokens().to_vec();
    let source = lexer.source().to_vec();

    // Parse into standalone pools FIRST. The analyzer's `DeclTable` ties the
    // module-AST borrow to the (invariant) `CodegenContext` lifetime, so the
    // module must be created BEFORE the context. The pools are then moved into
    // the context — the module references them only by index, so the move keeps
    // every `TypeIdx`/`NodeIdx`/`StringIdx` valid.
    let mut type_pool = TypePool::new();
    let mut string_pool = StringPool::new();
    let mut node_store = NodeStore::new();
    let module: ModuleAST = {
        let mut diags = Diagnostics::new();
        let mut p = Parser::new(
            tokens,
            source,
            &mut type_pool,
            &mut string_pool,
            &mut node_store,
            &mut diags,
            path,
        );
        match p.parse() {
            Ok(m) => m,
            Err(_) => {
                progress.error();
                eprint!("{}", diags.render_to_string());
                return 1;
            }
        }
    };

    // Resolve + load the transitive import closure into the SHARED pools BEFORE
    // the context exists — the loaded modules (like the entry) must outlive the
    // invariant context that the decl table borrows them for.
    let base_dir = std::path::Path::new(path)
        .parent()
        .and_then(|p| p.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(".");
    let mut resolver = ModuleResolver::new(base_dir);
    resolver.load_all(&module, &mut type_pool, &mut string_pool, &mut node_store);

    // Validate the entry module's imports: every import must resolve to a
    // loadable module, and every destructuring name must be EXPORTED. The
    // `--emit-tokens`/`--emit-ast` paths return earlier (separate functions),
    // so this only fires for jir/ir/build/test.
    if let Err(e) = resolver.validate_entry_imports(&module, path) {
        eprintln!("{e}");
        return 1;
    }

    let owner = Context::new();
    let mut cg = CodegenContext::new(&owner, "jam_module");
    cg.set_method_instantiator(jam_sema::astgen::instantiate_methods);
    cg.type_pool = type_pool;
    cg.string_pool = string_pool;
    cg.node_store = node_store;

    // Loaded modules in deterministic (sorted-by-identity) order — matches the
    // oracle's sorted `--emit-jir` iteration.
    let mut loaded_sorted: Vec<&(String, ModuleAST)> = resolver.loaded.iter().collect();
    loaded_sorted.sort_by(|a, b| a.0.cmp(&b.0));

    // For `--emit-ir` / `build`: the oracle iterates its loaded modules (a
    // libc++ `std::unordered_map`) in HASH-BUCKET order — NOT sorted. Reproduce
    // that exact iteration order with `libcxx_unordered_iteration_order`, fed
    // the loaded keys in their insertion (load) order. LLVM prints types/
    // globals/functions in value-CREATION order, so every imported-module pass
    // below must walk modules in this order for the IR to be byte-identical.
    // `--emit-jir` keeps `loaded_sorted` (the oracle sorts only for that mode).
    let test_mode = mode.is_test();
    let loaded_libcxx: Vec<&(String, ModuleAST)> = if matches!(mode, EmitMode::Jir) {
        loaded_sorted.clone()
    } else {
        let keys: Vec<String> = resolver.loaded.iter().map(|p| p.0.clone()).collect();
        let order = jam_sema::libcxx_order::libcxx_unordered_iteration_order(&keys);
        order.iter().map(|&i| &resolver.loaded[i]).collect()
    };

    // Register every module's types (qualified) + analyze, then register every
    // function (call-site name), struct method (qualified), and `cfn drop`.
    let mut all_modules: Vec<&ModuleAST> = Vec::with_capacity(1 + loaded_sorted.len());
    all_modules.push(&module);
    for pair in &loaded_sorted {
        all_modules.push(&pair.1);
    }
    // Register every function's call-site name BEFORE register_and_analyze so its
    // enum-payload / struct-field requalification can resolve a generic callee
    // (`Vec` -> `std/collections.Vec`) via get_function_ast. register_function_ast
    // touches no string pool, so the `--emit-jir` intern order is unaffected. The
    // loop further below re-registers (idempotent) + adds struct methods + the
    // drop registry.
    // In test mode an entry-file `tfn` registers under `__test_<name>` so it
    // doesn't shadow a regular function of the same source name.
    let reg_key = |func: &FunctionAST| -> String {
        if test_mode && func.is_test {
            format!("__test_{}", func.name)
        } else {
            func.name.clone()
        }
    };
    for m in &all_modules {
        for func in &m.functions {
            cg.register_function_ast(reg_key(func), func.clone());
            if !func.module_path.is_empty() {
                cg.register_function_ast(
                    format!("{}.{}", func.module_path, func.name),
                    func.clone(),
                );
            }
        }
    }
    // Struct-type registration / analysis ORDER: LLVM materializes named struct
    // types in creation order. The oracle declares IMPORTED modules' types FIRST
    // (libc++ iteration order) then the ENTRY module's. For `--emit-jir` the
    // original entry-first order over `loaded_sorted` is preserved (the JIR gate
    // doesn't observe LLVM type order).
    let reg_modules: Vec<&ModuleAST> = if matches!(mode, EmitMode::Jir) {
        all_modules.clone()
    } else {
        let mut v: Vec<&ModuleAST> = Vec::with_capacity(all_modules.len());
        for pair in &loaded_libcxx {
            v.push(&pair.1);
        }
        v.push(&module);
        v
    };
    register_and_analyze(&mut cg, &reg_modules);
    // Register each module's anonymous struct bodies (generic factories'
    // `return struct {...}`), keyed by module identity, so generic
    // instantiation can reach them.
    cg.register_anon_structs("", module.anon_structs.clone());
    cg.register_anon_enums("", module.anon_enums.clone());
    for pair in &loaded_sorted {
        cg.register_anon_structs(pair.0.clone(), pair.1.anon_structs.clone());
        cg.register_anon_enums(pair.0.clone(), pair.1.anon_enums.clone());
    }
    for m in &all_modules {
        for func in &m.functions {
            cg.register_function_ast(reg_key(func), func.clone());
            // Also register under the module-qualified name so a qualified
            // generic callee (`mod_x.Holder`) resolves to its factory.
            if !func.module_path.is_empty() {
                cg.register_function_ast(
                    format!("{}.{}", func.module_path, func.name),
                    func.clone(),
                );
            }
        }
        for s in &m.structs {
            let qself = qualify_type_name(&s.module_path, &s.name);
            for meth in &s.methods {
                cg.register_function_ast(format!("{qself}.{}", meth.name), meth.clone());
            }
        }
        // Methods of generic struct-returning factories register under the
        // factory's qualified name ("std/collections.Vec.push") — parameter
        // MODES don't depend on T, so the move analysis can resolve
        // `v.push(c)` through the receiver's requalified GenericCall callee.
        // Keys never collide with real struct methods (factories are
        // functions, not structs), and register_function_ast touches no string
        // pool, so the --emit-jir intern order is unaffected.
        for func in &m.functions {
            if !func.is_generic() {
                continue;
            }
            for &stmt in &func.body {
                let rn = *cg.node_store.get(stmt);
                if rn.tag != AstTag::Return {
                    continue;
                }
                if rn.lhs == 0 {
                    break;
                }
                let value = *cg.node_store.get(NodeIdx::new(rn.lhs));
                if value.tag != AstTag::StructExpr {
                    break;
                }
                let anon_idx = value.lhs as usize;
                if anon_idx >= m.anon_structs.len() {
                    break;
                }
                let qfn = qualify_type_name(&func.module_path, &func.name);
                for meth in &m.anon_structs[anon_idx].methods {
                    cg.register_function_ast(format!("{qfn}.{}", meth.name), meth.clone());
                }
            }
        }
        // Value consts inline at their use sites — register by bare name (and
        // module-qualified name for imports). Type-alias consts are skipped.
        for c in &m.consts {
            if !c.aliased_type.is_none() || c.init_expr.is_none() {
                continue;
            }
            let info = ModuleConstInfo {
                init_expr: c.init_expr,
                declared_type: c.declared_type,
                is_comp: c.is_comp,
                bare_name: c.name.clone(),
                module_path: c.module_path.clone(),
            };
            register_const_keyed(&cg, c, info);
        }
    }

    // Eager `comp const` validation: every comp-marked module const must fold
    // against the const fixpoint in its OWN module's scope. A non-foldable init
    // (`comp const X = 10 / 0;` — div-by-zero, runtime-dependent, or otherwise
    // unsupported) is rejected HERE at the declaration; without this the use site
    // re-lowers the unfoldable init and emits poison IR. Plain consts stay lazy.
    // Only fires for jir/ir/build/test (the token/ast paths return earlier).
    {
        let mut diags = Diagnostics::new();
        for m in &all_modules {
            for c in &m.consts {
                if !c.is_comp || !c.aliased_type.is_none() || c.init_expr.is_none() {
                    continue;
                }
                cg.push_body_module(c.module_path.clone());
                let v = cg.fold_comptime_expr(c.init_expr);
                cg.pop_body_module();
                if v.is_none() {
                    let file = if c.module_path.is_empty() {
                        path.to_string()
                    } else {
                        format!("{}.jam", c.module_path)
                    };
                    let line = cg.node_store.get_line(c.init_expr);
                    diags.error(
                        jam_core::diag::SrcLoc::new(&file, line),
                        format!(
                            "comp const `{}` must be compile-time evaluable — its initializer depends on a runtime value or an unsupported construct",
                            c.name
                        ),
                    );
                }
            }
        }
        if diags.has_errors() {
            eprint!("{}", diags.render_to_string());
            return 1;
        }
    }

    // Destructuring-import type aliases: `const { Token } = import("m")` brings
    // the type into scope under its bare name; resolve it to `m.Token`. Build a
    // bare-type-name -> qualified `Named` map over the loaded modules, then alias
    // each destructuring-imported type name.
    {
        let mut type_by_bare: std::collections::HashMap<String, TypeIdx> =
            std::collections::HashMap::new();
        for pair in &loaded_sorted {
            let lm = &pair.1;
            let names = lm
                .structs
                .iter()
                .map(|s| (&s.name, &s.module_path))
                .chain(lm.enums.iter().map(|e| (&e.name, &e.module_path)))
                .chain(lm.unions.iter().map(|u| (&u.name, &u.module_path)));
            for (name, module_path) in names {
                let q = qualify_type_name(module_path, name);
                let sid = cg.string_pool.intern(q.as_bytes());
                let ty = cg.type_pool.intern_named(sid);
                type_by_bare.insert(name.clone(), ty);
            }
        }
        for m in &all_modules {
            for di in &m.destructuring_imports {
                for name in &di.names {
                    if let Some(&ty) = type_by_bare.get(name) {
                        cg.register_type_alias(name.clone(), ty);
                    }
                }
            }
        }

        // Type-alias consts (`const CounterI32 = Counter(i32)`): a const whose
        // init expression resolves-as-type to a GenericCall is a type alias;
        // register it (bare + qualified) to the requalified target.
        // name_for_kinds chases these.
        for m in &all_modules {
            for c in &m.consts {
                if c.init_expr.is_none() {
                    continue;
                }
                let target = resolve_expr_as_type(&mut cg, c.init_expr);
                if cg.type_pool.get(target).kind != TypeKind::GenericCall {
                    continue;
                }
                let target = cg.requalify_type(target, &c.module_path);
                cg.register_type_alias(c.name.clone(), target);
                if !c.module_path.is_empty() {
                    cg.register_type_alias(format!("{}.{}", c.module_path, c.name), target);
                }
            }
        }

        // Import handles (`const lib = import("m")` / `const fmt = import("std")
        // .fmt`): map the handle name to the resolved module identity so
        // `lib.fn()` dispatches to `m.fn`. The identity is the import path plus
        // any `.chain` segments, validated against the loaded module set.
        let loaded_keys: std::collections::HashSet<&str> =
            loaded_sorted.iter().map(|p| p.0.as_str()).collect();
        for (i, m) in all_modules.iter().enumerate() {
            let id = if i == 0 {
                String::new()
            } else {
                loaded_sorted[i - 1].0.clone()
            };
            for imp in &m.imports {
                // Resolve the import spelling to its real identity (a RELATIVE
                // re-export like std/std.jam's `import("fmt")` resolves to
                // `std/fmt`), then fold any `.chain` segments through the SAME
                // identities table: `import("std").fmt` walks std/std.jam's
                // `fmt` re-export to `std/fmt`, not the naive `std/std/fmt`
                // concatenation. (The old global handle map hid this — the
                // entry's chained handle worked only because std's own
                // re-export leaked a same-named global handle.)
                let mut candidate = resolver
                    .import_identities
                    .get(&(id.clone(), imp.path.clone()))
                    .cloned()
                    .unwrap_or_else(|| imp.path.clone());
                for seg in &imp.chain {
                    candidate = resolver
                        .import_identities
                        .get(&(candidate.clone(), seg.clone()))
                        .cloned()
                        .unwrap_or_else(|| format!("{candidate}/{seg}"));
                }
                if loaded_keys.contains(candidate.as_str()) {
                    cg.register_import_handle(id.clone(), imp.name.clone(), candidate);
                }
            }
        }

        // Per-module namespaces for multi-dot chained access (`w.leaf.Point`):
        // each module's pub types + pub re-export module aliases, keyed by the
        // module's identity (entry = "", else its loaded key).
        for (i, m) in all_modules.iter().enumerate() {
            let id = if i == 0 {
                String::new()
            } else {
                loaded_sorted[i - 1].0.clone()
            };
            let mut ns = jam_sema::codegen_context::ModuleNamespace::default();
            let mut add_type = |name: &str, module_path: &str, cg: &mut CodegenContext| {
                let q = qualify_type_name(module_path, name);
                let sid = cg.string_pool.intern(q.as_bytes());
                ns.types
                    .insert(name.to_string(), cg.type_pool.intern_named(sid));
            };
            // Only PUB decls enter an imported module's namespace; non-pub
            // names are recorded so handle-qualified access reports "not
            // exported". The entry module (i == 0) has no import handle
            // pointing at it, so its namespace stays unrestricted.
            let entry = i == 0;
            for s in &m.structs {
                if entry || s.is_pub {
                    add_type(&s.name, &s.module_path, &mut cg);
                } else {
                    ns.private_names.insert(s.name.clone());
                }
            }
            for e in &m.enums {
                if entry || e.is_pub {
                    add_type(&e.name, &e.module_path, &mut cg);
                } else {
                    ns.private_names.insert(e.name.clone());
                }
            }
            for u in &m.unions {
                if entry || u.is_pub {
                    add_type(&u.name, &u.module_path, &mut cg);
                } else {
                    ns.private_names.insert(u.name.clone());
                }
            }
            for f in &m.functions {
                if !entry && !f.is_pub && !f.is_extern && !f.is_export {
                    ns.private_names.insert(f.name.clone());
                }
            }
            for imp in &m.imports {
                if !imp.is_pub {
                    continue;
                }
                // Resolve the spelling to its real identity so a RELATIVE
                // re-export (std/std.jam's `pub const fmt = import("fmt")`)
                // registers `module_aliases["fmt"] -> "std/fmt"`, which is what
                // walk_chain(std -> fmt -> println) needs.
                let base = resolver
                    .import_identities
                    .get(&(id.clone(), imp.path.clone()))
                    .cloned()
                    .unwrap_or_else(|| imp.path.clone());
                let candidate = if imp.chain.is_empty() {
                    base
                } else {
                    format!("{}/{}", base, imp.chain.join("/"))
                };
                if loaded_keys.contains(candidate.as_str()) {
                    // Intern the target path — the oracle interns this string
                    // at this exact point, so the string-pool order depends on it.
                    cg.string_pool.intern(candidate.as_bytes());
                    ns.module_aliases.insert(imp.name.clone(), candidate);
                }
            }
            cg.register_module_namespace(id, ns);
        }
    }

    // In-place FunctionAST signature requalification. Decl-type references
    // requalify lazily at use, but a cross-module call site lowers the callee's
    // param/return types against ITS OWN body module (""), where a bare imported
    // Named (`Inner`, `Pair`, `Status`) has no requalify entry — so the param
    // classify fails with "unresolved Named type". Rewrite each loaded module's
    // function (and struct-method) return + param types to their module-qualified
    // identity and RE-REGISTER the FunctionAST under the same keys the
    // registration loop used.
    {
        // Requalify a function's return + each param against `mpath`. Bare Named
        // leaves resolve through the requalify map; a GenericCall callee chases
        // qualify_generic_callee. Returns the rewritten clone and whether
        // anything changed.
        let requalify_fn =
            |cg: &CodegenContext, f: &FunctionAST, mpath: &str| -> (FunctionAST, bool) {
                let mut nf = f.clone();
                let mut changed = false;
                let rq = cg.requalify_type(nf.return_type, mpath);
                if rq != nf.return_type {
                    nf.return_type = rq;
                    changed = true;
                }
                for p in &mut nf.args {
                    let rq = cg.requalify_type(p.ty, mpath);
                    if rq != p.ty {
                        p.ty = rq;
                        changed = true;
                    }
                }
                (nf, changed)
            };
        // Only LOADED (imported) modules: the entry's own signatures already
        // lower against their own ("") body module, where bare references resolve
        // directly. Mirror the registration keys exactly.
        for pair in &loaded_sorted {
            let m = &pair.1;
            for func in &m.functions {
                let (nf, changed) = requalify_fn(&cg, func, &func.module_path);
                if !changed {
                    continue;
                }
                cg.register_function_ast(reg_key(&nf), nf.clone());
                if !nf.module_path.is_empty() {
                    cg.register_function_ast(format!("{}.{}", nf.module_path, nf.name), nf);
                }
            }
            for s in &m.structs {
                let qself = qualify_type_name(&s.module_path, &s.name);
                for meth in &s.methods {
                    let (nm, changed) = requalify_fn(&cg, meth, &meth.module_path);
                    if !changed {
                        continue;
                    }
                    cg.register_function_ast(format!("{qself}.{}", nm.name), nm);
                }
            }
        }
    }

    // Scoped handle.Type aliases. A type annotation `var s: a.Status` /
    // `fn f(pb: B.Pair)` carries the literal dotted Named `a.Status`; requalify
    // early-returns on dotted names and there is no `handle.Type` alias, so the
    // registry lookup misses. For every import handle in every module, alias each
    // PUB struct/enum/union of the handle's target module to its qualified
    // identity. PRIVATE types are NOT aliased — a handle-spelled private access
    // then misses and the `is not exported` diagnostic fires (privacy holds).
    {
        let loaded_keys: std::collections::HashSet<&str> =
            loaded_sorted.iter().map(|p| p.0.as_str()).collect();
        let loaded_by_id: std::collections::HashMap<&str, &ModuleAST> =
            loaded_sorted.iter().map(|p| (p.0.as_str(), &p.1)).collect();
        for (i, m) in all_modules.iter().enumerate() {
            let owner_id = if i == 0 {
                String::new()
            } else {
                loaded_sorted[i - 1].0.clone()
            };
            for imp in &m.imports {
                // Resolve the handle's import spelling to its real identity.
                let base = resolver
                    .import_identities
                    .get(&(owner_id.clone(), imp.path.clone()))
                    .cloned()
                    .unwrap_or_else(|| imp.path.clone());
                let target_id = if imp.chain.is_empty() {
                    base
                } else {
                    format!("{}/{}", base, imp.chain.join("/"))
                };
                if !loaded_keys.contains(target_id.as_str()) {
                    continue;
                }
                let Some(target_mod) = loaded_by_id.get(target_id.as_str()) else {
                    continue;
                };
                // Alias `<handle>.<Type>` -> the type's qualified `Named`. Skip the
                // no-op self-alias when the handle spelling equals the qualified
                // identity (a handle named like its module): the registries already
                // hold the identity, and a self-alias would tail-recurse forever.
                let alias_named = |name: &str, module_path: &str, cg: &mut CodegenContext| {
                    let qualified = qualify_type_name(module_path, name);
                    let key = format!("{}.{}", imp.name, name);
                    if key == qualified {
                        return;
                    }
                    let sid = cg.string_pool.intern(qualified.as_bytes());
                    let target = cg.type_pool.intern_named(sid);
                    cg.register_type_alias(key, target);
                };
                for s in &target_mod.structs {
                    if s.is_pub {
                        alias_named(&s.name, &s.module_path, &mut cg);
                    }
                }
                for e in &target_mod.enums {
                    if e.is_pub {
                        alias_named(&e.name, &e.module_path, &mut cg);
                    }
                }
                for u in &target_mod.unions {
                    if u.is_pub {
                        alias_named(&u.name, &u.module_path, &mut cg);
                    }
                }
            }
        }
    }

    // Lay out every type body now that import handles + module namespaces are
    // registered. For --emit-jir this runs HERE: its type/string interning sets
    // the dump's intern order. For --emit-ir/binary the fill is DEFERRED to after
    // the extern/prototype passes (below), so the ENTRY-module field/payload
    // generic instantiation (`items: Vec(u64)`) emits its methods AFTER Pass A's
    // externs, matching the oracle's demand-driven body fill.
    // Drop + clone registry population is ALSO deferred to after the type-body
    // fill: a conditional method like `Box(T).clone` instantiates DURING that
    // layout, when the drop registry must still be empty — so its
    // `self.ptr[0].clone()` over a drop-bearing T sees NO registered drop and
    // field-wise clones (Tier 2) instead of erroring. Populating the registry
    // before the fill turned that into a spurious owns-resources error,
    // withdrawing the clone body to a bare `declare`.
    let register_drop_clone = |cg: &mut CodegenContext, all_modules: &[&ModuleAST]| {
        for m in all_modules {
            let drops = build_drop_registry(m, &cg.type_pool, &cg.string_pool);
            for (type_name, drop_fn) in &drops {
                let mangled = mangled_function_name(drop_fn, &cg.type_pool, &cg.string_pool);
                cg.register_drop_fn(type_name.clone(), mangled.clone());
                // Imported drop-bearing types are referenced by their QUALIFIED
                // name (`mod.XThing`); register the drop there too so the field/
                // scope drop walk resolves it.
                if !drop_fn.module_path.is_empty() {
                    cg.register_drop_fn(format!("{}.{}", drop_fn.module_path, type_name), mangled);
                }
            }
            // A TOP-LEVEL `cfn clone(self: T) T` goes in the clone registry (NOT as
            // a `{T}.clone` method) so a direct `t.clone()` routes through
            // emit_clone_into's glue (passing `srcPtr`) rather than ordinary
            // method dispatch. In-struct clones already register as `{T}.clone`
            // methods.
            let mut clones = CloneRegistry::new();
            add_clone_candidates(&mut clones, m, &cg.type_pool, &cg.string_pool);
            for (type_name, clone_fn) in &clones {
                cg.register_clone_fn(type_name.clone(), (*clone_fn).clone());
            }
        }
    };

    if matches!(mode, EmitMode::Jir) {
        fill_type_bodies(&mut cg, &reg_modules);
        register_drop_clone(&mut cg, &all_modules);
    }
    // The analyzer pushes type-cycle diagnostics ("struct `A` depends on
    // itself", with the reference-trace chain) onto the context's GLOBAL
    // collector — flush them before lowering walks the infinite-sized types.
    if cg.has_errors() {
        progress.error();
        eprint!("{}", cg.diagnostics().render_to_string());
        return 1;
    }

    let mut jir_functions: Vec<JirFunction> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let lower = |cg: &mut CodegenContext,
                 f: &FunctionAST,
                 jir: &mut Vec<JirFunction>,
                 errs: &mut Vec<String>| {
        match astgen_function(f, cg) {
            Ok(mut jfn) => {
                jfn.name = mangled_function_name(f, &cg.type_pool, &cg.string_pool);
                if let Err(e) = jir_declare_prototype(&jfn, cg) {
                    errs.push(e);
                    return;
                }
                jir.push(jfn);
            }
            Err(e) => errs.push(e),
        }
    };
    // Like `lower`, but WITHOUT re-declaring the LLVM prototype — the function was
    // already declared by an earlier pass (Pass B for imported free fns, the method
    // pre-pass for struct methods). Re-declaring would make LLVM rename the
    // duplicate (`@helper.1`), which the oracle's IR does NOT carry for these.
    // Only the ENTRY module's free fns are declared at lowering time.
    let lower_nodecl = |cg: &mut CodegenContext,
                        f: &FunctionAST,
                        jir: &mut Vec<JirFunction>,
                        errs: &mut Vec<String>| {
        match astgen_function(f, cg) {
            Ok(mut jfn) => {
                jfn.name = mangled_function_name(f, &cg.type_pool, &cg.string_pool);
                jir.push(jfn);
            }
            Err(e) => errs.push(e),
        }
    };
    // Instantiate every IMPORTED-module decl-type generic monomorph EARLY
    // (type-only — method bodies deferred to the method pre-pass), in module LOAD
    // order, so the type monomorphs intern at the oracle's string-pool position
    // instead of lazily during body lowering. The ENTRY module is deliberately
    // skipped: its own decl types resolve lazily when its bodies lower (e.g. the
    // entry's `fn unwrapI32(o: Option(i32))` interns `Option__i32` late).
    // `--emit-jir` only: the early type-only instantiation reproduces the
    // oracle's string-pool order, but its deferred method lowering sets generic
    // LLVM struct bodies before all dependencies are ready, which the
    // object-emission backend can't consume. The IR/binary paths keep the
    // original coupled instantiation (correct codegen; intern order is
    // irrelevant to a binary).
    if matches!(mode, EmitMode::Jir) {
        cg.set_defer_method_lowering(true);
        for pair in &resolver.loaded {
            bind_decl_module(&cg, &pair.1, /*with_fns=*/ true);
        }
        // Entry: aggregate types only (its functions resolve lazily during lowering).
        bind_decl_module(&cg, &module, /*with_fns=*/ false);
        cg.set_defer_method_lowering(false);
        let _ = cg.drain_pending_method_lowering();
    }

    // Sig pre-pass: qualify each non-generic function's return/param GenericCall
    // callees (`Vec(u32)` -> `std/collections.Vec`) BEFORE any body lowers — the
    // oracle's prototype-declaration pass interns the qualified callee here, ahead
    // of the first body's drop-binding. Imported then entry, matching lowering.
    let qualify_sig = |cg: &mut CodegenContext, f: &FunctionAST, m: &str| {
        if cg.type_pool.get(f.return_type).kind == TypeKind::GenericCall {
            cg.qualify_generic_callee(f.return_type, m);
        }
        for p in &f.args {
            if cg.type_pool.get(p.ty).kind == TypeKind::GenericCall {
                cg.qualify_generic_callee(p.ty, m);
            }
        }
    };
    for pair in &loaded_libcxx {
        for func in &pair.1.functions {
            if func.is_generic() || func.is_extern {
                continue;
            }
            qualify_sig(&mut cg, func, &pair.0);
        }
    }
    for func in &module.functions {
        if func.is_test || func.is_generic() {
            continue;
        }
        qualify_sig(&mut cg, func, "");
    }
    // Pass A: declare `pub extern` fn prototypes (malloc/free/realloc from
    // std/box + std/collections) BEFORE any body lowers, so generic
    // instantiation — which calls them during method codegen — finds them
    // already in the LLVM module. jir_declare_prototype is LLVM-only (no
    // string/type interning), and extern sigs are primitives, so --emit-jir is
    // unaffected. These declarations only surface in --emit-ir.
    for pair in &loaded_libcxx {
        for func in &pair.1.functions {
            if func.is_pub
                && func.is_extern
                && !func.is_generic()
                && let Ok(mut jfn) = astgen_function(func, &mut cg)
            {
                jfn.name = mangled_function_name(func, &cg.type_pool, &cg.string_pool);
                let _ = jir_declare_prototype(&jfn, &cg);
            }
        }
    }
    // Pass B: for each imported module (libc++ iteration order), declare
    // prototypes for ALL its non-generic functions — pub AND private — INCLUDING
    // the externs already declared in Pass A. The second extern declaration is
    // intentional: LLVM auto-renames the duplicate (`@malloc.1`, `@free.2`),
    // which the oracle's IR carries. Private helpers also need prototypes so the
    // pub bodies that reference them resolve at LLVM codegen. Wrapped in
    // push/pop_body_module so each signature resolves in its own module's scope
    // (a private-type param must not trip another module's privacy gate).
    // Gated to non-Jir so the JIR gate is untouched.
    if !matches!(mode, EmitMode::Jir) {
        for pair in &loaded_libcxx {
            cg.push_body_module(pair.0.clone());
            for func in &pair.1.functions {
                if func.is_generic() {
                    continue;
                }
                let mut jfn = astgen_metadata(func, &cg);
                jfn.name = mangled_function_name(func, &cg.type_pool, &cg.string_pool);
                let _ = jir_declare_prototype(&jfn, &cg);
            }
            cg.pop_body_module();
        }
    }
    // (--emit-ir/binary) Fill the type bodies now — AFTER Pass A/B have declared
    // the libc externs + imported prototypes — so a field/payload generic's
    // methods emit after the externs, matching the oracle's emission order. The
    // JIR path already filled above.
    if !matches!(mode, EmitMode::Jir) {
        fill_type_bodies(&mut cg, &reg_modules);
        register_drop_clone(&mut cg, &all_modules);
        // Same global-collector flush as the JIR path: a type cycle must stop
        // the compile before codegen recurses into the cyclic layout.
        if cg.has_errors() {
            progress.error();
            eprint!("{}", cg.diagnostics().render_to_string());
            return 1;
        }
    }
    // Method-signature metadata pre-pass: run astgen_metadata() on every struct
    // method BEFORE any function body lowers, so a method whose RETURN/PARAM type
    // is a GenericCall (`fn readAll(self) Vec(u8)`) instantiates that generic —
    // interning its method names (`Vec__u8.withCapacity`) — at signature-
    // resolution time, ahead of the body call sites that name cross-module
    // callees. Without this the generic interns during readAll's body walk,
    // AFTER cpath/openMode bodies, reversing the oracle's string-pool order.
    // Imported (sorted) then entry; jir_declare_prototype is LLVM-only so
    // --emit-jir is unmoved.
    for pair in &loaded_libcxx {
        cg.push_body_module(pair.0.clone());
        for s in &pair.1.structs {
            for meth in &s.methods {
                if !meth.is_pub && !meth.is_cfn {
                    continue;
                }
                let mut jfn = astgen_metadata(meth, &cg);
                jfn.name = mangled_function_name(meth, &cg.type_pool, &cg.string_pool);
                let _ = jir_declare_prototype(&jfn, &cg);
            }
        }
        cg.pop_body_module();
    }
    for s in &module.structs {
        for meth in &s.methods {
            let mut jfn = astgen_metadata(meth, &cg);
            jfn.name = mangled_function_name(meth, &cg.type_pool, &cg.string_pool);
            let _ = jir_declare_prototype(&jfn, &cg);
        }
    }
    // Body lowering ORDER. The pushed `jir_functions` order is BOTH the `--emit-jir`
    // dump order AND the `--emit-ir` body-DEFINITION order (the later jir_define_body
    // loop walks `jir_functions` in push order, and LLVM prints each function — and
    // first-referenced struct type — in definition order).
    //
    // For `--emit-jir` the oracle dumps ENTRY FIRST then imported (sorted): keep
    // that. For `--emit-ir`/build the oracle emits all IMPORTED modules (libc++
    // order) FIRST and the ENTRY module LAST. Lower in that inverted order so the
    // function-definition sequence — and the struct-type-decl order that follows
    // from first reference — matches the oracle.
    // Only the ENTRY module's free fns get a fresh prototype at lowering time
    // (JIR keeps its original all-declare behavior — harmless, since the JIR
    // text dump ignores LLVM prototypes).
    let is_jir = matches!(mode, EmitMode::Jir);
    let lower_imported =
        |cg: &mut CodegenContext, jir: &mut Vec<JirFunction>, errs: &mut Vec<String>| {
            for pair in &loaded_libcxx {
                let (key, m) = (&pair.0, &pair.1);
                cg.push_body_module(key.clone());
                // Swap the display file to this imported module's defining file so
                // an astgen diagnostic in an imported body is attributed to it,
                // then restore.
                let prev_file = cg.current_file();
                cg.set_current_file(format!("{key}.jam"));
                for func in &m.functions {
                    if func.is_generic() || func.is_extern {
                        continue;
                    }
                    if is_jir {
                        lower(cg, func, jir, errs);
                    } else {
                        lower_nodecl(cg, func, jir, errs);
                    }
                }
                for s in &m.structs {
                    for meth in &s.methods {
                        if !meth.is_pub && !meth.is_cfn {
                            continue;
                        }
                        if is_jir {
                            lower(cg, meth, jir, errs);
                        } else {
                            lower_nodecl(cg, meth, jir, errs);
                        }
                    }
                }
                cg.set_current_file(prev_file);
                cg.pop_body_module();
            }
        };
    // Test-mode harness data: each `tfn`'s display name (its source name, no
    // prefix) and its mangled LLVM symbol (`__test_<name>`), in source order —
    // the synthesized `main` walks this to print "testing <name>... " + call.
    let test_mode = mode.is_test();
    let mut test_entries: Vec<(String, String)> = Vec::new();
    {
        // In test mode the harness calls each `tfn` by its mangled LLVM symbol;
        // record them (source order) before lowering so the harness, built
        // after body lowering, can look each up.
        for func in &module.functions {
            if func.is_test && !func.is_generic() {
                let mangled = mangled_function_name(func, &cg.type_pool, &cg.string_pool);
                test_entries.push((func.name.clone(), mangled));
            }
        }
    }
    let lower_entry =
        |cg: &mut CodegenContext, jir: &mut Vec<JirFunction>, errs: &mut Vec<String>| {
            for func in &module.functions {
                if func.is_generic() {
                    continue;
                }
                // Non-test mode: skip `tfn` test functions. Test mode: lower the
                // `tfn` bodies but skip the user's `fn main` (the harness supplies
                // its own).
                if !test_mode && func.is_test {
                    continue;
                }
                if test_mode && !func.is_test && func.name == "main" {
                    continue;
                }
                lower(cg, func, jir, errs);
            }
            for s in &module.structs {
                for m in &s.methods {
                    if is_jir {
                        lower(cg, m, jir, errs);
                    } else {
                        lower_nodecl(cg, m, jir, errs);
                    }
                }
            }
        };
    // Both JIR and IR push ENTRY bodies first, then imported (libc++ for IR,
    // sorted for JIR): the oracle's function list is entry-first, so the
    // body-DEFINITION order — and the @str global / first-referenced-type order
    // that follows it — is entry-first too.
    // Stamp the entry display file for astgen diagnostics; lower_imported swaps
    // per module.
    cg.set_current_file(path);
    lower_entry(&mut cg, &mut jir_functions, &mut errors);
    lower_imported(&mut cg, &mut jir_functions, &mut errors);
    if !errors.is_empty() {
        progress.error();
        for e in &errors {
            eprintln!("{e}");
        }
        return 1;
    }

    // Verify each JirFunction's structural invariants before codegen: malformed
    // dispatch, missing terminators, OOB refs, cross-block use-before-def. The
    // resolver resolves a body's bare GenericCall / ArrayExpr type refs against
    // its own module. Gated to the codegen path (the oracle's `--emit-jir` dump
    // returns before this pass) to keep the JIR gate byte-identical. All
    // diagnostics are stamped with the entry filename.
    if !matches!(mode, EmitMode::Jir) {
        let mut diags = Diagnostics::new();
        let resolver = |t: TypeIdx| -> TypeIdx {
            let k = cg.type_pool.get(t);
            if k.kind == TypeKind::GenericCall {
                let r = cg.resolve_generic_call(t);
                if !r.is_none() {
                    return r;
                }
            }
            if k.kind == TypeKind::ArrayExpr {
                return cg.resolve_array_expr(t);
            }
            t
        };
        for jfn in &jir_functions {
            cg.push_body_module(jfn.module_path.clone());
            let found = jam_sema::jir_verify::verify_jir_function(
                jfn,
                Some(&cg.type_pool),
                Some(&cg.string_pool),
                Some(&resolver),
            );
            cg.pop_body_module();
            for d in found {
                // jir_verify leaves the file empty; stamp the entry unit's file.
                let line = d.loc.line;
                diags.error(jam_core::diag::SrcLoc::new(path, line), d.message);
            }
        }
        if diags.has_errors() {
            progress.error();
            eprint!("{}", diags.render_to_string());
            return 1;
        }
    }

    // Whole-module definite-init / move-ownership sweep. Runs over EVERY
    // non-generic entry function, struct method, and imported-module
    // function/method — AFTER every prototype is in scope, BEFORE codegen.
    // Gated to the codegen path (same reason as the verify pass above).
    // Diagnostics funnel into a `Diagnostics` channel so they render
    // `file:line: error: message`.
    if !matches!(mode, EmitMode::Jir) {
        let mut diags = Diagnostics::new();
        let mut run_analysis_in = |cg: &CodegenContext, f: &FunctionAST, file: &str| {
            if f.is_extern {
                return;
            }
            // Resolve body-level type references against the function's own
            // module while analysis runs.
            cg.push_body_module(f.module_path.clone());
            let found = jam_sema::init_analysis::analyze(f, cg);
            cg.pop_body_module();
            for d in found {
                diags.error(jam_core::diag::SrcLoc::new(file, d.line), d.message);
            }
        };
        // Entry module: every non-generic function, with the same test-mode
        // selection as lowering (skip `tfn`s when not testing; skip the user's
        // `main` when testing). Then every struct method.
        for func in &module.functions {
            if func.is_test && !test_mode {
                continue;
            }
            if !func.is_test && test_mode && func.name == "main" {
                continue;
            }
            if func.is_generic() {
                continue;
            }
            run_analysis_in(&cg, func, path);
        }
        for s in &module.structs {
            for m in &s.methods {
                run_analysis_in(&cg, m, path);
            }
        }
        // Imported modules: diagnostics attributed to the DEFINING file
        // (`key + ".jam"`). Generic factory bodies stay skipped (their clones are
        // the analyzable artifacts).
        for pair in &loaded_sorted {
            let imported_file = format!("{}.jam", pair.0);
            for func in &pair.1.functions {
                if func.is_generic() {
                    continue;
                }
                run_analysis_in(&cg, func, &imported_file);
            }
            for s in &pair.1.structs {
                for m in &s.methods {
                    run_analysis_in(&cg, m, &imported_file);
                }
            }
        }
        if diags.has_errors() {
            progress.error();
            eprint!("{}", diags.render_to_string());
            return 1;
        }
    }

    // `--emit-ir` / `build`: lower each JIR function's body to LLVM (after all
    // prototypes were declared above), so cross-function references resolve.
    // Then either print the module IR or emit an object file and link it into a
    // native executable.
    if !matches!(mode, EmitMode::Jir) {
        for f in &jir_functions {
            // Push the function's owning module so `get_llvm_type` requalifies its
            // bare type references (`File` -> `std/fs.File`) against the right
            // module.
            cg.push_body_module(f.module_path.clone());
            if let Err(e) = jir_define_body(f, &cg) {
                errors.push(e);
            }
            cg.pop_body_module();
        }
        if !errors.is_empty() {
            for e in &errors {
                eprintln!("{e}");
            }
            return 1;
        }
        // Test mode: synthesize the harness `main` that drives every `tfn`.
        // A file with zero tests links/runs nothing and exits 0.
        if test_mode {
            if test_entries.is_empty() {
                return 0;
            }
            synthesize_test_main(&cg, &test_entries);
        }
        return match mode {
            EmitMode::Ir { .. } => {
                // Clear the indicator before the IR hits stdout.
                progress.stop();
                print!("{}", cg.module().print_to_string());
                0
            }
            EmitMode::Binary { output, libs } => {
                let rc = link_binary(&cg, &output, &libs, opt, lto, strip);
                if rc != 0 {
                    return rc;
                }
                // Clear the indicator before the success line.
                progress.stop();
                println!("Compilation successful: {output}");
                0
            }
            EmitMode::Run { output, libs } => {
                let rc = link_binary(&cg, &output, &libs, opt, lto, strip);
                if rc != 0 {
                    return rc;
                }
                // Clear the indicator before handing the terminal to the child.
                progress.stop();
                run_binary(&output)
            }
            EmitMode::Test { output, libs } => {
                link_and_run_test(&cg, &output, &libs, opt, lto, strip, &mut progress)
            }
            EmitMode::Jir => unreachable!(),
        };
    }

    let mut o: Vec<u8> = Vec::new();
    for f in &jir_functions {
        dump_jir_function(&mut o, &cg.type_pool, &cg.string_pool, f);
    }
    let stdout = io::stdout();
    let mut w = stdout.lock();
    let _ = w.write_all(&o);
    let _ = w.flush();
    0
}

/// Optimize the module and emit the intermediate object (or bitcode, under
/// LTO) next to `output`. Returns the intermediate's path.
fn emit_object(cg: &CodegenContext, output: &str, opt: OptLevel, lto: Lto) -> Result<String, i32> {
    // Initialize the native target + asm printer before creating the target
    // machine (object emission needs the backend registered).
    jam_llvm::init_native_target();
    jam_llvm::init_native_asm_printer();
    let triple = default_target_triple();
    cg.module().set_target_triple(&triple);
    let host = Target::from_triple_str(&triple);
    let pic = host.requires_pie() || host.requires_pic();
    let Some(tm) = TargetMachine::new(&triple, "generic", "", pic, opt, lto) else {
        eprintln!("Failed to create target machine");
        return Err(1);
    };
    tm.configure_module(cg.module());
    // Run the IR optimization pipeline before object emit — only at a non-zero
    // level; `--emit-ir` never optimizes.
    if !matches!(opt, OptLevel::None) {
        tm.run_optimization(cg.module());
    }
    // LTO emits LLVM bitcode (.bc) instead of a native object; clang's LTO
    // plugin re-runs the optimization at link time.
    let obj = format!("{output}.{}", if lto == Lto::Off { "o" } else { "bc" });
    if let Err(e) = tm.emit_to_file(cg.module(), &obj) {
        eprintln!("Failed to emit object file: {e}");
        return Err(1);
    }
    Ok(obj)
}

/// The link flags that follow `clang <obj> -o <out>` — libm, user `-l` libs,
/// LTO, dead-strip, and `-C strip` plumbing. Shared between the normal link
/// and the test-cache link (and FNV-mixed into the cache key).
fn link_flags(
    host: &Target,
    libs: &[String],
    opt: OptLevel,
    lto: Lto,
    strip: Strip,
) -> Vec<String> {
    let mut flags: Vec<String> = Vec::new();
    // libm: `frem` on floats lowers to a `fmod` libcall. glibc keeps math in
    // libm.so (pass `-lm`); macOS/Windows/musl bundle it into libc.
    if matches!(host.os, Os::Linux | Os::FreeBsd) && host.abi != Abi::Musl {
        flags.push("-lm".to_string());
    }
    // User-requested `-l<name>` libraries pass through to the link step
    // (e.g. `jam run -lncurses tetris.jam`).
    for lib in libs {
        flags.push(format!("-l{lib}"));
    }
    // Hand the bitcode to clang's LTO driver so it selects the right linker
    // plugin.
    match lto {
        Lto::Thin => flags.push("-flto=thin".to_string()),
        Lto::Fat => flags.push("-flto=full".to_string()),
        Lto::Off => {}
    }
    // Strip unreferenced functions/data at link time.
    // Pairs with FunctionSections/DataSections on the TargetMachine, which
    // split each symbol into its own section so the linker can GC them
    // individually. Mach-O uses -dead_strip; ELF (Linux/FreeBSD) uses
    // --gc-sections. Skipped in debug to keep link fast.
    if !matches!(opt, OptLevel::None) {
        match host.os {
            Os::MacOs => flags.push("-Wl,-dead_strip".to_string()),
            Os::Linux | Os::FreeBsd => flags.push("-Wl,--gc-sections".to_string()),
            // PE/COFF link.exe uses /OPT:REF; the safer minimum is to not
            // pass anything from here.
            _ => {}
        }
    }
    // `-C strip`: drop debug info / local symbols via linker flags. Mach-O:
    // `-Wl,-S` strips DWARF, `-Wl,-x` removes local symbols; ELF:
    // `--strip-debug` for debug only, `-s` for everything.
    if strip != Strip::None {
        match host.os {
            Os::MacOs => {
                flags.push("-Wl,-S".to_string());
                if strip == Strip::Symbols {
                    flags.push("-Wl,-x".to_string());
                }
            }
            Os::Linux | Os::FreeBsd => {
                flags.push(
                    if strip == Strip::Symbols {
                        "-Wl,-s"
                    } else {
                        "-Wl,--strip-debug"
                    }
                    .to_string(),
                );
            }
            _ => {}
        }
    }
    flags
}

/// `clang <obj> -o <out> <flags..>`. Does NOT remove the object. On any
/// failure — nonzero clang exit or a failed spawn — prints `Linking failed`
/// and returns 1.
fn run_clang_link(obj: &str, out: &str, flags: &[String]) -> i32 {
    let mut cmd = std::process::Command::new("clang");
    cmd.arg(obj).arg("-o").arg(out);
    for f in flags {
        cmd.arg(f);
    }
    match cmd.status() {
        Ok(s) if s.success() => 0,
        _ => {
            eprintln!("Linking failed");
            1
        }
    }
}

fn link_binary(
    cg: &CodegenContext,
    output: &str,
    libs: &[String],
    opt: OptLevel,
    lto: Lto,
    strip: Strip,
) -> i32 {
    let obj = match emit_object(cg, output, opt, lto) {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let host = Target::from_triple_str(&default_target_triple());
    let flags = link_flags(&host, libs, opt, lto, strip);
    let rc = run_clang_link(&obj, output, &flags);
    if rc != 0 {
        // On link failure the intermediate object is deliberately left in cwd.
        return rc;
    }
    let _ = std::fs::remove_file(&obj);
    0
}

/// FNV-1a over a file's bytes. Keys the linked-test-binary cache: macOS
/// assesses every fresh executable inode on first exec (60-290ms), so
/// re-running a byte-identical test binary from a cached inode instead of
/// relinking a new one is ~1ms vs ~100ms per test file.
fn hash_file_fnv(path: &str) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    let mut h: u64 = 1469598103934665603;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    Some(h)
}

fn hash_mix_string(h: &mut u64, s: &str) {
    for b in s.bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(1099511628211);
    }
}

/// Test-mode link + run with the cached-inode fast path: key the linked
/// binary by the object bytes + link flags, link once into
/// `output/testcache/t<hex>`, and re-exec the CACHED INODE when nothing
/// changed. The cached binary is deliberately NOT removed
/// after the run, and the link is skipped entirely on a cache hit. Falls back
/// to the ordinary link+run when the object can't be hashed.
fn link_and_run_test(
    cg: &CodegenContext,
    output: &str,
    libs: &[String],
    opt: OptLevel,
    lto: Lto,
    strip: Strip,
    progress: &mut ProgressGuard,
) -> i32 {
    let obj = match emit_object(cg, output, opt, lto) {
        Ok(o) => o,
        Err(rc) => return rc,
    };
    let host = Target::from_triple_str(&default_target_triple());
    let flags = link_flags(&host, libs, opt, lto, strip);

    if let Some(mut h) = hash_file_fnv(&obj) {
        for f in &flags {
            hash_mix_string(&mut h, f);
        }
        let cache_dir = std::path::Path::new("output").join("testcache");
        let _ = std::fs::create_dir_all(&cache_dir);
        let cache_path = cache_dir.join(format!("t{h:016x}"));
        let cache_path = cache_path.to_string_lossy().into_owned();
        if !std::path::Path::new(&cache_path).exists()
            && run_clang_link(&obj, &cache_path, &flags) != 0
        {
            // The cache path DOES clean up on failure, unlike the ordinary
            // link's leave-the-object-behind early return.
            let _ = std::fs::remove_file(&obj);
            return 1;
        }
        let _ = std::fs::remove_file(&obj);
        progress.stop();
        return run_binary_at(&cache_path, false);
    }

    let rc = run_clang_link(&obj, output, &flags);
    if rc != 0 {
        // Same as the ordinary link failure: intermediate left in cwd.
        return rc;
    }
    let _ = std::fs::remove_file(&obj);
    progress.stop();
    run_binary(output)
}

/// Synthesize the `jam test` harness `main`. For each `tfn` (in source order) it prints
/// `"testing <name>... "`, calls the test fn, then prints `"ok\n"`; after the
/// loop it prints `"<N> test(s) passed\n"` and returns 0. A failing `assert`
/// inside a test calls `exit(1)` before its `ok\n` ever prints — that IS the
/// fail path, no extra logic. `entries` = (display name, mangled symbol) pairs.
fn synthesize_test_main(cg: &CodegenContext, entries: &[(String, String)]) {
    let ctx = cg.context();
    let module = cg.module();
    let builder = cg.builder();

    let i32_ty = ctx.i32_type();
    // `main` is `i32()` with the C calling convention + external linkage.
    let main_ty = i32_ty.fn_type(&[], false);
    let main_func = module.add_function("main", main_ty);
    main_func.apply_default_attrs(false);
    main_func.set_linkage(Linkage::External);
    main_func.set_call_conv(CallConv::C);

    let entry_bb = main_func.append_basic_block("entry");
    builder.position_at_end(entry_bb);

    // Declare `printf` (`i32(ptr, ...)` varargs) if it isn't already present.
    let printf = match module.get_function("printf") {
        Some(f) => f,
        None => {
            let i8ptr = ctx.pointer_type(0);
            let printf_ty = i32_ty.fn_type(&[i8ptr], true);
            let f = module.add_function("printf", printf_ty);
            f.apply_default_attrs(true);
            f
        }
    };

    for (display, mangled) in entries {
        let Some(test_func) = module.get_function(mangled) else {
            // A test whose body failed to lower never got a prototype;
            // silently skip it.
            continue;
        };
        // "testing <name>... "
        let msg = format!("testing {display}... ");
        let msg_str = builder.global_string_ptr(&msg, "test_msg");
        builder.call(printf, &[msg_str], "");
        // Run the test (no args).
        builder.call(test_func, &[], "");
        // "ok\n" — reached only if the test didn't `exit(1)` via a failed assert.
        let ok_str = builder.global_string_ptr("ok\n", "test_pass");
        builder.call(printf, &[ok_str], "");
    }

    // "<N> test(s) passed\n"
    let summary = format!("{} test(s) passed\n", entries.len());
    let summary_str = builder.global_string_ptr(&summary, "test_summary");
    builder.call(printf, &[summary_str], "");

    builder.ret(i32_ty.const_int(0, false));
    let _ = main_func.verify();
}

/// Run the freshly-linked `output` binary, delete it, and propagate its exit
/// status as jam's own. A signal-killed child decodes to `128 + signal` (shell
/// convention) so a crashed test binary never silently reports success.
fn run_binary(output: &str) -> i32 {
    run_binary_at(output, true)
}

/// Run `output`; `remove_after=false` keeps the binary (the cached-inode test
/// path re-execs the same file on the next identical run).
fn run_binary_at(output: &str, remove_after: bool) -> i32 {
    let path = if output.starts_with('/') || output.starts_with("./") || output.contains('/') {
        output.to_string()
    } else {
        format!("./{output}")
    };
    let status = std::process::Command::new(&path).status();
    if remove_after {
        let _ = std::fs::remove_file(output);
    }
    match status {
        Ok(s) => {
            if let Some(code) = s.code() {
                code
            } else {
                // No exit code => terminated by a signal (Unix only).
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    s.signal().map(|sig| 128 + sig).unwrap_or(1)
                }
                #[cfg(not(unix))]
                {
                    1
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to run {path}: {e}");
            1
        }
    }
}
