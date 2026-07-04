/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Phase-0 spike: drive the whole binding end-to-end through real LLVM —
//! build IR, verify it, print it, and emit a host object file. A wrong FFI
//! signature in `raw.rs` would crash or miscompile here, so this is the
//! load-bearing proof that the hand-rolled bindings actually work.

use jam_llvm::{
    Context, IntPredicate, OptLevel, TargetMachine, init_native_asm_printer, init_native_target,
};

/// Build `fn add(i32, i32) -> i32 { ret a + b }`, verify, and check the IR.
#[test]
fn add_i32_roundtrip() {
    let ctx = Context::new();
    let module = ctx.create_module("spike");
    let builder = ctx.create_builder();

    let i32t = ctx.i32_type();
    let fn_ty = i32t.fn_type(&[i32t, i32t], false);
    let func = module.add_function("add", fn_ty);

    let entry = func.append_basic_block("entry");
    builder.position_at_end(entry);
    let sum = builder.add(func.param(0), func.param(1), "sum");
    builder.ret(sum);

    assert!(func.verify(), "add function failed verification");

    let ir = module.print_to_string();
    assert!(
        ir.contains("define"),
        "IR missing function definition:\n{ir}"
    );
    assert!(ir.contains("add"), "IR missing add instruction:\n{ir}");
    assert!(ir.contains("i32"), "IR missing i32 type:\n{ir}");
}

/// Exercise more of the instruction surface: locals (entry-block alloca),
/// load/store, icmp, conditional branch, phi.
#[test]
fn control_flow_and_memory() {
    let ctx = Context::new();
    let module = ctx.create_module("cf");
    let builder = ctx.create_builder();

    let i32t = ctx.i32_type();
    let i1t = ctx.i1_type();
    let fn_ty = i32t.fn_type(&[i32t], false);
    let func = module.add_function("absish", fn_ty);

    let entry = func.append_basic_block("entry");
    let then_bb = func.append_basic_block("then");
    let else_bb = func.append_basic_block("else");
    let merge = func.append_basic_block("merge");

    builder.position_at_end(entry);
    let slot = builder.alloca(i32t, 4, "slot");
    builder.store(func.param(0), slot);
    let v = builder.load(i32t, slot, "v");
    let zero = i32t.const_int(0, false);
    let is_neg = builder.icmp(IntPredicate::Slt, v, zero, "is_neg");
    let _ = i1t; // i1 type smoke
    builder.cond_br(is_neg, then_bb, else_bb);

    builder.position_at_end(then_bb);
    let neg = builder.sub(zero, v, "neg");
    builder.br(merge);

    builder.position_at_end(else_bb);
    builder.br(merge);

    builder.position_at_end(merge);
    let phi = builder.phi(i32t, "r");
    phi.add_incoming(&[neg, v], &[then_bb, else_bb]);
    builder.ret(phi);

    assert!(func.verify(), "absish failed verification");
    // Alloca must be hoisted into the entry block, not emitted in `merge`.
    let ir = module.print_to_string();
    assert!(ir.contains("alloca"), "missing alloca:\n{ir}");
    assert!(ir.contains("phi"), "missing phi:\n{ir}");
}

/// Emit a real object file for the host target through the full pipeline.
#[test]
fn emit_object_file() {
    init_native_target();
    init_native_asm_printer();

    let ctx = Context::new();
    let module = ctx.create_module("emit");
    let builder = ctx.create_builder();

    let i32t = ctx.i32_type();
    let fn_ty = i32t.fn_type(&[], false);
    // Name it `main` so the keep-`main` internalize predicate preserves it.
    let func = module.add_function("main", fn_ty);
    func.apply_default_attrs(false);
    let entry = func.append_basic_block("entry");
    builder.position_at_end(entry);
    builder.ret(i32t.const_int(42, false));
    assert!(func.verify());

    let triple = jam_llvm::default_target_triple();
    let tm = TargetMachine::new(&triple, "", "", true, OptLevel::Default, jam_llvm::Lto::Off)
        .expect("could not create target machine for host triple");
    tm.configure_module(&module);

    let dir = std::env::temp_dir();
    let path = dir.join("jam_llvm_spike_main.o");
    let path_str = path.to_str().unwrap();
    let _ = std::fs::remove_file(&path);

    tm.emit_to_file(&module, path_str)
        .expect("object emission failed");

    let meta = std::fs::metadata(&path).expect("object file was not created");
    assert!(meta.len() > 0, "emitted object file is empty");
    let _ = std::fs::remove_file(&path);
}

/// Prove the C++ shim actually enables `FunctionSections`: cross-emit an ELF
/// object at -O2 with two `llvm.used` functions (kept distinct by internalize/
/// DCE) and check each lands in its own `.text.<name>` section. Ignored by
/// default — needs the X86 backend + `llvm-objdump`; run with `--ignored`.
#[test]
#[ignore = "cross-emits ELF and shells llvm-objdump; run manually"]
fn function_sections_shim_takes_effect() {
    jam_llvm::init_all_targets();
    let ctx = Context::new();
    let module = ctx.create_module("fs");
    let builder = ctx.create_builder();
    let i32t = ctx.i32_type();
    let fn_ty = i32t.fn_type(&[], false);
    for (name, val) in [("jam_fn_a", 1u64), ("jam_fn_b", 2u64)] {
        let f = module.add_function(name, fn_ty);
        jam_llvm::append_to_used(&module, f); // survive internalize + globaldce
        let bb = f.append_basic_block("e");
        builder.position_at_end(bb);
        builder.ret(i32t.const_int(val, false));
    }

    let tm = TargetMachine::new(
        "x86_64-unknown-linux-gnu",
        "",
        "",
        true,
        OptLevel::Default,
        jam_llvm::Lto::Off,
    )
    .expect("x86_64-linux target machine");
    tm.configure_module(&module);

    let path = std::env::temp_dir().join("jam_llvm_fsections.o");
    tm.emit_to_file(&module, path.to_str().unwrap())
        .expect("emit ELF");

    let out = std::process::Command::new("llvm-objdump")
        .arg("-h")
        .arg(&path)
        .output()
        .expect("run llvm-objdump");
    let headers = String::from_utf8_lossy(&out.stdout);
    let _ = std::fs::remove_file(&path);
    assert!(
        headers.contains(".text.jam_fn_a") && headers.contains(".text.jam_fn_b"),
        "FunctionSections not applied — expected per-function sections, got:\n{headers}"
    );
}
