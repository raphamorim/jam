/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! stdout/stderr separation for the cfn print pipeline: `fmt.print` must land
//! on fd 1 and `fmt.eprint` on fd 2. The @-emit dispatch passes the fd as a
//! comp-known literal, so a wiring mistake would silently swap streams — only
//! capturing the two streams separately can catch it.

use std::process::Command;

#[test]
fn print_and_eprint_reach_their_own_streams() {
    let jam = env!("CARGO_BIN_EXE_jam");
    let repo = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let dir = std::env::temp_dir().join(format!("jam_print_streams_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("streams.jam");
    std::fs::write(
        &src,
        "const fmt = import(\"std\").fmt;\n\
         \n\
         fn main() i32 {\n\
             fmt.print(\"out-line\\n\");\n\
             fmt.eprint(\"err-line\\n\");\n\
             fmt.print(\"out-again\\n\");\n\
             return 0;\n\
         }\n",
    )
    .unwrap();
    let bin = dir.join("streams_bin");

    let compile = Command::new(jam)
        .arg("build")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .env("JAM_STD_PATH", format!("{repo}/std"))
        .output()
        .expect("spawn jam");
    assert!(
        compile.status.success(),
        "compile failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin).output().expect("run compiled binary");
    let out = String::from_utf8_lossy(&run.stdout);
    let err = String::from_utf8_lossy(&run.stderr);
    assert_eq!(out, "out-line\nout-again\n", "stdout carries only print");
    assert_eq!(err, "err-line\n", "stderr carries only eprint");
    assert!(run.status.success());

    let _ = std::fs::remove_dir_all(&dir);
}
