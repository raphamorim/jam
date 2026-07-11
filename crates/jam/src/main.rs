/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

mod cli;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Extract the global `-C opt-level=N` codegen flag (accepted anywhere in
    // argv, like the C++) up front, leaving clean args for the subcommand
    // parsers and carrying the level into every codegen path.
    let (args, opt, lto, strip) = match cli::run::extract_codegen_opts(&args) {
        Ok(v) => v,
        Err(code) => std::process::exit(code),
    };

    // Flag-style invocations driven by the differential harness, e.g.
    // `jam --emit-tokens <file>`. Handled before the subcommand dispatch.
    if args.iter().any(|a| a == "--emit-tokens") {
        match args.iter().skip(1).find(|a| !a.starts_with('-')) {
            Some(file) => std::process::exit(cli::emit::emit_tokens(file)),
            None => {
                eprintln!("Error: --emit-tokens requires an input file");
                std::process::exit(1);
            }
        }
    }
    if args.iter().any(|a| a == "--emit-ast") {
        match args.iter().skip(1).find(|a| !a.starts_with('-')) {
            Some(file) => std::process::exit(cli::emit::emit_ast(file)),
            None => {
                eprintln!("Error: --emit-ast requires an input file");
                std::process::exit(1);
            }
        }
    }
    if args.iter().any(|a| a == "--emit-jir") {
        match args.iter().skip(1).find(|a| !a.starts_with('-')) {
            Some(file) => std::process::exit(cli::emit::emit_jir(
                file,
                cli::emit::EmitMode::Jir,
                opt,
                lto,
                strip,
            )),
            None => {
                eprintln!("Error: --emit-jir requires an input file");
                std::process::exit(1);
            }
        }
    }
    if args.iter().any(|a| a == "--emit-ir") {
        match args.iter().skip(1).find(|a| !a.starts_with('-')) {
            Some(file) => std::process::exit(cli::emit::emit_jir(
                file,
                cli::emit::EmitMode::Ir,
                opt,
                lto,
                strip,
            )),
            None => {
                eprintln!("Error: --emit-ir requires an input file");
                std::process::exit(1);
            }
        }
    }
    // `--help`/`-h` and `--version`/`-V` are honored anywhere in argv (the C++
    // flag loop, main.cpp:2041-2048) — EXCEPT under `run`, whose strict
    // validation rejects every non-linker flag (main.cpp:2024-2031); the C++
    // loop hits the run check first, so `jam run --help` is an error there too.
    let run_pos = args.iter().position(|a| a == "run");
    let before_run = |p: Option<usize>| match (p, run_pos) {
        (Some(fp), Some(rp)) => fp < rp,
        (Some(_), None) => true,
        (None, _) => false,
    };
    if before_run(args.iter().position(|a| a == "--help" || a == "-h")) {
        cli::display_help();
        std::process::exit(0);
    }
    if before_run(args.iter().position(|a| a == "--version" || a == "-V")) {
        println!("{}-{}", env!("CARGO_PKG_VERSION"), env!("JAM_VERSION_SHA"));
        std::process::exit(0);
    }

    // `--target-info`: print the host target's properties, then CONTINUE into
    // the normal compile (the C++ `showTarget` flag, main.cpp:2089-2105 — it
    // still requires an input file, and compiles it after the printout).
    let mut args = args;
    if let Some(p) = args.iter().position(|a| a == "--target-info") {
        args.remove(p);
        // The C++ checks `filename.empty()` BEFORE the target block, so a bare
        // `jam --target-info` errors without printing (main.cpp:2079-2083).
        let has_positional = args
            .iter()
            .skip(1)
            .any(|a| !a.starts_with('-') && a != "run" && a != "test" && a != "build");
        if !has_positional {
            eprintln!("Error: No input file specified. Run `jam --help` for usage.");
            std::process::exit(1);
        }
        cli::emit::print_target_info();
    }

    if args.len() < 2 {
        // Help to stdout, but exit 1 — an argument-less invocation is a usage
        // error in the C++ (main.cpp:1908-1911).
        cli::display_help();
        std::process::exit(1);
    }
    let command = &args[1];
    match command.as_str() {
        // `jam build [-lLIB] [-o NAME] <file>`: compile to a native binary (linking
        // libraries). args index 2 skips the `build` token.
        "build" => std::process::exit(cli::run::build_command(&args, 2, opt, lto, strip)),
        "run" => {
            std::process::exit(cli::run::run_command(&args, opt, lto, strip));
        }
        "test" => {
            std::process::exit(cli::run::test_command(&args, opt, lto, strip));
        }
        "version" => {
            // `<base>-<git-short-sha>[-dirty]`, matching the C++ printVersion
            // (main.cpp) / the Makefile's JAM_VERSION_SHA. The SHA is baked at
            // build time by build.rs.
            println!("{}-{}", env!("CARGO_PKG_VERSION"), env!("JAM_VERSION_SHA"));
        }
        "help" => {
            cli::display_help();
        }
        // No subcommand: treat as the bare `jam [FLAGS] <file>` compile form (the
        // C++ default-compile path the jamstation Makefile uses). args index 1
        // includes the first token (a flag or the file). With no input file,
        // build_command prints the usage hint.
        _ => std::process::exit(cli::run::build_command(&args, 1, opt, lto, strip)),
    }
}
