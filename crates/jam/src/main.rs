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

    if args.len() < 2 {
        // Help to stdout, but exit 1 — an argument-less invocation is a usage error.
        cli::display_help();
        std::process::exit(1);
    }
    match args[1].as_str() {
        // `jam build [FLAGS] <file>`: same driver loop, with `run`/`test`
        // demoted to ordinary positionals.
        "build" => std::process::exit(cli::run::driver(&args, 2, false)),
        "version" => {
            // `<base>-<git-short-sha>[-dirty]`; the SHA is baked at build time
            // by build.rs.
            println!("{}-{}", env!("CARGO_PKG_VERSION"), env!("JAM_VERSION_SHA"));
        }
        "help" => {
            cli::display_help();
        }
        // Everything else — `run`/`test` (recognized anywhere in argv), flags,
        // or the bare `jam [FLAGS] <file>` compile form.
        _ => std::process::exit(cli::run::driver(&args, 1, true)),
    }
}
