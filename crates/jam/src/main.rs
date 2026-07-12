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
        // Help to stdout, but exit 1 — an argument-less invocation is a usage
        // error in the C++ (main.cpp:1908-1911).
        cli::display_help();
        std::process::exit(1);
    }
    match args[1].as_str() {
        // `jam build [FLAGS] <file>`: the bare compile form behind an explicit
        // subcommand token — same driver loop, with `run`/`test` demoted to
        // ordinary positionals.
        "build" => std::process::exit(cli::run::driver(&args, 2, false)),
        "version" => {
            // `<base>-<git-short-sha>[-dirty]`, matching the C++ printVersion
            // (main.cpp) / the Makefile's JAM_VERSION_SHA. The SHA is baked at
            // build time by build.rs.
            println!("{}-{}", env!("CARGO_PKG_VERSION"), env!("JAM_VERSION_SHA"));
        }
        "help" => {
            cli::display_help();
        }
        // Everything else — `run`/`test` (which the driver recognizes anywhere
        // in argv, like the C++), flags, or the bare `jam [FLAGS] <file>`
        // compile form the jamstation Makefile uses.
        _ => std::process::exit(cli::run::driver(&args, 1, true)),
    }
}
