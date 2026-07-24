/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

pub mod emit;
pub mod progress;
pub mod run;

pub fn display_help() {
    println!(
        r#"Jam programming language compiler

Usage: jam [OPTIONS] <file|directory>
       jam run [LINKER-FLAGS] <file>
       jam test [<file|directory>]

Subcommands:
  run             Compile, run, and clean up the executable.
                  Only linker flags (-l<name>) may accompany it.
  test            Compile test functions and run them.
  version         Print version and exit.
  help            Print help and exit.

Options:
  -o <name>       Output binary name (default: 'output')
  -C opt-level=N  Optimization level, Default is '0'
                    0  no optimizations
                    1  basic optimizations
                    2  LLVM default (-O2)
                    3  aggressive (-O3)
                    s  optimize for size (-Os)
                    z  aggressively optimize for size (-Oz)
  -C lto=MODE     Link-time optimization. Default is 'off'
                    off   regular object file (no LTO)
                    thin  ThinLTO bitcode — fast, parallel link
                    fat   full LTO bitcode — slowest link, most opt
  -C strip=MODE   Symbol / debug-info stripping. Default is 'none'
                    none       keep all symbols & debug info
                    debuginfo  strip DWARF / debug sections only
                    symbols    strip debug + local symbols
  -C target=T     Cross-compile for triple T (default: host).
                    x86_64-apple-darwin runs under Rosetta 2
  --emit-ir       Print LLVM IR to stdout
  --target-info   Show host target info (arch, triple, ...)
  --std-path <dir>
                  Override the standard-library root used to resolve
                  `import("std/...")`; takes precedence over the
                  JAM_STD_PATH env var.
  -l<name>, --library <name>
                  Link against system library <name>

Examples:

jam hello.jam                 # compile to './output'
jam run hello.jam             # compile and run
jam run -lncurses tetris.jam  # compile, link with ncurses, run
jam help                      # show help
jam version                   # show version
jam test                      # run tests in current working directory
jam test tests/unit           # run tests under tests/unit
jam test tests/unit/foo.jam   # run tests in a single file"#
    );
}
