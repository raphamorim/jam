/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam run` and `jam test` subcommands. Behaviorally identical to the C++
//! oracle (src/main.cpp): `run` compiles a file, runs it, propagates its exit
//! code, and deletes the temp binary; `test` compiles in test mode (synthesizing
//! a harness `main` over the file's `tfn`s), runs it, and propagates its output +
//! exit. `test <dir>` (or no arg → cwd) walks `*.jam` files, runs each one's
//! tests, and prints a `Summary:` line.

use std::path::Path;

use jam_llvm::{Lto, OptLevel};

use crate::cli::emit::{self, EmitMode, Strip};

/// Parse a `-C opt-level=N` value to an [`OptLevel`] (the C++ mapping,
/// src/main.cpp:2941-2947): `0`→None, `1`→Less, `2`→Default, `3`→Aggressive,
/// `s`→Size, `z`→Small. Returns `None` for an unrecognized value.
fn opt_level_from_value(value: &str) -> Option<OptLevel> {
    match value {
        "0" => Some(OptLevel::None),
        "1" => Some(OptLevel::Less),
        "2" => Some(OptLevel::Default),
        "3" => Some(OptLevel::Aggressive),
        "s" => Some(OptLevel::Size),
        "z" => Some(OptLevel::Small),
        _ => None,
    }
}

/// Pre-scan `argv` (skipping `argv[0]`) for a GLOBAL `-C opt-level=N` codegen
/// flag, which the C++ accepts ANYWHERE — before or after the subcommand
/// (`jam -C opt-level=3 test tests/unit` AND `jam test -C opt-level=3 …`). Both
/// the spaced `-C opt-level=3` and the joined `-Copt-level=3` forms are honored
/// (matching rustc / the C++ `-C` loop, src/main.cpp:2925-2953).
///
/// Returns the stripped argv (with the `-C` token(s) removed) and the parsed
/// [`OptLevel`] (default [`OptLevel::None`] when absent). On a bad `opt-level`
/// value it prints the C++'s error (src/main.cpp:2948-2952) and returns `Err`
/// with the process exit code.
pub fn extract_codegen_opts(args: &[String]) -> Result<(Vec<String>, OptLevel, Lto, Strip), i32> {
    let mut opt = OptLevel::None;
    let mut lto = Lto::Off;
    let mut strip = Strip::None;
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    out.push(args[0].clone());
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        // `--release` / `--release-small` were removed in favor of `-C opt-level`
        // (the C++ prints the same guidance, main.cpp:2906-2918).
        if arg == "--release" {
            eprintln!("Error: `--release` was removed; use `-C opt-level=3` instead");
            return Err(1);
        }
        if arg == "--release-small" {
            eprintln!("Error: `--release-small` was removed; use `-C opt-level=z` instead");
            return Err(1);
        }
        // `--std-path <dir>`: set the std-lib root (takes precedence over the
        // `JAM_STD_PATH` env, which the resolver reads — module_resolver.rs
        // std_root); stripped from argv so it isn't taken as the input file.
        if arg == "--std-path" && i + 1 < args.len() {
            // SAFETY: CLI startup is single-threaded; the resolver reads it later.
            unsafe { std::env::set_var("JAM_STD_PATH", &args[i + 1]) };
            i += 2;
            continue;
        }
        // `-C key=value` (spaced) or `-Ckey=value` (joined).
        let codegen: Option<String> = if arg == "-C" && i + 1 < args.len() {
            let v = args[i + 1].clone();
            i += 1; // consume the value token too
            Some(v)
        } else if arg.len() > 2 && arg.starts_with("-C") {
            Some(arg[2..].to_string())
        } else {
            None
        };
        if let Some(codegen) = codegen {
            let (key, value) = match codegen.split_once('=') {
                Some((k, v)) => (k, v),
                None => {
                    // The C++ dedicated malformed `-C` error (main.cpp:1966).
                    eprintln!("Error: -C expects key=value, got `{codegen}`");
                    return Err(1);
                }
            };
            // The C++ -C key mappings + error text (main.cpp:2941-2985).
            match key {
                "opt-level" => match opt_level_from_value(value) {
                    Some(l) => opt = l,
                    None => {
                        eprintln!("Error: -C opt-level expects one of 0|1|2|3|s|z, got `{value}`");
                        return Err(1);
                    }
                },
                "lto" => {
                    lto = match value {
                        "off" | "false" | "no" => Lto::Off,
                        "thin" => Lto::Thin,
                        "fat" | "full" | "true" | "yes" => Lto::Fat,
                        _ => {
                            eprintln!("Error: -C lto expects one of off|thin|fat, got `{value}`");
                            return Err(1);
                        }
                    }
                }
                "strip" => {
                    strip = match value {
                        "none" | "off" | "false" | "no" => Strip::None,
                        "debuginfo" => Strip::DebugInfo,
                        "symbols" => Strip::Symbols,
                        _ => {
                            eprintln!(
                                "Error: -C strip expects one of none|debuginfo|symbols, got `{value}`"
                            );
                            return Err(1);
                        }
                    }
                }
                _ => {
                    eprintln!("Error: unknown -C key `{key}` (supported: opt-level, lto, strip)");
                    return Err(1);
                }
            }
            i += 1;
            continue;
        }
        out.push(arg.clone());
        i += 1;
    }
    Ok((out, opt, lto, strip))
}

