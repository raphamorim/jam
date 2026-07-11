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

/// Strict `run` argument parsing (the C++ run branch, main.cpp:2024-2038):
/// only linker flags are permitted alongside `run`, and only ONE source file.
fn parse_run_args(args: &[String]) -> Result<(Vec<String>, Option<String>), i32> {
    let mut libs: Vec<String> = Vec::new();
    let mut path: Option<String> = None;
    let mut i = 2; // skip program name + subcommand
    while i < args.len() {
        let arg = &args[i];
        if (arg == "-l" || arg == "--library") && i + 1 < args.len() {
            libs.push(args[i + 1].clone());
            i += 2;
            continue;
        }
        if arg.len() > 2 && arg.starts_with("-l") {
            libs.push(arg[2..].to_string());
            i += 1;
            continue;
        }
        if arg.starts_with('-') {
            eprintln!(
                "Error: `run` only accepts linker flags (-l<name>, -l <name>, --library <name>); got `{arg}`"
            );
            return Err(1);
        }
        if let Some(prev) = &path {
            eprintln!("Error: `run` accepts only one source file; got `{arg}` after `{prev}`");
            return Err(1);
        }
        path = Some(arg.clone());
        i += 1;
    }
    Ok((libs, path))
}

/// `jam run [LINKER-FLAGS] <file>`: compile `<file>` to a temp executable, run
/// it, propagate its exit code, and delete the binary.
pub fn run_command(args: &[String], opt: OptLevel, lto: Lto, strip: Strip) -> i32 {
    let (libs, path) = match parse_run_args(args) {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let Some(file) = path else {
        eprintln!("Error: No input file specified. Run `jam --help` for usage.");
        return 1;
    };
    // Directory input is only meaningful with `test` (main.cpp:2112-2118).
    if Path::new(&file).is_dir() {
        eprintln!("Error: directory input is only supported with `test` (got '{file}')");
        return 1;
    }
    let output = pick_output_name("output");
    emit::emit_jir(&file, EmitMode::Run { output, libs }, opt, lto, strip)
}

/// `jam test [<file|directory>]`: with no arg, test the current directory.
pub fn test_command(args: &[String], opt: OptLevel, lto: Lto, strip: Strip) -> i32 {
    let (libs, output, path) = match parse_compile_args(args, 2) {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    // `jam test` with no path means "run every test under cwd" (main.cpp:3060).
    let target = path.unwrap_or_else(|| ".".to_string());

    if Path::new(&target).is_dir() {
        return test_directory(&target, &libs, opt, lto, strip);
    }

    let output = output.unwrap_or_else(|| pick_output_name("output"));
    emit::emit_jir(&target, EmitMode::Test { output, libs }, opt, lto, strip)
}

/// The default output name `output` collides with a same-named directory (this
/// repo has one): clang's link fails cryptically. Fall back to `<name>.bin`
/// (main.cpp:3245-3253). The note goes to stderr (the behavioral gate drops it).
fn pick_output_name(name: &str) -> String {
    if Path::new(name).is_dir() {
        let fallback = format!("{name}.bin");
        eprintln!("note: output name '{name}' is a directory; writing '{fallback}'");
        fallback
    } else {
        name.to_string()
    }
}

/// Parse `-l<name>` libs, `-o <name>` output, and the positional `<file>` from a
/// compile/test invocation. `start` is the first arg index to consider: 2 after
/// the `build`/`test` subcommand token, 1 for the bare `jam [FLAGS] <file>`
/// form. Like the C++ loop (main.cpp:2059-2073), any unmatched argument —
/// flag-shaped or not — is a positional, and a SECOND positional is an error.
fn parse_compile_args(
    args: &[String],
    start: usize,
) -> Result<(Vec<String>, Option<String>, Option<String>), i32> {
    let mut libs: Vec<String> = Vec::new();
    let mut output: Option<String> = None;
    let mut path: Option<String> = None;
    let mut i = start;
    while i < args.len() {
        let arg = &args[i];
        if (arg == "-l" || arg == "--library") && i + 1 < args.len() {
            libs.push(args[i + 1].clone());
            i += 2;
            continue;
        }
        if arg.len() > 2 && arg.starts_with("-l") {
            libs.push(arg[2..].to_string());
            i += 1;
            continue;
        }
        if arg == "-o" && i + 1 < args.len() {
            output = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(prev) = &path {
            eprintln!("Error: unexpected extra argument `{arg}` (already have `{prev}`)");
            return Err(1);
        }
        path = Some(arg.clone());
        i += 1;
    }
    Ok((libs, output, path))
}

/// `jam build [LINKER-FLAGS] [-o <name>] <file>` and the bare `jam [FLAGS] <file>`
/// form: compile `<file>` to a native executable (default name `output`, with the
/// directory-collision `.bin` fallback), linking `-l<name>` libraries — matching
/// the C++'s default-compile path. `start` is 2 for the `build` subcommand, 1 for
/// the bare form (no subcommand token).
pub fn build_command(args: &[String], start: usize, opt: OptLevel, lto: Lto, strip: Strip) -> i32 {
    let (libs, output, path) = match parse_compile_args(args, start) {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let Some(file) = path else {
        eprintln!("Error: No input file specified. Run `jam --help` for usage.");
        return 1;
    };
    // Directory input is only meaningful with `test` (main.cpp:2112-2118).
    if Path::new(&file).is_dir() {
        eprintln!("Error: directory input is only supported with `test` (got '{file}')");
        return 1;
    }
    let output = output.unwrap_or_else(|| pick_output_name("output"));
    emit::emit_jir(&file, EmitMode::Binary { output, libs }, opt, lto, strip)
}

/// Recursively discover `*.jam` files under `dir`, sorted (main.cpp's
/// `collectJamFiles`). Returns paths as strings.
fn collect_jam_files(dir: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    collect_jam_files_into(Path::new(dir), &mut files);
    files.sort();
    files
}

fn collect_jam_files_into(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_jam_files_into(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("jam")
            && let Some(s) = p.to_str()
        {
            out.push(s.to_string());
        }
    }
}

/// Cheap `tfn` substring scan — avoids lowering files with no test functions
/// (main.cpp's `fileHasTests`). Matches `tfn` at line start or after whitespace,
/// followed by whitespace or `(`.
fn file_has_tests(path: &str) -> bool {
    let Ok(src) = std::fs::read(path) else {
        return false;
    };
    let mut pos = 0usize;
    while let Some(rel) = find_subslice(&src[pos..], b"tfn") {
        let idx = pos + rel;
        let start_ok = idx == 0 || src[idx - 1].is_ascii_whitespace();
        let after = idx + 3;
        let end_ok = after < src.len() && (src[after].is_ascii_whitespace() || src[after] == b'(');
        if start_ok && end_ok {
            return true;
        }
        pos = idx + 3;
    }
    false
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

