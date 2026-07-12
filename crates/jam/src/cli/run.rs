/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The compile driver: one single-pass argument loop transliterated from the
//! C++ oracle (src/main.cpp:1904-2286), so flag interplay — position relative
//! to `run`, values that look like flags, trailing options — resolves
//! identically. `run` compiles a file, runs it, propagates its exit code, and
//! deletes the temp binary; `test` compiles in test mode (synthesizing a
//! harness `main` over the file's `tfn`s), runs it, and propagates its output +
//! exit. `test <dir>` (or no arg → cwd) walks `*.jam` files, runs each one's
//! tests, and prints a `Summary:` line. The `build` subcommand reuses the same
//! loop with `run`/`test` demoted to ordinary positionals.

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

/// Parse one `-C key=value` payload into the opt/lto/strip slots — the C++ key
/// mappings + error texts (main.cpp:1965-2020). Returns the process exit code
/// on a malformed or unknown option.
fn parse_codegen_opt(
    codegen: &str,
    opt: &mut OptLevel,
    lto: &mut Lto,
    strip: &mut Strip,
) -> Result<(), i32> {
    let (key, value) = match codegen.split_once('=') {
        Some((k, v)) => (k, v),
        None => {
            eprintln!("Error: -C expects key=value, got `{codegen}`");
            return Err(1);
        }
    };
    match key {
        "opt-level" => match opt_level_from_value(value) {
            Some(l) => *opt = l,
            None => {
                eprintln!("Error: -C opt-level expects one of 0|1|2|3|s|z, got `{value}`");
                return Err(1);
            }
        },
        "lto" => {
            *lto = match value {
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
            *strip = match value {
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
    Ok(())
}

/// Lexer/AST/JIR text dumps — flags the differential harness drives. They slot
/// beside `--emit-ir` in the driver loop so they obey the same position rules.
enum OracleEmit {
    Tokens,
    Ast,
    Jir,
}

/// The compile driver: the C++ argument loop (main.cpp:1917-2074) plus its
/// post-parse sequence (main.cpp:2076-2286), transliterated token for token so
/// every interplay — flag position relative to `run`, option values that look
/// like flags, trailing options falling through as positionals — resolves
/// identically.
///
/// `start` is the first argv index to consider: 1 for the bare
/// `jam [FLAGS] <file>` form (where `run`/`test` toggle modes anywhere in
/// argv, like the C++), 2 under the `build` subcommand. `mode_tokens` is
/// false under `build`, demoting `run`/`test` to ordinary positionals.
pub fn driver(args: &[String], start: usize, mode_tokens: bool) -> i32 {
    let mut run_flag = false;
    let mut test_mode = false;
    let mut emit_ir = false;
    let mut oracle: Option<OracleEmit> = None;
    let mut show_target = false;
    let mut opt = OptLevel::None;
    let mut lto = Lto::Off;
    let mut strip = Strip::None;
    let mut filename = String::new();
    let mut output_name = String::from("output");
    let mut std_path_override = String::new();
    let mut libs: Vec<String> = Vec::new();

    let mut i = start;
    while i < args.len() {
        let arg = &args[i];
        // `run`/`test` mode tokens — recognized anywhere in argv
        // (main.cpp:1919-1926).
        if mode_tokens && arg == "run" {
            run_flag = true;
            i += 1;
            continue;
        }
        if mode_tokens && arg == "test" {
            test_mode = true;
            i += 1;
            continue;
        }
        // Linker flags — accepted in every mode (main.cpp:1928-1935). The
        // spaced forms consume the next token unconditionally, so a value that
        // looks like a flag (`-l --release`) is still a library name.
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
        // `--release` / `--release-small` were removed in favor of `-C
        // opt-level` (main.cpp:1939-1950).
        if arg == "--release" {
            eprintln!("Error: `--release` was removed; use `-C opt-level=3` instead");
            return 1;
        }
        if arg == "--release-small" {
            eprintln!("Error: `--release-small` was removed; use `-C opt-level=z` instead");
            return 1;
        }
        // `-C key=value` (spaced) or `-Ckey=value` (joined) codegen options
        // (main.cpp:1958-2023). Handled BEFORE the `run` strictness check, so
        // `jam run -C opt-level=3 file` is accepted — as in the C++.
        let codegen: Option<String> = if arg == "-C" && i + 1 < args.len() {
            i += 1;
            Some(args[i].clone())
        } else if arg.len() > 2 && arg.starts_with("-C") {
            Some(arg[2..].to_string())
        } else {
            None
        };
        if let Some(codegen) = codegen {
            if let Err(rc) = parse_codegen_opt(&codegen, &mut opt, &mut lto, &mut strip) {
                return rc;
            }
            i += 1;
            continue;
        }
        // Inside `run`, anything else flag-shaped is an error
        // (main.cpp:2024-2038) — including the flags parsed below, which are
        // honored only BEFORE the `run` token.
        if run_flag {
            if arg.starts_with('-') {
                eprintln!(
                    "Error: `run` only accepts linker flags (-l<name>, -l <name>, --library <name>); got `{arg}`"
                );
                return 1;
            }
            if !filename.is_empty() {
                eprintln!(
                    "Error: `run` accepts only one source file; got `{arg}` after `{filename}`"
                );
                return 1;
            }
            filename = arg.clone();
            i += 1;
            continue;
        }
        // Compile-only / test-mode flags (main.cpp:2040-2064).
        if arg == "--help" || arg == "-h" {
            crate::cli::display_help();
            return 0;
        }
        if arg == "--version" || arg == "-V" {
            println!("{}-{}", env!("CARGO_PKG_VERSION"), env!("JAM_VERSION_SHA"));
            return 0;
        }
        if arg == "--target-info" {
            show_target = true;
            i += 1;
            continue;
        }
        if arg == "--emit-ir" {
            emit_ir = true;
            i += 1;
            continue;
        }
        if arg == "--emit-tokens" {
            oracle = Some(OracleEmit::Tokens);
            i += 1;
            continue;
        }
        if arg == "--emit-ast" {
            oracle = Some(OracleEmit::Ast);
            i += 1;
            continue;
        }
        if arg == "--emit-jir" {
            oracle = Some(OracleEmit::Jir);
            i += 1;
            continue;
        }
        if arg == "-o" && i + 1 < args.len() {
            output_name = args[i + 1].clone();
            i += 2;
            continue;
        }
        if arg == "--std-path" && i + 1 < args.len() {
            std_path_override = args[i + 1].clone();
            i += 2;
            continue;
        }
        // Positional: source file or directory. Flags may follow it; a second
        // positional is an error (main.cpp:2065-2073). A trailing option with
        // no value (`jam file -o`) falls through here, like the C++.
        if !filename.is_empty() {
            eprintln!("Error: unexpected extra argument `{arg}` (already have `{filename}`)");
            return 1;
        }
        filename = arg.clone();
        i += 1;
    }

    // `jam test` with no path means "run every test under cwd" (main.cpp:2077).
    if test_mode && filename.is_empty() {
        filename = ".".to_string();
    }
    if filename.is_empty() {
        eprintln!(
            "Error: No input file specified. Run `{} --help` for usage.",
            args[0]
        );
        return 1;
    }
    if !std_path_override.is_empty() {
        // Takes precedence over a caller-set JAM_STD_PATH env var, which the
        // module resolver reads (module_resolver.rs std_root).
        // SAFETY: CLI startup is single-threaded; the resolver reads it later.
        unsafe { std::env::set_var("JAM_STD_PATH", &std_path_override) };
    }
    // `--target-info` prints the host target block, then CONTINUES into the
    // normal compile (main.cpp:2089-2103). It still requires an input file —
    // the emptiness check above runs first, as in the C++.
    if show_target {
        emit::print_target_info();
    }
    // Directory input: only meaningful with `test` (main.cpp:2110-2118).
    if Path::new(&filename).is_dir() {
        if !test_mode {
            eprintln!("Error: directory input is only supported with `test` (got '{filename}')");
            return 1;
        }
        return test_directory(&filename, &libs, emit_ir, opt, lto, strip);
    }
    // The default output name `output` collides with a same-named directory
    // (this repo has one): the link fails with a cryptic `ld: errno=21`. Fall
    // back to a usable name instead (main.cpp:2265-2274). Applies to an
    // explicit `-o` name too, and the note prints even under `--emit-ir`,
    // which resolves the name it will never write — as in the C++.
    if Path::new(&output_name).is_dir() {
        let fallback = format!("{output_name}.bin");
        eprintln!("note: output name '{output_name}' is a directory; writing '{fallback}'");
        output_name = fallback;
    }
    match oracle {
        Some(OracleEmit::Tokens) => return emit::emit_tokens(&filename),
        Some(OracleEmit::Ast) => return emit::emit_ast(&filename),
        Some(OracleEmit::Jir) => {
            return emit::emit_jir(&filename, EmitMode::Jir, opt, lto, strip);
        }
        None => {}
    }
    let mode = if emit_ir {
        // Print-IR-and-exit wins over run/test execution (the early return
        // inside compileAndRun, main.cpp:1547-1553); under `test` the dump
        // includes the synthesized harness main.
        EmitMode::Ir { test: test_mode }
    } else if test_mode {
        EmitMode::Test {
            output: output_name,
            libs,
        }
    } else if run_flag {
        EmitMode::Run {
            output: output_name,
            libs,
        }
    } else {
        EmitMode::Binary {
            output: output_name,
            libs,
        }
    };
    emit::emit_jir(&filename, mode, opt, lto, strip)
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

/// Worker-pool size for `jam test <dir>` (the C++ main.cpp:2143-2148):
/// min(cpu count, 8), overridable via `JAM_TEST_JOBS` (>= 1). `JAM_TEST_JOBS=1`
/// forces the serial in-process path (useful under a debugger).
fn test_jobs() -> usize {
    let hw = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut jobs = if hw > 1 { hw.min(8) } else { 1 };
    if let Ok(env) = std::env::var("JAM_TEST_JOBS")
        && let Ok(v) = env.trim().parse::<i64>()
        && v >= 1
    {
        jobs = v as usize;
    }
    jobs
}

/// The `-C opt-level=` value spelling for a re-invoked worker (main.cpp:2150).
fn opt_name(opt: OptLevel) -> &'static str {
    match opt {
        OptLevel::Less => "1",
        OptLevel::Default => "2",
        OptLevel::Aggressive => "3",
        OptLevel::Size => "s",
        OptLevel::Small => "z",
        _ => "0",
    }
}

/// A `jam test <file>` re-invocation of this binary with stdout+stderr merged
/// into one buffered pipe (the C++ spawnAsync, main.cpp:113): the drain thread
/// keeps the pipe from filling while the parent polls for completion.
struct TestWorker {
    child: std::process::Child,
    file: String,
    drain: std::thread::JoinHandle<Vec<u8>>,
}

fn spawn_test_worker(exe: &Path, file: &str, libs: &[String], opt: OptLevel) -> Option<TestWorker> {
    let stem = Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");
    let (reader, writer) = std::io::pipe().ok()?;
    let writer2 = writer.try_clone().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("test")
        .arg(file)
        .arg("-o")
        .arg(format!("jam_test_{stem}"));
    // Only the opt level is forwarded (the C++ childArgv, main.cpp:2178-2183).
    if !matches!(opt, OptLevel::None) {
        cmd.arg("-C").arg(format!("opt-level={}", opt_name(opt)));
    }
    for lib in libs {
        cmd.arg(format!("-l{lib}"));
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::from(writer));
    cmd.stderr(std::process::Stdio::from(writer2));
    let child = cmd.spawn().ok()?;
    let drain = std::thread::spawn(move || {
        use std::io::Read;
        let mut reader = reader;
        let mut buf = Vec::new();
        let _ = reader.read_to_end(&mut buf);
        buf
    });
    Some(TestWorker {
        child,
        file: file.to_string(),
        drain,
    })
}

/// Decode a finished worker's status the shell way: exit code, or 128 + signal.
fn worker_exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map(|s| 128 + s).unwrap_or(1)
    }
    #[cfg(not(unix))]
    {
        1
    }
}

/// The parallel worker pool (the C++ main.cpp:2165-2242): per-file work is
/// ~98% blocked subprocess waits (clang link + first-exec assessment of the
/// freshly linked test binary), so a small pool of `jam test <file>`
/// re-invocations gives near-linear speedup. Each worker's output prints as a
/// coherent block on completion, in completion order.
fn run_tests_parallel(
    runnable: &[String],
    libs: &[String],
    opt: OptLevel,
    jobs: usize,
    passed: &mut i32,
    failed: &mut i32,
) {
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("jam"));
    let mut active: Vec<TestWorker> = Vec::new();
    let mut next = 0usize;
    while next < runnable.len() || !active.is_empty() {
        while active.len() < jobs && next < runnable.len() {
            let f = &runnable[next];
            next += 1;
            match spawn_test_worker(&exe, f, libs, opt) {
                Some(w) => active.push(w),
                None => {
                    println!();
                    println!("@{f}");
                    println!("error: failed to spawn test worker");
                    *failed += 1;
                }
            }
        }
        let mut reaped = false;
        let mut i = 0;
        while i < active.len() {
            match active[i].child.try_wait() {
                Ok(Some(status)) => {
                    let w = active.remove(i);
                    let out = w.drain.join().unwrap_or_default();
                    println!();
                    println!("@{}", w.file);
                    print!("{}", String::from_utf8_lossy(&out));
                    if worker_exit_code(status) != 0 {
                        *failed += 1;
                    } else {
                        *passed += 1;
                    }
                    reaped = true;
                }
                Ok(None) => i += 1,
                Err(_) => {
                    let w = active.remove(i);
                    let _ = w.drain.join();
                    println!();
                    println!("@{}", w.file);
                    println!("error: failed to wait on test worker");
                    *failed += 1;
                    reaped = true;
                }
            }
        }
        if !reaped && !active.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

/// `jam test <dir>`: run each file's tests, tracking passed/failed/skipped,
/// then print the `Summary:` line (main.cpp:3094-3243). Files run through the
/// parallel worker pool when more than one job is available; `JAM_TEST_JOBS=1`
/// (or a single runnable file) takes the serial in-process path, whose sorted,
/// deterministic output is what a differential comparison can rely on.
fn test_directory(
    dir: &str,
    libs: &[String],
    emit_ir: bool,
    opt: OptLevel,
    lto: Lto,
    strip: Strip,
) -> i32 {
    let files = collect_jam_files(dir);
    if files.is_empty() {
        println!("No .jam files found under {dir}");
        return 0;
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut runnable: Vec<String> = Vec::new();
    for f in &files {
        if file_has_tests(f) {
            runnable.push(f.clone());
        } else {
            skipped += 1;
        }
    }

    let jobs = test_jobs();
    if jobs > 1 && runnable.len() > 1 {
        run_tests_parallel(&runnable, libs, opt, jobs, &mut passed, &mut failed);
    } else {
        for f in &runnable {
            // Per-file block header: a leading blank line then `@<file>`.
            println!();
            println!("@{f}");
            // Flush so the header precedes the test binary's own stdout.
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let stem = Path::new(f)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("out");
            let output = format!("jam_test_{stem}");
            // The serial walker threads `--emit-ir` into each per-file compile
            // (the C++ passes emitIR through, main.cpp:2249-2251); the parallel
            // pool does not forward it, also as in the C++ (main.cpp:2178-2183).
            let mode = if emit_ir {
                EmitMode::Ir { test: true }
            } else {
                EmitMode::Test {
                    output,
                    libs: libs.to_vec(),
                }
            };
            let rc = emit::emit_jir(f, mode, opt, lto, strip);
            if rc != 0 {
                failed += 1;
            } else {
                passed += 1;
            }
        }
    }

    println!();
    println!(
        "Summary: {passed} file(s) passed, {failed} file(s) failed, {skipped} file(s) without tests, {} file(s) scanned",
        files.len()
    );
    if failed == 0 { 0 } else { 1 }
}
