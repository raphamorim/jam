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

fn spawn_test_worker(
    exe: &Path,
    file: &str,
    libs: &[String],
    opt: OptLevel,
) -> Option<TestWorker> {
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
fn test_directory(dir: &str, libs: &[String], opt: OptLevel, lto: Lto, strip: Strip) -> i32 {
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
            let rc = emit::emit_jir(
                f,
                EmitMode::Test {
                    output,
                    libs: libs.to_vec(),
                },
                opt,
                lto,
                strip,
            );
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
