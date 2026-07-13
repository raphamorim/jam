/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! The compile driver: one single-pass argument loop. `run` compiles a file,
//! runs it, propagates its exit code, and deletes the temp binary; `test`
//! compiles in test mode (synthesizing a harness `main` over the file's
//! `tfn`s), runs it, and propagates its output + exit. `test <dir>` (or no
//! arg → cwd) walks `*.jam` files, runs each one's tests, and prints a
//! `Summary:` line. The `build` subcommand reuses the same loop with
//! `run`/`test` demoted to ordinary positionals.

use std::path::Path;

use jam_llvm::{Lto, OptLevel};

use crate::cli::emit::{self, EmitMode, Strip};

/// Parse a `-C opt-level=N` value to an [`OptLevel`]. Returns `None` for an
/// unrecognized value.
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

/// Parse one `-C key=value` payload into the opt/lto/strip slots. Returns the
/// process exit code on a malformed or unknown option.
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

/// The compile driver: a single-pass argument loop, so flag interplay — flag
/// position relative to `run`, option values that look like flags, trailing
/// options falling through as positionals — resolves in one deterministic
/// order.
///
/// `start` is the first argv index to consider: 1 for the bare
/// `jam [FLAGS] <file>` form (where `run`/`test` toggle modes anywhere in
/// argv), 2 under the `build` subcommand. `mode_tokens` is
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
        // `run`/`test` mode tokens — recognized anywhere in argv.
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
        // Linker flags — accepted in every mode. The spaced forms consume the
        // next token unconditionally, so a value that looks like a flag
        // (`-l --release`) is still a library name.
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
        // `--release` / `--release-small` were removed in favor of `-C opt-level`.
        if arg == "--release" {
            eprintln!("Error: `--release` was removed; use `-C opt-level=3` instead");
            return 1;
        }
        if arg == "--release-small" {
            eprintln!("Error: `--release-small` was removed; use `-C opt-level=z` instead");
            return 1;
        }
        // `-C key=value` (spaced) or `-Ckey=value` (joined) codegen options.
        // Handled BEFORE the `run` strictness check, so
        // `jam run -C opt-level=3 file` is accepted.
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
        // Inside `run`, anything else flag-shaped is an error — including the
        // flags parsed below, which are honored only BEFORE the `run` token.
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
        // Compile-only / test-mode flags.
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
        // positional is an error. A trailing option with no value
        // (`jam file -o`) falls through here.
        if !filename.is_empty() {
            eprintln!("Error: unexpected extra argument `{arg}` (already have `{filename}`)");
            return 1;
        }
        filename = arg.clone();
        i += 1;
    }

    // `jam test` with no path means "run every test under cwd".
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
    // normal compile. It still requires an input file — the emptiness check
    // above runs first.
    if show_target {
        emit::print_target_info();
    }
    // Directory input: only meaningful with `test`.
    if Path::new(&filename).is_dir() {
        if !test_mode {
            eprintln!("Error: directory input is only supported with `test` (got '{filename}')");
            return 1;
        }
        return test_directory(&filename, &libs, emit_ir, opt, lto, strip);
    }
    // The default output name `output` collides with a same-named directory
    // (this repo has one): the link fails with a cryptic `ld: errno=21`. Fall
    // back to a usable name instead. Applies to an explicit `-o` name too, and
    // the note prints even under `--emit-ir`, which resolves the name it will
    // never write.
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
    // A file carrying `// expect-error:` directives is a MUST-FAIL test: the
    // compile itself is the assertion, so there is no harness to build or run.
    if test_mode && !emit_ir {
        let expected = expect_error_directives(&filename);
        if !expected.is_empty() {
            return run_must_fail(&filename, &expected);
        }
    }
    let mode = if emit_ir {
        // Print-IR-and-exit wins over run/test execution; under `test` the
        // dump includes the synthesized harness main.
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

/// `// expect-error: <substring>` directives make a file a MUST-FAIL test:
/// `jam test` compiles it expecting a non-zero exit, and the compiler's stderr
/// must contain every listed substring. Returns the (possibly empty) list.
fn expect_error_directives(path: &str) -> Vec<String> {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("// expect-error:"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Run one must-fail test: re-invoke this binary as `jam build <file>` with
/// captured output, require a failing exit, and require every expected
/// substring in stderr. Returns the test's exit code (0 = passed).
fn run_must_fail(path: &str, expected: &[String]) -> i32 {
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("jam"));
    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");
    let out_name = format!("jam_reject_{stem}");
    let output = match std::process::Command::new(exe)
        .arg("build")
        .arg(path)
        .arg("-o")
        .arg(&out_name)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            println!("error: failed to spawn compile for must-fail test: {e}");
            return 1;
        }
    };
    if output.status.success() {
        let _ = std::fs::remove_file(&out_name);
        println!("expected compile to FAIL, but it succeeded");
        return 1;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let missing: Vec<&String> = expected.iter().filter(|e| !stderr.contains(*e)).collect();
    if missing.is_empty() {
        println!("{} expected error(s) matched", expected.len());
        return 0;
    }
    for m in &missing {
        println!("expected error not found: {m}");
    }
    println!("--- compiler stderr ---");
    print!("{stderr}");
    1
}

/// Recursively discover `*.jam` files under `dir`, sorted.
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

/// Cheap `tfn` substring scan — avoids lowering files with no test functions.
/// Matches `tfn` at line start or after whitespace, followed by whitespace
/// or `(`.
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

/// Worker-pool size for `jam test <dir>`: min(cpu count, 8), overridable via
/// `JAM_TEST_JOBS` (>= 1). `JAM_TEST_JOBS=1` forces the serial in-process path
/// (useful under a debugger).
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

/// The `-C opt-level=` value spelling for a re-invoked worker.
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
/// into one buffered pipe. The drain thread keeps the pipe from filling (and
/// the child from blocking on a full pipe) while the parent polls for
/// completion.
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
    // Only the opt level is forwarded.
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

/// The parallel worker pool: per-file work is ~98% blocked subprocess waits
/// (clang link + first-exec assessment of the freshly linked test binary), so
/// a small pool of `jam test <file>` re-invocations gives near-linear speedup.
/// Each worker's output prints as a coherent block on completion, in
/// completion order.
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
/// then print the `Summary:` line. Files run through the
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
        // Both harness (`tfn`) files and `// expect-error:` must-fail files
        // count as runnable tests.
        if file_has_tests(f) || !expect_error_directives(f).is_empty() {
            runnable.push(f.clone());
        } else {
            skipped += 1;
        }
    }

    let jobs = test_jobs();
    if jobs > 1 && runnable.len() > 1 {
        // Must-fail files need no special casing here: each worker re-invokes
        // `jam test <file>`, whose single-file path dispatches on directives.
        run_tests_parallel(&runnable, libs, opt, jobs, &mut passed, &mut failed);
    } else {
        for f in &runnable {
            // Per-file block header: a leading blank line then `@<file>`.
            println!();
            println!("@{f}");
            // Flush so the header precedes the test binary's own stdout.
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let expected = expect_error_directives(f);
            if !expected.is_empty() && !emit_ir {
                if run_must_fail(f, &expected) != 0 {
                    failed += 1;
                } else {
                    passed += 1;
                }
                continue;
            }
            let stem = Path::new(f)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("out");
            let output = format!("jam_test_{stem}");
            // The serial walker threads `--emit-ir` into each per-file
            // compile; the parallel pool does not forward it.
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
