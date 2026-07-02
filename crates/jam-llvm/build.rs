/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Build script: locate the system LLVM via `llvm-config` and emit the link
//! directives for libLLVM. We link the C API only (the symbols our hand-rolled
//! `extern "C"` FFI in `src/raw.rs` declares) — no C++ toolchain participates
//! in our build, just the prebuilt library.
//!
//! The conventions here deliberately mirror the established Rust↔LLVM crates
//! (`llvm-sys`, which inkwell and most others link through, and rustc's own
//! `rustc_llvm`):
//!   * locate `llvm-config` via `$LLVM_CONFIG`, then the versioned
//!     `$LLVM_SYS_221_PREFIX/bin` convention, then `PATH`, then Homebrew;
//!   * pass an explicit `--link-shared` / `--link-static` to `llvm-config` so
//!     `--libs`/`--libdir`/`--system-libs` all answer for the same mode;
//!   * choose the C++ runtime by platform (only needed for a static link);
//!   * handle `--system-libs` entries that are absolute paths, not just `-lx`.
//!

// note: targets LLVM 22.1

#![allow(clippy::collapsible_if)]

use std::path::PathBuf;
use std::process::Command;

const LLVM_SYS_PREFIX: &str = "LLVM_SYS_221_PREFIX";

fn run(bin: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn locate_llvm_config() -> String {
    if let Ok(explicit) = std::env::var("LLVM_CONFIG") {
        if run(&explicit, &["--version"]).is_some() {
            return explicit;
        }
    }
    if let Ok(prefix) = std::env::var(LLVM_SYS_PREFIX) {
        let cand = PathBuf::from(prefix).join("bin").join("llvm-config");
        let cand = cand.to_string_lossy().into_owned();
        if run(&cand, &["--version"]).is_some() {
            return cand;
        }
    }
    for cand in [
        "llvm-config",
        "/opt/homebrew/opt/llvm/bin/llvm-config",
        "/usr/local/opt/llvm/bin/llvm-config",
    ] {
        if run(cand, &["--version"]).is_some() {
            return cand.to_string();
        }
    }
    panic!(
        "llvm-config not found. Install LLVM 22 (e.g. `brew install llvm`) and put it \
         on PATH, or set LLVM_CONFIG / {LLVM_SYS_PREFIX}."
    );
}

/// Emit a link directive for one token from `--libs` / `--system-libs`, which
/// may be `-lfoo` or an absolute path to a `.a`/`.so`/`.dylib`/`.tbd`.
fn link_token(tok: &str, default_kind: &str) {
    if let Some(name) = tok.strip_prefix("-l") {
        println!("cargo:rustc-link-lib={default_kind}={name}");
        return;
    }
    let path = PathBuf::from(tok);
    if path.is_absolute() {
        if let Some(dir) = path.parent() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            let name = stem.strip_prefix("lib").unwrap_or(stem);
            let kind = if tok.ends_with(".a") {
                "static"
            } else {
                "dylib"
            };
            println!("cargo:rustc-link-lib={kind}={name}");
        }
    }
}

fn main() {
    for var in [
        "LLVM_CONFIG",
        LLVM_SYS_PREFIX,
        "CXXSTDLIB",
        "JAM_LLVM_LINK_SHARED",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=build.rs");

    let lc = locate_llvm_config();

    let version = run(&lc, &["--version"]).unwrap_or_default();
    if !version.starts_with("22.") {
        println!("cargo:warning=jam_llvm targets LLVM 22.x; llvm-config reports {version}");
    }

    // Prefer the monolithic shared libLLVM (what the jam toolchain links). Fall
    // back to a static link if this LLVM was built static-only. Override with
    // JAM_LLVM_LINK_SHARED=0.
    let want_shared = std::env::var("JAM_LLVM_LINK_SHARED")
        .map(|v| v != "0")
        .unwrap_or(true);
    let shared_ok = want_shared
        && run(&lc, &["--link-shared", "--libs"])
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
    let (link_flag, kind) = if shared_ok {
        ("--link-shared", "dylib")
    } else {
        ("--link-static", "static")
    };

    // All three queries pinned to the chosen mode so their answers agree.
    let libdir = run(&lc, &[link_flag, "--libdir"])
        .or_else(|| run(&lc, &["--libdir"]))
        .expect("llvm-config --libdir failed");
    println!("cargo:rustc-link-search=native={libdir}");

    let mut linked_any = false;
    if let Some(libs) = run(&lc, &[link_flag, "--libs"]) {
        for tok in libs.split_whitespace() {
            link_token(tok, kind);
            linked_any = true;
        }
    }
    if !linked_any {
        println!("cargo:rustc-link-lib=dylib=LLVM");
    }

    // System libraries libLLVM depends on (zlib, zstd, curses, libxml2, …).
    // Empty for a stock shared build; populated for static / Linux.
    if let Some(sys) = run(&lc, &[link_flag, "--system-libs"]) {
        for tok in sys.split_whitespace() {
            link_token(tok, "dylib");
        }
    }

    // The C++ runtime. A shared libLLVM carries this in its own load commands,
    // so it is only our responsibility for a static link. Choice is by platform
    // (matching llvm-sys/rustc), overridable via CXXSTDLIB.
    if kind == "static" {
        if let Some(cxx) = cxx_stdlib() {
            println!("cargo:rustc-link-lib=dylib={cxx}");
        }
    }

    // Make a shared libLLVM discoverable at runtime without
    // DYLD_LIBRARY_PATH / LD_LIBRARY_PATH. Harmless for a static link.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");

    // Compile the one C++ shim (sets TargetOptions fields the C API can't).
    // This is the only C++ in the build; cc(true) also emits the C++-stdlib
    // link directive. We pass llvm-config's own --cxxflags so the shim is built
    // with the same include path, language std, and defines LLVM itself uses.
    let cxxflags = run(&lc, &["--cxxflags"]).unwrap_or_default();
    let mut shim = cc::Build::new();
    // Don't add -Wall/-Wextra: we'd only be warned about LLVM's own headers.
    shim.cpp(true).warnings(false).file("shim/jam_shim.cpp");
    for tok in cxxflags.split_whitespace() {
        shim.flag(tok);
    }
    shim.compile("jam_shim");
    println!("cargo:rerun-if-changed=shim/jam_shim.cpp");
}

/// The C++ standard library to link against a static libLLVM. `None` on MSVC,
/// where it is implicit.
fn cxx_stdlib() -> Option<String> {
    if let Ok(over) = std::env::var("CXXSTDLIB") {
        return if over.is_empty() { None } else { Some(over) };
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("msvc") {
        None
    } else if target.contains("apple")
        || target.contains("darwin")
        || target.contains("freebsd")
        || target.contains("openbsd")
    {
        Some("c++".to_string())
    } else {
        Some("stdc++".to_string())
    }
}
