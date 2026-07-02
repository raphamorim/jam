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

