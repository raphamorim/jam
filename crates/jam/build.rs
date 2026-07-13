/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Bake the git short-SHA (plus a `-dirty` marker when the worktree differs
//! from HEAD) into the binary as `JAM_VERSION_SHA`. Falls back to `unknown`
//! outside a git checkout. `jam version` prints
//! `<CARGO_PKG_VERSION>-<JAM_VERSION_SHA>`.

use std::process::Command;

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let version = match sha {
        Some(sha) => {
            // `git diff --quiet HEAD` exits non-zero when the worktree has any
            // tracked changes vs HEAD.
            let dirty = Command::new("git")
                .args(["diff", "--quiet", "HEAD"])
                .status()
                .map(|s| !s.success())
                .unwrap_or(false);
            if dirty { format!("{sha}-dirty") } else { sha }
        }
        None => "unknown".to_string(),
    };

    println!("cargo:rustc-env=JAM_VERSION_SHA={version}");
    // Re-stamp when HEAD moves or the index changes.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
