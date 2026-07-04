/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Target initialization, host queries, the `TargetMachine`, and object/bitcode
//! emission — the Rust reproduction of the C++ facade's target + emit surface.
//!
//! The emit path is the one place the LLVM-C API under-exposes the C++ API, so
//! two pieces live in the C++ shim (`shim/jam_shim.cpp`, compiled by `build.rs`)
//! rather than being driven through the C API:
//!   * the optimization pipeline (`jam_shim_optimize`) is the C++ facade's
//!     `PassBuilder` configuration copied VERBATIM — the tuning options, the
//!     analysis-manager wiring, the `OptimizationLevel` switch, the pre-pipeline
//!     `InternalizePass([main]) + GlobalDCEPass`, the size-favoring function
//!     attrs (Os/Oz), and the per-module/LTO/O0 pipeline selection. Running the
//!     identical pipeline on the same LLVM gives byte-identical optimized IR to
//!     the oracle, which the C-API textual-pipeline form could not guarantee.
//!   * `TargetOptions::{FunctionSections, DataSections}` are not reachable
//!     through the LLVM-C API at all (`TargetOptions` is C++-only), so
//!     [`TargetMachine::new`] flips those two bits via
//!     `jam_set_target_machine_sections` when `opt != None`, matching the C++.
//!
//! These two functions are the only C++ in the crate and the sole reason the
//! build needs a C++ compiler + LLVM's C++ headers. [`TargetMachine::run_optimization`]
//! invokes the optimizer; callers run it before [`TargetMachine::emit_to_file`].

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::LazyLock;

use crate::ll::{Function, Module};
use crate::raw;

fn cstr(s: &str) -> CString {
    CString::new(s).expect("string contained an interior NUL byte")
}

unsafe fn take_message(p: *mut c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe {
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        raw::LLVMDisposeMessage(p);
        s
    }
}

pub fn init_native_target() {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        {
            raw::LLVMInitializeAArch64TargetInfo();
            raw::LLVMInitializeAArch64Target();
            raw::LLVMInitializeAArch64TargetMC();
        }
        #[cfg(target_arch = "x86_64")]
        {
            raw::LLVMInitializeX86TargetInfo();
            raw::LLVMInitializeX86Target();
            raw::LLVMInitializeX86TargetMC();
        }
    }
}

pub fn init_native_asm_printer() {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        raw::LLVMInitializeAArch64AsmPrinter();
        #[cfg(target_arch = "x86_64")]
        raw::LLVMInitializeX86AsmPrinter();
    }
}

pub fn init_native_asm_parser() {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        raw::LLVMInitializeAArch64AsmParser();
        #[cfg(target_arch = "x86_64")]
        raw::LLVMInitializeX86AsmParser();
    }
}

pub fn init_all_targets() {
    unsafe {
        raw::LLVMInitializeAArch64TargetInfo();
        raw::LLVMInitializeAArch64Target();
        raw::LLVMInitializeAArch64TargetMC();
        raw::LLVMInitializeAArch64AsmParser();
        raw::LLVMInitializeAArch64AsmPrinter();
        raw::LLVMInitializeX86TargetInfo();
        raw::LLVMInitializeX86Target();
        raw::LLVMInitializeX86TargetMC();
        raw::LLVMInitializeX86AsmParser();
        raw::LLVMInitializeX86AsmPrinter();
    }
}

pub fn default_target_triple() -> String {
    unsafe { take_message(raw::LLVMGetDefaultTargetTriple()) }
}

/// Host CPU name (e.g. `apple-m1`), computed once and cached. The driver passes
/// this to [`TargetMachine::new`] for native builds.
pub fn host_cpu_name() -> String {
    static CPU: LazyLock<String> =
        LazyLock::new(|| unsafe { take_message(raw::LLVMGetHostCPUName()) });
    CPU.clone()
}

/// Host CPU feature string (`+feat,-feat,…`), computed once and cached.
pub fn host_cpu_features() -> String {
    static FEATS: LazyLock<String> =
        LazyLock::new(|| unsafe { take_message(raw::LLVMGetHostCPUFeatures()) });
    FEATS.clone()
}

