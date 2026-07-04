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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OptLevel {
    None,       // -O0
    Less,       // -O1
    Default,    // -O2
    Aggressive, // -O3
    Size,       // -Os
    Small,      // -Oz
}

impl OptLevel {
    fn codegen_level(self) -> raw::LLVMCodeGenOptLevel {
        use raw::LLVMCodeGenOptLevel as L;
        match self {
            OptLevel::None => L::None,
            OptLevel::Less => L::Less,
            OptLevel::Default => L::Default,
            OptLevel::Aggressive | OptLevel::Size | OptLevel::Small => L::Aggressive,
        }
    }

    /// The integer discriminant the C++ shim's `jam_shim_optimize` switches on.
    /// Matches the declaration order (None=0 … Small=5) and the C++ facade's
    /// `JamOptLevel` enum, so the shim selects the identical `OptimizationLevel`.
    fn shim_discriminant(self) -> c_int {
        match self {
            OptLevel::None => 0,
            OptLevel::Less => 1,
            OptLevel::Default => 2,
            OptLevel::Aggressive => 3,
            OptLevel::Size => 4,
            OptLevel::Small => 5,
        }
    }

    fn is_debug(self) -> bool {
        self == OptLevel::None
    }
}

impl Lto {
    /// The integer discriminant the C++ shim expects (Off=0, Thin=1, Fat=2).
    fn shim_discriminant(self) -> c_int {
        match self {
            Lto::Off => 0,
            Lto::Thin => 1,
            Lto::Fat => 2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Lto {
    Off,
    Thin,
    Fat,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Strip {
    None,
    DebugInfo,
    Symbols,
}

pub struct TargetMachine {
    ptr: raw::LLVMTargetMachineRef,
    opt: OptLevel,
    lto: Lto,
    pic: bool,
}

impl TargetMachine {
    pub fn new(
        triple: &str,
        cpu: &str,
        features: &str,
        is_pic: bool,
        opt: OptLevel,
        lto: Lto,
    ) -> Option<TargetMachine> {
        let ctriple = cstr(triple);
        // Match the C++ facade: a missing CPU falls back to "generic"
        // (`cpu ? cpu : "generic"`). An empty feature string is left as-is.
        let ccpu = cstr(if cpu.is_empty() { "generic" } else { cpu });
        let cfeat = cstr(features);
        unsafe {
            let mut target: raw::LLVMTargetRef = std::ptr::null_mut();
            let mut err: *mut c_char = std::ptr::null_mut();
            if raw::LLVMGetTargetFromTriple(ctriple.as_ptr(), &mut target, &mut err) != 0 {
                let _ = take_message(err);
                return None;
            }
            let reloc = if is_pic {
                raw::LLVMRelocMode::PIC
            } else {
                raw::LLVMRelocMode::Static
            };
            let tm = raw::LLVMCreateTargetMachine(
                target,
                ctriple.as_ptr(),
                ccpu.as_ptr(),
                cfeat.as_ptr(),
                opt.codegen_level(),
                reloc,
                raw::LLVMCodeModel::Default,
            );
            if tm.is_null() {
                return None;
            }
            // Mirror the C++ facade: enable per-function/-data sections when
            // optimizing. TargetOptions is not in the LLVM-C API, so this goes
            // through the C++ shim (see `shim/jam_shim.cpp`).
            if opt != OptLevel::None {
                raw::jam_set_target_machine_sections(tm, 1, 1);
            }
            Some(TargetMachine {
                ptr: tm,
                opt,
                lto,
                pic: is_pic,
            })
        }
    }

    /// Stamp the module's data layout from this machine, plus the PIC/PIE level
    /// module flags when relocation is PIC (so the emitted object advertises PIC
    /// and a linker building a PIE binary accepts it).
    pub fn configure_module(&self, module: &Module<'_>) {
        unsafe {
            let dl = raw::LLVMCreateTargetDataLayout(self.ptr);
            raw::LLVMSetModuleDataLayout(module.as_ptr(), dl);
            if self.pic {
                // PICLevel::BigPIC == 2, PIELevel::Large == 2. The C++ uses the
                // `Max` merge behavior, which the C enum lacks; `Override` is
                // codegen-inert for jam's single, never-IR-linked module.
                self.add_u32_module_flag(module, "PIC Level", 2);
                self.add_u32_module_flag(module, "PIE Level", 2);
            }
        }
    }

    unsafe fn add_u32_module_flag(&self, module: &Module<'_>, key: &str, val: u64) {
        unsafe {
            let ctx = raw::LLVMGetModuleContext(module.as_ptr());
            let i32ty = raw::LLVMInt32TypeInContext(ctx);
            let v = raw::LLVMConstInt(i32ty, val, 0);
            let md = raw::LLVMValueAsMetadata(v);
            raw::LLVMAddModuleFlag(
                module.as_ptr(),
                raw::LLVMModuleFlagBehavior::Override,
                key.as_ptr() as *const c_char,
                key.len(),
                md,
            );
        }
    }

    /// Run the new-PM optimization pipeline against `module` in place. This is
    /// the C++ facade's pipeline copied verbatim into the shim
    /// (`shim/jam_shim.cpp`, `jam_shim_optimize`): the size-favoring function
    /// attrs (Os/Oz), the pre-pipeline internalize+globaldce (keeping `main` and
    /// `llvm.used` members, skipped under LTO), and the OptimizationLevel switch
    /// driving `buildO0DefaultPipeline` / `buildLTOPreLinkDefaultPipeline` /
    /// `buildPerModuleDefaultPipeline`. Running the identical pipeline on the
    /// same LLVM gives byte-identical optimized IR to the oracle.
    ///
    /// A no-op at `OptLevel::None` is still well-defined (the shim runs the O0
    /// default pipeline); callers skip it at `None` to match the oracle, which
    /// only invokes the optimizer when emitting an object/bitcode at any level.
    pub fn run_optimization(&self, module: &Module<'_>) {
        unsafe {
            raw::jam_shim_optimize(
                module.as_ptr(),
                self.ptr,
                self.opt.shim_discriminant(),
                self.opt.is_debug() as c_int,
                self.lto.shim_discriminant(),
            );
        }
    }

    /// Emit an object file (LTO off) or LLVM bitcode (LTO on) to `filename`. The
    /// optimization pipeline is NOT run here — callers invoke
    /// [`run_optimization`] first (the C++ facade runs the pipeline as part of
    /// its emit; we split it so `--emit-ir` can print the UNoptimized module,
    /// matching the oracle's pre-optimization `--emit-ir`). Returns the LLVM
    /// error text on failure.
    pub fn emit_to_file(&self, module: &Module<'_>, filename: &str) -> Result<(), String> {
        unsafe {
            // Emit.
            let cfile = cstr(filename);
            if self.lto != Lto::Off {
                if raw::LLVMWriteBitcodeToFile(module.as_ptr(), cfile.as_ptr()) != 0 {
                    return Err("failed to write bitcode".to_string());
                }
                Ok(())
            } else {
                let mut err: *mut c_char = std::ptr::null_mut();
                let failed = raw::LLVMTargetMachineEmitToFile(
                    self.ptr,
                    module.as_ptr(),
                    cfile.as_ptr(),
                    raw::LLVMCodeGenFileType::ObjectFile,
                    &mut err,
                );
                if failed != 0 {
                    return Err(take_message(err));
                }
                Ok(())
            }
        }
    }
}

impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe { raw::LLVMDisposeTargetMachine(self.ptr) }
    }
}

/// Append `func` to the module's `@llvm.used` so whole-module internalization
/// never strips it (used for `export fn`). Rebuilds the `@llvm.used` appending
/// global with the new entry — the C API has no `appendToUsed`.
pub fn append_to_used(module: &Module<'_>, func: Function<'_>) {
    unsafe {
        let m = module.as_ptr();
        let ctx = raw::LLVMGetModuleContext(m);
        let ptr_ty = raw::LLVMPointerTypeInContext(ctx, 0);

        // Gather existing entries, then drop the old global.
        let mut entries: Vec<raw::LLVMValueRef> = Vec::new();
        let used_name = cstr("llvm.used");
        let old = raw::LLVMGetNamedGlobal(m, used_name.as_ptr());
        if !old.is_null() {
            let init = raw::LLVMGetInitializer(old);
            if !init.is_null() {
                let n = raw::LLVMGetNumOperands(init);
                for i in 0..n {
                    entries.push(raw::LLVMGetOperand(init, i as c_uint));
                }
            }
            raw::LLVMDeleteGlobal(old);
        }
        // Avoid duplicate entries.
        let fv = func.raw_value();
        if !entries.contains(&fv) {
            entries.push(fv);
        }

        let arr = raw::LLVMConstArray2(ptr_ty, entries.as_mut_ptr(), entries.len() as u64);
        let arr_ty = raw::LLVMArrayType2(ptr_ty, entries.len() as u64);
        let g = raw::LLVMAddGlobal(m, arr_ty, used_name.as_ptr());
        raw::LLVMSetLinkage(g, raw::LLVMLinkage::Appending);
        raw::LLVMSetInitializer(g, arr);
        let section = cstr("llvm.metadata");
        raw::LLVMSetSection(g, section.as_ptr());
    }
}
