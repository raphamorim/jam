/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Target triple model — CPU/OS/ABI classification plus target-specific
//! queries (pointer size, PIC/PIE requirements, libc, calling convention).
//! Ported from `src/target.{h,cpp}`.
//!
//! The C++ builds a [`Target`] from an `llvm::Triple`; here the triple is
//! parsed from its string form in pure Rust (the classification logic is
//! normative platform behavior, identical on any host). Host detection is the
//! driver's composition: `Target::from_triple_str(&jam_llvm::default_target_triple())`.

/// CPU architecture family.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    AArch64,
    Arm,
    Riscv64,
    #[default]
    Unknown,
}

/// Operating system.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Os {
    Linux,
    MacOs,
    Windows,
    FreeBsd,
    #[default]
    Unknown,
}

/// C-library / environment ABI.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Abi {
    /// No specific ABI (e.g. macOS).
    None,
    /// GNU C library.
    Gnu,
    /// Microsoft Visual C++.
    Msvc,
    /// Minimalist GNU for Windows.
    MinGw,
    /// Musl C library.
    Musl,
    #[default]
    Unknown,
}

/// LLVM calling convention for emitted functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    /// Standard C calling convention.
    C,
    /// Fast calling convention.
    Fast,
    /// Cold calling convention.
    Cold,
    /// Preserve all registers.
    Preserve,
}

/// A parsed target triple: `arch-vendor-os-environment`. The default is all
/// `Unknown` (mirrors the C++ default constructor).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Target {
    pub arch: Arch,
    pub os: Os,
    pub abi: Abi,
}

fn parse_arch(s: &str) -> Arch {
    match s {
        "x86_64" | "amd64" => Arch::X86_64,
        "aarch64" | "aarch64_be" | "arm64" | "arm64e" => Arch::AArch64,
        "riscv64" => Arch::Riscv64,
        "arm" | "armeb" | "thumb" | "thumbeb" => Arch::Arm,
        _ if s.starts_with("armv") || s.starts_with("thumbv") => Arch::Arm,
        _ => Arch::Unknown,
    }
}

fn parse_os(s: &str) -> Os {
    if s.starts_with("linux") {
        Os::Linux
    } else if s.starts_with("darwin") || s.starts_with("macosx") {
        Os::MacOs
    } else if s.starts_with("win32") || s.starts_with("windows") {
        Os::Windows
    } else if s.starts_with("freebsd") {
        Os::FreeBsd
    } else {
        Os::Unknown
    }
}

fn parse_abi(env: &str, os: Os) -> Abi {
    if env.starts_with("gnu") {
        // gnu, gnueabi, gnueabihf, gnux32
        Abi::Gnu
    } else if env == "msvc" {
        Abi::Msvc
    } else if env.starts_with("musl") {
        // musl, musleabi, musleabihf
        Abi::Musl
    } else if os == Os::Windows {
        // Windows with no recognized environment -> MinGW (the C++
        // `UnknownEnvironment` arm).
        Abi::MinGw
    } else if os == Os::MacOs {
        Abi::None
    } else {
        Abi::Unknown
    }
}

impl Target {
    pub fn new(arch: Arch, os: Os, abi: Abi) -> Target {
        Target { arch, os, abi }
    }

    /// Classify a target from its LLVM triple string (`arch-vendor-os-env`).
    /// Components are matched by content so both 3- and 4-field triples
    /// (`x86_64-linux-gnu` and `x86_64-unknown-linux-gnu`) classify correctly.
    pub fn from_triple_str(triple: &str) -> Target {
        let parts: Vec<&str> = triple.split('-').collect();
        let arch = parts
            .first()
            .map(|s| parse_arch(s))
            .unwrap_or(Arch::Unknown);

        // OS: the first component after the arch that matches a known OS.
        let mut os = Os::Unknown;
        let mut os_idx = None;
        for (i, p) in parts.iter().enumerate().skip(1) {
            let o = parse_os(p);
            if o != Os::Unknown {
                os = o;
                os_idx = Some(i);
                break;
            }
        }

        // Environment: the component right after the OS (or position 3 when no
        // OS field was recognized).
        let env = match os_idx {
            Some(i) => parts.get(i + 1).copied().unwrap_or(""),
            None => parts.get(3).copied().unwrap_or(""),
        };
        let abi = parse_abi(env, os);

        Target { arch, os, abi }
    }

    /// Reconstruct an LLVM triple string. Vendor is always `unknown`; macOS
    /// emits no ABI suffix and MinGW maps back to `-gnu`.
    pub fn to_llvm_triple(&self) -> String {
        let arch = match self.arch {
            Arch::X86_64 => "x86_64",
            Arch::AArch64 => "aarch64",
            Arch::Arm => "arm",
            Arch::Riscv64 => "riscv64",
            Arch::Unknown => "unknown",
        };
        let os = match self.os {
            Os::Linux => "linux",
            Os::MacOs => "darwin",
            Os::Windows => "windows",
            Os::FreeBsd => "freebsd",
            Os::Unknown => "unknown",
        };
        let abi = match self.abi {
            Abi::Gnu => "-gnu",
            Abi::Msvc => "-msvc",
            Abi::MinGw => "-gnu",
            Abi::Musl => "-musl",
            Abi::None | Abi::Unknown => "",
        };
        format!("{arch}-unknown-{os}{abi}")
    }

    /// Human-readable target name (e.g. `x86_64-linux-gnu`). Note macOS spells
    /// the OS `macos` here (vs `darwin` in [`Self::to_llvm_triple`]).
    pub fn name(&self) -> String {
        let arch = match self.arch {
            Arch::X86_64 => "x86_64",
            Arch::AArch64 => "aarch64",
            Arch::Arm => "arm",
            Arch::Riscv64 => "riscv64",
            Arch::Unknown => "unknown",
        };
        let os = match self.os {
            Os::Linux => "linux",
            Os::MacOs => "macos",
            Os::Windows => "windows",
            Os::FreeBsd => "freebsd",
            Os::Unknown => "unknown",
        };
        let mut out = format!("{arch}-{os}");
        if self.abi != Abi::None && self.abi != Abi::Unknown {
            let abi = match self.abi {
                Abi::Gnu => "gnu",
                Abi::Msvc => "msvc",
                Abi::MinGw => "mingw",
                Abi::Musl => "musl",
                _ => "",
            };
            out.push('-');
            out.push_str(abi);
        }
        out
    }

    /// macOS always links libSystem (which contains libc); FreeBSD requires
    /// libc for a stable syscall interface.
    pub fn requires_libc(&self) -> bool {
        self.os == Os::MacOs || self.os == Os::FreeBsd
    }

    /// PIC is required on Windows (DLLs) and when using GNU libc.
    pub fn requires_pic(&self) -> bool {
        self.os == Os::Windows || self.abi == Abi::Gnu
    }

    /// PIE is required on macOS/Darwin.
    pub fn requires_pie(&self) -> bool {
        self.os == Os::MacOs
    }

    pub fn can_dynamic_link(&self) -> bool {
        true
    }

    pub fn uses_cabi(&self) -> bool {
        true
    }

    pub fn libc_name(&self) -> &'static str {
        match self.os {
            Os::Windows => return "mingw",
            Os::MacOs => return "darwin",
            _ => {}
        }
        match self.abi {
            Abi::Gnu | Abi::MinGw => "glibc",
            Abi::Musl => "musl",
            Abi::Msvc => "msvcrt",
            _ => "unknown",
        }
    }

    pub fn pointer_size(&self) -> i32 {
        match self.arch {
            Arch::X86_64 | Arch::AArch64 | Arch::Riscv64 => 8,
            Arch::Arm => 4,
            Arch::Unknown => 8,
        }
    }

    pub fn pointer_alignment(&self) -> i32 {
        self.pointer_size()
    }

    pub fn default_cc(&self) -> CallingConvention {
        CallingConvention::C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_gnu_four_field() {
        let t = Target::from_triple_str("x86_64-unknown-linux-gnu");
        assert_eq!(t.arch, Arch::X86_64);
        assert_eq!(t.os, Os::Linux);
        assert_eq!(t.abi, Abi::Gnu);
        assert_eq!(t.name(), "x86_64-linux-gnu");
    }

    #[test]
    fn parses_linux_gnu_three_field() {
        // Vendor omitted: content-based scan still finds the OS + env.
        let t = Target::from_triple_str("x86_64-linux-gnu");
        assert_eq!(t.arch, Arch::X86_64);
        assert_eq!(t.os, Os::Linux);
        assert_eq!(t.abi, Abi::Gnu);
    }

    #[test]
    fn parses_macos_arm64() {
        // macOS spells the arch `arm64` (not `aarch64`) and carries a darwin
        // version suffix; the abi has no environment field -> None.
        let t = Target::from_triple_str("arm64-apple-darwin23.6.0");
        assert_eq!(t.arch, Arch::AArch64);
        assert_eq!(t.os, Os::MacOs);
        assert_eq!(t.abi, Abi::None);
        assert!(t.requires_pie());
        assert!(t.requires_libc());
        assert_eq!(t.name(), "aarch64-macos");
        assert_eq!(t.to_llvm_triple(), "aarch64-unknown-darwin");
        assert_eq!(t.libc_name(), "darwin");
    }

    #[test]
    fn parses_aarch64_linux_musl() {
        let t = Target::from_triple_str("aarch64-unknown-linux-musl");
        assert_eq!(t.arch, Arch::AArch64);
        assert_eq!(t.os, Os::Linux);
        assert_eq!(t.abi, Abi::Musl);
        assert_eq!(t.libc_name(), "musl");
        assert!(!t.requires_pic()); // musl, not gnu
    }

    #[test]
    fn parses_windows_msvc_and_gnu() {
        let msvc = Target::from_triple_str("x86_64-pc-windows-msvc");
        assert_eq!(msvc.os, Os::Windows);
        assert_eq!(msvc.abi, Abi::Msvc);
        assert!(msvc.requires_pic());
        assert_eq!(msvc.libc_name(), "mingw"); // os arm wins over abi

        // `-windows-gnu` is GNU env (not MinGW); MinGW only when env absent.
        let gnu = Target::from_triple_str("x86_64-pc-windows-gnu");
        assert_eq!(gnu.abi, Abi::Gnu);
        let bare = Target::from_triple_str("x86_64-pc-windows");
        assert_eq!(bare.abi, Abi::MinGw);
    }

    #[test]
    fn pointer_sizes_by_arch() {
        assert_eq!(
            Target::new(Arch::X86_64, Os::Linux, Abi::Gnu).pointer_size(),
            8
        );
        assert_eq!(
            Target::new(Arch::AArch64, Os::MacOs, Abi::None).pointer_size(),
            8
        );
        assert_eq!(
            Target::new(Arch::Riscv64, Os::Linux, Abi::Gnu).pointer_size(),
            8
        );
        assert_eq!(
            Target::new(Arch::Arm, Os::Linux, Abi::Gnu).pointer_size(),
            4
        );
        // Unknown defaults to 64-bit.
        assert_eq!(Target::default().pointer_size(), 8);
        assert_eq!(
            Target::new(Arch::X86_64, Os::Linux, Abi::Gnu).pointer_alignment(),
            8
        );
    }

    #[test]
    fn arm_string_variants_classify_as_arm() {
        assert_eq!(parse_arch("armv7"), Arch::Arm);
        assert_eq!(parse_arch("thumbv7m"), Arch::Arm);
        assert_eq!(parse_arch("armeb"), Arch::Arm);
        // arm64 is AArch64, must not fall into the arm arm.
        assert_eq!(parse_arch("arm64"), Arch::AArch64);
    }

    #[test]
    fn default_cc_is_c() {
        assert_eq!(Target::default().default_cc(), CallingConvention::C);
        assert!(Target::default().can_dynamic_link());
        assert!(Target::default().uses_cabi());
    }

    #[test]
    fn unknown_triple_is_all_unknown() {
        let t = Target::from_triple_str("wat-wat-wat");
        assert_eq!(t.arch, Arch::Unknown);
        assert_eq!(t.os, Os::Unknown);
        assert_eq!(t.abi, Abi::Unknown);
    }
}
