/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! C ABI helper for `extern "C"` function calling. Ported from
//! `src/cabi.{h,cpp}`. Picks the LLVM calling convention for the target,
//! applies linkage/attributes to functions, and checks C-ABI type
//! compatibility.
//!
//! Two behaviour-preserving omissions vs the C++: the explicit
//! `setVisibility(DefaultVisibility)` / `setDLLStorageClass(DefaultStorageClass)`
//! calls are dropped — they set the values a freshly-created function already
//! has, so they are no-ops. (If a future target needs non-default visibility,
//! add the wrapper then.)

use jam_llvm::{CallConv as LlvmCallConv, Function, Linkage, Module, Type};

use crate::target::{Abi, Os, Target};

/// C ABI helper bound to a target.
#[derive(Clone, Copy, Debug)]
pub struct CAbi {
    target: Target,
}

impl CAbi {
    pub fn new(target: Target) -> CAbi {
        CAbi { target }
    }

    /// The LLVM calling convention for the target. Windows MSVC uses the
    /// Microsoft x64 convention; everything else (incl. MinGW, which uses the
    /// System V AMD64 ABI on Windows) uses the C convention.
    pub fn calling_convention(&self) -> LlvmCallConv {
        match self.target.os {
            Os::Windows => {
                if self.target.abi == Abi::Msvc {
                    LlvmCallConv::Win64
                } else {
                    LlvmCallConv::C
                }
            }
            _ => LlvmCallConv::C,
        }
    }

    /// Apply C-ABI attributes (the calling convention) to a function.
    pub fn apply_function_attributes(&self, func: Function) {
        func.set_call_conv(self.calling_convention());
    }

    /// Is `ty` compatible with the C ABI? Primitives (int / float / pointer),
    /// void, structs, and arrays-of-compatible-elements are; everything else
    /// (functions, vectors, ...) is not.
    pub fn is_cabi_compatible(&self, ty: Type) -> bool {
        if ty.is_integer() || ty.is_float() || ty.is_pointer() {
            return true;
        }
        if ty.is_void() {
            return true;
        }
        if ty.is_struct() {
            // For now all structs are considered potentially compatible (layout
            // / padding checks can land later).
            return true;
        }
        if ty.is_array() {
            return self.is_cabi_compatible(ty.array_element_type());
        }
        false
    }

    /// Transform a Jam type to its C-ABI equivalent. Currently a pass-through
    /// (slice/string transforms can land later).
    pub fn to_cabi_type<'ctx>(&self, jam_type: Type<'ctx>) -> Type<'ctx> {
        jam_type
    }

    /// The external (linker-visible) name for a Jam function.
    pub fn extern_name(&self, jam_name: &str) -> String {
        self.platform_mangle(jam_name)
    }

    /// Platform name mangling. Every supported platform uses the C name as-is
    /// (modern macOS no longer requires the leading underscore for C symbols;
    /// MSVC `__cdecl` adds no decoration).
    fn platform_mangle(&self, name: &str) -> String {
        name.to_string()
    }

    /// External functions get external linkage (visibility stays default).
    pub fn set_extern_linkage(&self, func: Function) {
        func.set_linkage(Linkage::External);
    }

    /// Create an `extern` function declaration with external linkage + C-ABI
    /// attributes.
    pub fn create_extern_function<'ctx>(
        &self,
        module: &Module<'ctx>,
        name: &str,
        fn_type: Type<'ctx>,
    ) -> Function<'ctx> {
        let func = module.add_function(&self.extern_name(name), fn_type);
        func.set_linkage(Linkage::External);
        self.apply_function_attributes(func);
        func
    }
}

/// Calling-convention annotation for Jam functions.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CallConv {
    /// Default Jam calling convention (may be optimized).
    #[default]
    Jam,
    /// C calling convention (for interop).
    C,
    /// LLVM Fast calling convention.
    Fast,
    /// LLVM Cold calling convention.
    Cold,
}

/// Function attributes for C interop.
#[derive(Clone, Debug, Default)]
pub struct FunctionAbi {
    pub callconv: CallConv,
    /// External C function?
    pub is_extern: bool,
    /// Exported for C?
    pub is_export: bool,
    /// Inline hint.
    pub is_inline: bool,
    /// Never returns.
    pub no_return: bool,
    /// Custom external name.
    pub extern_name: String,
}

impl FunctionAbi {
    /// Apply linkage + calling convention + attributes to an LLVM function.
    pub fn apply_to_function(&self, func: Function, cabi: &CAbi) {
        // Linkage.
        if self.is_extern {
            cabi.set_extern_linkage(func);
        } else if self.is_export {
            func.set_linkage(Linkage::External);
        } else {
            func.set_linkage(Linkage::Internal);
        }
        // Calling convention.
        match self.callconv {
            CallConv::C => func.set_call_conv(cabi.calling_convention()),
            CallConv::Fast => func.set_call_conv(LlvmCallConv::Fast),
            CallConv::Cold => func.set_call_conv(LlvmCallConv::Cold),
            CallConv::Jam => func.set_call_conv(LlvmCallConv::C),
        }
        // Attributes.
        if self.is_inline {
            func.set_always_inline();
        }
        if self.no_return {
            func.set_no_return();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Abi, Arch, Os, Target};
    use jam_llvm::Context;

    #[test]
    fn calling_convention_per_target() {
        let linux = CAbi::new(Target::new(Arch::X86_64, Os::Linux, Abi::Gnu));
        assert_eq!(linux.calling_convention(), LlvmCallConv::C);
        let msvc = CAbi::new(Target::new(Arch::X86_64, Os::Windows, Abi::Msvc));
        assert_eq!(msvc.calling_convention(), LlvmCallConv::Win64);
        // MinGW on Windows uses the System V / C convention, not Win64.
        let mingw = CAbi::new(Target::new(Arch::X86_64, Os::Windows, Abi::MinGw));
        assert_eq!(mingw.calling_convention(), LlvmCallConv::C);
    }

    #[test]
    fn extern_name_is_passthrough() {
        let cabi = CAbi::new(Target::new(Arch::AArch64, Os::MacOs, Abi::None));
        assert_eq!(cabi.extern_name("printf"), "printf");
    }

    #[test]
    fn cabi_compatibility_by_type_kind() {
        let ctx = Context::new();
        let cabi = CAbi::new(Target::new(Arch::X86_64, Os::Linux, Abi::Gnu));
        assert!(cabi.is_cabi_compatible(ctx.i32_type()));
        assert!(cabi.is_cabi_compatible(ctx.f64_type()));
        assert!(cabi.is_cabi_compatible(ctx.void_type()));
        assert!(cabi.is_cabi_compatible(ctx.pointer_type(0)));
        // struct { i32, i64 } -> compatible
        let st = ctx.struct_type(&[ctx.i32_type(), ctx.i64_type()], false);
        assert!(cabi.is_cabi_compatible(st));
        // [4 x i32] -> compatible (element is compatible)
        assert!(cabi.is_cabi_compatible(ctx.i32_type().array_type(4)));
        // a function type is NOT C-ABI "compatible" by this predicate.
        let fnty = ctx.void_type().fn_type(&[ctx.i32_type()], false);
        assert!(!cabi.is_cabi_compatible(fnty));
    }

    #[test]
    fn create_extern_function_verifies() {
        let ctx = Context::new();
        let module = ctx.create_module("m");
        let cabi = CAbi::new(Target::new(Arch::X86_64, Os::Linux, Abi::Gnu));
        // extern fn puts(*i8) -> i32
        let fnty = ctx.i32_type().fn_type(&[ctx.pointer_type(0)], false);
        let func = cabi.create_extern_function(&module, "puts", fnty);
        // A bodyless external declaration verifies cleanly.
        assert!(func.verify());
    }

    #[test]
    fn function_abi_apply_to_internal_function_verifies() {
        let ctx = Context::new();
        let module = ctx.create_module("m");
        let cabi = CAbi::new(Target::new(Arch::X86_64, Os::Linux, Abi::Gnu));
        let fnty = ctx.void_type().fn_type(&[], false);
        let func = module.add_function("local", fnty);
        // Give it a body so an internal (non-extern) function verifies.
        let bb = func.append_basic_block("entry");
        let b = ctx.create_builder();
        b.position_at_end(bb);
        b.ret_void();
        let abi = FunctionAbi {
            callconv: CallConv::Fast,
            is_inline: true,
            ..Default::default()
        };
        abi.apply_to_function(func, &cabi);
        assert!(func.verify());
    }
}
