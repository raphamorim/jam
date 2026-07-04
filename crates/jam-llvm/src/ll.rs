/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Safe wrappers over the LLVM-C API, mirroring the operations the C++
//! `jam_llvm` facade exposed. RAII owners ([`Context`], [`Module`], [`Builder`])
//! dispose on drop; handle types ([`Type`], [`Value`], [`Function`],
//! [`BasicBlock`]) are `Copy` and borrow `'ctx` from their owning [`Context`].

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::{c_uint, c_ulonglong};

use crate::raw;
use crate::{CallConv, IntPredicate, Linkage, RealPredicate};

// LLVM attribute index sentinels (see `LLVMAttributeIndex`).
const ATTR_RETURN_INDEX: raw::LLVMAttributeIndex = 0;
const ATTR_FUNCTION_INDEX: raw::LLVMAttributeIndex = c_uint::MAX; // == -1
fn attr_param_index(arg_idx: u32) -> raw::LLVMAttributeIndex {
    arg_idx + 1
}

// `UWTableKind::Sync` — the unwind-table kind jam's non-extern functions carry.
const UWTABLE_SYNC: u64 = 1;

/// Convert a Rust `&str` into an owned, NUL-terminated C string. Names from
/// codegen never contain interior NULs; an empty name yields `c""`.
fn cstr(s: &str) -> CString {
    CString::new(s).expect("LLVM name contained an interior NUL byte")
}

/// The `LLVMContextRef` that owns `v` — recovered via its type, so handle types
/// need not each carry a context pointer.
unsafe fn value_context(v: raw::LLVMValueRef) -> raw::LLVMContextRef {
    unsafe { raw::LLVMGetTypeContext(raw::LLVMTypeOf(v)) }
}

// ===========================================================================
// Context
// ===========================================================================

/// Owns an `LLVMContext`. Everything else borrows `'ctx` from it.
pub struct Context {
    ptr: raw::LLVMContextRef,
}

impl Context {
    /// Create a context. Value names are discarded at construction (matching the
    /// C++ facade) so `--emit-ir` prints auto-numbered temporaries.
    pub fn new() -> Context {
        unsafe {
            let ptr = raw::LLVMContextCreate();
            // Discard SSA value names: they cost memory and don't affect codegen.
            raw::LLVMContextSetDiscardValueNames(ptr, 1);
            Context { ptr }
        }
    }

    pub fn create_module(&self, name: &str) -> Module<'_> {
        let cname = cstr(name);
        unsafe {
            Module {
                ptr: raw::LLVMModuleCreateWithNameInContext(cname.as_ptr(), self.ptr),
                _ctx: PhantomData,
            }
        }
    }

    pub fn create_builder(&self) -> Builder<'_> {
        unsafe {
            Builder {
                ptr: raw::LLVMCreateBuilderInContext(self.ptr),
                _ctx: PhantomData,
            }
        }
    }

    // ---- primitive types ----
    pub fn i1_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMInt1TypeInContext(self.ptr)) }
    }
    pub fn i8_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMInt8TypeInContext(self.ptr)) }
    }
    pub fn i16_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMInt16TypeInContext(self.ptr)) }
    }
    pub fn i32_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMInt32TypeInContext(self.ptr)) }
    }
    pub fn i64_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMInt64TypeInContext(self.ptr)) }
    }
    pub fn f32_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMFloatTypeInContext(self.ptr)) }
    }
    pub fn f64_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMDoubleTypeInContext(self.ptr)) }
    }
    pub fn void_type(&self) -> Type<'_> {
        unsafe { Type::new(raw::LLVMVoidTypeInContext(self.ptr)) }
    }
    /// Opaque pointer in the given address space.
    pub fn pointer_type(&self, address_space: u32) -> Type<'_> {
        unsafe { Type::new(raw::LLVMPointerTypeInContext(self.ptr, address_space)) }
    }

    pub fn struct_type(&self, fields: &[Type<'_>], packed: bool) -> Type<'_> {
        let mut raw_fields: Vec<raw::LLVMTypeRef> = fields.iter().map(|t| t.0).collect();
        unsafe {
            Type::new(raw::LLVMStructTypeInContext(
                self.ptr,
                raw_fields.as_mut_ptr(),
                raw_fields.len() as c_uint,
                packed as raw::LLVMBool,
            ))
        }
    }

    /// An opaque named struct whose body is set later via [`Type::set_body`].
    pub fn named_struct(&self, name: &str) -> Type<'_> {
        let cname = cstr(name);
        unsafe { Type::new(raw::LLVMStructCreateNamed(self.ptr, cname.as_ptr())) }
    }

    pub fn create_basic_block(&self, name: &str) -> BasicBlock<'_> {
        let cname = cstr(name);
        unsafe { BasicBlock::new(raw::LLVMCreateBasicBlockInContext(self.ptr, cname.as_ptr())) }
    }

    // ---- constants ----
    pub fn const_string(&self, bytes: &[u8], null_terminate: bool) -> Value<'_> {
        unsafe {
            Value::new(raw::LLVMConstStringInContext(
                self.ptr,
                bytes.as_ptr() as *const _,
                bytes.len() as c_uint,
                // C API takes DontNullTerminate.
                (!null_terminate) as raw::LLVMBool,
            ))
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Context::new()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { raw::LLVMContextDispose(self.ptr) }
    }
}

// ===========================================================================
// Module
// ===========================================================================

pub struct Module<'ctx> {
    ptr: raw::LLVMModuleRef,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Module<'ctx> {
    pub(crate) fn as_ptr(&self) -> raw::LLVMModuleRef {
        self.ptr
    }

    pub fn set_target_triple(&self, triple: &str) {
        let c = cstr(triple);
        unsafe { raw::LLVMSetTarget(self.ptr, c.as_ptr()) }
    }

    pub fn get_function(&self, name: &str) -> Option<Function<'ctx>> {
        let c = cstr(name);
        unsafe {
            let f = raw::LLVMGetNamedFunction(self.ptr, c.as_ptr());
            if f.is_null() {
                None
            } else {
                Some(Function(Value::new(f)))
            }
        }
    }

    pub fn add_function(&self, name: &str, fn_type: Type<'ctx>) -> Function<'ctx> {
        let c = cstr(name);
        unsafe {
            Function(Value::new(raw::LLVMAddFunction(
                self.ptr,
                c.as_ptr(),
                fn_type.0,
            )))
        }
    }

    /// Add a private, uninitialized global of `ty`. Set constness/initializer
    /// separately via [`Value::set_global_constant`] / [`Value::set_initializer`].
    pub fn add_global(&self, ty: Type<'ctx>, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            let g = raw::LLVMAddGlobal(self.ptr, ty.0, c.as_ptr());
            raw::LLVMSetLinkage(g, raw::LLVMLinkage::Private);
            Value::new(g)
        }
    }

    /// A private, constant, NUL-terminated string global (like the C++
    /// `JamLLVMAddGlobalString`).
    pub fn add_global_string(&self, s: &[u8], name: &str) -> Value<'ctx> {
        unsafe {
            let ctx = raw::LLVMGetModuleContext(self.ptr);
            let init = raw::LLVMConstStringInContext(
                ctx,
                s.as_ptr() as *const _,
                s.len() as c_uint,
                0, // null-terminate
            );
            let ty = raw::LLVMTypeOf(init);
            let c = cstr(name);
            let g = raw::LLVMAddGlobal(self.ptr, ty, c.as_ptr());
            raw::LLVMSetGlobalConstant(g, 1);
            raw::LLVMSetLinkage(g, raw::LLVMLinkage::Private);
            raw::LLVMSetInitializer(g, init);
            Value::new(g)
        }
    }

    /// Render the module's IR to a `String` (mirrors `LLVMPrintModuleToString`).
    pub fn print_to_string(&self) -> String {
        unsafe {
            let raw_str = raw::LLVMPrintModuleToString(self.ptr);
            let s = CStr::from_ptr(raw_str).to_string_lossy().into_owned();
            raw::LLVMDisposeMessage(raw_str);
            s
        }
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { raw::LLVMDisposeModule(self.ptr) }
    }
}

// ===========================================================================
// Builder
// ===========================================================================

