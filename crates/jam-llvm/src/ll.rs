/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Safe wrappers over the LLVM-C API. RAII owners ([`Context`], [`Module`],
//! [`Builder`]) dispose on drop; handle types ([`Type`], [`Value`],
//! [`Function`], [`BasicBlock`]) are `Copy` and borrow `'ctx` from their
//! owning [`Context`].

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
    /// Create a context. Value names are discarded at construction so
    /// `--emit-ir` prints auto-numbered temporaries.
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

    /// A private, constant, NUL-terminated string global.
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

pub struct Builder<'ctx> {
    ptr: raw::LLVMBuilderRef,
    _ctx: PhantomData<&'ctx Context>,
}

macro_rules! bin_op {
    ($(#[$m:meta])* $name:ident => $raw:ident) => {
        $(#[$m])*
        pub fn $name(&self, lhs: Value<'ctx>, rhs: Value<'ctx>, name: &str) -> Value<'ctx> {
            let c = cstr(name);
            unsafe { Value::new(raw::$raw(self.ptr, lhs.0, rhs.0, c.as_ptr())) }
        }
    };
}

macro_rules! cast_op {
    ($(#[$m:meta])* $name:ident => $raw:ident) => {
        $(#[$m])*
        pub fn $name(&self, val: Value<'ctx>, dest: Type<'ctx>, name: &str) -> Value<'ctx> {
            let c = cstr(name);
            unsafe { Value::new(raw::$raw(self.ptr, val.0, dest.0, c.as_ptr())) }
        }
    };
}

impl<'ctx> Builder<'ctx> {
    pub fn position_at_end(&self, block: BasicBlock<'ctx>) {
        unsafe { raw::LLVMPositionBuilderAtEnd(self.ptr, block.0) }
    }

    pub fn insert_block(&self) -> Option<BasicBlock<'ctx>> {
        unsafe {
            let b = raw::LLVMGetInsertBlock(self.ptr);
            if b.is_null() {
                None
            } else {
                Some(BasicBlock::new(b))
            }
        }
    }

    /// Position immediately before `block`'s terminator (to insert peer-type
    /// promotions after an arm already emitted its branch). Falls back to the
    /// block end if there is no terminator yet.
    pub fn position_before_terminator(&self, block: BasicBlock<'ctx>) {
        unsafe {
            let term = raw::LLVMGetBasicBlockTerminator(block.0);
            if term.is_null() {
                raw::LLVMPositionBuilderAtEnd(self.ptr, block.0);
            } else {
                raw::LLVMPositionBuilderBefore(self.ptr, term);
            }
        }
    }

    // ---- integer / bitwise ----
    bin_op!(/// `add`
        add => LLVMBuildAdd);
    bin_op!(/// `sub`
        sub => LLVMBuildSub);
    bin_op!(/// `mul`
        mul => LLVMBuildMul);
    bin_op!(/// `udiv`
        udiv => LLVMBuildUDiv);
    bin_op!(/// `sdiv`
        sdiv => LLVMBuildSDiv);
    bin_op!(/// `urem`
        urem => LLVMBuildURem);
    bin_op!(/// `srem`
        srem => LLVMBuildSRem);
    bin_op!(/// `and`
        and => LLVMBuildAnd);
    bin_op!(/// `or`
        or => LLVMBuildOr);
    bin_op!(/// `xor`
        xor => LLVMBuildXor);
    bin_op!(/// `shl`
        shl => LLVMBuildShl);
    bin_op!(/// `lshr`
        lshr => LLVMBuildLShr);
    bin_op!(/// `ashr`
        ashr => LLVMBuildAShr);

    // ---- floating point ----
    bin_op!(/// `fadd`
        fadd => LLVMBuildFAdd);
    bin_op!(/// `fsub`
        fsub => LLVMBuildFSub);
    bin_op!(/// `fmul`
        fmul => LLVMBuildFMul);
    bin_op!(/// `fdiv`
        fdiv => LLVMBuildFDiv);
    bin_op!(/// `frem`
        frem => LLVMBuildFRem);

    pub fn fneg(&self, val: Value<'ctx>, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe { Value::new(raw::LLVMBuildFNeg(self.ptr, val.0, c.as_ptr())) }
    }

    /// `not` — bitwise complement (`BitNot`; also `LogNot` on i1).
    pub fn not(&self, val: Value<'ctx>, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe { Value::new(raw::LLVMBuildNot(self.ptr, val.0, c.as_ptr())) }
    }

    pub fn icmp(
        &self,
        pred: IntPredicate,
        lhs: Value<'ctx>,
        rhs: Value<'ctx>,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildICmp(
                self.ptr,
                pred.to_raw(),
                lhs.0,
                rhs.0,
                c.as_ptr(),
            ))
        }
    }

    pub fn fcmp(
        &self,
        pred: RealPredicate,
        lhs: Value<'ctx>,
        rhs: Value<'ctx>,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildFCmp(
                self.ptr,
                pred.to_raw(),
                lhs.0,
                rhs.0,
                c.as_ptr(),
            ))
        }
    }

    // ---- casts ----
    cast_op!(/// `zext`
        zext => LLVMBuildZExt);
    cast_op!(/// `sext`
        sext => LLVMBuildSExt);
    cast_op!(/// `bitcast`
        bitcast => LLVMBuildBitCast);
    cast_op!(/// `ptrtoint`
        ptr_to_int => LLVMBuildPtrToInt);
    cast_op!(/// `inttoptr`
        int_to_ptr => LLVMBuildIntToPtr);
    cast_op!(/// `sitofp`
        si_to_fp => LLVMBuildSIToFP);
    cast_op!(/// `uitofp`
        ui_to_fp => LLVMBuildUIToFP);
    cast_op!(/// `fpcast` (fptrunc/fpext as needed)
        fp_cast => LLVMBuildFPCast);
    cast_op!(/// `fptosi`
        fp_to_si => LLVMBuildFPToSI);
    cast_op!(/// `fptoui`
        fp_to_ui => LLVMBuildFPToUI);
    cast_op!(/// `trunc` (integer truncation)
        trunc => LLVMBuildTrunc);

    /// Width-changing integer cast, sign-aware (zext/sext/trunc as needed).
    pub fn int_cast(
        &self,
        val: Value<'ctx>,
        dest: Type<'ctx>,
        is_signed: bool,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildIntCast2(
                self.ptr,
                val.0,
                dest.0,
                is_signed as raw::LLVMBool,
                c.as_ptr(),
            ))
        }
    }

    /// Allocate `ty` on the stack, always in the function's entry block.
    ///
    /// Emitting an alloca at the current insertion point inside a loop would
    /// allocate fresh stack every iteration (allocas only release at function
    /// return) until the stack overflows. So we hoist every alloca to the
    /// entry block. `align_bytes == 0` defers to LLVM's datalayout-derived
    /// alignment.
    pub fn alloca(&self, ty: Type<'ctx>, align_bytes: u64, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            let cur = raw::LLVMGetInsertBlock(self.ptr);
            let inst = if cur.is_null() {
                raw::LLVMBuildAlloca(self.ptr, ty.0, c.as_ptr())
            } else {
                let parent = raw::LLVMGetBasicBlockParent(cur);
                let entry = raw::LLVMGetEntryBasicBlock(parent);
                if entry == cur {
                    // Common case: laying out params/locals at the top of the body.
                    raw::LLVMBuildAlloca(self.ptr, ty.0, c.as_ptr())
                } else {
                    // Hoist: position in the entry block, build, then restore.
                    // Codegen always appends at the end of `cur`, so restoring
                    // with PositionBuilderAtEnd is exact.
                    let first = raw::LLVMGetFirstInstruction(entry);
                    if first.is_null() {
                        raw::LLVMPositionBuilderAtEnd(self.ptr, entry);
                    } else {
                        raw::LLVMPositionBuilderBefore(self.ptr, first);
                    }
                    let inst = raw::LLVMBuildAlloca(self.ptr, ty.0, c.as_ptr());
                    raw::LLVMPositionBuilderAtEnd(self.ptr, cur);
                    inst
                }
            };
            if align_bytes != 0 {
                raw::LLVMSetAlignment(inst, align_bytes as c_uint);
            }
            Value::new(inst)
        }
    }

    pub fn load(&self, ty: Type<'ctx>, ptr: Value<'ctx>, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe { Value::new(raw::LLVMBuildLoad2(self.ptr, ty.0, ptr.0, c.as_ptr())) }
    }

    pub fn store(&self, val: Value<'ctx>, ptr: Value<'ctx>) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildStore(self.ptr, val.0, ptr.0)) }
    }

    /// `@llvm.memcpy` of `size` bytes. The C API takes the size as a value, so
    /// we materialize an i64 constant.
    pub fn memcpy(
        &self,
        dst: Value<'ctx>,
        dst_align: u64,
        src: Value<'ctx>,
        src_align: u64,
        size: u64,
    ) -> Value<'ctx> {
        unsafe {
            let i64ty = raw::LLVMInt64TypeInContext(value_context(dst.0));
            let size_v = raw::LLVMConstInt(i64ty, size as c_ulonglong, 0);
            Value::new(raw::LLVMBuildMemCpy(
                self.ptr,
                dst.0,
                dst_align as c_uint,
                src.0,
                src_align as c_uint,
                size_v,
            ))
        }
    }

    pub fn memset(
        &self,
        dst: Value<'ctx>,
        value: Value<'ctx>,
        size: u64,
        align: u64,
    ) -> Value<'ctx> {
        unsafe {
            let i64ty = raw::LLVMInt64TypeInContext(value_context(dst.0));
            let len_v = raw::LLVMConstInt(i64ty, size as c_ulonglong, 0);
            Value::new(raw::LLVMBuildMemSet(
                self.ptr,
                dst.0,
                value.0,
                len_v,
                align as c_uint,
            ))
        }
    }

    /// In-bounds GEP into a fixed array: `gep [N x T], ptr, 0, idx`.
    pub fn array_gep(
        &self,
        array_ty: Type<'ctx>,
        ptr: Value<'ctx>,
        idx: Value<'ctx>,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            let i32ty = raw::LLVMInt32TypeInContext(value_context(ptr.0));
            let zero = raw::LLVMConstInt(i32ty, 0, 0);
            let mut indices = [zero, idx.0];
            Value::new(raw::LLVMBuildInBoundsGEP2(
                self.ptr,
                array_ty.0,
                ptr.0,
                indices.as_mut_ptr(),
                2,
                c.as_ptr(),
            ))
        }
    }

    /// Struct field GEP: pointer to field `field_idx` of `*struct_ty`.
    pub fn struct_gep(
        &self,
        struct_ty: Type<'ctx>,
        ptr: Value<'ctx>,
        field_idx: u32,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildStructGEP2(
                self.ptr,
                struct_ty.0,
                ptr.0,
                field_idx,
                c.as_ptr(),
            ))
        }
    }

    /// Single-index `gep elem_ty, ptr, idx` for many-item pointer stepping.
    pub fn ptr_gep(
        &self,
        elem_ty: Type<'ctx>,
        ptr: Value<'ctx>,
        idx: Value<'ctx>,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            let mut indices = [idx.0];
            Value::new(raw::LLVMBuildInBoundsGEP2(
                self.ptr,
                elem_ty.0,
                ptr.0,
                indices.as_mut_ptr(),
                1,
                c.as_ptr(),
            ))
        }
    }

    // ---- control flow ----
    pub fn br(&self, dest: BasicBlock<'ctx>) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildBr(self.ptr, dest.0)) }
    }

    pub fn cond_br(
        &self,
        cond: Value<'ctx>,
        then_bb: BasicBlock<'ctx>,
        else_bb: BasicBlock<'ctx>,
    ) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildCondBr(self.ptr, cond.0, then_bb.0, else_bb.0)) }
    }

    pub fn switch(
        &self,
        scrutinee: Value<'ctx>,
        default_bb: BasicBlock<'ctx>,
        num_cases_hint: u32,
    ) -> Value<'ctx> {
        unsafe {
            Value::new(raw::LLVMBuildSwitch(
                self.ptr,
                scrutinee.0,
                default_bb.0,
                num_cases_hint,
            ))
        }
    }

    pub fn ret(&self, val: Value<'ctx>) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildRet(self.ptr, val.0)) }
    }
    pub fn ret_void(&self) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildRetVoid(self.ptr)) }
    }
    pub fn unreachable(&self) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMBuildUnreachable(self.ptr)) }
    }

    /// Direct call to a function value (the function type is read from `func`).
    pub fn call(&self, func: Function<'ctx>, args: &[Value<'ctx>], name: &str) -> Value<'ctx> {
        let c = cstr(name);
        let mut raw_args: Vec<raw::LLVMValueRef> = args.iter().map(|v| v.0).collect();
        unsafe {
            let fn_ty = raw::LLVMGlobalGetValueType(func.0.0);
            Value::new(raw::LLVMBuildCall2(
                self.ptr,
                fn_ty,
                func.0.0,
                raw_args.as_mut_ptr(),
                raw_args.len() as c_uint,
                c.as_ptr(),
            ))
        }
    }

    /// Indirect call through a function-typed value; `fn_ty` is the explicit
    /// `FunctionType` built from the callee's Jam-level signature.
    pub fn indirect_call(
        &self,
        fn_ty: Type<'ctx>,
        callee: Value<'ctx>,
        args: &[Value<'ctx>],
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        let mut raw_args: Vec<raw::LLVMValueRef> = args.iter().map(|v| v.0).collect();
        unsafe {
            Value::new(raw::LLVMBuildCall2(
                self.ptr,
                fn_ty.0,
                callee.0,
                raw_args.as_mut_ptr(),
                raw_args.len() as c_uint,
                c.as_ptr(),
            ))
        }
    }

    pub fn phi(&self, ty: Type<'ctx>, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe { Value::new(raw::LLVMBuildPhi(self.ptr, ty.0, c.as_ptr())) }
    }

    /// A string global plus a pointer to its first element.
    pub fn global_string_ptr(&self, s: &str, name: &str) -> Value<'ctx> {
        let cs = cstr(s);
        let cn = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildGlobalStringPtr(
                self.ptr,
                cs.as_ptr(),
                cn.as_ptr(),
            ))
        }
    }

    pub fn insert_value(
        &self,
        agg: Value<'ctx>,
        elt: Value<'ctx>,
        index: u32,
        name: &str,
    ) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildInsertValue(
                self.ptr,
                agg.0,
                elt.0,
                index,
                c.as_ptr(),
            ))
        }
    }

    pub fn extract_value(&self, agg: Value<'ctx>, index: u32, name: &str) -> Value<'ctx> {
        let c = cstr(name);
        unsafe {
            Value::new(raw::LLVMBuildExtractValue(
                self.ptr,
                agg.0,
                index,
                c.as_ptr(),
            ))
        }
    }
}

impl Drop for Builder<'_> {
    fn drop(&mut self) {
        unsafe { raw::LLVMDisposeBuilder(self.ptr) }
    }
}

// ===========================================================================
// Type
// ===========================================================================

// LLVM interns types, so pointer identity is type identity — `PartialEq`/`Eq`
// compare the underlying `LLVMTypeRef`, and `Debug` prints it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Type<'ctx>(raw::LLVMTypeRef, PhantomData<&'ctx Context>);

impl<'ctx> Type<'ctx> {
    unsafe fn new(p: raw::LLVMTypeRef) -> Type<'ctx> {
        Type(p, PhantomData)
    }

    /// `[count x self]`.
    pub fn array_type(self, count: u64) -> Type<'ctx> {
        unsafe { Type::new(raw::LLVMArrayType2(self.0, count)) }
    }

    /// A function type returning `self` with the given parameter types.
    pub fn fn_type(self, params: &[Type<'ctx>], is_var_arg: bool) -> Type<'ctx> {
        let mut raw_params: Vec<raw::LLVMTypeRef> = params.iter().map(|t| t.0).collect();
        unsafe {
            Type::new(raw::LLVMFunctionType(
                self.0,
                raw_params.as_mut_ptr(),
                raw_params.len() as c_uint,
                is_var_arg as raw::LLVMBool,
            ))
        }
    }

    /// Set the body of an opaque named struct created by [`Context::named_struct`].
    pub fn set_body(self, fields: &[Type<'ctx>], packed: bool) {
        let mut raw_fields: Vec<raw::LLVMTypeRef> = fields.iter().map(|t| t.0).collect();
        unsafe {
            raw::LLVMStructSetBody(
                self.0,
                raw_fields.as_mut_ptr(),
                raw_fields.len() as c_uint,
                packed as raw::LLVMBool,
            )
        }
    }

    fn kind(self) -> raw::LLVMTypeKind {
        unsafe { raw::LLVMGetTypeKind(self.0) }
    }
    pub fn is_void(self) -> bool {
        self.kind() == raw::LLVMTypeKind::Void
    }
    pub fn is_struct(self) -> bool {
        self.kind() == raw::LLVMTypeKind::Struct
    }
    pub fn is_integer(self) -> bool {
        self.kind() == raw::LLVMTypeKind::Integer
    }
    pub fn is_float(self) -> bool {
        matches!(
            self.kind(),
            raw::LLVMTypeKind::Half
                | raw::LLVMTypeKind::Float
                | raw::LLVMTypeKind::Double
                | raw::LLVMTypeKind::X86_FP80
                | raw::LLVMTypeKind::FP128
                | raw::LLVMTypeKind::PPC_FP128
                | raw::LLVMTypeKind::BFloat
        )
    }
    pub fn is_pointer(self) -> bool {
        self.kind() == raw::LLVMTypeKind::Pointer
    }
    pub fn is_array(self) -> bool {
        self.kind() == raw::LLVMTypeKind::Array
    }

    /// Bit width of an integer type.
    pub fn int_width(self) -> u32 {
        unsafe { raw::LLVMGetIntTypeWidth(self.0) }
    }

    /// Element type of an array type (`[N x T] -> T`).
    pub fn array_element_type(self) -> Type<'ctx> {
        unsafe { Type::new(raw::LLVMGetElementType(self.0)) }
    }

    // ---- constants of this type ----
    pub fn const_int(self, val: u64, sign_extend: bool) -> Value<'ctx> {
        unsafe {
            Value::new(raw::LLVMConstInt(
                self.0,
                val as c_ulonglong,
                sign_extend as raw::LLVMBool,
            ))
        }
    }
    pub fn const_real(self, val: f64) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMConstReal(self.0, val)) }
    }
    pub fn const_null(self) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMConstNull(self.0)) }
    }
    pub fn undef(self) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMGetUndef(self.0)) }
    }

    pub(crate) fn as_ptr(self) -> raw::LLVMTypeRef {
        self.0
    }
}

// ===========================================================================
// Value
// ===========================================================================

#[derive(Copy, Clone)]
pub struct Value<'ctx>(raw::LLVMValueRef, PhantomData<&'ctx Context>);

impl<'ctx> Value<'ctx> {
    unsafe fn new(p: raw::LLVMValueRef) -> Value<'ctx> {
        Value(p, PhantomData)
    }

    pub fn type_of(self) -> Type<'ctx> {
        unsafe { Type::new(raw::LLVMTypeOf(self.0)) }
    }

    /// The allocated type of an `alloca` value.
    pub fn allocated_type(self) -> Type<'ctx> {
        unsafe { Type::new(raw::LLVMGetAllocatedType(self.0)) }
    }

    pub fn set_name(self, name: &str) {
        unsafe { raw::LLVMSetValueName2(self.0, name.as_ptr() as *const _, name.len()) }
    }

    pub fn set_global_constant(self, is_constant: bool) {
        unsafe { raw::LLVMSetGlobalConstant(self.0, is_constant as raw::LLVMBool) }
    }

    pub fn set_initializer(self, constant: Value<'ctx>) {
        unsafe { raw::LLVMSetInitializer(self.0, constant.0) }
    }

    pub fn set_linkage(self, linkage: Linkage) {
        unsafe { raw::LLVMSetLinkage(self.0, linkage.to_raw()) }
    }

    /// Add `(values[i], blocks[i])` incoming edges to a phi node.
    pub fn add_incoming(self, values: &[Value<'ctx>], blocks: &[BasicBlock<'ctx>]) {
        debug_assert_eq!(values.len(), blocks.len());
        let mut raw_vals: Vec<raw::LLVMValueRef> = values.iter().map(|v| v.0).collect();
        let mut raw_bbs: Vec<raw::LLVMBasicBlockRef> = blocks.iter().map(|b| b.0).collect();
        unsafe {
            raw::LLVMAddIncoming(
                self.0,
                raw_vals.as_mut_ptr(),
                raw_bbs.as_mut_ptr(),
                raw_vals.len() as c_uint,
            )
        }
    }

    /// Add a `case`: `on_val => dest` to a `switch` instruction.
    pub fn add_case(self, on_val: Value<'ctx>, dest: BasicBlock<'ctx>) {
        unsafe { raw::LLVMAddCase(self.0, on_val.0, dest.0) }
    }
}

// ===========================================================================
// Function (a Value known to be a function)
// ===========================================================================

#[derive(Copy, Clone)]
pub struct Function<'ctx>(Value<'ctx>);

impl<'ctx> Function<'ctx> {
    pub fn as_value(self) -> Value<'ctx> {
        self.0
    }

    pub fn set_call_conv(self, cc: CallConv) {
        unsafe { raw::LLVMSetFunctionCallConv(self.0.0, cc.to_raw()) }
    }

    pub fn set_linkage(self, linkage: Linkage) {
        self.0.set_linkage(linkage)
    }

    pub fn count_params(self) -> u32 {
        unsafe { raw::LLVMCountParams(self.0.0) }
    }

    pub fn param(self, index: u32) -> Value<'ctx> {
        unsafe { Value::new(raw::LLVMGetParam(self.0.0, index)) }
    }

    pub fn return_type(self) -> Type<'ctx> {
        unsafe {
            Type::new(raw::LLVMGetReturnType(raw::LLVMGlobalGetValueType(
                self.0.0,
            )))
        }
    }

    pub fn is_var_arg(self) -> bool {
        unsafe { raw::LLVMIsFunctionVarArg(raw::LLVMGlobalGetValueType(self.0.0)) != 0 }
    }

    /// Append a fresh basic block to this function.
    pub fn append_basic_block(self, name: &str) -> BasicBlock<'ctx> {
        let c = cstr(name);
        unsafe {
            let ctx = value_context(self.0.0);
            BasicBlock::new(raw::LLVMAppendBasicBlockInContext(
                ctx,
                self.0.0,
                c.as_ptr(),
            ))
        }
    }

    /// Run LLVM's function verifier; returns `true` when the function is
    /// valid. Failures are printed to stderr.
    pub fn verify(self) -> bool {
        unsafe {
            raw::LLVMVerifyFunction(self.0.0, raw::LLVMVerifierFailureAction::PrintMessage) == 0
        }
    }

    // ---- attributes ----

    fn add_enum_attr(self, index: raw::LLVMAttributeIndex, name: &str, val: u64) {
        unsafe {
            let ctx = value_context(self.0.0);
            let kind = raw::LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
            let attr = raw::LLVMCreateEnumAttribute(ctx, kind, val);
            raw::LLVMAddAttributeAtIndex(self.0.0, index, attr);
        }
    }

    fn add_string_attr(self, index: raw::LLVMAttributeIndex, key: &str, value: &str) {
        unsafe {
            let ctx = value_context(self.0.0);
            let attr = raw::LLVMCreateStringAttribute(
                ctx,
                key.as_ptr() as *const _,
                key.len() as c_uint,
                value.as_ptr() as *const _,
                value.len() as c_uint,
            );
            raw::LLVMAddAttributeAtIndex(self.0.0, index, attr);
        }
    }

    /// `zeroext` on parameter `arg_idx` (C ABI for i1 bool).
    pub fn add_param_attr_zeroext(self, arg_idx: u32) {
        self.add_enum_attr(attr_param_index(arg_idx), "zeroext", 0);
    }

    /// `zeroext` on the return slot.
    pub fn add_ret_attr_zeroext(self) {
        self.add_enum_attr(ATTR_RETURN_INDEX, "zeroext", 0);
    }

    pub fn set_no_return(self) {
        self.add_enum_attr(ATTR_FUNCTION_INDEX, "noreturn", 0);
    }

    /// `alwaysinline` function attribute (the `inline` hint for C interop).
    pub fn set_always_inline(self) {
        self.add_enum_attr(ATTR_FUNCTION_INDEX, "alwaysinline", 0);
    }

    /// Mark parameter `arg_idx` as the `sret(<pointee>) align <a> noalias` slot.
    pub fn add_param_attr_sret(self, arg_idx: u32, pointee: Type<'ctx>, align: u32) {
        let idx = attr_param_index(arg_idx);
        unsafe {
            let ctx = value_context(self.0.0);
            let sret_kind = raw::LLVMGetEnumAttributeKindForName("sret".as_ptr() as *const _, 4);
            let sret = raw::LLVMCreateTypeAttribute(ctx, sret_kind, pointee.as_ptr());
            raw::LLVMAddAttributeAtIndex(self.0.0, idx, sret);
        }
        self.add_enum_attr(idx, "noalias", 0);
        self.add_enum_attr(idx, "align", align as u64);
    }

    /// True if parameter 0 carries `sret`.
    pub fn uses_sret(self) -> bool {
        if self.count_params() == 0 {
            return false;
        }
        unsafe {
            let kind = raw::LLVMGetEnumAttributeKindForName("sret".as_ptr() as *const _, 4);
            !raw::LLVMGetEnumAttributeAtIndex(self.0.0, attr_param_index(0), kind).is_null()
        }
    }

    /// The pointee type of parameter 0's `sret`, if any.
    pub fn sret_pointee_type(self) -> Option<Type<'ctx>> {
        if self.count_params() == 0 {
            return None;
        }
        unsafe {
            let kind = raw::LLVMGetEnumAttributeKindForName("sret".as_ptr() as *const _, 4);
            let attr = raw::LLVMGetEnumAttributeAtIndex(self.0.0, attr_param_index(0), kind);
            if attr.is_null() {
                None
            } else {
                Some(Type::new(raw::LLVMGetTypeAttributeValue(attr)))
            }
        }
    }

    /// Default function attributes jam stamps on every definition. Non-extern
    /// functions also get `nounwind` + a sync unwind table; all get a frame
    /// pointer and the host CPU/feature strings.
    pub fn apply_default_attrs(self, is_extern: bool) {
        if !is_extern {
            self.add_enum_attr(ATTR_FUNCTION_INDEX, "nounwind", 0);
            self.add_enum_attr(ATTR_FUNCTION_INDEX, "uwtable", UWTABLE_SYNC);
        }
        self.add_string_attr(ATTR_FUNCTION_INDEX, "frame-pointer", "all");
        let cpu = crate::target::host_cpu_name();
        if !cpu.is_empty() {
            self.add_string_attr(ATTR_FUNCTION_INDEX, "target-cpu", &cpu);
        }
        let feats = crate::target::host_cpu_features();
        if !feats.is_empty() {
            self.add_string_attr(ATTR_FUNCTION_INDEX, "target-features", &feats);
        }
    }

    /// View the function as a generic `Value` pointer (for `ptrtoint` of a
    /// function address, etc.).
    pub fn as_value_ptr(self) -> Value<'ctx> {
        self.0
    }

    pub(crate) fn raw_value(self) -> raw::LLVMValueRef {
        self.0.0
    }
}

// ===========================================================================
// BasicBlock
// ===========================================================================

#[derive(Copy, Clone)]
pub struct BasicBlock<'ctx>(raw::LLVMBasicBlockRef, PhantomData<&'ctx Context>);

impl<'ctx> BasicBlock<'ctx> {
    unsafe fn new(p: raw::LLVMBasicBlockRef) -> BasicBlock<'ctx> {
        BasicBlock(p, PhantomData)
    }

    pub fn parent(self) -> Function<'ctx> {
        unsafe { Function(Value::new(raw::LLVMGetBasicBlockParent(self.0))) }
    }

    /// The block's terminator instruction, if it has one.
    pub fn terminator(self) -> Option<Value<'ctx>> {
        unsafe {
            let t = raw::LLVMGetBasicBlockTerminator(self.0);
            if t.is_null() {
                None
            } else {
                Some(Value::new(t))
            }
        }
    }
}
