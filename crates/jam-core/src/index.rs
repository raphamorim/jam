/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Frozen arena-index newtypes — the shared "interface decision" the whole
//! migration depends on. Each is a zero-cost `#[repr(transparent)]` wrapper
//! around a `u32`, so the C++ "everything is a u32 alias" model ports with a
//! strict type-safety upgrade: a `NodeIdx` can never be silently used where a
//! `TypeIdx` is expected.
//!
//! **Slot 0 is the reserved sentinel** in every pool (`kNoString`, `kNoType`,
//! invalid node/decl). [`Self::NONE`] is that sentinel; pools are constructed
//! pre-seeded so index 0 is never a live entry. The C++ holds raw
//! `FunctionAST*` in `Decl`, exported-symbol tables, generic-instance keys, and
//! the drop registry; all of those migrate to [`FunctionId`] indices into an
//! owned arena.

macro_rules! idx_newtype {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(pub u32);

        impl $name {
            /// The reserved slot-0 sentinel ("none" / invalid).
            pub const NONE: Self = Self(0);

            /// Wrap a raw index.
            #[inline]
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            /// Construct from a `usize` length/position, checking the `u32`
            /// bound in debug builds (the arenas can't exceed `u32::MAX`).
            #[inline]
            pub fn from_usize(i: usize) -> Self {
                debug_assert!(i < u32::MAX as usize, "arena index overflowed u32");
                Self(i as u32)
            }

            /// The index as a `usize`, for slicing into the backing `Vec`.
            #[inline]
            pub const fn index(self) -> usize {
                self.0 as usize
            }

            /// The raw `u32`.
            #[inline]
            pub const fn raw(self) -> u32 {
                self.0
            }

            /// True for the slot-0 sentinel.
            #[inline]
            pub const fn is_none(self) -> bool {
                self.0 == 0
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

idx_newtype!(
    /// Index into the AST node arena (`NodeStore`).
    NodeIdx
);
idx_newtype!(
    /// Index into the interned type pool (`TypePool`; `kNoType` == `NONE`).
    TypeIdx
);
idx_newtype!(
    /// Index into the interned string pool (`StringPool`; `kNoString` == `NONE`).
    StringIdx
);
idx_newtype!(
    /// Index into the AST "extra" side pool (variable-length node payloads).
    ExtraIdx
);
idx_newtype!(
    /// Index into the declaration table (`DeclTable`).
    DeclIndex
);
idx_newtype!(
    /// Owned-arena id for a function AST (replaces raw `FunctionAST*`).
    FunctionId
);
idx_newtype!(
    /// Owned-arena id for a module AST (replaces raw `ModuleAST*`).
    ModuleId
);
idx_newtype!(
    /// Reference to a JIR instruction within a function (`JirRef`).
    JirRef
);
idx_newtype!(
    /// Reference to a JIR basic block within a function (`JirBlockRef`).
    JirBlockRef
);

