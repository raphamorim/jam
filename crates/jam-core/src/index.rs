/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Arena-index newtypes: zero-cost `#[repr(transparent)]` wrappers around
//! `u32`, so a `NodeIdx` can never be silently used where a `TypeIdx` is
//! expected.
//!
//! **Slot 0 is the reserved sentinel** in every pool. `NONE` is that sentinel;
//! pools are constructed pre-seeded so index 0 is never a live entry.

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
    /// Index into the interned type pool (`TypePool`).
    TypeIdx
);
idx_newtype!(
    /// Index into the interned string pool (`StringPool`).
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
    /// Owned-arena id for a function AST.
    FunctionId
);
idx_newtype!(
    /// Owned-arena id for a module AST.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_and_roundtrip() {
        assert!(NodeIdx::NONE.is_none());
        assert_eq!(NodeIdx::NONE.raw(), 0);
        let n = NodeIdx::from_usize(7);
        assert_eq!(n.index(), 7);
        assert_eq!(n.raw(), 7);
        assert!(!n.is_none());
        assert_eq!(NodeIdx::new(7), n);
    }

    #[test]
    fn distinct_types_dont_mix() {
        // Compile-time guarantee: these are different types. This test just
        // exercises ordering/hashing used by interning maps.
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(TypeIdx::new(1));
        s.insert(TypeIdx::new(1));
        s.insert(TypeIdx::new(2));
        assert_eq!(s.len(), 2);
        assert!(TypeIdx::new(1) < TypeIdx::new(2));
    }

    #[test]
    fn debug_format() {
        assert_eq!(format!("{:?}", DeclIndex::new(3)), "DeclIndex(3)");
    }

    #[test]
    fn transparent_layout() {
        assert_eq!(std::mem::size_of::<TypeIdx>(), std::mem::size_of::<u32>());
    }
}
