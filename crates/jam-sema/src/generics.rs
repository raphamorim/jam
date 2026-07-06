/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Generic substitution primitives — the foundation of the monomorphization
//! engine. Jam generics are type-returning comptime functions
//! (`fn Pair(T: type) type { return struct { x: T } }`); a use `Pair(u64)` is a
//! `GenericCall` TypeIdx that resolves by binding the parameter names to the
//! concrete arguments and substituting them through the returned struct/enum
//! body. [`substitute_type`] is the recursive type rewrite at the core of that
//! (the C++ `substituteType`); `resolveGenericCall` / `instantiateStructExpr`
//! build on it and land next (see the migration memory for the full blueprint).

use std::collections::HashMap;

use jam_core::index::{StringIdx, TypeIdx};
use jam_syntax::ast_flat::{StringPool, TypeKind, TypePool};

/// Rewrite `ty`, replacing each `Named` leaf that names a generic parameter in
/// `subst` with its concrete argument. Recurses through pointer / slice / array
/// element types and `GenericCall` arguments, re-interning rewritten compound
/// types; non-parameter and scalar types pass through unchanged.
pub fn substitute_type(
    ty: TypeIdx,
    subst: &HashMap<String, TypeIdx>,
    tp: &TypePool,
    sp: &StringPool,
) -> TypeIdx {
    if subst.is_empty() {
        return ty;
    }
    let (kind, a, b) = {
        let k = tp.get(ty);
        (k.kind, k.a, k.b)
    };
    match kind {
        TypeKind::Named => {
            let name = String::from_utf8_lossy(&sp.get(StringIdx::new(a))).into_owned();
            subst.get(&name).copied().unwrap_or(ty)
        }
        TypeKind::PtrSingle => {
            let e = substitute_type(TypeIdx::new(a), subst, tp, sp);
            if e == TypeIdx::new(a) {
                ty
            } else {
                tp.intern_ptr_single(e)
            }
        }
        TypeKind::PtrMany => {
            let e = substitute_type(TypeIdx::new(a), subst, tp, sp);
            if e == TypeIdx::new(a) {
                ty
            } else {
                tp.intern_ptr_many(e)
            }
        }
        TypeKind::Slice => {
            let e = substitute_type(TypeIdx::new(a), subst, tp, sp);
            if e == TypeIdx::new(a) {
                ty
            } else {
                tp.intern_slice(e)
            }
        }
        TypeKind::Array => {
            let e = substitute_type(TypeIdx::new(a), subst, tp, sp);
            if e == TypeIdx::new(a) {
                ty
            } else {
                tp.intern_array(e, b)
            }
        }
        TypeKind::GenericCall => {
            let args: Vec<TypeIdx> = tp.generic_args_at(b).to_vec();
            let mut new_args = Vec::with_capacity(args.len());
            let mut changed = false;
            for arg in args {
                let s = substitute_type(arg, subst, tp, sp);
                if s != arg {
                    changed = true;
                }
                new_args.push(s);
            }
            if changed {
                tp.intern_generic_call(StringIdx::new(a), new_args)
            } else {
                ty
            }
        }
        TypeKind::Fn => {
            // `a` = return TypeIdx, `b` = fn-params index.
            let ret = TypeIdx::new(a);
            let new_ret = substitute_type(ret, subst, tp, sp);
            let params: Vec<TypeIdx> = tp.fn_params_at(b).to_vec();
            let mut new_params = Vec::with_capacity(params.len());
            let mut changed = new_ret != ret;
            for p in params {
                let s = substitute_type(p, subst, tp, sp);
                if s != p {
                    changed = true;
                }
                new_params.push(s);
            }
            if changed {
                tp.intern_fn(new_ret, new_params)
            } else {
                ty
            }
        }
        _ => ty,
    }
}

/// Spell a concrete type argument for an instantiated generic's mangled name
/// (the C++ `genericArgSpelling`): `Maybe(File)` → `Maybe__File`, `Vec(*const
/// u8)` → `Vec__p_u8`, `[4]u8` arg → `arr4_u8`. Every distinct argument spells
/// distinctly so instantiations never collide; the fallback embeds the TypeIdx.
pub fn generic_arg_spelling(ty: TypeIdx, tp: &TypePool, sp: &StringPool) -> String {
    let k = tp.get(ty);
    match k.kind {
        TypeKind::Int => format!("{}{}", if k.b != 0 { 'i' } else { 'u' }, k.a),
        TypeKind::Bool => "bool".to_string(),
        TypeKind::Float => if k.a == 32 { "f32" } else { "f64" }.to_string(),
        TypeKind::Struct | TypeKind::Named | TypeKind::Enum | TypeKind::Union => {
            String::from_utf8_lossy(&sp.get(StringIdx::new(k.a))).into_owned()
        }
        TypeKind::PtrSingle => format!("p_{}", generic_arg_spelling(TypeIdx::new(k.a), tp, sp)),
        TypeKind::PtrMany => format!("pm_{}", generic_arg_spelling(TypeIdx::new(k.a), tp, sp)),
        TypeKind::Slice => format!("sl_{}", generic_arg_spelling(TypeIdx::new(k.a), tp, sp)),
        TypeKind::Array => {
            format!(
                "arr{}_{}",
                k.b,
                generic_arg_spelling(TypeIdx::new(k.a), tp, sp)
            )
        }
        _ => format!("T{}", ty.index()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jam_syntax::ast_flat::builtin;

    fn subst_of(pairs: &[(&str, TypeIdx)], sp: &mut StringPool) -> HashMap<String, TypeIdx> {
        // Interning the names into `sp` isn't required (the map is keyed by the
        // String), but mirrors how the resolver builds the frame.
        let _ = sp;
        pairs.iter().map(|(n, t)| (n.to_string(), *t)).collect()
    }

    #[test]
    fn bare_param_substitutes() {
        let tp = TypePool::new();
        let mut sp = StringPool::new();
        let t = tp.intern_named(sp.intern(b"T"));
        let subst = subst_of(&[("T", builtin::I32)], &mut sp);
        assert_eq!(substitute_type(t, &subst, &tp, &sp), builtin::I32);
    }

    #[test]
    fn non_param_named_passes_through() {
        let tp = TypePool::new();
        let mut sp = StringPool::new();
        let counter = tp.intern_named(sp.intern(b"Counter"));
        let subst = subst_of(&[("T", builtin::I32)], &mut sp);
        assert_eq!(substitute_type(counter, &subst, &tp, &sp), counter);
    }

    #[test]
    fn pointer_element_substitutes() {
        let tp = TypePool::new();
        let mut sp = StringPool::new();
        let t = tp.intern_named(sp.intern(b"T"));
        let ptr_t = tp.intern_ptr_single(t);
        let subst = subst_of(&[("T", builtin::U64)], &mut sp);
        let want = tp.intern_ptr_single(builtin::U64);
        assert_eq!(substitute_type(ptr_t, &subst, &tp, &sp), want);
    }

    #[test]
    fn array_element_substitutes() {
        let tp = TypePool::new();
        let mut sp = StringPool::new();
        let t = tp.intern_named(sp.intern(b"T"));
        let arr_t = tp.intern_array(t, 4);
        let subst = subst_of(&[("T", builtin::U8)], &mut sp);
        let want = tp.intern_array(builtin::U8, 4);
        assert_eq!(substitute_type(arr_t, &subst, &tp, &sp), want);
    }

    #[test]
    fn nested_generic_call_args_substitute() {
        // `Vec(T)` with T->i32 becomes `Vec(i32)`.
        let tp = TypePool::new();
        let mut sp = StringPool::new();
        let vec_name = sp.intern(b"Vec");
        let t = tp.intern_named(sp.intern(b"T"));
        let vec_t = tp.intern_generic_call(vec_name, vec![t]);
        let subst = subst_of(&[("T", builtin::I32)], &mut sp);
        let want = tp.intern_generic_call(vec_name, vec![builtin::I32]);
        assert_eq!(substitute_type(vec_t, &subst, &tp, &sp), want);
    }

    #[test]
    fn empty_subst_is_identity() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let t = tp.intern_named(sp.intern(b"T"));
        let subst: HashMap<String, TypeIdx> = HashMap::new();
        assert_eq!(substitute_type(t, &subst, &tp, &sp), t);
    }

    #[test]
    fn arg_spelling_scalars_and_compounds() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        assert_eq!(generic_arg_spelling(builtin::I32, &tp, &sp), "i32");
        assert_eq!(generic_arg_spelling(builtin::U64, &tp, &sp), "u64");
        assert_eq!(generic_arg_spelling(builtin::BOOL, &tp, &sp), "bool");
        assert_eq!(generic_arg_spelling(builtin::F32, &tp, &sp), "f32");
        let file = tp.intern_named(sp.intern(b"File"));
        assert_eq!(generic_arg_spelling(file, &tp, &sp), "File");
        let p_u8 = tp.intern_ptr_single(builtin::U8);
        assert_eq!(generic_arg_spelling(p_u8, &tp, &sp), "p_u8");
        let arr = tp.intern_array(builtin::U8, 4);
        assert_eq!(generic_arg_spelling(arr, &tp, &sp), "arr4_u8");
    }
}
