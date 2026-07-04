/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Function name mangling — the single source of truth for the LLVM-level
//! symbol the linker sees. Ported from `src/mangling.h`.
//!
//! Builds a dot-separated fully-qualified name, walking up the namespace chain.
//! LLVM accepts `.` in symbol names, so no further escaping is needed.
//!
//! The rules:
//!   - `tfn t()` -> `__test_t` (the harness calls these by prefixed name).
//!   - Cloned instantiated methods carry their qualified name already
//!     (`Vec__i32.push`) — kept as-is by the caller before this runs.
//!   - Free-function `cfn drop(self: mut T)` -> `T.drop`. The receiver type
//!     qualifies it so two top-level drops for different types don't collide.
//!   - Otherwise: `[modulePath.][parentStruct.]Name`:
//!
//! ```text
//! free fn in entry module       -> name
//! free fn in module `m`         -> m.name
//! method on `T` in entry module -> T.name
//! method on `T` in module `m`   -> m.T.name
//! ```
//!
//! Centralised so every site that picks a function's LLVM symbol reads the
//! same rules — duplicating the mangling table per call site is exactly how
//! "unknown callee" miscompiles creep in. (The C++ memoizes the result on
//! `fn.mangledNameCache`; this is a pure function of its inputs, so callers
//! that want the cache can wrap it.)

use jam_core::param_mode::ParamMode;
use jam_syntax::ast::FunctionAST;
use jam_syntax::ast_flat::{StringPool, TypeKind, TypePool};

/// Translate a [`FunctionAST`] into the LLVM symbol name.
pub fn mangled_function_name(f: &FunctionAST, types: &TypePool, strings: &StringPool) -> String {
    if f.is_test {
        return format!("__test_{}", f.name);
    }
    // Free-function drop: qualify by receiver type so `cfn drop(self: mut A)`
    // and `cfn drop(self: mut B)` get distinct symbols. Only kicks in when
    // `parent_struct` is empty — in-struct `cfn drop` goes through the FQN path.
    if f.parent_struct.is_empty() && f.name == "drop" && f.args.len() == 1 {
        let p = &f.args[0];
        if p.name == "self" && p.mode == ParamMode::Mut {
            let k = types.get(p.ty);
            if k.kind == TypeKind::Struct || k.kind == TypeKind::Named {
                let ni = jam_core::index::StringIdx::new(k.a);
                if !ni.is_none() {
                    let recv = String::from_utf8_lossy(&strings.get(ni)).into_owned();
                    return format!("{recv}.drop");
                }
            }
        }
    }
    let mut out = String::new();
    if !f.module_path.is_empty() {
        out.push_str(&f.module_path);
        out.push('.');
    }
    if !f.parent_struct.is_empty() {
        out.push_str(&f.parent_struct);
        out.push('.');
    }
    out.push_str(&f.name);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jam_syntax::ast::{FunctionAST, Param};
    use jam_syntax::ast_flat::{StringPool, TypePool};

    fn empty_fn(name: &str) -> FunctionAST {
        FunctionAST::new(name, Vec::new(), jam_core::index::TypeIdx::NONE, Vec::new())
    }

    #[test]
    fn test_fn_gets_prefixed_name() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let mut f = empty_fn("roundtrip");
        f.is_test = true;
        assert_eq!(mangled_function_name(&f, &tp, &sp), "__test_roundtrip");
    }

    #[test]
    fn free_fn_entry_module_is_bare_name() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let f = empty_fn("main");
        assert_eq!(mangled_function_name(&f, &tp, &sp), "main");
    }

    #[test]
    fn free_fn_in_module_is_qualified() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let mut f = empty_fn("helper");
        f.module_path = "util".to_string();
        assert_eq!(mangled_function_name(&f, &tp, &sp), "util.helper");
    }

    #[test]
    fn method_in_module_is_fully_qualified() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let mut f = empty_fn("push");
        f.module_path = "collections".to_string();
        f.parent_struct = "Vec".to_string();
        assert_eq!(mangled_function_name(&f, &tp, &sp), "collections.Vec.push");
    }

    #[test]
    fn free_drop_is_qualified_by_receiver_type() {
        let tp = TypePool::new();
        let sp = StringPool::new();
        let recv = sp.intern(b"Counter");
        let recv_ty = tp.intern_named(recv);
        let mut f = empty_fn("drop");
        let mut p = Param::new("self", recv_ty);
        p.mode = ParamMode::Mut;
        f.args = vec![p];
        assert_eq!(mangled_function_name(&f, &tp, &sp), "Counter.drop");
    }

    #[test]
    fn in_struct_drop_uses_fqn_not_receiver_qualification() {
        // `parent_struct` set -> the free-drop special case is skipped.
        let tp = TypePool::new();
        let sp = StringPool::new();
        let recv = sp.intern(b"Counter");
        let recv_ty = tp.intern_named(recv);
        let mut f = empty_fn("drop");
        f.parent_struct = "Counter".to_string();
        let mut p = Param::new("self", recv_ty);
        p.mode = ParamMode::Mut;
        f.args = vec![p];
        assert_eq!(mangled_function_name(&f, &tp, &sp), "Counter.drop");
    }
}
