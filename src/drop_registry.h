/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef DROP_REGISTRY_H
#define DROP_REGISTRY_H

#include <string>
#include <unordered_map>

// Forward declarations — drop_registry intentionally has a slim include
// surface so codegen can pull it in without dragging in the analyzer.
class FunctionAST;
class ModuleAST;
class TypePool;
class StringPool;

namespace jam {
namespace drops {

// Map of struct name -> user-defined drop function for that type.
//
// A type T is "drop-bearing" when the program declares
//
//     fn drop(self: mut T) { ... }
//
// at module scope. The registry stores borrowed pointers; the FunctionAST
// objects live on the ModuleAST and outlive any registry derived from it.
//
// Two consumers:
//   - The init analyzer uses the registry to reject
//     `move` on drop-bearing bindings until move-aware drop tracking
//     lands.
//   - The codegen walks the registry to emit drop calls at
//     scope exit for in-scope bindings of drop-bearing types.
//
// See docs/MVS.md §6 for the full drop semantics.
using DropRegistry = std::unordered_map<std::string, const FunctionAST *>;

// Scan a module's functions for any named `drop` whose signature is
// exactly `fn drop(self: mut T)` for some struct or named type T, and
// produce the corresponding DropRegistry. Functions named `drop` with
// any other signature are silently ignored at this phase; a stricter
// validity check can land later.
DropRegistry buildDropRegistry(const ModuleAST &module, const TypePool &types,
                               const StringPool &strings);

// Add `module`'s drop fns to an existing registry. Used to fold imported
// modules' drops in, so a drop site in one module fires the destructor of a
// type defined (and imported) from another (e.g. `Bus.drop` in bus.jam,
// dropped in main.jam).
void addDropCandidates(DropRegistry &registry, const ModuleAST &module,
                       const TypePool &types, const StringPool &strings);

// Map of struct name -> user-defined `cfn clone(self: T) T`. The clone
// counterpart of DropRegistry: a type with its own cfn drop must
// provide a cfn clone to be cloneable (the compiler can clone structure
// but not resources — see CLONE_PLAN.md). Top-level form mirrors the
// top-level drop form; in-struct clones also register here so lookup is
// uniform.
using CloneRegistry = std::unordered_map<std::string, const FunctionAST *>;

void addCloneCandidates(CloneRegistry &registry, const ModuleAST &module,
                        const TypePool &types, const StringPool &strings);

}  // namespace drops
}  // namespace jam

#endif  // DROP_REGISTRY_H
