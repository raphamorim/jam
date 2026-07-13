/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam_sema` — semantic analysis and lowering for the Jam compiler.
//!
//! AstGen produces the typed flat [`jir`], codegen consumes it; ABI
//! classification and the JIR verifier live here too. Sits above `jam_core`
//! and `jam_syntax`.

pub mod abi;
pub mod analyzer;
pub mod astgen;
pub mod cabi;
pub mod codegen_context;
pub mod comptime;
pub mod decl;
pub mod drop_registry;
pub mod generics;
pub mod init_analysis;
pub mod jir;
pub mod jir_codegen;
pub mod jir_verify;
pub mod libcxx_order;
pub mod mangling;
pub mod module_resolver;
pub mod symbol_table;
pub mod target;
