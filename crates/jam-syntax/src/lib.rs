/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! `jam_syntax` — the Jam frontend: tokens, lexer, number-literal parsing,
//! the flat AST arenas, and the parser.

pub mod ast;
pub mod ast_flat;
pub mod lexer;
pub mod number_literal;
pub mod parser;
pub mod token;

pub use lexer::{LexError, Lexer};
pub use number_literal::{
    NumberBase, NumberError, NumberErrorKind, NumberResult, NumberResultKind, parse_number_literal,
};
pub use token::{Token, TokenType};
