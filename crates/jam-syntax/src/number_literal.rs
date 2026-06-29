/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Numeric-literal validation and value extraction — ported 1:1 from
//! `src/number_literal.{h,cpp}`.
//!
//! The lexer produces a permissive `NUMBER` token; this is where its bytes are
//! validated (base prefix, underscore placement, digit ranges, exponent rules)
//! and converted to a value. Integers accumulate with C++-faithful unsigned
//! wraparound (overflow → `BigInt`). The float value, which the C++ obtained
//! via `strtod` (decimal *and* hex), is obtained here via
//! [`float128::parse_decimal_float`] — correctly rounded for both bases, and
//! without an LLVM dependency.

use float128::{ParsedFloat, parse_decimal_float, quad_to_target_as_double};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NumberBase {
    Decimal = 10,
    Hex = 16,
    Binary = 2,
    Octal = 8,
}

/// Per-error variants with the byte offset into the lexeme where the problem
/// was found.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumberErrorKind {
    LeadingZero,
    DigitAfterBase,
    UpperCaseBase,
    InvalidFloatBase,
    RepeatedUnderscore,
    InvalidUnderscoreAfterSpecial,
    InvalidDigit,
    InvalidDigitExponent,
    DuplicatePeriod,
    DuplicateExponent,
    InvalidHexExponent,
    ExponentAfterUnderscore,
    SpecialAfterUnderscore,
    TrailingSpecial,
    TrailingUnderscore,
    InvalidCharacter,
    InvalidExponentSign,
    IntegerTooLarge,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumberError {
    pub kind: NumberErrorKind,
    pub pos: usize,
    /// Meaningful for `InvalidDigit`; otherwise `Decimal`.
    pub base: NumberBase,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumberResultKind {
    Int,
    BigInt,
    Float,
    Failure,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumberResult {
    pub kind: NumberResultKind,
    pub int_value: u64,
    pub float_value: f64,
    pub base: NumberBase,
    pub failure: Option<NumberError>,
}

impl NumberResult {
    fn fail(kind: NumberErrorKind, pos: usize, base: NumberBase) -> NumberResult {
        NumberResult {
            kind: NumberResultKind::Failure,
            int_value: 0,
            float_value: 0.0,
            base: NumberBase::Decimal,
            failure: Some(NumberError { kind, pos, base }),
        }
    }
    fn ok_int(value: u64, base: NumberBase) -> NumberResult {
        NumberResult {
            kind: NumberResultKind::Int,
            int_value: value,
            float_value: 0.0,
            base,
            failure: None,
        }
    }
    fn ok_big_int(base: NumberBase) -> NumberResult {
        NumberResult {
            kind: NumberResultKind::BigInt,
            int_value: 0,
            float_value: 0.0,
            base,
            failure: None,
        }
    }
    fn ok_float(base: NumberBase, value: f64) -> NumberResult {
        NumberResult {
            kind: NumberResultKind::Float,
            int_value: 0,
            float_value: value,
            base,
            failure: None,
        }
    }
}

