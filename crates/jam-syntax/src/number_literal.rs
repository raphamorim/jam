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

/// Validate and parse a numeric literal lexeme (raw token bytes).
pub fn parse_number_literal(bytes: &[u8]) -> NumberResult {
    let mut i = 0usize;
    let mut base_value: u8 = 10;
    let mut base = NumberBase::Decimal;

    if bytes.len() >= 2 && bytes[0] == b'0' {
        match bytes[1] {
            b'b' => {
                base = NumberBase::Binary;
                base_value = 2;
                i = 2;
            }
            b'o' => {
                base = NumberBase::Octal;
                base_value = 8;
                i = 2;
            }
            b'x' => {
                base = NumberBase::Hex;
                base_value = 16;
                i = 2;
            }
            b'B' | b'O' | b'X' => {
                return NumberResult::fail(NumberErrorKind::UpperCaseBase, 1, NumberBase::Decimal);
            }
            // "0." / "0e" / "0E" — decimal float; fall through to the scanner.
            b'.' | b'e' | b'E' => {}
            _ => {
                return NumberResult::fail(NumberErrorKind::LeadingZero, 0, NumberBase::Decimal);
            }
        }
    }

    if bytes.len() == 2 && base_value != 10 {
        return NumberResult::fail(NumberErrorKind::DigitAfterBase, 2, NumberBase::Decimal);
    }

    let mut x: u64 = 0;
    let mut overflow = false;
    let mut underscore = false;
    let mut period = false;
    let mut special: u8 = 0;
    let mut exponent = false;
    let mut is_float = false;

    while i < bytes.len() {
        let c = bytes[i];

        if c == b'_' {
            if i == 2 && base_value != 10 {
                return NumberResult::fail(
                    NumberErrorKind::InvalidUnderscoreAfterSpecial,
                    i,
                    NumberBase::Decimal,
                );
            }
            if special != 0 {
                return NumberResult::fail(
                    NumberErrorKind::InvalidUnderscoreAfterSpecial,
                    i,
                    NumberBase::Decimal,
                );
            }
            if underscore {
                return NumberResult::fail(
                    NumberErrorKind::RepeatedUnderscore,
                    i,
                    NumberBase::Decimal,
                );
            }
            underscore = true;
            i += 1;
            continue;
        }

        if (c == b'e' || c == b'E') && base_value == 10 {
            is_float = true;
            if exponent {
                return NumberResult::fail(
                    NumberErrorKind::DuplicateExponent,
                    i,
                    NumberBase::Decimal,
                );
            }
            if underscore {
                return NumberResult::fail(
                    NumberErrorKind::ExponentAfterUnderscore,
                    i,
                    NumberBase::Decimal,
                );
            }
            special = c;
            exponent = true;
            i += 1;
            continue;
        }

        if (c == b'p' || c == b'P') && base_value == 16 {
            is_float = true;
            if exponent {
                return NumberResult::fail(
                    NumberErrorKind::DuplicateExponent,
                    i,
                    NumberBase::Decimal,
                );
            }
            if underscore {
                return NumberResult::fail(
                    NumberErrorKind::ExponentAfterUnderscore,
                    i,
                    NumberBase::Decimal,
                );
            }
            special = c;
            exponent = true;
            i += 1;
            continue;
        }

        if c == b'.' {
            is_float = true;
            if base_value != 10 && base_value != 16 {
                return NumberResult::fail(
                    NumberErrorKind::InvalidFloatBase,
                    2,
                    NumberBase::Decimal,
                );
            }
            if period {
                return NumberResult::fail(
                    NumberErrorKind::DuplicatePeriod,
                    i,
                    NumberBase::Decimal,
                );
            }
            period = true;
            if underscore {
                return NumberResult::fail(
                    NumberErrorKind::SpecialAfterUnderscore,
                    i,
                    NumberBase::Decimal,
                );
            }
            special = b'.';
            i += 1;
            continue;
        }

        if c == b'+' || c == b'-' {
            if special != b'p' && special != b'P' && special != b'e' && special != b'E' {
                return NumberResult::fail(
                    NumberErrorKind::InvalidExponentSign,
                    i,
                    NumberBase::Decimal,
                );
            }
            special = c;
            i += 1;
            continue;
        }

        // Map `c` to a digit value in the current base.
        let digit: u8 = if c.is_ascii_digit() {
            c - b'0'
        } else if c.is_ascii_uppercase() {
            c - b'A' + 10
        } else if c.is_ascii_lowercase() {
            c - b'a' + 10
        } else {
            return NumberResult::fail(NumberErrorKind::InvalidCharacter, i, NumberBase::Decimal);
        };

        if digit >= base_value {
            return NumberResult::fail(NumberErrorKind::InvalidDigit, i, base);
        }
        if exponent && digit >= 10 {
            return NumberResult::fail(
                NumberErrorKind::InvalidDigitExponent,
                i,
                NumberBase::Decimal,
            );
        }

        underscore = false;
        special = 0;

        if !is_float {
            // Accumulate, tracking overflow with C++-faithful unsigned wrap.
            if x != 0 {
                let mul = x.wrapping_mul(base_value as u64);
                if mul / base_value as u64 != x {
                    overflow = true;
                }
                x = mul;
            }
            let add = x.wrapping_add(digit as u64);
            if add < x {
                overflow = true;
            }
            x = add;
        }

        i += 1;
    }

    if underscore {
        return NumberResult::fail(
            NumberErrorKind::TrailingUnderscore,
            bytes.len() - 1,
            NumberBase::Decimal,
        );
    }
    if special != 0 {
        return NumberResult::fail(
            NumberErrorKind::TrailingSpecial,
            bytes.len() - 1,
            NumberBase::Decimal,
        );
    }

    if is_float {
        // Underscores already validated — strip them and parse the pristine
        // numeric form. `parse_decimal_float` handles decimal AND hex floats
        // (`0x1.8p4`) with one correctly-rounded step.
        let clean: Vec<u8> = bytes.iter().copied().filter(|&c| c != b'_').collect();
        let clean_str = std::str::from_utf8(&clean).unwrap_or("");
        let v = match parse_decimal_float(clean_str) {
            ParsedFloat::Fits64(bits) => f64::from_bits(bits),
            ParsedFloat::Quad(q) => quad_to_target_as_double(&q, false),
        };
        return NumberResult::ok_float(base, v);
    }
    if overflow {
        return NumberResult::ok_big_int(base);
    }
    NumberResult::ok_int(x, base)
}

