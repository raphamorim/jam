/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Numeric-literal validation and value extraction. The lexer produces a
//! permissive `NUMBER` token; here its bytes are validated (base prefix,
//! underscore placement, digit ranges, exponent rules) and converted to a
//! value. Overflowing integers become `BigInt`; floats parse via
//! [`float128::parse_decimal_float`] (correctly rounded, decimal and hex).

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
            // Accumulate with wrapping arithmetic, tracking overflow.
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

/// Human-readable message for a numeric-literal error.
pub fn number_error_message(kind: NumberErrorKind) -> &'static str {
    use NumberErrorKind::*;
    match kind {
        LeadingZero => "leading zero is not allowed (use `0o` for octal)",
        DigitAfterBase => "expected digit after base prefix",
        UpperCaseBase => "base prefix must be lowercase (`0x`, `0b`, `0o`)",
        InvalidFloatBase => "float literals are only allowed in decimal or hex",
        RepeatedUnderscore => "consecutive underscores in numeric literal",
        InvalidUnderscoreAfterSpecial => {
            "underscore not allowed immediately after `0x`/`0b`/`0o` or after `.`/`+`/`-`/`e`/`E`/`p`/`P`"
        }
        InvalidDigit => "invalid digit for the literal's base",
        InvalidDigitExponent => "non-decimal digit in exponent",
        DuplicatePeriod => "more than one `.` in numeric literal",
        DuplicateExponent => "more than one exponent in numeric literal",
        InvalidHexExponent => "hex literal cannot use `e`/`E` exponent (use `p`/`P`)",
        ExponentAfterUnderscore => "exponent letter cannot follow `_`",
        SpecialAfterUnderscore => "`.`/`+`/`-` cannot follow `_`",
        TrailingSpecial => "numeric literal ends with `.`/`+`/`-`/`e`/`p`",
        TrailingUnderscore => "numeric literal ends with `_`",
        InvalidCharacter => "character not allowed in numeric literal",
        InvalidExponentSign => "`+`/`-` is only valid immediately after `e`/`E`/`p`/`P`",
        IntegerTooLarge => "integer literal exceeds u64 range",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> NumberResult {
        parse_number_literal(s.as_bytes())
    }

    #[test]
    fn integers_by_base() {
        assert_eq!(parse("42").kind, NumberResultKind::Int);
        assert_eq!(parse("42").int_value, 42);
        assert_eq!(parse("0x800").int_value, 0x800); // test_array_type_sizes
        assert_eq!(parse("0x800").base, NumberBase::Hex);
        assert_eq!(parse("0b1010").int_value, 0b1010);
        assert_eq!(parse("0o17").int_value, 0o17);
        assert_eq!(parse("1_000_000").int_value, 1_000_000);
    }

    #[test]
    fn overflow_is_bigint() {
        // 2^64 overflows u64.
        assert_eq!(parse("18446744073709551616").kind, NumberResultKind::BigInt);
        assert_eq!(parse("18446744073709551615").int_value, u64::MAX);
    }

    #[test]
    #[allow(clippy::approx_constant)] // `3.14` is the parse input under test, not an approximation of PI
    fn floats_decimal_and_hex() {
        let d = parse("3.14");
        assert_eq!(d.kind, NumberResultKind::Float);
        assert_eq!(d.float_value, 3.14_f64);
        assert_eq!(d.base, NumberBase::Decimal);

        let e = parse("1.5e-3");
        assert_eq!(e.float_value, 1.5e-3_f64);

        let h = parse("0x1.8p1");
        assert_eq!(h.kind, NumberResultKind::Float);
        assert_eq!(h.base, NumberBase::Hex);
        assert_eq!(h.float_value, 3.0_f64); // 1.5 * 2^1
    }

    #[test]
    fn float_matches_rust_parse() {
        for s in [
            "0.1",
            "3.141592653589793",
            "1e100",
            "2.5",
            "0.30000000000000004",
        ] {
            assert_eq!(parse(s).float_value, s.parse::<f64>().unwrap(), "{s}");
        }
    }

    #[test]
    fn rejects() {
        assert_eq!(
            parse("07").failure.unwrap().kind,
            NumberErrorKind::LeadingZero
        );
        assert_eq!(
            parse("0X42").failure.unwrap().kind,
            NumberErrorKind::UpperCaseBase
        );
        assert_eq!(
            parse("0x").failure.unwrap().kind,
            NumberErrorKind::DigitAfterBase
        );
        assert_eq!(
            parse("1__0").failure.unwrap().kind,
            NumberErrorKind::RepeatedUnderscore
        );
        assert_eq!(
            parse("1.0.5").failure.unwrap().kind,
            NumberErrorKind::DuplicatePeriod
        );
        assert_eq!(
            parse("0b2").failure.unwrap().kind,
            NumberErrorKind::InvalidDigit
        );
        assert_eq!(
            parse("1_").failure.unwrap().kind,
            NumberErrorKind::TrailingUnderscore
        );
        // base reported on InvalidDigit
        assert_eq!(parse("0b2").failure.unwrap().base, NumberBase::Binary);
    }
}
