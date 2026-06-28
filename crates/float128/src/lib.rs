/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

// the IEEE-754 binary128 (often called quadruple precision or quad precision)
// is a standard for representing floating-point numbers in computers.

// for anyone contributing to this (maybe in the future idk):
// 1. this crate should be fully in rust and, 2. dep free.

// we use float128 due to LLVM APFloat
// https://llvm.org/doxygen/classllvm_1_1APFloat.html
// Quadruple-precision floating-point format
// https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format

// The IEEE 754 standard specifies a binary128 as having:
// Sign bit: 1bit (in our case it's on position 127, 0 for positive and 1 for negative)
// Exponent width: 15bits, it basically encodes the magnitude (positions 112 to 126).
// Stored fraction field + Significand precision: first 112 is basically what
// lives in memory and (positions 0 to 111)
// note: the significand for a normal number is always 1.something,
// regardless of how big or small the actual number is.

// for example
// 15.3013
// 0     10000010     11101001101001000100000
// |     └──┬───┘     └──────────┬──────────┘
// sign  exponent            fraction

// -15.3013
// 1     10000010     11101001101001000100000
// |     └──┬───┘     └──────────┬──────────┘
// sign  exponent            fraction

//! Layout reminder — binary128: bit 127 = sign, bits 126..=112 = exponent
//! (15-bit, bias 16383), bits 111..=0 = fraction (112-bit). The `quad`
//! representation is the 128-bit pattern as four little-endian u32 words
//! (`quad[0]` = low 32 bits … `quad[3]` = high 32 bits incl. sign+exponent).

/// Result of parsing a float literal into its precision-preserving form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsedFloat {
    /// The value round-trips through f64 with no loss; the f64 bit pattern is
    /// stored inline (the common case for "nice" literals).
    Fits64(u64),
    /// The value needs the full binary128 pattern (4×u32, little-endian words)
    /// to preserve precision until the final f32/f64 rounding.
    Quad([u32; 4]),
}

const QUAD_BIAS: i64 = 16383;
const F128_FRAC_BITS: u32 = 112;

// ===========================================================================
// Public API (mirrors the two C facade entry points)
// ===========================================================================

/// Parse a sign-stripped, underscore-free decimal or hex float lexeme into
/// binary128, then pick the smallest lossless storage: `Fits64` if the value
/// round-trips through f64 exactly, else the full `Quad`.
pub fn parse_decimal_float(lexeme: &str) -> ParsedFloat {
    let bits = parse_to_binary128(lexeme);
    let (f64_bits, lost) = f128_to_target(bits, FloatFmt::F64);
    if !lost {
        ParsedFloat::Fits64(f64_bits)
    } else {
        ParsedFloat::Quad(to_quad(bits))
    }
}

/// Round a binary128 quad once to f32 (`to_f32`) or f64, returned widened to a
/// C `f64` (an f32 result is exact in f64, so the widening never rounds). This
/// is the single, final rounding — no double-rounding through f64.
pub fn quad_to_target_as_double(quad: &[u32; 4], to_f32: bool) -> f64 {
    let bits = from_quad(quad);
    if to_f32 {
        let (f32_bits, _) = f128_to_target(bits, FloatFmt::F32);
        // Widen f32 -> f64 exactly (the C-ABI return convention).
        f32::from_bits(f32_bits as u32) as f64
    } else {
        let (f64_bits, _) = f128_to_target(bits, FloatFmt::F64);
        f64::from_bits(f64_bits)
    }
}

// quad <-> u128 (little-endian word order).
fn to_quad(bits: u128) -> [u32; 4] {
    [
        bits as u32,
        (bits >> 32) as u32,
        (bits >> 64) as u32,
        (bits >> 96) as u32,
    ]
}
fn from_quad(quad: &[u32; 4]) -> u128 {
    (quad[0] as u128)
        | ((quad[1] as u128) << 32)
        | ((quad[2] as u128) << 64)
        | ((quad[3] as u128) << 96)
}

// ===========================================================================
// Lexeme parsing -> binary128 bit pattern
// ===========================================================================

fn parse_to_binary128(lexeme: &str) -> u128 {
    // Strip a leading sign defensively (the contract says lexemes are
    // sign-stripped). A leading '-' sets the sign bit.
    let mut s = lexeme.trim();
    let mut sign: u128 = 0;
    if let Some(rest) = s.strip_prefix('-') {
        sign = 1u128 << 127;
        s = rest;
    } else if let Some(rest) = s.strip_prefix('+') {
        s = rest;
    }

    let magnitude = if s.len() >= 2 && (&s[..2] == "0x" || &s[..2] == "0X") {
        parse_hex_float(&s[2..])
    } else {
        parse_decimal(s)
    };
    magnitude | sign
}

