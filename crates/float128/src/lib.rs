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

/// Decimal: `intdigits[.fracdigits][(e|E)[sign]expdigits]`.
fn parse_decimal(s: &str) -> u128 {
    let (mant, exp_str) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };

    // Significant digits = int ++ frac, as one big integer D.
    let mut digits: Vec<u8> = Vec::with_capacity(int_part.len() + frac_part.len());
    digits.extend(
        int_part
            .bytes()
            .filter(u8::is_ascii_digit)
            .map(|b| b - b'0'),
    );
    digits.extend(
        frac_part
            .bytes()
            .filter(u8::is_ascii_digit)
            .map(|b| b - b'0'),
    );

    let explicit_exp: i64 = exp_str.parse().unwrap_or(0);
    // The decimal point shifts the exponent left by the fraction length.
    let exp10 = explicit_exp - frac_part.bytes().filter(u8::is_ascii_digit).count() as i64;

    let d = Big::from_digits(&digits);
    if d.is_zero() {
        return 0;
    }

    // value = D * 10^exp10  ->  num/den
    let (num, den) = if exp10 >= 0 {
        let mut n = d;
        for _ in 0..exp10 {
            n.mul_small(10);
        }
        (n, Big::one())
    } else {
        let mut den = Big::one();
        for _ in 0..(-exp10) {
            den.mul_small(10);
        }
        (d, den)
    };
    ratio_to_binary128(num, den)
}

/// Hex float: `hexint[.hexfrac](p|P)[sign]binexp`.
fn parse_hex_float(s: &str) -> u128 {
    let (mant, exp_str) = match s.find(['p', 'P']) {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };

    // M = integer value of (hexint ++ hexfrac); value = M * 2^(p - 4*fraclen).
    let mut m = Big::zero();
    let mut hex_frac_digits = 0i64;
    for b in int_part.bytes().chain(frac_part.bytes()) {
        if let Some(v) = (b as char).to_digit(16) {
            m.mul_small(16);
            m.add_small(v);
        }
    }
    for b in frac_part.bytes() {
        if (b as char).is_ascii_hexdigit() {
            hex_frac_digits += 1;
        }
    }
    if m.is_zero() {
        return 0;
    }
    let p: i64 = exp_str.parse().unwrap_or(0);
    let bin_exp = p - 4 * hex_frac_digits;

    // value = M * 2^bin_exp  ->  num/den
    let (num, den) = if bin_exp >= 0 {
        (m.shl(bin_exp as usize), Big::one())
    } else {
        (m, Big::one().shl((-bin_exp) as usize))
    };
    ratio_to_binary128(num, den)
}

/// Correctly-rounded (nearest, ties-to-even) conversion of the positive
/// rational `num/den` into a binary128 bit pattern (sign bit left 0).
fn ratio_to_binary128(num: Big, den: Big) -> u128 {
    if num.is_zero() {
        return 0;
    }
    // Choose k so that q = floor((num<<k)/den) lands in [2^112, 2^113), i.e.
    //   den*2^112 <= num*2^k < den*2^113.
    // Equivalently scale one side; `k` is the net left shift of num/den.
    let mut k: i64 = 112 - (num.bit_len() as i64 - den.bit_len() as i64);
    let (a, b) = loop {
        let (a, b) = scaled_pair(&num, &den, k);
        let lo = b.shl(112); // den_scaled * 2^112
        let hi = b.shl(113);
        if a.cmp(&lo) == std::cmp::Ordering::Less {
            k += 1; // q would be < 2^112 -> need more left shift
            continue;
        }
        if a.cmp(&hi) != std::cmp::Ordering::Less {
            k -= 1; // q would be >= 2^113 -> too much
            continue;
        }
        break (a, b);
    };

    // Now q = floor(a/b) is exactly 113 bits; r = a mod b.
    let (mut q, r) = divmod_u128(&a, &b);
    let mut e: i64 = -k; // value == q * 2^e

    // Round nearest, ties to even, using 2*r vs b.
    let twice_r = r.shl(1);
    match twice_r.cmp(&b) {
        std::cmp::Ordering::Greater => q += 1,
        std::cmp::Ordering::Equal if (q & 1) == 1 => q += 1,
        _ => {}
    }
    if q == (1u128 << 113) {
        q = 1u128 << 112; // mantissa carry-out -> renormalize
        e += 1;
    }
    assemble_binary128(q, e)
}

/// Build (num<<k, den) for k>=0, or (num, den<<-k) for k<0.
fn scaled_pair(num: &Big, den: &Big, k: i64) -> (Big, Big) {
    if k >= 0 {
        (num.shl(k as usize), den.clone())
    } else {
        (num.clone(), den.shl((-k) as usize))
    }
}

/// Assemble a binary128 pattern from a 113-bit significand `q` (top bit at
/// position 112) and binary scale `e` such that the value is `q * 2^e`.
fn assemble_binary128(q: u128, e: i64) -> u128 {
    // value = q * 2^e = (q / 2^112) * 2^(112+e); q/2^112 in [1,2).
    let unbiased = 112 + e;
    let biased = unbiased + QUAD_BIAS;
    if biased >= 0x7FFF {
        // Overflow of binary128 itself -> +inf. Unreachable for any f32/f64
        // literal (binary128 max ~1.2e4932); the tokenizer rejects such ranges.
        return (0x7FFFu128) << F128_FRAC_BITS;
    }
    if biased <= 0 {
        // Underflow below binary128 normal range -> 0. Likewise unreachable for
        // real literals (binary128 min normal ~3.4e-4932).
        return 0;
    }
    let frac = q & ((1u128 << F128_FRAC_BITS) - 1);
    ((biased as u128) << F128_FRAC_BITS) | frac
}

// ===========================================================================
// binary128 -> f32 / f64 narrowing (single correct rounding)
// ===========================================================================

#[derive(Clone, Copy)]
enum FloatFmt {
    F32,
    F64,
}

impl FloatFmt {
    fn exp_bits(self) -> u32 {
        match self {
            FloatFmt::F32 => 8,
            FloatFmt::F64 => 11,
        }
    }
    fn frac_bits(self) -> u32 {
        match self {
            FloatFmt::F32 => 23,
            FloatFmt::F64 => 52,
        }
    }
    fn bias(self) -> i64 {
        match self {
            FloatFmt::F32 => 127,
            FloatFmt::F64 => 1023,
        }
    }
}

/// Round binary128 `bits` to the given target format. Returns `(target_bits,
/// lost)` where `target_bits` holds the result in the low bits (32 for f32, 64
/// for f64) and `lost` is true iff the conversion discarded any information
/// (the APFloat `lostInfo` semantics — used to decide the `Fits64` storage).
fn f128_to_target(bits: u128, fmt: FloatFmt) -> (u64, bool) {
    let exp_bits = fmt.exp_bits();
    let frac_bits = fmt.frac_bits();
    let bias = fmt.bias();
    let sign = ((bits >> 127) & 1) as u64;
    let src_exp = ((bits >> F128_FRAC_BITS) & 0x7FFF) as i64;
    let src_frac = bits & ((1u128 << F128_FRAC_BITS) - 1);
    let exp_all = (1u64 << exp_bits) - 1;
    let sign_shifted = sign << (exp_bits + frac_bits);

    // Inf / NaN.
    if src_exp == 0x7FFF {
        return if src_frac == 0 {
            (sign_shifted | (exp_all << frac_bits), false) // inf
        } else {
            // Quiet NaN (set the top fraction bit). lost=false: NaN -> NaN.
            (
                sign_shifted | (exp_all << frac_bits) | (1u64 << (frac_bits - 1)),
                false,
            )
        };
    }
    // Zero (binary128 subnormals don't arise from literals; treat as ~0).
    if src_exp == 0 {
        return (sign_shifted, src_frac != 0);
    }

    // Normal binary128: significand has implicit leading 1 at bit 112.
    let sig = (1u128 << F128_FRAC_BITS) | src_frac; // in [2^112, 2^113)
    let unbiased = src_exp - QUAD_BIAS;

    // Overflow of the target -> inf.
    if unbiased > bias {
        return (sign_shifted | (exp_all << frac_bits), true);
    }

    // Bits of `sig` to discard, and the target biased exponent.
    let (discard, mut biased_exp, subnormal) = if unbiased >= 1 - bias {
        (
            (F128_FRAC_BITS - frac_bits) as i64,
            (unbiased + bias) as u64,
            false,
        )
    } else {
        let extra = (1 - bias) - unbiased; // >= 1
        ((F128_FRAC_BITS - frac_bits) as i64 + extra, 0u64, true)
    };

    // Discarding >=114 bits means the whole significand (which is <2^113) is
    // below half a ULP of the smallest representable value -> rounds to zero.
    if discard >= 114 {
        return (sign_shifted, true);
    }

    let d = discard as u32;
    let mut keep = sig >> d;
    let rem = sig & ((1u128 << d) - 1);
    let half = 1u128 << (d - 1);
    let lost = rem != 0;

    // Round nearest, ties to even.
    if rem > half || (rem == half && (keep & 1) == 1) {
        keep += 1;
    }

    if subnormal {
        // Rounding may push a subnormal up into the smallest normal (mantissa
        // carries into the implicit bit).
        if keep >= (1u128 << frac_bits) {
            biased_exp = 1;
            keep &= (1u128 << frac_bits) - 1;
        }
    } else {
        // Normal: `keep` is in [2^frac_bits, 2^(frac_bits+1)); a round-up can
        // overflow it, bumping the exponent.
        if keep == (1u128 << (frac_bits + 1)) {
            keep >>= 1;
            biased_exp += 1;
            if biased_exp >= exp_all {
                return (sign_shifted | (exp_all << frac_bits), true); // -> inf
            }
        }
    }

    let frac_mask = (1u64 << frac_bits) - 1;
    let frac_t = (keep as u64) & frac_mask;
    (sign_shifted | (biased_exp << frac_bits) | frac_t, lost)
}

// ===========================================================================
// Minimal big unsigned integer (little-endian u32 limbs)
// ===========================================================================

#[derive(Clone, Debug)]
struct Big {
    /// little-endian base-2^32 limbs, normalized (no trailing zero limb except
    /// the canonical empty == zero).
    limbs: Vec<u32>,
}

impl Big {
    fn zero() -> Big {
        Big { limbs: Vec::new() }
    }
    fn one() -> Big {
        Big { limbs: vec![1] }
    }
    fn from_digits(digits: &[u8]) -> Big {
        let mut b = Big::zero();
        for &d in digits {
            b.mul_small(10);
            b.add_small(d as u32);
        }
        b
    }
    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }
    fn normalize(&mut self) {
        while let Some(&0) = self.limbs.last() {
            self.limbs.pop();
        }
    }
    fn mul_small(&mut self, m: u32) {
        if m == 0 {
            self.limbs.clear();
            return;
        }
        let mut carry: u64 = 0;
        for limb in self.limbs.iter_mut() {
            let v = (*limb as u64) * (m as u64) + carry;
            *limb = v as u32;
            carry = v >> 32;
        }
        while carry != 0 {
            self.limbs.push(carry as u32);
            carry >>= 32;
        }
    }
    fn add_small(&mut self, a: u32) {
        if a == 0 {
            return;
        }
        let mut carry = a as u64;
        let mut i = 0;
        while carry != 0 {
            if i == self.limbs.len() {
                self.limbs.push(0);
            }
            let v = self.limbs[i] as u64 + carry;
            self.limbs[i] = v as u32;
            carry = v >> 32;
            i += 1;
        }
    }
    /// Multiply by 2^bits (left shift).
    fn shl(&self, bits: usize) -> Big {
        if self.is_zero() {
            return Big::zero();
        }
        let limb_shift = bits / 32;
        let bit_shift = (bits % 32) as u32;
        let mut out = vec![0u32; limb_shift];
        if bit_shift == 0 {
            out.extend_from_slice(&self.limbs);
        } else {
            let mut carry: u32 = 0;
            for &limb in &self.limbs {
                let v = ((limb as u64) << bit_shift) | (carry as u64);
                out.push(v as u32);
                carry = (v >> 32) as u32;
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        let mut b = Big { limbs: out };
        b.normalize();
        b
    }
    fn bit_len(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() - 1) * 32 + (32 - top.leading_zeros() as usize),
        }
    }
    fn bit(&self, i: usize) -> u32 {
        let limb = i / 32;
        if limb >= self.limbs.len() {
            0
        } else {
            (self.limbs[limb] >> (i % 32)) & 1
        }
    }
    fn cmp(&self, other: &Big) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
    /// self -= other (requires self >= other).
    fn sub_assign(&mut self, other: &Big) {
        let mut borrow: i64 = 0;
        for i in 0..self.limbs.len() {
            let o = if i < other.limbs.len() {
                other.limbs[i] as i64
            } else {
                0
            };
            let mut v = self.limbs[i] as i64 - o - borrow;
            if v < 0 {
                v += 1 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            self.limbs[i] = v as u32;
        }
        self.normalize();
    }
    /// Append a bit at position 0 then OR — i.e. self = self*2 + bit.
    fn shl1_or(&mut self, bit: u32) {
        // multiply by 2
        let mut carry: u32 = 0;
        for limb in self.limbs.iter_mut() {
            let v = ((*limb as u64) << 1) | (carry as u64);
            *limb = v as u32;
            carry = (v >> 32) as u32;
        }
        if carry != 0 {
            self.limbs.push(carry);
        }
        if bit != 0 {
            self.add_small(1);
        }
    }
}

/// Long division producing `floor(a/b)` (known to fit u128 here) and `a mod b`.
fn divmod_u128(a: &Big, b: &Big) -> (u128, Big) {
    let mut q: u128 = 0;
    let mut r = Big::zero();
    let n = a.bit_len();
    for i in (0..n).rev() {
        r.shl1_or(a.bit(i));
        if r.cmp(b) != std::cmp::Ordering::Less {
            r.sub_assign(b);
            if i < 128 {
                q |= 1u128 << i;
            }
        }
    }
    (q, r)
}

