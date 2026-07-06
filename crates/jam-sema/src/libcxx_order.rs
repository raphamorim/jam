/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

//! Faithful pure-Rust emulation of macOS system libc++ (`_LIBCPP_VERSION`
//! 200100, the v2000.x line shipped in `/usr/lib/libc++.1.dylib`)
//! `std::unordered_map<std::string, V>` iteration order.
//!
//! The C++ frontend iterates its loaded-modules
//! `std::unordered_map<std::string, ...>` in `begin()..end()` order to emit
//! LLVM IR. To stay byte-compatible while porting the compiler to Rust we must
//! reproduce that order exactly, which means reproducing three things bit for
//! bit:
//!
//! 1. `std::hash<std::string>` — libc++ routes this through
//!    `__do_string_hash` → `__murmur2_or_cityhash<size_t>` which, for 64-bit
//!    `size_t`, is a CityHash64 variant. See [`libcxx_string_hash`].
//! 2. The prime bucket-count growth policy (`__hash_table::__next_prime`,
//!    `__rehash`, `max_load_factor == 1.0`).
//! 3. The single intrusive-list node chaining and rehash re-link order
//!    (`__node_insert_unique_perform` / `__do_rehash`).
//!
//! Every constant, branch, and arithmetic operation below was transcribed from
//! the SDK headers (`__functional/hash.h`, `__hash_table`,
//! `__string/char_traits.h`) and cross-checked against probes compiled with
//! `clang++ -std=c++17 -stdlib=libc++` linking the live system dylib.
//!
//! All integer arithmetic uses wrapping semantics to match libc++'s unsigned
//! `size_t` math (libc++ even tags these functions
//! `_LIBCPP_DISABLE_UBSAN_UNSIGNED_INTEGER_CHECK`).

// ===========================================================================
// std::hash<std::string>  ==  __murmur2_or_cityhash<uint64_t>
// ===========================================================================

// Some primes between 2^63 and 2^64, straight from libc++.
const K0: u64 = 0xc3a5_c85c_97cb_3127;
const K1: u64 = 0xb492_b66f_be98_f273;
const K2: u64 = 0x9ae1_6a3b_2f90_404f;
const K3: u64 = 0xc949_d7c7_509e_6557;

/// `std::__loadword<uint64_t>` — a `memcpy` of 8 little-endian bytes.
#[inline]
fn loadword64(s: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&s[off..off + 8]);
    u64::from_le_bytes(b)
}

/// `std::__loadword<uint32_t>` — a `memcpy` of 4 little-endian bytes.
#[inline]
fn loadword32(s: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&s[off..off + 4]);
    u32::from_le_bytes(b)
}

/// `__murmur2_or_cityhash<_Size,64>::__rotate`.
///
/// libc++ guards `shift == 0` (where `val << 64` would be UB in C++) by
/// returning `val`; `rotate_right` already has exactly that wrap-around
/// semantics, so they coincide for every shift in `0..64`.
#[inline]
fn rotate(val: u64, shift: u32) -> u64 {
    val.rotate_right(shift)
}

/// `__murmur2_or_cityhash<_Size,64>::__rotate_by_at_least_1` (shift is always
/// `>= 1` at every call site, so this matches `__rotate` here too).
#[inline]
fn rotate_by_at_least_1(val: u64, shift: u32) -> u64 {
    val.rotate_right(shift)
}

/// `__murmur2_or_cityhash<_Size,64>::__shift_mix`.
#[inline]
fn shift_mix(val: u64) -> u64 {
    val ^ (val >> 47)
}

/// `__murmur2_or_cityhash<_Size,64>::__hash_len_16`.
#[inline]
fn hash_len_16(u: u64, v: u64) -> u64 {
    const MUL: u64 = 0x9ddf_ea08_eb38_2d69;
    let mut a = (u ^ v).wrapping_mul(MUL);
    a ^= a >> 47;
    let mut b = (v ^ a).wrapping_mul(MUL);
    b ^= b >> 47;
    b = b.wrapping_mul(MUL);
    b
}

/// `__hash_len_0_to_16` — the dominant path for the short keys we care about.
///
/// `_LIBCPP_ABI_FIX_CITYHASH_IMPLEMENTATION` is *not* defined on the macOS
/// system libc++ (it is an ABI-breaking fix gated off on the stable ABI), so we
/// reproduce the legacy 32-bit-truncated `__a << 3` branch verbatim.
fn hash_len_0_to_16(s: &[u8], off: usize, len: usize) -> u64 {
    if len > 8 {
        let a = loadword64(s, off);
        let b = loadword64(s, off + len - 8);
        return hash_len_16(
            a,
            rotate_by_at_least_1(b.wrapping_add(len as u64), len as u32),
        ) ^ b;
    }
    if len >= 4 {
        // libc++ computes `__hash_len_16(__len + (__a << 3), __b)` where `__a`
        // and `__b` are `uint32_t`. Critically, on the macOS stable ABI
        // (`_LIBCPP_ABI_FIX_CITYHASH_IMPLEMENTATION` undefined) `__a << 3` is a
        // 32-bit shift that wraps mod 2^32 *before* widening to `size_t`.
        let a = loadword32(s, off);
        let b = loadword32(s, off + len - 4) as u64;
        let a_shifted = a.wrapping_shl(3) as u64;
        return hash_len_16((len as u64).wrapping_add(a_shifted), b);
    }
    if len > 0 {
        let a = s[off] as u32;
        let b = s[off + (len >> 1)] as u32;
        let c = s[off + len - 1] as u32;
        let y = a.wrapping_add(b << 8);
        let z = (len as u32).wrapping_add(c << 2);
        return shift_mix((y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K3))
            .wrapping_mul(K2);
    }
    K2
}

/// `__hash_len_17_to_32`.
fn hash_len_17_to_32(s: &[u8], off: usize, len: usize) -> u64 {
    let a = loadword64(s, off).wrapping_mul(K1);
    let b = loadword64(s, off + 8);
    let c = loadword64(s, off + len - 8).wrapping_mul(K2);
    let d = loadword64(s, off + len - 16).wrapping_mul(K0);
    hash_len_16(
        rotate(a.wrapping_sub(b), 43)
            .wrapping_add(rotate(c, 30))
            .wrapping_add(d),
        a.wrapping_add(rotate(b ^ K3, 20))
            .wrapping_sub(c)
            .wrapping_add(len as u64),
    )
}

/// `__weak_hash_len_32_with_seeds` (the 6-argument primitive form).
#[inline]
#[allow(clippy::too_many_arguments)]
fn weak_hash_len_32_with_seeds_vals(
    w: u64,
    x: u64,
    y: u64,
    z: u64,
    mut a: u64,
    mut b: u64,
) -> (u64, u64) {
    a = a.wrapping_add(w);
    b = rotate(b.wrapping_add(a).wrapping_add(z), 21);
    let c = a;
    a = a.wrapping_add(x);
    a = a.wrapping_add(y);
    b = b.wrapping_add(rotate(a, 44));
    (a.wrapping_add(z), b.wrapping_add(c))
}

/// `__weak_hash_len_32_with_seeds` (the buffer form).
#[inline]
fn weak_hash_len_32_with_seeds(s: &[u8], off: usize, a: u64, b: u64) -> (u64, u64) {
    weak_hash_len_32_with_seeds_vals(
        loadword64(s, off),
        loadword64(s, off + 8),
        loadword64(s, off + 16),
        loadword64(s, off + 24),
        a,
        b,
    )
}

/// `__hash_len_33_to_64`.
fn hash_len_33_to_64(s: &[u8], off: usize, len: usize) -> u64 {
    let mut z = loadword64(s, off + 24);
    let mut a = loadword64(s, off).wrapping_add(
        (len as u64)
            .wrapping_add(loadword64(s, off + len - 16))
            .wrapping_mul(K0),
    );
    let mut b = rotate(a.wrapping_add(z), 52);
    let mut c = rotate(a, 37);
    a = a.wrapping_add(loadword64(s, off + 8));
    c = c.wrapping_add(rotate(a, 7));
    a = a.wrapping_add(loadword64(s, off + 16));
    let vf = a.wrapping_add(z);
    let vs = b.wrapping_add(rotate(a, 31)).wrapping_add(c);
    a = loadword64(s, off + 16).wrapping_add(loadword64(s, off + len - 32));
    z = z.wrapping_add(loadword64(s, off + len - 8));
    b = rotate(a.wrapping_add(z), 52);
    c = rotate(a, 37);
    a = a.wrapping_add(loadword64(s, off + len - 24));
    c = c.wrapping_add(rotate(a, 7));
    a = a.wrapping_add(loadword64(s, off + len - 16));
    let wf = a.wrapping_add(z);
    let ws = b.wrapping_add(rotate(a, 31)).wrapping_add(c);
    let r = shift_mix(
        vf.wrapping_add(ws)
            .wrapping_mul(K2)
            .wrapping_add(wf.wrapping_add(vs).wrapping_mul(K0)),
    );
    shift_mix(r.wrapping_mul(K0).wrapping_add(vs)).wrapping_mul(K2)
}

/// `std::hash<std::string>{}(s)` for 64-bit `size_t` on the system libc++.
///
/// This is `__murmur2_or_cityhash<uint64_t>()(s.data(), s.size())`, i.e. the
/// CityHash64 specialization. Matches every probed value bit for bit.
pub fn libcxx_string_hash(s: &[u8]) -> u64 {
    let len = s.len();
    if len <= 32 {
        if len <= 16 {
            return hash_len_0_to_16(s, 0, len);
        }
        return hash_len_17_to_32(s, 0, len);
    }
    if len <= 64 {
        return hash_len_33_to_64(s, 0, len);
    }

    // For strings over 64 bytes we hash the end first, and then as we loop we
    // keep 56 bytes of state: v, w, x, y, and z.
    let mut x = loadword64(s, len - 40);
    let mut y = loadword64(s, len - 16).wrapping_add(loadword64(s, len - 56));
    let mut z = hash_len_16(
        loadword64(s, len - 48).wrapping_add(len as u64),
        loadword64(s, len - 24),
    );
    let mut v = weak_hash_len_32_with_seeds(s, len - 64, len as u64, z);
    let mut w = weak_hash_len_32_with_seeds(s, len - 32, y.wrapping_add(K1), x);
    x = x.wrapping_mul(K1).wrapping_add(loadword64(s, 0));

    // Decrease len to the nearest multiple of 64, and operate on 64-byte chunks.
    let mut rem = (len - 1) & !63usize;
    let mut off = 0usize;
    loop {
        x = rotate(
            x.wrapping_add(y)
                .wrapping_add(v.0)
                .wrapping_add(loadword64(s, off + 8)),
            37,
        )
        .wrapping_mul(K1);
        y = rotate(
            y.wrapping_add(v.1).wrapping_add(loadword64(s, off + 48)),
            42,
        )
        .wrapping_mul(K1);
        x ^= w.1;
        y = y.wrapping_add(v.0).wrapping_add(loadword64(s, off + 40));
        z = rotate(z.wrapping_add(w.0), 33).wrapping_mul(K1);
        v = weak_hash_len_32_with_seeds(s, off, v.1.wrapping_mul(K1), x.wrapping_add(w.0));
        w = weak_hash_len_32_with_seeds(
            s,
            off + 32,
            z.wrapping_add(w.1),
            y.wrapping_add(loadword64(s, off + 16)),
        );
        std::mem::swap(&mut z, &mut x);
        off += 64;
        rem -= 64;
        if rem == 0 {
            break;
        }
    }
    hash_len_16(
        hash_len_16(v.0, w.0)
            .wrapping_add(shift_mix(y).wrapping_mul(K1))
            .wrapping_add(z),
        hash_len_16(v.1, w.1).wrapping_add(x),
    )
}

// ===========================================================================
// __hash_table bucket policy
// ===========================================================================

