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

/// `std::__constrain_hash(h, bc)`.
///
/// `!(bc & (bc-1))` is true for powers of two (and for `bc == 0`, but this is
/// never called with `bc == 0`). For powers of two it masks; otherwise it
/// reduces modulo `bc` (with the `h < bc` fast path).
#[inline]
fn constrain_hash(h: u64, bc: u64) -> u64 {
    if bc & (bc.wrapping_sub(1)) == 0 {
        h & bc.wrapping_sub(1)
    } else if h < bc {
        h
    } else {
        h % bc
    }
}

/// `std::__is_hash_power2(bc)` — note the strict `> 2`, so `2` is *not* treated
/// as a hash-power-of-two by libc++'s growth math.
#[inline]
fn is_hash_power2(bc: u64) -> bool {
    bc > 2 && (bc & (bc.wrapping_sub(1)) == 0)
}

#[inline]
fn is_prime(x: u64) -> bool {
    if x < 2 {
        return false;
    }
    if x < 4 {
        return true;
    }
    if x.is_multiple_of(2) {
        return false;
    }
    let mut i = 3u64;
    while i.wrapping_mul(i) <= x {
        if x.is_multiple_of(i) {
            return false;
        }
        i += 2;
    }
    true
}

/// `std::__next_prime(n)` — the smallest prime `>= n`.
///
/// libc++'s `src/hash.cpp` implements this with a small static prime table plus
/// a 210-wheel search for larger inputs, but the observable result is exactly
/// "smallest prime >= n" (`__next_prime(0) == 0` is the lone special case it
/// never actually feeds back into bucketing). Verified to match the live system
/// dylib for all `n` in `0..=5000` with zero mismatches.
fn next_prime(mut n: u64) -> u64 {
    if n <= 1 {
        // libc++'s table starts at 2; n==0/1 only reach here via paths the
        // caller has already special-cased, but keep parity: next prime >= 2.
        return 2;
    }
    while !is_prime(n) {
        n += 1;
    }
    n
}

/// `__hash_table::__rehash<true>(n)` bucket-count selection (grow path only).
///
/// Mirrors the header verbatim:
/// ```text
/// if (n == 1) n = 2;
/// else if (n & (n - 1)) n = __next_prime(n);
/// // (powers of two and 0 pass through unchanged)
/// ```
/// We only ever grow (`n > bucket_count`) here, matching the insert path, so the
/// shrink branch of `__rehash` is irrelevant.
#[inline]
fn rehash_bucket_count(n: u64) -> u64 {
    if n == 1 {
        2
    } else if n & (n.wrapping_sub(1)) != 0 {
        next_prime(n)
    } else {
        n
    }
}

// ===========================================================================
// __hash_table node chaining + iteration
// ===========================================================================

/// One node in libc++'s single intrusive forward list of all elements.
struct Node {
    /// The insertion index of the key (its value in `m[k] = i`); also the
    /// answer index we ultimately report.
    idx: usize,
    /// Cached `std::hash<std::string>` of the key.
    hash: u64,
    /// `__next_` link: index into the node arena, or `None` for the tail.
    next: Option<usize>,
}

/// Faithful model of `std::unordered_map<std::string, V>`'s table.
///
/// libc++ keeps a sentinel head node (`__first_node_`) whose `__next_` is the
/// real list head, and a `__bucket_list_` of "node-before-first-of-bucket"
/// pointers. We model the sentinel as `head: Option<usize>` (the sentinel's
/// `__next_`) and bucket entries as `Option<BucketRef>` where `BucketRef`
/// distinguishes "points at the sentinel" from "points at arena node i".
struct Table {
    /// Arena of nodes; never shrinks. Indices are stable for the whole build.
    nodes: Vec<Node>,
    /// `__first_node_.__next_` — index of the first real node, or `None`.
    head: Option<usize>,
    /// `__bucket_list_`: for bucket `b`, the node *before* bucket `b`'s first
    /// node. `Some(BucketRef::Sentinel)` means "the sentinel head", i.e. this
    /// bucket owns the front of the list. `Some(BucketRef::Node(i))` means arena
    /// node `i`. `None` is an empty bucket.
    buckets: Vec<Option<BucketRef>>,
    /// Live element count (`size()`).
    size: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BucketRef {
    /// The sentinel `__first_node_` (the node before the list head).
    Sentinel,
    /// Arena node `i`.
    Node(usize),
}

impl Table {
    fn new() -> Self {
        Table {
            nodes: Vec::new(),
            head: None,
            buckets: Vec::new(),
            size: 0,
        }
    }

    #[inline]
    fn bucket_count(&self) -> u64 {
        self.buckets.len() as u64
    }

    /// `__next_` of a [`BucketRef`].
    #[inline]
    fn ref_next(&self, r: BucketRef) -> Option<usize> {
        match r {
            BucketRef::Sentinel => self.head,
            BucketRef::Node(i) => self.nodes[i].next,
        }
    }

    /// Set `__next_` of a [`BucketRef`].
    #[inline]
    fn set_ref_next(&mut self, r: BucketRef, v: Option<usize>) {
        match r {
            BucketRef::Sentinel => self.head = v,
            BucketRef::Node(i) => self.nodes[i].next = v,
        }
    }

    /// `__do_rehash<true>(nbc)`: reallocate the bucket array to `nbc`, then walk
    /// the existing list re-bucketing every node, preserving libc++'s exact
    /// re-link order.
    fn do_rehash(&mut self, nbc: u64) {
        let nbc_usize = nbc as usize;
        self.buckets = vec![None; nbc_usize];
        if nbc == 0 {
            return;
        }
        // __pp = __first_node_.__ptr() (sentinel); __cp = __pp->__next_
        let mut pp = BucketRef::Sentinel;
        let mut cp = self.head;
        if let Some(first) = cp {
            let chash = constrain_hash(self.nodes[first].hash, nbc) as usize;
            self.buckets[chash] = Some(pp); // sentinel
            let mut phash = chash;
            // for (__pp = __cp; __cp != nullptr; __cp = __pp->__next_)
            pp = BucketRef::Node(first);
            cp = self.nodes[first].next;
            while let Some(cur) = cp {
                let chash = constrain_hash(self.nodes[cur].hash, nbc) as usize;
                if chash == phash {
                    pp = BucketRef::Node(cur);
                } else if self.buckets[chash].is_none() {
                    self.buckets[chash] = Some(pp);
                    pp = BucketRef::Node(cur);
                    phash = chash;
                } else {
                    // Unique-keys branch: __np = __cp (no inner key_eq walk).
                    let np = cur;
                    // __pp->__next_ = __np->__next_;
                    let np_next = self.nodes[np].next;
                    self.set_ref_next(pp, np_next);
                    // __np->__next_ = __bucket_list_[__chash]->__next_;
                    let target = self.buckets[chash].unwrap();
                    let target_next = self.ref_next(target);
                    self.nodes[np].next = target_next;
                    // __bucket_list_[__chash]->__next_ = __cp;
                    self.set_ref_next(target, Some(cur));
                    // __pp stays; loop continues with __cp = __pp->__next_
                }
                cp = self.ref_next(pp);
            }
        }
    }

    /// `__rehash_unique(n)` → `__rehash<true>(n)` grow path.
    fn rehash_unique(&mut self, n: u64) {
        let nbc = rehash_bucket_count(n);
        if nbc > self.bucket_count() {
            self.do_rehash(nbc);
        }
        // We never shrink during a build of strictly-growing inserts.
    }

    /// `__node_insert_unique_perform(nd)`: splice node `nd` into the list and
    /// fix bucket pointers. `nd` must already be in the arena with its hash set
    /// and a known target bucket.
    fn node_insert_unique_perform(&mut self, nd: usize) {
        let bc = self.bucket_count();
        let chash = constrain_hash(self.nodes[nd].hash, bc) as usize;
        match self.buckets[chash] {
            None => {
                // Bucket empty: splice at the FRONT (right after the sentinel).
                let pn = BucketRef::Sentinel;
                // __nd->__next_ = __pn->__next_;
                self.nodes[nd].next = self.head;
                // __pn->__next_ = __nd;
                self.head = Some(nd);
                // __bucket_list_[__chash] = __pn;  (sentinel)
                self.buckets[chash] = Some(pn);
                // if (__nd->__next_ != nullptr)
                //   __bucket_list_[constrain(__nd->__next_->hash)] = __nd;
                if let Some(nxt) = self.nodes[nd].next {
                    let nxt_chash = constrain_hash(self.nodes[nxt].hash, bc) as usize;
                    self.buckets[nxt_chash] = Some(BucketRef::Node(nd));
                }
            }
            Some(pn) => {
                // Bucket non-empty: splice right after that bucket's "before"
                // node (i.e. immediately before its existing first element).
                let pn_next = self.ref_next(pn);
                self.nodes[nd].next = pn_next;
                self.set_ref_next(pn, Some(nd));
            }
        }
        self.size += 1;
    }

    /// `__node_insert_unique_prepare` rehash trigger + the full unique insert.
    ///
    /// On a duplicate key this models `operator[]`'s value assignment: the
    /// existing node stays in its original list position, but its stored value
    /// (here, the insertion index we ultimately report) is overwritten with the
    /// newer `idx` — exactly as `m[k] = i` overwrites the mapped value while
    /// leaving iteration position untouched. Returns the existing node's index
    /// if the key was already present (no new node), else `None`.
    fn insert_unique(
        &mut self,
        idx: usize,
        hash: u64,
        key: &[u8],
        keys: &[Vec<u8>],
    ) -> Option<usize> {
        // --- __node_insert_unique_prepare: existence check ---
        let bc = self.bucket_count();
        if bc != 0 {
            let chash = constrain_hash(hash, bc);
            if let Some(before) = self.buckets[chash as usize] {
                let mut ndptr = self.ref_next(before);
                while let Some(cur) = ndptr {
                    let ch = self.nodes[cur].hash;
                    if !(ch == hash || constrain_hash(ch, bc) == chash) {
                        break;
                    }
                    if ch == hash && keys[self.nodes[cur].idx] == key {
                        // operator[] on an existing key overwrites the value.
                        let prev = self.nodes[cur].idx;
                        self.nodes[cur].idx = idx;
                        return Some(prev);
                    }
                    ndptr = self.nodes[cur].next;
                }
            }
        }

        // --- rehash if load factor would exceed max_load_factor (1.0) ---
        // if (size() + 1 > bc * mlf || bc == 0)
        //   __rehash_unique(max(2*bc + !is_hash_power2(bc),
        //                       ceil((size()+1)/mlf)));
        // mlf == 1.0f, so `bc * 1.0` and `ceil((size+1)/1.0)` are exact.
        let bc = self.bucket_count();
        if self.size + 1 > bc || bc == 0 {
            let grow = 2u64
                .wrapping_mul(bc)
                .wrapping_add(u64::from(!is_hash_power2(bc)));
            let need = self.size + 1; // ceil((size+1)/1.0)
            self.rehash_unique(grow.max(need));
        }

        // --- perform the insert ---
        let nd = self.nodes.len();
        self.nodes.push(Node {
            idx,
            hash,
            next: None,
        });
        self.node_insert_unique_perform(nd);
        None
    }

    /// Walk the intrusive list head→tail, yielding insertion indices in
    /// `begin()..end()` order.
    fn iter_indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.size as usize);
        let mut cur = self.head;
        while let Some(i) = cur {
            out.push(self.nodes[i].idx);
            cur = self.nodes[i].next;
        }
        out
    }
}

/// Given keys in the order they were INSERTED into a libc++
/// `std::unordered_map<std::string, V>` (via `m[k] = i;`), return the keys'
/// insertion indices in the map's ITERATION order (`begin()..end()`).
///
/// Duplicate keys behave like `operator[]` assignment: the existing node keeps
/// its original iteration position (it is *not* moved), but its mapped value is
/// overwritten, so the index reported for a repeated key is the *last* one
/// written. Each distinct key therefore appears exactly once in the result.
pub fn libcxx_unordered_iteration_order(keys_in_insertion_order: &[String]) -> Vec<usize> {
    let key_bytes: Vec<Vec<u8>> = keys_in_insertion_order
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let mut table = Table::new();
    for (idx, kb) in key_bytes.iter().enumerate() {
        let hash = libcxx_string_hash(kb);
        table.insert_unique(idx, hash, kb, &key_bytes);
    }
    table.iter_indices()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vstr(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// Map an iteration-order index list back to the keys for readable asserts.
    fn order_keys<'a>(keys: &'a [String], order: &[usize]) -> Vec<&'a str> {
        order.iter().map(|&i| keys[i].as_str()).collect()
    }

    // --- std::hash<std::string> known-good values (system libc++) ---
    #[test]
    fn known_good_hashes() {
        let cases: &[(&str, u64)] = &[
            ("std/fs", 0x1ca4_c2e2_0494_201a),
            ("std/box", 0x854d_a794_1f29_9821),
            ("std/string", 0x415e_809c_7ab5_aa7e),
            ("std/process", 0xb1d3_0add_0c89_7d28),
            ("std/collections", 0x58f5_5832_94c7_007a),
            ("std/option", 0x2880_a1fe_bb86_2b98),
            ("std/fmt", 0x76db_1b38_fd4b_3c32),
            ("std", 0x0e57_9c9d_5463_1b22),
            ("test", 0x67c1_6835_6ac8_52a7),
        ];
        for (s, want) in cases {
            assert_eq!(
                libcxx_string_hash(s.as_bytes()),
                *want,
                "hash mismatch for {s:?}"
            );
        }
    }

    // hash<""> == __k2 (the len==0 path returns K2).
    #[test]
    fn hash_empty_is_k2() {
        assert_eq!(libcxx_string_hash(b""), 0x9ae1_6a3b_2f90_404f);
    }

    // Exercises the 17..32, 33..64, and >64 paths (values captured from probe).
    #[test]
    fn hash_long_strings() {
        // len 20 (17..32 path)
        assert_eq!(
            libcxx_string_hash(b"abcdefghijklmnopqrst"),
            0x5b90_8404_a851_21a6
        );
        // len 40 (33..64 path)
        assert_eq!(
            libcxx_string_hash(b"abcdefghijklmnopqrstuvwxyz0123456789ABCD"),
            0x0b73_978c_47b8_9ead
        );
        // len 82 (>64 main loop)
        assert_eq!(
            libcxx_string_hash(
                b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghij"
            ),
            0x6317_622a_c6e5_59e6
        );
    }

    // --- The canonical case from the spec ---
    #[test]
    fn canonical_module_paths() {
        let keys = vstr(&[
            "std/fs",
            "std/box",
            "std/string",
            "std/process",
            "std/collections",
            "std/option",
            "std/fmt",
            "std",
            "test",
        ]);
        let order = libcxx_unordered_iteration_order(&keys);
        assert_eq!(
            order_keys(&keys, &order),
            vec![
                "test",
                "std",
                "std/fmt",
                "std/option",
                "std/collections",
                "std/box",
                "std/process",
                "std/string",
                "std/fs",
            ]
        );
    }

    // Same set, different insertion order. The bucket_count is the same (11,
    // the set has 9 elements), but the per-bucket chain order depends on
    // insertion order because new nodes splice at the front of an empty bucket
    // and just after the existing first node of a non-empty bucket. So the
    // iteration order differs from `canonical_module_paths` — captured from the
    // C++ probe for THIS specific insertion order.
    #[test]
    fn module_paths_reversed_insertion() {
        let keys = vstr(&[
            "test",
            "std",
            "std/fmt",
            "std/option",
            "std/collections",
            "std/process",
            "std/string",
            "std/box",
            "std/fs",
        ]);
        let order = libcxx_unordered_iteration_order(&keys);
        // begin..end (probe indices 8 6 7 5 4 3 2 1 0) for THIS insertion order.
        assert_eq!(
            order_keys(&keys, &order),
            vec![
                "std/fs",
                "std/string",
                "std/box",
                "std/process",
                "std/collections",
                "std/option",
                "std/fmt",
                "std",
                "test",
            ]
        );
    }

    #[test]
    fn single_key() {
        let keys = vstr(&["onlyone"]);
        let order = libcxx_unordered_iteration_order(&keys);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn empty() {
        let keys: Vec<String> = vec![];
        let order = libcxx_unordered_iteration_order(&keys);
        assert!(order.is_empty());
    }

    #[test]
    fn two_keys() {
        let keys = vstr(&["alpha", "beta"]);
        let order = libcxx_unordered_iteration_order(&keys);
        // begin..end captured from probe: bucket_count=2, order = beta, alpha.
        assert_eq!(order_keys(&keys, &order), vec!["beta", "alpha"]);
    }

    #[test]
    fn duplicate_key_overwrites_value() {
        let keys = vstr(&["a", "b", "a", "c"]);
        let order = libcxx_unordered_iteration_order(&keys);
        // 3 distinct keys -> 3 entries. The repeated "a" keeps its ORIGINAL
        // position but its reported index becomes the LAST write (2), matching
        // `m["a"]=2` overwriting the value. Probe order indices: 1 3 2.
        assert_eq!(order, vec![1, 3, 2]);
        assert_eq!(order_keys(&keys, &order), vec!["b", "c", "a"]);
    }

    // Sequential keys "k0".."k29" — exercises several rehashes/prime growth.
    // begin..end captured from the C++ probe.
    #[test]
    fn sequential_keys_30() {
        let keys: Vec<String> = (0..30).map(|i| format!("k{i}")).collect();
        let order = libcxx_unordered_iteration_order(&keys);
        // Probe indices: 29 27 25 24 23 0 21 19 20 17 14 12 26 16 8 22 18 6 10
        //                4 9 7 15 28 5 1 11 3 13 2
        let want_idx = vec![
            29usize, 27, 25, 24, 23, 0, 21, 19, 20, 17, 14, 12, 26, 16, 8, 22, 18, 6, 10, 4, 9, 7,
            15, 28, 5, 1, 11, 3, 13, 2,
        ];
        assert_eq!(order, want_idx);
    }
}
