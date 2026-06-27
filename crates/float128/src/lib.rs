/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

// the IEEE-754 binary128 (often called quadruple precision or quad precision)
// is a standard for representing floating-point numbers in computers.

// for anyone contributing to this (maybe in the future idk):
// this crate should be fully in rust and dep free.

// we use float128 due to LLVM APFloat
// https://llvm.org/doxygen/classllvm_1_1APFloat.html
// Quadruple-precision floating-point format
// https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format

// The IEEE 754 standard specifies a binary128 as having:
// Sign bit: 1bit (in our case it's on position 127, 0 for positive and 1 for negative)
// Exponent width: 15bits, it basically encodes the magnitude (positions 112 to 126).

// for example
// 15.3013
// 0     10000010     11101001101001000100000
// |     └──┬───┘     └──────────┬──────────┘
// sign  exponent            mantissa

// -15.3013
// 1     10000010     11101001101001000100000
// |     └──┬───┘     └──────────┬──────────┘
// sign  exponent            mantissa
