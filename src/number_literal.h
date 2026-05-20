/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#ifndef NUMBER_LITERAL_H
#define NUMBER_LITERAL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

enum class NumberBase : uint8_t {
	Decimal = 10,
	Hex = 16,
	Binary = 2,
	Octal = 8,
};

// Per-error variants with the position where the problem was found.
// Position is a byte offset into the lexeme, not into the source file.
enum class NumberErrorKind {
	LeadingZero,                    // "07" — leading zero on decimal
	DigitAfterBase,                 // "0x" with no digit
	UpperCaseBase,                  // "0X42" — uppercase prefix
	InvalidFloatBase,               // float in non-decimal/hex base
	RepeatedUnderscore,             // "1__0"
	InvalidUnderscoreAfterSpecial,  // "0x_1", "1._5", etc.
	InvalidDigit,                   // digit out of range for the base
	InvalidDigitExponent,           // non-decimal digit in an exponent
	DuplicatePeriod,                // "1.0.5"
	DuplicateExponent,              // "1e10e2"
	InvalidHexExponent,             // hex literal with `e`/`E` exponent
	ExponentAfterUnderscore,        // "1_e10"
	SpecialAfterUnderscore,         // "1_."
	TrailingSpecial,                // "1.", "1+", "1e"
	TrailingUnderscore,             // "1_"
	InvalidCharacter,               // out of [0-9a-zA-Z._+-]
	InvalidExponentSign,            // "+" or "-" not after p/P/e/E
	IntegerTooLarge,                // overflows u64 (caller asks for int)
};

struct NumberError {
	NumberErrorKind kind;
	std::size_t pos;
	NumberBase base;  // meaningful for InvalidDigit, otherwise Decimal
};

enum class NumberResultKind {
	Int,
	BigInt,
	Float,
	Failure,
};

struct NumberResult {
	NumberResultKind kind = NumberResultKind::Failure;
	uint64_t intValue = 0;
	double floatValue = 0.0;
	NumberBase base = NumberBase::Decimal;
	NumberError failure{};
};

NumberResult parseNumberLiteral(std::string_view bytes);
const char *numberErrorMessage(NumberErrorKind kind);

#endif  // NUMBER_LITERAL_H
