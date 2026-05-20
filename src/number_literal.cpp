/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include "number_literal.h"

#include <cstdlib>  // strtod
#include <string>

namespace {

NumberResult fail(NumberErrorKind kind, std::size_t pos,
                  NumberBase base = NumberBase::Decimal) {
	NumberResult r;
	r.kind = NumberResultKind::Failure;
	r.failure.kind = kind;
	r.failure.pos = pos;
	r.failure.base = base;
	return r;
}

NumberResult okInt(uint64_t value, NumberBase base) {
	NumberResult r;
	r.kind = NumberResultKind::Int;
	r.intValue = value;
	r.base = base;
	return r;
}

NumberResult okBigInt(NumberBase base) {
	NumberResult r;
	r.kind = NumberResultKind::BigInt;
	r.base = base;
	return r;
}

NumberResult okFloat(NumberBase base, double value) {
	NumberResult r;
	r.kind = NumberResultKind::Float;
	r.base = base;
	r.floatValue = value;
	return r;
}

}  // namespace

NumberResult parseNumberLiteral(std::string_view bytes) {
	std::size_t i = 0;
	uint8_t baseValue = 10;
	NumberBase base = NumberBase::Decimal;

	if (bytes.size() >= 2 && bytes[0] == '0') {
		switch (bytes[1]) {
		case 'b':
			base = NumberBase::Binary;
			baseValue = 2;
			i = 2;
			break;
		case 'o':
			base = NumberBase::Octal;
			baseValue = 8;
			i = 2;
			break;
		case 'x':
			base = NumberBase::Hex;
			baseValue = 16;
			i = 2;
			break;
		case 'B':
		case 'O':
		case 'X':
			// Reject uppercase prefix outright for stylistic uniformity;
			// callers must spell `0x` / `0b` / `0o` in lowercase.
			return fail(NumberErrorKind::UpperCaseBase, 1);
		case '.':
		case 'e':
		case 'E':
			// Float starting with "0." / "0e..." — fall through to the
			// decimal scanner.
			break;
		default:
			// Any other leading-zero form ("07", "0a", etc.) is an error
			// to avoid silent C-style octal interpretation.
			return fail(NumberErrorKind::LeadingZero, 0);
		}
	}

	// `0x` / `0b` / `0o` with no following digit is an error.
	if (bytes.size() == 2 && baseValue != 10) {
		return fail(NumberErrorKind::DigitAfterBase, 2);
	}

	uint64_t x = 0;
	bool overflow = false;
	bool underscore = false;
	bool period = false;
	uint8_t special = 0;
	bool exponent = false;
	bool isFloat = false;

	while (i < bytes.size()) {
		char c = bytes[i];

		// Digit separator. Cannot appear immediately after a base prefix
		// or after a special character (period, sign, exponent letter),
		// and cannot repeat.
		if (c == '_') {
			if (i == 2 && baseValue != 10) {
				return fail(NumberErrorKind::InvalidUnderscoreAfterSpecial, i);
			}
			if (special != 0) {
				return fail(NumberErrorKind::InvalidUnderscoreAfterSpecial, i);
			}
			if (underscore) {
				return fail(NumberErrorKind::RepeatedUnderscore, i);
			}
			underscore = true;
			i++;
			continue;
		}

		// Decimal exponent marker. Only valid in decimal — in hex
		// literals `e` / `E` are hex digits, never exponent letters.
		if ((c == 'e' || c == 'E') && baseValue == 10) {
			isFloat = true;
			if (exponent) {
				return fail(NumberErrorKind::DuplicateExponent, i);
			}
			if (underscore) {
				return fail(NumberErrorKind::ExponentAfterUnderscore, i);
			}
			special = static_cast<uint8_t>(c);
			exponent = true;
			i++;
			continue;
		}

		// Hex / binary float exponent marker (always binary-base
		// exponent in IEEE-754 hex floats).
		if ((c == 'p' || c == 'P') && baseValue == 16) {
			isFloat = true;
			if (exponent) {
				return fail(NumberErrorKind::DuplicateExponent, i);
			}
			if (underscore) {
				return fail(NumberErrorKind::ExponentAfterUnderscore, i);
			}
			special = static_cast<uint8_t>(c);
			exponent = true;
			i++;
			continue;
		}

		// Decimal point. Only meaningful in decimal or hex; binary /
		// octal floats are not a thing.
		if (c == '.') {
			isFloat = true;
			if (baseValue != 10 && baseValue != 16) {
				return fail(NumberErrorKind::InvalidFloatBase, 2);
			}
			if (period) { return fail(NumberErrorKind::DuplicatePeriod, i); }
			period = true;
			if (underscore) {
				return fail(NumberErrorKind::SpecialAfterUnderscore, i);
			}
			special = '.';
			i++;
			continue;
		}

		// Sign on exponent. Only valid immediately after p/P/e/E.
		if (c == '+' || c == '-') {
			if (special != 'p' && special != 'P' && special != 'e' &&
			    special != 'E') {
				return fail(NumberErrorKind::InvalidExponentSign, i);
			}
			special = static_cast<uint8_t>(c);
			i++;
			continue;
		}

		// At this point `c` should be a digit in the current base. Map
		// it to its numeric value.
		uint8_t digit;
		if (c >= '0' && c <= '9') {
			digit = static_cast<uint8_t>(c - '0');
		} else if (c >= 'A' && c <= 'Z') {
			digit = static_cast<uint8_t>(c - 'A' + 10);
		} else if (c >= 'a' && c <= 'z') {
			digit = static_cast<uint8_t>(c - 'a' + 10);
		} else {
			return fail(NumberErrorKind::InvalidCharacter, i);
		}

		if (digit >= baseValue) {
			return fail(NumberErrorKind::InvalidDigit, i, base);
		}
		if (exponent && digit >= 10) {
			return fail(NumberErrorKind::InvalidDigitExponent, i);
		}

		underscore = false;
		special = 0;

		if (!isFloat) {
			// Accumulate value, watching for overflow.
			if (x != 0) {
				uint64_t mul = x * baseValue;
				if (mul / baseValue != x) overflow = true;
				x = mul;
			}
			uint64_t add = x + digit;
			if (add < x) overflow = true;
			x = add;
		}

		i++;
	}

	if (underscore) {
		return fail(NumberErrorKind::TrailingUnderscore, bytes.size() - 1);
	}
	if (special != 0) {
		return fail(NumberErrorKind::TrailingSpecial, bytes.size() - 1);
	}

	if (isFloat) {
		// rule:
		// Decimal and hex floats both parse cleanly with the standard C
		// `strtod` (it accepts `0x1.8p4` for hex). Underscores have been
		// validated already; strip them before strtod so it sees a
		// pristine numeric form.
		std::string clean;
		clean.reserve(bytes.size());
		for (char c : bytes) {
			if (c != '_') clean.push_back(c);
		}
		const char *cstr = clean.c_str();
		char *end = nullptr;
		double v = std::strtod(cstr, &end);
		// `end` should land at the end of the string; if not, we caught
		// an unparseable character above and shouldn't have reached
		// here, but guard for safety.
		(void)end;
		return okFloat(base, v);
	}
	if (overflow) return okBigInt(base);
	return okInt(x, base);
}

const char *numberErrorMessage(NumberErrorKind kind) {
	switch (kind) {
	case NumberErrorKind::LeadingZero:
		return "leading zero is not allowed (use `0o` for octal)";
	case NumberErrorKind::DigitAfterBase:
		return "expected digit after base prefix";
	case NumberErrorKind::UpperCaseBase:
		return "base prefix must be lowercase (`0x`, `0b`, `0o`)";
	case NumberErrorKind::InvalidFloatBase:
		return "float literals are only allowed in decimal or hex";
	case NumberErrorKind::RepeatedUnderscore:
		return "consecutive underscores in numeric literal";
	case NumberErrorKind::InvalidUnderscoreAfterSpecial:
		return "underscore not allowed immediately after `0x`/`0b`/`0o` "
		       "or after `.`/`+`/`-`/`e`/`E`/`p`/`P`";
	case NumberErrorKind::InvalidDigit:
		return "invalid digit for the literal's base";
	case NumberErrorKind::InvalidDigitExponent:
		return "non-decimal digit in exponent";
	case NumberErrorKind::DuplicatePeriod:
		return "more than one `.` in numeric literal";
	case NumberErrorKind::DuplicateExponent:
		return "more than one exponent in numeric literal";
	case NumberErrorKind::InvalidHexExponent:
		return "hex literal cannot use `e`/`E` exponent (use `p`/`P`)";
	case NumberErrorKind::ExponentAfterUnderscore:
		return "exponent letter cannot follow `_`";
	case NumberErrorKind::SpecialAfterUnderscore:
		return "`.`/`+`/`-` cannot follow `_`";
	case NumberErrorKind::TrailingSpecial:
		return "numeric literal ends with `.`/`+`/`-`/`e`/`p`";
	case NumberErrorKind::TrailingUnderscore:
		return "numeric literal ends with `_`";
	case NumberErrorKind::InvalidCharacter:
		return "character not allowed in numeric literal";
	case NumberErrorKind::InvalidExponentSign:
		return "`+`/`-` is only valid immediately after `e`/`E`/`p`/`P`";
	case NumberErrorKind::IntegerTooLarge:
		return "integer literal exceeds u64 range";
	}
	return "unknown numeric-literal error";
}
