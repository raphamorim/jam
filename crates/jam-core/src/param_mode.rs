/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ParamMode {
    #[default]
    // read-only borrow. Initialized at entry, cannot be reassigned,
    // the caller's binding is unchanged.
    // Not owned, so the callee does not drop it.
    Let = 0,
    // exclusive read-write borrow. Initialized at entry, may be read and
    // written. The caller's binding stays initialized after the call.
    Mut = 1,
    // consume ownership. The callee owns the value and drops it; the
    // caller's binding becomes uninitialized after the call.
    Move = 2,
}

impl ParamMode {
    pub fn from_u8(v: u8) -> ParamMode {
        match v {
            1 => ParamMode::Mut,
            2 => ParamMode::Move,
            _ => ParamMode::Let,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_cpp() {
        assert_eq!(ParamMode::Let as u8, 0);
        assert_eq!(ParamMode::Mut as u8, 1);
        assert_eq!(ParamMode::Move as u8, 2);
        assert_eq!(ParamMode::default(), ParamMode::Let);
        assert_eq!(ParamMode::from_u8(2), ParamMode::Move);
        assert_eq!(ParamMode::from_u8(7), ParamMode::Let);
    }
}
