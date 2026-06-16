/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

mod cli;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        cli::display_help();
        return;
    }
    let command = &args[1];
    match command.as_str() {
        "build" => {
            cli::progress::ProgressGuard::new();
        }
        "version" => {
            println!("0.0.1");
        }
        "help" => {
            cli::display_help();
        }
        _ => {
            cli::display_help();
        }
    }
}
