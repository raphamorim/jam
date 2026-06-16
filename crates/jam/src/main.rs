mod cli;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2  {
        cli::display_help();
        return;
    }
    let command = &args[1];
    match command.as_str() {
        "version" => {
            println!("0.0.1");
        },
        "help" | _ => {
            cli::display_help();
        }
    }
}

