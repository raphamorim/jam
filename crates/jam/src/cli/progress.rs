// Terminal progress reporting via OSC 9;4. Supported by ConEmu,
// Rio Terminal, Windows Terminal, iTerm2, WezTerm, Ghostty, etc.

// 0=remove, 1=normal+percent, 2=error, 3=indeterminate,
// 4=warning.

// in jam we use 3 during compile, 2 on error, 0 on success.
pub struct ProgressGuard {
	active: bool
}

impl Drop for ProgressGuard {
	fn drop(&mut self) {
    self.stop();
  }
}

impl ProgressGuard {
	pub fn error(&mut self) {
		if self.active {
			println!("\033]9;4;2;\033\\");
			self.active = false;
		}
	}

	pub fn stop(&mut self) {
		if self.active {
			println!("\033]9;4;0;\033\\");
			self.active = false;
		}

	}
}

// Pr
//   public:
// 	ProgressGuard(bool enabled) {
// 		if (!enabled) return;
// 		if (!isatty(STDERR_FILENO)) return;
// 		active = true;
// 		std::cerr << "\033]9;4;3;\033\\" << std::flush;
// 	}
// 	~ProgressGuard() { clear(); }

// 	void error() {
// 		if (active) {
// 			std::cerr << "\033]9;4;2;\033\\" << std::flush;
// 			active = false;
// 		}
// 	}
// 	// Stop without flagging error — same exit as success.
// 	void stop() { clear(); }

//   private:
// 	void clear() {
// 		if (active) {
// 			std::cerr << "\033]9;4;0;\033\\" << std::flush;
// 			active = false;
// 		}
// 	}
// };