.PHONY: build release test test-unit test-release test-x86_64 install uninstall clean fmt lint check-format info docs
.DEFAULT_GOAL := build

# Check if we're on macOS or Linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM = macOS
    PREFIX ?= $(HOME)/.local
else ifeq ($(UNAME_S),Linux)
    PLATFORM = Linux
    PREFIX ?= /usr/local
else
    PLATFORM = Unknown
    PREFIX ?= /usr/local
endif

BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib
STDDIR ?= $(LIBDIR)/jam/std

# The compiler is the Rust workspace; `jam --version` gets its git SHA
# baked in by crates/jam/build.rs, so no flags are needed here. LLVM is
# located by crates/jam-llvm/build.rs (LLVM_CONFIG / LLVM_SYS_221_PREFIX /
# PATH / Homebrew, in that order).
build:
	cargo build -p jam

release:
	cargo build --release -p jam

# Rust unit tests + the .jam corpus (must-pass tfn files and the
# `// expect-error:` must-fail files under tests/).
test: build
	cargo test --workspace
	./target/debug/jam test tests

test-unit: build
	./target/debug/jam test tests/unit

test-release: release
	./target/release/jam -C opt-level=3 test tests

# The whole corpus cross-compiled for x86_64 and executed under
# Rosetta 2 (macOS runs the binaries transparently).
test-x86_64: build
	./target/debug/jam -C target=x86_64-apple-darwin test tests

install: release
	@echo "Installing Jam compiler to $(PREFIX)..."
	@echo "Platform: $(PLATFORM)"
	install -d $(BINDIR)
	cp target/release/jam $(BINDIR)/jam
	chmod 755 $(BINDIR)/jam
	@echo "Installing Jam standard library to $(STDDIR)..."
	install -d $(STDDIR)
	rm -rf $(STDDIR)
	cp -R ./std $(STDDIR)

uninstall:
	@echo "Uninstalling Jam compiler..."
	rm -f $(BINDIR)/jam
	rm -rf $(LIBDIR)/jam

clean:
	cargo clean

fmt:
	cargo fmt --all

lint:
	cargo fmt -- --check --color always
	cargo clippy --all-targets --all-features -- -D warnings

check-format:
	cargo fmt --all -- --check

info:
	@echo "Platform: $(PLATFORM)"
	@echo "Install Prefix: $(PREFIX)"
	@echo "Binary Directory: $(BINDIR)"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the compiler (debug)"
	@echo "  make release        - Build the compiler (release)"
	@echo "  make test           - Rust unit tests + the .jam corpus"
	@echo "  make test-unit      - Run the must-pass .jam corpus only"
	@echo "  make test-release   - Corpus at -C opt-level=3"
	@echo "  make test-x86_64    - Corpus as x86_64 under Rosetta"
	@echo "  make install        - Install jam + std to $(PREFIX)"
	@echo "  make fmt            - cargo fmt"
	@echo "  make lint           - fmt --check + clippy"
	@echo "  make docs           - Serve the website locally"

docs:
	@echo "Serving documentation at http://localhost:4000..."
	cd docs && bundle install && bundle exec jekyll serve --livereload
