.PHONY: build check-llvm install uninstall clean cmake-build cmake-install cmake-uninstall test test-release test-unit test-unit-release test-init test-codegen-errors test-diagnostics test-decl test-analyzer test-comptime test-print docs format check-format fmt info
.DEFAULT_GOAL := build

LLVM_CONFIG=$(shell which llvm-config 2>/dev/null || echo "llvm-config")
OPTFLAGS ?= -O2 -DNDEBUG

# Short SHA of the build commit, baked into the binary so `jam --version`
# can report exactly which tree it was built from. Falls back to
# `unknown` outside a git checkout. Appends `-dirty` when the worktree
# has uncommitted changes — keeps "this isn't quite the tagged build"
# obvious in bug reports.
JAM_VERSION_SHA := $(shell \
  if git rev-parse --short HEAD >/dev/null 2>&1; then \
    sha=$$(git rev-parse --short HEAD); \
    if ! git diff --quiet HEAD 2>/dev/null; then sha="$$sha-dirty"; fi; \
    echo "$$sha"; \
  else \
    echo "unknown"; \
  fi)
VERSION_FLAGS := -DJAM_VERSION_SHA=\"$(JAM_VERSION_SHA)\"

CLANG_FORMAT ?= clang-format
CLANG_FORMAT_STYLE := file:clang-format
FORMAT_SOURCES := $(wildcard src/*.cpp src/*.h) $(wildcard tests/cpp/*.cpp tests/cpp/*.h)

OUT := output

SRC_NAMES := jam_llvm main lexer parser codegen target cabi \
             module_resolver symbol_table number_literal \
             init_analysis drop_registry abi diagnostics decl \
             analyzer comptime astgen jir_codegen jir_verify
OBJS := $(addprefix $(OUT)/, $(addsuffix .o, $(SRC_NAMES)))
# Compiler objects sans `main.o` — C++ tests in tests/cpp ship their own
# `main()`; linking them against the full OBJS list pulls in jam's CLI
# entry point and triggers a duplicate-symbol error.
LIB_OBJS := $(filter-out $(OUT)/main.o,$(OBJS))

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

# Incremental build: real per-object rules with compiler-generated
# header dependencies (-MMD), so an unchanged file is never recompiled
# and `make -j8` parallelizes the LLVM-header TUs. The old recipe was a
# serial shell for-loop that rebuilt all 19 objects on every invocation
# — ~14s of pure waste per `make test` with no changes.
LLVM_CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags 2>/dev/null)
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs 2>/dev/null)

# Version stamp: main.o bakes JAM_VERSION_SHA, which flips whenever the
# worktree dirty-state changes. Touch the stamp file only when the SHA
# actually differs so a flip rebuilds main.o alone (not all 19 TUs),
# and an unchanged SHA rebuilds nothing.
VERSION_STAMP := $(OUT)/.version_sha
$(shell mkdir -p $(OUT))
$(shell [ "`cat $(VERSION_STAMP) 2>/dev/null`" = "$(JAM_VERSION_SHA)" ] || echo "$(JAM_VERSION_SHA)" > $(VERSION_STAMP))

check-llvm:
	@if ! command -v $(LLVM_CONFIG) >/dev/null 2>&1; then \
		echo "error: llvm-config not found."; \
		exit 1; \
	fi

$(OUT):
	@mkdir -p $(OUT)

$(OUT)/%.o: src/%.cpp | $(OUT)
	@echo "  CC:  $< -> $@"
	@clang++ -c $< -o $@ $(LLVM_CXXFLAGS) -fexceptions $(OPTFLAGS) -MMD -MP $(EXTRA_CXXFLAGS)

$(OUT)/main.o: EXTRA_CXXFLAGS := $(VERSION_FLAGS)
$(OUT)/main.o: $(VERSION_STAMP)

$(OUT)/jam.out: $(OBJS)
	@echo "  LD: $(OUT)/jam.out"
	@clang++ -o $(OUT)/jam.out $(OBJS) $(LLVM_LDFLAGS)

build: check-llvm $(OUT)/jam.out

# C++ test objects share the pattern-rule + depfile treatment so
# unchanged test TUs (each compiles LLVM headers at -O2, multiple
# seconds) are skipped too.
$(OUT)/test_%.o: tests/cpp/test_%.cpp | $(OUT)
	@echo "  CC:  $< -> $@"
	@clang++ -c $< -o $@ $(LLVM_CXXFLAGS) -fexceptions $(OPTFLAGS) -MMD -MP

-include $(wildcard $(OUT)/*.d)

cmake-build:
	@echo "Building with CMake..."
	@mkdir -p build
	cd build && cmake .. && cmake --build .

cmake-install: cmake-build
	@echo "Installing Jam compiler using CMake..."
	cd build && make install
	@echo ""
	@echo "Jam compiler installed successfully!"
	@echo "Try: jam --help"

cmake-uninstall:
	@echo "Uninstalling Jam compiler..."
	@if [ -f build/cmake_uninstall.cmake ]; then \
		cd build && make uninstall; \
		echo "Jam compiler uninstalled successfully!"; \
	else \
		echo "No installation found. Run 'make cmake-install' first."; \
	fi

install: build
	@echo "Installing Jam compiler to $(PREFIX)..."
	@echo "Platform: $(PLATFORM)"
	install -d $(BINDIR)
	cp $(OUT)/jam.out $(BINDIR)/jam
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
	rm -rf $(OUT) build/

info:
	@echo "Platform: $(PLATFORM)"
	@echo "LLVM Config: $(LLVM_CONFIG)"
	@echo "Install Prefix: $(PREFIX)"
	@echo "Binary Directory: $(BINDIR)"
	@echo "Build Output: $(OUT)/"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the compiler"
	@echo "  make test-unit      - Run Jam unit tests (debug, -O0)"
	@echo "  make test-unit-release - Run Jam unit tests (release, -O3)"
	@echo "  make test           - Run all tests at default (debug)"
	@echo "  make test-release   - Run all tests + release (-O3)"
	@echo "  make docs           - Serve documentation site"
	@echo "  make install        - Install using manual method"
	@echo "  make uninstall      - Uninstall manual installation"
	@echo "  make cmake-install  - Install using CMake"
	@echo "  make cmake-uninstall- Uninstall CMake installation"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make info           - Show this information"

test-unit: build
	@echo "Running Jam unit tests (debug, -C opt-level=0)..."
	$(OUT)/jam.out test tests/unit

# -O3 (release) pass over the same unit suite.
# runs the optimizer to catch bugs that only surface when LLVM's
# passes are active (dead-load explosions, GEP/Load coalescing,
# mem2reg interactions, etc.).
test-unit-release: build
	@echo "Running Jam unit tests (release, -C opt-level=3)..."
	$(OUT)/jam.out -C opt-level=3 test tests/unit

# Test binaries are real file targets: an unchanged test TU (each
# compiles LLVM headers at -O2 — seconds apiece) is never recompiled,
# and an unchanged binary is never relinked. The phony test-* targets
# just run them.
$(OUT)/init_analysis_tests: $(OUT)/test_init_analysis.o $(LIB_OBJS)
	@echo "  LD: $@"
	@clang++ -o $@ $^ $(LLVM_LDFLAGS)

$(OUT)/analyzer_tests: $(OUT)/test_analyzer.o $(LIB_OBJS)
	@echo "  LD: $@"
	@clang++ -o $@ $^ $(LLVM_LDFLAGS)

$(OUT)/codegen_error_tests: $(OUT)/test_codegen_errors.o
	@echo "  LD: $@"
	@clang++ -o $@ $^

$(OUT)/diagnostic_tests: $(OUT)/test_diagnostics.o
	@echo "  LD: $@"
	@clang++ -o $@ $^

$(OUT)/decl_tests: $(OUT)/test_decl_table.o $(OUT)/decl.o
	@echo "  LD: $@"
	@clang++ -o $@ $^

$(OUT)/comptime_tests: $(OUT)/test_comptime.o $(OUT)/comptime.o $(OUT)/diagnostics.o $(OUT)/jam_llvm.o $(OUT)/target.o
	@echo "  LD: $@"
	@clang++ -o $@ $^ $(LLVM_LDFLAGS)

$(OUT)/print_tests: $(OUT)/test_print.o
	@echo "  LD: $@"
	@clang++ -o $@ $^

test-init: build $(OUT)/init_analysis_tests
	@echo ""
	@echo "Building and running init_analysis C++ tests..."
	@$(OUT)/init_analysis_tests

test-codegen-errors: build $(OUT)/codegen_error_tests
	@echo ""
	@echo "Building and running codegen-error C++ tests..."
	@$(OUT)/codegen_error_tests

test-diagnostics: build $(OUT)/diagnostic_tests
	@echo ""
	@echo "Building and running diagnostic-pipeline tests..."
	@$(OUT)/diagnostic_tests

test-decl: build $(OUT)/decl_tests
	@echo ""
	@echo "Building and running DeclTable C++ tests..."
	@$(OUT)/decl_tests

test-analyzer: build $(OUT)/analyzer_tests
	@echo ""
	@echo "Building and running analyzer C++ tests..."
	@$(OUT)/analyzer_tests

test-comptime: build $(OUT)/comptime_tests
	@echo ""
	@echo "Building and running Comptime C++ tests..."
	@$(OUT)/comptime_tests

test-print: build $(OUT)/print_tests
	@echo ""
	@echo "Building and running @-emit cfn-print end-to-end tests..."
	@$(OUT)/print_tests

test: test-unit test-init test-codegen-errors test-diagnostics test-decl test-analyzer test-comptime test-print
test-release: test test-unit-release

fmt: format

lint:
	cargo fmt -- --check --color always
	cargo clippy --all-targets --all-features -- -D warnings

format:
	@echo "Formatting $(words $(FORMAT_SOURCES)) C++ file(s)..."
	@$(CLANG_FORMAT) --style=$(CLANG_FORMAT_STYLE) -i $(FORMAT_SOURCES)
	@echo "Done."

check-format:
	@echo "Checking format of $(words $(FORMAT_SOURCES)) C++ file(s)..."
	@$(CLANG_FORMAT) --style=$(CLANG_FORMAT_STYLE) --dry-run --Werror $(FORMAT_SOURCES)

docs:
	@echo "Serving documentation at http://localhost:4000..."
	cd docs && bundle install && bundle exec jekyll serve --livereload
