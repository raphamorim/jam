.PHONY: build install uninstall clean cmake-build cmake-install cmake-uninstall test test-unit docs format check-format
.DEFAULT_GOAL := build

LLVM_CONFIG=$(shell which llvm-config 2>/dev/null || echo "llvm-config")
OPTFLAGS ?= -O2 -DNDEBUG

CLANG_FORMAT ?= clang-format
CLANG_FORMAT_STYLE := file:clang-format
FORMAT_SOURCES := $(wildcard src/*.cpp src/*.h) $(wildcard tests/cpp/*.cpp tests/cpp/*.h)

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

build:
	@if ! command -v $(LLVM_CONFIG) >/dev/null 2>&1; then \
		echo "error: llvm-config not found."; \
		exit 1; \
	fi
	clang++ -c ./src/jam_llvm.cpp -o ./jam_llvm.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/main.cpp -o ./main.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/lexer.cpp -o ./lexer.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/parser.cpp -o ./parser.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/codegen.cpp -o ./codegen.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/target.cpp -o ./target.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/cabi.cpp -o ./cabi.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/module_resolver.cpp -o ./module_resolver.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/symbol_table.cpp -o ./symbol_table.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/number_literal.cpp -o ./number_literal.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/init_analysis.cpp -o ./init_analysis.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/drop_registry.cpp -o ./drop_registry.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/abi.cpp -o ./abi.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/diagnostics.cpp -o ./diagnostics.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/decl.cpp -o ./decl.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/analyzer.cpp -o ./analyzer.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/astgen.cpp -o ./astgen.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/jir_codegen.cpp -o ./jir_codegen.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -c ./src/jir_verify.cpp -o ./jir_verify.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./jam.out ./jam_llvm.o ./main.o ./lexer.o ./parser.o ./codegen.o ./target.o ./cabi.o ./module_resolver.o ./symbol_table.o ./number_literal.o ./init_analysis.o ./drop_registry.o ./abi.o ./diagnostics.o ./decl.o ./analyzer.o ./astgen.o ./jir_codegen.o ./jir_verify.o `$(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs`

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
	cp ./jam.out $(BINDIR)/jam
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
	rm -f ./jam_llvm.o ./main.o ./lexer.o ./parser.o ./codegen.o ./target.o ./cabi.o ./module_resolver.o ./symbol_table.o ./number_literal.o ./init_analysis.o ./drop_registry.o ./abi.o ./diagnostics.o ./decl.o ./analyzer.o ./astgen.o ./jir_codegen.o ./jir_verify.o ./jam.out
	rm -rf build/

info:
	@echo "Platform: $(PLATFORM)"
	@echo "LLVM Config: $(LLVM_CONFIG)"
	@echo "Install Prefix: $(PREFIX)"
	@echo "Binary Directory: $(BINDIR)"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the compiler"
	@echo "  make test-unit      - Run Jam unit tests"
	@echo "  make test           - Run all tests (Jam + C++)"
	@echo "  make docs           - Serve documentation site"
	@echo "  make install        - Install using manual method"
	@echo "  make uninstall      - Uninstall manual installation"
	@echo "  make cmake-install  - Install using CMake (recommended)"
	@echo "  make cmake-uninstall- Uninstall CMake installation"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make info           - Show this information"

test-unit: build
	@echo "Running Jam unit tests..."
	./jam.out test tests/unit

test-init: build
	@echo ""
	@echo "Building and running init-analyzer C++ tests..."
	clang++ -c ./tests/cpp/test_init_analysis.cpp -o ./test_init_analysis.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./init_tests ./test_init_analysis.o ./jam_llvm.o ./lexer.o ./parser.o ./codegen.o ./target.o ./cabi.o ./module_resolver.o ./symbol_table.o ./number_literal.o ./init_analysis.o ./drop_registry.o ./abi.o ./diagnostics.o ./decl.o ./analyzer.o ./astgen.o ./jir_codegen.o ./jir_verify.o `$(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs`
	./init_tests

test-abi: build
	@echo ""
	@echo "Building and running ABI classifier C++ tests..."
	clang++ -c ./tests/cpp/test_abi.cpp -o ./test_abi.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./abi_tests ./test_abi.o ./jam_llvm.o ./lexer.o ./parser.o ./codegen.o ./target.o ./cabi.o ./module_resolver.o ./symbol_table.o ./number_literal.o ./init_analysis.o ./drop_registry.o ./abi.o ./diagnostics.o ./decl.o ./analyzer.o ./astgen.o ./jir_codegen.o ./jir_verify.o `$(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs`
	./abi_tests

test-codegen-errors: build
	@echo ""
	@echo "Building and running codegen-error C++ tests..."
	clang++ -c ./tests/cpp/test_codegen_errors.cpp -o ./test_codegen_errors.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./codegen_error_tests ./test_codegen_errors.o
	./codegen_error_tests

test-jir: build
	@echo ""
	@echo "Building and running JIR C++ tests..."
	clang++ -c ./tests/cpp/test_jir_skeleton.cpp -o ./test_jir_skeleton.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./jir_tests ./test_jir_skeleton.o
	./jir_tests

test-diagnostics: build
	@echo ""
	@echo "Building and running diagnostic-pipeline tests..."
	clang++ -c ./tests/cpp/test_diagnostics.cpp -o ./test_diagnostics.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./diagnostic_tests ./test_diagnostics.o
	./diagnostic_tests

test-decl: build
	@echo ""
	@echo "Building and running DeclTable C++ tests..."
	clang++ -c ./tests/cpp/test_decl_table.cpp -o ./test_decl_table.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./decl_tests ./test_decl_table.o ./decl.o
	./decl_tests

test-analyzer: build
	@echo ""
	@echo "Building and running Analyzer C++ tests..."
	clang++ -c ./tests/cpp/test_analyzer.cpp -o ./test_analyzer.o `$(LLVM_CONFIG) --cxxflags` -fexceptions $(OPTFLAGS)
	clang++ -o ./analyzer_tests ./test_analyzer.o ./jam_llvm.o ./lexer.o ./parser.o ./codegen.o ./target.o ./cabi.o ./module_resolver.o ./symbol_table.o ./number_literal.o ./init_analysis.o ./drop_registry.o ./abi.o ./diagnostics.o ./decl.o ./analyzer.o ./astgen.o ./jir_codegen.o ./jir_verify.o `$(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs`
	./analyzer_tests

test: test-unit test-init test-abi test-codegen-errors test-jir test-diagnostics test-decl test-analyzer

fmt: format

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
