.PHONY: build install uninstall clean cmake-build cmake-install cmake-uninstall test
.DEFAULT_GOAL := build

LLVM_CONFIG=$(shell which llvm-config 2>/dev/null || echo "llvm-config")
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DOCDIR ?= $(PREFIX)/share/doc/jam

# Check if we're on macOS or Linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM = macOS
else ifeq ($(UNAME_S),Linux)
    PLATFORM = Linux
else
    PLATFORM = Unknown
endif

build:
	@echo "Building Jam compiler for $(PLATFORM)..."
	@if ! command -v $(LLVM_CONFIG) >/dev/null 2>&1; then \
		echo "Error: llvm-config not found. Please install LLVM development packages."; \
		exit 1; \
	fi
	clang++ -c ./src/main.cpp -o ./main.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/lexer.cpp -o ./lexer.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/parser.cpp -o ./parser.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/ast.cpp -o ./ast.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/codegen.cpp -o ./codegen.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/target.cpp -o ./target.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -c ./src/cabi.cpp -o ./cabi.o `$(LLVM_CONFIG) --cxxflags` -fexceptions
	clang++ -o ./jam.out ./main.o ./lexer.o ./parser.o ./ast.o ./codegen.o ./target.o ./cabi.o `$(LLVM_CONFIG) --ldflags --libs --libfiles --system-libs`
	@echo "Build complete! Executable: ./jam.out"

# CMake-based build (recommended)
cmake-build:
	@echo "Building with CMake..."
	./build.sh

# Install using CMake (recommended)
cmake-install: cmake-build
	@echo "Installing Jam compiler using CMake..."
	cd build && sudo make install
	@echo ""
	@echo "Jam compiler installed successfully!"
	@echo "   Executable: $(PREFIX)/bin/jam"
	@echo "   Documentation: $(PREFIX)/share/doc/jam/"
	@echo ""
	@echo "You can now use 'jam' from anywhere in your terminal."
	@echo "Try: jam --help"

# Uninstall using CMake
cmake-uninstall:
	@echo "Uninstalling Jam compiler..."
	@if [ -f build/cmake_uninstall.cmake ]; then \
		cd build && sudo make uninstall; \
		echo "Jam compiler uninstalled successfully!"; \
	else \
		echo "No installation found. Run 'make cmake-install' first."; \
	fi

# Manual install (fallback)
install: build
	@echo "Installing Jam compiler to $(PREFIX)..."
	@echo "Platform: $(PLATFORM)"
	
	# Create directories
	sudo mkdir -p $(BINDIR)
	sudo mkdir -p $(DOCDIR)
	
	# Install executable
	sudo cp ./jam.out $(BINDIR)/jam
	sudo chmod 755 $(BINDIR)/jam
	
	# Install documentation
	sudo cp README.md $(DOCDIR)/ 2>/dev/null || true
	sudo cp LICENSE $(DOCDIR)/ 2>/dev/null || true
	sudo cp LANGUAGE_SPEC.md $(DOCDIR)/ 2>/dev/null || true
	
	@echo ""
	@echo "   Jam compiler installed successfully!"
	@echo "   Executable: $(BINDIR)/jam"
	@echo "   Documentation: $(DOCDIR)/"
	@echo ""
	@echo "You can now use 'jam' from anywhere in your terminal."
	@echo "Try: jam --help"

# Manual uninstall
uninstall:
	@echo "Uninstalling Jam compiler..."
	sudo rm -f $(BINDIR)/jam
	sudo rm -rf $(DOCDIR)
	@echo "Jam compiler uninstalled successfully!"

# Clean build artifacts
clean:
	rm -f ./main.o ./lexer.o ./parser.o ./ast.o ./codegen.o ./target.o ./cabi.o ./jam.out
	rm -rf build/
	@echo "Build artifacts cleaned!"

# Show installation info
info:
	@echo "Jam Compiler Installation Info"
	@echo "=============================="
	@echo "Platform: $(PLATFORM)"
	@echo "LLVM Config: $(LLVM_CONFIG)"
	@echo "Install Prefix: $(PREFIX)"
	@echo "Binary Directory: $(BINDIR)"
	@echo "Documentation Directory: $(DOCDIR)"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the compiler"
	@echo "  make test           - Run all tests (Jam + C++)"
	@echo "  make install        - Install using manual method"
	@echo "  make uninstall      - Uninstall manual installation"
	@echo "  make cmake-install  - Install using CMake (recommended)"
	@echo "  make cmake-uninstall- Uninstall CMake installation"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make info           - Show this information"

# Run all tests
test:
	@echo "Running comprehensive test suite..."
	./run_all_tests.sh
