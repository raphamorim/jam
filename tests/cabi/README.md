# Jam C ABI Tests

This directory contains tests for Jam's C Application Binary Interface (ABI) compatibility, demonstrating interoperability between Jam and C code.

## Overview

The C ABI is crucial for:
- Calling C libraries from Jam code
- Exposing Jam functions to C code
- System call interfaces
- FFI (Foreign Function Interface) compatibility

## Test Files

### C Helper Code
- **c_helpers.c** - C functions that can be called from Jam
- **c_helpers.h** - Header file for C functions
- **c_caller.c** - C program that calls Jam functions

### Jam Test Code
- **test_call_c.jam** - Jam code calling C functions
- **jam_functions.jam** - Jam functions exposed to C

### Test Runner
- **run_cabi_tests.sh** - Automated test suite

## What We're Testing

### 1. Calling Convention
- Verify Jam functions use C calling convention
- Check parameter passing (stack vs registers)
- Validate return value handling

### 2. Data Type Compatibility
- Integer types (i8, i16, i32, i64)
- Unsigned types (u8, u16, u32, u64)
- Pointers and pointer arithmetic
- Struct layout and alignment

### 3. ABI Details
- Pointer size (32-bit vs 64-bit)
- Struct padding and alignment
- Calling convention specifics (System V AMD64 ABI on Unix, Win64 ABI on Windows)

### 4. Function Signatures
- Simple functions (primitives)
- Functions with pointers
- Functions with structs
- Variadic functions (future)

## Running the Tests

```bash
# From the tests/cabi directory:
./run_cabi_tests.sh

# Or from the project root:
./tests/cabi/run_cabi_tests.sh
```

## Current Status

✅ **Working:**
- Target information detection
- C calling convention setup
- LLVM IR generation with correct ABI
- Basic type compatibility

🚧 **In Progress:**
- Extern function declarations in Jam syntax
- Direct linking with C object files
- Struct ABI compatibility

📋 **Planned:**
- `extern "C"` keyword support in Jam
- Full FFI with type marshalling
- Callback support (C calling Jam)
- Complex struct handling

## Example: Future Jam Syntax

```jam
// Future syntax for calling C functions
extern "C" fn printf(format: *u8, ...) -> i32;
extern "C" fn malloc(size: u64) -> *u8;
extern "C" fn free(ptr: *u8);

fn main() -> u8 {
    var msg: *u8 = "Hello from Jam!\n";
    printf(msg);
    return 0;
}
```

## C ABI Specifications

### System V AMD64 ABI (Linux, macOS, BSD)
- First 6 integer arguments in registers: RDI, RSI, RDX, RCX, R8, R9
- First 8 floating-point arguments in XMM0-XMM7
- Additional arguments on stack (right to left)
- Return value in RAX (integers) or XMM0 (floats)

### Microsoft x64 ABI (Windows)
- First 4 arguments in registers: RCX, RDX, R8, R9
- Additional arguments on stack (right to left)
- Shadow space (32 bytes) required
- Return value in RAX or XMM0

### ARM64 (AArch64)
- First 8 integer arguments in X0-X7
- First 8 floating-point arguments in V0-V7
- Additional arguments on stack
- Return value in X0 or V0

## Debugging

Test output is saved in `output/` directory:
- `*.log` - Test execution logs
- `*.ll` - LLVM IR files
- `*.o` - Object files
- `*.a` - Static libraries

## References

- [System V AMD64 ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)
- [Microsoft x64 Calling Convention](https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [ARM64 ABI](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [LLVM Calling Conventions](https://llvm.org/docs/LangRef.html#calling-conventions)
