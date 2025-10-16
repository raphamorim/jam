# Jam Programming Language

Jam is a systems programming language designed to provide developers with the control and performance characteristics of C while incorporating modern safety features to mitigate common programming errors. The language targets developers who require bare-metal performance and direct hardware access without sacrificing code reliability and maintainability.

## Abstract

This implementation presents a compiler for the Jam programming language, featuring LLVM-based code generation, just-in-time execution capabilities, and a comprehensive type system. The compiler supports multiple integer types with explicit bit-width specifications, UTF-8 string handling, and modern control flow constructs while maintaining compatibility with low-level system programming requirements.

#### Automated Installation
```bash
git clone <repository-url> --depth=1
cd jam
./install.sh
```

#### Manual Build Process
```bash
# CMake-based build (recommended)
make cmake-install

# Alternative make-based build
make install

# Local development build
./build.sh
```

### Platform-Specific Dependencies

#### macOS
```bash
brew install llvm cmake
```

#### Ubuntu/Debian
```bash
sudo apt-get install llvm-dev cmake clang
```

#### CentOS/RHEL
```bash
sudo yum install llvm-devel cmake clang
```

## Usage and Command-Line Interface

### Basic Operations
```bash
# Display compiler options
jam --help

# Compile source to executable
jam program.jam

# Execute via JIT compilation
jam --run program.jam

# Show target platform information
jam --target-info program.jam
```

### Target Information
```bash
$ jam --target-info program.jam

Target Information:
  Name: aarch64-macos
  Triple: aarch64-unknown-darwin
  Pointer size: 8 bytes
  Libc: darwin
  Requires PIC: no
  Requires PIE: yes
  Uses C ABI: yes
```

### Example Programs

#### Basic Program Structure
```jam
fn main() -> u32 {
    println("Hello, World!");
    return 0;
}
```

#### Type System Demonstration
```jam
fn demonstrate_types() -> u32 {
    const byte_value: u8 = 255;
    const word_value: u16 = 65535;
    const dword_value: u32 = 4294967295;
    const flag: bool = true;
    const message: str = "Type system demonstration";
    
    return 0;
}
```

#### Control Flow Examples
```jam
fn fibonacci(n: u32) -> u32 {
    if (n <= 1) {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

fn iterative_example() -> u32 {
    for i in 0:10 {
        if (i == 5) {
            continue;
        }
        if (i == 8) {
            break;
        }
        const result: u32 = fibonacci(i);
    }
    return 0;
}
```

## C ABI Interoperability

Jam provides first-class support for C ABI (Application Binary Interface), enabling seamless interoperability with C libraries and allowing Jam code to be called from C.

### Calling C Functions from Jam

Functions currently use C calling conventions by default, allowing them to interact with C code.

```jam
extern "C" fn printf(format: *u8, ...) -> i32;
extern "C" fn malloc(size: u64) -> *u8;
extern "C" fn free(ptr: *u8);
extern "C" fn strlen(str: *u8) -> u64;

fn main() -> u8 {
    // Call C standard library functions
    printf("Hello from Jam!\n");
    
    // Allocate memory using C malloc
    var buffer: *u8 = malloc(100);
    
    // Use the buffer...
    
    // Free memory
    free(buffer);
    
    return 0;
}
```

### Exposing Jam Functions to C

Jam functions can be exported for use in C programs:

```jam
// Export a Jam function for C to call
export fn add_numbers(a: i32, b: i32) -> i32 {
    return a + b;
}

export fn process_data(data: *u8, length: u64) -> bool {
    // Process the data
    return true;
}

// Callback function type compatible with C
export fn callback_handler(value: i32) -> i32 {
    return value + value;
}
```

### C Code Calling Jam

The corresponding C code to use exported Jam functions:

```c
// C header for Jam functions
extern int32_t add_numbers(int32_t a, int32_t b);
extern bool process_data(const uint8_t* data, uint64_t length);
extern int32_t callback_handler(int32_t value);

int main() {
    // Call Jam function from C
    int32_t result = add_numbers(10, 32);
    printf("Result: %d\n", result);  // Output: 42
    
    // Use callback
    int32_t doubled = callback_handler(21);
    printf("Doubled: %d\n", doubled);  // Output: 42
    
    return 0;
}
```

### Struct ABI Compatibility

Jam structs are compatible with C struct layout:

```jam
// Future struct syntax (compatible with C)
struct Point {
    x: i32,
    y: i32,
}

struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

export fn create_point(x: i32, y: i32) -> Point {
    return Point { x: x, y: y };
}

export fn print_point(p: Point) {
    printf("Point(%d, %d)\n", p.x, p.y);
}
```

Corresponding C code:

```c
struct Point {
    int32_t x;
    int32_t y;
};

extern struct Point create_point(int32_t x, int32_t y);
extern void print_point(struct Point p);

int main() {
    struct Point p = create_point(10, 20);
    print_point(p);  // Output: Point(10, 20)
    return 0;
}
```

### Type Compatibility Reference

| Jam Type | C Type       | Size (64-bit) | Alignment |
|----------|--------------|---------------|-----------|
| `u8`     | `uint8_t`    | 1 byte        | 1 byte    |
| `u16`    | `uint16_t`   | 2 bytes       | 2 bytes   |
| `u32`    | `uint32_t`   | 4 bytes       | 4 bytes   |
| `u64`    | `uint64_t`   | 8 bytes       | 8 bytes   |
| `i8`     | `int8_t`     | 1 byte        | 1 byte    |
| `i16`    | `int16_t`    | 2 bytes       | 2 bytes   |
| `i32`    | `int32_t`    | 4 bytes       | 4 bytes   |
| `i64`    | `int64_t`    | 8 bytes       | 8 bytes   |
| `bool`   | `_Bool`      | 1 byte        | 1 byte    |
| `*T`     | `T*`         | 8 bytes       | 8 bytes   |

### Platform-Specific Calling Conventions

Jam automatically uses the correct calling convention for your platform:

**Linux/macOS (System V AMD64 ABI)**
- First 6 integer args in registers: `RDI, RSI, RDX, RCX, R8, R9`
- First 8 float args in: `XMM0-XMM7`
- Return value in `RAX` or `XMM0`

**Windows (Microsoft x64 ABI)**
- First 4 args in registers: `RCX, RDX, R8, R9`
- Shadow space: 32 bytes required
- Return value in `RAX` or `XMM0`

**ARM64 (AArch64)**
- First 8 integer args in: `X0-X7`
- First 8 float args in: `V0-V7`
- Return value in `X0` or `V0`

### Building Mixed Jam/C Projects

Example workflow for combining Jam and C code:

```bash
# 1. Compile Jam to object file
jam program.jam              # Generates output.o

# 2. Compile C helper code
gcc -c c_helpers.c -o c_helpers.o

# 3. Link together
clang output.o c_helpers.o -o final_program

# 4. Run
./final_program
```

## License and Distribution

This software is distributed under the MIT License. See the LICENSE file for complete terms and conditions.
