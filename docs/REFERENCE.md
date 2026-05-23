---
layout: docs
title: Language Reference
permalink: /reference/
sidebar:
  - title: Basics
    items:
      - name: Variables & Constants
        url: "#variables"
      - name: Integer Types
        url: "#types"
      - name: Booleans
        url: "#bool"
      - name: Strings
        url: "#strings"
      - name: Pointers & Slices
        url: "#pointers"
      - name: Arrays
        url: "#arrays"
  - title: Control & Functions
    items:
      - name: Control Flow
        url: "#control-flow"
      - name: Functions
        url: "#fn-declaration"
      - name: Type Casts
        url: "#casts"
  - title: Types
    items:
      - name: Structs
        url: "#structs"
      - name: Enums
        url: "#enums"
      - name: Unions
        url: "#unions"
      - name: Pattern Matching
        url: "#match"
      - name: Generics
        url: "#generics"
  - title: System
    items:
      - name: Modules & Imports
        url: "#imports"
      - name: Comptime Intrinsics
        url: "#intrinsics"
      - name: extern / FFI
        url: "#extern"
  - title: Mutable Value Semantics
    items:
      - name: Overview
        url: "#mvs-overview"
      - name: Parameter Modes
        url: "#mvs-modes"
      - name: Exclusivity
        url: "#mvs-exclusivity"
      - name: Drop
        url: "#mvs-drop"
---

<div class="mb-12">
    <span class="text-pink-600 font-mono text-sm tracking-wide">Language Reference</span>
    <h1 class="text-4xl font-bold text-slate-900 mt-2 mb-6">Jam Reference</h1>
    <p class="text-lg text-slate-600 leading-relaxed">
        Jam is a statically typed systems language with mutable value semantics, generics
        over types, tagged unions, and an LLVM backend. The compiler infers types where it
        can; everything across a function boundary is fully explicit.
    </p>
</div>

## Variables & Constants {#variables}

Jam has two binding forms inside functions: `var` for mutable storage and `const` for
single-assignment values. Every binding must be initialized at its declaration, Jam has
no `undefined` placeholder.

```jam
fn example() {
    var counter: i32 = 0;
    counter = counter + 1;

    const limit: i32 = 100;
    // limit = 101;  // error: cannot reassign const binding
}
```

At module scope, `const` declarations bind a name to a compile-time value. They are
*inlined* at every use site, referring to one costs the same as a literal. The type may
be inferred when an initializer alone determines it.

```jam
const FLAG_Z: u8 = 0x80;
const PAGE:   u32 = 4096;
const SHIFT       = 4;            // type inferred

fn isZSet(f: u8) bool {
    return (f & FLAG_Z) != 0;     // FLAG_Z inlined here
}
```

Module-scope `const` is also how top-level types, imports, and function pointers are
named, Jam funnels every top-level declaration through the same form.

---

## Integer Types {#types}

An integer is a number without a fractional component. Jam provides both signed (`i`) and
unsigned (`u`) integers of various sizes.

| Length | Signed  | Unsigned | Range (Signed)                                            | Range (Unsigned)              |
|--------|---------|----------|-----------------------------------------------------------|-------------------------------|
| 8-bit  | `i8`    | `u8`     | -128 to 127                                               | 0 to 255                      |
| 16-bit | `i16`   | `u16`    | -32,768 to 32,767                                         | 0 to 65,535                   |
| 32-bit | `i32`   | `u32`    | -2,147,483,648 to 2,147,483,647                           | 0 to 4,294,967,295            |
| 64-bit | `i64`   | `u64`    | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807   | 0 to 18,446,744,073,709,551,615 |
| Arch   | `isize` | `usize`  | Depends on computer architecture (32 or 64-bit)           |                               |

```jam
fn main() u8 {
    const smallNum: u8  = 255;
    const bigNum:   u64 = 18446744073709551615;

    const temperature: i32 = -40;
    const altitude:    i64 = -11034;

    var counter: i8 = -128;
    return 0;
}
```

Numeric literals can use underscores for readability (`0x1234_5678`, `1_000_000`) and
the standard `0x`, `0o`, `0b` prefixes.

<div class="bg-purple-50 border border-purple-100 rounded-lg p-4 flex gap-3 mt-6 mb-12">
    <i data-lucide="lightbulb" class="w-5 h-5 text-purple-500 flex-shrink-0 mt-0.5"></i>
    <div>
        <h4 class="font-bold text-purple-800 text-sm">Choosing Integer Types</h4>
        <p class="text-sm text-purple-700 mt-1">
            Use unsigned types (<code class="font-mono">u8</code>, <code class="font-mono">u16</code>, etc.) for values that are never negative, indices, counts, byte sizes.
            Use signed types (<code class="font-mono">i8</code>, <code class="font-mono">i16</code>, etc.) when you need to represent negative values.
        </p>
    </div>
</div>

---

## Boolean Type {#bool}

The `bool` type represents a value that can be either `true` or `false`. Booleans are one
byte in size.

```jam
const isSweet: bool = true;
const isSour:  bool = false;

if (isSweet) {
    std.fmt.println("Delicious!");
}
```

### Logical Operators

Jam provides three logical operators for working with boolean values:

| Operator | Name | Description                                                            |
|----------|------|------------------------------------------------------------------------|
| `!`      | NOT  | Negates a boolean value                                                |
| `and`    | AND  | Returns true if both operands are true (short-circuits if first is false) |
| `or`     | OR   | Returns true if either operand is true (short-circuits if first is true)  |

The `and` and `or` operators use **short-circuit evaluation**: if the result can be
determined from the first operand alone, the second operand is not evaluated.

```jam
fn checkAccess(isAdmin: bool, isOwner: bool) bool {
    return isAdmin or isOwner;
}

fn validate(hasEmail: bool, hasPassword: bool) bool {
    return hasEmail and hasPassword;
}
```

---

## Strings {#strings}

Strings in Jam are represented as slices of bytes (`[]u8`). The `str` type is an alias for
`[]u8`. Element mutability follows the binding: `const s: []u8 = "..."` is read-only;
`var s: []u8 = "..."` permits writes.

```jam
const greeting: str   = "Hello, World!";
const message:  []u8  = "Same as str";

// Strings are UTF-8.
const chinese: str = "世界";
const emoji:   str = "🌍";
```

### Memory Layout

A slice is a two-word value: a pointer to the data and a length.

| Field | Type          | Description                                                       |
|-------|---------------|-------------------------------------------------------------------|
| `ptr` | `*const[] u8` | Many-item pointer to the first byte                                |
| `len` | `u64`         | Number of bytes (excludes any trailing null in literals)           |

### Escape Sequences

| Escape       | Description                                       |
|--------------|---------------------------------------------------|
| `\n`         | Newline                                           |
| `\r`         | Carriage return                                   |
| `\t`         | Tab                                               |
| `\\`         | Backslash                                         |
| `\"`         | Double quote                                      |
| `\'`         | Single quote                                      |
| `\0`         | Null byte                                         |
| `\xHH`       | Hex byte (2 hex digits)                           |
| `\u{HHHHHH}` | Unicode codepoint (1-6 hex digits, encoded as UTF-8) |

```jam
const newline: str = "Line1\nLine2";
const quote:   str = "She said \"Hello\"";
const hello:   str = "\x48\x65\x6C\x6C\x6F";   // "Hello"
const earth:   str = "\u{1F30D}";              // 🌍
```

---

## Pointers & Slices {#pointers}

Jam distinguishes three reference families. Pointer types take a *required* `const` or
`mut` qualifier marking whether the pointee may be written. Slices and fixed arrays carry
no qualifier, their element mutability follows the binding.

| Type          | Description                                                                      |
|---------------|----------------------------------------------------------------------------------|
| `*const T`    | Single-item pointer, read-only pointee                                           |
| `*mut T`      | Single-item pointer, writable pointee                                            |
| `*const[] T`  | Many-item pointer (indexable), read-only pointee                                 |
| `*mut[] T`    | Many-item pointer (indexable), writable pointee                                  |
| `[]T`         | Slice (pointer + length); element mutability follows the binding                 |
| `[N]T`        | Fixed-size array of N elements; element mutability follows the binding           |

The `&` prefix takes the address of a binding or array element; `p.*` dereferences a
pointer. Indexing through a many-item pointer is `p[i]`.

```jam
fn pointerExample() u8 {
    var x: u8 = 42;
    var p: *mut u8 = &x;
    p.* = 100;
    return x;                       // 100
}

fn indexThroughMany() u8 {
    var arr: [4]u8 = [10, 20, 30, 40];
    var p: *mut[] u8 = &arr[0];
    p[2] = 99;
    return arr[2];                  // 99
}
```

<div class="bg-pink-50 border border-pink-100 rounded-lg p-4 flex gap-3 mb-12">
    <i data-lucide="info" class="w-5 h-5 text-pink-500 flex-shrink-0 mt-0.5"></i>
    <div>
        <h4 class="font-bold text-pink-800 text-sm">No bare <code class="font-mono">*T</code></h4>
        <p class="text-sm text-pink-700 mt-1">
            The mutability qualifier is part of the pointer type, there is no shorthand
            <code class="font-mono">*T</code>. Writing <code class="font-mono">*const u8</code>
            vs <code class="font-mono">*mut u8</code> documents intent at every signature.
        </p>
    </div>
</div>

---

## Arrays {#arrays}

Fixed-size arrays have a length known at compile time. Array literals come in three forms:
comma-separated, fill-with-count, and empty (which produces a zero-length slice).

```jam
var a: [4]u8  = [10, 20, 30, 40];     // [a, b, c, d]
var b: [16]u8 = [0; 16];              // [expr; N], fill 16 slots with 0
const empty: []u8 = [];               // empty slice

// Index with []. Out-of-bounds is undefined at runtime;
// bounds checks are not yet inserted.
var x: u8 = a[2];                     // 30
a[0] = 99;
```

A fixed array implicitly coerces to a slice when bound to a `[]T` location, producing a
two-word `{ptr, len}` view over the same storage.

---

## Control Flow {#control-flow}

Jam has `if`/`else`, `while`, `for`, and `return`. Conditions require parentheses.

```jam
fn classify(n: i32) i32 {
    if (n < 0) {
        return -1;
    } else if (n == 0) {
        return 0;
    } else {
        return 1;
    }
}

fn sumTo(n: u32) u32 {
    var total: u32 = 0;
    var i: u32 = 0;
    while (i < n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

fn fillIndices() [16]u8 {
    var arr: [16]u8 = [0; 16];
    for i in 0:16 {
        arr[i] = i as u8;
    }
    return arr;
}
```

The `for` loop iterates a half-open integer range, `for i in 0:N` binds `i = 0, 1, …, N-1`.

---

## Functions {#fn-declaration}

Functions are declared with `fn`. Parameters are strictly typed; the return type goes
directly after the parameter list. Jam code uses *camelCase* by convention for functions
and variables.

```jam
// Two i32 inputs, returns i32.
fn mixIngredients(sugar: i32, fruit: i32) i32 {
    return sugar + fruit;
}

// No return value, omit the return type.
fn greet() {
    std.fmt.println("Hello!");
}
```

Functions can be prefixed with `pub` (visible outside the defining module) and `extern`
(declared but defined elsewhere, typically libc).

```jam
pub fn add(a: i32, b: i32) i32 { return a + b; }

pub extern fn malloc(size: u64) *mut[] u8;
pub extern fn free(ptr: *mut[] u8);
```

<div class="bg-pink-50 border border-pink-100 rounded-lg p-4 flex gap-3 mb-12">
    <i data-lucide="info" class="w-5 h-5 text-pink-500 flex-shrink-0 mt-0.5"></i>
    <div>
        <h4 class="font-bold text-pink-800 text-sm">Void Functions</h4>
        <p class="text-sm text-pink-700 mt-1">
            A function with no declared return type returns nothing. Don't write
            <code class="font-mono">void</code>, just omit the return-type slot.
        </p>
    </div>
</div>

---

## Type Casts {#casts}

The `as` operator performs explicit conversions. Integer ↔ integer casts truncate or
extend; integer → float converts numerically; an enum tag can be extracted to its
underlying integer type.

```jam
const big: u32 = 0x1234_5678;
const low: u8  = big as u8;            // truncate → 0x78

const small: u8  = 200;
const wide:  u32 = small as u32;       // zero-extend → 200

const negative: i32 = 250;
const narrow:   i8  = negative as i8;  // wrap → -6
```

Casts are explicit at every narrowing or sign-changing step. There are no implicit
numeric conversions.

---

## Structs {#structs}

A struct groups fields under a single name. Top-level structs are declared via
`const Name = struct { … };`. Field names are `field: Type`; instances are built with
`{ field: value, … }`.

```jam
const Vec3   = struct { x: f32, y: f32, z: f32 };
const Pixel  = struct { r: u8, g: u8, b: u8 };
const Player = struct { hp: u32, level: u8, alive: bool };

fn main() {
    const v: Vec3 = { x: 0, y: 100, z: 50 };
    var px:  Pixel = { r: 10, g: 20, b: 30 };
    px.r = 100;

    // Nested literals work the same way.
    const Outer = struct { inner: Pixel, c: u8 };
    const x: Outer = {
        inner: { r: 1, g: 2, b: 3 },
        c: 4,
    };
}
```

### Methods

Functions declared *inside* a struct body are methods. The first parameter is
conventionally `self`; the parameter mode chooses borrow semantics.

```jam
const Counter = struct {
    value: u32,
    sink:  *mut u32,

    fn drop(self: mut Counter) {
        var p: *mut u32 = self.sink;
        p.* = p.* + 1;
    }
};

fn observe() u32 {
    var hits: u32 = 0;
    var c: Counter = { value: 5, sink: &hits };
    return c.value;
    // c.drop() fires automatically here, see Drop, below.
}
```

A method can also be invoked by-name on the type: `Counter.drop(&c)` calls it
explicitly while `c` is still in scope (and the automatic drop will still fire at scope
exit, calling drop manually is currently a footgun).

---

## Enums {#enums}

Enums describe a closed set of named variants. The simplest form is *payload-less*, each
variant is a u8 discriminant.

```jam
const Color = enum { Red, Green, Blue };

fn classifyColor(c: Color) u8 {
    match (c) {
        Color.Red   { return 100; }
        Color.Green { return 200; }
        Color.Blue  { return 50; }
        _           { return 0; }
    }
    return 99;
}
```

Discriminant values are assigned in declaration order starting at 0. You can also pin them
explicitly:

```jam
const Phase = enum { Idle = 0, Running = 5, Stopping = 9 };

fn phaseAsByte(p: Phase) u8 {
    return p as u8;
}
```

### Variants with Payloads (Tagged Unions)

Variants can carry positional fields. Pattern matching destructures them by name.

```jam
const Op = enum {
    Nop,
    LdRR(u8, u8),
    Imm(u8),
    Wide(u16),
};

fn srcReg(op: Op) u8 {
    match (op) {
        Op.Nop          { return 0xFF; }
        Op.LdRR(d, s)   { return s; }
        Op.Imm(v)       { return v; }
        Op.Wide(w)      { return 0xEE; }
        _               { return 0xCC; }
    }
    return 0;
}
```

Payload-carrying enums are how `Option(T)` and `Result(T, E)` are built, see
[Generics](#generics).

---

## Unions {#unions}

A `union` is *untagged*, every field shares the same storage. Use it for type punning
(read float bits as `u32`, etc.) and for matching C-side `union { … }` types at FFI
boundaries.

```jam
const FloatBits = union {
    i: u32,
    f: f32,
};

fn floatBits(x: f32) u32 {
    var b: FloatBits = { f: x };
    return b.i;
}
```

Unions size to their largest field; alignment is the max of all fields'. Unlike enums,
there is no discriminant, the compiler trusts the program to know which field is live.

---

## Pattern Matching {#match}

The `match` expression dispatches on a scrutinee. Patterns include integer literals,
enum variants (with or without payloads), and `_` as a wildcard.

```jam
fn dispatch(x: u8) u8 {
    match (x) {
        0           { return 10; }
        1 or 2 or 3 { return 20; }       // `or`-joined patterns
        4..=9       { return 30; }       // inclusive range
        _           { return 255; }
    }
    return 99;
}
```

A `match` can also be used as an *expression*, each arm produces a value, and the result
is the value of the matched arm.

```jam
fn unwrap(o: Option(i32), fallback: i32) i32 {
    return match (o) {
        Option(i32).Some(x) { x }
        Option(i32).None    { fallback }
    };
}
```

Arms that bind payload fields introduce those names into the arm's scope.

---

## Generics {#generics}

A generic function takes one or more type parameters declared `T: type` and returns a
type. Calling it with concrete type arguments at compile time produces a concrete type;
each distinct argument list yields a fresh struct (or enum).

```jam
fn Box(T: type) type {
    return struct {
        value: T,
    };
}

fn Pair(A: type, B: type) type {
    return struct {
        first:  A,
        second: B,
    };
}

fn main() {
    var b: Box(i32)        = { value: 17 };
    var p: Pair(i32, u8)   = { first: 7, second: 35 };
}
```

A type alias `const Name = Generic(arg);` registers `Name` as a synonym, subsequent
uses resolve to the same instantiated struct.

```jam
const BoxI32 = Box(i32);
var b: BoxI32 = { value: 17 };
```

### Generic Methods

Methods on generic types are cloned per instantiation. Each instantiation gets its own
LLVM symbol with substituted parameter and return types.

```jam
fn Holder(T: type) type {
    return struct {
        value: T,
        fn unwrap(self: move Self) T {
            return self.value;
        }
    };
}
```

`Self` inside a struct body refers to the enclosing struct type, for a generic, that's
the specific instantiation in play.

### Generic Enums

The same mechanism produces tagged unions parameterized by type. The standard library's
`Option(T)` is built this way:

```jam
pub fn Option(T: type) type {
    return enum {
        Some(T),
        None,
    };
}
```

Construct variants by qualifying with the instantiated type: `Option(i32).Some(42)`,
`Option(i32).None`.

---

## Modules & Imports {#imports}

`import("name")` returns a *module value*, a compile-time record of the symbols another
file exports. Bind it through `const` either whole or destructured.

```jam
// Whole-module binding.
const std = import("std");

fn show() {
    std.fmt.println("Hello!");
}

// Destructured binding, pulls specific names into the current scope.
// Re-exports under `std` can be reached via chained access on the
// import expression — pick the names you need without binding the
// whole module.
const { Vec }    = import("std").collections;
const { Option } = import("std").option;
const { assert } = import("test");
```

The path is resolved relative to the file (`./collections.jam` or `std/string.jam`) or
to a built-in name (`std`, `test`). Modules can re-export by binding to a local `pub`
name.

---

## Comptime Intrinsics {#intrinsics}

Intrinsics are compiler builtins prefixed with `@`. They run at compile time and their
results are substituted as constants before LLVM sees the code.

| Intrinsic            | Description                                             |
|----------------------|---------------------------------------------------------|
| `@sizeOf(T)`         | Size of `T` in bytes (u64)                              |
| `@alignOf(T)`        | Alignment of `T` in bytes (u64)                         |

```jam
fn bytesFor(n: u64) u64 {
    return n * @sizeOf(u32);          // n * 4, fully constant-folded
}

fn pointerSize() u64 {
    return @sizeOf(*const u8);        // 8 on a 64-bit target
}

fn sliceSize() u64 {
    return @sizeOf([]u8);             // 16, {ptr, len}
}
```

Intrinsics compose freely with runtime arithmetic. They produce a `u64` and require an
explicit cast to narrow to other widths.

---

## extern / FFI {#extern}

`extern` declarations link to symbols defined in C (or any system providing the C ABI).
The compiler does not generate a body; it just emits a call. Most stdlib allocators use
this.

```jam
pub extern fn malloc(size: u64)  *mut[] u8;
pub extern fn free(ptr: *mut[] u8);
pub extern fn realloc(ptr: *mut[] u8, size: u64) *mut[] u8;

fn allocOne() *mut[] u32 {
    var raw: *mut[] u8 = malloc(@sizeOf(u32));
    return raw as *mut[] u32;
}
```

Param types must match the C ABI. Slice arguments are passed as a `{ptr, len}` pair, and
struct arguments larger than two words use sret-style return ABI, these match the
platform's C compiler.

---

## Mutable Value Semantics {#mvs-overview}

Jam is a *mutable value semantics* language. Every binding owns its value; passing a
value to a function does not silently share storage with the caller. Functions opt into
a specific borrowing or transfer behavior via a per-parameter **mode** keyword.

The result is the same memory safety as Rust, no use-after-free, no double-free, no
data races, no reads of uninitialized memory, with no lifetime annotations.

The design is inspired by Hylo's parameter-mode system and Rust's drop semantics. The
compiler enforces three rules: *definite initialization* (every binding is written
before it is read), *exclusivity* (no overlapping mutable borrows), and *linear drops*
(every owned value gets exactly one destructor call).

## Parameter Modes {#mvs-modes}

Every function parameter is declared in one of three modes. The default mode is
read-only; the other two are introduced by an explicit keyword between the colon and
the type.

| Mode                  | Keyword     | Pointee Mutability | Caller After Call        |
|-----------------------|-------------|--------------------|--------------------------|
| Read-only borrow      | `(default)` | read-only          | unchanged                |
| Exclusive read-write  | `mut`       | read-write         | unchanged                |
| Consume ownership     | `move`      | read-write         | becomes uninitialized    |

```jam
// Default read-only borrow, caller's value is unchanged.
fn distance(a: Point, b: Point) f64 {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Exclusive read-write, caller passes `&binding`.
fn scale(p: mut Point, factor: f64) {
    p.x = p.x * factor;
    p.y = p.y * factor;
}

// Consume, caller's binding is uninitialized after the call.
fn storeIn(buf: move []u8, db: mut Database) {
    db.append(buf);
}
```

At the call site, a `mut` parameter is passed with `&` to make the borrow explicit:

```jam
var p: Point = { x: 3.0, y: 4.0 };
scale(&p, 2.0);                  // mut borrow, explicit &
distance(p, otherPoint);          // read-only, no sigil
```

`move` parameters take a plain expression, the binding is dead in the caller after the
call.

## Exclusivity {#mvs-exclusivity}

At any call boundary, at most one argument may be a `mut` borrow of a given binding,
and a `mut` borrow cannot coexist with any other borrow of the same binding. The
compiler rejects programs that violate this rule, the same "law of exclusivity"
Rust's borrow checker enforces, applied locally at each call rather than across
whole-program lifetimes.

```jam
fn modify(x: mut u32, y: u32) u32 {
    x = x + y;
    return x;
}

fn caller() u32 {
    var n: u32 = 5;
    return modify(&n, n);   // error: conflicting borrows of `n`
}
```

## Drop {#mvs-drop}

A struct may define a `drop` method that runs automatically when an owned instance goes
out of scope. The drop method takes `self: mut Self` and runs exactly once per owned
value, even on early returns, in match arms, and through nested control flow. There is
no manual `defer` ceremony.

```jam
const File = struct {
    fd: i32,

    fn drop(self: mut File) {
        close(self.fd);
    }
};

fn readFile(path: []u8) i32 {
    var f: File = openFile(path);
    return f.fd;
    // f.drop() runs here automatically
}
```

A value *moved* into another function (via the `move` mode) becomes uninitialized in
the caller, so the drop fires at the new owner's scope exit, never twice.
