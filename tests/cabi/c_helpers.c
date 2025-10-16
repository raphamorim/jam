/*
 * C helper functions for Jam C ABI tests
 */

#include <stdio.h>
#include <stdint.h>

// Simple function that adds two numbers
int32_t add_numbers(int32_t a, int32_t b) {
    printf("[C] add_numbers(%d, %d) = %d\n", a, b, a + b);
    return a + b;
}

// Function that takes a callback
typedef int32_t (*callback_fn)(int32_t);

int32_t call_with_callback(callback_fn fn, int32_t value) {
    printf("[C] Calling callback with value %d\n", value);
    return fn(value);
}

// Function that works with strings
void print_from_c(const char* message) {
    printf("[C] Message from Jam: %s\n", message);
}

// Function that returns a pointer
const char* get_c_string() {
    return "Hello from C!";
}

// Function with multiple return values via pointer
void get_coordinates(int32_t* x, int32_t* y) {
    *x = 42;
    *y = 84;
    printf("[C] Returning coordinates: x=%d, y=%d\n", *x, *y);
}

// Test pointer size compatibility
size_t get_pointer_size() {
    return sizeof(void*);
}

// Test struct ABI
struct Point {
    int32_t x;
    int32_t y;
};

struct Point create_point(int32_t x, int32_t y) {
    struct Point p = {x, y};
    printf("[C] Created point: (%d, %d)\n", p.x, p.y);
    return p;
}

void print_point(struct Point p) {
    printf("[C] Point: (%d, %d)\n", p.x, p.y);
}
