/*
 * Header file for C helper functions
 */

#ifndef C_HELPERS_H
#define C_HELPERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple arithmetic
int32_t add_numbers(int32_t a, int32_t b);

// Callbacks
typedef int32_t (*callback_fn)(int32_t);
int32_t call_with_callback(callback_fn fn, int32_t value);

// String handling
void print_from_c(const char* message);
const char* get_c_string();

// Multiple return values
void get_coordinates(int32_t* x, int32_t* y);

// Pointer size
size_t get_pointer_size();

// Struct handling
struct Point {
    int32_t x;
    int32_t y;
};

struct Point create_point(int32_t x, int32_t y);
void print_point(struct Point p);

#ifdef __cplusplus
}
#endif

#endif // C_HELPERS_H
