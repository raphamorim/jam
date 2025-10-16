/*
 * Jam code that will be called from C
 * This tests the reverse direction: C calling Jam
 */

#include <stdio.h>
#include <stdint.h>

// Forward declarations of Jam functions
// These will be provided by the compiled Jam code
extern int32_t jam_multiply(int32_t a, int32_t b);
extern int32_t jam_square(int32_t x);
extern void jam_greet(const char* name);

int main() {
    printf("=== C Calling Jam Functions ===\n\n");
    
    // Test 1: Simple arithmetic
    printf("Test 1: Calling jam_multiply(6, 7)\n");
    int32_t result1 = jam_multiply(6, 7);
    printf("Result: %d\n", result1);
    printf("Expected: 42\n");
    printf("Status: %s\n\n", result1 == 42 ? "PASS" : "FAIL");
    
    // Test 2: Another function
    printf("Test 2: Calling jam_square(8)\n");
    int32_t result2 = jam_square(8);
    printf("Result: %d\n", result2);
    printf("Expected: 64\n");
    printf("Status: %s\n\n", result2 == 64 ? "PASS" : "FAIL");
    
    // Test 3: String parameter
    printf("Test 3: Calling jam_greet(\"C World\")\n");
    jam_greet("C World");
    printf("\n");
    
    printf("=== All Tests Complete ===\n");
    return 0;
}
