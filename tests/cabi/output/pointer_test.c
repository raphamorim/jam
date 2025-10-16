#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

int main() {
    printf("Pointer size: %zu bytes\n", sizeof(void*));
    printf("Pointer alignment: %zu bytes\n", alignof(void*));
    printf("int32_t size: %zu bytes\n", sizeof(int32_t));
    printf("int32_t alignment: %zu bytes\n", alignof(int32_t));
    
    return 0;
}
