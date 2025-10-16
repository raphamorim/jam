#include <stdio.h>
#include <stdint.h>

struct Point {
    int32_t x;
    int32_t y;
};

int main() {
    printf("Struct Point size: %zu bytes\n", sizeof(struct Point));
    printf("int32_t size: %zu bytes\n", sizeof(int32_t));
    printf("Pointer size: %zu bytes\n", sizeof(void*));
    
    struct Point p = {10, 20};
    printf("Point: x=%d, y=%d\n", p.x, p.y);
    
    return 0;
}
