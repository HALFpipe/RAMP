#include <math.h>
#include <stdio.h>
#include <stdint.h>

#define TARGET_CLONES __attribute__ ((target_clones("default", "arch=sandybridge", "arch=broadwell", "arch=cascadelake", "arch=znver4")))

TARGET_CLONES
void c_set_lower_triangle(
        double * const __restrict__ a,
        double alpha,
        size_t const n,
        size_t const m
) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = i + 1; j < n; j++) {
            a[i * n + j] = alpha;
        }
    }
}

TARGET_CLONES
void c_set_upper_triangle(
        double * const __restrict__ a,
        double alpha,
        size_t const n,
        size_t const m
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < m; j++) {
            a[j * n + i] = alpha;
        }
    }
}

TARGET_CLONES
void c_copy_upper_to_lower_triangle(
    double * const __restrict__ a,
    size_t const n,
    size_t const m
) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < m; j++) {
            a[i * n + j] = a[j * n + i];
        }
    }
}
