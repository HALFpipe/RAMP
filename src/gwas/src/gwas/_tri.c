#include <math.h>
#include <stdio.h>
#include <stdint.h>

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
