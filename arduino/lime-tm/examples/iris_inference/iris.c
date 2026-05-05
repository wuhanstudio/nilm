#include "iris.h"

#include <math.h>
#include <stddef.h>

#ifdef __AVR__
float erff(float x) {
    // Abramowitz & Stegun approximation
    float t = 1.0f / (1.0f + 0.5f * fabsf(x));
    float tau = t * expf(-x*x - 1.26551223f +
                         t*(1.00002368f +
                         t*(0.37409196f +
                         t*(0.09678418f +
                         t*(-0.18628806f +
                         t*(0.27886807f +
                         t*(-1.13520398f +
                         t*(1.48851587f +
                         t*(-0.82215223f +
                         t*0.17087277f)))))))));
    return (x >= 0) ? 1.0f - tau : tau - 1.0f;
}
#endif

static inline float norm_cdf(float x) {
    return 0.5f * (1.0f + erff(x / 1.41421356237f)); // sqrt(2)
}

void iris_normalize(float* X) {
  for (int i = 0; i < IRIS_FEATURES; i++) {
    X[i] = (X[i] - IRIS_X_MEAN) / IRIS_X_STD;
    X[i] = norm_cdf(X[i]);
  }
}

static int iris_booleanize_n_bit(float x, int num_bits, uint8_t *out_bits) {
    if (x < 0.0f || x > 1.0f)
        return -1;

    if (!(num_bits == 1 || num_bits == 2 || num_bits == 4 || num_bits == 8))
        return -2;

    int max_val = (1 << num_bits) - 1;

    /* round-to-nearest-even */
    int int_val = (int) lrintf(x * max_val);

    for (int i = 0; i < num_bits; i++) {
        out_bits[i] = (int_val >> (num_bits - 1 - i)) & 1;
    }

    return 0;
}

uint8_t* iris_booleanize_features(
    float* X,
    int num_bits
) {
    uint8_t* X_bool = malloc(IRIS_FEATURES * num_bits * sizeof(uint8_t));
    if(!X_bool) {
        return NULL;
    }

    int offset = 0;
    for (int i = 0; i < IRIS_FEATURES; i++) {
        iris_booleanize_n_bit(X[i], num_bits, &X_bool[offset]);
        offset += num_bits;
    }
    return X_bool;
}
