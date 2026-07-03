#ifndef REDD_H
#define REDD_H

#include "redd_test.h"

#define REDD_MODEL_BITS 8
#define REDD_X_MEAN 691.789150058184
#define REDD_X_STD  3003.659450676937

#ifdef __cplusplus
extern "C" {
#endif

float* redd_normalize(const float* X);
uint8_t* redd_booleanize_features(
    const float* X,
    int num_bits
);

#ifdef __cplusplus
}
#endif

#endif // REDD_H
