#ifndef IRIS_H
#define IRIS_H

#include "iris_test.h"

#define IRIS_MODEL_BITS 4
#define IRIS_X_MEAN 3.4636666666666662
#define IRIS_X_STD  1.974000985027335

#ifdef __cplusplus
extern "C" {
#endif

void iris_normalize(float* X);
uint8_t* iris_booleanize_features(
    float* X,
    int num_bits
);

#ifdef __cplusplus
}
#endif

#endif // IRIS_H
