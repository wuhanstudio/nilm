#ifndef IRIS_TEST_H
#define IRIS_TEST_H

#include <stdint.h>

/* Number of samples and features */
#define IRIS_TEST_SAMPLES 10
#define IRIS_FEATURES 4

/* Feature matrix: [sepal length, sepal width, petal length, petal width] */
static const float iris_X_test[IRIS_TEST_SAMPLES][IRIS_FEATURES] = {
    {5.1f, 3.5f, 1.4f, 0.2f},  // setosa
    {4.9f, 3.0f, 1.4f, 0.2f},  // setosa
    {6.2f, 3.4f, 5.4f, 2.3f},  // virginica
    {5.9f, 3.0f, 5.1f, 1.8f},  // virginica
    {6.0f, 2.2f, 4.0f, 1.0f},  // versicolor
    {5.5f, 2.3f, 4.0f, 1.3f},  // versicolor
    {5.7f, 2.8f, 4.5f, 1.3f},  // versicolor
    {6.3f, 3.3f, 6.0f, 2.5f},  // virginica
    {4.8f, 3.4f, 1.6f, 0.2f},  // setosa
    {5.0f, 3.5f, 1.3f, 0.3f}   // setosa
};

/* Labels: 0=setosa, 1=versicolor, 2=virginica */
static const uint8_t iris_y_test[IRIS_TEST_SAMPLES] = {
    0, 0, 2, 2, 1, 1, 1, 2, 0, 0
};

#endif /* IRIS_TEST_H */