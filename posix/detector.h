#ifndef __DETECTOR_H__
#define __DETECTOR_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef EDGE_DETECTOR_MALLOC
#include <stdlib.h>
#define EDGE_DETECTOR_MALLOC(sz) malloc(sz)
#define EDGE_DETECTOR_CALLOC(n, sz) calloc((n), (sz))
#define EDGE_DETECTOR_REALLOC(ptr, sz) realloc((ptr), (sz))
#define EDGE_DETECTOR_FREE(ptr) free(ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * On Arduino ESP32 you can override allocator hooks before including this file,
 * for example to place allocations in PSRAM:
 *
 *   #define EDGE_DETECTOR_MALLOC(sz) heap_caps_malloc((sz), MALLOC_CAP_SPIRAM)
 *   #define EDGE_DETECTOR_CALLOC(n, sz) heap_caps_calloc((n), (sz), MALLOC_CAP_SPIRAM)
 *   #define EDGE_DETECTOR_REALLOC(ptr, sz) heap_caps_realloc((ptr), (sz), MALLOC_CAP_SPIRAM)
 *   #define EDGE_DETECTOR_FREE(ptr) heap_caps_free((ptr))
 */

typedef struct {
    bool transition;
    int64_t transition_start_time;
    int64_t transition_end_time;
    double transition_power_change;
    const double *transition_data;
    size_t transition_data_len;
} EdgeDetectorOutput;

typedef struct {
    double *data;
    size_t len;
    size_t cap;
} EdgeDetectorDoubleVec;

typedef struct {
    int64_t *data;
    size_t len;
    size_t cap;
} EdgeDetectorI64Vec;

typedef struct {
    double state_threshold;
    double noise_level;
    size_t min_n_samples;

    size_t N;
    double estimated_steady_power;

    bool ongoing_change;
    bool *instantaneous_change_queue;
    size_t instantaneous_change_queue_len;

    EdgeDetectorI64Vec tran_start_time;
    EdgeDetectorDoubleVec tran_data;
    int64_t tran_end_time;

    EdgeDetectorI64Vec index_transitions_start;
    EdgeDetectorI64Vec index_transitions_end;

    int64_t previous_time;
    double previous_measurement;
    double last_steady_power;

    EdgeDetectorDoubleVec transitions;
    EdgeDetectorDoubleVec steady_states;
    EdgeDetectorI64Vec index_steady_states;

    EdgeDetectorDoubleVec emitted_transition_data;
} EdgeDetector;

int edge_detector_init(
    EdgeDetector *detector,
    int64_t current_time,
    double current_measurement,
    double state_threshold,
    double noise_level,
    size_t min_n_samples
);

void edge_detector_free(EdgeDetector *detector);

EdgeDetectorOutput edge_detector_update(
    EdgeDetector *detector,
    int64_t current_time,
    double current_measurement
);

size_t edge_detector_num_transitions(const EdgeDetector *detector);

#ifdef __cplusplus
}
#endif

#endif // __DETECTOR_H__
