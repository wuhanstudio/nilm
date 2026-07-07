#ifndef __DETECTOR_H__
#define __DETECTOR_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#define EDGE_DETECTOR_MAX_SAMPLES 512

typedef struct {
    bool transition;
    int64_t transition_start_time;
    int64_t transition_end_time;
    float transition_power_change;
    const float *transition_data;
    size_t transition_data_len;
} EdgeDetectorOutput;

typedef struct {
    float state_threshold;
    float noise_level;
    size_t min_n_samples;

    size_t N;
    float estimated_steady_power;

    bool ongoing_change;
    bool instantaneous_change_queue[EDGE_DETECTOR_MAX_SAMPLES];
    size_t instantaneous_change_queue_len;

    // Current transition being detected
    int64_t tran_start_time;
    float tran_data[EDGE_DETECTOR_MAX_SAMPLES];
    size_t tran_data_len;
    int64_t tran_end_time;

    int64_t previous_time;
    float previous_measurement;
    float last_steady_power;

    // Most recent detected transition (output)
    bool last_transition_valid;
    int64_t last_transition_start_time;
    int64_t last_transition_end_time;
    float last_transition_power_change;
    float last_transition_data[EDGE_DETECTOR_MAX_SAMPLES];
    size_t last_transition_data_len;
} EdgeDetector;

int edge_detector_init(
    EdgeDetector *detector,
    int64_t current_time,
    float current_measurement,
    float state_threshold,
    float noise_level,
    size_t min_n_samples
);

void edge_detector_free(EdgeDetector *detector);

EdgeDetectorOutput edge_detector_update(
    EdgeDetector *detector,
    int64_t current_time,
    float current_measurement
);

#ifdef __cplusplus
}
#endif

#endif // __DETECTOR_H__
