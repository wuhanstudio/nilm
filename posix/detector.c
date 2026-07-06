#include "detector.h"

#include <math.h>
#include <string.h>

static int push_to_transition_data(EdgeDetector *detector, double value) {
    if (detector->tran_data_len >= EDGE_DETECTOR_MAX_SAMPLES) {
        return -1;
    }
    detector->tran_data[detector->tran_data_len++] = value;
    return 0;
}

static bool all_queue_false(const EdgeDetector *detector) {
    size_t i;
    for (i = 0; i < detector->instantaneous_change_queue_len; i++) {
        if (detector->instantaneous_change_queue[i]) {
            return false;
        }
    }
    return true;
}

static void queue_push(EdgeDetector *detector, bool value) {
    if (detector->instantaneous_change_queue_len < detector->min_n_samples) {
        detector->instantaneous_change_queue[detector->instantaneous_change_queue_len++] = value;
        return;
    }

    if (detector->min_n_samples == 0) {
        return;
    }

    memmove(
        detector->instantaneous_change_queue,
        detector->instantaneous_change_queue + 1,
        (detector->min_n_samples - 1) * sizeof(bool)
    );
    detector->instantaneous_change_queue[detector->min_n_samples - 1] = value;
}

int edge_detector_init(
    EdgeDetector *detector,
    int64_t current_time,
    double current_measurement,
    double state_threshold,
    double noise_level,
    size_t min_n_samples
) {
    if (!detector || min_n_samples > EDGE_DETECTOR_MAX_SAMPLES) {
        return -1;
    }

    memset(detector, 0, sizeof(*detector));

    detector->state_threshold = state_threshold;
    detector->noise_level = noise_level;
    detector->min_n_samples = min_n_samples;

    detector->N = 0;
    detector->estimated_steady_power = 0.0;
    detector->ongoing_change = false;
    detector->tran_end_time = current_time;

    detector->previous_time = current_time;
    detector->previous_measurement = current_measurement;
    detector->last_steady_power = current_measurement;

    detector->last_transition_valid = false;

    return 0;
}

void edge_detector_free(EdgeDetector *detector) {
    if (!detector) {
        return;
    }

    detector->instantaneous_change_queue_len = 0;
    detector->tran_data_len = 0;
}

EdgeDetectorOutput edge_detector_update(
    EdgeDetector *detector,
    int64_t current_time,
    double current_measurement
) {
    EdgeDetectorOutput output;
    double state_change;
    bool instantaneous_change;

    memset(&output, 0, sizeof(output));

    if (!detector) {
        return output;
    }

    state_change = fabs(current_measurement - detector->previous_measurement);

    instantaneous_change = state_change > detector->state_threshold;

    // Collect data for transitions
    if (!detector->ongoing_change && instantaneous_change) {
        // Starting a new transition
        if (detector->tran_data_len == 0) {
            detector->tran_start_time = detector->previous_time;
        }
        if (push_to_transition_data(detector, detector->previous_measurement) != 0) {
            return output;
        }
    } else if (detector->ongoing_change) {
        // Continuing or ending a transition
        if (push_to_transition_data(detector, detector->previous_measurement) != 0) {
            return output;
        }
        if (!instantaneous_change) {
            detector->tran_end_time = detector->previous_time;
        }
    }

    // Update the instantaneous change queue
    queue_push(detector, instantaneous_change);

    if (detector->instantaneous_change_queue_len == detector->min_n_samples &&
        all_queue_false(detector)) {
        double last_transition = detector->estimated_steady_power - detector->last_steady_power;

        if (fabs(last_transition) > detector->noise_level) {
            int64_t transition_start_time = detector->previous_time;
            if (detector->tran_data_len > 0) {
                transition_start_time = detector->tran_start_time;
            }

            // Store the last detected transition
            detector->last_transition_valid = true;
            detector->last_transition_start_time = transition_start_time;
            detector->last_transition_end_time = detector->tran_end_time;
            detector->last_transition_power_change = last_transition;

            // Prepare output with current transition data
            output.transition = true;
            output.transition_start_time = transition_start_time;
            output.transition_end_time = detector->tran_end_time;
            output.transition_power_change = last_transition;
            output.transition_data = detector->tran_data;
            output.transition_data_len = detector->tran_data_len;

            // Reset current transition data
            detector->tran_data_len = 0;
        }

        detector->last_steady_power = detector->estimated_steady_power;
    }

    detector->estimated_steady_power =
        ((double)detector->N * detector->estimated_steady_power + current_measurement) /
        (double)(detector->N + 1);

    if (instantaneous_change) {
        detector->N = 0;
    } else {
        detector->N += 1;
    }

    detector->ongoing_change = instantaneous_change;
    detector->previous_measurement = current_measurement;
    detector->previous_time = current_time;

    return output;
}
