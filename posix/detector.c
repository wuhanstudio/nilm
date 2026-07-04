#include "detector_posix.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int ensure_double_capacity(EdgeDetectorDoubleVec *vec, size_t needed) {
    if (needed <= vec->cap) {
        return 0;
    }

    size_t new_cap = vec->cap ? vec->cap : 8;
    while (new_cap < needed) {
        if (new_cap > ((size_t)-1) / 2) {
            return -1;
        }
        new_cap *= 2;
    }

    double *new_data = (double *)realloc(vec->data, new_cap * sizeof(double));
    if (!new_data) {
        return -1;
    }

    vec->data = new_data;
    vec->cap = new_cap;
    return 0;
}

static int ensure_i64_capacity(EdgeDetectorI64Vec *vec, size_t needed) {
    if (needed <= vec->cap) {
        return 0;
    }

    size_t new_cap = vec->cap ? vec->cap : 8;
    while (new_cap < needed) {
        if (new_cap > ((size_t)-1) / 2) {
            return -1;
        }
        new_cap *= 2;
    }

    int64_t *new_data = (int64_t *)realloc(vec->data, new_cap * sizeof(int64_t));
    if (!new_data) {
        return -1;
    }

    vec->data = new_data;
    vec->cap = new_cap;
    return 0;
}

static int push_double(EdgeDetectorDoubleVec *vec, double value) {
    if (ensure_double_capacity(vec, vec->len + 1) != 0) {
        return -1;
    }
    vec->data[vec->len++] = value;
    return 0;
}

static int push_i64(EdgeDetectorI64Vec *vec, int64_t value) {
    if (ensure_i64_capacity(vec, vec->len + 1) != 0) {
        return -1;
    }
    vec->data[vec->len++] = value;
    return 0;
}

static void clear_double_vec(EdgeDetectorDoubleVec *vec) {
    vec->len = 0;
}

static void clear_i64_vec(EdgeDetectorI64Vec *vec) {
    vec->len = 0;
}

static void free_double_vec(EdgeDetectorDoubleVec *vec) {
    free(vec->data);
    vec->data = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static void free_i64_vec(EdgeDetectorI64Vec *vec) {
    free(vec->data);
    vec->data = NULL;
    vec->len = 0;
    vec->cap = 0;
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
    if (!detector) {
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

    if (min_n_samples > 0) {
        detector->instantaneous_change_queue = (bool *)calloc(min_n_samples, sizeof(bool));
        if (!detector->instantaneous_change_queue) {
            edge_detector_free(detector);
            return -1;
        }
    }

    return 0;
}

void edge_detector_free(EdgeDetector *detector) {
    if (!detector) {
        return;
    }

    free(detector->instantaneous_change_queue);
    detector->instantaneous_change_queue = NULL;
    detector->instantaneous_change_queue_len = 0;

    free_i64_vec(&detector->tran_start_time);
    free_double_vec(&detector->tran_data);
    free_i64_vec(&detector->index_transitions_start);
    free_i64_vec(&detector->index_transitions_end);
    free_double_vec(&detector->transitions);
    free_double_vec(&detector->steady_states);
    free_i64_vec(&detector->index_steady_states);
    free_double_vec(&detector->emitted_transition_data);
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

    clear_double_vec(&detector->emitted_transition_data);

    state_change = fabs(current_measurement - detector->previous_measurement);

    if (fabs(current_measurement - detector->last_steady_power) > detector->noise_level) {
        if (push_i64(&detector->tran_start_time, detector->previous_time) != 0 ||
            push_double(&detector->tran_data, detector->previous_measurement) != 0) {
            return output;
        }
    }

    instantaneous_change = state_change > detector->state_threshold;

    if (detector->ongoing_change) {
        if (push_double(&detector->tran_data, detector->previous_measurement) != 0) {
            return output;
        }
        if (!instantaneous_change) {
            detector->tran_end_time = detector->previous_time;
        }
    }

    if (detector->instantaneous_change_queue_len == detector->min_n_samples &&
        all_queue_false(detector)) {
        double last_transition = detector->estimated_steady_power - detector->last_steady_power;

        if (fabs(last_transition) > detector->noise_level) {
            int64_t transition_start_time = detector->previous_time;
            if (detector->tran_start_time.len > 0) {
                transition_start_time = detector->tran_start_time.data[0];
            }

            if (push_i64(&detector->index_transitions_end, detector->tran_end_time) != 0 ||
                push_i64(&detector->index_transitions_start, transition_start_time) != 0) {
                return output;
            }

            clear_i64_vec(&detector->tran_start_time);

            if (ensure_double_capacity(&detector->emitted_transition_data, detector->tran_data.len) != 0) {
                return output;
            }

            if (detector->tran_data.len > 0) {
                memcpy(
                    detector->emitted_transition_data.data,
                    detector->tran_data.data,
                    detector->tran_data.len * sizeof(double)
                );
            }
            detector->emitted_transition_data.len = detector->tran_data.len;
            clear_double_vec(&detector->tran_data);

            if (push_double(&detector->transitions, last_transition) != 0 ||
                push_i64(&detector->index_steady_states, detector->tran_end_time) != 0 ||
                push_double(&detector->steady_states, detector->estimated_steady_power) != 0) {
                return output;
            }

            output.transition = true;
            output.transition_start_time = transition_start_time;
            output.transition_end_time = detector->tran_end_time;
            output.transition_power_change = last_transition;
            output.transition_data = detector->emitted_transition_data.data;
            output.transition_data_len = detector->emitted_transition_data.len;
        }

        detector->last_steady_power = detector->estimated_steady_power;
    }

    if (instantaneous_change) {
        detector->N = 0;
    }

    detector->estimated_steady_power =
        ((double)detector->N * detector->estimated_steady_power + current_measurement) /
        (double)(detector->N + 1);

    queue_push(detector, instantaneous_change);

    detector->N += 1;
    detector->ongoing_change = instantaneous_change;
    detector->previous_measurement = current_measurement;
    detector->previous_time = current_time;

    return output;
}

size_t edge_detector_num_transitions(const EdgeDetector *detector) {
    if (!detector) {
        return 0;
    }
    return detector->transitions.len;
}
