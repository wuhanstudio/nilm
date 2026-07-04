#include <stdio.h>

#include "detector_posix.h"

int main(void) {
    EdgeDetector detector;
    EdgeDetectorOutput output;

    int64_t t;
    double samples[] = {
        100.0, 101.0, 99.0, 102.0,
        210.0, 212.0, 209.0,
        211.0, 210.0, 209.0,
        105.0, 102.0, 100.0,
    };
    size_t num_samples = sizeof(samples) / sizeof(samples[0]);

    if (edge_detector_init(&detector, 0, samples[0], 15.0, 50.0, 2) != 0) {
        fprintf(stderr, "Failed to initialize edge detector\n");
        return 1;
    }

    for (t = 1; t < (int64_t)num_samples; t++) {
        output = edge_detector_update(&detector, t, samples[t]);
        if (output.transition) {
            printf(
                "transition start=%lld end=%lld delta=%.2f samples=%zu\n",
                (long long)output.transition_start_time,
                (long long)output.transition_end_time,
                output.transition_power_change,
                output.transition_data_len
            );
        }
    }

    printf("total transitions: %zu\n", edge_detector_num_transitions(&detector));
    edge_detector_free(&detector);
    return 0;
}
