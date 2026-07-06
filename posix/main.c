#include <stdio.h>

#include "detector.h"

int main(void) {
    EdgeDetector detector;
    EdgeDetectorOutput output;
    FILE *input_file = NULL;
    const char *input_paths[] = {
        "tests/redd_building_1_pruned.bin",
        "../tests/redd_building_1_pruned.bin",
        "redd_building_1_pruned.bin",
    };
    double sample;
    size_t i;

    int64_t t;

    for (i = 0; i < sizeof(input_paths) / sizeof(input_paths[0]); i++) {
        input_file = fopen(input_paths[i], "rb");
        if (input_file != NULL) {
            printf("Loaded input from %s\n", input_paths[i]);
            break;
        }
    }

    if (input_file == NULL) {
        fprintf(stderr, "Failed to open redd_building_1_pruned.bin\n");
        return 1;
    }

    if (fread(&sample, sizeof(double), 1, input_file) != 1) {
        fprintf(stderr, "Input file has no readable samples\n");
        fclose(input_file);
        return 1;
    }

    if (edge_detector_init(&detector, 0, sample, 15.0, 50.0, 2) != 0) {
        fprintf(stderr, "Failed to initialize edge detector\n");
        fclose(input_file);
        return 1;
    }

    t = 1;
    while (fread(&sample, sizeof(double), 1, input_file) == 1) {
        output = edge_detector_update(&detector, t, sample);
        if (output.transition) {
            printf(
                "transition start=%lld end=%lld delta=%.2f samples=%zu\n",
                (long long)output.transition_start_time,
                (long long)output.transition_end_time,
                output.transition_power_change,
                output.transition_end_time - output.transition_start_time + 1
            );
            printf("transition data: ");
            for (size_t i = 0; i < output.transition_data_len; i++) {
                printf("%.2f ", output.transition_data[i]);
            }
            printf("\n");
        }

        t += 1;
    }

    if (ferror(input_file)) {
        fprintf(stderr, "Error while reading sample data\n");
        fclose(input_file);
        edge_detector_free(&detector);
        return 1;
    }

    fclose(input_file);
    
    edge_detector_free(&detector);
    return 0;
}
