#include <stdio.h>

#include "detector.h"

#define MAX_STORED_EDGES 4096
#define STABLE_MATCH_WINDOW 200

typedef struct {
    int64_t start_time;
    int64_t end_time;
    double delta;
} StoredEdge;

static StoredEdge rising_edges[MAX_STORED_EDGES];
static size_t rising_count = 0;

static StoredEdge falling_edges[MAX_STORED_EDGES];
static size_t falling_count = 0;

static void match_edges_if_possible(void) {
    size_t rise_idx = 0;
    size_t fall_search_idx = 0;
    size_t match_count = 0;

    while (rise_idx < rising_count && fall_search_idx < falling_count) {
        while (fall_search_idx < falling_count &&
               falling_edges[fall_search_idx].start_time <= rising_edges[rise_idx].start_time) {
            fall_search_idx++;
        }

        if (fall_search_idx >= falling_count) {
            break;
        }

        printf(
            "matched transition rise=(%lld,%lld,%.2f) fall=(%lld,%lld,%.2f)\n",
            (long long)rising_edges[rise_idx].start_time,
            (long long)rising_edges[rise_idx].end_time,
            rising_edges[rise_idx].delta,
            (long long)falling_edges[fall_search_idx].start_time,
            (long long)falling_edges[fall_search_idx].end_time,
            falling_edges[fall_search_idx].delta
        );

        match_count++;
        rise_idx++;
        fall_search_idx++;
    }

    if (match_count == 0) {
        return;
    }

    for (size_t i = 0; i < rising_count - match_count; i++) {
        rising_edges[i] = rising_edges[i + match_count];
    }
    rising_count -= match_count;

    for (size_t i = 0; i < falling_count - match_count; i++) {
        falling_edges[i] = falling_edges[i + match_count];
    }
    falling_count -= match_count;
}

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
    size_t stable_samples = 0;

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
            stable_samples = 0;

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

            if (output.transition_power_change > 0.0) {
                if (rising_count < MAX_STORED_EDGES) {
                    rising_edges[rising_count].start_time = output.transition_start_time;
                    rising_edges[rising_count].end_time = output.transition_end_time;
                    rising_edges[rising_count].delta = output.transition_power_change;
                    rising_count++;
                }
            } else if (output.transition_power_change < 0.0) {
                if (falling_count < MAX_STORED_EDGES) {
                    falling_edges[falling_count].start_time = output.transition_start_time;
                    falling_edges[falling_count].end_time = output.transition_end_time;
                    falling_edges[falling_count].delta = output.transition_power_change;
                    falling_count++;
                }
            }
        } else {
            stable_samples++;
            if (stable_samples >= STABLE_MATCH_WINDOW) {
                printf(
                    "stable window reached (%zu samples), matching stored edges (rising=%zu, falling=%zu)\n",
                    stable_samples,
                    rising_count,
                    falling_count
                );
                match_edges_if_possible();
                stable_samples = 0;
            }
        }

        t += 1;
    }

    if (rising_count > 0 && falling_count > 0) {
        printf(
            "end of stream, final matching pass (rising=%zu, falling=%zu)\n",
            rising_count,
            falling_count
        );
        match_edges_if_possible();
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
