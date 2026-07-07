#include <stdio.h>

#include "detector.h"
#include "nilm_features.h"
#include "tsetlin.h"

#define STABLE_MATCH_WINDOW 200

static const char *TAG = "main";

static void log_transition_data(const float *data, size_t len) {
    const size_t values_per_line = 16;
    size_t i = 0;

    if (data == NULL || len == 0) {
        LOGI(TAG, "transition data: <empty>");
        return;
    }

    while (i < len) {
        char line[1024];
        size_t end = i + values_per_line;
        if (end > len) {
            end = len;
        }

        int offset = snprintf(line, sizeof(line), "transition data [%zu:%zu/%zu]:", i, end, len);
        if (offset < 0) {
            return;
        }

        for (size_t j = i; j < end && offset < (int)(sizeof(line) - 1); j++) {
            int wrote = snprintf(line + offset, sizeof(line) - (size_t)offset, " %.2f", data[j]);
            if (wrote < 0) {
                break;
            }
            if (wrote >= (int)(sizeof(line) - (size_t)offset)) {
                offset = (int)sizeof(line) - 1;
                break;
            }
            offset += wrote;
        }

        LOGI(TAG, "%s\n", line);
        i = end;
    }
}

int main(void) {
    EdgeDetector detector;
    EdgeDetectorOutput output;

    FILE *input_file = NULL;
    const char input_paths[] = "redd_building_1_pruned.bin";

    float sample_f32;

    size_t i;
    size_t stable_samples = 0;

    int64_t t;
    int64_t current_index;

    input_file = fopen(input_paths, "rb");
    if (input_file == NULL) {
        LOGE(TAG, "Failed to open redd_building_1_pruned.bin");
        return 1;
    }

    if (fread(&sample_f32, sizeof(float), 1, input_file) != 1) {
        LOGE(TAG, "Input file has no readable samples");
        fclose(input_file);
        return 1;
    }

    if (edge_detector_init(&detector, 0, sample_f32, 15.0f, 50.0f, 2) != 0) {
        LOGE(TAG, "Failed to initialize edge detector");
        fclose(input_file);
        return 1;
    }

    t = 1;
    current_index = 1;
    while (fread(&sample_f32, sizeof(float), 1, input_file) == 1) {
        output = edge_detector_update(&detector, t, sample_f32);
        if (output.transition) {
            stable_samples = 0;

            LOGI(
                TAG,
                "transition start=%lld end=%lld delta=%.2f samples=%zu",
                (long long)output.transition_start_time,
                (long long)output.transition_end_time,
                output.transition_power_change,
                output.transition_end_time - output.transition_start_time + 1
            );
            log_transition_data(output.transition_data, output.transition_data_len);

            features_record_transition(&output);
        } else {
            stable_samples++;
            if (stable_samples >= STABLE_MATCH_WINDOW) {
                LOGI(
                    TAG,
                    "stable window reached (%zu samples), matching stored edges (rising=%zu, falling=%zu)\n",
                    stable_samples,
                    features_get_rising_count(),
                    features_get_falling_count()
                );
                features_match_pending_edges(input_file, current_index);
                stable_samples = 0;
            }
        }

        t += 1;
        current_index += 1;
    }

    features_finalize(input_file, current_index);

    if (ferror(input_file)) {
        LOGE(TAG, "Error while reading sample data");
        fclose(input_file);
        edge_detector_free(&detector);
        return 1;
    }

    fclose(input_file);
    
    edge_detector_free(&detector);
    return 0;
}
