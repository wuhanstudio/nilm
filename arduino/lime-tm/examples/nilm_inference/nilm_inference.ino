// Install TFT_eSPI and LVGL arduino library
// - TFT_eSPI Setup: https://github.com/witnessmenow/ESP32-Cheap-Yellow-Display/blob/main/DisplayConfig/User_Setup.h
// - LVGL Setup: https://github.com/witnessmenow/ESP32-Cheap-Yellow-Display/blob/main/Examples/LVGL9/lv_conf.h

//  * Note you MUST move the 'examples' and 'demos' folders into the 'src' folder inside the lvgl library folder
// In `Arduino\libraries\lvgl\src\demos\widgets\lv_demo_widgets.h`, replace

// ```
// #include "../lv_demos.h"
// #include "../../src/draw/lv_draw.h"
// #include "../../src/draw/lv_draw_triangle.h"
// ```

// with

// ```
// #include "../lv_demos.h"
// #include "../../draw/lv_draw.h"
// #include "../../draw/lv_draw_triangle.h"
// ```

#include <SPI.h>
#include <SD.h>
#include <math.h>
#include <string.h>

#include <tsetlin.h>

#include "redd.h"
#include "redd_test.h"
#include "redd_model.h"

#include "chart.h"

#define Console Serial
static const char* TAG = "main";

#define EDGE_DETECTOR_MAX_SAMPLES 512
#define MAX_STORED_EDGES 256
#define STABLE_MATCH_WINDOW 20

typedef struct {
  bool transition;
  int64_t transition_start_time;
  int64_t transition_end_time;
  double transition_power_change;
  const double* transition_data;
  size_t transition_data_len;
} EdgeDetectorOutput;

typedef struct {
  double state_threshold;
  double noise_level;
  size_t min_n_samples;

  size_t N;
  double estimated_steady_power;

  bool ongoing_change;
  bool instantaneous_change_queue[EDGE_DETECTOR_MAX_SAMPLES];
  size_t instantaneous_change_queue_len;

  int64_t tran_start_time;
  double tran_data[EDGE_DETECTOR_MAX_SAMPLES];
  size_t tran_data_len;
  int64_t tran_end_time;

  int64_t previous_time;
  double previous_measurement;
  double last_steady_power;
} EdgeDetector;

typedef struct {
  int64_t start_time;
  int64_t end_time;
  double delta;
} StoredEdge;

static EdgeDetector detector;
static bool detector_initialized = false;
static int64_t sample_time_index = 0;
static size_t stable_samples = 0;

static StoredEdge rising_edges[MAX_STORED_EDGES];
static size_t rising_count = 0;

static StoredEdge falling_edges[MAX_STORED_EDGES];
static size_t falling_count = 0;

static int push_to_transition_data(EdgeDetector* det, double value) {
  if (det->tran_data_len >= EDGE_DETECTOR_MAX_SAMPLES) {
    return -1;
  }
  det->tran_data[det->tran_data_len++] = value;
  return 0;
}

static bool all_queue_false(const EdgeDetector* det) {
  for (size_t i = 0; i < det->instantaneous_change_queue_len; i++) {
    if (det->instantaneous_change_queue[i]) {
      return false;
    }
  }
  return true;
}

static void queue_push(EdgeDetector* det, bool value) {
  if (det->instantaneous_change_queue_len < det->min_n_samples) {
    det->instantaneous_change_queue[det->instantaneous_change_queue_len++] = value;
    return;
  }

  if (det->min_n_samples == 0) {
    return;
  }

  memmove(
      det->instantaneous_change_queue,
      det->instantaneous_change_queue + 1,
      (det->min_n_samples - 1) * sizeof(bool));
  det->instantaneous_change_queue[det->min_n_samples - 1] = value;
}

static int edge_detector_init(
    EdgeDetector* det,
    int64_t current_time,
    double current_measurement,
    double state_threshold,
    double noise_level,
    size_t min_n_samples) {
  if (!det || min_n_samples > EDGE_DETECTOR_MAX_SAMPLES) {
    return -1;
  }

  memset(det, 0, sizeof(*det));

  det->state_threshold = state_threshold;
  det->noise_level = noise_level;
  det->min_n_samples = min_n_samples;

  det->N = 0;
  det->estimated_steady_power = 0.0;
  det->ongoing_change = false;
  det->tran_end_time = current_time;

  det->previous_time = current_time;
  det->previous_measurement = current_measurement;
  det->last_steady_power = current_measurement;

  return 0;
}

static EdgeDetectorOutput edge_detector_update(
    EdgeDetector* det,
    int64_t current_time,
    double current_measurement) {
  EdgeDetectorOutput output;
  double state_change;
  bool instantaneous_change;

  memset(&output, 0, sizeof(output));

  if (!det) {
    return output;
  }

  state_change = fabs(current_measurement - det->previous_measurement);
  instantaneous_change = state_change > det->state_threshold;

  if (!det->ongoing_change && instantaneous_change) {
    if (det->tran_data_len == 0) {
      det->tran_start_time = det->previous_time;
    }
    if (push_to_transition_data(det, det->previous_measurement) != 0) {
      return output;
    }
  } else if (det->ongoing_change) {
    if (push_to_transition_data(det, det->previous_measurement) != 0) {
      return output;
    }
    if (!instantaneous_change) {
      det->tran_end_time = det->previous_time;
    }
  }

  queue_push(det, instantaneous_change);

  if (det->instantaneous_change_queue_len == det->min_n_samples && all_queue_false(det)) {
    double last_transition = det->estimated_steady_power - det->last_steady_power;

    if (fabs(last_transition) > det->noise_level) {
      int64_t transition_start_time = det->previous_time;
      if (det->tran_data_len > 0) {
        transition_start_time = det->tran_start_time;
      }

      output.transition = true;
      output.transition_start_time = transition_start_time;
      output.transition_end_time = det->tran_end_time;
      output.transition_power_change = last_transition;
      output.transition_data = det->tran_data;
      output.transition_data_len = det->tran_data_len;

      det->tran_data_len = 0;
    }

    det->last_steady_power = det->estimated_steady_power;
  }

  det->estimated_steady_power =
      ((double)det->N * det->estimated_steady_power + current_measurement) / (double)(det->N + 1);

  if (instantaneous_change) {
    det->N = 0;
  } else {
    det->N += 1;
  }

  det->ongoing_change = instantaneous_change;
  det->previous_measurement = current_measurement;
  det->previous_time = current_time;

  return output;
}

static void reset_event_pairing_state(void) {
  detector_initialized = false;
  sample_time_index = 0;
  stable_samples = 0;
  rising_count = 0;
  falling_count = 0;
}

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

    LOGI(
        TAG,
        "MATCH rise=(%lld,%lld,%.2f) fall=(%lld,%lld,%.2f)",
        (long long)rising_edges[rise_idx].start_time,
        (long long)rising_edges[rise_idx].end_time,
        rising_edges[rise_idx].delta,
        (long long)falling_edges[fall_search_idx].start_time,
        (long long)falling_edges[fall_search_idx].end_time,
        falling_edges[fall_search_idx].delta);

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

static void process_edge_event(double value) {
  EdgeDetectorOutput output;

  if (!detector_initialized) {
    if (edge_detector_init(&detector, 0, value, 15.0, 50.0, 2) != 0) {
      LOGE(TAG, "Failed to initialize edge detector");
      return;
    }
    detector_initialized = true;
    sample_time_index = 1;
    return;
  }

  output = edge_detector_update(&detector, sample_time_index, value);
  if (output.transition) {
    stable_samples = 0;

    LOGI(
        TAG,
        "EDGE start=%lld end=%lld delta=%.2f len=%u",
        (long long)output.transition_start_time,
        (long long)output.transition_end_time,
        output.transition_power_change,
        (unsigned int)output.transition_data_len);

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
      LOGI(
          TAG,
          "Stable window reached (%u), matching edges rising=%u falling=%u",
          (unsigned int)stable_samples,
          (unsigned int)rising_count,
          (unsigned int)falling_count);
      match_edges_if_possible();
      stable_samples = 0;
    }
  }

  sample_time_index++;
}

/* =========================
   ESP32 SD SPI PINS
   Change if needed
   ========================= */
#define SD_SCK 18
#define SD_MISO 19
#define SD_MOSI 23
#define SD_CS 5

#ifdef __AVR__
FILE f_out;
int sput(char c, __attribute__((unused)) FILE* f) {
  return !Console.write(c);
}
#endif

// Printf requries std library and _write implementation
extern "C" int _write(int file, char* ptr, int len) {
  (void)file;
  Console.write((uint8_t*)ptr, len);
  return len;
}

void print_progress(const char* label, int percent) {
  const int bar_width = 40;
  int filled = percent * bar_width / 100;

  printf("%s [", label);
  for (int i = 0; i < bar_width; i++) {
    if (i < filled) printf("=");
    else printf(" ");
  }
  printf("] %3d%%\n", percent);  // stay on same line
                                 // fflush(stdout);
}

int tm_redd_main() {
  // Step 0: Load Tsetlin model
  Tsetlin* model = &tsetlin_model;

  LOGI(TAG, "n_class   = %u", model->n_class);
  LOGI(TAG, "n_feature = %u", model->n_feature);
  LOGI(TAG, "n_clause  = %u", model->n_clause);
  LOGI(TAG, "n_state   = %u", model->n_state);
  LOGI(TAG, "model_type = %u", model->model_type);
  LOGI(TAG, "");

  // Outputs for model evaluation
  uint8_t predicted_class = 0;
  int32_t votes[10];

  // Step 1: Evaluate model on testing images
  int correct = 0;
  long total_boolean_time = 0;
  long total_calc_time = 0;
  for (size_t i = 0; i < REDD_TEST_SAMPLES; i++) {
    const float* input = redd_X_test[i];

    // LOGI(TAG, "Evaluating model on test sample %d (label %d)", i, redd_y_test[i]);

    // Booleanize the input using a threshold
    uint32_t start_bool = micros();
    float* X_norm = redd_normalize(input);
    uint8_t* bool_input = redd_booleanize_features(X_norm, REDD_MODEL_BITS);
    total_boolean_time += (micros() - start_bool);

    if (bool_input != NULL) {
      // Evaluate
      uint32_t start_calc = micros();
      tsetlin_evaluate(model, bool_input, votes, &predicted_class);
      total_calc_time += (micros() - start_calc);

      free(bool_input);

      // for (size_t i = 0; i < model->n_class; i++) {
      //   LOGI(TAG, "Class %d: %d votes", i, votes[i]);
      // }
      // LOGI(TAG, "Predicted class: %d with %d votes", predicted_class, votes[predicted_class]);
      // LOGI(TAG, "");

      if (predicted_class == redd_y_test[i]) {
        correct++;
      }

      // Print progress every 1000 images
      if ((i + 1) % 100 == 0) {
        char message[32];
        snprintf(message, sizeof(message), "Testing %d/%d", i + 1, REDD_TEST_SAMPLES);
        print_progress(message, (i + 1) * 100 / REDD_TEST_SAMPLES);
      }
    }
  }

  printf("[BOOL] Achieved %d us/image\n", (int)(total_boolean_time / REDD_TEST_SAMPLES));
  printf("[TM] Achieved %d us/image\n", (int)(total_calc_time / REDD_TEST_SAMPLES));

  printf("Correct predictions on test set %d / %d\n", (int)correct, (int)REDD_TEST_SAMPLES);

  return 0;
}

File fp;
uint32_t lastNILMTick = 0;  //Used to track the tick timer
uint32_t lastTick = 0;      //Used to track the tick timer
int current_index = 0;

void setup() {
  // Initialize Console
  Serial.begin(115200);
#ifdef __AVR__
  fdev_setup_stream(&f_out, sput, nullptr, _FDEV_SETUP_WRITE);  // cf https://www.nongnu.org/avr-libc/user-manual/group__avr__stdio.html#gaf41f158c022cbb6203ccd87d27301226
  stdout = &f_out;
#endif
  while (!Serial) { ; }

  //Initialise LVGL
  lv_ui_init();

  //Or try out the large standard widgets demo
  // lv_demo_widgets();
  // lv_demo_benchmark();
  lv_chart_ui();

  LOGI(TAG, "Initializing SD card...");
  if (!SD.begin(SD_CS)) {
    LOGE(TAG, "SD card initialization failed!");
    while (1) {
      delay(1000);
    }
  }
  LOGI(TAG, "SD card initialized.");

  fp = SD.open("/main.bin", "rb");
  if (fp == NULL) {
    Serial.println("Please upload data\n");
    while (1)
      ;
  }

  reset_event_pairing_state();
}

void loop() {
  // int ret = tm_redd_main();
  // if (ret < 0) {
  //   LOGE(TAG, "Inference Failed.");
  // }

  float value;
  if (millis() - lastNILMTick > 1000) {
    if (fp != NULL) {
      if (fp.read((uint8_t*)&value, sizeof(float)) == sizeof(float)) {
        // Initialize the chart using the first value
        if (current_index == 0) {
          for (uint16_t i = 0; i < LV_CHART_POINT; i++) {
            // lv_chart_set_next_value(power_chart, power_series, (lv_coord_t)value);
            lv_update_chart(value);
          }
        }

        lv_update_chart(value);
        Serial.println(value);

        process_edge_event((double)value);

        current_index = current_index + 1;
      } else {
        if (rising_count > 0 && falling_count > 0) {
          LOGI(
              TAG,
              "End of file, final matching rising=%u falling=%u",
              (unsigned int)rising_count,
              (unsigned int)falling_count);
          match_edges_if_possible();
        }
        fp.seek(0);
        reset_event_pairing_state();
      }
    }
    lastNILMTick = millis();
  }

  lv_tick_inc(millis() - lastTick);  //Update the tick timer. Tick is new for LVGL 9
  lastTick = millis();
  lv_timer_handler();  //Update the UI

  delay(5);
}
