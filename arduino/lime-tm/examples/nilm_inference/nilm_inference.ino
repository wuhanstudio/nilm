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
#include "detector.h"
#include "features.h"

#include "chart.h"

#define Console Serial
static const char *TAG = "main";

#define MAX_STORED_EDGES 256
#define STABLE_MATCH_WINDOW 20
#define MATCH_BATCH_LIMIT 1

// Forward declarations to prevent Arduino auto-prototype ordering issues.
static void reset_event_pairing_state(void);
static void match_edges_if_possible(size_t max_matches);
static void process_edge_event(float value);

static EdgeDetector detector;
static bool detector_initialized = false;
static int64_t sample_time_index = 0;
static size_t stable_samples = 0;

static StoredEdge rising_edges[MAX_STORED_EDGES];
static size_t rising_count = 0;

static StoredEdge falling_edges[MAX_STORED_EDGES];
static size_t falling_count = 0;
static size_t matched_event_count = 0;

extern File fp;
extern int current_index;

static void reset_event_pairing_state(void)
{
  detector_initialized = false;
  sample_time_index = 0;
  stable_samples = 0;
  rising_count = 0;
  falling_count = 0;
  matched_event_count = 0;
}

static void match_edges_if_possible(size_t max_matches)
{
  if (max_matches == 0)
  {
    return;
  }

  size_t rise_idx = 0;
  size_t fall_search_idx = 0;
  size_t match_count = 0;

  while (rise_idx < rising_count && fall_search_idx < falling_count && match_count < max_matches)
  {
    while (fall_search_idx < falling_count &&
           falling_edges[fall_search_idx].start_time <= rising_edges[rise_idx].start_time)
    {
      fall_search_idx++;
    }

    if (fall_search_idx >= falling_count)
    {
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

    LOGI(TAG, "MATCHING rising edge %zu with falling edge %zu", rise_idx, fall_search_idx);

    features_extract_and_log_matched_episode_features(
        &rising_edges[rise_idx],
        &falling_edges[fall_search_idx],
        fp);

    fp.seek((uint32_t)current_index * sizeof(float));

    match_count++;
    rise_idx++;
    fall_search_idx++;
  }

  if (match_count == 0)
  {
    return;
  }

  matched_event_count += match_count;

  for (size_t i = 0; i < rising_count - match_count; i++)
  {
    rising_edges[i] = rising_edges[i + match_count];
  }
  rising_count -= match_count;

  for (size_t i = 0; i < falling_count - match_count; i++)
  {
    falling_edges[i] = falling_edges[i + match_count];
  }
  falling_count -= match_count;
}

static void process_edge_event(float value)
{
  EdgeDetectorOutput output;

  if (!detector_initialized)
  {
    if (edge_detector_init(&detector, 0, value, 15.0f, 50.0f, 2) != 0)
    {
      LOGE(TAG, "Failed to initialize edge detector");
      return;
    }
    detector_initialized = true;
    sample_time_index = 1;
    return;
  }

  output = edge_detector_update(&detector, sample_time_index, value);
  if (output.transition)
  {
    stable_samples = 0;

    LOGI(
        TAG,
        "EDGE start=%lld end=%lld delta=%.2f len=%u",
        (long long)output.transition_start_time,
        (long long)output.transition_end_time,
        output.transition_power_change,
        (unsigned int)output.transition_data_len);

    if (output.transition_power_change > 0.0)
    {
      if (rising_count < MAX_STORED_EDGES)
      {
        rising_edges[rising_count].start_time = output.transition_start_time;
        rising_edges[rising_count].end_time = output.transition_end_time;
        rising_edges[rising_count].delta = output.transition_power_change;
        rising_count++;
      }
    }
    else if (output.transition_power_change < 0.0)
    {
      if (falling_count < MAX_STORED_EDGES)
      {
        falling_edges[falling_count].start_time = output.transition_start_time;
        falling_edges[falling_count].end_time = output.transition_end_time;
        falling_edges[falling_count].delta = output.transition_power_change;
        falling_count++;
      }
    }
  }
  else
  {
    stable_samples++;
    if (stable_samples >= STABLE_MATCH_WINDOW)
    {
      delay(5);
      LOGI(
          TAG,
          "Stable window reached (%u), matching edges rising=%u falling=%u",
          (unsigned int)stable_samples,
          (unsigned int)rising_count,
          (unsigned int)falling_count);
      match_edges_if_possible(MATCH_BATCH_LIMIT);
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
int sput(char c, __attribute__((unused)) FILE *f)
{
  return !Console.write(c);
}
#endif

// Printf requries std library and _write implementation
extern "C" int _write(int file, char *ptr, int len)
{
  (void)file;
  Console.write((uint8_t *)ptr, len);
  return len;
}

void print_progress(const char *label, int percent)
{
  const int bar_width = 40;
  int filled = percent * bar_width / 100;

  printf("%s [", label);
  for (int i = 0; i < bar_width; i++)
  {
    if (i < filled)
      printf("=");
    else
      printf(" ");
  }
  printf("] %3d%%\n", percent); // stay on same line
                                // fflush(stdout);
}

int tm_redd_main()
{
  // Step 0: Load Tsetlin model
  Tsetlin *model = &tsetlin_model;

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
  for (size_t i = 0; i < REDD_TEST_SAMPLES; i++)
  {
    const float *input = redd_X_test[i];

    // LOGI(TAG, "Evaluating model on test sample %d (label %d)", i, redd_y_test[i]);

    // Booleanize the input using a threshold
    uint32_t start_bool = micros();
    float *X_norm = redd_normalize(input);
    uint8_t *bool_input = redd_booleanize_features(X_norm, REDD_MODEL_BITS);
    total_boolean_time += (micros() - start_bool);

    if (bool_input != NULL)
    {
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

      if (predicted_class == redd_y_test[i])
      {
        correct++;
      }

      // Print progress every 1000 images
      if ((i + 1) % 100 == 0)
      {
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
uint32_t lastNILMTick = 0; // Used to track the tick timer
uint32_t lastTick = 0;     // Used to track the tick timer
int current_index = 0;

Tsetlin *nilm_get_tm_model(void)
{
  return &tsetlin_model;
}

void setup()
{
  // Initialize Console
  Serial.begin(115200);
#ifdef __AVR__
  fdev_setup_stream(&f_out, sput, nullptr, _FDEV_SETUP_WRITE); // cf https://www.nongnu.org/avr-libc/user-manual/group__avr__stdio.html#gaf41f158c022cbb6203ccd87d27301226
  stdout = &f_out;
#endif
  while (!Serial)
  {
    ;
  }

  // Initialise LVGL
  lv_ui_init();
  lv_chart_ui();

  LOGI(TAG, "Initializing SD card...");
  if (!SD.begin(SD_CS))
  {
    LOGE(TAG, "SD card initialization failed!");
    while (1)
    {
      delay(1000);
    }
  }
  LOGI(TAG, "SD card initialized.");

  fp = SD.open("/main.bin", "rb");
  if (fp == NULL)
  {
    Serial.println("Please upload data\n");
    while (1)
      ;
  }

  reset_event_pairing_state();
}

void loop()
{
  // int ret = tm_redd_main();
  // if (ret < 0) {
  //   LOGE(TAG, "Inference Failed.");
  // }

  float value;
  if (millis() - lastNILMTick > 100)
  {
    if (fp != NULL)
    {
      if (fp.read((uint8_t *)&value, sizeof(float)) == sizeof(float))
      {
        // Initialize the chart using the first value
        if (current_index == 0)
        {
          for (uint16_t i = 0; i < LV_CHART_POINT; i++)
          {
            // lv_chart_set_next_value(power_chart, power_series, (lv_coord_t)value);
            lv_update_chart(value, rising_count, falling_count, matched_event_count);
          }
        }

        // Edge detection and event processing
        process_edge_event(value);

        lv_update_chart(value, rising_count, falling_count, matched_event_count);
        Serial.println(value);
        current_index++;
      }
      else
      {
        if (rising_count > 0 && falling_count > 0)
        {
          LOGI(
              TAG,
              "End of file, final matching rising=%u falling=%u",
              (unsigned int)rising_count,
              (unsigned int)falling_count);
          while (rising_count > 0 && falling_count > 0)
          {
            match_edges_if_possible(MAX_STORED_EDGES);
          }
        }
        fp.seek(0);
        reset_event_pairing_state();
      }
    }
    lastNILMTick = millis();
  }

  lv_tick_inc(millis() - lastTick); // Update the tick timer. Tick is new for LVGL 9
  lastTick = millis();
  lv_timer_handler(); // Update the UI

  delay(5);
}
