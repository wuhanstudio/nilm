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

#include <tsetlin.h>

#include "redd.h"
#include "redd_test.h"
#include "redd_model.h"

#include "chart.h"

#define Console Serial
static const char* TAG = "main";

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

        current_index = current_index + 1;
      } else {
        fp.seek(0);
      }
    }
    lastNILMTick = millis();
  }

  lv_tick_inc(millis() - lastTick);  //Update the tick timer. Tick is new for LVGL 9
  lastTick = millis();
  lv_timer_handler();  //Update the UI

  delay(5);
}
