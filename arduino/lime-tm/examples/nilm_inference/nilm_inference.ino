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

#include <lvgl.h>
#include <TFT_eSPI.h>

#include <examples/lv_examples.h>
#include <demos/lv_demos.h>

/*Set to your screen resolution and rotation*/
#define TFT_HOR_RES 240
#define TFT_VER_RES 320
#define TFT_ROTATION LV_DISPLAY_ROTATION_90

/*LVGL draw into this buffer, 1/10 screen size usually works well. The size is in bytes*/
#define DRAW_BUF_SIZE (TFT_HOR_RES * TFT_VER_RES / 10 * (LV_COLOR_DEPTH / 8))

#if LV_USE_LOG != 0
void my_print(lv_log_level_t level, const char* buf) {
  LV_UNUSED(level);
  Serial.println(buf);
  Serial.flush();
}
#endif

/* LVGL calls it when a rendered image needs to copied to the display*/
void my_disp_flush(lv_display_t* disp, const lv_area_t* area, uint8_t* px_map) {
  /*Call it to tell LVGL you are ready*/
  lv_disp_flush_ready(disp);
}

#include <tsetlin.h>

#include "redd.h"
#include "redd_test.h"
#include "redd_model.h"

#define Console Serial
static const char* TAG = "main";

/* =========================
   ESP32 SD SPI PINS
   Change if needed
   ========================= */
#define SD_SCK  18
#define SD_MISO 19
#define SD_MOSI 23
#define SD_CS    5

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

void printDirectory(File dir) {
  while (true) {
    File entry = dir.openNextFile();
    if (!entry) {
      // no more files
      break;
    }

    if (entry.isDirectory()) {
      LOGI(TAG, "/");
      printDirectory(entry);
    } else {
      // files have sizes, directories do not
      LOGI(TAG, "%s | %d Bytes", entry.name(), entry.size());
    }
    entry.close();
  }
}

uint8_t* draw_buf;      //draw_buf is allocated on heap otherwise the static area is too big on ESP32 at compile
uint32_t lastTick = 0;  //Used to track the tick timer

File fp;
lv_obj_t* power_chart = nullptr;
lv_chart_series_t* power_series = nullptr;
lv_obj_t* value_label = nullptr;
float chart_min = 0.0f;
float chart_max = 1000.0f;

void create_chart_ui() {
  lv_obj_t* scr = lv_scr_act();

  value_label = lv_label_create(scr);
  lv_label_set_text(value_label, "Value: --");
  lv_obj_align(value_label, LV_ALIGN_TOP_MID, 0, 8);

  power_chart = lv_chart_create(scr);
  lv_obj_set_size(power_chart, TFT_HOR_RES - 20, TFT_VER_RES - 50);
  lv_obj_align(power_chart, LV_ALIGN_BOTTOM_MID, 0, -6);
  lv_chart_set_type(power_chart, LV_CHART_TYPE_LINE);
  lv_chart_set_point_count(power_chart, 60);
  lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);
  lv_chart_set_div_line_count(power_chart, 5, 6);

  power_series = lv_chart_add_series(power_chart, lv_palette_main(LV_PALETTE_GREEN), LV_CHART_AXIS_PRIMARY_Y);

  for (uint16_t i = 0; i < 60; i++) {
    lv_chart_set_next_value(power_chart, power_series, 0);
  }
}

void setup() {
  // Initialize Console
  Serial.begin(115200);
#ifdef __AVR__
  fdev_setup_stream(&f_out, sput, nullptr, _FDEV_SETUP_WRITE);  // cf https://www.nongnu.org/avr-libc/user-manual/group__avr__stdio.html#gaf41f158c022cbb6203ccd87d27301226
  stdout = &f_out;
#endif
  while (!Serial) { ; }

  //Initialise LVGL
  lv_init();
  draw_buf = new uint8_t[DRAW_BUF_SIZE];
  lv_display_t* disp;
  disp = lv_tft_espi_create(TFT_HOR_RES, TFT_VER_RES, draw_buf, DRAW_BUF_SIZE);
  lv_display_set_rotation(disp, TFT_ROTATION);

  //Or try out the large standard widgets demo
  // lv_demo_widgets();
  // lv_demo_benchmark();
  // lv_demo_keypad_encoder();

  create_chart_ui();

  LOGI(TAG, "Initializing SD card...");
  if (!SD.begin(SD_CS)) {
    LOGE(TAG, "SD card initialization failed!");
    while (1) {
      delay(1000);
    }
  }
  LOGI(TAG, "SD card initialized.");

  // Print files on the SD card
  File root = SD.open("/");
  if (root) {
    printDirectory(root);
  } else {
    LOGI(TAG, "Could not open root");
  }
  root.close();

  fp = SD.open("/main.bin", "rb");

  if (fp == NULL) {
    Serial.println("Please upload data\n");
    while (1)
      ;
  }
}

uint32_t lastNILMTick = 0;  //Used to track the tick timer

void loop() {
  // int ret = tm_redd_main();
  // if (ret < 0) {
  //   LOGE(TAG, "Inference Failed.");
  // }

  float value;
  if (millis() - lastNILMTick > 1000) {
    if (fp != NULL) {
      if (fp.read((uint8_t*)&value, sizeof(float)) == sizeof(float)) {
        Serial.println(value);

        if (value < chart_min) {
          chart_min = value;
          lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);
        }
        if (value > chart_max) {
          chart_max = value;
          lv_chart_set_range(power_chart, LV_CHART_AXIS_PRIMARY_Y, (int32_t)chart_min, (int32_t)chart_max);
        }

        lv_chart_set_next_value(power_chart, power_series, (lv_coord_t)value);

        char label_text[32];
        snprintf(label_text, sizeof(label_text), "Value: %.2f", value);
        lv_label_set_text(value_label, label_text);
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
