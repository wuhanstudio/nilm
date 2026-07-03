#include <tsetlin.h>

#include "redd.h"
#include "redd_test.h"
#include "redd_model.h"

#define Console Serial
static const char* TAG = "main";

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

void print_progress(const char *label, int percent) 
{
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

    if(bool_input != NULL) {
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

  printf("Correct predictions on test set %d / %d\n", (int) correct, (int) REDD_TEST_SAMPLES);

  return 0;
}

void setup() {
  // Initialize Console
  Serial.begin(115200);
#ifdef __AVR__
  fdev_setup_stream(&f_out, sput, nullptr, _FDEV_SETUP_WRITE);  // cf https://www.nongnu.org/avr-libc/user-manual/group__avr__stdio.html#gaf41f158c022cbb6203ccd87d27301226
  stdout = &f_out;
#endif
  while (!Serial) { ; }
}

void loop() {
    int ret = tm_redd_main();
    if (ret < 0) {
      LOGE(TAG, "Inference Failed.");
    }
    delay(10000);
}
