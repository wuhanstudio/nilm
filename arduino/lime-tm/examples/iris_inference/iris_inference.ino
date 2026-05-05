#include <tsetlin.h>

#include "iris.h"
#include "iris_test.h"
#include "iris_model.h"

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

int tm_iris_main() {
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
  for (size_t i = 0; i < IRIS_TEST_SAMPLES; i++) {
    const float* input = iris_X_test[i];

    LOGI(TAG, "Evaluating model on test sample %d (label %d)", i, iris_y_test[i]);

    // Booleanize the input using a threshold
    float* X_norm = iris_normalize(input);
    uint8_t* bool_input = iris_booleanize_features(X_norm, IRIS_MODEL_BITS);
  
    if(bool_input != NULL) {
      // Evaluate
      tsetlin_evaluate(model, bool_input, votes, &predicted_class);
      free(bool_input);
    
      for (size_t i = 0; i < model->n_class; i++) {
        LOGI(TAG, "Class %d: %d votes", i, votes[i]);
      }
      LOGI(TAG, "Predicted class: %d with %d votes", predicted_class, votes[predicted_class]);
      LOGI(TAG, "");

      if (predicted_class == iris_y_test[i]) {
            correct++;
      }
    }
  }
  printf("Correct predictions on test set %d / %d\n", (int) correct, (int) IRIS_TEST_SAMPLES);

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
    int ret = tm_iris_main();
    if (ret < 0) {
      LOGE(TAG, "Inference Failed.");
    }
    delay(10000);
}
