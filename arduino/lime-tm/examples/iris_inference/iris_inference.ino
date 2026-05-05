#include <tsetlin.h>

#include "iris_model.h"
#include "iris_test.h"

static const char* TAG = "main";

// #define CONSOLE_USE_SERIAL
#define CONSOLE_USE_CDC
// #define CONSOLE_USE_RTT

// Print to Serial 1
#if defined(CONSOLE_USE_SERIAL)

#ifndef HAVE_HWSERIAL1
HardwareSerial Serial1(PA10, PA9);
#endif

#define Console Serial1
#endif

// Print to USB CDC
#if defined(CONSOLE_USE_CDC)
#define Console Serial
#endif

// Print to ST-Link / CMSIS-DAP Debugger (RTT)
#if defined(CONSOLE_USE_RTT)
#include <RTTStream.h>
RTTStream rtt;
#define Console rtt
#endif

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

#define IRIS_X_MEAN 3.4636666666666662
#define IRIS_X_STD  1.974000985027335

#ifdef __AVR__
float erff(float x) {
    // Abramowitz & Stegun approximation
    float t = 1.0f / (1.0f + 0.5f * fabsf(x));
    float tau = t * expf(-x*x - 1.26551223f +
                         t*(1.00002368f +
                         t*(0.37409196f +
                         t*(0.09678418f +
                         t*(-0.18628806f +
                         t*(0.27886807f +
                         t*(-1.13520398f +
                         t*(1.48851587f +
                         t*(-0.82215223f +
                         t*0.17087277f)))))))));
    return (x >= 0) ? 1.0f - tau : tau - 1.0f;
}
#endif

static inline float norm_cdf(float x) {
    return 0.5f * (1.0f + erff(x / 1.41421356237f)); // sqrt(2)
}

static void iris_normalize_img(float* X) {
  for (int i = 0; i < IRIS_FEATURES; i++) {
    X[i] = (X[i] - IRIS_X_MEAN) / IRIS_X_STD;
    X[i] = norm_cdf(X[i]);
  }
}

#define IRIS_MODEL_BITS 4

static int iris_booleanize_n_bit(float x, int num_bits, uint8_t *out_bits) {
    if (x < 0.0f || x > 1.0f)
        return -1;

    if (!(num_bits == 1 || num_bits == 2 || num_bits == 4 || num_bits == 8))
        return -2;

    int max_val = (1 << num_bits) - 1;

    /* round-to-nearest-even */
    int int_val = (int) lrintf(x * max_val);

    for (int i = 0; i < num_bits; i++) {
        out_bits[i] = (int_val >> (num_bits - 1 - i)) & 1;
    }

    return 0;
}

static uint8_t* iris_booleanize_features(
    float* X,
    int num_bits
) {
    uint8_t* X_bool = malloc(IRIS_FEATURES * num_bits * sizeof(uint8_t));
    if(!X_bool) {
        LOGE(TAG, "Failed to allocate memory for booleanized features");
        return NULL;
    }

    int offset = 0;
    for (int i = 0; i < IRIS_FEATURES; i++) {
        iris_booleanize_n_bit(X[i], num_bits, &X_bool[offset]);
        offset += num_bits;
    }
    return X_bool;
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
  for (size_t i = 0; i < IRIS_TEST_SAMPLES; i++) {
    float* input = iris_X_test[i];

    LOGI(TAG, "Evaluating model on test sample %d (label %d)", i, iris_y_test[i]);

    // Booleanize the input using a threshold
    iris_normalize_img(input);
    uint8_t* bool_input = iris_booleanize_features(input, IRIS_MODEL_BITS);
    if(bool_input != NULL) {
      // Evaluate
      tsetlin_evaluate(model, bool_input, votes, predicted_class);
      free(bool_input);
    
      for (size_t i = 0; i < model->n_class; i++) {
        LOGI(TAG, "Class %d: %d votes", i, votes[i]);
      }
      LOGI(TAG, "Predicted class: %d with %d votes", predicted_class, votes[predicted_class]);
      LOGI(TAG, "");
    }
  }

  return 0;
}

void setup() {
  // Initialize Console
  #if defined(CONSOLE_USE_SERIAL)
  // Print to Serial 1
  Serial1.begin(115200);
  while (!Serial1) { ; }
#elif defined(CONSOLE_USE_CDC)
  // Print to USB CDC
  Serial.begin(115200);
#ifdef __AVR__
  fdev_setup_stream(&f_out, sput, nullptr, _FDEV_SETUP_WRITE);  // cf https://www.nongnu.org/avr-libc/user-manual/group__avr__stdio.html#gaf41f158c022cbb6203ccd87d27301226
  stdout = &f_out;
#endif
  while (!Serial) { ; }
#elif defined(CONSOLE_USE_RTT)
  // Print to ST-Link / CMSIS-DAP Debugger (RTT)
  rtt.blockUpBufferFull();
#endif
}

void loop() {
  int ret = tm_iris_main();

  if (ret < 0) {
    LOGE(TAG, "Inference Failed.");
  }

  while (1) {
    delay(500);
  };
}
