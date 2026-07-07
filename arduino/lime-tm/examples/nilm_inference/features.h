#ifndef __NILM_FEATURES_H__
#define __NILM_FEATURES_H__

#include <Arduino.h>
#include <SD.h>

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int64_t start_time;
  int64_t end_time;
  float delta;
} StoredEdge;

void features_extract_and_log_matched_episode_features(
    const StoredEdge* rise,
    const StoredEdge* fall,
    File f_data);

const char* features_get_latest_predicted_label(void);
void features_reset_prediction_stats(void);
size_t features_get_class_count(void);
const char* features_get_class_label(size_t class_index);
size_t features_get_class_event_count(size_t class_index);

#endif // __NILM_FEATURES_H__
