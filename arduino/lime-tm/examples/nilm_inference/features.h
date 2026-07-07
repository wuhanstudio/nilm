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

#endif // __NILM_FEATURES_H__
