#ifndef __NILM_FEATURES_H__
#define __NILM_FEATURES_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int64_t start_time;
  int64_t end_time;
  double delta;
} StoredEdge;

typedef size_t (*FeatureReadRangeFn)(
    int64_t start,
    int64_t end,
    double* out,
    size_t out_cap,
    void* user_ctx);

typedef int (*FeatureRestorePositionFn)(int64_t current_index, void* user_ctx);

typedef void (*FeatureYieldFn)(void* user_ctx);

void features_extract_and_log_matched_episode_features(
    const StoredEdge* rise,
    const StoredEdge* fall,
    FeatureReadRangeFn read_range,
    FeatureRestorePositionFn restore_position,
    int64_t current_index,
    FeatureYieldFn yield_fn,
    void* user_ctx,
    const char* tag);

#ifdef __cplusplus
}
#endif

#endif // __NILM_FEATURES_H__
