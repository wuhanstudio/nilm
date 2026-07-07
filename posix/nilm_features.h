#ifndef __NILM_FEATURES_H__
#define __NILM_FEATURES_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "detector.h"

#ifdef __cplusplus
extern "C" {
#endif

void features_record_transition(const EdgeDetectorOutput *output);
void features_match_pending_edges(FILE *input_file, int64_t current_index);
void features_finalize(FILE *input_file, int64_t current_index);

size_t features_get_rising_count(void);
size_t features_get_falling_count(void);

#ifdef __cplusplus
}
#endif

#endif // __NILM_FEATURES_H__
