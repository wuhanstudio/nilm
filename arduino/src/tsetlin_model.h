#ifndef __TSETLIN_MODEL_H__
#define __TSETLIN_MODEL_H__

#include <stdint.h>

typedef enum {
    MODEL_TYPE__INFERENCE = 0,
    MODEL_TYPE__TRAINING = 1,

    MODEL_TYPE__COMPRESSED = 2,
} ModelType;

typedef struct {
    uint32_t *data;
} Clause;

typedef struct {
    const uint16_t n_pos_literal;
    const uint16_t n_neg_literal;
    const uint16_t *position;
    const uint16_t *data;
} ClauseCompressed;

typedef struct {
    const uint32_t n_class;
    const uint32_t n_feature;
    const uint32_t n_clause;
    const uint32_t n_state;

    const ModelType model_type;

    const Clause *clauses;
    const ClauseCompressed *clauses_compressed;
} Tsetlin;

#endif // __TSETLIN_MODEL_H__
