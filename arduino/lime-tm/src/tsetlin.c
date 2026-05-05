#include "tsetlin.h"

#if defined(__ZEPHYR__)
    /* Zephyr RTOS */
    #include <zephyr/fs/fs.h>
    LOG_MODULE_REGISTER(tsetlin);
#endif

static const char* TAG = "tsetlin";

int tsetlin_evaluate(Tsetlin* model, uint8_t* input, int32_t *out_votes, uint8_t* out_class) {
    memset(out_votes, 0, model->n_class * sizeof(int32_t));

    for (size_t c = 0; c < model->n_class; c++)
    {
        for (uint32_t j = 0; j <(size_t) model->n_clause / 2; j++)
        {
            ClauseCompressed* p_clause = &model->clauses_compressed[c * model->n_clause + j * 2];
            ClauseCompressed* n_clause = &model->clauses_compressed[c * model->n_clause + j * 2 + 1];

            out_votes[c] += clause_evaluate(p_clause, input, model->n_state, model->n_feature, model->model_type);
            out_votes[c] -= clause_evaluate(n_clause, input, model->n_state, model->n_feature, model->model_type);
        }
    }

    // Find class with maximum votes
    uint8_t max_class = 0;
    int32_t max_votes = out_votes[0];
    for (size_t c = 1; c < model->n_class; c++)
    {
        if (out_votes[c] > max_votes)
        {
            max_votes = out_votes[c];
            max_class = c;
        }
    }

    *out_class = max_class;

    return 0;
}
