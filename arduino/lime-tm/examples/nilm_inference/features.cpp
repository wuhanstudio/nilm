#include "features.h"

#include <math.h>
#include <string.h>

#include <tsetlin.h>

#define TAG "features"

#define EPISODE_MAX_SAMPLES 1024
#define CONTEXT_WINDOW 32
#define EPISODE_FEATURE_COUNT 23
#define EPISODE_FEATURE_BITS 8
#define TM_MAX_CLASSES 16

// Implemented in nilm_inference.ino to avoid duplicating model storage in this translation unit.
extern Tsetlin *nilm_get_tm_model(void);

static const char *REDD_CLASS_NAMES[] = {
    "fridge",
    "microwave",
    "dish washer",
    "electric furnace",
};

#define REDD_CLASS_COUNT (sizeof(REDD_CLASS_NAMES) / sizeof(REDD_CLASS_NAMES[0]))

static const char *latest_predicted_label = "--";
static size_t matched_event_class_counts[REDD_CLASS_COUNT] = {0};

typedef struct
{
    float pos_transition_magnitude;
    float neg_transition_magnitude;
    float abs_transition;
    float log_abs_transition;
    float duration;
    float log_duration;
    float transition_duration_product;
    float transition_duration_ratio;
    float episode_mean_main;
    float episode_std_main;
    float episode_min_main;
    float episode_max_main;
    float episode_range_main;
    float internal_diff_mean_abs;
    float internal_diff_max_abs;
    float internal_edge_count;
    float subcycle_count_proxy;
    float active_fraction_proxy;
    float episode_energy_estimate;
    float post_minus_pre_mean;
    float event_internal_edge_count;
} EpisodeFeatures;

static const float EPISODE_FEATURE_MEAN[EPISODE_FEATURE_COUNT] = {
    5.53064420e+02,
    2.34699651e+02,
    5.53064420e+02,
    5.14110892e+02,
    5.33587656e+02,
    5.85662218e+00,
    2.34699651e+02,
    5.00175457e+00,
    8.00592420e+04,
    1.70234171e+01,
    8.71226609e+02,
    2.09001330e+02,
    3.25109966e+02,
    1.36429093e+03,
    1.03918096e+03,
    3.38986639e+01,
    8.55259768e+02,
    4.25378347e+00,
    3.25611176e+00,
    7.20509590e-01,
    7.64846182e+04,
    4.09780040e+01,
    4.96973225e+00,
};

static const float EPISODE_FEATURE_STD[EPISODE_FEATURE_COUNT] = {
    6.10272619e+02,
    1.52976379e+02,
    6.10272619e+02,
    6.22051295e+02,
    5.97094304e+02,
    8.35577268e-01,
    1.52976379e+02,
    1.20954040e+00,
    9.65427333e+04,
    4.26523956e+01,
    9.83710637e+02,
    3.20441496e+02,
    5.89947086e+02,
    1.29423519e+03,
    1.06204987e+03,
    6.72437546e+01,
    8.51834332e+02,
    8.32535961e+00,
    8.32430942e+00,
    3.73134104e-01,
    1.03515350e+05,
    5.67766889e+02,
    9.40915841e+00,
};

static size_t clamp_episode_len(int64_t start, int64_t end)
{
    if (end < start)
    {
        return 0;
    }

    int64_t len = end - start + 1;
    if (len <= 0)
    {
        return 0;
    }
    if (len > EPISODE_MAX_SAMPLES)
    {
        return EPISODE_MAX_SAMPLES;
    }
    return (size_t)len;
}

static size_t read_float_range_from_sd(File f_data, int64_t start, int64_t end, float *out, size_t out_cap)
{
    if (!f_data || out == NULL || out_cap == 0 || end < start || start < 0)
    {
        return 0;
    }

    size_t requested = (size_t)(end - start + 1);
    if (requested > out_cap)
    {
        requested = out_cap;
    }

    uint32_t byte_pos = (uint32_t)start * sizeof(float);
    if (!f_data.seek(byte_pos))
    {
        return 0;
    }

    for (size_t i = 0; i < requested; i++)
    {
        float v;
        if (f_data.read((uint8_t *)&v, sizeof(float)) != sizeof(float))
        {
            return i;
        }
        out[i] = v;
    }

    return requested;
}

static float mean_nan_safe(const float *arr, size_t len)
{
    if (arr == NULL || len == 0)
    {
        return 0.0;
    }

    float sum = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < len; i++)
    {
        if (!isnan(arr[i]))
        {
            sum += arr[i];
            cnt++;
        }
    }
    return cnt ? (sum / (float)cnt) : 0.0;
}

static float min_nan_safe(const float *arr, size_t len)
{
    if (arr == NULL || len == 0)
    {
        return 0.0;
    }

    int found = 0;
    float vmin = 0.0;
    for (size_t i = 0; i < len; i++)
    {
        if (!isnan(arr[i]))
        {
            if (!found || arr[i] < vmin)
            {
                vmin = arr[i];
            }
            found = 1;
        }
    }
    return found ? vmin : 0.0;
}

static float max_nan_safe(const float *arr, size_t len)
{
    if (arr == NULL || len == 0)
    {
        return 0.0;
    }

    int found = 0;
    float vmax = 0.0;
    for (size_t i = 0; i < len; i++)
    {
        if (!isnan(arr[i]))
        {
            if (!found || arr[i] > vmax)
            {
                vmax = arr[i];
            }
            found = 1;
        }
    }
    return found ? vmax : 0.0;
}

static float std_nan_safe(const float *arr, size_t len, float mean)
{
    if (arr == NULL || len == 0)
    {
        return 0.0;
    }

    float acc = 0.0;
    size_t cnt = 0;
    for (size_t i = 0; i < len; i++)
    {
        if (!isnan(arr[i]))
        {
            float d = arr[i] - mean;
            acc += d * d;
            cnt++;
        }
    }

    return cnt ? sqrtf(acc / (float)cnt) : 0.0f;
}

static size_t fill_nan_ffill_bfill(const float *in, size_t len, float *out)
{
    if (in == NULL || out == NULL || len == 0)
    {
        return 0;
    }

    for (size_t i = 0; i < len; i++)
    {
        out[i] = in[i];
    }

    int found_valid = 0;
    size_t first_valid = 0;
    float last_valid = 0.0;

    for (size_t i = 0; i < len; i++)
    {
        if (!isnan(out[i]))
        {
            if (!found_valid)
            {
                first_valid = i;
            }
            found_valid = 1;
            last_valid = out[i];
        }
        else if (found_valid)
        {
            out[i] = last_valid;
        }
    }

    if (!found_valid)
    {
        for (size_t i = 0; i < len; i++)
        {
            out[i] = 0.0;
        }
        return len;
    }

    for (size_t i = 0; i < first_valid; i++)
    {
        out[i] = out[first_valid];
    }

    return len;
}

static inline float norm_cdf_double(float x)
{
    return 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

static void episode_features_to_vector23(const EpisodeFeatures *f, float *out23)
{
    if (f == NULL || out23 == NULL)
    {
        return;
    }

    out23[0] = f->pos_transition_magnitude;
    out23[1] = f->duration;
    out23[2] = f->pos_transition_magnitude;
    out23[3] = f->neg_transition_magnitude;
    out23[4] = f->abs_transition;
    out23[5] = f->log_abs_transition;
    out23[6] = f->duration;
    out23[7] = f->log_duration;
    out23[8] = f->transition_duration_product;
    out23[9] = f->transition_duration_ratio;
    out23[10] = f->episode_mean_main;
    out23[11] = f->episode_std_main;
    out23[12] = f->episode_min_main;
    out23[13] = f->episode_max_main;
    out23[14] = f->episode_range_main;
    out23[15] = f->internal_diff_mean_abs;
    out23[16] = f->internal_diff_max_abs;
    out23[17] = f->internal_edge_count;
    out23[18] = f->subcycle_count_proxy;
    out23[19] = f->active_fraction_proxy;
    out23[20] = f->episode_energy_estimate;
    out23[21] = f->post_minus_pre_mean;
    out23[22] = f->event_internal_edge_count;
}

static void normalize_scale_and_booleanize8(
    const float *features23,
    float *normalized23,
    float *scaled23,
    uint8_t *bool184)
{
    if (features23 == NULL || normalized23 == NULL || scaled23 == NULL || bool184 == NULL)
    {
        return;
    }

    for (size_t i = 0; i < EPISODE_FEATURE_COUNT; i++)
    {
        float std = EPISODE_FEATURE_STD[i];
        float z = (std > 0.0) ? ((features23[i] - EPISODE_FEATURE_MEAN[i]) / std) : 0.0;
        float norm01 = norm_cdf_double(z);

        if (norm01 < 0.0)
        {
            norm01 = 0.0;
        }
        else if (norm01 > 1.0)
        {
            norm01 = 1.0;
        }

        float scaled = norm01 * 256.0;
        if (scaled < 0.0)
        {
            scaled = 0.0;
        }
        if (scaled > 256.0)
        {
            scaled = 256.0;
        }

        normalized23[i] = norm01;
        scaled23[i] = scaled;

        int quantized = (int)floorf(scaled);
        if (quantized > 255)
        {
            quantized = 255;
        }
        if (quantized < 0)
        {
            quantized = 0;
        }

        size_t bit_offset = i * EPISODE_FEATURE_BITS;
        for (size_t b = 0; b < EPISODE_FEATURE_BITS; b++)
        {
            bool184[bit_offset + b] = (uint8_t)((quantized >> (EPISODE_FEATURE_BITS - 1 - b)) & 0x1);
        }
    }
}

static EpisodeFeatures generate_episode_features(
    const float *ep,
    size_t ep_len,
    const float *pre,
    size_t pre_len,
    const float *post,
    size_t post_len,
    float pos_delta,
    float neg_delta,
    float duration)
{
    EpisodeFeatures f;
    memset(&f, 0, sizeof(f));

    float pre_mean = mean_nan_safe(pre, pre_len);
    float post_mean = mean_nan_safe(post, post_len);
    float ep_mean = mean_nan_safe(ep, ep_len);
    float ep_max = max_nan_safe(ep, ep_len);
    float ep_min = min_nan_safe(ep, ep_len);
    float ep_std = std_nan_safe(ep, ep_len, ep_mean);

    float baseline = pre_len ? pre_mean : ep_min;
    float neg_mag = fabsf(neg_delta);
    float abs_transition = 0.5 * (pos_delta + neg_mag);
    float ep_range = ep_max - ep_min;

    float diff_sum = 0.0;
    float diff_max = 0.0;
    int internal_edge_count = 0;
    if (ep_len > 1)
    {
        float edge_threshold = fmaxf(1.0f, 0.25f * abs_transition);
        for (size_t i = 1; i < ep_len; i++)
        {
            float d = fabsf(ep[i] - ep[i - 1]);
            diff_sum += d;
            if (d > diff_max)
            {
                diff_max = d;
            }
            if (d >= edge_threshold)
            {
                internal_edge_count++;
            }
        }
    }

    float active_fraction = 0.0;
    if (ep_len > 0 && ep_range > 0.0)
    {
        size_t active_count = 0;
        float threshold = ep_min + 0.25 * ep_range;
        for (size_t i = 0; i < ep_len; i++)
        {
            if (ep[i] >= threshold)
            {
                active_count++;
            }
        }
        active_fraction = (float)active_count / (float)ep_len;
    }

    float energy = 0.0;
    for (size_t i = 0; i < ep_len; i++)
    {
        float p = ep[i] - baseline;
        if (p > 0.0)
        {
            energy += p;
        }
    }

    float filled[EPISODE_MAX_SAMPLES];
    size_t filled_len = fill_nan_ffill_bfill(ep, ep_len, filled);

    float event_internal_edge_count = 0.0;
    if (filled_len > 0)
    {
        for (size_t i = 1; i < filled_len; i++)
        {
            float delta = fabsf(filled[i] - filled[i - 1]);
            if (delta >= 50.0)
            {
                event_internal_edge_count += 1.0;
            }
        }
    }

    f.pos_transition_magnitude = pos_delta;
    f.neg_transition_magnitude = neg_mag;
    f.abs_transition = abs_transition;
    f.log_abs_transition = log1pf(abs_transition);
    f.duration = duration;
    f.log_duration = log1pf(duration);
    f.transition_duration_product = abs_transition * fmaxf(1.0f, duration);
    f.transition_duration_ratio = abs_transition / fmaxf(1.0f, duration);
    f.episode_mean_main = ep_mean;
    f.episode_std_main = ep_std;
    f.episode_min_main = ep_min;
    f.episode_max_main = ep_max;
    f.episode_range_main = ep_range;
    f.internal_diff_mean_abs = (ep_len > 1) ? (diff_sum / (float)(ep_len - 1)) : 0.0;
    f.internal_diff_max_abs = diff_max;
    f.internal_edge_count = (float)internal_edge_count;
    f.subcycle_count_proxy = (float)((internal_edge_count > 0) ? (internal_edge_count - 1) : 0);
    f.active_fraction_proxy = active_fraction;
    f.episode_energy_estimate = energy;
    f.post_minus_pre_mean = post_mean - pre_mean;
    f.event_internal_edge_count = event_internal_edge_count;

    return f;
}

void features_extract_and_log_matched_episode_features(
    const StoredEdge *rise,
    const StoredEdge *fall,
    File f_data)
{
    if (rise == NULL || fall == NULL || !f_data)
    {
        LOGE(TAG, "Invalid arguments to features_extract_and_log_matched_episode_features");
        return;
    }

    LOGI(TAG, "Extracting and logging features for matched episode: rise=%lld-%lld fall=%lld-%lld",
        (long long)rise->start_time, (long long)rise->end_time,
        (long long)fall->start_time, (long long)fall->end_time);

    // Episode is the full mains segment from rising edge to falling edge.
    int64_t episode_start = rise->start_time;
    int64_t episode_end = fall->end_time;
    if (episode_start < 0 || episode_end < episode_start)
    {
        LOGE(TAG, "Invalid episode range: start=%lld end=%lld", (long long)episode_start, (long long)episode_end);
        return;
    }

    size_t ep_cap = clamp_episode_len(episode_start, episode_end);
    if (ep_cap == 0)
    {
        return;
    }

    float episode[EPISODE_MAX_SAMPLES];
    float pre[CONTEXT_WINDOW];
    float post[CONTEXT_WINDOW];

    size_t ep_len = read_float_range_from_sd(f_data, episode_start, episode_end, episode, ep_cap);
    if (ep_len != ep_cap)
    {
        LOGW(TAG, "Partial episode read from SD: got=%u expected=%u", (unsigned int)ep_len, (unsigned int)ep_cap);
    }

    size_t pre_len = 0;
    int64_t pre_start = episode_start - CONTEXT_WINDOW;
    int64_t pre_end = episode_start - 1;
    if (pre_end >= 0)
    {
        if (pre_start < 0)
        {
            pre_start = 0;
        }
        pre_len = read_float_range_from_sd(f_data, pre_start, pre_end, pre, CONTEXT_WINDOW);
    }

    int64_t post_start = episode_end + 1;
    int64_t post_end = episode_end + CONTEXT_WINDOW;
    size_t post_len = read_float_range_from_sd(f_data, post_start, post_end, post, CONTEXT_WINDOW);

    if (ep_len == 0)
    {
        return;
    }

    float duration = (float)(episode_end - episode_start + 1);
    EpisodeFeatures feat = generate_episode_features(
        episode,
        ep_len,
        pre,
        pre_len,
        post,
        post_len,
        rise->delta,
        fall->delta,
        duration);

    float feature_vec23[EPISODE_FEATURE_COUNT];
    float norm_vec23[EPISODE_FEATURE_COUNT];
    float scaled_vec23[EPISODE_FEATURE_COUNT];
    uint8_t bool_vec184[EPISODE_FEATURE_COUNT * EPISODE_FEATURE_BITS];

    episode_features_to_vector23(&feat, feature_vec23);
    normalize_scale_and_booleanize8(feature_vec23, norm_vec23, scaled_vec23, bool_vec184);

    latest_predicted_label = "--";

    LOGI(
    TAG,
    "EP_FEATURE start=%lld end=%lld pos=%.3f neg=%.3f abs=%.3f dur=%.3f mean=%.3f std=%.3f min=%.3f max=%.3f range=%.3f diff_mean=%.3f diff_max=%.3f internal_edges=%.0f active=%.3f energy=%.3f post_pre=%.3f event_edges=%.0f",
    (long long)episode_start,
    (long long)episode_end,
    feat.pos_transition_magnitude,
    feat.neg_transition_magnitude,
    feat.abs_transition,
    feat.duration,
    feat.episode_mean_main,
    feat.episode_std_main,
    feat.episode_min_main,
    feat.episode_max_main,
    feat.episode_range_main,
    feat.internal_diff_mean_abs,
    feat.internal_diff_max_abs,
    feat.internal_edge_count,
    feat.active_fraction_proxy,
    feat.episode_energy_estimate,
    feat.post_minus_pre_mean,
    feat.event_internal_edge_count);

    Tsetlin *model = nilm_get_tm_model();
    if (model == NULL)
    {
        LOGW(TAG, "TM model is not available");
    }
    else if (model->n_feature != (EPISODE_FEATURE_COUNT * EPISODE_FEATURE_BITS))
    {
        LOGW(
            TAG,
            "TM feature mismatch model=%u expected=%u",
            (unsigned int)model->n_feature,
            (unsigned int)(EPISODE_FEATURE_COUNT * EPISODE_FEATURE_BITS));
    }
    else if (model->n_class > TM_MAX_CLASSES)
    {
        LOGW(
            TAG,
            "TM class count %u exceeds TM_MAX_CLASSES=%u",
            (unsigned int)model->n_class,
            (unsigned int)TM_MAX_CLASSES);
    }
    else
    {
        uint8_t predicted_class = 0;
        int32_t votes[TM_MAX_CLASSES] = {0};
        if (tsetlin_evaluate(model, bool_vec184, votes, &predicted_class) == 0)
        {
            if (predicted_class < REDD_CLASS_COUNT)
            {
                latest_predicted_label = REDD_CLASS_NAMES[predicted_class];
                matched_event_class_counts[predicted_class]++;
            }
            else
            {
                latest_predicted_label = "unknown";
            }
            LOGI(
                TAG,
                "TM_INFER class=%u label=%s vote=%d",
                (unsigned int)predicted_class,
                latest_predicted_label,
                votes[predicted_class]);
        }
        else
        {
            LOGW(TAG, "TM inference failed");
        }
    }
}

const char *features_get_latest_predicted_label(void)
{
    return latest_predicted_label;
}

void features_reset_prediction_stats(void)
{
    latest_predicted_label = "--";
    memset(matched_event_class_counts, 0, sizeof(matched_event_class_counts));
}

size_t features_get_class_count(void)
{
    return REDD_CLASS_COUNT;
}

const char *features_get_class_label(size_t class_index)
{
    if (class_index >= REDD_CLASS_COUNT)
    {
        return "unknown";
    }
    return REDD_CLASS_NAMES[class_index];
}

size_t features_get_class_event_count(size_t class_index)
{
    if (class_index >= REDD_CLASS_COUNT)
    {
        return 0;
    }
    return matched_event_class_counts[class_index];
}
