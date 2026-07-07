#include "features.h"

#include <math.h>
#include <string.h>

#include <tsetlin.h>

#define EPISODE_MAX_SAMPLES 1024
#define CONTEXT_WINDOW 32
#define EPISODE_FEATURE_COUNT 23
#define EPISODE_FEATURE_BITS 8

typedef struct {
  double pos_transition_magnitude;
  double neg_transition_magnitude;
  double abs_transition;
  double log_abs_transition;
  double duration;
  double log_duration;
  double transition_duration_product;
  double transition_duration_ratio;
  double episode_mean_main;
  double episode_std_main;
  double episode_min_main;
  double episode_max_main;
  double episode_range_main;
  double internal_diff_mean_abs;
  double internal_diff_max_abs;
  double internal_edge_count;
  double subcycle_count_proxy;
  double active_fraction_proxy;
  double episode_energy_estimate;
  double post_minus_pre_mean;
  double event_internal_edge_count;
} EpisodeFeatures;

static const double EPISODE_FEATURE_MEAN[EPISODE_FEATURE_COUNT] = {
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

static const double EPISODE_FEATURE_STD[EPISODE_FEATURE_COUNT] = {
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

typedef struct {
  FeatureReadRangeFn read_range;
  FeatureYieldFn yield_fn;
  void* user_ctx;
} FeatureOps;

static void maybe_yield(const FeatureOps* ops, size_t i) {
  if (ops != NULL && ops->yield_fn != NULL && (i & 0x3F) == 0) {
    ops->yield_fn(ops->user_ctx);
  }
}

static size_t clamp_episode_len(int64_t start, int64_t end) {
  if (end < start) {
    return 0;
  }

  int64_t len = end - start + 1;
  if (len <= 0) {
    return 0;
  }
  if (len > EPISODE_MAX_SAMPLES) {
    return EPISODE_MAX_SAMPLES;
  }
  return (size_t)len;
}

static double mean_nan_safe(const double* arr, size_t len) {
  if (arr == NULL || len == 0) {
    return 0.0;
  }

  double sum = 0.0;
  size_t cnt = 0;
  for (size_t i = 0; i < len; i++) {
    if (!isnan(arr[i])) {
      sum += arr[i];
      cnt++;
    }
  }
  return cnt ? (sum / (double)cnt) : 0.0;
}

static double min_nan_safe(const double* arr, size_t len) {
  if (arr == NULL || len == 0) {
    return 0.0;
  }

  int found = 0;
  double vmin = 0.0;
  for (size_t i = 0; i < len; i++) {
    if (!isnan(arr[i])) {
      if (!found || arr[i] < vmin) {
        vmin = arr[i];
      }
      found = 1;
    }
  }
  return found ? vmin : 0.0;
}

static double max_nan_safe(const double* arr, size_t len) {
  if (arr == NULL || len == 0) {
    return 0.0;
  }

  int found = 0;
  double vmax = 0.0;
  for (size_t i = 0; i < len; i++) {
    if (!isnan(arr[i])) {
      if (!found || arr[i] > vmax) {
        vmax = arr[i];
      }
      found = 1;
    }
  }
  return found ? vmax : 0.0;
}

static double std_nan_safe(const double* arr, size_t len, double mean) {
  if (arr == NULL || len == 0) {
    return 0.0;
  }

  double acc = 0.0;
  size_t cnt = 0;
  for (size_t i = 0; i < len; i++) {
    if (!isnan(arr[i])) {
      double d = arr[i] - mean;
      acc += d * d;
      cnt++;
    }
  }

  return cnt ? sqrt(acc / (double)cnt) : 0.0;
}

static size_t fill_nan_ffill_bfill(const double* in, size_t len, double* out) {
  if (in == NULL || out == NULL || len == 0) {
    return 0;
  }

  for (size_t i = 0; i < len; i++) {
    out[i] = in[i];
  }

  int found_valid = 0;
  size_t first_valid = 0;
  double last_valid = 0.0;

  for (size_t i = 0; i < len; i++) {
    if (!isnan(out[i])) {
      if (!found_valid) {
        first_valid = i;
      }
      found_valid = 1;
      last_valid = out[i];
    } else if (found_valid) {
      out[i] = last_valid;
    }
  }

  if (!found_valid) {
    for (size_t i = 0; i < len; i++) {
      out[i] = 0.0;
    }
    return len;
  }

  for (size_t i = 0; i < first_valid; i++) {
    out[i] = out[first_valid];
  }

  return len;
}

static inline double norm_cdf_double(double x) {
  return 0.5 * (1.0 + erf(x / 1.4142135623730951));
}

static void episode_features_to_vector23(const EpisodeFeatures* f, double* out23) {
  if (f == NULL || out23 == NULL) {
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
    const double* features23,
    double* normalized23,
    double* scaled23,
    uint8_t* bool184) {
  if (features23 == NULL || normalized23 == NULL || scaled23 == NULL || bool184 == NULL) {
    return;
  }

  for (size_t i = 0; i < EPISODE_FEATURE_COUNT; i++) {
    double std = EPISODE_FEATURE_STD[i];
    double z = (std > 0.0) ? ((features23[i] - EPISODE_FEATURE_MEAN[i]) / std) : 0.0;
    double norm01 = norm_cdf_double(z);

    if (norm01 < 0.0) {
      norm01 = 0.0;
    } else if (norm01 > 1.0) {
      norm01 = 1.0;
    }

    double scaled = norm01 * 256.0;
    if (scaled < 0.0) {
      scaled = 0.0;
    }
    if (scaled > 256.0) {
      scaled = 256.0;
    }

    normalized23[i] = norm01;
    scaled23[i] = scaled;

    int quantized = (int)floor(scaled);
    if (quantized > 255) {
      quantized = 255;
    }
    if (quantized < 0) {
      quantized = 0;
    }

    size_t bit_offset = i * EPISODE_FEATURE_BITS;
    for (size_t b = 0; b < EPISODE_FEATURE_BITS; b++) {
      bool184[bit_offset + b] = (uint8_t)((quantized >> (EPISODE_FEATURE_BITS - 1 - b)) & 0x1);
    }
  }
}

static EpisodeFeatures generate_episode_features(
    const double* ep,
    size_t ep_len,
    const double* pre,
    size_t pre_len,
    const double* post,
    size_t post_len,
    double pos_delta,
    double neg_delta,
    double duration,
    const FeatureOps* ops) {
  EpisodeFeatures f;
  memset(&f, 0, sizeof(f));

  double pre_mean = mean_nan_safe(pre, pre_len);
  double post_mean = mean_nan_safe(post, post_len);
  double ep_mean = mean_nan_safe(ep, ep_len);
  double ep_max = max_nan_safe(ep, ep_len);
  double ep_min = min_nan_safe(ep, ep_len);
  double ep_std = std_nan_safe(ep, ep_len, ep_mean);

  double baseline = pre_len ? pre_mean : ep_min;
  double neg_mag = fabs(neg_delta);
  double abs_transition = 0.5 * (pos_delta + neg_mag);
  double ep_range = ep_max - ep_min;

  double diff_sum = 0.0;
  double diff_max = 0.0;
  int internal_edge_count = 0;
  if (ep_len > 1) {
    double edge_threshold = fmax(1.0, 0.25 * abs_transition);
    for (size_t i = 1; i < ep_len; i++) {
      double d = fabs(ep[i] - ep[i - 1]);
      diff_sum += d;
      if (d > diff_max) {
        diff_max = d;
      }
      if (d >= edge_threshold) {
        internal_edge_count++;
      }
      maybe_yield(ops, i);
    }
  }

  double active_fraction = 0.0;
  if (ep_len > 0 && ep_range > 0.0) {
    size_t active_count = 0;
    double threshold = ep_min + 0.25 * ep_range;
    for (size_t i = 0; i < ep_len; i++) {
      if (ep[i] >= threshold) {
        active_count++;
      }
    }
    active_fraction = (double)active_count / (double)ep_len;
  }

  double energy = 0.0;
  for (size_t i = 0; i < ep_len; i++) {
    double p = ep[i] - baseline;
    if (p > 0.0) {
      energy += p;
    }
    maybe_yield(ops, i);
  }

  double filled[EPISODE_MAX_SAMPLES];
  size_t filled_len = fill_nan_ffill_bfill(ep, ep_len, filled);

  double event_internal_edge_count = 0.0;
  if (filled_len > 0) {
    for (size_t i = 1; i < filled_len; i++) {
      double delta = fabs(filled[i] - filled[i - 1]);
      if (delta >= 50.0) {
        event_internal_edge_count += 1.0;
      }
      maybe_yield(ops, i);
    }
  }

  f.pos_transition_magnitude = pos_delta;
  f.neg_transition_magnitude = neg_mag;
  f.abs_transition = abs_transition;
  f.log_abs_transition = log1p(abs_transition);
  f.duration = duration;
  f.log_duration = log1p(duration);
  f.transition_duration_product = abs_transition * fmax(1.0, duration);
  f.transition_duration_ratio = abs_transition / fmax(1.0, duration);
  f.episode_mean_main = ep_mean;
  f.episode_std_main = ep_std;
  f.episode_min_main = ep_min;
  f.episode_max_main = ep_max;
  f.episode_range_main = ep_range;
  f.internal_diff_mean_abs = (ep_len > 1) ? (diff_sum / (double)(ep_len - 1)) : 0.0;
  f.internal_diff_max_abs = diff_max;
  f.internal_edge_count = (double)internal_edge_count;
  f.subcycle_count_proxy = (double)((internal_edge_count > 0) ? (internal_edge_count - 1) : 0);
  f.active_fraction_proxy = active_fraction;
  f.episode_energy_estimate = energy;
  f.post_minus_pre_mean = post_mean - pre_mean;
  f.event_internal_edge_count = event_internal_edge_count;

  return f;
}

void features_extract_and_log_matched_episode_features(
    const StoredEdge* rise,
    const StoredEdge* fall,
    FeatureReadRangeFn read_range,
    FeatureRestorePositionFn restore_position,
    int64_t current_index,
    FeatureYieldFn yield_fn,
    void* user_ctx,
    const char* tag) {
  if (rise == NULL || fall == NULL || read_range == NULL || restore_position == NULL || tag == NULL) {
    return;
  }

  int64_t start = rise->start_time;
  int64_t end = fall->end_time;
  if (start < 0 || end < start) {
    return;
  }

  size_t ep_cap = clamp_episode_len(start, end);
  if (ep_cap == 0) {
    return;
  }

  FeatureOps ops;
  ops.read_range = read_range;
  ops.yield_fn = yield_fn;
  ops.user_ctx = user_ctx;

  double episode[EPISODE_MAX_SAMPLES];
  double pre[CONTEXT_WINDOW];
  double post[CONTEXT_WINDOW];

  size_t ep_len = read_range(start, end, episode, ep_cap, user_ctx);

  size_t pre_len = 0;
  int64_t pre_start = start - CONTEXT_WINDOW;
  int64_t pre_end = start - 1;
  if (pre_end >= 0) {
    if (pre_start < 0) {
      pre_start = 0;
    }
    pre_len = read_range(pre_start, pre_end, pre, CONTEXT_WINDOW, user_ctx);
  }

  int64_t post_start = end + 1;
  int64_t post_end = end + CONTEXT_WINDOW;
  size_t post_len = read_range(post_start, post_end, post, CONTEXT_WINDOW, user_ctx);

  (void)restore_position(current_index, user_ctx);

  if (ep_len == 0) {
    return;
  }

  double duration = (double)(end - start + 1);
  EpisodeFeatures feat = generate_episode_features(
      episode,
      ep_len,
      pre,
      pre_len,
      post,
      post_len,
      rise->delta,
      fall->delta,
      duration,
      &ops);

  double feature_vec23[EPISODE_FEATURE_COUNT];
  double norm_vec23[EPISODE_FEATURE_COUNT];
  double scaled_vec23[EPISODE_FEATURE_COUNT];
  uint8_t bool_vec184[EPISODE_FEATURE_COUNT * EPISODE_FEATURE_BITS];

  episode_features_to_vector23(&feat, feature_vec23);
  normalize_scale_and_booleanize8(feature_vec23, norm_vec23, scaled_vec23, bool_vec184);

  LOGI(
      tag,
      "EP_FEATURE start=%lld end=%lld pos=%.3f neg=%.3f abs=%.3f dur=%.3f mean=%.3f std=%.3f min=%.3f max=%.3f range=%.3f diff_mean=%.3f diff_max=%.3f internal_edges=%.0f active=%.3f energy=%.3f post_pre=%.3f event_edges=%.0f",
      (long long)start,
      (long long)end,
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

  LOGI(
      tag,
      "EP_NORM first3=(%.4f, %.4f, %.4f) scaled_first3=(%.2f, %.2f, %.2f) bool_first8=%u%u%u%u%u%u%u%u",
      norm_vec23[0],
      norm_vec23[1],
      norm_vec23[2],
      scaled_vec23[0],
      scaled_vec23[1],
      scaled_vec23[2],
      (unsigned int)bool_vec184[0],
      (unsigned int)bool_vec184[1],
      (unsigned int)bool_vec184[2],
      (unsigned int)bool_vec184[3],
      (unsigned int)bool_vec184[4],
      (unsigned int)bool_vec184[5],
      (unsigned int)bool_vec184[6],
      (unsigned int)bool_vec184[7]);
}
