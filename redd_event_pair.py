import pandas as pd
from loguru import logger

import numpy as np
from scipy.signal import medfilt
from sklearn.cluster import KMeans

building_list = [1, 2, 3, 4, 5, 6]
appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]

def best_subset_dp(active, target, max_subset_size=6):
    """
    Approximate subset selection using DP-like pruning.
    Avoids full combinatorial explosion.
    """

    # dp: list of (sum, indices, error)
    dp = [(0.0, [], float("inf"))]

    for i, r in enumerate(active):

        new_dp = dp.copy()
        val = r["transition"]

        for s, idxs, _ in dp:
            new_sum = s + val
            new_idxs = idxs + [i]
            new_err = abs(new_sum - target)

            new_dp.append((new_sum, new_idxs, new_err))

        # prune: keep only best candidates
        new_dp.sort(key=lambda x: x[2])
        dp = new_dp[:200]  # beam width (important for speed)

    # filter valid subset size
    dp = [x for x in dp if len(x[1]) <= max_subset_size]

    if not dp:
        return [], float("inf")

    best = min(dp, key=lambda x: x[2])

    return best[1], best[2]

def match_edges_stateful(
    df,
    appliance_col,
    power_weight=1.0,
    time_weight=0.01,
    max_duration=None,
    max_time_gap=500,
):
    """
    Stateful NILM edge matching:
    - many rises → one fall
    - missing edges allowed
    - noisy power tolerated
    """

    events = df[df[appliance_col] == 1].copy()

    if events.empty:
        logger.warning(f"No events found for appliance column: {appliance_col}")
        return [], [], []

    events = events.sort_values("start")

    active_rises = []
    pairs = []

    for idx, row in events.iterrows():

        power = row["transition"]
        t = row["start"]

        # -------------------------
        # RISING EDGE
        # -------------------------
        if power > 0:
            active_rises.append({
                "idx": idx,
                "transition": power,
                "time": t
            })
            continue

        # -------------------------
        # FALLING EDGE
        # -------------------------
        target = abs(power)

        # remove stale rises (missing fall handling)
        active_rises = [
            r for r in active_rises
            if (t - r["time"]) <= max_time_gap
        ]

        if len(active_rises) == 0:
            continue

        subset_idx, error = best_subset_dp(active_rises, target)

        if not subset_idx:
            continue

        used = [active_rises[i] for i in subset_idx]

        rise_sum = sum(u["transition"] for u in used)

        # duration check (optional)
        duration = t - min(u["time"] for u in used)

        if max_duration is not None and duration > max_duration:
            continue

        pairs.append({
            "rise_indices": [u["idx"] for u in used],
            "fall_idx": idx,
            "rise_time": min(u["time"] for u in used),
            "fall_time": t,
            "duration": duration,
            "rise_sum": rise_sum,
            "fall_power": power,
            "power_error": abs(rise_sum + power),
        })

        # remove used rises
        for i in sorted(subset_idx, reverse=True):
            active_rises.pop(i)

    unmatched_rises = [r["idx"] for r in active_rises]
    unmatched_falls = []  # unknown falls are implicitly handled (no forced matching)

    # Print avearage power error for matched pairs
    if pairs:
        avg_power_error = np.mean([p["power_error"] for p in pairs])
        logger.info(f"Average power error for {appliance_col}: {avg_power_error:.2f}")

    return pairs, unmatched_rises, unmatched_falls


import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

def safe_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end < start:
        return np.asarray([], dtype=np.float64)
    return values[start : end + 1]

def episode_feature_row(values: np.ndarray, episode: Mapping[str, Any]) -> Dict[str, float]:
    start = int(episode["start"])
    end = int(episode["end"])
    ep = safe_slice(values, start, end)
    pre = safe_slice(values, start - 32, start - 1)
    post = safe_slice(values, end + 1, end + 32)
    pre_mean = float(np.nanmean(pre)) if len(pre) else 0.0
    post_mean = float(np.nanmean(post)) if len(post) else 0.0
    ep_mean = float(np.nanmean(ep)) if len(ep) else 0.0
    ep_max = float(np.nanmax(ep)) if len(ep) else 0.0
    ep_min = float(np.nanmin(ep)) if len(ep) else 0.0
    ep_std = float(np.nanstd(ep)) if len(ep) else 0.0
    baseline = pre_mean if len(pre) else ep_min
    filled = pd.Series(ep).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64) if len(ep) else np.asarray([], dtype=np.float64)
    delta = np.diff(filled, prepend=filled[0]) if len(filled) else np.asarray([], dtype=np.float64)
    duration = float(episode["duration"])
    pos_mag = float(episode["pos_delta"])
    neg_mag = float(abs(episode["neg_delta"]))
    abs_transition = 0.5 * (pos_mag + neg_mag)
    ep_range = ep_max - ep_min
    diffs = np.abs(np.diff(ep)) if len(ep) > 1 else np.asarray([], dtype=np.float64)
    edge_threshold = max(1.0, 0.25 * abs_transition)
    internal_edge_count = int(np.sum(diffs >= edge_threshold)) if len(diffs) else 0
    active_fraction = float(np.mean(ep >= (ep_min + 0.25 * ep_range))) if len(ep) and ep_range > 0 else 0.0
    return {
        "pos_transition_magnitude": pos_mag,
        "neg_transition_magnitude": neg_mag,
        "abs_transition": float(abs_transition),
        "log_abs_transition": float(math.log1p(abs_transition)),
        "duration": duration,
        "log_duration": float(math.log1p(duration)),
        "transition_duration_product": float(abs_transition * max(1.0, duration)),
        "transition_duration_ratio": float(abs_transition / max(1.0, duration)),
        "episode_mean_main": ep_mean,
        "episode_std_main": ep_std,
        "episode_min_main": ep_min,
        "episode_max_main": ep_max,
        "episode_range_main": ep_range,
        "internal_diff_mean_abs": float(np.mean(diffs)) if len(diffs) else 0.0,
        "internal_diff_max_abs": float(np.max(diffs)) if len(diffs) else 0.0,
        "internal_edge_count": internal_edge_count,
        "subcycle_count_proxy": int(max(0, internal_edge_count - 1)),
        "active_fraction_proxy": active_fraction,
        "episode_energy_estimate": float(np.nansum(np.maximum(ep - baseline, 0.0))) if len(ep) else 0.0,
        "post_minus_pre_mean": post_mean - pre_mean,
        "event_internal_edge_count": float(np.count_nonzero(np.abs(delta) >= 50.0)) if len(delta) else 0.0,
    }

if __name__ == "__main__":
    for i in building_list:
        logger.info(f"========== Processing building {i} ==========")

        building_raw = pd.read_csv(f"building_{i}_raw.csv")
        df = pd.read_csv(f"building_{i}_main_transients_train.csv", index_col=0)

        for appliance in appliance_names:
            logger.info(f"Processing appliance {appliance}")

            pairs, unmatched_on, unmatched_off = match_edges_stateful(
                df,
                appliance_col=f"{appliance}_label",
                power_weight=1.0,
                time_weight=0.05,
                max_duration=2000,
                max_time_gap=500,
            )

            if pairs:
                logger.info(f"\tMatched: {len(pairs)}")
                logger.info(f"\tUnmatched rises: {len(unmatched_on)}")
                logger.info(f"\tUnmatched falls: {len(unmatched_off)}")

                matched_events = []

                for pair in pairs:
                    # Create a new row for the matched event
                    matched_row = {
                        "appliance": appliance,
                        "transition": pair["rise_sum"],
                        "duration": pair["duration"],
                        "start": pair["rise_time"],
                        "end": pair["fall_time"]
                    }

                    episode = episode_feature_row(
                        building_raw["main"].to_numpy(dtype=np.float64), 
                        {"start": pair["rise_time"], 
                         "end": pair["fall_time"], 
                         "duration": pair["duration"], 
                         "pos_delta": pair["rise_sum"], 
                         "neg_delta": pair["fall_power"]
                         }
                    )

                    # Append the matched row to a new DataFrame
                    matched_events.append(episode | matched_row)

                pairs_df = pd.DataFrame(matched_events)
                pairs_df.to_csv(f"temp/building_{i}_{appliance}_matched_transitions.csv", index=False)
