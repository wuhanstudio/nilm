import argparse
import pandas as pd
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import medfilt
from sklearn.cluster import KMeans

building_list = [1, 2, 3, 4, 5, 6]
appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["fridge", "microwave", "dish washer", "electric furnace", "unknown"]

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


def plot_matched_edges(
    building_raw: pd.DataFrame,
    pairs: Sequence[Mapping[str, Any]],
    appliance: str,
    window_size: int = 2000,
) -> None:
    if "main" not in building_raw.columns:
        logger.warning("Column 'main' is missing in building_raw. Skipping plot.")
        return

    if building_raw.empty:
        logger.warning("building_raw is empty. Skipping plot.")
        return

    values = pd.to_numeric(building_raw["main"], errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0).to_numpy(dtype=np.float64)
    x = building_raw.index.to_numpy()

    window_size = min(max(1, int(window_size)), len(building_raw))
    max_start = max(0, len(building_raw) - window_size)

    rises = np.asarray([int(p["rise_time"]) for p in pairs], dtype=np.int64) if pairs else np.asarray([], dtype=np.int64)
    falls = np.asarray([int(p["fall_time"]) for p in pairs], dtype=np.int64) if pairs else np.asarray([], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)

    line_main, = ax.plot(x[:window_size], values[:window_size], color="black", linewidth=1.0, label="main")

    rise_vlines = []
    fall_vlines = []

    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x[0], x[window_size - 1])
    ax.set_title(f"Matched edges on main signal - {appliance}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.grid(alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="main"),
        plt.Line2D([0], [0], color="tab:green", linestyle="--", label="rise"),
        plt.Line2D([0], [0], color="tab:red", linestyle="--", label="fall"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    slider_ax = fig.add_axes([0.15, 0.07, 0.72, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="Scroll",
        valmin=0,
        valmax=max_start,
        valinit=0,
        valstep=1,
    )

    def redraw_edge_lines(left_idx: int, right_idx: int) -> None:
        nonlocal rise_vlines, fall_vlines

        for ln in rise_vlines:
            ln.remove()
        for ln in fall_vlines:
            ln.remove()
        rise_vlines = []
        fall_vlines = []

        if len(rises) > 0:
            visible_rises = rises[(rises >= left_idx) & (rises <= right_idx)]
            for rx in visible_rises:
                rise_vlines.append(ax.axvline(x=rx, color="tab:green", linestyle="--", alpha=0.6))

        if len(falls) > 0:
            visible_falls = falls[(falls >= left_idx) & (falls <= right_idx)]
            for fx in visible_falls:
                fall_vlines.append(ax.axvline(x=fx, color="tab:red", linestyle="--", alpha=0.6))

    redraw_edge_lines(0, window_size - 1)

    def update(val: float) -> None:
        start = int(val)
        end = start + window_size

        x_window = x[start:end]
        y_window = values[start:end]
        line_main.set_data(x_window, y_window)
        ax.set_xlim(x_window[0], x_window[-1])

        redraw_edge_lines(start, end - 1)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_all_matched_edges(
    building_raw: pd.DataFrame,
    appliance_pairs: Mapping[str, Sequence[Mapping[str, Any]]],
    window_size: int = 2000,
) -> None:
    if "main" not in building_raw.columns:
        logger.warning("Column 'main' is missing in building_raw. Skipping plot.")
        return

    if building_raw.empty:
        logger.warning("building_raw is empty. Skipping plot.")
        return

    values = (
        pd.to_numeric(building_raw["main"], errors="coerce")
        .ffill()
        .bfill()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    x = building_raw.index.to_numpy()

    window_size = min(max(1, int(window_size)), len(building_raw))
    max_start = max(0, len(building_raw) - window_size)

    non_empty_apps = [name for name, pairs in appliance_pairs.items() if pairs]
    if not non_empty_apps:
        logger.warning("No matched pairs found for any appliance. Skipping plot.")
        return

    cmap = plt.get_cmap("tab10")
    appliance_color = {app: cmap(i % 10) for i, app in enumerate(non_empty_apps)}

    rise_dict = {
        app: np.asarray([int(p["rise_time"]) for p in pairs], dtype=np.int64)
        for app, pairs in appliance_pairs.items()
        if pairs
    }
    fall_dict = {
        app: np.asarray([int(p["fall_time"]) for p in pairs], dtype=np.int64)
        for app, pairs in appliance_pairs.items()
        if pairs
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.22)

    line_main, = ax.plot(x[:window_size], values[:window_size], color="black", linewidth=1.0, label="main")

    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x[0], x[window_size - 1])
    ax.set_title("Matched rising/falling edges on main signal (all appliances)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.grid(alpha=0.25)

    edge_lines = []

    def redraw_edge_lines(left_idx: int, right_idx: int) -> None:
        nonlocal edge_lines
        for ln in edge_lines:
            ln.remove()
        edge_lines = []

        for app in non_empty_apps:
            color = appliance_color[app]
            rises = rise_dict[app]
            falls = fall_dict[app]

            visible_rises = rises[(rises >= left_idx) & (rises <= right_idx)]
            visible_falls = falls[(falls >= left_idx) & (falls <= right_idx)]

            for rx in visible_rises:
                edge_lines.append(ax.axvline(x=rx, color=color, linestyle="--", alpha=0.75))
            for fx in visible_falls:
                edge_lines.append(ax.axvline(x=fx, color=color, linestyle="-", alpha=0.35))

    redraw_edge_lines(0, window_size - 1)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="main"),
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color=appliance_color[app], linewidth=2.0, label=app)
            for app in non_empty_apps
        ]
    )
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color="gray", linestyle="--", label="rise"),
            plt.Line2D([0], [0], color="gray", linestyle="-", label="fall"),
        ]
    )
    ax.legend(handles=legend_handles, loc="upper right", ncol=2)

    slider_ax = fig.add_axes([0.15, 0.08, 0.72, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="Scroll",
        valmin=0,
        valmax=max_start,
        valinit=0,
        valstep=1,
    )

    def update(val: float) -> None:
        start = int(val)
        end = start + window_size

        x_window = x[start:end]
        y_window = values[start:end]
        line_main.set_data(x_window, y_window)
        ax.set_xlim(x_window[0], x_window[-1])

        redraw_edge_lines(start, end - 1)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair REDD events and optionally plot matched edges")
    parser.add_argument("--plot", action="store_true", help="Show interactive matched-edge plot")
    parser.add_argument("--window-size", type=int, default=2000, help="Visible samples per plot window")
    parser.add_argument("--building-id", type=int, default=None, help="Process a single building id (1-6)")
    parser.add_argument("--appliance", type=str, default=None, help="Process a single appliance name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    selected_buildings = [args.building_id] if args.building_id is not None else building_list
    selected_appliances = [args.appliance] if args.appliance is not None else appliance_names

    for i in selected_buildings:
        logger.info(f"========== Processing building {i} ==========")

        building_raw = pd.read_csv(f"building_{i}_raw.csv")
        df = pd.read_csv(f"building_{i}_main_transients_train.csv", index_col=0)
        building_pairs = {}

        for appliance in selected_appliances:
            logger.info(f"Processing appliance {appliance}")

            pairs, unmatched_on, unmatched_off = match_edges_stateful(
                df,
                appliance_col=f"{appliance}_label" if appliance != "unknown" else "unknown",
                power_weight=1.0,
                time_weight=0.05,
                max_duration=2000,
                max_time_gap=500,
            )

            if pairs:
                logger.info(f"\tMatched: {len(pairs)}")
                logger.info(f"\tUnmatched rises: {len(unmatched_on)}")
                logger.info(f"\tUnmatched falls: {len(unmatched_off)}")
                building_pairs[appliance] = pairs

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
                    building_raw.loc[int(pair["rise_time"]):int(pair["fall_time"]), f"{appliance}"] = 1

                    # Append the matched row to a new DataFrame
                    matched_events.append(episode | matched_row)

                pairs_df = pd.DataFrame(matched_events)
                pairs_df.to_csv(f"temp/building_{i}_{appliance}_matched_transitions.csv", index=False)

        if args.plot:
            plot_all_matched_edges(
                building_raw=building_raw,
                appliance_pairs=building_pairs,
                window_size=args.window_size,
            )

        building_raw.to_csv(f"building_{i}_labeled.csv", index=False)
