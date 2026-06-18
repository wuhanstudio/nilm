import os
import glob
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from loguru import logger
from tqdm import tqdm
from detector import EdgeDetector
from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features

inference_appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]

tm_features = ["transition", "duration"]
tm_features += [
    "pos_transition_magnitude",
    "neg_transition_magnitude",
    "abs_transition",
    "log_abs_transition",
    "duration",
    "log_duration",
    "transition_duration_product",
    "transition_duration_ratio",
    "episode_mean_main",
    "episode_std_main",
    "episode_min_main",
    "episode_max_main",
    "episode_range_main",
    "internal_diff_mean_abs",
    "internal_diff_max_abs",
    "internal_edge_count",
    "subcycle_count_proxy",
    "active_fraction_proxy",
    "episode_energy_estimate",
    "post_minus_pre_mean",
    "event_internal_edge_count",
]


def safe_slice(values: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(values) - 1, int(end))
    if end < start:
        return np.asarray([], dtype=np.float64)
    return values[start : end + 1]


def episode_feature_row(values: np.ndarray, episode: dict) -> dict:
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


def load_normalization_stats(stats_csv: str, expected_feature_count: int):
    if not os.path.exists(stats_csv):
        logger.warning(f"Stats file not found: {stats_csv}. Falling back to zero-mean/unit-std.")
        return np.zeros(expected_feature_count), np.ones(expected_feature_count)

    stats_df = pd.read_csv(stats_csv)
    numeric_df = stats_df.select_dtypes(include=[np.number])

    if "label" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["label"])

    if numeric_df.shape[1] < expected_feature_count:
        logger.warning(
            f"Stats CSV has {numeric_df.shape[1]} numeric feature columns, expected {expected_feature_count}. "
            "Falling back to zero-mean/unit-std."
        )
        return np.zeros(expected_feature_count), np.ones(expected_feature_count)

    numeric_vals = numeric_df.iloc[:, :expected_feature_count].to_numpy(dtype=np.float64)
    mean = np.mean(numeric_vals, axis=0)
    std = np.std(numeric_vals, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def infer_with_tsetlin(model, feature_df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, n_bit: int):
    if feature_df.empty:
        return []

    X = feature_df[tm_features].to_numpy(dtype=np.float64)
    X_bool = booleanize_features(X.copy(), mean, std, num_bits=n_bit)
    return model.predict(X_bool)


def infer_ground_truth_appliance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    candidates: list[str],
) -> str:
    # Use appliance-channel transition strength in the episode window as a proxy for ground truth.
    if df.empty or start_idx >= len(df):
        return "unknown"

    s = max(0, int(start_idx))
    e = min(len(df) - 1, int(end_idx))
    if e < s:
        return "unknown"

    best_label = "unknown"
    best_score = 0.0

    for appliance in candidates:
        if appliance not in df.columns:
            continue

        window = pd.to_numeric(df.loc[s:e, appliance], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if window.size == 0:
            continue

        score = float(abs(window[-1] - window[0]) + (np.max(window) - np.min(window)))
        if score > best_score:
            best_score = score
            best_label = appliance

    return best_label if best_score > 0.0 else "unknown"


def best_subset_dp(active, target, max_subset_size=6, beam_width=200):
    dp = [(0.0, [], abs(target))]

    for i, rise in enumerate(active):
        val = float(rise["transition"])
        expanded = list(dp)

        for s, idxs, _ in dp:
            new_sum = s + val
            new_idxs = idxs + [i]
            new_err = abs(new_sum - target)
            expanded.append((new_sum, new_idxs, new_err))

        expanded.sort(key=lambda x: x[2])
        dp = expanded[:beam_width]

    valid = [x for x in dp if len(x[1]) <= max_subset_size]
    if not valid:
        return [], float("inf")

    best = min(valid, key=lambda x: x[2])
    return best[1], best[2]


def match_edges_stateful(transients, max_duration=2000, max_time_gap=500, max_subset_size=6):
    if transients.empty:
        return []

    events = transients.sort_values("start")
    active_rises = []
    pairs = []

    for _, row in events.iterrows():
        power = float(row["transition"])
        t = float(row["start"])

        if power > 0:
            active_rises.append({"transition": power, "time": t})
            continue

        target = abs(power)
        active_rises = [r for r in active_rises if (t - r["time"]) <= max_time_gap]
        if not active_rises:
            continue

        subset_idx, _ = best_subset_dp(active_rises, target, max_subset_size=max_subset_size)
        if not subset_idx:
            continue

        used = [active_rises[i] for i in subset_idx]
        rise_time = min(u["time"] for u in used)
        rise_sum = sum(u["transition"] for u in used)
        duration = t - rise_time

        if max_duration is not None and duration > max_duration:
            continue

        pairs.append(
            {
                "transition": rise_sum,
                "duration": duration,
                "start": rise_time,
                "end": t,
            }
        )

        for i in sorted(subset_idx, reverse=True):
            active_rises.pop(i)

    return pairs


def init_realtime_plot(building_id: int):
    fig, ax = plt.subplots(figsize=(14, 6))
    (line_main,) = ax.plot([], [], color="black", linewidth=1.0, label="main")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Building {building_id} Real-Time NILM (main-only input)")
    ax.grid(alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="main"),
        plt.Line2D([0], [0], color="tab:green", linestyle="--", label="detected rise"),
        plt.Line2D([0], [0], color="tab:red", linestyle="-", label="detected fall"),
        plt.Line2D([0], [0], color="tab:orange", linewidth=4.0, alpha=0.5, label="paired/correct"),
        plt.Line2D([0], [0], color="red", linewidth=4.0, alpha=0.5, label="paired/incorrect (pred→GT)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    return fig, ax, line_main


def redraw_realtime_plot(
    ax,
    line_main,
    all_values,
    detected_edges,
    finalized_events,
    window_size,
    edge_artists,
    event_span_artists,
    event_text_artists,
):
    if len(all_values) == 0:
        return

    x = np.arange(len(all_values), dtype=np.int64)
    y = np.asarray(all_values, dtype=np.float64)
    line_main.set_data(x, y)

    right = len(all_values) - 1
    left = max(0, right - int(window_size) + 1)
    ax.set_xlim(left, max(left + 1, right))

    y_window = y[left : right + 1]
    y_min = float(np.min(y_window)) if y_window.size else 0.0
    y_max = float(np.max(y_window)) if y_window.size else 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    for artist in edge_artists:
        artist.remove()
    for artist in event_span_artists:
        artist.remove()
    for artist in event_text_artists:
        artist.remove()
    edge_artists.clear()
    event_span_artists.clear()
    event_text_artists.clear()

    for edge in detected_edges:
        t = int(edge["time"])
        if left <= t <= right:
            color = "tab:green" if float(edge["transition"]) > 0 else "tab:red"
            style = "--" if float(edge["transition"]) > 0 else "-"
            edge_artists.append(ax.axvline(x=t, color=color, linestyle=style, alpha=0.5, linewidth=1.0))

    text_y = y_max - 0.06 * (y_max - y_min)
    for ann in finalized_events:
        s = int(ann["start"])
        e = int(ann["end"])
        if e < left or s > right:
            continue

        vis_s = max(s, left)
        vis_e = min(e, right)
        color = str(ann.get("color", "tab:orange"))
        label = str(ann.get("label", "unclassified"))

        event_span_artists.append(ax.axvspan(vis_s, vis_e, color=color, alpha=0.16))
        event_text_artists.append(
            ax.text(
                vis_s + max(1, (vis_e - vis_s) * 0.02),
                text_y,
                label,
                fontsize=8,
                color=color,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time edge detection and TM inference pipeline")
    parser.add_argument("--building-id", type=int, default=1, help="Building ID for REDD dataset")
    parser.add_argument("--model-path", type=str, default="tsetlin_redd_inference_model.ipb", help="Path to Tsetlin inference model")
    parser.add_argument("--stats-csv", type=str, default="redd_data_train.csv", help="Training CSV for normalization stats")
    parser.add_argument("--n-bit", type=int, default=8, choices=[1, 2, 4, 8], help="Bits for feature booleanization")
    parser.add_argument("--idle-threshold", type=float, default=200.0, help="Power threshold below which appliances are considered off (W)")
    parser.add_argument("--idle-duration", type=int, default=50, help="Number of samples to consider idle before triggering inference (time units)")
    parser.add_argument("--window-size", type=int, default=2000, help="Visible window size in animated plot")
    parser.add_argument("--pause-ms", type=float, default=10.0, help="Pause per sample for animation in milliseconds")
    parser.add_argument("--no-plot", action="store_true", help="Disable live animated plot")
    parser.add_argument("--no-inference", action="store_true", help="Skip Tsetlin inference")
    parser.add_argument("--output-dir", type=str, default="temp", help="Output directory for results")
    return parser.parse_args()


def process_real_time(args):
    """Main real-time processing loop."""
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load TM model and normalization stats
    tsetlin_model = None
    X_mean = None
    X_std = None
    label_to_appliance = {idx: name for idx, name in enumerate(inference_appliance_names)}

    if not args.no_inference:
        if os.path.exists(args.model_path):
            tsetlin_model = Tsetlin.load_model(args.model_path)
            expected_feature_count = max(1, tsetlin_model.n_features // args.n_bit)
            X_mean, X_std = load_normalization_stats(args.stats_csv, expected_feature_count)
            logger.info(f"Loaded Tsetlin inference model: {args.model_path}")
        else:
            logger.warning(f"Model file not found: {args.model_path}. Inference disabled.")

    # Load building data from REDD files
    building_pattern = f"redd_house{args.building_id}_*.csv"
    csv_files = glob.glob("redd/" + building_pattern)
    
    if not csv_files:
        logger.error(f"No REDD files found for building {args.building_id} matching pattern: redd/{building_pattern}")
        return
    
    logger.info(f"Loading REDD data from {len(csv_files)} files for building {args.building_id}")
    df = pd.concat((pd.read_csv(f, index_col=0) for f in csv_files), ignore_index=True)
    df = df.bfill()
    logger.info(f"Loaded {len(df)} data points")

    # Initialize detector and state
    detector = None
    main_values_buffer = deque(maxlen=10000)  # Keep episode-local history for features
    accumulated_transients = []
    idle_counter = 0
    episode_start_idx = None

    all_main_values = []
    detected_edges = []
    finalized_events = []

    fig = None
    ax = None
    line_main = None
    edge_artists = []
    event_span_artists = []
    event_text_artists = []
    if not args.no_plot:
        plt.ion()
        fig, ax, line_main = init_realtime_plot(args.building_id)

    episode_count = 0

    try:
        # Process data points one by one, use tqdm for progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data points"):
            try:
                timestamp = idx
                main_power = float(row["main"]) if "main" in row.index else float(row.iloc[0])
                all_main_values.append(main_power)
                
                # Initialize detector on first iteration
                if detector is None:
                    detector = EdgeDetector(timestamp, main_power, state_threshold=15, noise_level=80)
                    episode_start_idx = int(timestamp)
                    main_values_buffer.clear()
                    main_values_buffer.append(main_power)
                    logger.info(f"Initialized detector at sample {timestamp} with power {main_power}W")
                    if not args.no_plot:
                        redraw_realtime_plot(
                            ax,
                            line_main,
                            all_main_values,
                            detected_edges,
                            finalized_events,
                            args.window_size,
                            edge_artists,
                            event_span_artists,
                            event_text_artists,
                        )
                        plt.pause(max(0.0, args.pause_ms / 1000.0))
                    continue

                # Feed data point to edge detector (main meter only)
                detector.update(timestamp, main_power)
                main_values_buffer.append(main_power)

                # Collect transients as they are detected
                if len(detector.index_transitions_end) > len(accumulated_transients):
                    # New transients detected
                    new_count = len(detector.index_transitions_end) - len(accumulated_transients)
                    for i in range(new_count):
                        idx_start = len(accumulated_transients)
                        accumulated_transients.append({
                            "transition": detector.transitions[idx_start],
                            "duration": len(detector.tran_data_list[idx_start]),
                            "start": detector.index_transitions_start[idx_start],
                            "end": detector.index_transitions_end[idx_start],
                        })
                        detected_edges.append(
                            {
                                "time": int(detector.index_transitions_start[idx_start]),
                                "transition": float(detector.transitions[idx_start]),
                            }
                        )

                # Track idle time (main power below threshold)
                if main_power < args.idle_threshold:
                    idle_counter += 1
                else:
                    idle_counter = 0

                # When idle duration exceeded and we have events, trigger inference pipeline
                if idle_counter >= args.idle_duration and len(accumulated_transients) > 0:
                    logger.info(f"Idle threshold reached after {idle_counter} samples. Triggering inference pipeline with {len(accumulated_transients)} accumulated transients.")
                    
                    # Create transients dataframe
                    transients_df = pd.DataFrame(accumulated_transients)
                    
                    # Match edges
                    matched_pairs = match_edges_stateful(
                        transients_df,
                        max_duration=2000,
                        max_time_gap=500,
                        max_subset_size=6,
                    )
                    logger.info(f"Matched {len(matched_pairs)} rise/fall pairs")

                    # Generate features
                    feature_rows = []
                    main_array = np.array([val for val in main_values_buffer], dtype=np.float64)
                    
                    for p in matched_pairs:
                        local_start = int(p["start"]) - int(episode_start_idx)
                        local_end = int(p["end"]) - int(episode_start_idx)
                        if local_end < 0:
                            continue

                        row = {
                            "transition": p["transition"],
                            "duration": p["duration"],
                            "start": p["start"],
                            "end": p["end"],
                        }
                        row.update(
                            episode_feature_row(
                                main_array,
                                {
                                    "start": local_start,
                                    "end": local_end,
                                    "duration": p["duration"],
                                    "pos_delta": p["transition"],
                                    "neg_delta": -p["transition"],
                                },
                            )
                        )
                        feature_rows.append(row)

                    features_df = pd.DataFrame(feature_rows)

                    # Run Tsetlin inference
                    if tsetlin_model is not None and not features_df.empty:
                        raw_pred = infer_with_tsetlin(tsetlin_model, features_df, X_mean, X_std, args.n_bit)
                        features_df["pred_label"] = raw_pred
                        features_df["pred_appliance"] = [label_to_appliance.get(int(p), "unknown") for p in raw_pred]
                        logger.info(f"Tsetlin predictions: {list(features_df['pred_appliance'])}")

                    # Add ground truth and correctness
                    for idx_row, (_, p) in enumerate(features_df.iterrows() if not features_df.empty else []):
                        gt_label = infer_ground_truth_appliance(
                            df=df,
                            start_idx=int(p["start"]) if idx_row < len(feature_rows) else 0,
                            end_idx=int(p["end"]) if idx_row < len(feature_rows) else 0,
                            candidates=inference_appliance_names,
                        )
                        pred_label = str(p.get("pred_appliance", "unclassified")) if not features_df.empty else "unclassified"
                        is_correct = (pred_label != "unclassified" and gt_label != "unknown" and pred_label == gt_label)
                        
                        features_df.loc[idx_row, "gt_appliance"] = gt_label
                        features_df.loc[idx_row, "is_correct"] = int(is_correct)
                        if is_correct:
                            display_label = pred_label
                            color = "tab:orange"
                        else:
                            display_label = f"{pred_label}→{gt_label}"
                            color = "red"
                        finalized_events.append(
                            {
                                "start": int(p["start"]),
                                "end": int(p["end"]),
                                "label": display_label,
                                "color": color,
                            }
                        )

                    # Save results
                    if not features_df.empty:
                        output_file = f"{output_dir}/building_{args.building_id}_episode_{episode_count}_inference.csv"
                        features_df.to_csv(output_file, index=False)
                        logger.info(f"Saved inference results to {output_file}")
                        
                        episode_count += 1

                    # Clear accumulated events and reset
                    logger.info("Clearing accumulated transients and waiting for new activity...")
                    accumulated_transients = []
                    idle_counter = 0
                    main_values_buffer.clear()
                    detector = None  # Will reinitialize on next data point
                    episode_start_idx = None

                if not args.no_plot:
                    redraw_realtime_plot(
                        ax,
                        line_main,
                        all_main_values,
                        detected_edges,
                        finalized_events,
                        args.window_size,
                        edge_artists,
                        event_span_artists,
                        event_text_artists,
                    )
                    plt.pause(max(0.0, args.pause_ms / 1000.0))

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if not args.no_plot and fig is not None:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    args = parse_args()
    process_real_time(args)
