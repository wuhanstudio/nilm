import os
import glob
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector
from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features

building_list = [1, 2, 3, 4, 5, 6]
output_dir = "temp"

appliance_names = ["main"]

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
    building_df: pd.DataFrame,
    start: int,
    end: int,
    candidates: list[str],
) -> str:
    # Use appliance-channel transition strength in the episode window as a proxy for ground truth.
    if building_df.empty:
        return "unknown"

    s = max(0, int(start))
    e = min(len(building_df) - 1, int(end))
    if e < s:
        return "unknown"

    best_label = "unknown"
    best_score = 0.0

    for appliance in candidates:
        if appliance not in building_df.columns:
            continue

        window = pd.to_numeric(building_df.loc[s:e, appliance], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if window.size == 0:
            continue

        # Transition-focused score: ignores constant always-on offsets better than simple energy sum.
        score = float(abs(window[-1] - window[0]) + (np.max(window) - np.min(window)))
        if score > best_score:
            best_score = score
            best_label = appliance

    return best_label if best_score > 0.0 else "unknown"


def best_subset_dp(active, target, max_subset_size=6, beam_width=200):
    # dp entries: (sum, indices, error)
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


def plot_main_matches(building_df, matched_pairs, building_id, window_size=2000, event_annotations=None):
    if "main" not in building_df.columns:
        logger.warning("Column 'main' not found, skipping plot.")
        return

    if building_df.empty:
        logger.warning("Empty building dataframe, skipping plot.")
        return

    values = pd.to_numeric(building_df["main"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy()
    x = building_df.index.to_numpy()

    window_size = min(max(1, int(window_size)), len(building_df))
    max_start = max(0, len(building_df) - window_size)

    rises = [int(p["start"]) for p in matched_pairs]
    falls = [int(p["end"]) for p in matched_pairs]

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)

    line_main, = ax.plot(x[:window_size], values[:window_size], color="black", linewidth=1.0, label="main")

    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x[0], x[window_size - 1])
    ax.set_title(f"Building {building_id} main signal with matched edges")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.grid(alpha=0.25)

    rise_lines = []
    fall_lines = []
    event_spans = []
    event_texts = []

    if event_annotations is None:
        event_annotations = [
            {
                "start": int(p["start"]),
                "end": int(p["end"]),
                "label": "unclassified",
            }
            for p in matched_pairs
        ]

    def redraw_edges(left_idx, right_idx):
        nonlocal rise_lines, fall_lines, event_spans, event_texts
        for ln in rise_lines:
            ln.remove()
        for ln in fall_lines:
            ln.remove()
        for sp in event_spans:
            sp.remove()
        for tx in event_texts:
            tx.remove()
        rise_lines = []
        fall_lines = []
        event_spans = []
        event_texts = []

        for r in rises:
            if left_idx <= r <= right_idx:
                rise_lines.append(ax.axvline(x=r, color="tab:green", linestyle="--", alpha=0.7))

        for f in falls:
            if left_idx <= f <= right_idx:
                fall_lines.append(ax.axvline(x=f, color="tab:red", linestyle="-", alpha=0.5))

        for ann in event_annotations:
            s = int(ann["start"])
            e = int(ann["end"])
            label = str(ann.get("label", "unclassified"))
            span_color = str(ann.get("color", "tab:orange"))

            if e < left_idx or s > right_idx:
                continue

            vis_s = max(s, left_idx)
            vis_e = min(e, right_idx)
            span = ax.axvspan(vis_s, vis_e, color=span_color, alpha=0.16)
            event_spans.append(span)

            text_x = vis_s + max(1, (vis_e - vis_s) * 0.02)
            text_y = y_max - 0.08 * (y_max - y_min)
            txt = ax.text(
                text_x,
                text_y,
                label,
                fontsize=8,
                color=span_color,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
            )
            event_texts.append(txt)

    redraw_edges(0, window_size - 1)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="main"),
        plt.Line2D([0], [0], color="tab:green", linestyle="--", label="rise"),
        plt.Line2D([0], [0], color="tab:red", linestyle="-", label="fall"),
        plt.Line2D([0], [0], color="tab:orange", linewidth=4.0, alpha=0.5, label="correct"),
        plt.Line2D([0], [0], color="red", linewidth=4.0, alpha=0.5, label="incorrect (show GT)"),
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

    def update(val):
        start = int(val)
        end = start + window_size
        x_window = x[start:end]
        y_window = values[start:end]
        line_main.set_data(x_window, y_window)
        ax.set_xlim(x_window[0], x_window[-1])
        redraw_edges(start, end - 1)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run main-edge detection and stateful matching")
    parser.add_argument("--building-id", type=int, default=None, help="Process only one building id")
    parser.add_argument("--window-size", type=int, default=2000, help="Visible samples in plot window")
    parser.add_argument("--model-path", type=str, default="tsetlin_redd_inference_model.ipb", help="Path to Tsetlin inference model")
    parser.add_argument("--stats-csv", type=str, default="redd_data_train.csv", help="Training CSV used to compute normalization stats")
    parser.add_argument("--n-bit", type=int, default=8, choices=[1, 2, 4, 8], help="Bits for feature booleanization")
    parser.add_argument("--no-inference", action="store_true", help="Skip Tsetlin inference")
    return parser.parse_args()

def edge_detection(dataframe, noise_level=50, state_threshold=15):
    detector = None
    with tqdm(total=dataframe.shape[0]) as pbar:
        for index, row in dataframe.iterrows():
            row = row.to_frame().iloc[0]
            current_time = row.index[0]
            current_measurement = row.iloc[0].item()

            # Initialize detector on first iteration
            if index == dataframe.index[0]:
                detector = EdgeDetector(current_time, current_measurement, state_threshold=state_threshold, noise_level=noise_level)
                continue

            output = detector.update(current_time, current_measurement)
            pbar.update(1)

    # Prepare DataFrames for steady states and transients
    steady_states = pd.DataFrame()
    transients = pd.DataFrame()

    assert len(detector.transitions) == len(detector.tran_data_list)

    # Create DataFrames if we have detected any transitions
    if len(detector.index_transitions_end) > 0:
        transients = pd.DataFrame({
            "transition": detector.transitions,
            "duration": [len(tran) for tran in detector.tran_data_list],
            "start": detector.index_transitions_start,
            "end": detector.index_transitions_end,
            "sequence": detector.tran_data_list
        })
        steady_states = pd.DataFrame(
            data=detector.steady_states, index=detector.index_steady_states, columns=["active average"]
        )
    
    return transients, steady_states

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_buildings = [args.building_id] if args.building_id is not None else building_list

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

    for building_id in selected_buildings:
        # Process each building
        logger.info(f"Processing Building {building_id}")

        # Pattern for files starting with 'redd_house_1' and ending with .csv
        building_pattern = f"redd_house{building_id}_*.csv"

        # Get list of matching files
        csv_files = glob.glob("redd/" + building_pattern)

        # Read and concatenate
        df = pd.concat((pd.read_csv(f, index_col=0) for f in csv_files), ignore_index=True)

        # Fill missing values using backward fill method
        df = df.bfill()
        df.to_csv(f"building_{building_id}_raw.csv", index=False)

        # Apply a threshold based classification
        # df_binary = df.copy()
        # cols_to_convert = [col for col in df.columns if col not in ["index", "main"]]
        # df_binary[cols_to_convert] = (df[cols_to_convert] >= 80).astype(int)
        # df_binary.to_csv(f"{output_dir}/building_{building_id}_binary.csv", index=False)

        # Output: Reset all appliance states as 0
        df_output = df.copy()
        df_output.loc[:, df_output.columns != "main"] = 0

        for appliance in appliance_names:
            logger.info(f"Performing edge detection on Building {building_id} {appliance}...")

            if appliance in df.columns.to_list():
                appliance_df = df[[appliance]]
                transients, steady_states = edge_detection(appliance_df, noise_level=80, state_threshold=15)

                # transients.to_csv(f"{output_dir}/building_{building_id}_{appliance}_transients.csv", index=False)
                # steady_states.to_csv(f"{output_dir}/building_{building_id}_{appliance}_steady_states.csv", index=True)

                logger.info(f"Processing building {building_id}, appliance: {appliance}")

                matched_pairs = match_edges_stateful(
                    transients,
                    max_duration=2000,
                    max_time_gap=500,
                    max_subset_size=6,
                )
                results = [
                    {
                        "appliance": appliance,
                        "transition": p["transition"],
                        "duration": p["duration"],
                        "start": p["start"],
                        "end": p["end"],
                    }
                    for p in matched_pairs
                ]

                # Convert to DataFrame
                matched_df = pd.DataFrame(results)

                # Generate richer episode features for each matched rise/fall pair.
                feature_rows = []
                main_values = df["main"].to_numpy(dtype=np.float64)
                for p in matched_pairs:
                    row = {
                        "transition": p["transition"],
                        "duration": p["duration"],
                        "start": p["start"],
                        "end": p["end"],
                    }
                    row.update(
                        episode_feature_row(
                            main_values,
                            {
                                "start": p["start"],
                                "end": p["end"],
                                "duration": p["duration"],
                                "pos_delta": p["transition"],
                                "neg_delta": -p["transition"],
                            },
                        )
                    )
                    feature_rows.append(row)

                features_df = pd.DataFrame(feature_rows)

                if tsetlin_model is not None and not features_df.empty:
                    raw_pred = infer_with_tsetlin(tsetlin_model, features_df, X_mean, X_std, args.n_bit)
                    features_df["pred_label"] = raw_pred
                    features_df["pred_appliance"] = [label_to_appliance.get(int(p), "unknown") for p in raw_pred]

                event_annotations = []
                for idx, p in enumerate(matched_pairs):
                    pred_label = "unclassified"
                    if "pred_appliance" in features_df.columns and idx < len(features_df):
                        pred_label = str(features_df.iloc[idx]["pred_appliance"])

                    gt_label = infer_ground_truth_appliance(
                        building_df=df,
                        start=int(p["start"]),
                        end=int(p["end"]),
                        candidates=inference_appliance_names,
                    )

                    is_incorrect = (
                        pred_label != "unclassified"
                        and gt_label != "unknown"
                        and pred_label != gt_label
                    )

                    if is_incorrect:
                        display_label = f"{pred_label}→{gt_label}"
                    else:
                        display_label = pred_label
                    span_color = "red" if is_incorrect else "tab:orange"

                    if idx < len(features_df):
                        features_df.loc[idx, "gt_appliance"] = gt_label
                        features_df.loc[idx, "is_correct"] = int(not is_incorrect)

                    event_annotations.append(
                        {
                            "start": int(p["start"]),
                            "end": int(p["end"]),
                            "label": display_label,
                            "color": span_color,
                        }
                    )

                if not features_df.empty:
                    features_df.to_csv(f"{output_dir}/building_{building_id}_main_matched_features_inference.csv", index=False)

                # Save if needed
                # matched_df.to_csv(f"{output_dir}/building_{building_id}_{appliance}_matched_transitions.csv", index=False)

                logger.info(f"Total transitions: {len(transients)}")
                logger.info(f"Total matches: {len(matched_df) * 2}")

                for res in results:
                    df_output.loc[res['start']:res['end'], res['appliance']] = 1

                plot_main_matches(
                    building_df=df,
                    matched_pairs=matched_pairs,
                    building_id=building_id,
                    window_size=args.window_size,
                    event_annotations=event_annotations,
                )
            else:
                logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")

        df_output.to_csv(f"building_{building_id}_output.csv", index=False)
