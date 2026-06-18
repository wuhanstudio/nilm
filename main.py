import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector

building_list = [1, 2, 3, 4, 5, 6]
output_dir = "temp"

appliance_names = ["main"]

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


def plot_main_matches(building_df, matched_pairs, building_id, window_size=2000):
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

    def redraw_edges(left_idx, right_idx):
        nonlocal rise_lines, fall_lines
        for ln in rise_lines:
            ln.remove()
        for ln in fall_lines:
            ln.remove()
        rise_lines = []
        fall_lines = []

        for r in rises:
            if left_idx <= r <= right_idx:
                rise_lines.append(ax.axvline(x=r, color="tab:green", linestyle="--", alpha=0.7))

        for f in falls:
            if left_idx <= f <= right_idx:
                fall_lines.append(ax.axvline(x=f, color="tab:red", linestyle="-", alpha=0.5))

    redraw_edges(0, window_size - 1)

    legend_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="main"),
        plt.Line2D([0], [0], color="tab:green", linestyle="--", label="rise"),
        plt.Line2D([0], [0], color="tab:red", linestyle="-", label="fall"),
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
                )
            else:
                logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")

        df_output.to_csv(f"building_{building_id}_output.csv", index=False)
