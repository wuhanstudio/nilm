import argparse
import glob
from loguru import logger

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

building_list = [1, 2, 3, 4, 5, 6]

def build_keep_mask(
    series: pd.Series,
    *,
    small_threshold: float,
    near_const_delta: float,
    min_run_length: int,
    keep_points: int,
    keep_last: bool,
) -> np.ndarray:
    """
    Build a boolean mask for rows to keep.

    A run is considered redundant when:
    1) values stay within +/- small_threshold, and
    2) each next value differs from the previous by <= near_const_delta.

    If run length is greater than min_run_length, only keep_points are kept
    (and optionally the last point of the run).
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    keep = np.ones(n, dtype=bool)

    i = 0
    while i < n:
        v = values[i]
        if np.isnan(v) or abs(v) > small_threshold:
            i += 1
            continue

        j = i + 1
        while j < n:
            curr = values[j]
            prev = values[j - 1]
            if np.isnan(curr) or np.isnan(prev):
                break
            if abs(curr) > small_threshold:
                break
            if abs(curr - prev) > near_const_delta:
                break
            j += 1

        run_len = j - i
        if run_len > min_run_length and keep_points < run_len:
            drop_start = i + keep_points
            drop_end = j - (1 if keep_last else 0)
            if drop_start < drop_end:
                keep[drop_start:drop_end] = False

        i = j

    return keep


def prune_csv(
    input_csv: str,
    output_csv: str,
    column: str,
    *,
    small_threshold: float,
    near_const_delta: float,
    min_run_length: int,
    keep_points: int,
    keep_last: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    keep_mask = build_keep_mask(
        df[column],
        small_threshold=small_threshold,
        near_const_delta=near_const_delta,
        min_run_length=min_run_length,
        keep_points=keep_points,
        keep_last=keep_last,
    )
    pruned = df.loc[keep_mask].reset_index(drop=True)
    pruned.to_csv(output_csv, index=False)

    removed = len(df) - len(pruned)
    print(f"Input rows:  {len(df)}")
    print(f"Output rows: {len(pruned)}")
    print(f"Removed rows: {removed}")

    return df, keep_mask, pruned


def plot_with_scrollbar(
    pruned_df: pd.DataFrame,
    column: str,
    window_size: int,
) -> None:
    values = pd.to_numeric(pruned_df[column], errors="coerce").to_numpy(dtype=float)
    x = np.arange(len(values))

    if len(values) == 0:
        print("Nothing to plot: pruned CSV is empty.")
        return

    window_size = max(1, min(window_size, len(values)))
    max_start = max(0, len(values) - window_size)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    ax.plot(x, values, color="tab:blue", linewidth=1.0, label="Pruned")
    ax.set_title(f"{column}: pruned signal")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, window_size - 1)

    slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="Scroll",
        valmin=0,
        valmax=max_start,
        valinit=0,
        valstep=1,
    )

    def update(start: float) -> None:
        # Move a fixed-width window across the series.
        start_idx = int(start)
        end_idx = start_idx + window_size - 1
        ax.set_xlim(start_idx, end_idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove redundant near-constant small-value samples from a CSV column."
    )
    parser.add_argument(
        "--small-threshold",
        type=float,
        default=200.0,
        help="Values with abs(value) <= threshold are considered small (default: 200.0)",
    )
    parser.add_argument(
        "--near-const-delta",
        type=float,
        default=10.0,
        help="Max absolute point-to-point delta for near-constant values (default: 10.0)",
    )
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=10,
        help="Only runs longer than this are pruned (default: 10)",
    )
    parser.add_argument(
        "--keep-points",
        type=int,
        default=10,
        help="How many points to keep at the start of a long run (default: 10)",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        help="Also drop the final point of each pruned run (default keeps it)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show an interactive plot with a scroll slider",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2000,
        help="Number of samples visible in one window when plotting (default: 2000)",
    )
    return parser.parse_args()

if __name__ == "__main__":

    for building_id in building_list:
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
        df.to_csv(f"tests/building_{building_id}_combined.csv", index=False)

        input_csv = f"tests/building_{building_id}_combined.csv"
        output_csv = f"tests/building_{building_id}_pruned.csv"

        args = parse_args()
        _, _, pruned_df = prune_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            column="main",
            small_threshold=args.small_threshold,
            near_const_delta=args.near_const_delta,
            min_run_length=args.min_run_length,
            keep_points=args.keep_points,
            keep_last=not args.drop_last,
        )

        if args.plot:
            plot_with_scrollbar(
                pruned_df=pruned_df,
                column="main",
                window_size=args.window_size,
            )
