import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector

building_list = [1, 2, 3, 4, 5, 6]
output_dir = "temp"

appliance_names = ["main", "fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["CE appliance"] # Always On and spikes

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

def plot_edge_detection(dataframe, steady_states, transients, window_size=1200, title=None):
    if dataframe.empty:
        logger.warning("Skipping plot because dataframe is empty.")
        return

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    y_min = dataframe.min().item()
    y_max = dataframe.max().item()
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    ax.set_ylim(y_min, y_max)

    window_size = min(max(1, int(window_size)), len(dataframe))

    x0 = dataframe.index[:window_size]
    y0 = dataframe.iloc[:window_size].values
    line_main, = ax.plot(x0, y0, color="blue")

    line_states = None
    if not steady_states.empty:
        mask = (steady_states.index >= x0[0]) & (steady_states.index <= x0[-1])
        line_states, = ax.plot(
            steady_states.index[mask],
            steady_states["active average"][mask],
            "o",
            color="orange",
        )

    for _, tran in transients.iterrows():
        ax.axvline(x=tran["start"], color="r", linestyle="--", alpha=0.3)

    ax.set_xlim(x0[0], x0[-1])
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time")
    if title:
        ax.set_title(title)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Start",
        0,
        max(0, len(dataframe) - window_size),
        valinit=0,
        valstep=1,
    )

    def update(val):
        start = int(val)
        end = start + window_size

        x = dataframe.index[start:end]
        y = dataframe.iloc[start:end].values
        line_main.set_data(x, y)
        ax.set_xlim(x[0], x[-1])

        if line_states is not None:
            mask = (steady_states.index >= x[0]) & (steady_states.index <= x[-1])
            line_states.set_data(steady_states.index[mask], steady_states["active average"][mask])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def edge_detection(dataframe, noise_level=50, state_threshold=15, plot=False, window_size=1200, plot_title=None):
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
            "duration": [detector.index_transitions_end[i] - detector.index_transitions_start[i] for i in range(len(detector.index_transitions_start))],
            "start": detector.index_transitions_start,
            "end": detector.index_transitions_end,
            # "sequence": detector.tran_data_list
        })
        steady_states = pd.DataFrame(
            data=detector.steady_states, index=detector.index_steady_states, columns=["active average"]
        )

    if plot:
        plot_edge_detection(
            dataframe=dataframe,
            steady_states=steady_states,
            transients=transients,
            window_size=window_size,
            title=plot_title,
        )
    
    return transients, steady_states


def parse_args():
    parser = argparse.ArgumentParser(description="REDD edge detection with optional interactive plotting")
    parser.add_argument("--plot", action="store_true", help="Show interactive plot with slider")
    parser.add_argument("--window-size", type=int, default=1200, help="Visible samples per plot window")
    parser.add_argument("--building-id", type=int, default=None, help="Process a single building id (1-6)")
    return parser.parse_args()

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

        for appliance in appliance_names:
            logger.info(f"Performing edge detection on Building {building_id} {appliance}...")

            if appliance in df.columns.to_list():
                appliance_df = df[[appliance]]
                transients, steady_states = edge_detection(
                    appliance_df,
                    noise_level=80,
                    state_threshold=15,
                    plot=args.plot,
                    window_size=args.window_size,
                    plot_title=f"Building {building_id} - {appliance}",
                )

                transients.to_csv(f"{output_dir}/building_{building_id}_{appliance}_transients.csv", index=False)
                # steady_states.to_csv(f"{output_dir}/building_{building_id}_{appliance}_steady_states.csv", index=True)
            else:
                logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")
