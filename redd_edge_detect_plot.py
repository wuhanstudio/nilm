import glob
from tqdm import tqdm
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from detector import EdgeDetector

building_id = 1

appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["CE appliance"] # Always On and spikes

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

def plot_edge_detection(dataframe, noise_level=50, state_threshold=15):
    detector = None
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        row = row.to_frame().iloc[0]
        current_time = row.index[0]
        current_measurement = row.iloc[0].item()

        # Initialize detector on first iteration
        if index == dataframe.index[0]:
            detector = EdgeDetector(current_time, current_measurement, state_threshold=state_threshold, noise_level=noise_level)
            continue

        output = detector.update(current_time, current_measurement)
        # if output.get('transition', False):
        #     logger.info(f"Duration: {len(output['transition_data'])} samples")
        #     logger.info(f"Transition: {output['transition_power_change']}")
        #     logger.info(f"Transition: {output['transition_data']}")
        #     logger.info("---")

    # Prepare DataFrames for steady states and transients
    steady_states = pd.DataFrame()
    transients = pd.DataFrame()

    assert len(detector.transitions) == len(detector.tran_data_list)

    # Create DataFrames if we have detected any transitions
    if len(detector.index_transitions_end) > 0:
        transients = pd.DataFrame({
            "active transition": detector.transitions,
            "start time": detector.index_transitions_start,
            "end time": detector.index_transitions_end
        })
        steady_states = pd.DataFrame(
            data=detector.steady_states, index=detector.index_steady_states, columns=["active average"]
        )

    # Plot steady states with main
    # ax = dataframe.plot()
    # if not steady_states.empty:
    #     steady_states.plot(style="o", ax=ax)
    #     for _, tran in transients.iterrows():
    #         plt.axvline(x=tran["start time"], color='r', linestyle='--', label='Start Time')
    #         # plt.axvline(x=tran["end time"], color='g', linestyle='--', label='End Time')

    # plt.legend(["Measurement", "Steady states"])
    # plt.ylabel("Power (W)")
    # plt.xlabel("Time")
    # plt.show()

    # --- Interactive plotting ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Fixed y-axis
    y_min = dataframe.min().item()
    y_max = dataframe.max().item()
    ax.set_ylim(y_min, y_max)

    window_size = 1200

    # Initial plot window
    x0 = dataframe.index[:window_size]
    y0 = dataframe.iloc[:window_size].values
    line_main, = ax.plot(x0, y0, color='blue')

    # Steady states circles
    line_states = None
    if not steady_states.empty:
        mask = (steady_states.index >= x0[0]) & (steady_states.index <= x0[-1])
        line_states, = ax.plot(
            steady_states.index[mask],
            steady_states["active average"][mask],
            'o',
            # label="Steady states",
            color='orange'
        )

    # Transient start lines
    for i, (_, tran) in enumerate(transients.iterrows()):
        ax.axvline(x=tran["start time"], color='r', linestyle='--')

    ax.set_xlim(x0[0], x0[-1])
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time")
    # ax.legend()

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Start",
        0,
        max(0, len(dataframe) - window_size),
        valinit=0,
        valstep=None  # smooth continuous sliding
    )

    # Update function
    def update(val):
        start = int(slider.val)
        end = start + window_size

        x = dataframe.index[start:end]
        y = dataframe.iloc[start:end].values

        line_main.set_data(x, y)
        ax.set_xlim(x[0], x[-1])

        # Update steady states circles
        if line_states is not None:
            mask = (steady_states.index >= x[0]) & (steady_states.index <= x[-1])
            line_states.set_data(steady_states.index[mask], steady_states["active average"][mask])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return transients, steady_states

if __name__ == "__main__":
    # Load REDD dataset
    file_pattern = f"redd_house{building_id}_*.csv"

    # Get list of matching files
    csv_files = glob.glob("redd/" + file_pattern)

    print("Files found:", csv_files)
    df = pd.concat((pd.read_csv(f, index_col=0) for f in csv_files), ignore_index=True)

    # Fill missing values using backward fill method
    df = df.bfill()

    for appliance in appliance_names:
        logger.info(f"Performing edge detection on Building {building_id} {appliance}...")

        if appliance in df.columns.to_list():
            appliance_df = df[[appliance]]
            transients, steady_states = plot_edge_detection(appliance_df, noise_level=80, state_threshold=15)
            logger.info(f"Detected {len(transients)} edges for {appliance}.")
        else:
            logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")
