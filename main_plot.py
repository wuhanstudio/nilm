import glob
from tqdm import tqdm
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt

from detector import EdgeDetector

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
    ax = dataframe.plot()
    if not steady_states.empty:
        steady_states.plot(style="o", ax=ax)
        for _, tran in transients.iterrows():
            plt.axvline(x=tran["start time"], color='r', linestyle='--', label='Start Time')
            # plt.axvline(x=tran["end time"], color='g', linestyle='--', label='End Time')

    plt.legend(["Measurement", "Steady states"])
    plt.ylabel("Power (W)")
    plt.xlabel("Time")
    plt.show()

    return transients, steady_states

if __name__ == "__main__":
    # Load REDD dataset
    building_id = 1
    file_pattern = f"redd_house{building_id}_*.csv"

    # Get list of matching files
    csv_files = glob.glob("redd/" + file_pattern)

    print("Files found:", csv_files)
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Fill missing values using backward fill method
    df = df.bfill()

    # Get main meter data
    main_df = df[['main']]

    # Get fridge data
    fridge_df = df[['fridge']]

    # Get microwave data
    microwave_df = df[['microwave']]

    # Plot edge detection results
    logger.info(f"Performing edge detection on Building {building_id} main meter...")
    transients_main, steady_states_main = plot_edge_detection(main_df, noise_level=80, state_threshold=15)
    logger.info(f"Detected {len(transients_main)} edges for main meter.")

    logger.info(f"Performing edge detection on Building {building_id} fridge...")
    transients_fridge, steady_states_fridge = plot_edge_detection(fridge_df, noise_level=80, state_threshold=15)
    logger.info(f"Detected {len(transients_fridge)} edges for fridge.")

    logger.info(f"Performing edge detection on Building {building_id} microwave...")
    transients_microwave, steady_states_microwave = plot_edge_detection(microwave_df, noise_level=80, state_threshold=15)
    logger.info(f"Detected {len(transients_microwave)} edges for microwave.")
