import glob
import pandas as pd
from tqdm import tqdm
from loguru import logger

from detector import EdgeDetector

NUM_BUILDINGS = 6

appliance_names = ["main", "fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["CE appliance"] # Always On and spikes

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

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
    for i in range(NUM_BUILDINGS):
        # Process each building
        building_id = i + 1
        logger.info(f"Processing Building {building_id}")

        # Pattern for files starting with 'redd_house_1' and ending with .csv
        building_pattern = f"redd_house{building_id}_*.csv"

        # Get list of matching files
        csv_files = glob.glob("redd/" + building_pattern)

        # Read and concatenate
        df = pd.concat((pd.read_csv(f, index_col=0) for f in csv_files), ignore_index=True)

        # Fill missing values using backward fill method
        df = df.bfill()

        for appliance in appliance_names:
            logger.info(f"Performing edge detection on Building {building_id} {appliance}...")

            if appliance in df.columns.to_list():
                appliance_df = df[[appliance]]
                transients, steady_states = edge_detection(appliance_df, noise_level=80, state_threshold=15)

                transients.to_csv(f"building_{building_id}_{appliance}_transients.csv", index=False)
                # steady_states.to_csv(f"building_{building_id}_{appliance}_steady_states.csv", index=True)
            else:
                logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")
