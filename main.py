import os
import glob
import pandas as pd
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        df.to_csv(f"{output_dir}/building_{building_id}_combined.csv", index=False)

        df_binary = df.copy()
        # Columns to convert (exclude index and main)
        cols_to_convert = [col for col in df.columns if col not in ["index", "main"]]

        # Apply threshold
        df_binary[cols_to_convert] = (df[cols_to_convert] >= 80).astype(int)

        # Save result while keeping index column
        df_binary.to_csv(f"{output_dir}/building_{building_id}_binary.csv", index=False)

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

                stacks = []
                results = []
                for _, row in transients.iterrows():
                    trans = row['transition']
                
                    # Rising edge
                    if trans > 0:
                        stacks.append(row)

                    # Falling edge
                    else:
                        if stacks:
                            rise = stacks.pop()

                            # If stack is empty, we have a match
                            # if not stacks:
                            results.append({
                                'appliance': appliance,
                                'transition': rise['transition'],
                                'duration': row['end'] - rise['start'],
                                'start': rise['start'],
                                'end': row['end']
                            })

                # Convert to DataFrame
                matched_df = pd.DataFrame(results)

                # Save if needed
                matched_df.to_csv(f"{output_dir}/building_{building_id}_{appliance}_matched_transitions.csv", index=False)

                logger.info(f"Total transitions: {len(transients)}")
                logger.info(f"Total matches: {len(matched_df) * 2}")

                for res in results:
                    df_output.loc[res['start']:res['end'], res['appliance']] = 1
            else:
                logger.warning(f"{appliance} not found in Building {building_id}. Skipping...")

        df_output.to_csv(f"building_{building_id}_output.csv", index=False)
