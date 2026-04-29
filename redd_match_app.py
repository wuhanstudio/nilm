import pandas as pd
from loguru import logger

building_list = [1, 2, 3, 4, 5, 6]
output_dir = "temp"

appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["CE appliance"] # Always On and spikes

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

# Match rising and falling edges to get duration
for i in building_list:
    for appliance in appliance_names:
        logger.info(f"Processing building {i}, appliance: {appliance}")

        # Check if the file exists
        try:
            df = pd.read_csv(f"{output_dir}/building_{i}_{appliance}_transients.csv")
        except FileNotFoundError:
            logger.warning(f"File for building {i}, appliance {appliance} not found. Skipping...")
            continue
        except pd.errors.EmptyDataError:
            logger.warning(f"File for building {i}, appliance {appliance} is empty. Skipping...")
            continue

        stacks = []
        results = []
        for _, row in df.iterrows():
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
        matched_df.to_csv(f"{output_dir}/building_{i}_{appliance}_matched_transitions.csv", index=False)

        logger.info(f"Total transitions: {len(df)}")
        logger.info(f"Total matches: {len(matched_df) * 2}")
