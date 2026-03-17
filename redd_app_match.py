import pandas as pd
from loguru import logger

building_list = [1, 2, 3, 5]
appliance_list = ['fridge', 'microwave']

# Match rising and falling edges to get duration
for i in building_list:
    for appliance in appliance_list:
        logger.info(f"Processing building {i}, appliance: {appliance}")

        # Load data
        df = pd.read_csv(f"building_{i}_{appliance}_transients.csv")

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
        matched_df.to_csv(f"building_{i}_{appliance}_matched_transitions.csv", index=False)

        logger.info(f"Total transitions: {len(df)}")
        logger.info(f"Total matches: {len(matched_df) * 2}")
