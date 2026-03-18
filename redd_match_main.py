import pandas as pd
import itertools
from tqdm import tqdm
from loguru import logger

building_list = [1, 2, 3, 4, 5, 6]
appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]

# Not working ones
# appliance_name = ["washer dryer"] # Bug
# appliance_name = ["CE appliance"] # Always On and spikes
# appliance_name = ["waste disposal unit"] # Spikes
# appliance_name = ["electric stove", "electric space heater"] # Low threshold

tolerance = 2

def find_match(building_main, building_app, app_name, tolerance):
    building_main[app_name] = 0

    # building_main['start'] = pd.to_datetime(building_main['start'])
    # building_main['end'] = pd.to_datetime(building_main['end'])
    
    # building_app['start'] = pd.to_datetime(building_app['start'])
    # building_app['end'] = pd.to_datetime(building_app['end'])

    current_main = 0
    not_found_list = []
    for i, f_tran in tqdm(building_app.iterrows(), total=building_app.shape[0]):
        f_interval = pd.Interval(f_tran['start'] - tolerance, f_tran['end'] + tolerance, closed='both')
    
        found = False
        for j, m_tran in itertools.islice(building_main.iterrows(), current_main, None):
            m_interval = pd.Interval(m_tran['start'] - tolerance, m_tran['end'] + tolerance, closed='both')
            if f_interval.overlaps(m_interval):
                found = True
                current_main = j
                building_main.loc[j, app_name] = 1
                break
        if not found:
            not_found_list.append(i)

    return building_main, not_found_list

# Match appliance with main transitions
for i in building_list:
    building_main =  pd.read_csv(f"building_{i}_main_transients.csv")

    for appliance in appliance_names:
        logger.info(f"Processing building {i}, appliance: {appliance}")

        try:
            building_app = pd.read_csv(f"building_{i}_{appliance}_transients.csv")
        except FileNotFoundError:
            logger.warning(f"File for building {i}, appliance {appliance} not found. Skipping...")
            continue
        except pd.errors.EmptyDataError:
            logger.warning(f"File for building {i}, appliance {appliance} is empty. Skipping...")
            continue

        building_main, not_found_list = find_match(building_main, building_app, f"{appliance}_label", tolerance)
        logger.info(f"main: {len(building_main)}, {appliance}: {len(building_app)}, not found: {len(not_found_list)}")
        # logger.info(not_found_list)

    building_main.to_csv(f"building_{i}_main_transients_train.csv")

# Match rising and falling edges to get duration
for i in building_list:
    logger.info(f"Processing building {i}")

    # Load data
    df = pd.read_csv(f"building_{i}_main_transients_train.csv", index_col=0)

    results = []

    # Separate stacks per appliance
    stacks = {appliance: [] for appliance in appliance_names}
    stacks['unknown'] = []

    for _, row in df.iterrows():
        trans = row['transition']

        # Determine appliance
        found_appliance = False
        for appliance in appliance_names:
            if f"{appliance}_label" in row and row[f"{appliance}_label"] == 1:
                key = appliance
                found_appliance = True
                break
        if not found_appliance:
            key = 'unknown'
        
        # Rising edge
        if trans > 0:
            stacks[key].append(row)

        # Falling edge
        else:
            if stacks[key]:
                rise = stacks[key].pop()

                if not stacks[key]:  # If stack is empty, we have a match
                    results.append({
                        'appliance': key,
                        'transition': rise['transition'],
                        'duration': row['end'] - rise['start'],
                        'start': rise['start'],
                        'end': row['end']
                    })

    # Convert to DataFrame
    matched_df = pd.DataFrame(results)

    # Save if needed
    matched_df.to_csv(f"building_{i}_matched_transitions.csv", index=False)

    logger.info(f"Total matches: {len(matched_df) * 2} / {len(df)}")
