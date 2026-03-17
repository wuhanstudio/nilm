import pandas as pd

import itertools
from tqdm import tqdm

building_list = [1, 2, 3, 5]

tolerance = 2

def find_match(building_main, building_app, app_name, tolerance):
    print(f"Processing {app_name}")

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
    print(f"Processing building {i}")

    building_main =  pd.read_csv(f"building_{i}_main_transients.csv")
    building_fridge = pd.read_csv(f"building_{i}_fridge_transients.csv")
    building_microwave = pd.read_csv(f"building_{i}_microwave_transients.csv")
    
    building_main, not_found_list = find_match(building_main, building_fridge, "fridge_label", tolerance)
    print(f"main: {len(building_main)}, fridge: {len(building_fridge)}, not found: {len(not_found_list)}")
    # print(not_found_list)

    building_main, not_found_list = find_match(building_main, building_microwave, "microwave_label", tolerance)
    print(f"main: {len(building_main)}, microwave: {len(building_microwave)}, not found: {len(not_found_list)}")
    # print(not_found_list)

    building_main.to_csv(f"building_{i}_main_transients_train.csv")

# Match rising and falling edges to get duration
for i in building_list:
    print(f"Processing building {i}")

    # Load data
    df = pd.read_csv(f"building_{i}_main_transients_train.csv")

    results = []

    # Separate stacks per appliance
    stacks = {
        'fridge': [],
        'microwave': [],
        'unknown': []
    }

    for _, row in df.iterrows():
        trans = row['transition']
        
        # Determine appliance
        if row['fridge_label'] == 1:
            key = 'fridge'
        elif row['microwave_label'] == 1:
            key = 'microwave'
        else:
            key = 'unknown'
        
        # Rising edge
        if trans > 0:
            stacks[key].append(row)

        # Falling edge
        else:
            if stacks[key]:
                rise = stacks[key].pop()

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

    print(f"Total matches: {len(matched_df) * 2} / {len(df)}")
