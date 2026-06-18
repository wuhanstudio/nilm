import argparse
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from loguru import logger

building_list = [1, 2, 3, 4, 5, 6]
output_dir = "temp"

appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["CE appliance"] # Always On and spikes

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

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


def plot_edge_match(building_raw, building_app, appliance, window_size=1000):
    if appliance not in building_raw.columns:
        logger.warning(f"{appliance} not found in raw data. Skipping plotting...")
        return

    if building_raw.empty:
        logger.warning("Raw data is empty. Skipping plotting...")
        return

    window_size = min(max(1, int(window_size)), len(building_raw))

    main_df = building_raw["main"]
    app_df = building_raw[appliance]

    fig, (ax_main, ax_app) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    plt.subplots_adjust(bottom=0.25)

    ax_main.set_ylabel("Main Power (W)")
    ax_app.set_ylabel(appliance)

    main_min, main_max = main_df.min(), main_df.max()
    app_min, app_max = app_df.min(), app_df.max()
    if main_min == main_max:
        main_min -= 1.0
        main_max += 1.0
    if app_min == app_max:
        app_min -= 1.0
        app_max += 1.0

    ax_main.set_ylim(main_min, main_max)
    ax_app.set_ylim(app_min, app_max)

    x0 = building_raw.index[:window_size]
    y_main = main_df.iloc[:window_size].values
    y_app = app_df.iloc[:window_size].values

    line_main, = ax_main.plot(x0, y_main, color="black")
    line_app, = ax_app.plot(x0, y_app, color="blue")

    ax_main.set_title("Main Power Signal")
    ax_app.set_title(f"Appliance: {appliance}")
    ax_main.grid(True)
    ax_app.grid(True)
    ax_app.set_xlabel("Time")
    ax_main.set_xlim(x0[0], x0[-1])

    for _, tran in building_app.iterrows():
        ax_main.axvline(x=tran["start"], color="r", linestyle="--", alpha=0.3)
        ax_app.axvline(x=tran["start"], color="r", linestyle="--", alpha=0.3)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Start",
        0,
        max(0, len(building_raw) - window_size),
        valinit=0,
        valstep=1,
    )

    def update(val):
        start = int(val)
        end = start + window_size

        x = building_raw.index[start:end]
        y_main = main_df.iloc[start:end].values
        y_app = app_df.iloc[start:end].values

        line_main.set_data(x, y_main)
        line_app.set_data(x, y_app)
        ax_main.set_xlim(x[0], x[-1])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="REDD edge matching with optional interactive plotting")
    parser.add_argument("--plot", action="store_true", help="Show interactive plot with slider")
    parser.add_argument("--window-size", type=int, default=1000, help="Visible samples per plot window")
    parser.add_argument("--building-id", type=int, default=None, help="Process a single building id (1-6)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    selected_buildings = [args.building_id] if args.building_id is not None else building_list

    # Match appliance with main transitions
    for i in selected_buildings:
        try:
            building_main = pd.read_csv(f"{output_dir}/building_{i}_main_transients.csv")
        except FileNotFoundError:
            logger.warning(f"Main transient file for building {i} not found. Skipping building...")
            continue
        except pd.errors.EmptyDataError:
            logger.warning(f"Main transient file for building {i} is empty. Skipping building...")
            continue

        building_raw = None
        if args.plot:
            try:
                building_raw = pd.read_csv(f"building_{i}_raw.csv")
            except FileNotFoundError:
                logger.warning(f"Raw file building_{i}_raw.csv not found. Plotting disabled for building {i}.")
            except pd.errors.EmptyDataError:
                logger.warning(f"Raw file building_{i}_raw.csv is empty. Plotting disabled for building {i}.")

        for appliance in appliance_names:
            if f"{appliance}_label" not in building_main.columns:
                building_main[f"{appliance}_label"] = 0

        for appliance in appliance_names:
            logger.info(f"Processing building {i}, appliance: {appliance}")

            try:
                building_app = pd.read_csv(f"{output_dir}/building_{i}_{appliance}_transients.csv")
            except FileNotFoundError:
                logger.warning(f"File for building {i}, appliance {appliance} not found. Skipping...")
                continue
            except pd.errors.EmptyDataError:
                logger.warning(f"File for building {i}, appliance {appliance} is empty. Skipping...")
                continue

            building_main, not_found_list = find_match(building_main, building_app, f"{appliance}_label", tolerance)
            logger.info(f"main: {len(building_main)}, {appliance}: {len(building_app)}, not found: {len(not_found_list)}")

            if args.plot and building_raw is not None:
                plot_edge_match(
                    building_raw=building_raw,
                    building_app=building_app,
                    appliance=appliance,
                    window_size=args.window_size,
                )

        building_main["unknown"] = (
            building_main[[f"{appliance}_label" for appliance in appliance_names]].sum(axis=1) == 0
        ).astype(int)

        # Save the matched transitions to a new CSV file
        building_main.to_csv(f"building_{i}_main_transients_train.csv")
