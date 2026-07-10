import random
random.seed(0)

import argparse
from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd
from bitarray import bitarray
from sklearn.metrics import classification_report, confusion_matrix

from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features
from tsetlin.utils.split import train_test_split
from tsetlin.compiler.write import tsetlin_compile

building_list_train = [1, 2, 4, 5, 6]
building_list_test  = [3]
output_dir = "temp"

features = ["transition", "duration"]
features += [
        "pos_transition_magnitude",
        "neg_transition_magnitude",
        "abs_transition",
        "log_abs_transition",
        "duration",
        "log_duration",
        "transition_duration_product",
        "transition_duration_ratio",
        "episode_mean_main",
        "episode_std_main",
        "episode_min_main",
        "episode_max_main",
        "episode_range_main",
        "internal_diff_mean_abs",
        "internal_diff_max_abs",
        "internal_edge_count",
        "subcycle_count_proxy",
        "active_fraction_proxy",
        "episode_energy_estimate",
        "post_minus_pre_mean",
        "event_internal_edge_count",
]
print(f"Features: {features}")

appliance_names = ["fridge", "microwave"]
# appliance_names = ["fridge", "microwave", "dish washer", "electric furnace"]
# appliance_names = ["fridge", "microwave", "dish washer", "electric furnace", "unknown"]
# appliance_names = ["fridge", "microwave", "dish washer", "electric furnace", "CE appliance"]

# Not working ones
# appliance_names = ["washer dryer"] # Bug

# appliance_names = ["waste disposal unit"] # Spikes
# appliance_names = ["electric stove", "electric space heater"] # Low threshold

# Auto generate dictionary for appliances
appliance_dict = {name: idx for idx, name in enumerate(appliance_names)}

def read_redd_data(building_list, appliance_names, output_dir):
    redd_data = pd.DataFrame()
    # Concatenate matched transitions for each building
    for i in building_list:
        for appliance in appliance_names:
            try:
                df = pd.read_csv(f"{output_dir}/building_{i}_{appliance}_matched_transitions.csv")
                redd_data = pd.concat([redd_data, df], ignore_index=True)
            except FileNotFoundError:
                logger.warning(f"File for building {i}, appliance {appliance} not found. Skipping...")
            except pd.errors.EmptyDataError:
                logger.warning(f"File for building {i}, appliance {appliance} is empty. Skipping...")
        
        if 'unknown' in appliance_names:
            try:
                df = pd.read_csv(f"{output_dir}/building_{i}_matched_transitions.csv")
                df = df[df['appliance'] == 'unknown']
                redd_data = pd.concat([redd_data, df], ignore_index=True)
            except FileNotFoundError:
                logger.warning(f"File for building {i}, appliance unknown not found. Skipping...")
            except pd.errors.EmptyDataError:
                logger.warning(f"File for building {i}, appliance unknown is empty. Skipping...")

    # Draw a scatter plot with publication-friendly typography.
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 20,
        # 'font.weight': 'bold',
        'axes.labelsize': 24,
        # 'axes.labelweight': 'bold',
        'axes.titlesize': 28,
        # 'axes.titleweight': 'bold',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18,
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    for appliance in appliance_names:
        subset = redd_data[redd_data['appliance'] == appliance]
        ax.scatter(subset['transition'], subset['duration'], label=appliance, alpha=0.6)

    ax.set_xlabel('Transition', fontsize=24)
    ax.set_ylabel('Duration', fontsize=24)
    ax.set_title('Transition vs Duration for Appliances', fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    legend = ax.legend(fontsize=18)
    legend.set_title('Appliance', prop={'size': 18, 'weight': 'bold'})
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.show()

    redd_data['label'] = redd_data['appliance'].map(appliance_dict)

    X = redd_data[features]
    y = redd_data['label']

    # Convert dataframe to numpy array
    X = X.values
    y = y.values

    return X, y

# Read the REDD data for training
X_train, y_train = read_redd_data(building_list_train, appliance_names, output_dir)

# Save the training data to a CSV file
train_data_df = pd.DataFrame(X_train, columns=features)
train_data_df['label'] = y_train
train_data_df.to_csv('redd_data_train.csv', index=False)

# Read the REDD data for testing
X_test, y_test = read_redd_data(building_list_test, appliance_names, output_dir)

# Save the test data to a CSV file
test_data_df = pd.DataFrame(X_test, columns=features)
test_data_df['label'] = y_test
test_data_df.to_csv('redd_data_test.csv', index=False)

def objective(trial):
    experiment_results = dict(
        accuracy=[],
        train_time=[],
        test_time=[],
    )

    n_epochs = 10
    n_state = trial.suggest_int("n_state", 2, 256, step=2)
    n_clause = trial.suggest_int("n_clause", 2, 500, step=2)

    T = trial.suggest_int("T", 1, n_state)
    s = trial.suggest_float("s", 1.0, 10.0, step=0.1)

    logger.info(f"Number of clauses: {n_clause}, Number of states: {n_state}")
    logger.info(f"Threshold T: {T}, Specificity s: {s}")

    m_tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=len(np.unique(y_train)), N_clause=n_clause, N_state=n_state)

    logger.info(f"Running for {n_epochs} epochs")

    y_pred = m_tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    for epoch in range(n_epochs):
        logger.info(f"[Epoch {epoch+1}/{n_epochs}] Train Accuracy: {accuracy * 100:.2f}%")
        for i in tqdm(range(len(X_train))):
            m_tsetlin.step(X_train[i], y_train[i], T=T, s=s)

        y_pred = m_tsetlin.predict(X_train)
        accuracy = sum([ 1 if pred == train else 0 for pred, train in zip(y_pred, y_train)]) / len(y_train)

    logger.info("")

    # Final evaluation
    y_pred = m_tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    return (1.0 - accuracy)  # Optuna minimizes the objective, so we return 1 - accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=200, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=50, help="Number of states")
    parser.add_argument("--n_bit", type=int, default=8, help="Number of bits in [1, 2, 4, 8]")
    
    parser.add_argument("--T", type=int, default=20, help="Threshold T")
    parser.add_argument("--s", type=float, default=6.0, help="Specificity s")

    parser.add_argument("--optuna", action='store_true')

    args = parser.parse_args()

    N_EPOCHS = args.epochs

    N_BIT = args.n_bit
    if N_BIT not in {1, 2, 4, 8}:
        raise ValueError("n_bit must be one of [1, 2, 4, 8]")

    # Normalization
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    logger.info(f"Feature mean: {X_mean}, Feature std: {X_std}")

    logger.info(f"Using {N_BIT} bits for booleanization")
    X_train = booleanize_features(X_train, X_mean, X_std, num_bits=N_BIT)
    X_test = booleanize_features(X_test, X_mean, X_std, num_bits=N_BIT)

    # Convert to bitarray
    X_train = [bitarray(list(map(bool, x))) for x in X_train]
    X_test = [bitarray(list(map(bool, x))) for x in X_test]

    # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    N_CLAUSE = args.n_clause
    N_STATE  = args.n_state

    logger.info(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    logger.info(f"Threshold T: {args.T}, Specificity s: {args.s}")

    if args.optuna:
        import optuna
            
        # Create a new study.
        # study = optuna.create_study()

        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            study_name=f"tsetlin-machine-redd",  # Name your study.}",
            load_if_exists="True"
        )
        
        # Invoke optimization of the objective function.
        study.optimize(objective, n_trials=100)  

        print(f"Best value: {study.best_value} (params: {study.best_params})")

    m_tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=len(np.unique(y_train)), N_clause=N_CLAUSE, N_state=N_STATE)

    y_pred = m_tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    for epoch in range(N_EPOCHS):
        logger.info(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
        for i in tqdm(range(len(X_train))):
            m_tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s)

        y_pred = m_tsetlin.predict(X_train)
        accuracy = sum([ 1 if pred == train else 0 for pred, train in zip(y_pred, y_train)]) / len(y_train)

    logger.info("")

    # Final evaluation
    y_pred = m_tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model
    m_tsetlin.save_model("tsetlin_redd_model.pb", type="training")
    logger.info("Model saved to tsetlin_redd_model.pb")

    # Load the model and evaluate again
    m_tsetlin_loaded = Tsetlin.load_model("tsetlin_redd_model.pb")
    y_pred_loaded = m_tsetlin_loaded.predict(X_test)
    accuracy_loaded = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred_loaded, y_test)]) / len(y_test)

    logger.info(f"Test Accuracy after loading model: {accuracy_loaded * 100:.2f}%")
    logger.info("")

    # Save inference model
    m_tsetlin.save_model("tsetlin_redd_inference_model.ipb", type="inference")
    logger.info("Inference Model saved to tsetlin_redd_inference_model.ipb")

    # Load inference model and evaluate
    m_tsetlin_inference = Tsetlin.load_model("tsetlin_redd_inference_model.ipb")
    y_pred_inference = m_tsetlin_inference.predict(X_test)

    accuracy_inference = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred_inference, y_test)]) / len(y_test)
    logger.info(f"Test Accuracy after loading inference model: {accuracy_inference * 100:.2f}%")

    # Compile the inference model to C header
    tsetlin_compile("tsetlin_redd_inference_model.ipb", "redd_model.h")
