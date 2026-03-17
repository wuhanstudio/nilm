import argparse
from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features
from tsetlin.utils.split import train_test_split

building_list = [1, 2, 3, 5]

redd_data = pd.DataFrame()
# Concatenate matched transitions for each building
for i in building_list:
    redd_data = pd.concat([redd_data, pd.read_csv(f"building_{i}_matched_transitions.csv")], ignore_index=True)

    # Draw a scatter plot of duration vs transition for each appliance
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for appliance in ['fridge', 'microwave',]: # 'unknown']:
        subset = redd_data[redd_data['appliance'] == appliance]
        plt.scatter(subset['transition'], subset['duration'], label=appliance, alpha=0.6)
    plt.xlabel('Transition')
    plt.ylabel('Duration')
    plt.title('Transition vs Duration for Appliances')
    plt.legend()
    plt.show()

redd_data['label'] = redd_data['appliance'].map({
    'fridge': 0,
    'microwave': 1,
    'unknown': 2
})

X = redd_data[['transition', 'duration']]
y = redd_data['label']

# Convert dataframe to numpy array
X = X.values
y = y.values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=200, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=50, help="Number of states")
    parser.add_argument("--n_bit", type=int, default=8, help="Number of bits in [1, 2, 4, 8]")
    
    parser.add_argument("--T", type=int, default=20, help="Threshold T")
    parser.add_argument("--s", type=float, default=6.0, help="Specificity s")

    args = parser.parse_args()

    N_EPOCHS = args.epochs

    N_BIT = args.n_bit
    if N_BIT not in {1, 2, 4, 8}:
        raise ValueError("n_bit must be one of [1, 2, 4, 8]")

    # Normalization
    X_mean = np.mean(X)
    X_std = np.std(X)

    logger.info(f"Using {N_BIT} bits for booleanization")
    X_bool = booleanize_features(X, X_mean, X_std, num_bits=N_BIT)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    N_CLAUSE = args.n_clause
    N_STATE  = args.n_state

    logger.info(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    logger.info(f"Threshold T: {args.T}, Specificity s: {args.s}")

    tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

    y_pred = tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    for epoch in range(N_EPOCHS):
        logger.info(f"[Epoch {epoch+1}/{N_EPOCHS}] Train Accuracy: {accuracy * 100:.2f}%")
        for i in tqdm(range(len(X_train))):
            tsetlin.step(X_train[i], y_train[i], T=args.T, s=args.s)

        y_pred = tsetlin.predict(X_train)
        accuracy = sum([ 1 if pred == train else 0 for pred, train in zip(y_pred, y_train)]) / len(y_train)

    logger.info("")

    # Final evaluation
    y_pred = tsetlin.predict(X_test)
    accuracy = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]) / len(y_test)

    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))