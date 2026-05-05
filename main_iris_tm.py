import random

from tsetlin.compiler.clause_compressed import emit_clausec_arrays
random.seed(0)

import argparse
import numpy as np

from tqdm import tqdm
from loguru import logger

from iris import load_iris_X_y

from tsetlin import Tsetlin
from tsetlin.utils.booleanize import booleanize_features
from tsetlin.utils.split import train_test_split
from tsetlin.compiler.write import tsetlin_compile

X_train, X_test, y_train, y_test = None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsetlin Machine on Iris Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--n_clause", type=int, default=100, help="Number of clauses")
    parser.add_argument("--n_state", type=int, default=10, help="Number of states")
    parser.add_argument("--n_bit", type=int, default=4, help="Number of bits in [1, 2, 4, 8]")
    
    parser.add_argument("--T", type=int, default=30, help="Threshold T")
    parser.add_argument("--s", type=float, default=6.0, help="Specificity s")

    args = parser.parse_args()

    N_EPOCHS = args.epochs

    N_BIT = args.n_bit
    if N_BIT not in {1, 2, 4, 8}:
        raise ValueError("n_bit must be one of [1, 2, 4, 8]")

    # Example usage
    X, y = load_iris_X_y('iris.csv')

    # Normalization
    X_mean = np.mean(X)
    X_std = np.std(X)

    print(f"Feature mean: {X_mean}, Feature std: {X_std}")

    logger.info(f"Using {N_BIT} bits for booleanization")
    X_bool = booleanize_features(X, X_mean, X_std, num_bits=N_BIT)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_bool, y, test_size=0.2, random_state=0)

    N_CLAUSE = args.n_clause
    N_STATE  = args.n_state

    logger.info(f"Number of clauses: {N_CLAUSE}, Number of states: {N_STATE}")
    logger.info(f"Threshold T: {args.T}, Specificity s: {args.s}")

    m_tsetlin = Tsetlin(N_feature=len(X_train[0]), N_class=3, N_clause=N_CLAUSE, N_state=N_STATE)

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
    logger.info("")

    # Save the model
    m_tsetlin.save_model("tsetlin_iris_model.pb", type="training")
    logger.info("Model saved to tsetlin_iris_model.pb")

    # Load the model and evaluate again
    m_tsetlin_loaded = Tsetlin.load_model("tsetlin_iris_model.pb")
    y_pred_loaded = m_tsetlin_loaded.predict(X_test)
    accuracy_loaded = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred_loaded, y_test)]) / len(y_test)

    logger.info(f"Test Accuracy after loading model: {accuracy_loaded * 100:.2f}%")
    logger.info("")

    # Save inference model
    m_tsetlin.save_model("tsetlin_iris_inference_model.ipb", type="inference")
    logger.info("Inference Model saved to tsetlin_iris_inference_model.ipb")

    # Load inference model and evaluate
    m_tsetlin_inference = Tsetlin.load_model("tsetlin_iris_inference_model.ipb")
    y_pred_inference = m_tsetlin_inference.predict(X_test)

    accuracy_inference = sum([ 1 if pred == test else 0 for pred, test in zip(y_pred_inference, y_test)]) / len(y_test)
    logger.info(f"Test Accuracy after loading inference model: {accuracy_inference * 100:.2f}%")

    # Compile the inference model to C header
    tsetlin_compile("tsetlin_iris_inference_model.ipb", "iris_model.h")
