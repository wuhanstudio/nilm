import random

def shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = random.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]

def train_test_split(X, y, test_size=0.25, random_state=None):
    # Optional: seed the random number generator for reproducibility
    if random_state is not None:
        random.seed(random_state)

    # Create a list of indices and shuffle them
    indices = list(range(len(X)))
    shuffle(indices)

    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))

    # Split indices
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Use the indices to split X and y
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
