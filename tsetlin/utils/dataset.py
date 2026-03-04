import numpy as np

def balance_dataset(X_train, y_train, num_per_class=1000, seed=0):
    np.random.seed(seed)

    indices = []

    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        chosen = np.random.choice(cls_idx, num_per_class, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices
