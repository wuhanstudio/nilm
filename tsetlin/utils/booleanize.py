import math
from tqdm import tqdm

def erf(x):
    # Approximation of the error function (Abramowitz and Stegun, formula 7.1.26)
    # with maximal error ~1.5e-7
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y

def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + erf(z))

def booleanize(x: float, num_bits: int=8) -> list[bool]:
    if not (0.0 <= x <= 1.0):
        raise ValueError("Input float must be between 0.0 and 1.0")

    if num_bits not in {1, 2, 4, 8}:
        raise ValueError("num_bits must be 1, 2, 4, or 8")

    # Scale float to integer range
    max_val = (1 << num_bits) - 1  # 2^num_bits - 1

    # !Important: Round to even to avoid bias
    int_val = round(x * max_val)

    # Convert to boolean list
    bool_bits = [(int_val >> i) & 1 == 1 for i in reversed(range(num_bits))]
    return bool_bits

def booleanize_features(X, mean, std, num_bits: int=8):
    # Normalization to [0, 255]
    for i in tqdm(range(len(X)), desc="Normalizing features"):
        for j in range(len(X[i])):
            if isinstance(mean, list) and isinstance(std, list):
                X[i][j] = (X[i][j] - mean[j]) / std[j]
            else:
                X[i][j] = (X[i][j] - mean) / std

            # Map to [0, 1] using CDF of standard normal distribution
            X[i][j] = norm_cdf(X[i][j])

    # Mistake: Shouldn't do this
    # X = [[int(x* 255) for x in row] for row in X]

    X_bool = []
    for x_features in tqdm(X, desc="Booleanizing features"):
        bool_features = [booleanize(x, num_bits=num_bits) for x in x_features]
        X_bool.append([b for row in bool_features for b in row])

    return X_bool
