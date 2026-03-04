def encode_labels(y):
    label_map = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }
    return [label_map[label] for label in y]

def load_iris_X_y(filename):
    X = []
    y = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Extract header
        headers = lines[0].strip().split(',')

        # Assume last column is the label (species)
        feature_indices = list(range(len(headers) - 1))
        label_index = len(headers) - 1

        for line in lines[1:]:
            values = line.strip().split(',')
            # Extract features and convert to float
            features = [float(values[i]) for i in feature_indices]
            label = values[label_index]
            X.append(features)
            y.append(label)

    y = encode_labels(y)

    return X, y

if __name__ == '__main__':

    # Example usage
    X, y = load_iris_X_y('iris.csv')

    # Print first 3 entries
    for i in range(3):
        print("X:", X[i], "y:", y[i])
