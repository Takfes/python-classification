from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

# set the random seed for reproducibility
SEED = 1990
TEST_RATIO = 1 / 4
n_samples = 1000
n_features = 5
majority_class_weight = 0.7
np.random.seed(SEED)


def main():
    # generate the synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=(n_features - n_features // 2),
        n_redundant=n_features // 2,
        n_clusters_per_class=2,
        n_classes=2,
        weights=(majority_class_weight, 1 - majority_class_weight),
        random_state=SEED,
    )

    # create some categorical features
    colors = np.random.choice(["red", "green", "blue"], size=n_samples)
    sizes = np.random.choice(["small", "medium", "large"], size=n_samples)

    # combine the numerical and categorical features into a DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["color"] = colors
    df["size"] = sizes
    df["target"] = y

    # split data
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    # also split dataset to align with pycaret experiment process
    train = X_train.assign(target=y_train)
    test = X_test.assign(target=y_test)

    # output data for ml training
    data = {}
    data["dataset"] = df
    data["X_train"] = X_train
    data["X_test"] = X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["train"] = train
    data["test"] = test

    # write the DataFrame to disk
    with open("classification_data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
