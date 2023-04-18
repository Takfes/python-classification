from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# set the random seed for reproducibility
SEED = 1990
np.random.seed(SEED)

# define the number of samples and features
n_samples = 1000
n_features = 5
majority_class_weight = 0.7


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

    df.target.value_counts()

    # write the DataFrame to a CSV file
    df.to_pickle("classification_dataset.pkl")


if __name__ == "__main__":
    main()
