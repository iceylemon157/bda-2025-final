import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["id"])

    for column in data.select_dtypes(include=[np.number]).columns:
        mean_value = data[column].mean()
        std_dev = data[column].std()
        outliers = (data[column] < mean_value - 3 * std_dev) | (
            data[column] > mean_value + 3 * std_dev
        )
        data.loc[outliers, column] = mean_value

    return data


def plot_feature_pairs(data_path):
    dataset_name = data_path.split("_")[0]
    data = load_and_preprocess_data(data_path)
    pd.plotting.scatter_matrix(data, figsize=(12, 12), diagonal="kde")
    plt.suptitle("Pairwise Feature Relationships")
    plt.savefig(f"plots/{dataset_name}_feature_pairs_plot.png")
    plt.close()


if __name__ == "__main__":
    public_path = "public_data.csv"
    plot_feature_pairs(public_path)

    private_path = "private_data.csv"
    plot_feature_pairs(private_path)
