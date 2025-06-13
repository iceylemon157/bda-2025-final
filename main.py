import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Tuple

PUBLIC_DATA_PATH = 'public_data.csv'
PRIVATE_DATA_PATH = 'private_data.csv'
PUBLIC_SUBMISSION_PATH = 'public_submission.csv'
PRIVATE_SUBMISSION_PATH = 'private_submission.csv'

RANDOM_STATE = 157
np.random.seed(RANDOM_STATE)


def data_transformation(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != 'id']
    feature_cols_numeric = sorted([int(col) for col in feature_cols])
    original_dim_cols = [str(col) for col in feature_cols_numeric]
    n_dimensions = len(original_dim_cols)
    new_df = df.copy()
    for k in range(n_dimensions - 1):
        dim_k_col = original_dim_cols[k]
        dim_k_plus_1_col = original_dim_cols[k+1]

        x_vals = df[dim_k_col].values
        x_vals = x_vals.astype(float)
        y_vals = df[dim_k_plus_1_col].values
        y_vals = y_vals.astype(float)

        theta_k = np.arctan2(y_vals, x_vals)
        new_df[f'{k+1}_angle'] = theta_k
    
    new_df.drop(columns=original_dim_cols, inplace=True)
    return new_df


def multi_stage_angle_clustering(df: pd.DataFrame, dataset_name: str) -> Tuple[np.ndarray, int]:
    feature_cols = [col for col in df.columns if col != 'id']
    feature_cols_numeric = sorted([int(col) for col in feature_cols])
    original_dim_cols = [str(col) for col in feature_cols_numeric]
    
    n_dimensions = len(original_dim_cols)

    local_cluster_labels = []

    for k in range(n_dimensions - 1):
        dim_k_col = original_dim_cols[k]
        dim_k_plus_1_col = original_dim_cols[k+1]

        x_vals = df[dim_k_col].values
        x_vals = x_vals.astype(float)
        y_vals = df[dim_k_plus_1_col].values
        y_vals = y_vals.astype(float)

        theta_k = np.arctan2(y_vals, x_vals)

        kmeans_angle = GaussianMixture(n_components=n_dimensions + 1, 
                           random_state=RANDOM_STATE, 
                           n_init=10)
        
        pair_labels = kmeans_angle.fit_predict(theta_k.reshape(-1, 1))
        local_cluster_labels.append(pair_labels)

        degree_range = {}
        for label, angle in zip(pair_labels, theta_k):
            if label not in degree_range:
                degree_range[label] = (180, -180)
            angle_degrees = np.degrees(angle)
            current_min, current_max = degree_range[label]
            degree_range[label] = (min(current_min, angle_degrees), max(current_max, angle_degrees))
        
        plt.figure(figsize=(8, 6))
        plt.scatter(theta_k, np.random.randn(len(theta_k)), s=0.1)
        plt.title(f"Clustering of Angles for Dimensions {dim_k_col} and {dim_k_plus_1_col}")
        plt.xlabel(f"Theta")
        plt.ylabel(f"Random Value for Visualization")
        plt.grid()
        plt.savefig(f"plots/{dataset_name}/angle_clustering_{dim_k_col}_{dim_k_plus_1_col}.png")
        plt.close()

        # when drawing, remove the outliers (out 3 standard deviations)
        new_vals = np.column_stack((x_vals, y_vals))
        mean_vals = np.mean(new_vals, axis=0)
        std_vals = np.std(new_vals, axis=0)
        outlier_mask = np.all(np.abs(new_vals - mean_vals) <= 3 * std_vals, axis=1)
        new_vals = new_vals[outlier_mask]
        new_x, new_y = new_vals[:, 0], new_vals[:, 1]
        plt.figure(figsize=(10, 10))
        plt.scatter(new_x, new_y, s=8)
        plt.title(f"Scatter Plot for Dimensions {dim_k_col} and {dim_k_plus_1_col}")
        plt.xlabel(dim_k_col)
        plt.ylabel(dim_k_plus_1_col)
        plt.grid()

        def draw_angle_lines(angle, color='gray'):
            x_line = np.cos(np.radians(angle)) * max(new_x)
            y_line = np.sin(np.radians(angle)) * max(new_y)
            plt.plot([0, x_line], [0, y_line], linestyle='--', color=color, alpha=0.5)

        for min_angle, max_angle in degree_range.values():
            draw_angle_lines(min_angle, color='red')
            draw_angle_lines(max_angle, color='blue')
        
        plt.savefig(f"plots/{dataset_name}/scatter_{dim_k_col}_{dim_k_plus_1_col}.png")
        plt.close()

        with open("plots/angle_ranges.txt", "a") as f:
            f.write(f"Dimension pair ({dim_k_col}, {dim_k_plus_1_col}): {sorted(degree_range.values())}\n")

    all_local_labels = np.column_stack(local_cluster_labels)
    combined_labels_df = pd.DataFrame(all_local_labels)

    global_cluster_ids = pd.factorize(combined_labels_df.apply(lambda row: '_'.join(row.values.astype(str)), axis=1))[0]
    

    # Keep only the first 4n-1 clusters with most data
    top_n_clusters = 4 * n_dimensions - 1
    cluster_counts = pd.Series(global_cluster_ids).value_counts()
    top_clusters_idx = cluster_counts.nlargest(top_n_clusters).index
    top_clusters_idx = top_clusters_idx.astype(int)

    transformed_df = data_transformation(df)
    clustered_data = transformed_df.copy()
    clustered_data['global_cluster_id'] = global_cluster_ids
    clustered_data = clustered_data[clustered_data['global_cluster_id'].isin(top_clusters_idx)]

    centroids = transformed_df.drop(columns=['id']).groupby(global_cluster_ids).mean().loc[top_clusters_idx].values

    global_cluster_ids = global_cluster_ids.astype(int)
    for idx, row in transformed_df.iterrows():
        if row['id'] not in clustered_data['id'].values:
            distances = np.linalg.norm(centroids - row.drop('id').values, axis=1)
            nearest_centroid_idx = np.argmin(distances)
            global_cluster_ids[idx] = top_clusters_idx[int(nearest_centroid_idx)]

    return global_cluster_ids, n_dimensions

if __name__ == "__main__":
    with open("plots/angle_ranges.txt", "w") as f:
        f.write("Angle Ranges for Dimension Pairs:\n")

    # Public
    public_df = pd.read_csv(PUBLIC_DATA_PATH)
    public_id_col = public_df['id']
    
    public_cluster_labels, n_public_dims = multi_stage_angle_clustering(public_df, dataset_name='public')
    
    public_submission_df = pd.DataFrame({'id': public_id_col, 'label': public_cluster_labels})
    public_submission_df.to_csv(PUBLIC_SUBMISSION_PATH, index=False)

    # Private
    private_df = pd.read_csv(PRIVATE_DATA_PATH)
    private_id_col = private_df['id']

    private_cluster_labels, n_private_dims = multi_stage_angle_clustering(private_df, dataset_name='private')
    
    private_submission_df = pd.DataFrame({'id': private_id_col, 'label': private_cluster_labels})
    private_submission_df.to_csv(PRIVATE_SUBMISSION_PATH, index=False)