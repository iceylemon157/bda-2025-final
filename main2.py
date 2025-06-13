import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple

PUBLIC_DATA_PATH = 'public_data.csv'
PRIVATE_DATA_PATH = 'private_data.csv'
PUBLIC_SUBMISSION_PATH = 'public_submission.csv'
PRIVATE_SUBMISSION_PATH = 'private_submission.csv'

RANDOM_STATE = 157
np.random.seed(RANDOM_STATE)


def multi_stage_angle_clustering(df: pd.DataFrame, dataset_name: str) -> np.ndarray:
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
    print(new_df)

    cluster_counts = 4 * n_dimensions - 1
    kmeans_angle = KMeans(n_clusters=cluster_counts, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans_angle.fit_predict(new_df.values)

    return labels

    all_local_labels = np.column_stack(local_cluster_labels)
    combined_labels_df = pd.DataFrame(all_local_labels)

    global_cluster_ids = pd.factorize(combined_labels_df.apply(lambda row: '_'.join(row.values.astype(str)), axis=1))[0]
    

    # Keep only the first 4n-1 clusters with most data
    top_n_clusters = 4 * n_dimensions - 1
    cluster_counts = pd.Series(global_cluster_ids).value_counts()
    top_clusters_idx = cluster_counts.nlargest(top_n_clusters).index
    top_clusters_idx = top_clusters_idx.astype(int)

    clustered_data = df.copy()
    clustered_data['global_cluster_id'] = global_cluster_ids
    clustered_data = clustered_data[clustered_data['global_cluster_id'].isin(top_clusters_idx)]

    centroids = df[original_dim_cols].groupby(global_cluster_ids).mean().loc[top_clusters_idx].values

    global_cluster_ids = global_cluster_ids.astype(int)
    for idx, row in df.iterrows():
        if row['id'] not in clustered_data['id'].values:
            distances = np.linalg.norm(centroids - row[original_dim_cols].values, axis=1)
            nearest_centroid_idx = np.argmin(distances)
            global_cluster_ids[idx] = top_clusters_idx[int(nearest_centroid_idx)]

    return global_cluster_ids, n_dimensions

if __name__ == "__main__":
    with open("plots/angle_ranges.txt", "w") as f:
        f.write("Angle Ranges for Dimension Pairs:\n")

    # Public
    public_df = pd.read_csv(PUBLIC_DATA_PATH)
    public_id_col = public_df['id']
    
    public_cluster_labels = multi_stage_angle_clustering(public_df, dataset_name='public')
    
    public_submission_df = pd.DataFrame({'id': public_id_col, 'label': public_cluster_labels})
    public_submission_df.to_csv(PUBLIC_SUBMISSION_PATH, index=False)

    # Private
    private_df = pd.read_csv(PRIVATE_DATA_PATH)
    private_id_col = private_df['id']

    private_cluster_labels = multi_stage_angle_clustering(private_df, dataset_name='private')
    
    private_submission_df = pd.DataFrame({'id': private_id_col, 'label': private_cluster_labels})
    private_submission_df.to_csv(PRIVATE_SUBMISSION_PATH, index=False)