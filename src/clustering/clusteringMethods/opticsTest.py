import pandas as pd
import os.path as path

from dbscan import optics

def downsample_dataset(dataset, n):
    """
    Downsample a dataset to have n rows.

    Parameters:
    - dataset: pandas DataFrame, the input dataset.
    - n: int, the number of rows for the downsampled dataset.

    Returns:
    - downsampled_dataset: pandas DataFrame, the downsampled dataset.
    """
    # Check if the requested number of rows is greater than the dataset size
    if n >= len(dataset):
        return dataset  # No need to downsample

    # Use sample method to randomly select n rows from the dataset
    downsampled_dataset = dataset.sample(n, random_state=42)  # You can change the random_state if you want reproducibility

    return downsampled_dataset

def useOpticsWithDownsampledDataset (downsampledDataset: pd.DataFrame):
    continuousFeatures = [
        "duration_ms",
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "n_beats"
    ]

    downsampledDatasetWithOptics, _ = optics (downsampledDataset, continuousFeatures, min_samples=[1, 20], xi=[0.01, 0.05, 0.1], min_cluster_size=[0.01, 0.05, 0.1])

    downsampledDatasetWithOptics.to_csv (path.join(path.dirname(path.abspath (__file__)), "Dataset_nomissing_nofeatur_noutlier_noinconsistencies with optics.csv"))


originalDatasetPath = path.join(path.dirname(path.abspath (__file__)), "../../../dataset (missing + split)/Dataset_nomissing_nofeatur_noutlier_noinconsistencies.csv")
originalDataset = pd.read_csv (originalDatasetPath)

useOpticsWithDownsampledDataset (downsample_dataset (originalDataset, 5000))