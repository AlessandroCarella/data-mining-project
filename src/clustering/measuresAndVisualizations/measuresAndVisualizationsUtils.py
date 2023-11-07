import csv
import os.path as path  
import os
import numpy as np
from scipy.cluster.hierarchy import fcluster


def saveDictToFile (data:dict, filePath, custom_headers=["", "value"]):
    # Open the CSV file for writing
    with open(filePath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the custom headers to the CSV file
        writer.writerow(custom_headers)

        # Write the dictionary data to the CSV file
        for key, value in data.items():
            writer.writerow([key, value])

def getMetricsFolderPath ():
    return path.join (path.dirname(__file__), "../metrics")

def getPlotsFolderPath ():
    return path.join (path.dirname(__file__), "../plots")

def checkSubFoldersExists (path):
    os.makedirs(path, exist_ok=True)

def visualizazionAlreadyExists (imgPath):
    path.exists(imgPath)

def reorderDistanceMatrix(distance_matrix, linkage_matrix):
    # Extract cluster assignments
    clusters = fcluster(linkage_matrix, t=0, criterion='inconsistent')

    # Get the order of data points within each cluster
    order_within_clusters = [np.where(clusters == cluster_id)[0] for cluster_id in np.unique(clusters)]

    # Reorder rows and columns of the distance matrix based on clusters
    sorted_distance_matrix = distance_matrix[np.concatenate(order_within_clusters)][:, np.concatenate(order_within_clusters)]

    return sorted_distance_matrix