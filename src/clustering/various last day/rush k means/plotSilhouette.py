import matplotlib.pyplot as plt
import pandas as pd

def plotSilhoutte  (filePath, type):
    # Assuming your data is in a file named 'data.csv'
    data = pd.read_csv(filePath)

    # Extracting values for x and y
    x_values = [int(value.split('=')[-1]) for value in data['clusteringColumn']]
    y_values = data['value']

    # Plotting the data
    plt.figure(figsize=(40, 10))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Scores vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(rotation=90)
    plt.xticks(range(min(x_values), max(x_values)+1))
    plt.grid(True)
    plt.savefig("plots/Silhouette " + str (type) + ".png")

plotSilhoutte  ("metrics/silhouetteKMeans.csv", "kMeans")
plotSilhoutte  ("metrics/silhouetteBisectingKMeans.csv", "bisecting kMeans")