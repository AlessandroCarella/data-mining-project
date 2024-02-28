import matplotlib.pyplot as plt
import pandas as pd

def plotSSE (filePath, type):
    # Assuming your data is in a file named 'data.csv'
    data = pd.read_csv(filePath)

    # Extracting values for x and y
    x_values = [int(value.split('=')[-1]) for value in data['clusteringColumn']]
    y_values = data['value']

    # Plotting the data
    plt.figure(figsize=(40, 10))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title('SSE Scores vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE Score')
    plt.xticks(rotation=90)
    plt.xticks(range(min(x_values), max(x_values)+1))
    plt.grid(True)
    plt.savefig("plots/sse " + str (type) + ".png")

plotSSE ("metrics/sseKMeans.csv", "kMeans")
plotSSE ("metrics/sseBisectingKMeans.csv", "bisecting kMeans")