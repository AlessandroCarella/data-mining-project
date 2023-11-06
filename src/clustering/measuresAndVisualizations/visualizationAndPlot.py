import matplotlib.pyplot as plt
import os.path as path

from measuresAndVisualizationsUtils import getPlotsFolderPath, checkSubFoldersExists

def dendogram(df, listOfClusteringColumns):
    #@SaraHoxha
    # TODO: Write the dendogram method
    for clusteringType in listOfClusteringColumns:
        pass
    # plt.savefig("""path""")

def correlationMatrix(df, listOfClusteringColumns):
    #@RafaelUrbina
    # TODO: Write the correlationMatrix method
    for clusteringType in listOfClusteringColumns:
        pass
    # plt.savefig("""path""")

def clusterBarChart(df, listOfClusteringColumns, featuresToPlotOn):
    #@AlessandroCarella
    for clusteringType in listOfClusteringColumns:
        for clusteringColumn in clusteringType:
            if clusteringColumn in df.columns:
                for categoricalColumn in featuresToPlotOn:
                    # Group the data by the categorical column and the cluster column
                    grouped = df.groupby([categoricalColumn, clusteringColumn]).size().unstack().fillna(0)

                    # Create the plot
                    fig, ax = plt.subplots()

                    # Plot the data
                    for cluster in grouped.columns:
                        ax.bar(grouped.index, grouped[cluster], label=f'Cluster {cluster}')

                    # Set plot labels and legend
                    ax.set_xlabel(categoricalColumn)
                    ax.set_ylabel('Count')
                    ax.set_title(f'{categoricalColumn} for {clusteringColumn}')
                    ax.legend()
                    
                    # Set the figure size to 21:9 (3440x1440 pixels)
                    fig.set_size_inches(21, 9)

                    imgPath = path.join(getPlotsFolderPath(), "clusterBarChart", f'{categoricalColumn} for {clusteringColumn}.png')
                    checkSubFoldersExists(imgPath)
                    plt.savefig(imgPath, dpi=100)

def similarityMatrix(df, listOfClusteringColumns):
    #@RafaelUrbina
    # TODO: Write the similarityMatrix method
    for clusteringType in listOfClusteringColumns:
        pass
    # plt.savefig("""path""")