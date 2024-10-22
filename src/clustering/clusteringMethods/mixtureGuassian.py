import numpy as np
from sklearn.mixture import GaussianMixture
from clusteringMethods.clusteringUtility import columnAlreadyInDf, copyAndScaleDataset, saveMidRunObjectToFile,getMidRunObjectFolderPath
import os.path as path

# n_components:
# Number of Gaussian components (clusters) to identify
# Determines the granularity of clustering

# tol:
# Convergence tolerance for the EM algorithm
# Controls when the algorithm stops iterating
# Smaller values for higher precision, larger values for faster convergence
def mixtureGuassian(df, columnsToUse, n_components=[2, 10], tols=[1e-4]):
    #@AlessandroCarella
    tempDfScal = copyAndScaleDataset (df, columnsToUse)

    columnsNames = []
    for n_component in range (n_components[0], n_components[1]):
        for tol in tols:
            newColumnName = 'mixtureGaussian' + ' ' + 'nComponents=' + ' ' + str (n_component) + ' ' + 'tol=' + str(tol)
            columnsNames.append (newColumnName)
            if not columnAlreadyInDf (newColumnName, df):
                # Create and fit a Gaussian Mixture Model
                gmm = GaussianMixture(n_components=n_component, tol=tol)
                gmm.fit(tempDfScal)

                # Predict the cluster assignments for each data point
                labels = gmm.predict(tempDfScal)
                component_means = gmm.means_
                saveMidRunObjectToFile (component_means, path.join(getMidRunObjectFolderPath(), "(just object) gmmComponent_means "+ newColumnName))

                # Add the cluster assignment as a new column in the DataFrame
                df[newColumnName] = labels

    return df, columnsNames
