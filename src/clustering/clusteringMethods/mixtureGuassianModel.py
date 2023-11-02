# n_components:
# Number of Gaussian components (clusters) to identify
# Determines the granularity of clustering

# tol:
# Convergence tolerance for the EM algorithm
# Controls when the algorithm stops iterating
# Smaller values for higher precision, larger values for faster convergence
def mixtureGuassian(df, columnsToUse, n_components=[2, 10], tols=[1e-4]):
    #@AlessandroCarella
    import numpy as np
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt 
    from sklearn.preprocessing import StandardScaler

    #create a copy of the dataset to select only certain features
    tempDf = df.copy()
    tempDf = tempDf [columnsToUse]

    #scale the temp dataset to use the clustering algorithm
    scaler = StandardScaler()
    scaler.fit(tempDf)
    tempDfScal = scaler.transform(tempDf)

    columnsNames = []
    for n_component in range (n_components[0], n_components[1]):
        for tol in tols:
            # Create and fit a Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_component, tol=tol)
            gmm.fit(tempDfScal)

            # Predict the cluster assignments for each data point
            labels = gmm.predict(tempDfScal)

            # Add the cluster assignment as a new column in the DataFrame
            df['mixtureGaussian' + ' ' + 'nComponents=' + ' ' + str (n_component) + ' ' + 'tol=' + str(tol)] = labels
            columnsNames.append ('mixtureGaussian' + ' ' + 'nComponents=' + ' ' + str (n_component) + ' ' + 'tol=' + str(tol))

    return df, columnsNames
