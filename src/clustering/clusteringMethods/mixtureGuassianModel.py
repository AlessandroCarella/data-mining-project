# n_components:
# Number of Gaussian components (clusters) to identify
# Determines the granularity of clustering

# tol:
# Convergence tolerance for the EM algorithm
# Controls when the algorithm stops iterating
# Smaller values for higher precision, larger values for faster convergence
def mixtureGuassian(df, columnsToUse, n_components=10, tol=1e-4):
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

    # Create and fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, tol=tol)
    gmm.fit(tempDf)

    # Predict the cluster assignments for each data point
    labels = gmm.predict(tempDf)

    # Add the cluster assignment as a new column in the DataFrame
    df['mixtureGaussian'] = labels

    return df, "mixtureGaussian"
