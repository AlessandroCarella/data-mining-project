import pandas as pd
import re
import matplotlib.pyplot as plt

def split_entropy_dataset(entropy, clustering_types)->dict:
    # Initialize an empty list to store the split datasets
    split_datasets = {}

    # Iterate through each clustering type
    for clustering_type in clustering_types:
        # Filter the dataset based on whether the clustering type is present in the 'clustering type' column
        subset = entropy[entropy['clustering type'].str.contains(clustering_type)]

        # Append the filtered subset to the list
        split_datasets[clustering_type] = subset

    return split_datasets

def getListOfValues (df):
    names = list(df["clustering type"])
    thresholdKetc = []
    for name in names:
        thresholdKetc.append (int(re.search(r'\d+$', name).group()))

    return thresholdKetc

def extract_valuesForHierarchicalGroupAverage(input_string):
    # Define a regular expression pattern to capture the threshold value and criterion
    pattern = r'threshold(\d+)\s+criterion=(\w+)'

    # Use re.search to find the pattern in the input string
    match = re.search(pattern, input_string)

    if match:
        # Extract the threshold value and criterion from the matched groups
        threshold_value = int(match.group(1))
        criterion = match.group(2)

        return threshold_value, criterion

def extract_valuesForMixtureGaussian (input_string):
    # Define a regular expression pattern to extract values
    pattern = r'nComponents= (\d+) tol=([0-9eE.-]+)'

    # Use re.search to find the pattern in the string
    match = re.search(pattern, input_string)

    # Extract values from the matched groups
    n_components = match.group(1)
    tol = match.group(2)

    return n_components, tol

def extract_valuesForDBSCAN (input_string):
    # Define a regular expression pattern to extract values
    pattern = r'min_samples=(\d+) eps=([0-9eE.-]+)'

    # Use re.search to find the pattern in the string
    match = re.search(pattern, input_string)

    # Extract values from the matched groups
    min_samples = match.group(1)
    eps = match.group(2)

    return min_samples, eps

def plot (clustering_type, values, thresholdKetc):
    plt.plot (thresholdKetc, values, marker='o', linestyle='-')
    plt.title('Plot for '+ clustering_type)
    plt.show()
    plt.close()

def plot2ValuesDicts (clustering_type, category, ):
    plt.plot (thresholdKetc, values, marker='o', linestyle='-')
    plt.title('Plot for '+ clustering_type + ' using ' + category)
    plt.show()
    plt.close()

def dataPreparationAndPlotting (datasets:dict):
    for clustering_type, df in datasets.items():
        if clustering_type != "hierarchicalGroupAverage" and clustering_type != "mixtureGaussian" and clustering_type != "dbscan" and clustering_type != "optics":
            thresholdKetc = getListOfValues (df)
            plot (clustering_type, list(df["value"]), thresholdKetc)
        elif clustering_type == "hierarchicalGroupAverage":
            thresholdsCriterionAndValue = {}
            names = list(df["clustering type"])
            values = list(df["value"])
            for name, value in zip(names, values):
                threshold_value, criterion = extract_valuesForHierarchicalGroupAverage (name)
                if not criterion in thresholdsCriterionAndValue:
                    thresholdsCriterionAndValue [criterion] = []
                thresholdsCriterionAndValue [criterion].append ((threshold_value, value))
            
            for key, value in thresholdsCriterionAndValue.items():
                plot2ValuesDicts (clustering_type, key, value)
        elif clustering_type == "mixtureGaussian":
            nComponentTolAndValue = {}
            names = list(df["clustering type"])
            values = list(df["value"])
            for name, value in zip(names, values):
                n_components, tol = extract_valuesForMixtureGaussian (name)
                if not float(tol) in nComponentTolAndValue:
                    nComponentTolAndValue [float(tol)] = []
                nComponentTolAndValue [float(tol)].append ((int(n_components), value))
        elif clustering_type == "dbscan":
            minSamplesAndEps = {}
            names = list(df["clustering type"])
            values = list(df["value"])
            for name, value in zip(names, values):
                if "hdbscan" not in name:
                    min_samples, eps = extract_valuesForDBSCAN (name)
                    if not int(min_samples) in minSamplesAndEps:
                        minSamplesAndEps [int(min_samples)] = []
                    minSamplesAndEps [int(min_samples)].append ((float(eps), value))

    print (minSamplesAndEps)    

def main ():
    entropy = pd.read_csv ("entropy.csv")

    clusteringsTypes = [
        "hierarchicalCentroidLinkage",
        "hierarchicalCompleteLinkage",
        "hierarchicalSingleLinkage",
        "hierarchicalGroupAverage",
        "kMeans",
        "bisectingKmeans",
        "kModes",
        "mixtureGaussian",
        "hdbscan",
        "dbscan",
        "optics"
    ]

    datasets = split_entropy_dataset (entropy, clusteringsTypes)

    dataPreparationAndPlotting (datasets)
    



main ()