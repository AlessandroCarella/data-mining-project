import os.path as path
import pandas as pd

def compareValues(originalValues, filledValues, groundTruthValues, feature):
    total_values = len(originalValues)
    different_values_count_originalFilled = 0
    different_values_count_filledGroundTruth = 0
    different_values_count_originalGroundTruth = 0

    for i in range(total_values):
        if originalValues[i] != filledValues[i]:
            different_values_count_originalFilled += 1
        if originalValues[i] != groundTruthValues[i]:
            different_values_count_filledGroundTruth += 1
        #if originalValues [i] != groundTruthValues [i]:
        #   different_values_count_originalGroundTruth += 1

    percentage_difference_originalFilled = (different_values_count_originalFilled / total_values) * 100
    percentage_difference_filledGroundTruth = (different_values_count_filledGroundTruth / total_values) * 100
    #percentage_difference_originalGroundTruth = (different_values_count_originalGroundTruth / total_values) * 100

    print(f"Percentage of different values: {percentage_difference_originalFilled:.2f}% for", feature, "when comparing original values and filled values")
    print(f"Percentage of different values: {percentage_difference_filledGroundTruth:.2f}% for", feature, "when comparing filled and spotify (or ground truth) values")
    #print(f"Percentage of different values: {percentage_difference_originalGroundTruth:.2f}% for", feature, "when comparing original and spotify (or ground truth) values")

def analyisisOnFilledValues(original, filled, groundTruth, listOfFeatures):
    for feature in listOfFeatures:
        originalValues = list(original[feature])
        filledValues = list(filled[feature])
        groundTruthValues = list(groundTruth[feature])
        compareValues (originalValues, filledValues, groundTruthValues, feature)

def printMissingValues (df):
    # Count missing values in the "mode" column
    missing_mode = df['mode'].isnull().sum()
    # Count missing values in the "time_signature" column
    missing_time_signature = df['time_signature'].isnull().sum()
    print(f"Missing values in 'mode' column: {missing_mode}")
    print(f"Missing values in 'time_signature' column: {missing_time_signature}")

def findTimeSignatureMissingValues(df):
    mode_time_signature = df['time_signature'].mode().iloc[0]

    # Replace null values with the mean value
    df['time_signature'].fillna(mode_time_signature, inplace=True)

    return df

def findModeMissingValues (df):
    #tried mean, mode, median, forward fill and backward fill, linear interpolation
    #and they all return 100% different values from the spotify values (ground truth)

    mean_mode = df['mode'].mean()
    # Replace null values with the mean value
    df['mode'].fillna(mean_mode, inplace=True)

    return df

trainDf = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/train.csv"))

filledTrainDf = findModeMissingValues (trainDf)
filledTrainDf = findTimeSignatureMissingValues (filledTrainDf)

printMissingValues (filledTrainDf)

spotifyDf = pd.read_csv(path.join(path.abspath(path.dirname(__file__)), "../spotifyCheck/train with all values.csv"))
analyisisOnFilledValues (trainDf, filledTrainDf, spotifyDf, ["mode", "time_signature"])

filledTrainDf.to_csv (path.join(path.abspath(path.dirname(__file__)), "../../dataset (missing + split)/trainFilled.csv"), index = False)
