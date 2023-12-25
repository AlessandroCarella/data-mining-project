import os.path as path

def getTrainDataset ():
    return path.join(path.dirname(__file__), "../../../dataset (missing + split)/trainFilledWithoutUselessFeatures.csv")
