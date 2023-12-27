import os.path as path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor

from classificationUtils import continuousFeatures, downsampleDataset, getTrainDatasetPath, getTestDatasetPath, copyAndScaleDataset
from findBestModel import getBestDecisionTreeRegressorModel
from metrics import decisionTreeRegMetrics


def visualize_decision_tree(regressor):
    """
    Visualize the Decision Tree Regressor.

    Parameters:
    - regressor: sklearn.tree.DecisionTreeRegressor
      The Decision Tree Regressor to visualize.
    """
    plt.figure(figsize=(100, 50))
    plot_tree(regressor, filled=True, feature_names=None, rounded=True, fontsize=10)
    plt.title("Decision Tree Regressor")
    plt.savefig(path.join(path.dirname(__file__), "../results", "representation decision tree regressor.png"))

def modelTest (targetVariable="popularity")->(DecisionTreeRegressor, dict):
    columnsToUse = continuousFeatures
    if targetVariable in columnsToUse:
        columnsToUse.remove (targetVariable)
        
    originalDataset = pd.read_csv (getTrainDatasetPath())
    decisionTreeRegressorModel = getBestDecisionTreeRegressorModel (maxDepth=10 , criterion="squared_error", minSampleLeaf=100, minSampleSplit=2)
    decisionTreeRegressorModel.fit (originalDataset[columnsToUse], originalDataset[targetVariable])

    testDataset = pd.read_csv (getTestDatasetPath())
    predictions = decisionTreeRegressorModel.predict (testDataset[columnsToUse])

    return decisionTreeRegressorModel, decisionTreeRegMetrics (predictions=predictions, groundTruth=testDataset[targetVariable])

model, metrics = modelTest()
for key, value in metrics.items():
    print (key,":")
    print (value)

visualize_decision_tree (model)