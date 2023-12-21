import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path
import matplotlib.pyplot as plt

def plotDecisionTree(tree, rank=""):

        base_directory='src/classification/dst'
        directoryName= 'treePlots'
        fileName = f"Decision Tree Rank {rank}"
        model_directory = os.path.join(base_directory, directoryName)
        file_path = os.path.join(base_directory, directoryName, fileName + ".png")
        
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            
        figsize = (20, 15)
        plt.figure(figsize=figsize)
        plot_tree(tree, filled=True, feature_names=None, class_names=None, rounded=True)
        plt.title(f"Decision Tree Rank {rank}")

        plt.savefig(file_path)
