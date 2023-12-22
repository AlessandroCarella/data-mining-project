import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

base_directory='src/classification/dst/REPORT FINDINGS'
directoryName= 'Mode' #'Genre'
def plotDecisionTree(tree):
        fileName = f"Decision Tree {tree}"
        model_directory = os.path.join(base_directory, directoryName)
        file_path = os.path.join(base_directory, directoryName, fileName + ".png")
        
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            
        figsize = (20, 15)
        plt.figure(figsize=figsize)
        plot_tree(tree, filled=True, feature_names=None, class_names=None, rounded=True)
        plt.title(f"Decision Tree")

        plt.savefig(file_path)
        
        
def plotConfusionMatrix(y_test, y_test_pred):
        fileName = "Confusion Matrix"
        file_path = os.path.join(base_directory, directoryName, fileName + ".png")
        cf = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cf, annot=True, cmap="Greens")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(file_path)
