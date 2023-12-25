import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

base_directory='src/classification/dst/REPORT FINDINGS'
directoryName = 'Genre' #'Mode'
def plotDecisionTree(tree):
        fileName = f"Decision Tree {tree}"
        model_directory = os.path.join(base_directory, directoryName)
        file_path = os.path.join(base_directory, directoryName, fileName + ".png")
        
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            
        figsize = (20, 15)
        plt.figure(figsize=figsize)
        features = [
        "mode",
        "key",
        #"grouped_genres",
        #"time_signature",
        #"duration_ms",
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "n_beats"
        ]
        classes = ['0', '1', '2', '3' ]

        plot_tree(tree,
          feature_names=features,
          class_names=classes,
          rounded=True, 
          filled=True, 
          proportion=True); 

        plt.title(f"Decision Tree")
        plt.savefig(file_path)
        
        
def plotConfusionMatrix(y_test, y_test_pred):
        fileName = "Confusion Matrix"
        file_path = os.path.join(base_directory, directoryName, fileName + ".png")
        
        cf = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cf, annot=True, fmt='d', cbar=False, cmap="Blues",
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], vmin=0, vmax=cf.max())
    
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(file_path)
        