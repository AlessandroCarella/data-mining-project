import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as path
from dstModel import getDecisionTreeModel
from noTuningModel import getNoTuningDecisionTreeModel
from dstTest import testModel
from utils import continousAndCategorialFeaturesForClassification, continousAndCategorialFeaturesForClassificationForNoTuning

def compareMetrics(metrics1: dict, metrics2: dict, metrics_to_compare: list = ['accuracyScore', 'f1Score', 'precisionScore', 'recallScore']):
    # Return True if metrics1 are better than metrics2, False otherwise
    # Check if metrics in metrics_to_compare are better in the first model

    conditions = [metrics1.get(metric, float('-inf')) > metrics2.get(metric, float('-inf')) for metric in metrics_to_compare]

    # Return True if all conditions are true, False otherwise
    return all(conditions)

def calculate_overall_score(metrics):
    # Customize the scoring method as per your preference
    return metrics['accuracyScore'] + metrics['f1Score'] + metrics['precisionScore'] + metrics['recallScore']

'''def find_best_decision_tree_models(target_variable, num_best_models=5):
    #decision_tree_classifier_dict = getDecisionTreeModel(targetVariable=target_variable)
    decision_tree_classifier_dict = getNoTuningDecisionTreeModel(targetVariable=target_variable)

    # Sort models based on the overall score
    sorted_models = sorted(
        decision_tree_classifier_dict.items(),
        key=lambda x: calculate_overall_score(x[1]['metrics']),
        reverse=True
    )

    print(f"Top {num_best_models} Decision Tree Models:")
    for i, (model_key, model_details) in enumerate(sorted_models[:num_best_models], start=1):
        print(f"\nRank {i}:")
        print(f"Key: {model_key}")
        print(f"Details: {model_details}")
        plotDecisionTree(model_details['model'], i)
        '''

#find_best_decision_tree_models("genre")
#find_best_decision_tree_models("mode")
#find_best_decision_tree_models("grouped_genres")

def find_best_decision_tree_model(target_variable, tuning = True):
    columnsToUse = continousAndCategorialFeaturesForClassification
    if(tuning == False):
        columnsToUse = continousAndCategorialFeaturesForClassificationForNoTuning
        decision_tree_classifier_dict = getNoTuningDecisionTreeModel(targetVariable=target_variable)
    else:
        decision_tree_classifier_dict = getDecisionTreeModel(targetVariable=target_variable)

    best_model_key = None
    best_metrics = {'accuracyScore': float('-inf'), 'f1Score': float('-inf'), 'precisionScore': float('-inf'), 'recallScore': float('-inf')}

    for model_key, model_details in decision_tree_classifier_dict.items():
        if compareMetrics(model_details['metrics'], best_metrics):
            best_model_key = model_key
            best_metrics = model_details['metrics']

    if best_model_key is not None:
        best_model_details = decision_tree_classifier_dict[best_model_key]
        print(f"Best Decision Tree Model:")
        print(f"Key: {best_model_key}")
        print(f"Details: {best_model_details}")
        testModel(target_variable, model_details['model'], f"Model key: {best_model_key}", columnsToUse)
    else:
        print("No best model found.")

#find_best_decision_tree_model("genre")
#find_best_decision_tree_model("genre", False)
find_best_decision_tree_model("mode")
#find_best_decision_tree_model("mode", False)