

from classificationUtils import getModelFromPickleFile

def compareModels ():
    models = getModelFromPickleFile ("knn")
    for kValueModel in models:
        pass
        #TODO implement the method to compare the various models
        #this methods prints which value of k is the best for the various models

def getBestKnnModel ():
    pass
    #this method is needed just to get the best model found by the metrics
    #this method should train the classifier on the whole training set, not the train/validation split

def learningCurveForDifferentDatasetSize ():
    #(plot) 
    pass