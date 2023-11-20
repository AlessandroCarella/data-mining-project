

from classificationUtils import getModelFromPickleFile

def compareModels ():
    models = getModelFromPickleFile ("knn")
    for kValueModel in models:
        pass
        #TODO implement the method to compare the various models

def getBestKnnModel ():
    pass
    #this method is needed just to get the best model found by the metrics

def learningCurveForDifferentDatasetSize ():
    #(plot) 
    pass