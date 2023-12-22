import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import getTrainDatasetPath, continousAndCategorialFeaturesForClassificationForNoTuning

def makeDecisionTreeClassifierDictValue (splitNumber, metrics,model):
    return {
        "splitNumber": splitNumber,
        "metrics": metrics,
        "model": model
    }

def getNoTuningDecisionTreeModel (targetVariable):
    dataset = pd.read_csv (getTrainDatasetPath())[continousAndCategorialFeaturesForClassificationForNoTuning]
    dataset["genre"]= LabelEncoder().fit_transform(dataset["genre"])  
    X = dataset.copy().drop(targetVariable, axis=1)

    Y =  dataset[targetVariable]
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    decisionTreeClassifierDict = {}
                        
    for trainIndex, testIndex in kf.split (X):
        splitNumber = 1
        X_train, X_test = X.iloc[trainIndex], X.iloc[testIndex]
        y_train, y_test = Y.iloc[trainIndex], Y.iloc[testIndex]
        model = DecisionTreeClassifier()
        model.fit (X_train, y_train)

        predictions=model.predict(X_test)
        accuracyScore = accuracy_score(predictions, y_test)
        print('accuracyScore' + str(accuracyScore))
        f1Score = f1_score(predictions, y_test, average="weighted")
        recallScore = recall_score(predictions, y_test, average="weighted")
        precisionScore = precision_score(predictions, y_test, average="weighted")
                                
        metrics = {'accuracyScore':accuracyScore, "f1Score":f1Score, "precisionScore": precisionScore, "recallScore":recallScore}

        decisionTreeRegKey = f"NO TUNING DST with splitNumber:{splitNumber}, metrics:{metrics}"
        decisionTreeClassifierDict[decisionTreeRegKey] = makeDecisionTreeClassifierDictValue (splitNumber, metrics, model)

        print(decisionTreeRegKey)
        splitNumber += 1
    return decisionTreeClassifierDict