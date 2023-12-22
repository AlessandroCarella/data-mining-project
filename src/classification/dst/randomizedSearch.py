from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.stats import uniform
from utils import getTrainDatasetPath, getTestDatasetPath, continousAndCategorialFeaturesForClassification, continousAndCategorialFeaturesForClassificationForNoTuning
from plotTrees import plotDecisionTree, plotConfusionMatrix

targetVariable= "mode" #"genre"
dataset = pd.read_csv (getTrainDatasetPath())[continousAndCategorialFeaturesForClassification]
dataset['genre'] = LabelEncoder().fit_transform(dataset['genre'])
X_train = dataset.copy().drop(targetVariable, axis=1)
Y_train = dataset[targetVariable]

param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 6, 7,8, 9, 10, None],
    'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf': [0.01, 0.05, 0.1, 0.2, 1, 2 ,3 ,4 ,5 , 6],
    'ccp_alpha': uniform(0.0, 0.1)
}

testDataset = pd.read_csv (getTestDatasetPath())[continousAndCategorialFeaturesForClassification]
testDataset['genre'] = LabelEncoder().fit_transform(testDataset['genre'])
X_test = testDataset.drop(targetVariable, axis=1) 
if(targetVariable=="mode"):
    mode_value = testDataset[targetVariable].mode().iloc[0]  # Get the mode value
    testDataset[targetVariable].fillna(mode_value, inplace=True)
Y_test = testDataset[targetVariable]


dt_classifier = DecisionTreeClassifier()


random_search = RandomizedSearchCV(
    dt_classifier,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=33,  # Number of cross-validation folds
    scoring='accuracy',  # Use an appropriate scoring metric
    random_state=42
)

# Fit the randomized search to your data
random_search.fit(X_train, Y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print(best_params)

# Get the best model
best_model = random_search.best_estimator_
print(best_model)

dtp = DecisionTreeClassifier(random_state=0, **random_search.best_params_)

dtp.fit(X_train, Y_train)
y_train_pred = dtp.predict(X_train)
y_test_pred = dtp.predict(X_test)

print('Train Accuracy %s' % accuracy_score(Y_train, y_train_pred))
print('Train F1-score %s' % f1_score(Y_train, y_train_pred, average='weighted'))
print()

print('Test Accuracy %s' % accuracy_score(Y_test, y_test_pred))
print('Test F1-score %s' % f1_score(Y_test, y_test_pred, average='weighted'))

print('\nClassification Report\n')
print(classification_report(Y_test, y_test_pred, zero_division=1))

# Feature Importance
zipped = zip(continousAndCategorialFeaturesForClassificationForNoTuning, dtp.feature_importances_)
zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
for col, imp in zipped:
    print(col, imp)

plotDecisionTree(dtp)
plotConfusionMatrix(Y_test, y_test_pred)

