from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from scipy.stats import uniform, randint
from utils import getTrainDatasetPath, getTestDatasetPath, continousAndCategorialFeaturesForClassification

targetVariable= "genre"
dataset = pd.read_csv (getTrainDatasetPath())[continousAndCategorialFeaturesForClassification]

X_train = dataset.copy().drop(targetVariable, axis=1)
Y_train = LabelEncoder().fit_transform(dataset[targetVariable])

param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(2, 20),
    'min_samples_split': uniform(0.1, 0.9),
    'min_samples_leaf': randint(1, 100),
    'ccp_alpha': uniform(0.0, 0.5)
}

testDataset = pd.read_csv (getTestDatasetPath())[continousAndCategorialFeaturesForClassification]
testDataset[targetVariable] = LabelEncoder().fit_transform(testDataset[targetVariable])

X_test = testDataset.drop(targetVariable, axis=1) 
Y_test = testDataset[targetVariable]


dt_classifier = DecisionTreeClassifier()


random_search = RandomizedSearchCV(
    dt_classifier,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
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
print('Train F1-score %s' % f1_score(Y_train, y_train_pred, average=None))
print()

print('Test Accuracy %s' % accuracy_score(Y_test, y_test_pred))
print('Test F1-score %s' % f1_score(Y_test, y_test_pred, average=None))