Part 2: This is me just trying to collect my thoughts in one place, my mind is a scary place.

So:
# BINARY CLASS CLASSIFICATION
# TARGET VARIABLE = "MODE"

## Scenario 1 Random:
Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth, using RandomizedSearchCV  and using all these features
[
    "mode",
    "key",
    "genre",
    "time_signature",
    "duration_ms",
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

### Results:
DecisionTreeClassifier(ccp_alpha=0.03745401188473625, min_samples_leaf=4,
                       min_samples_split=0.2)
Train Accuracy **0.7407333333333334**
Train F1-score 0.6304077259903232

Test Accuracy **0.7436**
Test F1-score 0.6342520761642578

Classification Report

              precision    recall  f1-score   support

         0.0       1.00      0.00      0.00      1282
         1.0       0.74      1.00      0.85      3718

    accuracy                           0.74      5000
   macro avg       0.87      0.50      0.43      5000
weighted avg       0.81      0.74      0.63      5000

### Notes:
Acceptable scores, weird plot.


## Scenario 2 Random:
Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth, using RandomizedSearchCV  and using  these features
continousAndCategorialFeaturesForClassification = [
    "mode",
    "key",
    "genre",
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

### Results:

DecisionTreeClassifier(ccp_alpha=0.03745401188473625, min_samples_leaf=4,
                       min_samples_split=0.2)
Train Accuracy **0.7407333333333334**
Train F1-score 0.6304077259903232

Test Accuracy **0.7436**
Test F1-score 0.6342520761642578

Classification Report

              precision    recall  f1-score   support

         0.0       1.00      0.00      0.00      1282
         1.0       0.74      1.00      0.85      3718

    accuracy                           0.74      5000
   macro avg       0.87      0.50      0.43      5000
weighted avg       0.81      0.74      0.63      5000

### Notes:
Exactly same as Scenario 1, even though I used less attributes.


## Scenario 3 No Tuning:
Doing no hyperparameter tuning and using all these features
[
    "mode",
    "key",
    "genre",
    "time_signature",
    "duration_ms",
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

### Results:
Details: {'splitNumber': 1, 'metrics': {'accuracyScore': 0.68, 'f1Score': 0.6828067792773674, 'precisionScore': 0.6858332715271083, 'recallScore': 0.68}, 'model': DecisionTreeClassifier()}
Model key: NO TUNING DST with splitNumber:1, metrics:{'accuracyScore': 0.68, 'f1Score': 0.6828067792773674, 'precisionScore': 0.6858332715271083, 'recallScore': 0.68}

Train Accuracy **0.9824666666666667**
Train F1-score 0.9824615304430977
Train Precision Score 0.9824569458960175
Train Recall Score 0.9824666666666667

Test Accuracy **0.4472**
Test F1-score 0.36054540935185225
Test Precision Score 0.5993272882387175
Test Recall Score 0.4472

### Notes:
Definitely overfitting, graph too complex. Not ideal.

## Scenario 4 Tuning:
Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth,  and using  these features
continousAndCategorialFeaturesForClassification = [
    "mode",
    "key",
    "genre",
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

### Results:
Details: {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 0.01, 'min_samples_split': 0.2, 'splitNumber': 1, 'ccp_alpha': 0.0, 'metrics': {'accuracyScore': 0.7613333333333333, 'f1Score': 0.8644965934897804, 'precisionScore': 1.0, 'recallScore': 0.7613333333333333}, 'model': DecisionTreeClassifier(max_depth=2, min_samples_leaf=0.01,
                       min_samples_split=0.2)}

Model key: DST with criterion:gini, maxDepth:2, minSampleLeaf:0.01, minSampleSplit:0.2, splitNumber:1, ccp_alpha:0.0, metrics:{'accuracyScore': 0.7613333333333333, 'f1Score': 0.8644965934897804, 'precisionScore': 1.0, 'recallScore': 0.7613333333333333}

Train Accuracy 0.7407333333333334
Train F1-score 0.6304077259903232
Train Precision Score 0.8079525377777778
Train Recall Score 0.7407333333333334
Test Accuracy 0.4514
Test F1-score 0.2807798814937302
Test Precision Score 0.75236196
Test Recall Score 0.4514

### Notes:
Clear graph, but overfitting.

# CONCLUSION
Randoms are better but they have weird graphs.
