This is me just trying to collect my thoughts in one place, my mind is a scary place.

So:

# MULTI CLASS CLASSIFICATION
# TARGET VARIABLE = "GENRE"
## Scenario 1:

Doing hyperparameter tuning through Grid Search and KF, not allowing "None" depth,  and using these features
[
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
Model key: DST with criterion:gini, maxDepth:9, minSampleLeaf:6, minSampleSplit:0.002, splitNumber:7, ccp_alpha:0.0, metrics:{'accuracyScore': 0.49066666666666664, 'f1Score': 0.48810293354615863, 'precisionScore': 0.5000136447686478, 'recallScore': 0.49066666666666664}

1. Train Accuracy **0.5188**
2. Train F1-score 0.5201761827952048
3. Train Precision Score 0.5316504102129872
4. Train Recall Score 0.5188

1. Test Accuracy **0.4482**
2. Test F1-score 0.4505221520093799
3. Test Precision Score 0.4646942415986265
4. Test Recall Score 0.4482

### Notes
It looks okay, not much difference between train and test. 

## Scenario 2:

Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth,  and using these features
[
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
Best Decision Tree Model:
Details: {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 6, 'min_samples_split': 4, 'splitNumber': 1, 'ccp_alpha': 0.0, 'metrics': {'accuracyScore': 0.5013333333333333, 'f1Score': 0.4984253889135204, 'precisionScore': 0.4993090870494506, 'recallScore': 0.5013333333333333}, 'model': DecisionTreeClassifier(min_samples_leaf=6, min_samples_split=4)}
Model key: DST with criterion:gini, maxDepth:None, minSampleLeaf:6, minSampleSplit:4, splitNumber:1, ccp_alpha:0.0, metrics:{'accuracyScore': 0.5013333333333333, 'f1Score': 0.4984253889135204, 'precisionScore': 0.4993090870494506, 'recallScore': 0.5013333333333333}

Train Accuracy **0.6802666666666667**
Train F1-score 0.6801085996073379
Train Precision Score 0.6836207086528381
Train Recall Score 0.6802666666666667

Test Accuracy **0.4554**
Test F1-score 0.456377989227202
Test Precision Score 0.46087909836073176
Test Recall Score 0.4554

### Notes: 
Graph too complex, bigger difference between train & test -> overfitting.

## Scenario 3:
Doing no hyperparameter tuning  and using all these features
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
Best Decision Tree Model:
Details: {'splitNumber': 1, 'metrics': {'accuracyScore': 0.4786666666666667, 'f1Score': 0.477005882187021, 'precisionScore': 0.4780827468482218, 'recallScore': 0.4786666666666667}, 'model': DecisionTreeClassifier()}
Model key: NO TUNING DST with splitNumber:1, metrics:{'accuracyScore': 0.4786666666666667, 'f1Score': 0.477005882187021, 'precisionScore': 0.4780827468482218, 'recallScore': 0.4786666666666667}

Train Accuracy **0.9714666666666667**
Train F1-score 0.971471949104446
Train Precision Score 0.9715213755534157
Train Recall Score 0.9714666666666667

Test Accuracy **0.4326**
Test F1-score 0.4326352456573069
Test Precision Score 0.43337126577445895
Test Recall Score 0.4326

### Notes:
Too much difference between train & test -> overfitting. Graph too complex.

## Scenario 4 Random:
Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth, using RandomizedSearchCV  and using these features
[
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
DecisionTreeClassifier(ccp_alpha=0.0020584494295802446, criterion='entropy',
                       max_depth=8, min_samples_leaf=2, min_samples_split=0.01)
                       
Train Accuracy **0.46826666666666666**
Train F1-score 0.4614843291218691

Test Accuracy **0.43**
Test F1-score 0.4239233431437542

Classification Report

              precision    recall  f1-score   support

           0       0.25      0.16      0.19       250
           1       0.66      0.70      0.68       250
           2       0.37      0.47      0.42       250
           3       0.31      0.33      0.32       250
           4       0.38      0.18      0.25       250
           5       0.56      0.56      0.56       250
           6       0.45      0.38      0.41       250
           7       0.44      0.54      0.48       250
           8       0.36      0.42      0.39       250
           9       0.41      0.36      0.39       250
          10       0.28      0.30      0.29       250
          11       0.34      0.36      0.35       250
          12       0.56      0.54      0.55       250
          13       0.41      0.37      0.39       250
          14       0.37      0.32      0.34       250
          15       0.37      0.43      0.40       250
          16       0.70      0.69      0.69       250
          17       0.21      0.25      0.23       250
          18       0.62      0.80      0.70       250
          19       0.48      0.45      0.46       250

    accuracy                           0.43      5000
   macro avg       0.43      0.43      0.42      5000
weighted avg       0.43      0.43      0.42      5000

### Notes:
Not much difference between train & test. Graph not complex, even less than Scenario 1.

## Scenario 5 Random:
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
DecisionTreeClassifier(ccp_alpha=0.0020584494295802446, criterion='entropy',
                       max_depth=8, min_samples_leaf=2, min_samples_split=0.01)

Train Accuracy **0.4816666666666667**
Train F1-score 0.4766607782084671

Test Accuracy **0.4528**
Test F1-score 0.45080154571736586

Classification Report

              precision    recall  f1-score   support

           0       0.27      0.15      0.19       250
           1       0.66      0.70      0.68       250
           2       0.34      0.44      0.38       250
           3       0.33      0.42      0.37       250
           4       0.25      0.22      0.23       250
           5       0.61      0.52      0.56       250
           6       0.56      0.44      0.49       250
           7       0.42      0.50      0.46       250
           8       0.51      0.43      0.47       250
           9       0.41      0.42      0.42       250
          10       0.41      0.26      0.31       250
          11       0.34      0.36      0.35       250
          12       0.54      0.70      0.61       250
          13       0.50      0.44      0.47       250
          14       0.46      0.47      0.46       250
          15       0.30      0.50      0.37       250
          16       0.75      0.64      0.69       250
          17       0.30      0.26      0.28       250
          18       0.77      0.69      0.73       250
          19       0.48      0.50      0.49       250

    accuracy                           0.45      5000
   macro avg       0.46      0.45      0.45      5000
weighted avg       0.46      0.45      0.45      5000

### Features Importance:
duration_ms 0.23503674319330495
acousticness 0.1871459110022447
speechiness 0.16271662434798798
popularity 0.1462553170090014
time_signature 0.08075669841943142
danceability 0.07851840322762624
loudness 0.03400904467383511
liveness 0.030294067814722547
energy 0.029145201151355295
valence 0.011696896751543282
tempo 0.0022304872706650357
instrumentalness 0.0021946051382821445
mode 0.0
key 0.0
genre 0.0

### Notes:
Very similar to Scenario 4 in terms of graph, but better accuracy when I include all features. 
Did a Confusion Matrix for this one, as till now it look

## Scenario 6 Random:
Doing hyperparameter tuning through Grid Search and KF, and allowing "None" depth, using RandomizedSearchCV  and using these features. I removed the ones with least importance according to Features Importance of Scenario 5.
continousAndCategorialFeaturesForClassificationForNoTuning = [
    #"mode",
    #"key",
    "genre",
    "time_signature",
    "duration_ms",
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    #"instrumentalness",
    "liveness",
    "valence",
    #"tempo",
    "n_beats"
]

### Results: 
DecisionTreeClassifier(ccp_alpha=0.0020584494295802446, criterion='entropy',
                       max_depth=8, min_samples_leaf=2, min_samples_split=0.01)
Train Accuracy **0.4524**
Train F1-score 0.4449498287379101

Test Accuracy **0.417**
Test F1-score 0.4072528512822606

Classification Report

              precision    recall  f1-score   support

           0       0.21      0.16      0.18       250
           1       0.58      0.67      0.62       250
           2       0.50      0.50      0.50       250
           3       0.29      0.38      0.33       250
           4       0.29      0.21      0.24       250
           5       0.50      0.56      0.53       250
           6       0.46      0.31      0.37       250
           7       0.43      0.57      0.49       250
           8       0.36      0.23      0.28       250
           9       0.31      0.28      0.29       250
          10       0.28      0.25      0.26       250
          11       0.34      0.29      0.31       250
          12       0.55      0.62      0.58       250
          13       0.41      0.32      0.36       250
          14       0.41      0.49      0.45       250
          15       0.36      0.54      0.43       250
          16       0.67      0.72      0.69       250
          17       0.29      0.24      0.26       250
          18       0.47      0.62      0.54       250
          19       0.46      0.39      0.42       250

    accuracy                           0.42      5000
   macro avg       0.41      0.42      0.41      5000
weighted avg       0.41      0.42      0.41      5000

### Notes: 
Less accuracy than Scenario 5 so not really worth it.


# TARGET VARIABLE = "GROUPED_GENRES"
# Scenario 6 Grouped Genres:


## Results:
Details: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_leaf': 0.1, 'min_samples_split': 0.05, 'splitNumber': 18, 'ccp_alpha': 0.01, 'metrics': {'accuracyScore': 0.688, 'f1Score': 0.6829508616177088, 'precisionScore': 0.6921920946387429, 'recallScore': 0.688}, 'model': DecisionTreeClassifier(ccp_alpha=0.01, criterion='entropy', max_depth=2,
                       min_samples_leaf=0.1, min_samples_split=0.05)}
Model key: DST with criterion:entropy, maxDepth:2, minSampleLeaf:0.1, minSampleSplit:0.05, splitNumber:18, ccp_alpha:0.01, metrics:{'accuracyScore': 0.688, 'f1Score': 0.6829508616177088, 'precisionScore': 0.6921920946387429, 'recallScore': 0.688}

Train Accuracy **0.6626**
Train F1-score 0.6592883558878498
Train Precision Score 0.6682486125819765
Train Recall Score 0.6626

Test Accuracy **0.6566**
Test F1-score 0.6521851778668034
Test Precision Score 0.6604963203588354
Test Recall Score 0.6566

# Scenario 7 Grouped Genres No Tuning:
Using no hyperparameter tuning.

## Results:
Details: {'splitNumber': 1, 'metrics': {'accuracyScore': 0.764, 'f1Score': 0.7653346388680372, 'precisionScore': 0.7678635601068698, 'recallScore': 0.764}, 'model': DecisionTreeClassifier()}
Model key: NO TUNING DST with splitNumber:1, metrics:{'accuracyScore': 0.764, 'f1Score': 0.7653346388680372, 'precisionScore': 0.7678635601068698, 'recallScore': 0.764}

Train Accuracy **0.9864**
Train F1-score 0.9864071402342695
Train Precision Score 0.9864216553689213
Train Recall Score 0.9864

Test Accuracy **0.7238**
Test F1-score 0.7238630977487077
Test Precision Score 0.724496177915007
Test Recall Score 0.7238

# Conclusion:
Definitely 6.