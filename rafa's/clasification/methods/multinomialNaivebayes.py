# !pip install scikit-plot
#Load libs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall

#set wd
os.chdir("methods")

#load categorical data
df = pd.read_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

#Set y,x
frame = df
df_orig = frame[frame.columns.difference(['genre'])]
X = df_orig.values
y = np.array(frame['genre'].values)

#Check number of categories in genre.
np.unique(y, return_counts=True)

random_state = 70

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state,stratify=y
)

#partitioning
print(np.unique(y, return_counts=True)[1] / len(y))
print(np.unique(y_train, return_counts=True)[1] / len(y_train))
print(np.unique(y_test, return_counts=True)[1] / len(y_test))


############## Multinomial NB ##################

clf = MultinomialNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred

print(classification_report(y_test, y_pred))

clf.predict_proba(X_test)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
plot_roc(y_test, clf.predict_proba(X_test))
plt.show()
print(roc_auc_score(y_test, clf.predict_proba(X_test), multi_class="ovr", average="macro"))


########### TOTAL CONFUSION MATRIX ################
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True, cmap="Greens")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()


########### PER CLASS CONFUSION MATRIX ################
cm = multilabel_confusion_matrix(y_test, y_pred)
for clas in range(20):

    disp = ConfusionMatrixDisplay(confusion_matrix=cm[clas])
    disp.plot()
    # Set title for each plot
    plt.title(f'Confusion Matrix for Class {clas}')
    plt.show()