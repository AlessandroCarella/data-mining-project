# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os

#set wd
os.chdir("../methods")

#load ds normalized
df = pd.read_csv("../dataset (missing + split)/R_LogisticReg_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

#Set y,x
frame = df
df_orig = frame[frame.columns.difference(['explicit','genre'])]
X = df_orig.values
y = np.array(frame['explicit'].values)

#Check number of categories in genre.
np.unique(y, return_counts=True)

random_state = 70

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=random_state,stratify=y
)


# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))



# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

df['explicit'].value_counts()


# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 3], y=X_test[:, 4], hue=y_test, palette={
                0: 'blue', 1: 'red'}, marker='o')
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(
    accuracy * 100))
plt.legend(title="Explicit", loc="upper right")
plt.show()


