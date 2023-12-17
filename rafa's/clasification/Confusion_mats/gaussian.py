# Load the necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the data
df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)

# Separate features and target
X = df.drop('genre', axis=1)
y = df['genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multinomial Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the genre labels for the test data
y_pred = model.predict(X_test)

# Create a dictionary to store confusion matrices for each category
confusion_matrices = {}

# Iterate over the unique categories and create confusion matrices
for category in np.unique(y):
    # Get the indices of the data points belonging to the current category
    category_indices = np.where(y_test == category)[0]

    # Create a confusion matrix for the current category
    confusion_matrix_current_category = confusion_matrix(y_test[category_indices], y_pred[category_indices])

    # Store the confusion matrix in the dictionary
    confusion_matrices[category] = confusion_matrix_current_category

# Print the confusion matrices for each category
for category, confusion_matrix in confusion_matrices.items():
    print(f"Confusion matrix for category {category}:")
    print(confusion_matrix)
