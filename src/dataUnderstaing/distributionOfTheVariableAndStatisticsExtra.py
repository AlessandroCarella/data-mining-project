import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as path

# Load your dataset
data = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

corr_matrix = data.select_dtypes(include=['int', 'float']).corr()

# Create a larger figure with extra height for better readability at the bottom
plt.figure(figsize=(21, 9))

# Use hierarchical clustering to reorder the correlation matrix and display values
clustermap = sns.clustermap(corr_matrix, cmap='coolwarm', annot=True, figsize=(21, 12), row_cluster=False, col_cluster=False)
clustermap.ax_cbar.remove()

# Create a directory to save files
output_directory = path.join(path.dirname(__file__), 'analysis')

# Save the clustermap image with higher resolution
clustermap.savefig(os.path.join(output_directory, "visualizations", "clustermapWithoutDendograms.png"), dpi=600)

# Close the previous figure
plt.close()

# Create a larger figure with extra height for better readability at the bottom
plt.figure(figsize=(10, 15))

# Adjust the color intensity and range for better visibility
cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(corr_matrix, cmap=cmap, center=0)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Save the heatmap image
plt.savefig(os.path.join(output_directory, "visualizations", "heatmap.png"), dpi=600)



"""Dimensionality Reduction:
If you have too many variables and the visualizations are still too messy, consider using dimensionality reduction techniques like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to project your data into a lower-dimensional space. This can help you visualize the data while preserving important relationships.

python
Copy code
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1])
plt.show()"""