import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as path

# Sample dataset (replace with your dataset)
data = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

# Create a scatter matrix
sns.set(style="ticks")
scatter_matrix = sns.pairplot(data)

# Specify the output folder
output_folder = path.join(path.dirname(__file__), "scatterMatrixDivided")

# Create the output folder if it doesn't exist
import os
os.makedirs(output_folder, exist_ok=True)

# Suppress the RuntimeWarning from NumPy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Loop through the upper triangle of the scatter plot grid
    for i, j in zip(*plt.np.triu_indices_from(scatter_matrix.axes, 1)):
        scatter_plot = scatter_matrix.axes[i, j]
        scatter_plot.get_figure().savefig(f"{output_folder}/scatter_plot_{i}_{j}.png")
