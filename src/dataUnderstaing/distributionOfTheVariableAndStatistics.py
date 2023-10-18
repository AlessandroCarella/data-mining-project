import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as path
import tqdm as tqdm
import numpy as np

# Load your dataset
data = pd.read_csv(path.join(path.dirname(__file__), "../../dataset (missing + split)/train.csv"))

# Create a directory to save files
output_directory = path.join(path.dirname(__file__), 'analysis')
os.makedirs(output_directory, exist_ok=True)

# Define subdirectories for different types of output
subdirectories = ["histograms", "box_plots", "frequency_tables", "pdfs", "cdfs", "skew_kurtosis", "correlation", "outliers", "data_transformation", "imputation", "statistics", "visualizations", "data_quality"]
for subdir in subdirectories:
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)


# Summary statistics
summary_stats = data.describe()
summary_stats.to_csv(os.path.join(output_directory, "summary_stats.csv"))

"""
# Histograms for numerical variables
for col in data.select_dtypes(include=['int', 'float']):
    data[col].plot(kind='hist')
    plt.title(f'Histogram of {col}')
    plt.savefig(os.path.join(output_directory, "histograms", f"{col}_histogram.png"))
    plt.close()"""

# Box plots
for col in data.select_dtypes(include=['int', 'float']):
    sns.boxplot(x=col, data=data)
    plt.title(f'Box Plot of {col}')
    plt.savefig(os.path.join(output_directory, "box_plots", f"{col}_box_plot.png"))
    plt.close()

# Frequency tables for categorical variables
for col in data.select_dtypes(include=['object']):
    freq_table = data[col].value_counts()
    freq_table.to_csv(os.path.join(output_directory, "frequency_tables", f"{col}_frequency_table.csv"))

# Probability Density Functions (PDFs) with KDE for numerical variables
for col in data.select_dtypes(include=['int', 'float']):
    sns.kdeplot(data[col], shade=True)
    plt.title(f'PDF of {col}')
    plt.savefig(os.path.join(output_directory, "pdfs", f"{col}_pdf.png"))
    plt.close()

"""# Cumulative Distribution Functions (CDFs)
for col in data.select_dtypes(include=['int', 'float']):
    sns.ecdfplot(data[col])
    plt.title(f'CDF of {col}')
    plt.savefig(os.path.join(output_directory, "cdfs", f"{col}_cdf.png"))
    plt.close()
"""
# Skewness and Kurtosis
skewness = data.select_dtypes(include=['int', 'float']).skew()
kurtosis = data.select_dtypes(include=['int', 'float']).kurt()
skewness.to_csv(os.path.join(output_directory, "skew_kurtosis", "skewness.csv"))
kurtosis.to_csv(os.path.join(output_directory, "skew_kurtosis", "kurtosis.csv"))

# Correlation matrix
correlation_matrix = data.select_dtypes(include=['int', 'float']).corr()
correlation_matrix.to_csv(os.path.join(output_directory, "correlation", "correlation_matrix.csv"))

# Outlier Detection (using Z-scores as an example)
numerical_data = data.select_dtypes(include=['int', 'float'])
z_scores = (numerical_data - numerical_data.mean()) / numerical_data.std()
outliers = (z_scores.abs() > 3).any(axis=1)
outliers.to_csv(os.path.join(output_directory, "outliers", "outliers.csv"))

# Data Quality Assessment
missing_values = data.isnull().sum()
missing_values.to_csv(os.path.join(output_directory, "data_quality", "missing_values.csv"))

"""# Data Transformation (e.g., logarithm)
log_transformed = data['numeric_column'].apply(lambda x: np.log(x + 1))
log_transformed.to_csv(os.path.join(output_directory, "data_transformation", "log_transformed.csv"))"""

# Data Imputation (e.g., filling missing values with the mean) for numeric columns
data.select_dtypes(include=['int', 'float']).fillna(numerical_data.mean(), inplace=True)

"""# Statistical Tests (e.g., t-test)
from scipy.stats import ttest_ind
group1 = data[data['group'] == 'A']['value_column']
group2 = data[data['group'] == 'B']['value_column']
t_stat, p_value = ttest_ind(group1, group2)"""

# Data Visualization (scatter plot, pair plot, heatmap, etc.)
sns.pairplot(data)
plt.savefig(os.path.join(output_directory, "visualizations", "pairplot.png"))
plt.close()

sns.heatmap(correlation_matrix, annot=True)
plt.savefig(os.path.join(output_directory, "visualizations", "correlation_heatmap.png"))
plt.close()
