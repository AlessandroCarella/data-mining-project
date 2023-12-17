from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir("../Methods")

df = pd.read_csv("../dataset (missing + split)/R_Norm_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
df_id =  pd.read_csv("../dataset (missing + split)/R_Cat_Prueba_Dataset_nomissing_nofeatur_noutlier.csv", skip_blank_lines=True)
plot_folder = '../results/plots/'

eps_values = [0.3,0.5, 1, 2.5,3,4,5,6,7]
min_samples_values = [3,5,7,10,15, 25,35, 50]

results_labels = {}
results_silhouettes = {}

results_labels['genre'] = df_id['genre']

for eps_val, min_samples_val in itertools.product(eps_values, min_samples_values):
    dbscan_model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    cluster_labels = dbscan_model.fit_predict(df)
    col_name = f"DBSCAN_eps{eps_val}_minpts{min_samples_val}"
    results_labels[col_name] = cluster_labels

    # Calculate silhouette score excluding points with label -1 (noise/outliers)
    valid_indices = np.where(cluster_labels != -1)[0]
    valid_data = df.iloc[valid_indices]
    valid_labels = cluster_labels[valid_indices]
    
    if len(np.unique(valid_labels)) > 1:  # At least 2 clusters required for silhouette
        silhouette = silhouette_score(valid_data, valid_labels)
    else:
        silhouette = None
    
    col_name = f"DBSCAN_eps{eps_val}_minpts{min_samples_val}"
    results_silhouettes[col_name] = {'silhouette_score': silhouette}

# Convert results to a DataFrame
labels_df = pd.DataFrame(results_labels)
silhouettes_df = pd.DataFrame(results_silhouettes)

# Accessing the DataFrame
print("Labels DataFrame:")
print(labels_df.head())

print(labels_df.nunique().to_markdown())


# Concatenate multiple columns into a single series
concatenated = pd.concat([labels_df[col] for col in labels_df.columns])

unique_values = concatenated.unique()
print(unique_values)

silhouettes_df

labels_df["DBSCAN_eps0.5_minpts5"].value_counts()

#Let's save the results
folder_path_labs = ('../results/labels/')
folder_path_silh = ('../results/silhouette/')

#Labels
labels_df.to_csv(folder_path_labs + 'dbscan.csv', index=False)
#Silhouette
silhouettes_df.to_csv(folder_path_silh + 'dbscan.csv', index=False)


filtered_df = labels_df[labels_df['DBSCAN_eps0.3_minpts3'] != -1]

# Perform the groupby operation on the filtered DataFrame
#grouped = labels_df.groupby(['genre','DBSCAN_eps0.5_minpts5']).size().unstack()

grouped = filtered_df.groupby(['genre', 'DBSCAN_eps0.3_minpts3']).size().unstack()

# Plotting the grouped bar chart
colors_part1 = sns.color_palette('tab20', 20)
colors_part2 = sns.color_palette('tab20b', 4)
colors = colors_part1 + colors_part2
grouped.plot(kind='bar', stacked=False, color=colors)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Genre by DBSCAN_eps0.3_minpts3')
plt.legend(title='dbscan')
plt.savefig(plot_folder+'dbscan_grouped_bar_chart.png')  # Specify the desired filename and extension
plt.show()