import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fim import fpgrowth
import numpy as np

from featureTransformation import getDatasetWithPatterMiningFeatures

X = getDatasetWithPatterMiningFeatures().values.tolist()

len_r = []
zmin=2
min_sup = 1
max_sup = 20
min_conf = 50
max_conf = 90
for i in range(min_sup, max_sup):  # support
    len_r_wrt_i = []
    for j in range(min_conf, max_conf):  # confidence
        rules = fpgrowth(X, target="r", supp=i, zmin=zmin, conf=j, report="aScl")
        len_r_wrt_i.append(len(rules))

    len_r.append(len_r_wrt_i)
len_r = np.array(len_r)

# Specify the desired values of support and confidence
desired_support = 15
desired_confidence = 60

# Find the indices corresponding to the desired support and confidence values
support_index = desired_support - min_sup
confidence_index = desired_confidence - min_conf

# Access the length of rules at the specified indices
length_of_rules = len_r[support_index, confidence_index]

print(f"The length of rules for support {desired_support} and confidence {desired_confidence} is {length_of_rules}")

sns.heatmap(len_r, cmap="Greens", fmt='g')
plt.yticks(np.arange(0, max_sup-min_sup +1, 5), np.arange(min_sup, max_sup+1,5 ))
plt.xticks(np.arange(0, max_conf-min_conf+1, 5), np.arange(min_conf, max_conf+1, 5))
plt.xlabel("%confidence")
plt.ylabel("%support")
title='Optimization heatmap for rules extraction'
plt.title(title)
plt.grid(True)
plt.savefig(os.path.join('src/patterMining/fpgrowth/charts', title + ".png"))

