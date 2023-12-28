import os
import pandas as pd
from fim import fpgrowth
import matplotlib.pyplot as plt
import numpy as np

from featureTransformation import getDatasetWithPatterMiningFeatures

def findCommonHighestSupport(df1, df2):
    highest_support_df1 = df1['support'].max()
    highest_support_df2 = df2['support'].max()

    # Find the common highest support value
    common_highest_support = None
    if highest_support_df1 == highest_support_df2:
        common_highest_support = highest_support_df1
    else:
        common_highest_support = min(highest_support_df1, highest_support_df2)

    return common_highest_support

supp = 25
zmin = 2 # minimum number of items per item set

X = getDatasetWithPatterMiningFeatures().values.tolist()

allFrequentItemSets = fpgrowth(X, target="s", supp=supp, zmin=zmin,report="S")
print('\nFrequent Item Set\n')
print (pd.DataFrame(allFrequentItemSets, columns=["frequent_itemset", "support"]))
closedFrequentItemSets = fpgrowth(X, target="c", supp=supp, zmin=zmin, report="S")
print('\nClosed Item Set\n')
print (pd.DataFrame(closedFrequentItemSets, columns=["frequent_itemset", "support"]))
maximalFrequentItemSets = fpgrowth(X, target="m", supp=supp, zmin=zmin, report="S")
print('\nMaximal Item Set\n')
print (pd.DataFrame(maximalFrequentItemSets, columns=["frequent_itemset", "support"]))

len_max_it, len_cl_it, len_fr_it, support_thresholds = [], [], [], []
max_supp = findCommonHighestSupport(pd.DataFrame(closedFrequentItemSets, columns=["frequent_itemset", "support"]),pd.DataFrame(maximalFrequentItemSets, columns=["frequent_itemset", "support"]))
print('max is' + str(max_supp))

for i in range(2, int(max_supp)):
    support_thresholds.append(i)
    max_itemsets = fpgrowth(X, target="m", supp=i, zmin=zmin)
    cl_itemsets = fpgrowth(X, target="c", supp=i, zmin=zmin)
    freq_itemsets = fpgrowth(X, target="s", supp=i, zmin=zmin)
    len_max_it.append(len(max_itemsets))
    len_cl_it.append(len(cl_itemsets))
    len_fr_it.append(len(freq_itemsets))


# Plotting as a line chart
plt.plot(support_thresholds, len_max_it, label='Maximal', marker='o')
plt.plot(support_thresholds, len_cl_it, label='Closed', marker='o')
#plt.plot(support_thresholds, len_fr_it, label='Frequent', marker='o')

plt.xlabel('Support Threshold')
plt.ylabel('Number of Itemsets')
title='Closed vs. Maximal Itemsets FP-Growth'
plt.title(title)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join('src/patterMining/fpgrowth/charts', title + ".png"))


conf = 70
rules = fpgrowth(X, target="r", supp=supp, zmin=zmin, conf=conf, report="aScl")
generatedRules = fpgrowth(X, target="g", supp=supp, zmin=zmin, conf=conf, report="aScl")

rules_df = pd.DataFrame(
    rules,
    columns=[
        "consequent",
        "antecedent",
        "abs_support",
        "%_support",
        "confidence",
        "lift",
    ],
)

csv_filename = 'rules_output.csv'
csv_filename_classification = 'classification_rules_output_NotExplicit.csv'

rules_df.sort_values(by="lift", axis=0, ascending=False).to_csv(os.path.join('src/patterMining/fpgrowth/results',csv_filename), index=False)
#classification rules for "explicit" attribute when value is False
rules_df[rules_df["consequent"] == "notExplicit"].sort_values(by="lift", ascending=False).to_csv(os.path.join('src/patterMining/fpgrowth/results',csv_filename_classification), index=False)