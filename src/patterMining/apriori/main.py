import pandas as pd
from fim import apriori
import matplotlib.pyplot as plt
import numpy as np

from featureTransformation import getDatasetWithPatterMiningFeatures

def findCommonHighestSupport(df1, df2):
    # Find the highest support value in each DataFrame
    highest_support_df1 = df1['support'].max()
    highest_support_df2 = df2['support'].max()

    # Find the common highest support value
    common_highest_support = None
    if highest_support_df1 == highest_support_df2:
        common_highest_support = highest_support_df1
    else:
        if highest_support_df1<highest_support_df2:
            common_highest_support = highest_support_df1
        else:
            common_highest_support = highest_support_df2

    return common_highest_support


patterMiningDf = getDatasetWithPatterMiningFeatures().values.tolist()

supp = 20  # 20%
zmin = 2  # minimum number of items per item set
allFrequentItemSets = apriori(patterMiningDf, target="s", supp=supp, zmin=zmin)
#print (pd.DataFrame(allFrequentItemSets, columns=["frequent_itemset", "support"]))

closedFrequentItemSets = apriori(patterMiningDf, target="c", supp=supp, zmin=zmin)
print (pd.DataFrame(closedFrequentItemSets, columns=["frequent_itemset", "support"]))
maximalFrequentItemSets = apriori(patterMiningDf, target="m", supp=supp, zmin=zmin)
print (pd.DataFrame(maximalFrequentItemSets, columns=["frequent_itemset", "support"]))

len_max_it = []
len_cl_it = []
max_supp = findCommonHighestSupport(pd.DataFrame(closedFrequentItemSets, columns=["frequent_itemset", "support"]), pd.DataFrame(maximalFrequentItemSets, columns=["frequent_itemset", "support"])) #depends on output of the closedFrequentItemSets and maximalFrequentItemSets (the highest support in both)
for i in range(2, max_supp):
    len_max_it.append(len(maximalFrequentItemSets))
    len_cl_it.append(len(maximalFrequentItemSets))

plt.plot(np.arange(2, max_supp), len_max_it, label="maximal")
plt.plot(np.arange(2, max_supp), len_cl_it, label="closed")
plt.legend()
plt.xlabel("%support")
plt.ylabel("itemsets")

plt.show()

gensFrequentItemSets = apriori(patterMiningDf, target="g", supp=supp, zmin=zmin)
rulesFrequentItemSets = apriori(patterMiningDf, target="r", supp=supp, zmin=zmin)