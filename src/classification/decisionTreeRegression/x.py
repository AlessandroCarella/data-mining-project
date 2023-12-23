
from classificationUtils import downsampleDataset, getTrainDatasetPath, copyAndScaleDataset, continuousFeatures, modelNameToModelObject

decisionTreeRegFull = modelNameToModelObject ("decisionTreeReg")

output = {}
i = 0
for key, value in decisionTreeRegFull.items():
    output[key] = value
    if i>20:
        break
    i+=1

import pickle
with open ("decisionTreeRegTemp.pickle", "wb") as f:
    pickle.dump(output, f)