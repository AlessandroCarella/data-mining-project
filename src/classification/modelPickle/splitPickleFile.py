import os
import pickle

def split_pickle(input_file, output_dir, chunk_size=99e6):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Pickle the data to get the actual size
    pickled_data = pickle.dumps(data)

    # Determine the number of chunks
    num_chunks = int((len(pickled_data) / chunk_size) + 1)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the pickled data into chunks and save each chunk
    for i in range(num_chunks):
        start_idx = int(i * len(pickled_data) / num_chunks)
        end_idx = int((i + 1) * len(pickled_data) / num_chunks)
        chunk_data = pickled_data[start_idx:end_idx]

        output_file = os.path.join(output_dir, f'chunk_{i + 1}.pkl')
        with open(output_file, 'wb') as f:
            f.write(chunk_data)

def read_split_pickle_files(input_dir):
    pass
    #implementation in classificationUtils

def modelNameToModelObject (modelName:str):
    fodlerPath = os.path.join(os.path.dirname(os.path.abspath (__file__)), f"{modelName}SplittedPickle")
    return read_split_pickle_files (fodlerPath)

#code to split the object
split_pickle (
    os.path.join(os.path.dirname(os.path.abspath (__file__)), "knn.pickle"), 
    os.path.join(os.path.dirname(os.path.abspath (__file__)), "knnSplittedPickle")
)

#example of usage
"""knnModel = modelNameToModelObject ("knn")
for key, value in knnModel.items():
    print (key, value)
print ()
"""