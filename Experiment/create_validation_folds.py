import os
from PreprocessingAndEmbCreation.preprocessUsers import split_json_into_folds

dataDirectory = 'Dataset/EchoData'
k = 5  # Number of folds
seed = 0  # Random seed

file_path = os.path.join(dataDirectory, 'haters.json')
split_json_into_folds(file_path, k, seed)
file_path = os.path.join(dataDirectory, 'nonhaters.json')
split_json_into_folds(file_path, k, seed)
