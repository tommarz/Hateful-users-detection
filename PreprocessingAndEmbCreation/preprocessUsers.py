# import json
# import random
# import os
# import sys
# import numpy as np
# import argparse
#
# f = os.path.dirname(__file__)
# sys.path.append(os.path.join(f, ".."))
#
# datasets = ['parler', 'echo', 'gab']
# parser = argparse.ArgumentParser()
# parser.add_argument("dataset", type=str, default='parler',
#                     choices=[e.lower() for e in datasets] + [e.capitalize() for e in datasets])
# parser.add_argument("--overwrite", action="store_true", default=False)
# parser.add_argument("--k", type=int, default=5)
# parser.add_argument("--seed", type=int, default=0)
# args = parser.parse_args()
#
# dataset = args.dataset.lower()
#
# data_directory = f'Dataset/{dataset.capitalize()}Data/'
#
#
# def split_json_into_folds(path, k, seed):
#     # Load JSON data
#     file_name = path.split('/')[-1].split('.')[0]
#     with open(path, 'r') as file:
#         data = json.load(file)
#
#     # Shuffle the data with a seed
#     random.seed(seed)
#     random.shuffle(data)
#
#     # Calculate the size of each fold
#     fold_size = np.ceil(len(data) / k).astype(int)
#
#     # Split the data into k folds
#     folds = [data[i:i + fold_size] for i in range(0, len(data), fold_size)]
#
#     # Save each fold as a separate JSON file
#     for i, fold in enumerate(folds):
#         print(f'Saving fold {i + 1} out of {k}... - {len(fold)} samples')
#         fold_path = os.path.join(data_directory, f'{file_name[:-2]}val{i + 1}.json')
#         with open(fold_path, 'w') as file:
#             json.dump(fold, file, indent=4)
#
#     print(f'Successfully split the data into {k} folds.')
#
#
# file_path = os.path.join(data_directory, 'haters.json')
# split_json_into_folds(file_path, args.k, args.seed)
# file_path = os.path.join(data_directory, 'nonhaters.json')
# split_json_into_folds(file_path, args.k, args.seed)