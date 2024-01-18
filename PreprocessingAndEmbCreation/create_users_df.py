import pandas as pd
import pickle
from tqdm import tqdm
import json
import os
import sys
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))

from config.data_config import raw_data_path_config, processed_data_path_config

datasets = ['parler', 'echo', 'gab']
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default='parler',
                    choices=[e.lower() for e in datasets] + [e.capitalize() for e in datasets])
parser.add_argument("--overwrite", action="store_true", default=False)
args = parser.parse_args()
dataset = args.dataset.lower()
in_path = raw_data_path_config[dataset]
out_path = processed_data_path_config[dataset]
# data_directory = f'Dataset/{dataset.capitalize()}Data/'

with open(out_path['network_nodes'], 'rb') as f:
    nodes_lst = pickle.load(f)
users_path_path = in_path['users_info']

if dataset == 'parler':
    dfs = []
    for f in tqdm(os.listdir(users_path_path)):
        df_all = pd.read_json(os.path.join(users_path_path, f), lines=True, chunksize=100000)
        for df in df_all:
            dfs.append(df.query("`username` in @nodes_lst"))
    users_df = pd.concat(dfs)
elif dataset == 'echo':
    users_df = pd.read_csv(users_path_path, usecols=['screen_name', 'name', 'statuses_count', 'description',
                                                     'followers_count', 'friends_count', 'twitter_id'])

with open(out_path['users_info'], 'wb') as f:
    pickle.dump(users_df, f)
