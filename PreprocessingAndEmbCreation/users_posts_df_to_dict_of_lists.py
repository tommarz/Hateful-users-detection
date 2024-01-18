import pandas as pd
import pickle
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


with open(out_path['users_posts'], 'rb') as f:
    df_all = pickle.load(f)

# users_posts_list = df_all.groupby('username')['text'].apply(list).to_dict()
# with open(out_path['users_posts_list'], 'wb') as f:
#     pickle.dump(users_posts_list, f)

users_posts_proc_list = df_all.groupby('username')['text'].apply(list).to_dict()
with open(out_path['users_posts_proc_list'], 'wb') as f:
    pickle.dump(users_posts_proc_list, f)