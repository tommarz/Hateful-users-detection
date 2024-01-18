import pandas as pd
import pickle
from tqdm import tqdm
import json
import os
import sys
import argparse
import gzip
import logging
from nltk.stem.snowball import SnowballStemmer
from preprocess_doc import stop_words, only_words
from preprocess import tweet_preprocess2

stemmer = SnowballStemmer("english")
import re

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
raw_data_path = raw_data_path_config[dataset]
processed_data_path = processed_data_path_config[dataset]
# data_directory = f'Dataset/{dataset.capitalize()}Data/'

with open(processed_data_path['network_nodes'], 'rb') as f:
    nodes_lst = pickle.load(f)
users_posts_path = raw_data_path['users_posts']


def preprocess(post_text):
    return [stemmer.stem(elem) for elem in re.sub(only_words, '', tweet_preprocess2(post_text)).lower().split(' ')
            if elem not in stop_words]


dfs = []
for f in tqdm(os.listdir(users_posts_path)):
    full_path = os.path.join(users_posts_path, f)
    if dataset == 'parler':
        df = pd.read_pickle(full_path).query("`text`!='' and `username` in @nodes_lst")[[
            'username', 'text', 'createdAtformatted', 'datatype', 'upvotes', 'comments', 'impressions', 'reposts']]
        df['text_proc'] = df['text'].apply(lambda x: preprocess(x))
        dfs.append(df)
    elif dataset == 'echo':
        with gzip.open(full_path, 'r') as fin:
            data = [json.loads(line) for line in fin.readlines()]
            if len(data) == 0:
                continue
            df = pd.DataFrame(data).query('`lang`=="en" and `full_text`!=""').reset_index(drop=True)
            df['full_text'] = df['full_text'].apply(lambda x: preprocess(x))
            if len(df) == 0:
                continue
            df = df.join(pd.json_normalize(df['user']).rename(
                columns={c: f'user_{c}' for c in ['id', 'id_str', 'lang', 'created_at']})).drop('user', axis='columns')[
                ['id', 'full_text', 'retweet_count', 'favorite_count', 'user_id_str']]
            dfs.append(df.query("`user_id_str` in @nodes_lst").rename(
                columns={'user_id_str': 'username', 'full_text': 'text'}))
df_all = pd.concat(dfs, ignore_index=True)

users_posts_dict_proc = df_all.sort_values('createdAtformatted', ascending=False).groupby('username')[
    'text_proc'].apply(list).to_dict()
with open(processed_data_path['users_posts_dict_proc'], 'wb') as f:
    pickle.dump(users_posts_dict_proc, f)

with open(processed_data_path['users_posts_df'], 'wb') as f:
    pickle.dump(df_all, f)