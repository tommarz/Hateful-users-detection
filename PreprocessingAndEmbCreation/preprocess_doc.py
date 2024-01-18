import os
import re
import sys
import json
import pickle
import argparse
import stop_words
import preprocess
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import logging
import itertools
import multiprocessing

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))
stemmer = SnowballStemmer("english")

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

stop_words = stop_words.get_stop_words('en')
stop_words_2 = ['i', 'me', 'we', 'us', 'you', 'u', 'she', 'her', 'his', 'he', 'him', 'it', 'they', 'them', 'who',
                'which', 'whom', 'whose', 'that', 'this', 'these', 'those', 'anyone', 'someone', 'some', 'all', 'most',
                'himself', 'herself', 'myself', 'itself', 'hers', 'ours', 'yours', 'theirs', 'to', 'in', 'at', 'for',
                'from', 'etc', ' ', ',']
stop_words.extend(stop_words_2)
stop_words.extend(
    ['with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite',
     'towards', 'upon', 'concerning', 'of', 'to', 'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before',
     'between', 'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across', 'behind', 'beyond',
     'plus', 'except', 'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near', 'and', 'or', 'but', 'nor', 'so',
     'for', 'yet', 'after', 'although', 'as', 'as', 'if', 'long', 'because', 'before', 'even', 'if', 'even though',
     'once', 'since', 'so', 'that', 'though', 'till', 'unless', 'until', 'what', 'when', 'whenever', 'wherever',
     'whether', 'while', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'yours', 'his', 'her', 'its', 'ours',
     'their', 'few', 'many', 'little', 'much', 'many', 'lot', 'most', 'some', 'any', 'enough', 'all', 'both', 'half',
     'either', 'neither', 'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite'])
stop_words = list(set(stop_words))
stopword_file = open(os.path.join("PreprocessingAndEmbCreation", "stopword.txt"), 'r')
stop_words.extend([line.rstrip() for line in stopword_file])

only_words = '[^a-z0-9\' ]+'

if dataset != 'echo':
    with open(processed_data_path['users_posts_list'], 'rb') as f:
        users_posts = pickle.load(f)
    if dataset == 'parler':
        with open(processed_data_path['network_nodes'], 'rb') as f:
            users = pickle.load(f)
    else:
        with open(processed_data_path["ego_network_nodes"], 'rb') as f:
            users = pickle.load(f)
    users_posts = {user: users_posts[user] for user in users}
else:
    with open(raw_data_path['users_posts_list'], 'rb') as f:
        users_posts = pickle.load(f)
    users = list(users_posts.keys())
logging.info("Loaded users posts - total users: {}".format(len(users)))


logging.info("Started processing user sentences")
users_sentences = {user: [
    [stemmer.stem(elem) for elem in re.sub(only_words, '', preprocess.tweet_preprocess2(post_text)).lower().split(' ')
     if elem not in stop_words] for post_text in users_posts[user]] for user in tqdm(users_posts)}

logging.info("Processed user sentences")

with open(processed_data_path['user_sentences_proc'], 'wb') as f:
    pickle.dump(users_sentences, f)
logging.info("Saved user sentences")
logging.info("Done")
