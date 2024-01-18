import os
import sys
import json
import pickle
import argparse
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))
stemmer = SnowballStemmer("english")

from config.data_config import raw_data_path_config, processed_data_path_config, processed_data_output_dir

datasets = ['parler', 'echo', 'gab']
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default='parler',
                    choices=[e.lower() for e in datasets] + [e.capitalize() for e in datasets])
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--vector_size", type=int, default=100)
parser.add_argument("--window", type=int, default=5)
parser.add_argument("--min_count", type=int, default=3)
args = parser.parse_args()

dataset = args.dataset.lower()

processed_data_path = processed_data_path_config[dataset]

data_directory = processed_data_output_dir[dataset]

with open(processed_data_path['user_sentences_proc'], 'rb') as f:
    user_sentences = pickle.load(f)

chosen_users = {user: [sentence for sublist in user_sentences[user] if sublist for sentence in sublist] for user in
                tqdm(user_sentences)}

logging.info("StartDoc2Vec Training")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate([chosen_users[user] for user in chosen_users])]
model = Doc2Vec(documents, vector_size=args.vector_size, window=args.window, min_count=args.min_count, workers=30)
logging.info("Finished Doc2Vec Training")

logging.info("Start Saving Doc2Vec model")
model.save(os.path.join(data_directory, f"doc2vec{model.vector_size}.model"))
logging.info("Saved Doc2Vec model")

logging.info("Start Dumping Doc2Vec user vectors")
user_vectors = {user: model.infer_vector(chosen_users[user]) for user in tqdm(chosen_users)}

with open(os.path.join(data_directory, f'Doc2Vec{model.vector_size}.p'), 'wb') as handle:
    pickle.dump(user_vectors, handle)
logging.info("Finished Dumping Doc2Vec user vectors")
