import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import pickle
import networkx as nx
from tqdm import tqdm
import os
import sys
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))

from config.data_config import raw_data_path_config, processed_data_path_config, processed_data_output_dir

datasets = ['parler', 'echo', 'gab']

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default='parler',
                    choices=[e.lower() for e in datasets] + [e.capitalize() for e in datasets])
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--radius", type=int, default=1)
# parser.add_argument("--min_weight", type=int, nargs='+', default=[1, 1, 1])
parser.add_argument("--vector_size", type=int, default=100)
# parser.add_argument("--interactions", type=str, nargs='+', default=['repost', 'reply', 'mention'])
parser.add_argument("--multigraph", action="store_true", default=True)
parser.add_argument("--repost_weight", type=int, default=1)
parser.add_argument("--reply_weight", type=int, default=1)
parser.add_argument("--mention_weight", type=int, default=1)
args = parser.parse_args()

min_weight = [args.repost_weight, args.reply_weight, args.mention_weight]

dataset = args.dataset.lower()

in_path = raw_data_path_config[dataset]
out_path = processed_data_path_config[dataset]

data_directory = processed_data_output_dir[dataset]

input_graph_path = out_path['network']
output_graph_path = out_path['ego_network']
doc_vectors_path = os.path.join(data_directory, f'Doc2Vec{args.vector_size}.p')

logging.info("Start loading labeled users")
labeled_users_df = pd.read_csv(
    in_path['users_labels'], sep='\t',
    index_col=0)
labeled_nodes = labeled_users_df.index.astype(str).tolist()

logging.info("Started loading doc vectors")
if dataset == 'parler':
    with open(out_path['users_with_posts'], 'rb') as handle:
        doc_vectors = pickle.load(handle)
else:
    with open(doc_vectors_path, 'rb') as handle:
        doc_vectors = pickle.load(handle).keys()

logging.info("Started loading graph")
logging.debug('reposts key is 0, replies key is 1, mentions key is 2')
G = nx.read_weighted_edgelist(input_graph_path, delimiter='\t', create_using=nx.MultiDiGraph)
logging.info("Finished loading graph")
logging.info("Number of nodes: %d", G.number_of_nodes())
logging.info("Number of edges: %d", G.number_of_edges())
logging.info("Number of labeled nodes: %d", len(set(labeled_nodes).intersection(G.nodes)))
logging.info("Number of nodes with doc2vec vectors: %d", len(set(doc_vectors).intersection(G.nodes)))

logging.info("Started creating ego network of radius %d", args.radius)
nbrs_set = set(labeled_nodes)
missing_labeled_users = []
for node in tqdm(labeled_nodes):
    if node not in G.nodes:
        missing_labeled_users.append(node)
        continue
    for d in range(args.radius + 1):
        nbrs_set.update(nx.descendants_at_distance(G, node, d))
logging.info("Number of missing labeled users in the ego-network: %d", len(missing_labeled_users))
proximity_list = list(nbrs_set.intersection(set(doc_vectors)))

H = G.subgraph(proximity_list).copy()
logging.info("Number of nodes: %d", H.number_of_nodes())
logging.info("Number of edges: %d", H.number_of_edges())
logging.info("Number of labeled nodes: %d", len(set(labeled_nodes).intersection(H.nodes)))
logging.info("Number of nodes with doc2vec vectors: %d", len(set(doc_vectors).intersection(H.nodes)))
H.remove_nodes_from(missing_labeled_users)

logging.info("Started removing self-loops and isolates")
H.remove_edges_from(nx.selfloop_edges(H))
H.remove_nodes_from(list(nx.isolates(H)))
H.add_nodes_from(labeled_nodes)
logging.info("Number of nodes: %d", H.number_of_nodes())
logging.info("Number of edges: %d", H.number_of_edges())
logging.info("Number of labeled nodes: %d", len(set(labeled_nodes).intersection(H.nodes)))
logging.info("Number of nodes with doc2vec vectors: %d", len(set(doc_vectors).intersection(H.nodes)))

logging.info("Started saving ego network")
nx.write_weighted_edgelist(H, os.path.join(data_directory, "network_with_docs_no_isolates.weighted.edgelist.gz"),
                           delimiter='\t')
logging.info("Finished saving ego network")

# logging.info("Started filtering out edges with weight < %d", min(min_weight))
# filtered_edges = [(u, v, k, attr) for u, v, k, attr in tqdm(H.edges(keys=True, data=True)) if
#                   attr['weight'] >= min_weight[k]]

# G_filtered = nx.edge_subgraph(G, filtered_edges).copy()
#
# subgraph = nx.subgraph_view(G, filter_node=lambda n: n in doc_vectors.keys() and n in proximity_list,
#                             filter_edge=lambda u, v, k: (u, v, k) in filtered_edges)
# H = nx.subgraph(subgraph, proximity_list)
# H = nx.MultiDiGraph(subgraph)
# H = nx.subgraph(G, proximity_list)
# nodes_list = list(subgraph.nodes)

# H_merged = nx.DiGraph()
# logging.info("Started merging edges")
# for u, v, k, attr in tqdm(filtered_edges):
#     w = attr['weight'] if 'weight' in attr else 1.0
#     if H_merged.has_edge(u, v):
#         H_merged[u][v]['weight'] += w
#     else:
#         H_merged.add_edge(u, v, weight=w)
# H_merged.add_nodes_from(labeled_nodes)
# # logging.info("Finished merging edges")
# logging.info("Finished creating ego network of radius %d", args.radius)
# logging.info("Number of nodes: %d", H_merged.number_of_nodes())
# logging.info("Number of edges: %d", H_merged.number_of_edges())
# logging.info("Number of labeled nodes: %d", len(set(labeled_nodes).intersection(H_merged.nodes)))
# logging.info("Number of nodes with doc2vec vectors: %d", len(set(doc_vectors).intersection(H_merged.nodes)))
# logging.info("Number of labeled nodes with doc2vec vectors: %d",
#              len(set(labeled_nodes).intersection(doc_vectors)))
# logging.info("Number of isolated nodes: %d", len(list(nx.isolates(H_merged))))
# logging.info("Number of isolated labeled nodes: %d", len(set(nx.isolates(H_merged)).intersection(labeled_nodes)))

# logging.info("Finished loading doc vectors")


# logging.info("Start filtering out edges with weight < %d", args.min_weight)
# logging.info("Started filtering out edges with weight < %d and nodes without doc2vec vectors", args.min_weight)
# H_filtered = nx.subgraph_view(H_merged, filter_node=lambda n: n in doc_vectors.keys(),
#                               filter_edge=lambda e: e['weight'] >= args.min_weight)
# logging.info("Finished filtering out edges with weight < %d", args.min_weight)

# logging.info("Started filtering out nodes without doc vectors")
# nodes_with_docs = list(set.intersection(*map(set, [doc_vectors.keys(), H_filtered.nodes])))
# logging.info("Number of nodes with doc vectors: %d", len(nodes_with_docs))
# # logging.info("Finished filtering out nodes without doc vectors")
#
# logging.info("Start saving ego network")
# # H_filtered_with_docs = H_filtered.subgraph(nodes_with_docs)
# nx.write_weighted_edgelist(H_merged.reverse(), output_graph_path, delimiter='\t')
# logging.info("Done")

# labeled_users = list(set.intersection(*map(set, [nodes_with_docs, labeled_nodes])))

# users_df = pd.read_pickle(os.path.join(data_directory, 'users_df.p')).sort_values(
#     ['username', 'comments', 'posts', 'likes']).drop_duplicates(
#     subset=['username'], keep='last').set_index('username')
#
# users_with_info = set(users_df.index)
# len(users_with_info)
#
# users_df['days_since_joined'] = (
#         pd.Timestamp.today() - pd.to_datetime(users_df['joined'].astype(int).astype(str))).dt.days
#
# users_df = users_df.reset_index().sort_values(
#     ['username', 'days_since_joined', 'posts', 'likes', 'comments']).drop_duplicates(subset=['username'],
#                                                                                      keep='last').set_index('username')
#
# parler_users_intersection = list(set.intersection(*map(set, [nodes_with_docs, users_with_info])))
#
# feats_to_uses = ['days_since_joined', 'user_followers', 'user_following', 'comments', 'posts', 'likes']
#
# scaler = StandardScaler()
# scaler.set_output(transform='pandas')
#
# users_df_transformed = scaler.fit_transform(users_df.loc[parler_users_intersection, feats_to_uses])
#
# users_df_transformed.to_csv(os.path.join(data_directory, 'users_info.tsv'), sep='\t')
#
# labeled_users_new = set(labeled_users).intersection(set(parler_users_intersection))
#
# parler_labeled_users_for_gnn = labeled_users_df.loc[list(labeled_users)]
#
# parler_labeled_users_for_gnn.to_csv('parler_users_2_labels_for_gnn.tsv', sep='\t')
#
# non_haters = parler_labeled_users_for_gnn.query('`label`==0').index.tolist()
# with open(os.path.join(data_directory, 'nonhaters.json'), 'w') as f:
#     json.dump(non_haters, f)
# haters = parler_labeled_users_for_gnn.query('`label`==1').index.tolist()
# with open(os.path.join(data_directory, 'haters.json'), 'w') as f:
#     json.dump(haters, f)
