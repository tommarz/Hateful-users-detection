import pandas as pd
import numpy as np
import json
import pickle
import networkx as nx
from tqdm import tqdm
import sys
import os
import argparse

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))

from config.data_config import raw_data_path_config, processed_data_path_config

datasets = ['parler', 'echo', 'gab']
datasets.extend([e.capitalize() for e in datasets])

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default='parler', choices=datasets)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--multigraph", action="store_true", default=True)
args = parser.parse_args()
dataset = args.dataset.lower()
raw_data_path = raw_data_path_config[dataset]
processed_data_path = processed_data_path_config[dataset]

if not os.path.exists(processed_data_path['network']) or args.overwrite:
    if not os.path.exists(processed_data_path['edges']) or args.overwrite:
        if dataset == 'parler':
            if not os.path.exists(processed_data_path['reposts']):
                reposts_edge_dict = pd.read_pickle(raw_data_path['reposts'])
                # d = [(k[0], k[1], v) for k, v in echos_edge_dict.items()]
                reposts_edge_df = pd.Series(reposts_edge_dict).rename_axis(['source', 'target']).reset_index(
                    name='weight')
                reposts_edge_df.to_csv(processed_data_path['reposts'], sep='\t', index=False, header=None)
            else:
                reposts_edge_df = pd.read_csv(processed_data_path['reposts'], sep='\t', header=None,
                                              names=['source', 'target', 'weight'])
        elif dataset == 'echo':
            reposts_edge_df = pd.read_csv(raw_data_path['reposts'], sep='\t', header=None,
                                          names=['source', 'target', 'weight'])
        else:
            reposts_edge_df = pd.read_csv(raw_data_path['reposts'], sep='\t', skiprows=1,
                                          names=['source', 'target', 'weight'])
        reposts_edge_df.insert(2, 'type', 'repost')

        if dataset == 'parler':
            if not os.path.exists(processed_data_path['replies']):
                replies_edge_dict = pd.read_pickle(raw_data_path['replies'])
                # d = [(k[0], k[1], v) for k, v in comments_edge_dict.items()]
                replies_edge_df = pd.Series(replies_edge_dict).rename_axis(['source', 'target']).reset_index(
                    name='weight')
                replies_edge_df.to_csv(processed_data_path['replies'], sep='\t', index=False, header=None)
            else:
                replies_edge_df = pd.read_csv(processed_data_path['replies'], sep='\t', header=None,
                                              names=['source', 'target', 'weight'])
        elif dataset == 'echo':
            replies_edge_df = pd.read_csv(raw_data_path['replies'], sep='\t', header=None,
                                          names=['source', 'target', 'weight'])
        else:
            replies_edge_df = pd.read_csv(raw_data_path['replies'], sep='\t', skiprows=1,
                                          names=['source', 'target', 'weight'])
        replies_edge_df.insert(2, 'type', 'reply')

        if dataset != 'parler':
            if dataset == 'echo':
                mentions_edge_df = pd.read_csv(raw_data_path['mentions'], sep='\t', header=None,
                                               names=['source', 'target', 'weight'])
            else:
                mentions_edge_df = pd.read_csv(raw_data_path['mentions'], sep='\t', skiprows=1,
                                               names=['source', 'target', 'weight'])
            mentions_edge_df.insert(2, 'type', 'mention')
            edge_df = pd.concat([reposts_edge_df, replies_edge_df, mentions_edge_df])
        else:
            edge_df = pd.concat([reposts_edge_df, replies_edge_df])
        edge_df.to_csv(processed_data_path['edges'], sep='\t', index=False, header=None)

    else:
        edge_df = pd.read_csv(processed_data_path['edges'], sep='\t', header=None,
                              names=['source', 'target', 'type', 'weight'])

    G = nx.from_pandas_edgelist(edge_df, edge_attr='weight', edge_key='type', create_using=nx.MultiDiGraph)
    # G = nx.read_weighted_edgelist(processed_data_path['edges'], delimiter='\t', create_using=nx.MultiDiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    labeled_users = pd.read_csv(raw_data_path['users_labels'], sep='\t')['user_id'].values
    G.add_nodes_from(labeled_users)
    if not args.multigraph:
        G_ = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if G_.has_edge(u, v):
                G_[u][v]['weight'] += w
            else:
                G_.add_edge(u, v, weight=w)
        G = G_

    nx.write_weighted_edgelist(G, processed_data_path['network'], delimiter='\t')

    with open(processed_data_path['network_nodes'], 'wb') as f:
        pickle.dump(list(G.nodes), f)

if not os.path.exists(processed_data_path['network_nodes']):
    # reposts key is 0, replies key is 1, mentions key is 2
    G = nx.read_weighted_edgelist(processed_data_path['network'], delimiter='\t',
                                  create_using=nx.MultiDiGraph if args.multigraph else nx.DiGraph)
    with open(processed_data_path['network_nodes'], 'wb') as f:
        pickle.dump(list(G.nodes), f)
