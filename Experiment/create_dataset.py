# import os
# import torch
# from torch_geometric.data import Data, DataLoader
# import pickle
# import json
# from tqdm import tqdm
# import warnings
# import networkx as nx
#
# warnings.simplefilter(action='ignore', category=UserWarning)
#
#
# def get_dataset(dataDirectory):
#     dataset_path = os.path.join(dataDirectory, 'dataset.pt')
#     if os.path.exists(dataset_path):
#         data = torch.load(dataset_path)
#         return data
#
#     with open(os.path.join(dataDirectory, 'haters.json')) as json_file:
#         haters = json.load(json_file)
#
#     with open(os.path.join(dataDirectory, 'nonhaters.json')) as json_file:
#         non_haters = json.load(json_file)
#
#     # Read the Doc2vec embedding of all the users
#     with open(os.path.join(dataDirectory, 'Doc2vec100.p', 'rb')) as handle:
#         doc_vectors = pickle.load(handle)
#
#     final_list = nx.read_weighted_edgelist("network.weighted.edgelist.gz",
#                                            nodetype=str, create_using=nx.DiGraph).edges()
#
#     graph = {}
#     graph_dict = {}
#     inv_graph_dict = {}
#     nodes = 0
#
#     for i in final_list:
#         if i[0] not in graph:
#             graph[i[0]] = []
#             graph_dict[i[0]] = nodes
#             inv_graph_dict[nodes] = i[0]
#             nodes += 1
#         if i[1] not in graph:
#             graph[i[1]] = []
#             graph_dict[i[1]] = nodes
#             inv_graph_dict[nodes] = i[1]
#             nodes += 1
#         graph[i[0]].append(i[1])
#
#     # print("Number of Users in the Network:", nodes)
#
#     X = []
#     y = []
#
#     for i in range(0, nodes):
#         X.append(doc_vectors[str(inv_graph_dict[i])])
#         if inv_graph_dict[i] in haters:
#             y.append(1)
#         elif inv_graph_dict[i] in non_haters:
#             y.append(0)
#         else:
#             y.append(2)
#
#     featureVector = torch.FloatTensor(X)
#     labels = torch.LongTensor(y)
#
#     rows = []
#     cols = []
#
#     for elem in tqdm(graph_dict):
#         neighbours = graph[elem]
#         r = graph_dict[elem]
#         for neighbour in neighbours:
#             c = graph_dict[neighbour]
#             rows.append(r)
#             cols.append(c)
#
#     edge_index = torch.LongTensor([rows, cols])
#
#     # print("Edge Index Created")
#
#     data = Data(x=featureVector, edge_index=edge_index, y=labels, num_nodes=nodes, num_classes=2, graph_dict=graph_dict)
#     # data.num_features = 100
#
#     torch.save(data, dataset_path)
#
#     return data
#
# # print("Data Created")
