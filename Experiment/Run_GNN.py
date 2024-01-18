import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv, GCNConv, ARMAConv, ChebConv, GATConv, SAGEConv, SGConv, GINConv, GatedGraphConv
from torch_geometric.nn import GraphSAGE, GCN, GAT, GIN, PNA
from torch_geometric.data import Data, DataLoader
import pickle
import json
import gzip
import random
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import logging
from sklearn.metrics import *
from tqdm import tqdm
import warnings
import networkx as nx
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))

from config.data_config import raw_data_path_config, processed_data_path_config, processed_data_output_dir

warnings.simplefilter(action='ignore', category=UserWarning)

models = ['GCN', 'GAT', 'GRAPHSAGE', 'ARMA', 'Cheb', 'GIN', 'PNA', 'AGNN']
datasets = ['parler', 'echo', 'gab']


class AGNN(torch.nn.Module):
    def __init__(self):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(data.num_features, args.hidden_channels)
        self.prop1 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(args.hidden_channels, data.num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class ARMA(torch.nn.Module):
    def __init__(self):
        super(ARMA, self).__init__()
        self.conv1 = ARMAConv(data.num_features, args.hidden_channels)
        self.conv2 = ARMAConv(args.hidden_channels, data.num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


# class GraphSage(torch.nn.Module):
#     def __init__(self):
#         super(GraphSage, self).__init__()
#         self.conv1 = SAGEConv(num_features, 32)
#         self.conv2 = SAGEConv(32, num_classes)
#
#     def forward(self):
#         x = X
#         # x = F.dropout(x, 0.5)
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, 0.2)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


class CHEBY(torch.nn.Module):
    def __init__(self):
        super(CHEBY, self).__init__()
        self.conv1 = ChebConv(data.num_features, args.hidden_channels, K=1)
        self.conv2 = ChebConv(args.hidden_channels, data.num_classes, K=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# class GAT(torch.nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(num_features, 32, heads=2, concat=True)
#         self.conv2 = GATConv(2 * 32, num_classes, heads=2, concat=False)
#
#     def forward(self):
#         x = X
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, 0.2)
#         x = self.conv2(x, edge_index)
#         return x.log_softmax(dim=1)


# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         self.conv1 = GINConv(num_features, 32)
#         self.conv2 = GCNConv(32, num_classes)
#
#     def forward(self):
#         x = X
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, 0.2)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
#

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default='parler',
                    choices=[e.lower() for e in datasets] + [e.capitalize() for e in datasets])
parser.add_argument("model", type=str, default='agnn',
                    choices=[e.upper() for e in models] + [e.lower() for e in models])
parser.add_argument("--percentages", type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.20, 0.5, 0.8])
parser.add_argument("--hidden_channels", type=int, default=32)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--include_user_features", default=False, action="store_true")
parser.add_argument("--overwrite", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

print(args)


def concatenate_args():
    d = vars(args)
    excluded_keys = ['dataset', 'model', 'percentages', 'include_user_features', 'overwrite']
    result = []

    for key, value in d.items():
        if key not in excluded_keys:
            if isinstance(value, list):
                value_str = '-'.join(map(str, value))
            else:
                value_str = str(value)
            result.append(value_str)

    return '_'.join(result)


args_str = concatenate_args()

res_dir = f'Results/{args.dataset}/{args.model}/{args_str}'
Path(res_dir).mkdir(parents=True, exist_ok=True)

args_dict = vars(args)
args_str = str(args_dict)
with open(os.path.join(res_dir, 'args.txt'), 'w') as f:
    f.write(args_str)

dataset_name = args.dataset.lower()

data_directory = processed_data_output_dir[dataset_name]

# users_kfold_directory = os.path.join(data_directory, 'users', str(args.seed))
#
# # Read the hateful users and non hateful users
# with open(os.path.join(data_directory, 'haters.json')) as json_file:
#     haters = json.load(json_file)
#
# with open(os.path.join(data_directory, 'nonhaters.json')) as json_file:
#     non_haters = json.load(json_file)
# print("User Info loading Done")
logging.info("Loading User Info")
labeled_users_df = pd.read_csv(raw_data_path_config[dataset_name]['users_labels'], sep='\t', index_col=0)
labeled_users_df.index = labeled_users_df.index.astype(str)
haters = labeled_users_df.query('`label` == 1').index.tolist()
non_haters = labeled_users_df.query('`label` == 0').index.tolist()


def get_dataset():
    # dataset_path = os.path.join(data_directory, 'dataset.pt')

    processed_data_path = processed_data_path_config[dataset_name]

    dataset_path = processed_data_path['dataset']

    if os.path.exists(dataset_path) and not args.overwrite:
        data = torch.load(dataset_path)
        return data

    # Read the Doc2vec embedding of all the users
    with open(os.path.join(data_directory, 'Doc2Vec100.p'), 'rb') as handle:
        doc_vectors = pickle.load(handle)

    edge_list = nx.read_weighted_edgelist(processed_data_path['ego_network'],
                                          nodetype=str, create_using=nx.DiGraph).edges(data=True)

    graph = {u: [] for u in labeled_users_df.index}
    graph_dict = {u: idx for idx, u in enumerate(labeled_users_df.index)}
    edge_weights_dict = {}
    inv_graph_dict = {v: k for k, v in graph_dict.items()}
    nodes = len(inv_graph_dict)

    for i in tqdm(edge_list):
        if i[0] not in graph:
            graph[i[0]] = []
            graph_dict[i[0]] = nodes
            inv_graph_dict[nodes] = i[0]
            nodes += 1
        if i[1] not in graph:
            graph[i[1]] = []
            graph_dict[i[1]] = nodes
            inv_graph_dict[nodes] = i[1]
            nodes += 1
        edge_weights_dict[(graph_dict[i[0]], graph_dict[i[1]])] = i[2]['weight']
        graph[i[0]].append(i[1])

    print("Number of Users in the Network:", nodes)

    X = []
    y = []
    # labeled_usernames = []

    users_info_df = None
    if args.include_user_features:
        users_info_df = pd.read_pickle(processed_data_path['users_info'])
        users_info_df.index = users_info_df.index.astype(str)
    for i in tqdm(range(nodes)):
        if args.include_user_features:
            X.append(
                np.concatenate(
                    [doc_vectors[str(inv_graph_dict[i])], users_info_df.loc[str(inv_graph_dict[i])].values]))
        else:
            X.append(doc_vectors[str(inv_graph_dict[i])])
        if inv_graph_dict[i] in haters:
            y.append(1)
        elif inv_graph_dict[i] in non_haters:
            y.append(0)
        else:
            y.append(2)
        # labeled_usernames.append(inv_graph_dict[i])

    featureVector = torch.FloatTensor(X)
    labels = torch.LongTensor(y)

    print("Feature Vector Created")

    rows = []
    cols = []

    edge_weight = []

    for elem in tqdm(graph_dict):
        neighbours = graph[elem]
        r = graph_dict[elem]
        for neighbour in neighbours:
            c = graph_dict[neighbour]
            rows.append(r)
            cols.append(c)
            edge_weight.append(edge_weights_dict[(r, c)])

    edge_index = torch.LongTensor([rows, cols])
    edge_weight = torch.FloatTensor(edge_weight)

    print("Edge Index Created")

    data = Data(x=featureVector, edge_index=edge_index, y=labels, num_nodes=nodes, num_classes=2, graph_dict=graph_dict,
                nodes=nodes, edge_weight=edge_weight)
    # data.num_features = 100

    torch.save(data, dataset_path)

    return data


logging.info('Loading dataset...')
data = get_dataset()

validationFold = [str(i + 1) for i in range(args.k)]
percentages = args.percentages  # [0.05, 0.1, 0.15, 0.20, 0.5, 0.8]

model = args.model.lower()
logging.info(f'Using {model.upper()} model...')


def get_model():
    if model == 'agnn':
        Net = AGNN()
    elif model == 'arma':
        Net = ARMA()
    elif model == 'cheb':
        Net = CHEBY()
    elif model == 'gat':
        Net = GAT(in_channels=data.num_features, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                  out_channels=data.num_classes, dropout=args.dropout, v2=False, heads=2, concat=True)
    elif model == 'gcn':
        Net = GCN(in_channels=data.num_features, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                  out_channels=data.num_classes, dropout=args.dropout)
    elif model == 'gin':
        Net = GIN(in_channels=data.num_features, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                  out_channels=data.num_classes, dropout=args.dropout)
    elif model == 'graphsage':
        Net = GraphSAGE(in_channels=data.num_features, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                        out_channels=data.num_classes, dropout=args.dropout)
    else:
        Net = AGNN()
    return Net


# percentages = [1.0]


def Diff(li1, li2):
    return (list(set(li1) - set(li2)))


def ratio_split(percentage, train_haters, train_non_haters, test_haters, test_non_haters, nodes):
    np.random.seed(args.seed)
    np.random.shuffle(train_haters)
    np.random.shuffle(train_non_haters)

    # Calculate Total number of training point and split training Data Point
    hlen = len(train_haters)
    nhlen = len(train_non_haters)
    htrain_len = int(percentage * hlen)
    nhtrain_len = int(percentage * nhlen)

    # Creating Training List
    trainList = train_haters[0:htrain_len]
    trainList.extend(train_non_haters[0:nhtrain_len])
    # Creating Validation List
    valList = train_haters[htrain_len:]
    valList.extend(train_non_haters[nhtrain_len:])
    # Creating Testing DataPoint
    testList = list(test_haters)
    testList.extend(test_non_haters)

    train_mask = [0] * nodes
    test_mask = [0] * nodes
    val_mask = [0] * nodes

    for i in trainList:
        train_mask[data.graph_dict[i]] = 1

    for i in valList:
        val_mask[data.graph_dict[i]] = 1

    for i in testList:
        test_mask[data.graph_dict[i]] = 1

    train_mask = torch.ByteTensor(train_mask)
    test_mask = torch.ByteTensor(test_mask)
    val_mask = torch.ByteTensor(val_mask)
    # print("Splitting done")
    return train_mask, val_mask, test_mask


def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred)
    precisionScore = precision_score(y_true, y_pred)
    return {"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
            'precision': precisionScore, 'recall': recallScore}


def train():
    model.train()
    optimizer.zero_grad()
    logits = model(X, edge_index, edge_weight=edge_weight)
    loss = F.nll_loss(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    logits = model(X, edge_index, edge_weight=edge_weight)
    accs = []
    Mf1_score = []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        mfc = f1_score(y[mask].detach().cpu(), pred.detach().cpu(), average='macro')
        accs.append(acc)
        Mf1_score.append(mfc)
    test_pred = logits[test_mask].max(1)[1]
    test_Metrics = evalMetric(y[test_mask].detach().cpu(), test_pred.detach().cpu())
    return accs, Mf1_score, test_Metrics


# def split_json_into_folds(file_path, k, seed):
#     # Load JSON data
#     file_name = file_path.split('/')[-1].split('.')[0]
#     with open(file_path, 'r') as file:
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
#         fold_path = os.path.join(dataDirectory, f'{file_name[:-2]}val{i + 1}.json')
#         with open(fold_path, 'w') as file:
#             json.dump(fold, file, indent=4)
#
#     print(f'Successfully split the data into {k} folds.')
#
#
# k = len(validationFold)
# seed = 42
# file_path = os.path.join(dataDirectory, 'haters.json')
# split_json_into_folds(file_path, k, seed)
# file_path = os.path.join(dataDirectory, 'nonhaters.json')
# split_json_into_folds(file_path, k, seed)

test_accs = {}
test_mF1Score = {}
test_f1Score = {}
test_precision = {}
test_recall = {}
test_roc = {}
models_dict = {}
import copy

fin_macrof1Score = {}
fin_accs = {}

val_macrof1Score = {}
val_accs = {}

# Net = AGNN

logging.info("Starting Training")
for percent in tqdm(percentages):
    logging.info("Starting Training for " + str(percent) + " percent of training data")
    test_accs[percent] = {}
    test_mF1Score[percent] = {}
    test_f1Score[percent] = {}
    test_precision[percent] = {}
    test_recall[percent] = {}
    test_roc[percent] = {}

    fin_macrof1Score[percent] = {}
    fin_accs[percent] = {}
    val_macrof1Score[percent] = {}
    val_accs[percent] = {}

    hateval_dict = {}
    nonhateval_dict = {}

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    # random.seed(args.seed)
    # random.shuffle(haters)
    # random.seed(args.seed)
    # random.shuffle(non_haters)
    # logging.info("Starting KFold")
    for fold, (hate_train_idx, hate_test_idx), (nonhate_train_idx, nonhate_test_idx) in zip(validationFold,
                                                                                            kfold.split(haters),
                                                                                            kfold.split(non_haters)):
        test_haters = set(np.array(haters)[hate_test_idx])
        test_non_haters = set(np.array(non_haters)[nonhate_test_idx])
        # for fold in validationFold:
        #     with open(os.path.join(users_kfold_directory, f'hateval{fold}.json')) as json_file:
        #         test_haters = json.load(json_file)
        #     with open(os.path.join(users_kfold_directory, f'nonhateval{fold}.json')) as json_file:
        #         test_non_haters = json.load(json_file)

        train_haters = Diff(haters, test_haters)
        train_non_haters = Diff(non_haters, test_non_haters)

        logging.info(fold)
        logging.info(f'{percent}, Created features and labels and masking function')

        test_accs[percent][fold] = []
        test_mF1Score[percent][fold] = []
        test_f1Score[percent][fold] = []
        test_precision[percent][fold] = []
        test_recall[percent][fold] = []
        test_roc[percent][fold] = []

        fin_macrof1Score[percent][fold] = []
        fin_accs[percent][fold] = []
        val_macrof1Score[percent][fold] = []
        val_accs[percent][fold] = []

        logging.info(f"-------------{fold}-------------")
        for i in range(5):
            logging.info(f"------{i}------")
            torch.manual_seed(30 + i)
            np.random.seed(30 + i)
            # X, y = getData()
            train_mask, val_mask, test_mask = ratio_split(percent, train_haters, train_non_haters,
                                                          test_haters, test_non_haters, data.nodes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_model().to(device)
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device)
            # use_gpu = torch.cuda.is_available()
            y = data.y.to(device)
            X = data.x.to(device)
            train_mask = train_mask.to(device)
            test_mask = test_mask.to(device)
            val_mask = val_mask.to(device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val_acc = best_MfScore = test_acc = test_mfscore = 0
            evalObject = None
            for epoch in range(1, args.epochs + 1):
                loss = train()
                Accuracy, F1Score, test_Metrics = test()
                if Accuracy[1] > best_val_acc:
                    best_val_acc = Accuracy[1]
                    test_acc = Accuracy[2]
                    best_MfScore = F1Score[1]
                    test_mfscore = F1Score[2]
                    evalObject = copy.deepcopy(test_Metrics)
                # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                # print(log.format(epoch, train_acc, val_acc, test_acc))
                logging.info('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
            test_accs[percent][fold].append(evalObject['accuracy'])
            test_mF1Score[percent][fold].append(evalObject['mF1Score'])
            test_f1Score[percent][fold].append(evalObject['f1Score'])
            test_precision[percent][fold].append(evalObject['precision'])
            test_recall[percent][fold].append(evalObject['recall'])
            test_roc[percent][fold].append(evalObject['auc'])
            val_macrof1Score[percent][fold].append(best_MfScore)
            val_accs[percent][fold].append(best_val_acc)


def getFoldWiseResult(dict_fold):
    FinalRes = {}
    FinalResMean = {}
    for percent in percentages:
        FinalRes[percent] = []
        for fold in validationFold:
            FinalRes[percent].extend(dict_fold[percent][fold])
        FinalResMean[percent] = [np.mean(FinalRes[percent]), np.std(FinalRes[percent])]
        print(percent, '\t', np.mean(FinalRes[percent]), '\t', np.std(FinalRes[percent]))
    print("\n")
    return FinalResMean


def dict_to_excel(data, output_file):
    """
    Save a dictionary of dictionaries to an Excel file.

    Parameters:
        data (dict): The input dictionary of dictionaries.
        output_file (str): The filename of the output Excel file.
    """
    df = pd.DataFrame.from_dict(data, orient='index', columns=['mean', 'std'])
    df.index.name = 'percent'
    df.to_excel(output_file,
                index=True)  # Set index=False if you don't want to include the row index in the Excel file.


print("Test Accuracy:")
test_acc_dict = getFoldWiseResult(test_accs)
dict_to_excel(test_acc_dict, os.path.join(res_dir, 'test_acc.xlsx'))
print("Test Macro-F1Score:")
test_mf1_dict = getFoldWiseResult(test_mF1Score)
dict_to_excel(test_mf1_dict, os.path.join(res_dir, 'test_mf1.xlsx'))
print("Test F1Score:")
test_f1_dict = getFoldWiseResult(test_f1Score)
dict_to_excel(test_f1_dict, os.path.join(res_dir, 'test_f1.xlsx'))
print("Test Precision:")
test_precision_dict = getFoldWiseResult(test_precision)
dict_to_excel(test_precision_dict, os.path.join(res_dir, 'test_precision.xlsx'))
print("Test Recall:")
test_recall_dict = getFoldWiseResult(test_recall)
dict_to_excel(test_recall_dict, os.path.join(res_dir, 'test_recall.xlsx'))
print("Test Roc:")
test_roc_auc_dict = getFoldWiseResult(test_roc)
dict_to_excel(test_roc_auc_dict, os.path.join(res_dir, 'test_roc_auc.xlsx'))

print("Validation Accuracy")
ValFinalAcc = {}

for percent in percentages:
    print("Val:" + str(percent))
    ValFinalAcc[percent] = []
    for zz in validationFold:
        print(np.mean(val_accs[percent][zz]), np.std(val_accs[percent][zz]), val_accs[percent][zz])
        ValFinalAcc[percent].extend(val_accs[percent][zz])

print("\nValidation Macro F1 Score")
ValFinalF1 = {}

for percent in percentages:
    print("Validation F1-Score:" + str(percent))
    ValFinalF1[percent] = []
    for zz in validationFold:
        print(np.mean(val_macrof1Score[percent][zz]), np.std(val_macrof1Score[percent][zz]),
              val_macrof1Score[percent][zz])
        ValFinalF1[percent].extend(val_macrof1Score[percent][zz])

print("\nTest Accuracy")
TestFinalAcc = {}
for percent in percentages:
    print("Test:" + str(percent))
    TestFinalAcc[percent] = []
    for zz in validationFold:
        print(np.mean(test_accs[percent][zz]), np.std(test_accs[percent][zz]), test_accs[percent][zz])
        TestFinalAcc[percent].extend(test_accs[percent][zz])

print("\nTest Macro F1 Score")
TestFinalF1 = {}
for percent in percentages:
    print("Test F1-Score:" + str(percent))
    TestFinalF1[percent] = []
    for zz in validationFold:
        print(np.mean(test_mF1Score[percent][zz]), np.std(test_mF1Score[percent][zz]),
              test_mF1Score[percent][zz])
        TestFinalF1[percent].extend(test_mF1Score[percent][zz])

# print("\nFinal Test Accuracy")
# endFinalAcc = {}
# for percent in percentages:
#     print("Fin Test:" + str(percent))
#     endFinalAcc[percent] = []
#     for zz in validationFold:
#         print(np.mean(fin_accs[percent][zz]), np.std(fin_accs[percent][zz]), fin_accs[percent][zz])
#         endFinalAcc[percent].extend(fin_accs[percent][zz])
#
# print("\nFinal Test Macro F1 Score")
# endFinalF1 = {}
# for percent in percentages:
#     print("Fin Test F1-Score:" + str(percent))
#     endFinalF1[percent] = []
#     for zz in validationFold:
#         print(np.mean(fin_macrof1Score[percent][zz]), np.std(fin_macrof1Score[percent][zz]),
#               fin_macrof1Score[percent][zz])
#         endFinalF1[percent].extend(fin_macrof1Score[percent][zz])

print("-----------------------------------------------------------------------------------------")
print("\n--Validation--")
print("Accuracy:")
for i in ValFinalAcc:
    print(i, np.mean(ValFinalAcc[i]), np.std(ValFinalAcc[i]))
print("Macro F1-Score:")
for i in ValFinalF1:
    print(i, np.mean(ValFinalF1[i]), np.std(ValFinalF1[i]))

print("\n--Test--")
print("Accuracy:")
for i in TestFinalAcc:
    print(i, np.mean(TestFinalAcc[i]), np.std(TestFinalAcc[i]))
print("Macro F1-Score:")
for i in TestFinalF1:
    print(i, np.mean(TestFinalF1[i]), np.std(TestFinalF1[i]))

# print("\n--Fin--")
# print("Accuracy:")
# for i in endFinalAcc:
#     print(i, np.mean(endFinalAcc[i]), np.std(endFinalAcc[i]))
# print("Macro F1-Score:")
# for i in endFinalF1:
#     print(i, np.mean(endFinalF1[i]), np.std(endFinalF1[i]))
