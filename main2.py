import torch.nn as nn
import time
import argparse
from utils import *
from model import GCN, LogReg
from copy import deepcopy
import scipy
from robcon import get_contrastive_emb
from deeprobust.graph.data import Dataset

# from dataset import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--threshold', type=float, default=1,  help='threshold')
parser.add_argument('--jt', type=float, default=0.03,  help='jaccard threshold')
parser.add_argument('--cos', type=float, default=0.1,  help='cosine similarity threshold')
parser.add_argument('--k', type=int, default=3 ,  help='add k neighbors')
parser.add_argument('--alpha', type=float, default=0.3,  help='add k neighbors')
parser.add_argument('--beta', type=float, default=2,  help='the weight of selfloop')
parser.add_argument("--log", action='store_true', help='run prepare_data or not')
parser.add_argument('--attack', type=str, default='mettack',  help='attack method')
parser.add_argument("--label_rate", type=float, default=0.05, help='rate of labeled data')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--s_rate', type=float, default=0.7, help='rate of delete edges')
parser.add_argument('--edge_rate', type=float, default=0.5, help="noise edge_rate")


args = parser.parse_args()
if args.log:
    logger = get_logger('./log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
else:
    logger = get_logger('./log/try.log')

if args.attack == 'nettack':
    args.ptb_rate = int(args.ptb_rate)
seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(15)

if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))    # 生成标签的索引
    np.random.shuffle(idx)
    # 将标签划分为测试集、训练集、评估集
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]




features = scipy.sparse.csr_matrix(features)

n_nodes = features.shape[0]
n_class = labels.max() + 1

train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes), idx_to_mask(idx_val, n_nodes), \
                                  idx_to_mask(idx_test, n_nodes)
train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

'''
# 删除正常边
from torch_geometric.utils import from_scipy_sparse_matrix
num_nodes = adj.shape[0]
edge_index, _ = from_scipy_sparse_matrix(adj)
edge_index = edge_index.to(device)
masks = torch.bernoulli(1. - torch.ones(edge_index.size(1)) * args.s_rate).to(torch.bool)
edge_index = edge_index[:, masks]
rows = edge_index[0].cpu().numpy()
cols = edge_index[1].cpu().numpy()
cut_adj = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(num_nodes, num_nodes))


# 添加噪声边
num_nodes = cut_adj.shape[0]
num_noise_edges = int((cut_adj.nnz) * args.edge_rate)  # 设置要添加的噪声边数量
graph = cut_adj + cut_adj.T     # 构建原始邻接矩阵的无向图表示
# 随机选择节点对来添加噪声边（确保节点对之间不存在原有边）
noise_edges = []
while len(noise_edges) < num_noise_edges:
    node1, node2 = np.random.choice(num_nodes, size=2, replace=False)
    if graph[node1, node2] == 0:
        noise_edges.append((node1, node2))
# 构建噪声边的稀疏矩阵
indices = np.array(noise_edges).T
values = np.ones(num_noise_edges)
noise_edges_sparse = sp.coo_matrix((values, indices), shape=cut_adj.shape)
# 将原始邻接矩阵和噪声边相加
noisy_adj_matrix = cut_adj + noise_edges_sparse


perturbed_adj = noisy_adj_matrix
perturbed_adj = torch.FloatTensor(perturbed_adj.todense()).to(device)
# perturbed_adj = perturbed_adj.to_dense().to(device)
'''
perturbed_adj = adj
perturbed_adj = torch.FloatTensor(perturbed_adj.todense()).to(device)
# Training parameters
epochs = 200
n_hidden = 16
dropout = 0.5
weight_decay = 5e-4
lr = 1e-2
loss = nn.CrossEntropyLoss()


def train(model, optim, adj, embeds):
    best_loss_val = 9999
    best_acc_val = 0
    for epoch in range(epochs):
        model.train()
        logits = model(adj, embeds)
        l = loss(logits[train_mask], labels[train_mask])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, embeds, labels, val_mask)
        val_loss = loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())

    model.load_state_dict(weights)
    torch.save(weights, './save_model/%s_%s_%s.pth' % (args.attack, args.dataset, args.ptb_rate))
    acc = evaluate(model, adj, embeds, labels, test_mask)
    return acc



if __name__ == '__main__':
    logger.info(args)
    perturbed_adj_sparse = to_scipy(perturbed_adj)

    logger.info('===start preprocessing the graph===')
    # if args.dataset == 'dblp':
    #     args.jt = 0
    adj_pre = preprocess_adj(features, perturbed_adj_sparse, logger, threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    _, features = to_tensor(perturbed_adj_sparse, features)
    embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(), adj_delete=adj_delete,
                                    lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta)
    embeds = embeds.squeeze(dim=0)
    acc_total = []
    embeds = embeds.to('cpu')
    embeds = to_scipy(embeds)

    # prune the perturbed graph by the representations
    # 根据余弦相似度筛选边
    adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False, threshold=args.cos)
    embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    features = features.to_dense()
    labels = torch.LongTensor(labels)
    adj_clean = adj_clean.to(device)
    features = features.to(device)
    labels = labels.to(device)
    logger.info('===train ours on perturbed graph===')
    for run in range(10):
        adj_temp = adj_clean.clone()
        # add k new neighbors to each node
        get_reliable_neighbors(adj_temp, embeds, k=args.k, degree_threshold=args.threshold)
        model = GCN(embeds.shape[1], n_hidden, n_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        adj_temp = adj_new_norm(adj_temp, args.alpha)
        acc = train(model, optimizer, adj_temp, embeds=embeds)
        acc_total.append(acc)
    logger.info('Mean Accuracy:%f' % np.mean(acc_total))
    logger.info('Standard Deviation:%f' % np.std(acc_total, ddof=1))
    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

