import torch.nn as nn
import time
import sys
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from utils import *  # 假设包含idx_to_mask、to_tensor、sparse_mx_to_sparse_tensor等工具函数
from model import GCN, LogReg
from copy import deepcopy
from robcon import get_contrastive_emb
from deeprobust.graph.data import Dataset


# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate')
parser.add_argument('--threshold', type=float, default=0.5, help='degree threshold for neighbors')

parser.add_argument('--high_threshold', type=float, default=80,
                    help='High threshold (percentile) for Fine-grained View')# 细粒度视图阈值(例如 90，表示保留前 10% 高置信度)
parser.add_argument('--low_threshold', type=float, default=40,
                    help='Low threshold (percentile) for Coarse-grained View')# 粗粒度视图阈值(例如 60，表示保留前 40%)

parser.add_argument('--pseudo_threshold', type=float, default=80,
                    help='cosine similarity threshold for pseudo labels')  # 新增伪标签阈值
parser.add_argument('--cos', type=float, default=0.1, help='cosine similarity threshold for graph pruning')
parser.add_argument('--k', type=int, default=3, help='add k neighbors')
parser.add_argument('--alpha', type=float, default=0.3, help='parameter for adj normalization')
parser.add_argument('--beta', type=float, default=2, help='the weight of selfloop in contrastive learning')
parser.add_argument("--log", action='store_true', help='enable logging')
parser.add_argument('--attack', type=str, default='mettack', help='attack method')
parser.add_argument("--label_rate", type=float, default=0.05, help='rate of labeled data')
parser.add_argument('--seed', type=int, default=11, help='Random seed')
parser.add_argument('--s_rate', type=float, default=0.7, help='rate of delete edges')
parser.add_argument('--edge_rate', type=float, default=0.5, help="noise edge rate")

args = parser.parse_args()
if args.log:
    logger = get_logger(f'./log/{args.attack}/ours_{args.dataset}_{args.ptb_rate}.log')
else:
    logger = get_logger('./log/try.log')

# 特殊处理nettack的扰动率
if args.attack == 'nettack':
    args.ptb_rate = int(args.ptb_rate)
seed = 15
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(15)

# 数据集加载
if args.dataset == 'dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data', 'dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9 + args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]  # 按标签率截取训练集

clean_labels = labels.copy()
clean_labels = torch.LongTensor(clean_labels).to(device)

from utils import noisify_with_P
ptb = args.ptb_rate         # 设置噪声率
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10) # y是添加噪声之后的标签，P是混淆矩阵
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y   # 所有的标签集合->包含噪声的标签集合
labels = noise_labels

# 数据预处理
features = sp.csr_matrix(features)
n_nodes = features.shape[0]
n_class = labels.max() + 1

real_labels = labels.copy()
real_labels = torch.LongTensor(real_labels).to(device)

real_idx_train = idx_train.copy()
# 初始掩码（扩展标签前）
train_mask, val_mask, test_mask = idx_to_mask(idx_train, n_nodes), idx_to_mask(idx_val, n_nodes), \
                                  idx_to_mask(idx_test, n_nodes)
train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 扰动图设置（使用原始图或扰动图，此处保留原逻辑）
# 注释掉的扰动图生成代码（需要时可启用）
'''
from torch_geometric.utils import from_scipy_sparse_matrix
num_nodes = adj.shape[0]
edge_index, _ = from_scipy_sparse_matrix(adj)
edge_index = edge_index.to(device)
masks = torch.bernoulli(1. - torch.ones(edge_index.size(1)) * args.s_rate).to(torch.bool)
edge_index = edge_index[:, masks]
rows = edge_index[0].cpu().numpy()
cols = edge_index[1].cpu().numpy()
cut_adj = sp.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(num_nodes, num_nodes))

num_noise_edges = int((cut_adj.nnz) * args.edge_rate)
graph = cut_adj + cut_adj.T
noise_edges = []
while len(noise_edges) < num_noise_edges:
    node1, node2 = np.random.choice(num_nodes, size=2, replace=False)
    if graph[node1, node2] == 0:
        noise_edges.append((node1, node2))
indices = np.array(noise_edges).T
values = np.ones(num_noise_edges)
noise_edges_sparse = sp.coo_matrix((values, indices), shape=cut_adj.shape)
noisy_adj_matrix = cut_adj + noise_edges_sparse
perturbed_adj = noisy_adj_matrix
'''
perturbed_adj = adj  # 不使用扰动时直接用原始图
perturbed_adj = torch.FloatTensor(perturbed_adj.todense()).to(device)

# 训练参数
epochs = 200
n_hidden = 16
dropout = 0.5
weight_decay = 5e-4
lr = 1e-2
loss = nn.CrossEntropyLoss()


# 模型训练函数
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

    acc = evaluate(model, adj, embeds, real_labels, test_mask)

    return acc



if __name__ == '__main__':
    logger.info(args)
    perturbed_adj_sparse = to_scipy(perturbed_adj)  # 扰动图转为稀疏矩阵

    # 核心修改：用标签扩展
    logger.info('===start expanding labeled nodes===')
    # 准备扩展标签所需参数
    adj_for_expand = perturbed_adj_sparse  # 用于扩展标签的邻接矩阵
    val_nodes = set(idx_val.tolist())
    test_nodes = set(idx_test.tolist())

    # 确保标签格式为numpy数组
    if isinstance(labels, torch.Tensor):
        original_labels_np = labels.cpu().numpy()
    else:
        original_labels_np = labels

    # 调用多粒度视图生成函数
    idx_train_F, labels_F, idx_train_C, labels_C = expand_multigranularity_nodes(
        features=features,
        adj=adj_for_expand,
        idx_train=idx_train,
        original_labels=original_labels_np,
        high_threshold=args.high_threshold, # 传入细粒度阈值
        low_threshold=args.low_threshold,   # 传入粗粒度阈值
        val_nodes=val_nodes,
        clean_labels=clean_labels # 用于打印准确率以监控质量
    )

    # sys.exit(0)
    # 更新训练集索引和标签
    idx_train = idx_train_F
    labels = labels_F
    # 更新训练掩码（因训练集已扩展）
    train_mask = idx_to_mask(idx_train, n_nodes).to(device)

    # 特征转换
    _, features = to_tensor(perturbed_adj_sparse, features)

    # 对比学习获取嵌入（使用原始图结构，无图清洗）
    adj_for_contrastive = adj_for_expand  # 对比学习用图
    adj_delete_dummy = sp.csr_matrix((n_nodes, n_nodes))  # 虚拟删除边矩阵

    embeds, _ = get_contrastive_emb(
        logger=logger,
        adj=adj_for_contrastive,
        features=features.unsqueeze(dim=0).to_dense(),
        idx_train_F=idx_train_F,  # 传入细粒度索引
        idx_train_C=idx_train_C,
        lr=0.001,
        weight_decay=0.0,
        nb_epochs=10000,
        beta=args.beta
    )
    embeds = embeds.squeeze(dim=0)

    # 后续处理
    acc_total = []
    # embeds = embeds.to('cpu')
    # embeds = to_scipy(embeds)
    embeds = embeds.to(device)

    # 基于嵌入修剪图（可选，保留原逻辑）
    adj_clean = adj_for_expand

    # 转换为模型输入格式
    # embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    features = features.to_dense()
    labels = torch.LongTensor(labels).to(device)  # 扩展后的标签转为tensor
    adj_clean = adj_clean.to(device)
    features = features.to(device)

    # 训练模型（10次实验取平均）
    logger.info('===train ours on perturbed graph===')
    for run in range(5):
        adj_temp = adj_clean.clone()
        # get_reliable_neighbors(adj_temp, embeds, k=args.k, degree_threshold=args.threshold)   # 添加可靠邻居
        model = GCN(embeds.shape[1], n_hidden, n_class).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        adj_temp = adj_new_norm(adj_temp, args.alpha)       # 优化邻接矩阵
        acc = train(model, optimizer, adj_temp, embeds=embeds)
        acc_total.append(acc)

    # 输出结果
    logger.info(f'Mean Accuracy: {np.mean(acc_total):.4f}')
    logger.info(f'Standard Deviation: {np.std(acc_total, ddof=1):.4f}')

    # 提取各子集嵌入（用于后续分析）
