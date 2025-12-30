import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import argparse
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

# --- 修正导入 ---
from utils import (
    expand_multigranularity_nodes,
    idx_to_mask,
    to_tensor,
    get_logger,
    to_scipy,
    sparse_mx_to_torch_sparse_tensor,
    noisify_with_P,
    build_path_specific_adj,
    normalize_adj
)
from deeprobust.graph.data import Dataset
from robcon import GCN_Contrastive, compute_asymmetric_loss

# ==========================================
# 1. 参数设置
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate (noise level)')

# --- 新增的多粒度阈值参数 ---
parser.add_argument('--high_threshold', type=float, default=70,
                    help='percentile for Fine-grained View')
parser.add_argument('--low_threshold', type=float, default=10,
                    help='percentile for Coarse-grained View')

parser.add_argument('--eta', type=float, default=1.0, help='weight for classification loss')
parser.add_argument('--tau', type=float, default=0.5, help='temperature for fine-grained boosting')

# 其他通用参数
parser.add_argument('--k', type=int, default=3, help='add k neighbors')
parser.add_argument('--alpha', type=float, default=0.3, help='parameter for adj normalization')
parser.add_argument("--log", action='store_true', help='enable logging')
parser.add_argument('--attack', type=str, default='mettack', help='attack method')
parser.add_argument("--label_rate", type=float, default=0.05, help='rate of labeled data')
parser.add_argument('--seed', type=int, default=20, help='Random seed')
parser.add_argument('--n_hidden', type=int, default=512, help='hidden dimension')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.8, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')

args = parser.parse_args()

if args.log:
    logger = get_logger(f'./log/{args.attack}/ours_{args.dataset}_{args.ptb_rate}.log')
else:
    logger = get_logger('./log/try.log')

# ==========================================
# 2. 环境初始化
# ==========================================
if args.attack == 'nettack':
    args.ptb_rate = int(args.ptb_rate)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(15)

# ==========================================
# 3. 数据加载与预处理
# ==========================================
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
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

clean_labels = labels.copy()
clean_labels = torch.LongTensor(clean_labels).to(device)

ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10)
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y
labels = noise_labels

features = sp.csr_matrix(features)
n_nodes = features.shape[0]
n_class = labels.max() + 1
perturbed_adj = adj
perturbed_adj_sparse = to_scipy(torch.FloatTensor(perturbed_adj.todense())) if not sp.issparse(
    perturbed_adj) else perturbed_adj

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    logger.info(args)

    # ---------------------------------------------------------
    # Step A: 多粒度视图生成
    # ---------------------------------------------------------
    logger.info('=== Start Expanding Labeled Nodes ===')

    adj_for_expand = perturbed_adj_sparse
    val_nodes = set(idx_val.tolist())

    if isinstance(labels, torch.Tensor):
        original_labels_np = labels.cpu().numpy()
    else:
        original_labels_np = labels

    # [MODIFIED] 接收 6 个参数
    idx_train_F, labels_F, idx_train_C, labels_C, fine_edges, coarse_edges = expand_multigranularity_nodes(
        features=features,
        adj=adj_for_expand,
        idx_train=idx_train,
        original_labels=original_labels_np,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        val_nodes=val_nodes,
        clean_labels=clean_labels.cpu().numpy(),
        k=args.k
    )

    logger.info(f"Fine-grained edges: {len(fine_edges)}")
    logger.info(f"Coarse-grained edges: {len(coarse_edges)}")

    # ---------------------------------------------------------
    # Step B: 联合训练
    # ---------------------------------------------------------
    logger.info('=== Start Joint Training ===')

    # 1. 准备多视图图结构

    # --- A. 细粒度视图 (Teacher) ---
    # 强增强 (tau=args.tau, default 0.5)
    adj_fine_raw = build_path_specific_adj(adj_for_expand, fine_edges, tau=args.tau)
    adj_fine_norm = normalize_adj(adj_fine_raw + sp.eye(adj_fine_raw.shape[0]))
    sp_adj_fine = sparse_mx_to_torch_sparse_tensor(adj_fine_norm).to(device)

    # --- B. 粗粒度视图 (Student) ---
    # [MODIFIED] 弱增强 (tau = args.tau * 2, e.g., 1.0)
    # 对满足粗粒度阈值的边也进行适当增强，但比Teacher温和
    # 这有助于 Student 更好地抵抗噪声，同时保持与 Teacher 的梯度差异

    adj_coarse_raw = build_path_specific_adj(adj_for_expand, coarse_edges, tau=args.tau * 2)
    adj_coarse_norm = normalize_adj(adj_coarse_raw + sp.eye(adj_coarse_raw.shape[0]))
    sp_adj_coarse = sparse_mx_to_torch_sparse_tensor(adj_coarse_norm).to(device)
    '''
    adj_coarse_norm = normalize_adj(adj_for_expand + sp.eye(adj_for_expand.shape[0]))
    sp_adj_coarse = sparse_mx_to_torch_sparse_tensor(adj_coarse_norm).to(device)
    '''
    # 2. 准备特征
    if sp.issparse(features):
        features_tensor = torch.FloatTensor(features.todense()).to(device)
    else:
        features_tensor = torch.FloatTensor(features).to(device)

    if len(features_tensor.shape) == 2:
        features_tensor = features_tensor.unsqueeze(0)

    # 3. 准备标签
    labels_F_tensor = torch.LongTensor(labels_F).to(device)
    labels_val_tensor = torch.LongTensor(labels[idx_val]).to(device)

    # 4. 初始化模型
    encoder = GCN_Contrastive(n_feat=features_tensor.shape[2], n_hidden=args.n_hidden, dropout=args.dropout).to(device)
    classifier = nn.Linear(encoder.fc2.out_features, n_class).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 5e-3
    )

    # 6. 训练循环
    best_acc_val = 0
    best_state_encoder = None
    best_state_classifier = None

    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()

        # --- 前向传播 ---
        # Teacher View (Fine & Sharp)
        z_fine_view = encoder(features_tensor, sp_adj_fine, sparse=True)

        # Student View (Coarse & Smooth)
        z_coarse_view = encoder(features_tensor, sp_adj_coarse, sparse=True)

        z_nodes_student = z_coarse_view.squeeze(0)
        logits = classifier(z_nodes_student)

        # --- 计算损失 ---
        loss_con = compute_asymmetric_loss(
            z_fine=z_fine_view,
            z_coarse=z_coarse_view,
            idx_train_F=idx_train_F,
            idx_train_C=idx_train_C,
            device=device,
            tau=0.5, gamma=0.8, beta=0.1
        )   # 对比学习损失

        loss_cls = F.cross_entropy(logits[idx_train_F], labels_F_tensor[idx_train_F])   # 交叉熵损失
        loss_total = loss_con + args.eta * loss_cls

        loss_total.backward()
        optimizer.step()

        # --- 验证 ---
        if epoch % 10 == 0:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                # 验证和测试建议使用 Student View (sp_adj_coarse) 或 原始视图
                # 这里使用 sp_adj_coarse，因为它包含了全局信息且经过了温和去噪
                z_eval = encoder(features_tensor, sp_adj_coarse, sparse=True).squeeze(0)
                logits_eval = classifier(z_eval)

                preds = logits_eval[idx_val].max(1)[1]
                acc_val = preds.eq(labels_val_tensor).sum().item() / len(idx_val)

                logger.info(
                    f"Epoch {epoch:03d} | Total: {loss_total:.4f} (Con: {loss_con:.4f}, Cls: {loss_cls:.4f}) | Val Acc: {acc_val:.4f}")

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_state_encoder = deepcopy(encoder.state_dict())
                    best_state_classifier = deepcopy(classifier.state_dict())

    # ---------------------------------------------------------
    # Step C: 最终测试
    # ---------------------------------------------------------
    logger.info("=== Final Testing on Clean Labels ===")

    if best_state_encoder is not None:
        encoder.load_state_dict(best_state_encoder)
        classifier.load_state_dict(best_state_classifier)

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        z_test = encoder(features_tensor, sp_adj_coarse, sparse=True).squeeze(0)
        logits_test = classifier(z_test)

        test_labels_tensor = clean_labels[idx_test]
        preds_test = logits_test[idx_test].max(1)[1]
        acc_test = preds_test.eq(test_labels_tensor).sum().item() / len(idx_test)

        logger.info(f"Test Accuracy: {acc_test:.4f}")