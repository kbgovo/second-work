import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import argparse
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

# --- 导入依赖 ---
# 确保 utils.py 和 robcon.py 在同一目录下
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
from robcon import GCN_Contrastive

# ==========================================
# 1. 参数设置
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'dblp', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate (noise level)')

# --- 多粒度阈值参数 ---
parser.add_argument('--high_threshold', type=float, default=60,
                    help='percentile for Fine-grained View (Teacher)')
parser.add_argument('--low_threshold', type=float, default=20,
                    help='percentile for Coarse-grained View (Student)')

# --- 训练权重参数 (已优化默认值) ---
parser.add_argument('--eta', type=float, default=0.1,
                    help='weight for student classification loss (low trust in coarse labels)')
parser.add_argument('--beta', type=float, default=2.0, help='weight for distillation loss (high trust in teacher)')
parser.add_argument('--tau', type=float, default=0.5, help='temperature for adjacency construction')
parser.add_argument('--warmup', type=int, default=60, help='epochs for teacher warmup')

# 通用参数
parser.add_argument('--k', type=int, default=3, help='add k neighbors')
parser.add_argument('--alpha', type=float, default=0.3, help='parameter for adj normalization')
parser.add_argument("--log", action='store_true', help='enable logging')
parser.add_argument('--attack', type=str, default='mettack', help='attack method')
parser.add_argument("--label_rate", type=float, default=0.05, help='rate of labeled data')
parser.add_argument('--seed', type=int, default=12, help='Random seed')
parser.add_argument('--n_hidden', type=int, default=512, help='hidden dimension')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')  # 增加 dropout 防止过拟合
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight_decay')


args = parser.parse_args()

if args.log:
    logger = get_logger(f'./log/{args.attack}/ours_warmup_{args.dataset}_{args.ptb_rate}.log')
else:
    logger = get_logger('./log/try_warmup.log')

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
# 添加噪声标签
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
    # Step A: 多粒度视图生成 (Expanded Graphs)
    # ---------------------------------------------------------
    logger.info('=== Start Expanding Labeled Nodes ===')

    adj_for_expand = perturbed_adj_sparse
    val_nodes = set(idx_val.tolist())

    if isinstance(labels, torch.Tensor):
        original_labels_np = labels.cpu().numpy()
    else:
        original_labels_np = labels

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

    logger.info(f"Fine-grained edges (Teacher View): {len(fine_edges)}")
    logger.info(f"Coarse-grained edges (Student View): {len(coarse_edges)}")

    # ---------------------------------------------------------
    # Step B: 准备数据与模型 (Teacher-Student)
    # ---------------------------------------------------------

    # 1. 准备多视图图结构
    # Teacher 使用更纯净的图结构 (Fine edges)
    adj_fine_raw = build_path_specific_adj(adj_for_expand, fine_edges, tau=args.tau)
    adj_fine_norm = normalize_adj(adj_fine_raw + sp.eye(adj_fine_raw.shape[0]))
    sp_adj_fine = sparse_mx_to_torch_sparse_tensor(adj_fine_norm).to(device)

    # Student 使用信息更丰富的图结构 (Coarse edges)
    # 这里的 tau 可以大一点，保留更多结构
    adj_coarse_raw = build_path_specific_adj(adj_for_expand, coarse_edges, tau=args.tau * 2)
    adj_coarse_norm = normalize_adj(adj_coarse_raw + sp.eye(adj_coarse_raw.shape[0]))
    sp_adj_coarse = sparse_mx_to_torch_sparse_tensor(adj_coarse_norm).to(device)

    # 2. 准备特征
    if sp.issparse(features):
        features_tensor = torch.FloatTensor(features.todense()).to(device)
    else:
        features_tensor = torch.FloatTensor(features).to(device)
    if len(features_tensor.shape) == 2:
        features_tensor = features_tensor.unsqueeze(0)

    # 3. 准备标签
    labels_F_tensor = torch.LongTensor(labels_F).to(device)
    labels_C_tensor = torch.LongTensor(labels_C).to(device)  # Student 的训练标签
    labels_val_tensor = torch.LongTensor(labels[idx_val]).to(device)

    # 4. 初始化双模型 (Teacher & Student)
    # Teacher: 在 Fine 数据上训练，作为指导者
    encoder_teacher = GCN_Contrastive(n_feat=features_tensor.shape[2], n_hidden=args.n_hidden, dropout=args.dropout).to(
        device)
    classifier_teacher = nn.Linear(encoder_teacher.fc2.out_features, n_class).to(device)

    # Student: 在 Coarse 数据上训练，作为最终模型
    encoder_student = GCN_Contrastive(n_feat=features_tensor.shape[2], n_hidden=args.n_hidden, dropout=args.dropout).to(
        device)
    classifier_student = nn.Linear(encoder_student.fc2.out_features, n_class).to(device)

    # 联合优化器
    optimizer = torch.optim.Adam(
        list(encoder_teacher.parameters()) + list(classifier_teacher.parameters()) +
        list(encoder_student.parameters()) + list(classifier_student.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # ---------------------------------------------------------
    # Step C: 训练循环 (Teacher Warmup + Distillation)
    # ---------------------------------------------------------
    logger.info('=== Start Training (Warmup + Distillation) ===')

    best_acc_val = 0
    best_state_encoder = None
    best_state_classifier = None

    # 预热轮数
    warmup_epochs = args.warmup

    for epoch in range(args.epochs):
        # --- 阶段判断 ---
        is_warmup = epoch < warmup_epochs

        # 模式设置
        encoder_teacher.train()
        classifier_teacher.train()

        if is_warmup:
            # 预热期：Student 不训练
            encoder_student.eval()
            classifier_student.eval()
        else:
            # 蒸馏期：Student 开始训练
            encoder_student.train()
            classifier_student.train()

        optimizer.zero_grad()

        # --- 1. Teacher Forward (Fine View) ---
        # Teacher 始终计算，保持对 Fine 数据的敏感度
        z_teacher = encoder_teacher(features_tensor, sp_adj_fine, sparse=True).squeeze(0)
        logits_teacher = classifier_teacher(z_teacher)

        # Teacher Loss (只看最准的 Fine 标签)
        loss_sup_teacher = F.cross_entropy(logits_teacher[idx_train_F], labels_F_tensor[idx_train_F])

        if is_warmup:
            # === Warmup Phase ===
            # 只优化 Teacher 的 loss
            loss_total = loss_sup_teacher
            loss_total.backward()
            optimizer.step()

            phase_status = "Warmup(T)"
            loss_sup_student = 0.0
            loss_distill = 0.0

        else:
            # === Distillation Phase ===
            # Teacher 已经预热好，现在让 Student 进场

            # Student Forward (Coarse View)
            z_student = encoder_student(features_tensor, sp_adj_coarse, sparse=True).squeeze(0)
            logits_student = classifier_student(z_student)

            # Student Supervision Loss (降低权重 eta，因为这里有噪声)
            loss_sup_student = F.cross_entropy(logits_student[idx_train_C], labels_C_tensor[idx_train_C])

            # Distillation Loss: Student 模仿 Teacher
            # Teacher detach，只作为 Target (软标签)
            probs_teacher = F.softmax(logits_teacher.detach(), dim=1)
            log_probs_student = F.log_softmax(logits_student, dim=1)

            # 在全图节点上做蒸馏，传递结构知识
            loss_distill = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean')

            # 总损失：
            # 1. 保持 Teacher 继续微调 (loss_sup_teacher)
            # 2. Student 学习粗标签 (eta * loss_sup_student)
            # 3. Student 模仿 Teacher (beta * loss_distill)
            loss_total = loss_sup_teacher + args.eta * loss_sup_student + args.beta * loss_distill

            loss_total.backward()
            optimizer.step()

            phase_status = "Distill(T+S)"

        # --- 4. Validation ---
        if epoch % 10 == 0:
            # 验证集模型选择：预热期看 Teacher，蒸馏期看 Student
            if is_warmup:
                val_model_enc = encoder_teacher
                val_model_cls = classifier_teacher
                view_adj = sp_adj_fine
                prefix = "[Teacher Val]"
            else:
                val_model_enc = encoder_student
                val_model_cls = classifier_student
                view_adj = sp_adj_coarse
                prefix = "[Student Val]"

            val_model_enc.eval()
            val_model_cls.eval()
            with torch.no_grad():
                z_eval = val_model_enc(features_tensor, view_adj, sparse=True).squeeze(0)
                logits_eval = val_model_cls(z_eval)

                preds = logits_eval[idx_val].max(1)[1]
                acc_val = preds.eq(labels_val_tensor).sum().item() / len(idx_val)

                if is_warmup:
                    logger.info(
                        f"Epoch {epoch:03d} {phase_status} | Loss: {loss_total:.4f} | {prefix} Acc: {acc_val:.4f}")
                else:
                    logger.info(
                        f"Epoch {epoch:03d} {phase_status} | Total: {loss_total:.4f} (T: {loss_sup_teacher:.3f}, S: {loss_sup_student:.3f}, KD: {loss_distill:.3f}) | {prefix} Acc: {acc_val:.4f}")

                    # 只在蒸馏阶段保存最好的 Student 模型
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        best_state_encoder = deepcopy(encoder_student.state_dict())
                        best_state_classifier = deepcopy(classifier_student.state_dict())

    # ---------------------------------------------------------
    # Step D: 最终测试
    # ---------------------------------------------------------
    logger.info("=== Final Testing on Clean Labels (Student Model) ===")

    if best_state_encoder is not None:
        encoder_student.load_state_dict(best_state_encoder)
        classifier_student.load_state_dict(best_state_classifier)
    else:
        logger.warning("No best student model saved! Using last epoch model.")

    encoder_student.eval()
    classifier_student.eval()
    with torch.no_grad():
        # 使用 Student 的视图和参数进行测试
        z_test = encoder_student(features_tensor, sp_adj_coarse, sparse=True).squeeze(0)
        logits_test = classifier_student(z_test)

        test_labels_tensor = clean_labels[idx_test]
        preds_test = logits_test[idx_test].max(1)[1]
        acc_test = preds_test.eq(test_labels_tensor).sum().item() / len(idx_test)

        logger.info(f"Test Accuracy: {acc_test:.4f}")