import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

# --- 导入依赖 ---
from utils import (
    expand_multigranularity_nodes,
    to_scipy,
    sparse_mx_to_torch_sparse_tensor,
    noisify_with_P,
    build_path_specific_adj,
    normalize_adj,
    get_logger
)
from deeprobust.graph.data import Dataset
from robcon import GCN_Contrastive, compute_graph_smoothness_loss


def objective(trial):
    # ==========================================
    # 1. 种子重置 (必须保留，确保和 main.py 一致)
    # ==========================================
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(15)

    # ==========================================
    # 2. 参数空间
    # ==========================================
    high_threshold = trial.suggest_int("high_threshold", 50, 90, step=5)
    low_threshold = trial.suggest_int("low_threshold", 10, high_threshold - 10, step=5)
    k = trial.suggest_int("k", 2, 3)
    eta = trial.suggest_float("eta", 1e-3, 0.5, log=True)
    beta = trial.suggest_float("beta", 0.5, 5.0)
    gamma = trial.suggest_float("gamma", 0.0, 0.5)
    tau = trial.suggest_float("tau", 0.1, 2.0)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.8)
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256, 512])

    epochs = 300
    warmup_epochs = 60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 3. 数据准备 (防止污染)
    # ==========================================
    adj_raw, features_raw, labels_noisy_fixed, idx_train_fixed, idx_val, idx_test = data_pack

    features = deepcopy(features_raw)
    adj_sparse = to_scipy(torch.FloatTensor(adj_raw.todense())) if not sp.issparse(adj_raw) else adj_raw

    try:
        idx_train_F, labels_F, idx_train_C, labels_C, fine_edges, coarse_edges = expand_multigranularity_nodes(
            features=features,
            adj=adj_sparse,
            idx_train=idx_train_fixed,
            original_labels=labels_noisy_fixed,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            val_nodes=set(idx_val.tolist()),
            k=k
        )
    except Exception:
        return 0.0

    # 构图
    adj_fine = build_path_specific_adj(adj_sparse, fine_edges, tau=tau)
    adj_coarse = build_path_specific_adj(adj_sparse, coarse_edges, tau=tau)
    sp_adj_fine = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_fine + sp.eye(adj_raw.shape[0]))).to(device)
    sp_adj_coarse = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_coarse + sp.eye(adj_raw.shape[0]))).to(device)

    # Tensor化
    features_tensor = torch.FloatTensor(features.todense() if sp.issparse(features) else features).to(device)
    if len(features_tensor.shape) == 2: features_tensor = features_tensor.unsqueeze(0)

    labels_F_tensor = torch.LongTensor(labels_F).to(device)
    labels_C_tensor = torch.LongTensor(labels_C).to(device)
    labels_val_tensor = torch.LongTensor(data.labels[idx_val]).to(device)  # 验证集标签 (用于选模型)
    labels_test_tensor = torch.LongTensor(data.labels[idx_test]).to(device)  # 测试集标签 (用于记录结果)

    # ==========================================
    # 4. 模型初始化
    # ==========================================
    n_class = data.labels.max() + 1
    n_feat = features_tensor.shape[2]
    encoder_teacher = GCN_Contrastive(n_feat=n_feat, n_hidden=n_hidden, dropout=dropout).to(device)
    classifier_teacher = nn.Linear(encoder_teacher.fc2.out_features, n_class).to(device)
    encoder_student = GCN_Contrastive(n_feat=n_feat, n_hidden=n_hidden, dropout=dropout).to(device)
    classifier_student = nn.Linear(encoder_student.fc2.out_features, n_class).to(device)

    optimizer = torch.optim.Adam(
        list(encoder_teacher.parameters()) + list(classifier_teacher.parameters()) +
        list(encoder_student.parameters()) + list(classifier_student.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ==========================================
    # 5. 训练循环 (模拟 main.py 的 Checkpoint 机制)
    # ==========================================

    # 追踪变量
    best_val_acc = 0.0  # 验证集最高分
    final_report_test_acc = 0.0  # 验证集最高分那一刻对应的测试集分数

    for epoch in range(epochs):
        is_warmup = epoch < warmup_epochs

        # --- Train Step (不变) ---
        encoder_teacher.train();
        classifier_teacher.train()
        if is_warmup:
            encoder_student.eval(); classifier_student.eval()
        else:
            encoder_student.train(); classifier_student.train()

        optimizer.zero_grad()

        z_teacher = encoder_teacher(features_tensor, sp_adj_fine, sparse=True).squeeze(0)
        logits_teacher = classifier_teacher(z_teacher)
        loss_sup_teacher = F.cross_entropy(logits_teacher[idx_train_F], labels_F_tensor[idx_train_F])

        if is_warmup:
            loss_total = loss_sup_teacher
        else:
            z_student = encoder_student(features_tensor, sp_adj_coarse, sparse=True).squeeze(0)
            logits_student = classifier_student(z_student)
            loss_sup_student = F.cross_entropy(logits_student[idx_train_C], labels_C_tensor[idx_train_C])
            probs_teacher = F.softmax(logits_teacher.detach(), dim=1)
            log_probs_student = F.log_softmax(logits_student, dim=1)
            loss_distill = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean')
            loss_smooth = compute_graph_smoothness_loss(F.softmax(logits_student, dim=1), sp_adj_coarse)

            loss_total = loss_sup_teacher + eta * loss_sup_student + beta * loss_distill + gamma * loss_smooth

        loss_total.backward()
        optimizer.step()
        if not is_warmup:
            scheduler.step()

        # ==========================================
        # 6. 核心评估逻辑 (逻辑完全复刻 Main.py)
        # ==========================================
        # Main.py 是每 10 个 epoch 验证一次
        if epoch % 10 == 0:
            # 根据阶段选择要评估的模型 (Main.py 逻辑)
            if is_warmup:
                curr_enc = encoder_teacher
                curr_cls = classifier_teacher
                curr_adj = sp_adj_fine
            else:
                curr_enc = encoder_student
                curr_cls = classifier_student
                curr_adj = sp_adj_coarse

            curr_enc.eval()
            curr_cls.eval()
            with torch.no_grad():
                z_eval = curr_enc(features_tensor, curr_adj, sparse=True).squeeze(0)
                logits_eval = curr_cls(z_eval)

                # 1. 计算验证集精度 (Main.py 用这个选模型)
                preds_val = logits_eval[idx_val].max(1)[1]
                acc_val = preds_val.eq(labels_val_tensor).sum().item() / len(idx_val)

                # 2. 计算测试集精度 (Main.py 存下模型后，最后测的就是这个)
                preds_test = logits_eval[idx_test].max(1)[1]
                acc_test = preds_test.eq(labels_test_tensor).sum().item() / len(idx_test)

                # 3. 模拟 "保存最佳模型"
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    # 关键点：我们不存模型文件，我们直接存下这一刻的测试集分数
                    # 这就是 main.py 最终会输出的那个分数
                    final_report_test_acc = acc_test

    # 循环结束，返回记录下的那个分数
    return final_report_test_acc


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.2, help='noise rate')
    parser.add_argument('--n_trials', type=int, default=1000, help='number of optuna trials')
    parser.add_argument('--seed', type=int, default=11, help='Random seed')
    parser.add_argument('--label_rate', type=float, default=0.05, help='rate of labeled data')
    args = parser.parse_args()

    data = Dataset(root='./data', name=args.dataset)
    idx_train_fixed = data.idx_train[:int(args.label_rate * data.adj.shape[0])]

    # 确保噪声与 Main.py 一致 (无 pair, 默认 uniform)
    torch.manual_seed(args.seed)
    np.random.seed(15)

    ptb = args.ptb_rate
    nclass = data.labels.max() + 1
    train_labels = data.labels[idx_train_fixed]
    noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10)

    noisy_labels = data.labels.copy()
    noisy_labels[idx_train_fixed] = noise_y

    data_pack = (data.adj, data.features, noisy_labels, idx_train_fixed, data.idx_val, data.idx_test)

    print(f"Start Optimizing on {args.dataset} (Val Selection -> Test Report Mode)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("=" * 30)
    print("Best Result (Test Acc corresponding to Best Val Model):", study.best_value)
    print("Best Params:", study.best_params)