import torch
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
import logging
from numpy.testing import assert_array_almost_equal
from collections import defaultdict


def noisify_with_P(y_train, nb_classes, noise, random_state=None, noise_type='pair'):
    if noise > 0.0:
        if noise_type == 'uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        cnt = (y_train_noisy != y_train).sum()
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return y_train, P


def multiclass_noisify(y, P, random_state=0):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def build_uniform_P(size, noise):
    assert (noise >= 0.) and (noise <= 1.)
    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise)) * np.ones(size))
    diag_idx = np.arange(size)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def build_pair_p(size, noise):
    assert (noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i, i - 1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if tensor.layout == torch.sparse_coo:
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def to_tensor(adj, features, labels=None, device='cpu'):
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def idx_to_mask(idx, nodes_num):
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()


def preprocess_adj_for_smoothing(adj, self_loop_weight=2.0):
    adj = sp.coo_matrix(adj)
    adj = adj + self_loop_weight * sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()


def filter_noisy_seeds(features_np, idx_train, labels_np, keep_percentile=75):
    print(f">>> [Seed Cleaning] 正在提纯种子节点 (保留 Top {keep_percentile}%)...")
    clean_idx_list = []
    unique_classes = np.unique(labels_np[idx_train])

    for c in unique_classes:
        c_mask = (labels_np[idx_train] == c)
        c_nodes = idx_train[c_mask]
        if len(c_nodes) < 2:
            clean_idx_list.append(c_nodes)
            continue
        c_features = features_np[c_nodes]
        prototype = np.mean(c_features, axis=0).reshape(1, -1)
        sims = cosine_similarity(c_features, prototype).flatten()
        cut_off = np.percentile(sims, 100 - keep_percentile)
        clean_c_nodes = c_nodes[sims >= cut_off]
        clean_idx_list.append(clean_c_nodes)

    clean_idx_train = np.concatenate(clean_idx_list)
    clean_idx_train = np.sort(clean_idx_train)
    removed_count = len(idx_train) - len(clean_idx_train)
    print(f"    - 原始种子数: {len(idx_train)}")
    print(f"    - 提纯后种子数: {len(clean_idx_train)} (剔除了 {removed_count} 个疑似噪声)")
    return clean_idx_train


def expand_multigranularity_nodes(features, adj, idx_train, original_labels,
                                  high_threshold, low_threshold,
                                  val_nodes, clean_labels=None, k=2):
    """
    生成多粒度视图，并返回细粒度和粗粒度的传导路径。
    """
    if isinstance(features, torch.Tensor):
        x = features.cpu().numpy()
    elif sp.issparse(features):
        x = features.toarray()
    else:
        x = features

    if isinstance(original_labels, torch.Tensor):
        labels_np = original_labels.cpu().numpy()
    else:
        labels_np = original_labels

    clean_idx_train = filter_noisy_seeds(x, idx_train, labels_np, keep_percentile=75)

    print(f">>> [Feature Smoothing] 执行 {k} 阶特征平滑...")
    norm_adj = preprocess_adj_for_smoothing(adj, self_loop_weight=2.0)
    smoothed_x = x
    for _ in range(k):
        smoothed_x = norm_adj.dot(smoothed_x)
    if sp.issparse(smoothed_x):
        features_np = smoothed_x.toarray()
    else:
        features_np = smoothed_x

    print(f">>> [Expansion] 计算相似度并生成多粒度视图...")
    seed_nodes_set = set(clean_idx_train.tolist())
    val_nodes_set = set(val_nodes.tolist()) if hasattr(val_nodes, 'tolist') else set(val_nodes)
    original_labeled_set = set(idx_train.tolist())

    candidates = []
    all_sims = []

    for node in seed_nodes_set:
        neighbors = adj[node].indices.tolist()
        if not neighbors: continue
        node_feat = features_np[node].reshape(1, -1)
        neighbor_feats = features_np[neighbors]
        sims = cosine_similarity(node_feat, neighbor_feats).flatten()

        for i, neighbor in enumerate(neighbors):
            if neighbor not in original_labeled_set and neighbor not in val_nodes_set:
                sim_val = sims[i]
                all_sims.append(sim_val)
                candidates.append({
                    'target': neighbor,
                    'label': labels_np[node],
                    'sim': sim_val,
                    'source': node
                })

    if not candidates:
        print("警告: 未找到可扩展节点，返回原始集合。")
        return idx_train, original_labels, idx_train, original_labels, [], []

    cut_off_high = np.percentile(all_sims, high_threshold)
    cut_off_low = np.percentile(all_sims, low_threshold)

    print(f"    - 细粒度阈值 (Top {100 - high_threshold}%): {cut_off_high:.4f}")
    print(f"    - 粗粒度阈值 (Top {100 - low_threshold}%): {cut_off_low:.4f}")

    candidates.sort(key=lambda x: x['sim'], reverse=True)

    fine_new_labels = {}
    fine_nodes_set = set(idx_train.tolist())
    coarse_new_labels = {}
    coarse_nodes_set = set(idx_train.tolist())

    fine_grained_edges = []
    coarse_grained_edges = []  # [ADD] 新增粗粒度路径列表

    for item in candidates:
        neighbor = item['target']
        label = item['label']
        sim = item['sim']
        source = item['source']

        # --- 构建细粒度视图 ---
        if sim >= cut_off_high:
            if neighbor not in fine_new_labels:
                fine_new_labels[neighbor] = label
                fine_nodes_set.add(neighbor)
                fine_grained_edges.append((source, neighbor, sim))

        # --- 构建粗粒度视图 ---
        if sim >= cut_off_low:
            if neighbor not in coarse_new_labels:
                coarse_new_labels[neighbor] = label
                coarse_nodes_set.add(neighbor)
                # [ADD] 只要满足粗粒度阈值，就记录下来用于温和增强
                coarse_grained_edges.append((source, neighbor, sim))

    def format_output(new_labels_map, base_nodes_set):
        final_labels = labels_np.copy()
        for n, l in new_labels_map.items():
            final_labels[n] = l
        return np.array(sorted(list(base_nodes_set))), final_labels

    idx_train_F, labels_F = format_output(fine_new_labels, fine_nodes_set)
    idx_train_C, labels_C = format_output(coarse_new_labels, coarse_nodes_set)

    print(f"--- 多粒度扩展完成 ---")
    print(f"细粒度集合 (F): 总数 {len(idx_train_F)}, 新增 {len(idx_train_F) - len(idx_train)}")
    print(f"粗粒度集合 (C): 总数 {len(idx_train_C)}, 新增 {len(idx_train_C) - len(idx_train)}")

    # [MODIFIED] 返回6个值
    return idx_train_F, labels_F, idx_train_C, labels_C, fine_grained_edges, coarse_grained_edges


def build_path_specific_adj(adj, specific_edges, tau=0.5):
    """
    仅针对提供的特定边修改权重为指数增强值。
    """
    if not specific_edges:
        print(">>> [View Generation] 没有特定的增强边，返回原矩阵。")
        return adj

    print(f">>> [View Generation] 构建特异性增强矩阵 (Edges: {len(specific_edges)}, tau={tau})...")

    adj_lil = adj.tolil()
    cnt = 0
    max_w = 0

    for src, dst, sim in specific_edges:
        weight = np.exp(sim / tau)
        adj_lil[src, dst] = weight
        adj_lil[dst, src] = weight
        if weight > max_w: max_w = weight
        cnt += 1

    print(f"    - 已增强 {cnt} 条路径 (Max Weight: {max_w:.2f})")
    return adj_lil.tocsr()