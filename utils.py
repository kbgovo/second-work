import torch
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
import logging
from numpy.testing import assert_array_almost_equal
from collections import defaultdict

def noisify_with_P(y_train, nb_classes, noise, random_state=None,  noise_type='pair'):

    if noise > 0.0:
        if noise_type=='uniform':
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
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    实操根据概率改变标签类型
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix 行随机矩阵（每一行元素之和为）
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)       # 生成随机数

    # 更新y标签
    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    生成均匀噪声，原标签有p概率变到其他类型的标签，计算混淆矩阵
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise)) * np.ones(size))

    diag_idx = np.arange(size)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    """
    生成配对噪声，有p概率变到最相似的类型标签中，计算混淆矩阵
    """
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
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


def adj_norm(adj, neighbor_only=False):
    if not neighbor_only:
        adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    if adj.is_sparse:
        degree = adj.to_dense().sum(dim=1)
    else:
        degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), -0.5).expand(adj.shape[0], adj.shape[0])
    in_degree_norm = torch.where(torch.isinf(in_degree_norm), torch.full_like(in_degree_norm, 0), in_degree_norm)
    out_degree_norm = torch.pow(degree.view(-1, 1), -0.5).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.where(torch.isinf(out_degree_norm), torch.full_like(out_degree_norm, 0), out_degree_norm)
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    return adj


def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adj, features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices == test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]


def get_reliable_neighbors(adj, features, k, degree_threshold):
    degree = adj.sum(dim=1)
    degree_mask = degree > degree_threshold
    assert degree_mask.sum().item() >= k
    sim = cosine_similarity(features.to('cpu'))
    sim = torch.FloatTensor(sim).to('cuda')
    sim[:, degree_mask == False] = 0
    _, top_k_indices = sim.topk(k=k, dim=1)
    for i in range(adj.shape[0]):
        adj[i][top_k_indices[i]] = 1
        adj[i][i] = 0
    return


def adj_new_norm(adj, alpha):
    adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    if alpha != -0.5:
        return adj / (adj.sum(dim=1).reshape(adj.shape[0], -1))
    else:
        return adj


def preprocess_adj(features, adj, logger, metric='similarity', threshold=0.03, jaccard=True):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_triu = sp.triu(adj, format='csr')
    if sp.issparse(features):
        features = features.todense().A  # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if jaccard:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                           threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features,
                                          threshold=threshold)
    logger.info('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj


import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_adj_for_smoothing(adj, self_loop_weight=2.0):
    """
    [辅助函数] 构建归一化邻接矩阵: D^-1/2 * (A + lambda*I) * D^-1/2

    Args:
        self_loop_weight: 自环权重。建议设为 >1 的值，保证平滑时保留更多自身特征，
                          防止节点特征被邻居完全“同化”。
    """
    adj = sp.coo_matrix(adj)

    # 添加加权自环
    adj = adj + self_loop_weight * sp.eye(adj.shape[0])

    # 归一化步骤
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 转为 CSR 格式加速计算
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()

def filter_noisy_seeds(features_np, idx_train, labels_np, keep_percentile=75):
    """
    [核心步骤 1: 种子提纯]
    利用“特征原型”识别并剔除可能标记错误的训练节点。

    Args:
        keep_percentile: 每个类别保留置信度最高的百分之多少 (推荐 70-80)
    """
    print(f">>> [Seed Cleaning] 正在提纯种子节点 (保留 Top {keep_percentile}%)...")

    clean_idx_list = []
    unique_classes = np.unique(labels_np[idx_train])

    for c in unique_classes:
        # 1. 获取该类别下的所有训练节点
        c_mask = (labels_np[idx_train] == c)
        c_nodes = idx_train[c_mask]

        if len(c_nodes) < 2:
            clean_idx_list.append(c_nodes)
            continue

        # 2. 计算类原型 (Prototype) - 该类节点的特征中心
        c_features = features_np[c_nodes]
        prototype = np.mean(c_features, axis=0).reshape(1, -1)

        # 3. 计算每个节点到原型的相似度
        # 如果标签是错的（比如狗标成了猫），它离猫的中心通常比较远
        sims = cosine_similarity(c_features, prototype).flatten()

        # 4. 动态截断，剔除离群点
        # cut_off 是第 (100-keep) 分位处的数值
        cut_off = np.percentile(sims, 100 - keep_percentile)

        # 保留那些离中心近的“典型”节点
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
    生成多粒度视图，F表示细粒度，C表示粗粒度视图。
    Args:
        high_threshold (int): 细粒度视图的百分位阈值 (例如 90，代表 Top 10%)，对应论文 theta_h
        low_threshold (int): 粗粒度视图的百分位阈值 (例如 60，代表 Top 40%)，对应论文 theta_l
        k (int): 结构特征平滑的阶数
    
    Returns:
        idx_train_F, labels_F: 细粒度视图的训练集索引和标签
        idx_train_C, labels_C: 粗粒度视图的训练集索引和标签
    """

    # ---------------------------------------------------------
    # 1. 种子提纯 (Prototype-based Label Purification)
    # ---------------------------------------------------------
    # 复用原有的 filter_noisy_seeds 函数，确保种子节点的质量
    if isinstance(features, torch.Tensor):
        m = skdfhas()
        x = features.cpu().numpy()
    elif sp.issparse(features):
        x = features.toarray()
    else:
        x = features

    if isinstance(original_labels, torch.Tensor):
        labels_np = original_labels.cpu().numpy()
    else:
        labels_np = original_labels

    # 这里的 keep_percentile 可以作为超参数，或者固定为 75
    clean_idx_train = filter_noisy_seeds(x, idx_train, labels_np, keep_percentile=75)
    
    # ---------------------------------------------------------
    # 2. 结构感知特征平滑 (Structure-Aware Feature Smoothing)
    # ---------------------------------------------------------
    print(f">>> [Feature Smoothing] 执行 {k} 阶特征平滑...")
    # 使用你在 utils.py 中已有的 preprocess_adj_for_smoothing
    norm_adj = preprocess_adj_for_smoothing(adj, self_loop_weight=2.0)

    smoothed_x = x
    for _ in range(k):
        smoothed_x = norm_adj.dot(smoothed_x)

    if sp.issparse(smoothed_x):
        features_np = smoothed_x.toarray()
    else:
        features_np = smoothed_x

    # ---------------------------------------------------------
    # 3. 计算相似度并收集候选 (Candidate Collection)
    # ---------------------------------------------------------
    print(f">>> [Expansion] 计算相似度并生成多粒度视图...")
    
    seed_nodes_set = set(clean_idx_train.tolist())
    val_nodes_set = set(val_nodes.tolist()) if hasattr(val_nodes, 'tolist') else set(val_nodes)
    # 注意：这里用原始的训练集作为“已标记”的基准，防止覆盖原始标签
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
            # 排除：已有的训练节点、验证集节点
            if neighbor not in original_labeled_set and neighbor not in val_nodes_set:
                sim_val = sims[i]
                all_sims.append(sim_val)
                candidates.append({
                    'target': neighbor,
                    'label': labels_np[node],
                    'sim': sim_val
                })

    if not candidates:
        print("警告: 未找到可扩展节点，返回原始集合。")
        return idx_train, original_labels, idx_train, original_labels

    # ---------------------------------------------------------
    # 4. 双阈值筛选 (Dual Thresholding)
    # ---------------------------------------------------------
    # 计算两个截断值
    # high_threshold 对应细粒度 (Strict)，例如 90 分位
    cut_off_high = np.percentile(all_sims, high_threshold)
    # low_threshold 对应粗粒度 (Loose)，例如 60 分位
    cut_off_low = np.percentile(all_sims, low_threshold)
    
    print(f"    - 细粒度阈值 (Top {100-high_threshold}%): {cut_off_high:.4f}")
    print(f"    - 粗粒度阈值 (Top {100-low_threshold}%): {cut_off_low:.4f}")

    # 按相似度排序，确保高质量优先分配
    candidates.sort(key=lambda x: x['sim'], reverse=True)

    # 初始化两组集合
    # 细粒度集合 (F)
    fine_new_labels = {}
    fine_nodes_set = set(idx_train.tolist()) # 包含原始标签
    
    # 粗粒度集合 (C)
    coarse_new_labels = {}
    coarse_nodes_set = set(idx_train.tolist()) # 包含原始标签

    # 开始分配
    for item in candidates:
        neighbor = item['target']
        label = item['label']
        sim = item['sim']
        
        # --- 构建细粒度视图 (Fine-grained) ---
        if sim >= cut_off_high:
            if neighbor not in fine_new_labels:
                fine_new_labels[neighbor] = label
                fine_nodes_set.add(neighbor)
        
        # --- 构建粗粒度视图 (Coarse-grained) ---
        # 注意：粗粒度集合是细粒度集合的超集 (Superset)，或者包含更多节点
        # 只要满足低阈值即可入选粗粒度
        if sim >= cut_off_low:
            if neighbor not in coarse_new_labels:
                coarse_new_labels[neighbor] = label
                coarse_nodes_set.add(neighbor)

    # ---------------------------------------------------------
    # 5. 结果封装
    # ---------------------------------------------------------
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
    
    return idx_train_F, labels_F, idx_train_C, labels_C

'''
def preprocess_adj_for_smoothing(adj):
    """
    [预处理] 构建归一化邻接矩阵: D^-1/2 * (A+I) * D^-1/2
    用于特征平滑传播。
    """
    # 转换为 COO 格式处理
    adj = sp.coo_matrix(adj)

    # 1. 添加自环 (Self-loops): A = A + I
    # 保证节点在平滑时保留自身的语义特征
    adj = adj + sp.eye(adj.shape[0])

    # 2. 计算度矩阵 D
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 3. 归一化并转为 CSR 格式 (加速矩阵乘法)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()

def expand_labeled_nodes(features, adj, idx_train, original_labels, threshold, val_nodes, clean_labels, test_nodes,
                         k=4):
    """
    Args:
        percentile (int): 0-100 之间的整数。
                          例如 80 表示保留相似度排名前 20% (100-80) 的连边。
        k (int): 特征平滑的阶数，建议设为 2。
    """

    # ============================================================
    # 阶段 1: 特征平滑 (Structure-Aware Smoothing)
    # 利用图结构聚合邻居特征，解决单纯语义不可靠的问题
    # ============================================================
    print(f">>> [Step 1] 执行 {k} 阶特征平滑...")

    percentile = threshold
    # 1. 数据格式统一化 (转为 numpy 或 sparse matrix)
    if isinstance(features, torch.Tensor):
        x = features.cpu().numpy()
    else:
        x = features

    # 2. 获取归一化邻接矩阵
    norm_adj = preprocess_adj_for_smoothing(adj)

    # 3. 执行传播: X' = (D^-0.5(A+I)D^-0.5)^k * X
    smoothed_x = x
    for _ in range(k):
        smoothed_x = norm_adj.dot(smoothed_x)

    # 4. 转为 Dense Numpy Array 以便切片和计算余弦相似度
    if sp.issparse(smoothed_x):
        features_np = smoothed_x.toarray()
    else:
        features_np = smoothed_x

    print(">>> 特征平滑完成。")

    # ============================================================
    # 阶段 2: 收集候选边与动态阈值计算
    # ============================================================
    original_labeled = set(idx_train.tolist())
    val_nodes_set = set(val_nodes.tolist()) if isinstance(val_nodes, torch.Tensor) else set(val_nodes)

    candidates = []  # 存储字典: {'target', 'label', 'sim'}
    all_sims = []  # 存储所有计算出的相似度用于统计分布

    print(f">>> [Step 2] 计算相似度并收集候选节点...")

    # 遍历每一个种子节点 (Seed Node)
    for node in original_labeled:
        # 使用原始物理邻接矩阵寻找邻居
        neighbors = adj[node].indices.tolist()
        if not neighbors:
            continue

        # 使用【平滑后】的特征计算相似度
        node_feat = features_np[node].reshape(1, -1)
        neighbor_feats = features_np[neighbors]

        # 计算当前节点与所有邻居的余弦相似度
        sims = cosine_similarity(node_feat, neighbor_feats).flatten()

        for i, neighbor in enumerate(neighbors):
            # 排除已标记、验证集、测试集节点
            if neighbor not in original_labeled and neighbor not in val_nodes_set:
                sim_val = sims[i]
                all_sims.append(sim_val)

                # 暂时存入候选列表，等待阈值筛选
                candidates.append({
                    'target': neighbor,  # 目标无标签节点
                    'label': original_labels[node],  # 拟赋予的标签
                    'sim': sim_val  # 置信度
                })

    # 如果没有候选节点，直接返回
    if not candidates:
        print("警告: 未找到任何可扩展的邻居节点。")
        return idx_train, original_labels

    # ============================================================
    # 阶段 3: 动态确定阈值 (Dynamic Thresholding)
    # ============================================================
    # 计算分位点数值。例如 percentile=80，则 cut_off 是第 80% 位置的数
    # 这意味着只有比这个数大的(即前 20%)会被选中
    cut_off = np.percentile(all_sims, percentile)

    print(f"--- 相似度分布统计 ---")
    print(f"Min: {np.min(all_sims):.4f}, Max: {np.max(all_sims):.4f}, Mean: {np.mean(all_sims):.4f}")
    print(f"设定百分位: {percentile}% (保留 Top {100 - percentile}%)")
    print(f"动态计算出的截断阈值: {cut_off:.4f}")

    # ============================================================
    # 阶段 4: 过滤与冲突解决 (Global Sorting)
    # ============================================================
    # 关键步骤：按相似度从高到低排序！
    # 这样如果一个节点被多个邻居推荐，它会被相似度最高的那个邻居优先“抢到”
    candidates.sort(key=lambda x: x['sim'], reverse=True)

    expanded_labeled = set(original_labeled)
    new_labels = {}
    cnt_correct = 0

    for item in candidates:
        # 仅处理高于动态阈值的边
        if item['sim'] >= cut_off:
            neighbor = item['target']

            # 如果该节点还没被分配标签 (由于已排序，先被分配的一定是置信度更高的)
            if neighbor not in new_labels:
                new_labels[neighbor] = item['label']
                expanded_labeled.add(neighbor)

                # 统计准确率 (仅用于监控，不用于训练)
                if clean_labels is not None:
                    # 注意: 需确保 clean_labels 索引访问方式正确
                    true_label = clean_labels[neighbor]
                    # 处理 tensor 类型的 label
                    if isinstance(true_label, torch.Tensor):
                        true_label = true_label.item()

                    assigned_label = item['label']
                    if isinstance(assigned_label, torch.Tensor):
                        assigned_label = assigned_label.item()

                    if assigned_label == true_label:
                        cnt_correct += 1

    # ============================================================
    # 阶段 5: 结果整合
    # ============================================================
    # 合并标签
    final_labels = original_labels.copy()
    if isinstance(original_labels, torch.Tensor):
        # 如果是Tensor，可能需要转numpy或保持Tensor，这里建议转为字典更新后再转回
        # 为简单起见，假设调用方能处理 tensor/numpy 混用，或者这里统一转 tensor
        pass

        # 简单处理：更新字典或数组
    # 如果 original_labels 是类似 numpy 数组
    for node, label in new_labels.items():
        # 确保不越界 (通常 original_labels 长度是全图节点数)
        final_labels[node] = label

    expanded_idx_train = np.array(sorted(list(expanded_labeled)))

    # 打印最终统计
    added_count = len(expanded_idx_train) - len(idx_train)
    acc = cnt_correct / added_count if added_count > 0 else 0.0

    print(f"--- 扩展完成 ---")
    print(f"正确标签数: {cnt_correct}")
    print(f"新增标签数: {added_count}")
    print(f"新增标签准确率: {acc:.4f}")

    return expanded_idx_train, final_labels

def expand_labeled_nodes2(features, adj, idx_train, original_labels, threshold, val_nodes, clean_labels, test_nodes):
    # 将特征转换为numpy数组以便计算余弦相似度
    cnt = 0
    if sp.issparse(features):
        features_np = features.toarray()
    else:
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
    # 存储原始标签节点
    original_labeled = set(idx_train.tolist())
    # 存储扩展后的标签节点
    expanded_labeled = set(original_labeled)
    # 存储新添加节点的标签
    new_labels = {}
    # 遍历每个原始标签节点
    for node in original_labeled:
        # 获取该节点的所有邻居
        neighbors = adj[node].indices.tolist()
        # 计算当前节点与所有邻居的余弦相似度
        node_feat = features_np[node].reshape(1, -1)
        neighbor_feats = features_np[neighbors]
        similarities = cosine_similarity(node_feat, neighbor_feats).flatten()
        # 检查每个邻居
        for i, neighbor in enumerate(neighbors):
            # 如果邻居不在原始标签节点中且相似度大于阈值
            if neighbor not in original_labeled and similarities[i] > threshold and neighbor not in val_nodes:
                # 将邻居添加到扩展标签节点集合
                expanded_labeled.add(neighbor)
                # 为新节点分配与当前节点相同的标签
                # 如果一个节点被多个标签节点覆盖，保留第一个分配的标签
                if neighbor not in new_labels:
                    new_labels[neighbor] = original_labels[node]
                    k = original_labels[node]
                    t = clean_labels[neighbor]
                    if k==t:
                        cnt+=1
    # 创建新的标签数组
    expanded_labels = original_labels.copy()
    for node, label in new_labels.items():
        expanded_labels[node] = label
    # 转换为numpy数组并排序
    expanded_idx_train = np.array(sorted(expanded_labeled))
    # 打印扩展信息
    print(f"原始标签节点数: {len(idx_train)}")
    print(f"扩展后标签节点数: {len(expanded_idx_train)}")
    print(f"新增标签节点数: {len(expanded_idx_train) - len(idx_train)}")
    print(f"拓展的正确的标签节点数: {cnt}")

    return expanded_idx_train, expanded_labels

def expand_labeled_nodes_weighted(features, adj, idx_train, original_labels, threshold, val_nodes, clean_labels,
                                  test_nodes):
    # --- 1. 数据准备 ---
    if sp.issparse(features):
        features_np = features.toarray()
    else:
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features

    original_labeled = set(idx_train.tolist())
    val_nodes_set = set(val_nodes.tolist()) if hasattr(val_nodes, 'tolist') else set(val_nodes)

    # 用于存储候选节点的投票箱
    # 结构: { neighbor_id: { label_id: total_similarity_score } }
    candidate_votes = defaultdict(lambda: defaultdict(float))

    # --- 2. 收集投票 (Accumulate Votes) ---
    # 遍历每个原始训练节点
    for node in original_labeled:
        # 获取该节点的真实标签
        current_label = original_labels[node]

        # 获取该节点的所有邻居
        neighbors = adj[node].indices.tolist()

        if not neighbors:
            continue

        # 计算当前节点与所有邻居的余弦相似度
        node_feat = features_np[node].reshape(1, -1)
        neighbor_feats = features_np[neighbors]
        similarities = cosine_similarity(node_feat, neighbor_feats).flatten()

        # 遍历邻居进行筛选和投票
        for i, neighbor in enumerate(neighbors):
            sim = similarities[i]

            # 筛选条件：
            # 1. 相似度大于阈值
            # 2. 邻居不是原始训练节点
            # 3. 邻居不在验证集中 (防泄露)
            if sim > threshold and neighbor not in original_labeled and neighbor not in val_nodes_set:
                # 核心修改：不再直接赋值，而是累加相似度作为分数
                candidate_votes[neighbor][current_label] += sim

    # --- 3. 决议阶段 (Decision Making) ---
    expanded_labeled = set(original_labeled)
    new_labels = {}
    cnt = 0  # 记录预测正确的数量

    # 遍历所有候选人（即满足阈值条件的未标记邻居）
    for neighbor, votes in candidate_votes.items():
        # 找出总相似度分数最高的标签
        # votes 是一个字典 {label: score, label2: score...}
        best_label = max(votes, key=votes.get)

        # 将该节点加入扩展集合
        expanded_labeled.add(neighbor)
        new_labels[neighbor] = best_label

        # 验证准确性 (仅用于统计，不参与逻辑)
        if clean_labels[neighbor] == best_label:
            cnt += 1

    # --- 4. 构建输出结果 ---
    # 创建新的标签数组
    expanded_labels = original_labels.copy()
    # 确保如果是Tensor则先转为numpy，如果是numpy则直接copy，防止原地修改出错
    if isinstance(expanded_labels, torch.Tensor):
        expanded_labels = expanded_labels.clone()
    elif isinstance(expanded_labels, np.ndarray):
        expanded_labels = expanded_labels.copy()

    # 填入新标签
    for node, label in new_labels.items():
        expanded_labels[node] = label

    # 转换为排序后的numpy数组，作为新的训练集索引
    expanded_idx_train = np.array(sorted(expanded_labeled))

    # 打印扩展信息
    print(f"--- 标签扩充统计 (加权投票机制) ---")
    print(f"阈值 (Threshold): {threshold}")
    print(f"原始标签节点数: {len(idx_train)}")
    print(f"扩展后标签节点数: {len(expanded_idx_train)}")
    num_new = len(expanded_idx_train) - len(idx_train)
    print(f"新增标签节点数: {num_new}")

    acc = cnt / num_new if num_new > 0 else 0
    print(f"新增节点的标签准确率: {acc:.4f} ({cnt}/{num_new})")

    return expanded_idx_train, expanded_labels
'''
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_both(A, iA, jA, features, threshold1=2.5, threshold2=0.01):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C1 = np.linalg.norm(features[n1] - features[n2])

            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C2 = inner_product / (np.sqrt(np.square(a).sum() + np.square(b).sum())+ 1e-6)
            if C1 > threshold1 or threshold2 < 0:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            if (np.count_nonzero(a) + np.count_nonzero(b) - intersection) == 0:
                J = 0
            else:
                J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)
            if C <= threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def idx_to_mask(idx, nodes_num):
    """Convert a indices array to a tensor mask matrix
    Args:
        idx : numpy.array
            indices of nodes set
        nodes_num: int
            number of nodes
    """
    mask = torch.zeros(nodes_num)
    mask[idx] = 1
    return mask.bool()


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.
    Args:
        tensor : torch.Tensor
                 given tensor
    Returns:
        bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
