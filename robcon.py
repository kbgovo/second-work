import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import copy
from deeprobust.graph.utils import *
from utils import sparse_dense_mul

'''
def get_contrastive_emb(logger, adj, features, adj_delete, lr, weight_decay, nb_epochs, beta, recover_percent=0.2):
    ft_size = features.shape[2]
    nb_nodes = features.shape[1]
    aug_features1 = features
    aug_features2 = features
    # 随机drop增强
    aug_adj1 = aug_random_edge(adj, adj_delete=adj_delete, recover_percent=recover_percent)  # random drop edges
    aug_adj2 = aug_random_edge(adj, adj_delete=adj_delete, recover_percent=recover_percent)  # random drop edges
    # 对原图和增强图进行归一化
    adj = normalize_adj(adj + (sp.eye(adj.shape[0]) * beta))
    aug_adj1 = normalize_adj2(aug_adj1 + (sp.eye(adj.shape[0]) * beta))
    aug_adj2 = normalize_adj2(aug_adj2 + (sp.eye(adj.shape[0]) * beta))
    sp_adj = sparse_mx_to_torch_sparse_tensor((adj))
    sp_aug_adj1 = sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = sparse_mx_to_torch_sparse_tensor(aug_adj2)
    model = DGI(ft_size, 512, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        features = features.cuda()
        aug_features1 = aug_features1.cuda()
        aug_features2 = aug_features2.cuda()
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        # 生成负样本：随机打乱节点顺序（用于构造负样本对）
        idx = np.random.permutation(nb_nodes)
        # 打乱后的特征（负样本特征）
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, aug_features1, aug_features2,
                       sp_adj, sp_aug_adj1, sp_aug_adj2,
                       True, None, None, None, aug_type='edge')
        loss = b_xent(logits, lbl)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            weights = copy.deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == 20:
            logger.info('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    logger.info('Loading {}th epoch'.format(best_t))
    model.load_state_dict(weights)

    return model.embed(features, sp_adj, True, None)
'''

def get_contrastive_emb(logger, adj, features, real_idx_train, expanded_idx_train,  # 新增expanded_idx_train参数
                        lr, weight_decay, nb_epochs, beta, mask_rate=0.5):  # 替换adj_delete为mask_rate
    ft_size = features.shape[2]
    nb_nodes = features.shape[1]
    # 标签增强：生成两个不同的标签视图（随机屏蔽部分标签节点）
    aug_labels1 = aug_random_label(real_idx_train, expanded_idx_train, mask_rate=mask_rate)  # 视图1：屏蔽部分标签
    aug_labels2 = aug_random_label(real_idx_train, expanded_idx_train, mask_rate=mask_rate)  # 视图2：屏蔽另一部分标签

    # 生成两个视图的特征掩码（仅保留当前视图的标签节点特征，其他置0）
    mask1 = torch.zeros(nb_nodes, dtype=torch.float32)
    mask1[aug_labels1] = 1.0  # 视图1保留的节点掩码
    mask2 = torch.zeros(nb_nodes, dtype=torch.float32)
    mask2[aug_labels2] = 1.0  # 视图2保留的节点掩码

    # 图结构不变（使用原始清洗后的adj），仅归一化一次
    adj = normalize_adj(adj + (sp.eye(adj.shape[0]) * beta))
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj)

    # 初始化DGI模型（保持不变）
    model = DGI(ft_size, 512, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        features = features.cuda()
        sp_adj = sp_adj.cuda()
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        # 生成负样本（保持不变：随机打乱特征）
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, features * mask1.reshape(1, -1, 1), features * mask2.reshape(1, -1, 1),
                       sp_adj, sp_adj, sp_adj,
                       True, None, None, None, aug_type='edge')
        loss = b_xent(logits, lbl)

        # 早停机制（保持不变）
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            weights = copy.deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == 20:
            logger.info('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    logger.info('Loading {}th epoch'.format(best_t))
    model.load_state_dict(weights)

    return model.embed(features, sp_adj, True, None)


def aug_random_edge(input_adj, adj_delete, recover_percent=0.2):
    percent = recover_percent
    adj_delete = sp.tril(adj_delete)
    row_idx, col_idx = adj_delete.nonzero()
    edge_num = int(len(row_idx))
    add_edge_num = int(edge_num * percent)
    print("the number of recovering edges: {:04d}" .format(add_edge_num))
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_list = [(i, j) for i, j in zip(row_idx, col_idx)]
    edge_idx = [i for i in range(edge_num)]
    add_idx = random.sample(edge_idx, add_edge_num)

    for i in add_idx:
        aug_adj[edge_list[i][0]][edge_list[i][1]] = 1
        aug_adj[edge_list[i][1]][edge_list[i][0]] = 1


    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj

'''
def aug_random_label(real_idx_train, expanded_idx_train, mask_rate=0.2):
    """
    随机屏蔽扩展后标签节点中的部分节点，生成新的训练节点子集
    :param expanded_idx_train: 扩展后的标签节点索引（np.array）
    :param mask_rate: 屏蔽比例（删除多少比例的标签节点）
    :return: 新的训练节点子集（np.array）
    """
    # 计算需要屏蔽的节点数量
    total = len(expanded_idx_train)
    mask_num = int(total * mask_rate)
    # 随机选择要屏蔽的节点索引
    mask_indices = np.random.choice(total, size=mask_num, replace=False)
    # 保留未被屏蔽的节点（新的训练集）
    new_idx_train = np.delete(expanded_idx_train, mask_indices)
    print(f"屏蔽 {mask_num} 个标签节点，剩余 {len(new_idx_train)} 个")
    return new_idx_train
'''

def aug_random_label(original_idx_train, expanded_idx_train, mask_rate=0.5):
    """
    仅随机屏蔽扩展后新增的伪标签节点，保留所有原始标签节点
    :param original_idx_train: 原始标签节点索引（无伪标签，np.array）
    :param expanded_idx_train: 扩展后标签节点索引（包含原始+伪标签，np.array）
    :param mask_rate: 伪标签节点的屏蔽比例（0~1，默认屏蔽20%伪标签）
    :return: 新的训练节点子集（原始标签节点 + 未被屏蔽的伪标签节点，np.array）
    """
    # 步骤1：区分原始标签节点和伪标签节点（核心）
    original_labeled = set(original_idx_train.tolist())
    expanded_labeled = set(expanded_idx_train.tolist())
    # 伪标签节点 = 扩展后节点 - 原始节点
    pseudo_labeled = expanded_labeled - original_labeled
    pseudo_labeled = np.array(sorted(pseudo_labeled))  # 转为有序数组
    pseudo_num = len(pseudo_labeled)

    if pseudo_num == 0:
        print("无新增伪标签节点，直接返回原始标签节点")
        return original_idx_train

    # 步骤2：计算需要屏蔽的伪标签节点数量（不影响原始节点）
    mask_num = int(pseudo_num * mask_rate)
    if mask_num == 0:
        print("屏蔽比例过低，未屏蔽任何伪标签节点")
        return expanded_idx_train
    if mask_num >= pseudo_num:
        mask_num = pseudo_num - 1  # 避免屏蔽所有伪标签（至少保留1个）

    # 步骤3：随机选择要屏蔽的伪标签节点索引
    mask_indices = np.random.choice(pseudo_num, size=mask_num, replace=False)
    masked_pseudo = pseudo_labeled[mask_indices]  # 被屏蔽的伪标签节点
    remaining_pseudo = np.delete(pseudo_labeled, mask_indices)  # 未被屏蔽的伪标签节点

    # 步骤4：新训练集 = 原始标签节点 + 未被屏蔽的伪标签节点
    new_idx_train = np.concatenate([original_idx_train, remaining_pseudo])
    new_idx_train = np.sort(new_idx_train)  # 保持索引有序（可选，增强可读性）

    # 打印日志（明确屏蔽细节）
    print(f"原始标签节点数：{len(original_idx_train)}（全部保留）")
    print(f"伪标签节点数：{pseudo_num} → 屏蔽 {mask_num} 个，剩余 {len(remaining_pseudo)} 个")
    print(f"新训练集总节点数：{len(new_idx_train)}")

    return new_idx_train
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj2(adj, alpha=-0.5):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.add(torch.eye(adj.shape[0]), adj)
    degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    if alpha > 0:
        return to_scipy((adj / (adj.sum(dim=1).reshape(adj.shape[0], -1)))).tocoo()
    else:
        return to_scipy(adj).tocoo()


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN_DGI(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    # (features, shuf_fts, aug_features1, aug_features2,
    #  sp_adj if sparse else adj,
    #  sp_aug_adj1 if sparse else aug_adj1,
    #  sp_aug_adj2 if sparse else aug_adj2,
    #  sparse, None, None, None, aug_type=aug_type
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):
        h_0 = self.gcn(seq1, adj, sparse)
        if aug_type == 'edge':

            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = self.gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class GCN_DGI(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_DGI, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        tmp = self.f_k(h_pl, c_x)
        sc_1 = torch.squeeze(tmp, 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits