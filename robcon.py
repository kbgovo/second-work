import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.utils import *
from utils import sparse_dense_mul, to_tensor

class GCN_Contrastive(nn.Module):
    """
    共享的GCN编码器，用于生成节点嵌入
    对应论文中的 f_theta
    """
    def __init__(self, n_feat, n_hidden, dropout=0.5):
        super(GCN_Contrastive, self).__init__()
        self.fc1 = nn.Linear(n_feat, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden) # 输出维度保持为 hidden
        self.dropout = dropout
        self.act = nn.PReLU()

        # 初始化权重
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj, sparse=False):
        # Layer 1
        x = self.fc1(x)
        if sparse:
            x = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        else:
            x = torch.bmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2
        x = self.fc2(x)
        if sparse:
            x = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        else:
            x = torch.bmm(adj, x)
        
        # 论文建议对输出进行归一化，有助于对比学习稳定性
        # 但如果后续任务需要幅度信息，可考虑去掉
        return x

def compute_asymmetric_loss(z_fine, z_coarse, idx_train_F, idx_train_C, device, tau=0.5, gamma=0.5, beta=0.1):
    """
    计算非对称对比损失 (Asymmetric Gradient Alignment)
    论文公式: L_con
    """
    # 1. 构造权重 alpha_i (论文公式 78)
    # 细粒度集合 (包含原始标签 + 高置信伪标签) -> 权重 1.0
    # 粗粒度集合 \ 细粒度集合 -> 权重 gamma (例如 0.5)
    # 其他节点 -> 权重 beta (例如 0.1)

    if z_fine.dim() == 3:
        num_nodes = z_fine.shape[1]  # 取 N
    else:
        num_nodes = z_fine.shape[0]  # 取 N

    weights = torch.ones(num_nodes).to(device) * beta # 默认为 beta

    # 将 numpy 索引转为 tensor 或 set
    set_F = set(idx_train_F.tolist()) if isinstance(idx_train_F, np.ndarray) else set(idx_train_F)
    set_C = set(idx_train_C.tolist()) if isinstance(idx_train_C, np.ndarray) else set(idx_train_C)
    
    # 只有 C 中有但 F 中没有的，才是“低置信度”伪标签
    set_C_only = set_C - set_F
    
    # 转换为 list 以便索引 tensor
    list_F = list(set_F)
    list_C_only = list(set_C_only)
    
    if list_F:
        weights[list_F] = 1.0
    if list_C_only:
        weights[list_C_only] = gamma
        
    # 2. 梯度阻断 (Gradient Stop / Asymmetric)
    # 论文公式: l_pos = exp(sim(z^C, sg(z^F)) / tau)
    # 关键：z_fine 必须 detach，不传梯度
    z_fine_target = z_fine.detach() 
    
    # 3. 计算相似度矩阵 (Cosine Similarity)
    # z: (1, N, d) -> (N, d)
    z_c = z_coarse.squeeze(0)
    z_f = z_fine_target.squeeze(0)
    
    # 归一化以便计算余弦相似度
    z_c_norm = F.normalize(z_c, dim=1)
    z_f_norm = F.normalize(z_f, dim=1)
    
    # 相似度矩阵 (N, N)
    sim_matrix = torch.mm(z_c_norm, z_f_norm.t()) / tau
    
    # 4. InfoNCE Loss
    # 正样本: 对角线元素 (node i 的 coarse view 和 fine view)
    # exp(sim(i, i))
    exp_sim = torch.exp(sim_matrix)
    
    # 分子: 正样本
    pos_sim = torch.diag(exp_sim)
    
    # 分母: 所有样本 (batch 内所有其他节点的 fine view 都是负样本)
    # sum(exp(sim(i, k)))
    neg_sim_sum = exp_sim.sum(dim=1) 
    
    # Log-Softmax 形式: -log(pos / sum) = - (log_pos - log_sum)
    # 加上权重 alpha_i
    loss_per_node = -torch.log(pos_sim / (neg_sim_sum + 1e-8))
    
    # 加权平均
    weighted_loss = (loss_per_node * weights).mean()
    
    return weighted_loss

def get_contrastive_emb(logger, adj, features, idx_train_F, idx_train_C, 
                        lr=0.001, weight_decay=5e-4, nb_epochs=200, beta=1.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(features.shape) == 3:
        # 输入形状: (Batch, Nodes, Features) -> (1, N, D)
        n_feat = features.shape[2] 
    else:
        # 输入形状: (Nodes, Features) -> (N, D)
        n_feat = features.shape[1]
        features = features.unsqueeze(0) # 升维到 (1, N, D)
    
    if torch.is_tensor(features):
        features = features.to(device)
    else:
        features = torch.FloatTensor(features).to(device)
        
    n_hidden = 512  # 嵌入维度，可调整
    
    # 预处理图结构：A = A + I 并归一化
    # 注意：这里我们使用同一张图及其增强视图，或者简单地在同一张图上通过不同监督信号学习
    # 论文中提到 Shared GCN，输入是 Structure-aware smoothed features 或 raw features + adj
    # 这里我们简化为使用处理过的 adj
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    if torch.is_tensor(features):
        features = features.to(device)
    else:
        features = torch.FloatTensor(features).to(device)
        
    # 如果 features 是 (N, D)，变为 (1, N, D) 适配 batch 形式
    if len(features.shape) == 2:
        features = features.unsqueeze(0)

    # 初始化模型
    model = GCN_Contrastive(n_feat, n_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练循环
    model.train()
    best_loss = 1e9
    wait = 0
    patience = 20
    
    for epoch in range(nb_epochs):
        optimizer.zero_grad()
        
        # 前向传播：生成嵌入
        # 论文中，两个视图可能使用相同的图输入，但通过 Loss 区分
        # 或者你可以做轻微的 DropEdge 增强来区分 z_fine 和 z_coarse 的输入
        # 这里为了稳定，暂使用相同的输入，仅靠非对称梯度和权重区分
        z = model(features, sp_adj, sparse=True)
        
        # 这里的 z 既是 z_fine 也是 z_coarse 的来源 (Shared Encoder)
        # 在计算 loss 时，我们将 z 视为 z_coarse，将 z.detach() 视为 z_fine
        # 如果你想引入更多差异，可以在这里对 z 加不同的 Dropout mask
        z_coarse = z 
        z_fine = z   # 之后会在 loss 函数里 detach
        
        loss = compute_asymmetric_loss(
            z_fine, z_coarse, 
            idx_train_F, idx_train_C, 
            device=device,
            tau=0.5, gamma=0.5, beta=0.1 # 这些超参可根据论文微调
        )
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
            # 保存最佳模型权重
            best_state = model.state_dict()
        else:
            wait += 1
            if wait > patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
                
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 加载最佳模型并推理
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeds = model(features, sp_adj, sparse=True)
        
    return embeds.squeeze(0).cpu(), None # 返回 CPU 上的 Tensor

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
