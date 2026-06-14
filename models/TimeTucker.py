import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_orthogonal_loss(matrix):
    """
    计算矩阵的非对角线正交损失（Feature Decorrelation / 特征解耦）。
    【还原至原版精妙逻辑】：
    仅惩罚非对角线元素（迫使特征相互独立/正交），
    绝对不干涉对角线元素（保留时间序列特征的真实模长与振幅能量）。
    """
    # 1. 计算 Gram 矩阵: A^T A
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix) 
    
    # 2. 提取并构造对角线矩阵
    diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    diag_matrix = torch.diag_embed(diag)
    
    # 3. 仅保留非对角线元素（扣除对角线）
    off_diagonal = gram_matrix - diag_matrix
    
    # 4. 计算非对角线元素的 Frobenius 范数
    loss = torch.norm(off_diagonal, dim=(-2, -1)) 
    
    return loss.mean()


class Model(nn.Module):
    """
    TimeTucker: 3D Tensor-based Low-Rank Architecture for Long-term Time Series Forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_period_norm = configs.use_period_norm
        self.use_orthogonal = configs.use_orthogonal

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0

        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError('seq_len and pred_len must be positive integers.')
        if self.enc_in <= 0:
            raise ValueError('enc_in must be a positive integer.')
        if self.period_len <= 0:
            raise ValueError('period_len must be a positive integer.')

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 序列 Padding 逻辑
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1

        # ==========================================
        # 安全获取张量秩 r_n 和 r_c (保留了安全的默认值机制)
        # ==========================================
        configured_r_n = getattr(configs, 'r_n', None)
        configured_r_c = getattr(configs, 'r_c', None)
        
        self.r_n = configured_r_n if configured_r_n is not None else 6  
        self.r_c = configured_r_c if configured_r_c is not None else min(self.enc_in, 16) 

        if self.r_n <= 0 or self.r_c <= 0:
            raise ValueError('r_n and r_c must be positive integers.')
        if self.r_n > self.seg_num_x:
            raise ValueError(f'r_n ({self.r_n}) must be <= seg_num_x ({self.seg_num_x}).')
        if self.r_c > self.enc_in:
            raise ValueError(f'r_c ({self.r_c}) must be <= enc_in ({self.enc_in}).')

        # ==========================================
        # 核心参数：3D 张量投影矩阵 (Tucker Decomposition Basis)
        # ==========================================
        self.W_seg = nn.Parameter(torch.Tensor(self.seg_num_x, self.r_n))
        self.W_var = nn.Parameter(torch.Tensor(self.enc_in, self.r_c))

        self.W_pred_seg = nn.Parameter(torch.Tensor(self.r_n, self.seg_num_y))
        self.W_pred_var = nn.Parameter(torch.Tensor(self.r_c, self.enc_in))

        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.gelu

        self._reset_parameters()

    def _reset_parameters(self):
        """对静态提取矩阵使用正交初始化，赋予良好的等距映射起点"""
        nn.init.orthogonal_(self.W_seg)
        nn.init.orthogonal_(self.W_var)
        nn.init.xavier_uniform_(self.W_pred_seg)
        nn.init.xavier_uniform_(self.W_pred_var)

    def _normalize_input(self, x, b, c):
        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - period_mean
            return x, {'period_mean': period_mean}
        else:
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {'mean': mean}

    def _denormalize_output(self, x, norm_stats, b, c):
        if self.use_period_norm:
            x = x + norm_stats['period_mean']
        else:
            x = x.reshape(b, c, -1)
            x = x + norm_stats['mean']
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(self, x):
        '''
        x: [B, T, C]
        '''
        b, t, c = x.shape
        batch_size = b
        x = x.permute(0, 2, 1)  # [B, C, T]

        # 1. Padding 填充
        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            x = torch.cat([x, x[:, :, pad_start - self.pad_seq_len:pad_start]], dim=-1)

        # 2. 转换为规范形态用于归一化
        x_norm_shape = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x_norm_shape = x_norm_shape.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.seg_num_x)
        x_norm, norm_stats = self._normalize_input(x_norm_shape, b, c)

        # ==========================================
        # 3. 核心计算流：3D 建模与 Tucker 张量收缩
        # ==========================================
        x_3d = x_norm.reshape(b, self.enc_in, self.period_len, self.seg_num_x).permute(0, 3, 2, 1)

        # X_his x_1 W_seg x_3 W_var
        core_tensor = torch.einsum('bnpc, nr, cd -> brpd', x_3d, self.W_seg, self.W_var)

        core_tensor = self.dropout(self.activation(core_tensor))

        # 核心张量驱动未来分段重建
        out_3d = torch.einsum('brpd, rm, dc -> bmpc', core_tensor, self.W_pred_seg, self.W_pred_var)


        # ==========================================
        # 4. 展平与反归一化
        # ==========================================
        x_out = out_3d.permute(0, 3, 2, 1).reshape(-1, self.period_len, self.seg_num_y)
        x_out = self._denormalize_output(x_out, norm_stats, b, c)

        x_out = x_out.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y).permute(0, 1, 3, 2)
        x_out = x_out.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        # ==========================================
        # 5. 三重正交约束机制 
        # ==========================================
        if self.use_orthogonal:
            loss_seg = cal_orthogonal_loss(self.W_seg.unsqueeze(0))
            loss_var = cal_orthogonal_loss(self.W_var.unsqueeze(0))
            
            # 使用 transpose 确保矩阵形态符合 gram_matrix 计算
            core_unfold_T = core_tensor.reshape(b, self.r_n, -1).transpose(-2, -1) 
            loss_core = cal_orthogonal_loss(core_unfold_T) 
            
            # 合并损失
            orthogonal_loss = loss_seg + loss_var + 0.5 * loss_core
            
            return x_out[:, :self.pred_len, :], orthogonal_loss
        else:
            return x_out[:, :self.pred_len, :]