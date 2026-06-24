import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_orthogonal_loss(matrix: torch.Tensor) -> torch.Tensor:
    """
    严格的 Stiefel 正交损失: ||M^T M - I||_F^2
    强制列向量既单位化又彼此正交。
    matrix: [..., N, R]，要求 N >= R
    """
    # 兼容 batched / 非 batched
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    r = matrix.shape[-1]
    gram = torch.matmul(matrix.transpose(-2, -1), matrix)        # [..., R, R]
    eye = torch.eye(r, device=matrix.device, dtype=matrix.dtype)
    diff = gram - eye
    # 用平方 F-范数，数值更稳定，梯度更平滑
    loss = (diff ** 2).flatten(-2).sum(dim=-1)
    return loss.mean()


class RevIN(nn.Module):
    """
    可逆实例归一化 (Kim et al., ICLR 2022)
    解决 LTSF 中分布漂移问题，对纯线性模型尤其重要。
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        # x: [B, T, C]
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
            return x
        else:
            raise ValueError(mode)


class Model(nn.Module):
    """
    TimeTucker++  :  Pure Multilinear Tucker for LTSF

    改进点：
      1. RevIN 可逆实例归一化（推荐与 period_norm 二选一或叠加）
      2. 严格 Stiefel 正交损失 + 可选解码端共享因子
      3. 核张量 dropout 防过拟合
      4. pred_len 任意长度 (用 ceil 段数 + 截断)
      5. forward 始终返回 (output, aux_loss) 元组，接口统一
    """
    def __init__(self, configs):
        super().__init__()
        self.use_period_norm = getattr(configs, 'use_period_norm', True)
        self.use_revin       = getattr(configs, 'use_revin', True)
        self.use_orthogonal  = getattr(configs, 'use_orthogonal', True)
        self.share_factors   = getattr(configs, 'share_factors', True)  # 编解码共享因子

        self.pred_len   = configs.pred_len
        self.seq_len    = configs.seq_len
        self.enc_in     = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len  // self.period_len
        # 用 ceil 兼容 pred_len 不是 period_len 整数倍的情况
        self.seg_num_y = (self.pred_len + self.period_len - 1) // self.period_len

        # Tucker Ranks
        self.r_c = getattr(configs, 'r_c', min(self.enc_in, 8))
        self.r_n = getattr(configs, 'r_n', min(self.seg_num_x, 6))
        self.r_p = getattr(configs, 'r_p', min(self.period_len, 8))

        # --- 编码因子 (Stiefel 流形上) ---
        self.W_enc_var = nn.Parameter(torch.empty(self.enc_in,    self.r_c))
        self.W_enc_seg = nn.Parameter(torch.empty(self.seg_num_x, self.r_n))
        self.W_enc_per = nn.Parameter(torch.empty(self.period_len, self.r_p))

        # --- 解码因子 ---
        if self.share_factors:
            # var / per 维度共享 (转置即可)，只为段(时间)维新建解码因子
            self.W_dec_seg = nn.Parameter(torch.empty(self.r_n, self.seg_num_y))
        else:
            self.W_dec_var = nn.Parameter(torch.empty(self.r_c, self.enc_in))
            self.W_dec_seg = nn.Parameter(torch.empty(self.r_n, self.seg_num_y))
            self.W_dec_per = nn.Parameter(torch.empty(self.r_p, self.period_len))

        # 核张量上的可学习缩放 (类似 Tucker 的 G 核心)，提升表达力又几乎零代价
        self.core_scale = nn.Parameter(torch.ones(self.r_c, self.r_n, self.r_p))

        self.dropout = nn.Dropout(getattr(configs, 'dropout', 0.1))

        if self.use_revin:
            self.revin = RevIN(self.enc_in, affine=True)

        self._reset_parameters()

    def _reset_parameters(self):
        # 所有因子均用正交初始化，给优化一个良好的 Stiefel 起点
        for p in [self.W_enc_var, self.W_enc_seg, self.W_enc_per, self.W_dec_seg]:
            nn.init.orthogonal_(p)
        if not self.share_factors:
            nn.init.orthogonal_(self.W_dec_var)
            nn.init.orthogonal_(self.W_dec_per)

    # ---------- 主前向 ----------
    def forward(self, x):
        """
        x : [B, T, C]
        return : (pred [B, pred_len, C], aux_loss scalar)
        """
        B, T, C = x.shape

        # 1) RevIN 归一化
        if self.use_revin:
            x = self.revin(x, 'norm')

        # 2) 形态重构 -> [B, C, S, P]
        x_3d = x.permute(0, 2, 1).reshape(B, self.enc_in, self.seg_num_x, self.period_len)

        # 3) 周期均值 / 全局均值去趋势
        if self.use_period_norm:
            period_mean = x_3d.mean(dim=2, keepdim=True)   # [B,C,1,P]
            x_3d = x_3d - period_mean
        else:
            period_mean = x_3d.mean(dim=(2, 3), keepdim=True)
            x_3d = x_3d - period_mean

        # 4) 多线性收缩 -> 核张量 [B, r_c, r_n, r_p]
        core = torch.einsum('bcsp, ci, sj, pk -> bijk',
                            x_3d, self.W_enc_var, self.W_enc_seg, self.W_enc_per)
        core = core * self.core_scale          # 可学习的核心缩放
        core = self.dropout(core)              # 仅在核张量做 dropout，最小化噪声注入

        # 5) 多线性展开
        if self.share_factors:
            W_dec_var = self.W_enc_var.transpose(0, 1)   # [r_c, C]
            W_dec_per = self.W_enc_per.transpose(0, 1)   # [r_p, P]
        else:
            W_dec_var = self.W_dec_var
            W_dec_per = self.W_dec_per

        out_3d = torch.einsum('bijk, ic, jy, kp -> bcyp',
                              core, W_dec_var, self.W_dec_seg, W_dec_per)

        # 6) 去归一化（与编码端对称）
        out_3d = out_3d + period_mean

        # 7) 还原 2D
        x_out = out_3d.reshape(B, self.enc_in, -1).permute(0, 2, 1)  # [B, S_y*P, C]
        x_out = x_out[:, :self.pred_len, :]

        # 8) RevIN 反归一化
        if self.use_revin:
            x_out = self.revin(x_out, 'denorm')

        # 9) 正交损失（即使关闭也返回 0 张量，统一接口）
        if self.use_orthogonal:
            l_var = cal_orthogonal_loss(self.W_enc_var)
            l_seg = cal_orthogonal_loss(self.W_enc_seg)
            l_per = cal_orthogonal_loss(self.W_enc_per)
            aux_loss = l_var + l_seg + l_per
            if not self.share_factors:
                aux_loss = aux_loss + \
                           cal_orthogonal_loss(self.W_dec_var.transpose(0,1)) + \
                           cal_orthogonal_loss(self.W_dec_per.transpose(0,1))
        else:
            aux_loss = x_out.new_zeros(())

        return x_out, aux_loss