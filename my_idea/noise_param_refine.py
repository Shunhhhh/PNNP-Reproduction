"""
NoiseParamRefineNet: 基于 residual 结构修正噪声参数估计。

核心思想：
  正确的 alpha/sigma2 归一化后的 residual 应该无结构（白噪声）。
  如果 residual 有结构（行相关、亮度相关、通道不平衡），
  说明参数估计不准，网络通过消除这些结构来学习修正量。

输入特征（全部从 G1-G2 差图和初步参数计算，不需要 GT）：
  1. 亮度相关特征：局部方差 vs 局部均值的偏差
  2. 行相关特征：行方差的不均匀性
  3. 通道特征：G1/G2 残差的不平衡
  4. 频域特征：条带噪声的频率响应

输出：
  delta_log_alpha, delta_log_sigma2（log 域修正量）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ────────────────────────────────────────────────
# 特征提取
# ────────────────────────────────────────────────

def extract_residual_features(
    G1: torch.Tensor,
    G2: torch.Tensor,
    alpha: torch.Tensor,
    sigma2: torch.Tensor,
    patch_size: int = 16,
    n_bins: int = 8,
) -> torch.Tensor:
    B, _, H, W = G1.shape

    diff = G1 - G2
    mean = (G1 + G2) / 2.0

    var_map = (
        alpha[:, None, None, None] * mean.clamp(min=0)
        + sigma2[:, None, None, None]
    ).clamp(min=1e-10)
    residual = diff / torch.sqrt(2.0 * var_map)   # [B, 1, H, W]

    features = []

    # ── 特征 1：亮度分箱 residual 方差（向量化）──────────────────
    mean_flat = mean.reshape(B, -1)        # [B, HW]
    res_flat  = residual.reshape(B, -1)    # [B, HW]

    brightness_bias = torch.zeros(B, n_bins, device=G1.device)
    for b in range(n_bins):
        low  = b       / n_bins
        high = (b + 1) / n_bins
        mask = (mean_flat >= low) & (mean_flat < high)   # [B, HW]
        # 分子：masked sum of squares；分母：count
        count    = mask.float().sum(dim=1).clamp(min=1)           # [B]
        res_sum  = (res_flat * mask.float()).sum(dim=1)            # [B]
        res_mean = res_sum / count                                 # [B]
        res2_sum = ((res_flat - res_mean.unsqueeze(1)) ** 2 * mask.float()).sum(dim=1)
        bin_var  = torch.where(count > 10, res2_sum / count, torch.ones_like(count))
        brightness_bias[:, b] = bin_var - 1.0
    features.append(brightness_bias)                               # [B, n_bins]

    # ── 特征 2：行相关性 ──────────────────────────────────────────
    row_mean = residual.mean(dim=3)                                # [B, 1, H]
    row_mean = row_mean - row_mean.mean(dim=2, keepdim=True)
    row_var      = row_mean.var(dim=2).squeeze(1)                  # [B]
    row_std_feat = row_mean.squeeze(1).std(dim=1)                  # [B]
    features.append(row_var.unsqueeze(1))
    features.append(row_std_feat.unsqueeze(1))

    # ── 特征 3：局部方差 vs 预测方差（向量化）────────────────────
    pH = H // patch_size
    pW = W // patch_size
    H_crop = pH * patch_size
    W_crop = pW * patch_size

    diff_crop = diff[:, :, :H_crop, :W_crop]
    mean_crop = mean[:, :, :H_crop, :W_crop]

    diff_patches = (
        diff_crop.reshape(B, pH, patch_size, pW, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, pH * pW, -1)
    )
    mean_patches = (
        mean_crop.reshape(B, pH, patch_size, pW, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, pH * pW, -1)
    )

    local_var  = diff_patches.var(dim=2) / 2.0          # [B, N]
    local_mean = mean_patches.mean(dim=2)               # [B, N]
    pred_var   = alpha[:, None] * local_mean.clamp(min=0) + sigma2[:, None]

    log_ratio = (
        torch.log(local_var.clamp(min=1e-10))
        - torch.log(pred_var.clamp(min=1e-10))
    )                                                    # [B, N]

    log_ratio_mean = log_ratio.mean(dim=1, keepdim=True)
    log_ratio_std  = log_ratio.std(dim=1, keepdim=True)

    # 每个 batch 独立算中位数（修复原版 batch 共用阈值的 bug）
    threshold   = local_mean.median(dim=1, keepdim=True).values   # [B, 1]
    bright_mask = (local_mean > threshold).float()                 # [B, N]
    dark_mask   = 1.0 - bright_mask

    bright_count = bright_mask.sum(dim=1, keepdim=True).clamp(min=1)
    dark_count   = dark_mask.sum(dim=1,   keepdim=True).clamp(min=1)
    log_ratio_bright = (log_ratio * bright_mask).sum(dim=1, keepdim=True) / bright_count
    log_ratio_dark   = (log_ratio * dark_mask  ).sum(dim=1, keepdim=True) / dark_count

    features.append(log_ratio_mean)
    features.append(log_ratio_std)
    features.append(log_ratio_bright)
    features.append(log_ratio_dark)

    # ── 特征 4：频域条带检测 ──────────────────────────────────────
    res_col_mean = residual.mean(dim=3).squeeze(1)       # [B, H]
    fft_mag = torch.fft.rfft(res_col_mean, dim=1).abs()  # [B, H//2+1]
    n_freq = fft_mag.shape[1]
    low_freq_energy  = fft_mag[:, 1:n_freq // 4].mean(dim=1, keepdim=True)
    high_freq_energy = fft_mag[:, n_freq // 4:].mean(dim=1, keepdim=True)
    freq_ratio = low_freq_energy / high_freq_energy.clamp(min=1e-10)
    features.append(freq_ratio)

    # ── 特征 5：初步参数先验 ─────────────────────────────────────
    log_alpha  = torch.log(alpha.clamp(min=1e-6)).unsqueeze(1)
    log_sigma2 = torch.log(sigma2.clamp(min=1e-10)).unsqueeze(1)
    features.append(log_alpha)
    features.append(log_sigma2)

    return torch.cat(features, dim=1)   # [B, F]
# ────────────────────────────────────────────────
# 修正网络
# ────────────────────────────────────────────────

class NoiseParamRefineNet(nn.Module):
    def __init__(
        self,
        n_bins: int = 8,
        hidden_dim: int = 64,
        delta_scale: float = 1.5,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.delta_scale = delta_scale
        self.net = None  # 延迟初始化，第一次 forward 时根据实际特征维度构建

    def _build_net(self, in_dim: int, device):
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Tanh(),
        ).to(device)
        # 用小随机初始化代替零初始化
        # 让网络初始就有少量修正信号，梯度能正常流动
        nn.init.normal_(self.net[-2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-2].bias)   # bias 保持 0 没问题

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            self._build_net(features.shape[1], features.device)
        return self.net(features) * self.delta_scale

    def refine(
        self,
        G1: torch.Tensor,
        G2: torch.Tensor,
        alpha: torch.Tensor,
        sigma2: torch.Tensor,
    ):
        features = extract_residual_features(G1, G2, alpha, sigma2)
        delta = self.forward(features)
        alpha_final = torch.exp(torch.log(alpha.clamp(min=1e-6)) + delta[:, 0])
        sigma2_final = torch.exp(torch.log(sigma2.clamp(min=1e-10)) + delta[:, 1])
        return alpha_final, sigma2_final

# ────────────────────────────────────────────────
# 自监督损失函数
# ────────────────────────────────────────────────

def residual_whitening_loss(
    G1, G2, alpha, sigma2,
    patch_size=16, flat_percentile=0.5,
):
    B, _, H, W = G1.shape
    diff = G1 - G2
    mean = (G1 + G2) / 2.0

    pH, pW = H // patch_size, W // patch_size
    H_crop, W_crop = pH * patch_size, pW * patch_size

    diff_p = (diff[:, :, :H_crop, :W_crop]
              .reshape(B, pH, patch_size, pW, patch_size)
              .permute(0, 1, 3, 2, 4).reshape(B, pH * pW, -1))
    mean_p = (mean[:, :, :H_crop, :W_crop]
              .reshape(B, pH, patch_size, pW, patch_size)
              .permute(0, 1, 3, 2, 4).reshape(B, pH * pW, -1))

    local_var  = diff_p.var(dim=2) / 2.0
    local_mean = mean_p.mean(dim=2)
    local_flatness = mean_p.var(dim=2)

    flat_threshold = local_flatness.quantile(flat_percentile, dim=1, keepdim=True)
    flat_mask = (local_flatness < flat_threshold).float()

    pred_var = (alpha[:, None] * local_mean.clamp(min=0)
                + sigma2[:, None]).clamp(min=1e-10)

    # clamp 防止单个 patch 爆炸，smooth_l1 替代 pow(2) 降低大值的影响
    log_ratio = (torch.log(pred_var)
                 - torch.log(local_var.clamp(min=1e-10))).clamp(-3.0, 3.0)
    loss = F.smooth_l1_loss(
        log_ratio * flat_mask,
        torch.zeros_like(log_ratio),
        reduction='sum',
        beta=0.5,
    ) / flat_mask.sum().clamp(min=1)
    return loss


def brightness_correlation_loss(
    G1, G2, alpha, sigma2, n_bins=8,
):
    diff = G1 - G2
    mean = (G1 + G2) / 2.0
    B = G1.shape[0]

    var_map = (alpha[:, None, None, None] * mean.clamp(min=0)
               + sigma2[:, None, None, None]).clamp(min=1e-10)
    residual = diff / torch.sqrt(2.0 * var_map)

    mean_flat = mean.reshape(B, -1)
    res_flat  = residual.reshape(B, -1)

    bin_losses = []
    for b in range(n_bins):
        low  = b       / n_bins
        high = (b + 1) / n_bins
        mask  = ((mean_flat >= low) & (mean_flat < high)).float()
        count = mask.sum(dim=1).clamp(min=1)

        res_sum  = (res_flat * mask).sum(dim=1)
        res_mean = res_sum / count
        res2_sum = ((res_flat - res_mean.unsqueeze(1)).pow(2) * mask).sum(dim=1)
        bin_var  = res2_sum / count

        valid    = (count > 10).float()
        # smooth_l1 替代 pow(2)，降低离群 bin 的影响
        bin_loss = F.smooth_l1_loss(
            bin_var * valid,
            torch.ones_like(bin_var) * valid,
            reduction='mean',
            beta=0.5,
        )
        bin_losses.append(bin_loss)

    return torch.stack(bin_losses).mean()


def row_whitening_loss(
    G1: torch.Tensor,
    G2: torch.Tensor,
    alpha: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    # 这个函数本身没问题，保持不变
    diff = G1 - G2
    mean = (G1 + G2) / 2.0

    var_map = (alpha[:, None, None, None] * mean.clamp(min=0)
               + sigma2[:, None, None, None]).clamp(min=1e-10)
    residual = diff / torch.sqrt(2.0 * var_map)

    row_mean = residual.mean(dim=3)
    row_mean = row_mean - row_mean.mean(dim=2, keepdim=True)
    return row_mean.pow(2).mean()


def param_regularization_loss(
    alpha: torch.Tensor,
    sigma2: torch.Tensor,
    alpha_init: torch.Tensor,
    sigma2_init: torch.Tensor,
) -> torch.Tensor:
    # 没有问题，保持不变
    loss_alpha = F.mse_loss(
        torch.log(alpha.clamp(min=1e-6)),
        torch.log(alpha_init.clamp(min=1e-6).detach()),
    )
    loss_sigma2 = F.mse_loss(
        torch.log(sigma2.clamp(min=1e-10)),
        torch.log(sigma2_init.clamp(min=1e-10).detach()),
    )
    return loss_alpha + loss_sigma2

def refine_loss(
    G1: torch.Tensor,
    G2: torch.Tensor,
    alpha_refined: torch.Tensor,
    sigma2_refined: torch.Tensor,
    alpha_init: torch.Tensor,
    sigma2_init: torch.Tensor,
    w_whitening: float = 1.0,
    w_row: float = 0.5,
    w_brightness: float = 0.5,
    w_reg: float = 0.1,
) -> torch.Tensor:
    """
    修正网络的总损失。
    """
    loss = (
        w_whitening * residual_whitening_loss(G1, G2, alpha_refined, sigma2_refined)
        + w_row * row_whitening_loss(G1, G2, alpha_refined, sigma2_refined)
        + w_brightness * brightness_correlation_loss(G1, G2, alpha_refined, sigma2_refined)
        + w_reg * param_regularization_loss(alpha_refined, sigma2_refined, alpha_init, sigma2_init)
    )
    return loss