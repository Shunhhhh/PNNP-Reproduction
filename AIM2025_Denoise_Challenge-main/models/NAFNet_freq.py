import torch
import torch.nn as nn
import torch.fft as fft

class SimpleGate(nn.Module):
    def forward(self, x):
        # NAFNet 的核心：将通道分为两半，相乘
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FrequencyModulationModule(nn.Module):
    """
    频域调制模块：
    1. FFT 变换到频域
    2. 学习一个频率权重 (Magnitude Modulation)
    3. IFFT 变回空域
    """
    def __init__(self, channels):
        super().__init__()
        # 学习频域的幅度增益 (实数)，初始化为全1
        # 形状: [1, C, 1, 1] -> 广播到 [B, C, H, W] 的频谱上
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. FFT 
        x_fft = fft.fft2(x, norm='ortho')
        
        # 2. Shift zero-frequency component to center 
        x_fft_shifted = fft.fftshift(x_fft, dim=(-2, -1))
        
        # 3. Modulate Magnitude (只调制幅度，保留相位，或者简单地对复数进行缩放)
        x_modulated = x_fft_shifted * self.freq_weight
        
        # 4. IShift back
        x_fft_back = fft.ifftshift(x_modulated, dim=(-2, -1))
        
        # 5. IFFT
        x_rec = fft.ifft2(x_fft_back, norm='ortho')
        
        # 6. Take Real Part (由于数值误差可能有微小虚部，取实部)
        return torch.real(x_rec)


class FreqNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        
        dw_channel = c * DW_Expand
        
        # --- Spatial Branch  ---
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        
        # Simple Gate is applied after conv2 split
        
        # --- Frequency Branch ---
        # 我们在 Depthwise Conv 之后引入频域处理，或者并行处理
        # 这里采用并行分支：输入经过 Conv1 后，分两路
        # 一路走 Spatial (Conv2 + SG)，一路走 Frequency
        
        self.freq_module = FrequencyModulationModule(dw_channel)
        
        # Fusion Layer: 将空域和频域特征融合
        # 输入是 dw_channel (spatial) + dw_channel (freq) ? 
        # 或者更简单的：在 Simple Gate 之前加入频域信息
        
        # 1. Input -> Conv1 (Expand channels)
        # 2. Split into two paths:
        #    Path A: Conv2 (DW) -> Simple Gate
        #    Path B: FFT -> Modulate -> IFFT
        # 3. Add Path A and Path B
        # 4. Conv3 (Project back)
        

        # 修改 NAFBlock 的内部，在 DW Conv 之后，SG 之前，插入频域残差
        
        self.dw_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        
        # 频域分支独立处理 dw_channel
        self.freq_branch = FrequencyModulationModule(dw_channel)
        
        self.sg = SimpleGate()
        
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c * FFN_Expand, kernel_size=1, bias=True),
            nn.GELU(), # NAFNet 通常不用 GELU，但在 FFN 中有时会用，或者用 Simple Gate 再次替代
            # 为了纯粹 NAFNet 风格，FFN 也可以用 Simple Gate 结构，但这里简化为线性投影
            nn.Conv2d(in_channels=c * FFN_Expand, out_channels=c, kernel_size=1, bias=True)
        )
        
        # Layer Norms
        self.norm1 = LayerNorm(c, data_format='channels_first')
        self.norm2 = LayerNorm(c, data_format='channels_first')
        
        # Dropout
        self.drop_path = DropPath(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        
        # --- Part 1: Attention-like Block with Freq ---
        input_x = x
        x = self.norm1(x)
        
        # Expand channels
        x = self.conv1(x) # [B, 2C, H, W]
        
        # Depthwise Conv
        x_dw = self.dw_conv(x) # [B, 2C, H, W]
        
        # FFT 
        freq_feat = self.freq_branch(x_dw) # [B, 2C, H, W]
        
        # Combine Spatial and Freq
        combined = x_dw + freq_feat 
        
        # Simple Gate 
        x1_comb, x2_comb = combined.chunk(2, dim=1)
        x_sg = x1_comb * x2_comb # Simple Gate Output [B, C, H, W]
        
        # Scale by global context (SCA in NAFNet)
        sca = self.sca(x_sg)
        x_sg = x_sg * sca
        
        # Project back
        x_proj = self.conv3(x_sg) # [B, C, H, W]
        
        # Residual Connect
        x = input_x + self.drop_path(x_proj)
        
        # --- Part 2: FFN Block (Standard NAFNet FFN) ---
        input_x = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = input_x + self.drop_path(x)
        
        return x

# Helper Classes from NAFNet
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output

import torch.nn.functional as F