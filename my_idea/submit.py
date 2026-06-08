"""
  SIDD_Benchmark_Submit_Raw.py
  使用 ConditionalNAFNet 在 SIDD Benchmark 上生成 SubmitRaw.mat

  用法:
      python SIDD_Benchmark_Submit_Raw.py \
          --ckpt ./checkpoints/SIDD_Noisier2Noise/best.pth \
          --data_dir /path/to/SIDD_Benchmark_Data \
          --blocks_mat ./BenchmarkBlocks32.mat \
          --out_dir ./Submit
  """

import os
import sys
import time
import argparse
import numpy as np
import scipy.io as sio


sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..")
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
from torch import nn

from my_idea.conditional_denoiser import ConditionalNAFNet


# =========================================================
# 1. NLF → alpha, sigma2
# =========================================================
def nlf_to_noise_params(nlf):
    """
    从 SIDD metadata 的 NLF (Noise Level Function) 中拟合 Poisson-Gaussian 噪声参数

    NLF 格式: [N, 2] 矩阵, 第一列 = 强度(intensity), 第二列 = 噪声标准差(std)

    Poisson-Gaussian 噪声模型:
        variance = a * intensity + b
        其中 a  = Poisson / shot noise 参数 → alpha
            b  = Gaussian / read noise 方差 → sigma2

    返回 (alpha, sigma2)，均为标量 float
    """
    # nlf 可能是 MATLAB 存的结构体, 先尝试常规提取
    if isinstance(nlf, np.ndarray) and nlf.ndim == 2 and nlf.shape[1] == 2:
        intensities = nlf[:, 0]
        stds = nlf[:, 1]
    else:
        # 兜底: 可能嵌套在奇怪的结构里
        raise ValueError(f"Unexpected NLF format: type={type(nlf)}, shape={getattr(nlf, 'shape', 'N/A')}")

    # 过滤掉零强度点（避免除零）
    mask = intensities > 0.001
    intensities = intensities[mask]
    stds = stds[mask]

    if len(intensities) < 2:
        return 0.0, float(np.mean(stds ** 2))

    variances = stds ** 2

    # 最小二乘拟合: variance = a * intensity + b
    A = np.stack([intensities, np.ones_like(intensities)], axis=1)
    params, residuals, rank, s = np.linalg.lstsq(A, variances, rcond=None)

    a = max(float(params[0]), 0.0)  # Poisson 参数, 非负
    b = max(float(params[1]), 1e-10)  # Gaussian 方差, 正数

    return a, b


# =========================================================
# 2. Bayer 打包 / 解包
# =========================================================
def bayer_to_4ch(bayer):
    """
    单通道 Bayer (RGGB) → 4 通道 packed RAW [H/2, W/2, 4]

    Bayer pattern:
        R  Gr      →  channel 0
        Gb B       →  channel 1
    """
    H, W = bayer.shape
    H2, W2 = H // 2, W // 2

    packed = np.zeros((4, H2, W2), dtype=np.float32)
    packed[0, :, :] = bayer[0:H:2, 0:W:2]  # R
    packed[1, :, :] = bayer[0:H:2, 1:W:2]  # Gr
    packed[2, :, :] = bayer[1:H:2, 0:W:2]  # Gb
    packed[3, :, :] = bayer[1:H:2, 1:W:2]  # B

    return packed


def ch4_to_bayer(packed):
    """
    4 通道 packed RAW [4, H/2, W/2] → 单通道 Bayer (RGGB)
    """
    C, H2, W2 = packed.shape
    H, W = H2 * 2, W2 * 2

    bayer = np.zeros((H, W), dtype=np.float32)
    bayer[0:H:2, 0:W:2] = packed[0, :, :]
    bayer[0:H:2, 1:W:2] = packed[1, :, :]
    bayer[1:H:2, 0:W:2] = packed[2, :, :]
    bayer[1:H:2, 1:W:2] = packed[3, :, :]

    return bayer


# =========================================================
# 3. 模型推理（处理任意尺寸，padding 到 16 的倍数）
# =========================================================
def pad_to_multiple(x, multiple=16):
    """padding 到 multiple 的倍数，返回 (padded, pad_h, pad_w, orig_h, orig_w)"""
    _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0, h, w
    padded = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return padded, pad_h, pad_w, h, w


def denoise_block(model, noisy_block_4ch, alpha, sigma2, device):
    """
    对单个 block（4ch packed）进行降噪

    Args:
        noisy_block_4ch: np.ndarray [4, H2, W2], float32, 范围 ~[0, 1]
        alpha:  float, Poisson 噪声参数
        sigma2: float, Gaussian 噪声方差
        device: torch.device

    Returns:
        denoised_block_4ch: np.ndarray [4, H2, W2], float32
    """
    # padding
    padded, pad_h, pad_w, orig_h, orig_w = pad_to_multiple(noisy_block_4ch, multiple=16)

    # to tensor
    x = torch.from_numpy(padded).unsqueeze(0).to(device)           # [1, 4, H2, W2]
    a = torch.tensor([alpha], device=device).float()                # [1]
    s = torch.tensor([sigma2], device=device).float()               # [1]

    with torch.no_grad():
        out = model(x, a, s)                                        # [1, 4, H2', W2']

    out = out.squeeze(0).cpu().numpy()                              # [4, H2', W2']

    # crop back to original size
    if pad_h > 0 or pad_w > 0:
        out = out[:, :orig_h, :orig_w]

    return out


# =========================================================
# 4. 主流程
# =========================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 加载模型 ----
    model = ConditionalNAFNet(
        img_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    # ---- 加载 BenchmarkNoisyBlocksRaw.mat ----
    # 结构: BenchmarkNoisyBlocksRaw[i, j] = ndarray [H, W], 共 40 张图 × 32 blocks
    noisy_data = sio.loadmat(args.noisy_mat)
    noisy_blocks = noisy_data["BenchmarkNoisyBlocksRaw"]  # shape: (40, 32), dtype=object
    n_images, n_blocks = noisy_blocks.shape
    print(f"Loaded noisy blocks: {n_images} images × {n_blocks} blocks")

    # ---- 降噪 ----
    DenoisedBlocksRaw = np.empty((n_images, n_blocks), dtype=object)
    TimeMP = 0.0

    # 没有 metadata 时使用固定的默认噪声参数
    # 如果有 metadata 可以按需扩展
    default_alpha  = 0.01
    default_sigma2 = 1e-4

    for img_idx in range(n_images):
        print(f"\nImage {img_idx + 1}/{n_images}")
        for blk_idx in range(n_blocks):
            noisy_block = noisy_blocks[img_idx, blk_idx].astype(np.float32)  # [H, W]

            # 归一化到 [0, 1]
            noisy_block_norm = noisy_block / args.raw_max_val
            noisy_block_norm = np.clip(noisy_block_norm, 0.0, 1.0)

            # Bayer → 4ch packed
            noisy_block_4ch = bayer_to_4ch(noisy_block_norm)  # [4, H/2, W/2]

            # 降噪
            t0 = time.time()
            denoised_block_4ch = denoise_block(
                model, noisy_block_4ch, default_alpha, default_sigma2, device
            )
            t1 = time.time()
            block_time = t1 - t0

            # 4ch packed → Bayer
            denoised_block_bayer = ch4_to_bayer(denoised_block_4ch)

            # 反归一化
            denoised_block_bayer = denoised_block_bayer * args.raw_max_val
            denoised_block_bayer = np.clip(denoised_block_bayer, 0, args.raw_max_val)

            DenoisedBlocksRaw[img_idx, blk_idx] = denoised_block_bayer.astype(np.float32)
            TimeMP += block_time

            print(f"  Block {blk_idx + 1}/{n_blocks}: time={block_time:.3f}s, "
                  f"range=[{denoised_block_bayer.min():.1f}, {denoised_block_bayer.max():.1f}]")

    # ---- 计算 TimeMP ----
    sample_block = noisy_blocks[0, 0]
    blk_h, blk_w = sample_block.shape
    n_total_pixels = n_images * n_blocks * blk_h * blk_w
    TimeMP = TimeMP * 1e6 / n_total_pixels
    print(f"\nTimeMP: {TimeMP:.4f} s/MP")

    # ---- 保存 SubmitRaw.mat ----
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "SubmitRaw.mat")
    sio.savemat(
        out_path,
        {
            "DenoisedBlocksRaw": DenoisedBlocksRaw,
            "TimeMPRaw": TimeMP,
            "OptionalData": {
                "MethodName": args.method_name,
                "Authors": args.authors,
                "PaperTitle": args.paper_title,
                "Venue": args.venue,
                "MachineSpecs": args.machine_specs,
            },
        },
        do_compression=True,
    )
    print(f"Saved to: {out_path}")

# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/SIDD_Noisier2Noise/best.pth",
                        help="训练好的模型 checkpoint 路径")
    parser.add_argument("--noisy_mat", type=str, default="../../data/xml196414/SIDD/SIDD_Benchmark/BenchmarkNoisyBlocksRaw.mat",
                        help="SIDD Benchmark Data 目录路径")
    parser.add_argument("--blocks_mat", type=str, default="../../data/xml196414/SIDD/SIDD_Benchmark/BenchmarkBlocks32.mat",
                        help="BenchmarkBlocks32.mat 路径")
    parser.add_argument("--out_dir", type=str, default="./Submit",
                        help="输出目录")
    parser.add_argument("--raw_max_val", type=float, default=16383.0,
                        help="RAW 数据最大值 (用于归一化), 10-bit=1023, 14-bit=16383")
    parser.add_argument("--method_name", type=str, default="ConditionalNAFNet-Noisier2Noise")
    parser.add_argument("--authors", type=str, default="xml")
    parser.add_argument("--paper_title", type=str, default="")
    parser.add_argument("--venue", type=str, default="")
    parser.add_argument("--machine_specs", type=str, default="NVIDIA GPU")
    args = parser.parse_args()

    main(args)