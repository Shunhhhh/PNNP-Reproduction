import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.pnnp_train_dataset import NoiseSynthesisDataset
import os

def diagnose_noise_gap(
    real_noisy_path,
    real_clean_path,
    dataset,
    iso,
    top=0, left=0,
    patch_size=512,
):
    # ── 真实噪声 ──────────────────────────────────────────────
    real_noisy = torch.from_numpy(np.load(real_noisy_path).astype(np.float32))
    real_clean  = torch.from_numpy(np.load(real_clean_path).astype(np.float32))

    print("real_noisy raw shape:", real_noisy.shape)
    print("real_noisy min/max:", real_noisy.min().item(), real_noisy.max().item())

    # npy 已经是 pack 后的 [4, H, W]，直接裁剪
    assert real_noisy.ndim == 3 and real_noisy.shape[0] == 4, \
        f"Expected [4, H, W], got {real_noisy.shape}"

    real_noisy_pack = real_noisy[:, top:top+patch_size, left:left+patch_size]
    real_clean_pack  = real_clean[:,  top:top+patch_size, left:left+patch_size]

    print("pack shape:", real_noisy_pack.shape)

    # 判断值域：已归一化[0,1] 直接用，ADU 值域才 normalize
    if real_noisy.max() <= 2.0:
        print("检测到已归一化数据，直接使用")
        real_noisy_n = real_noisy_pack
        real_clean_n  = real_clean_pack
    else:
        print("检测到 ADU 数据，执行归一化")
        real_noisy_n = dataset.normalize_raw(real_noisy_pack)
        real_clean_n  = dataset.normalize_raw(real_clean_pack)

    print("real_noisy_n min/max:", real_noisy_n.min().item(), real_noisy_n.max().item())

    real_noise = real_noisy_n - real_clean_n   # [4, P, P]

    # ── 合成噪声 ──────────────────────────────────────────────
    target_iso = min(dataset.shadings.keys(), key=lambda x: abs(x - iso))

    shading_adu = dataset.shadings[target_iso][
        :,
        top : top + patch_size,
        left : left + patch_size,
    ]

    print("shading_adu min/max/mean:",
          shading_adu.min().item(),
          shading_adu.max().item(),
          shading_adu.mean().item())

    # shading 存的是减过 bl 的残差，直接除动态范围
    shading_normed = shading_adu / (dataset.wl - dataset.bl + 1e-8)

    b_sigma_row = dataset.band_params[target_iso]["row"]
    b_sigma_col = dataset.band_params[target_iso]["col"]

    n1 = torch.randn(1, 4, patch_size, patch_size)
    n2 = torch.randn(1, 4, patch_size, patch_size)
    iso_t = torch.tensor([[float(iso)]])

    with torch.no_grad():
        gen_noise = dataset.model(n1, n2, iso_t).squeeze(0)

    print("gen_noise min/max:", gen_noise.min().item(), gen_noise.max().item())

    row_noise  = torch.randn(4, patch_size, 1) * b_sigma_row
    col_noise  = torch.randn(4, 1, patch_size) * b_sigma_col
    band_noise = row_noise + col_noise

    syn_noise = gen_noise + shading_normed + band_noise

    # ── 统计对比 ──────────────────────────────────────────────
    print(f"\n{'':20s} {'Real':>12s} {'Syn':>12s}")
    print("-" * 46)
    for name, rn, sn in [
        ("total noise",  real_noise, syn_noise),
        ("shading only", real_noise, shading_normed),
    ]:
        print(f"{name:20s} std={rn.std():.6f}  std={sn.std():.6f}")

    print("\n--- Per-component std (channel mean) ---")
    print(f"  real total noise : {real_noise.std(dim=(-2,-1)).mean():.6f}")
    print(f"  syn  gen_noise   : {gen_noise.std(dim=(-2,-1)).mean():.6f}")
    print(f"  syn  shading     : {shading_normed.std(dim=(-2,-1)).mean():.6f}")
    print(f"  syn  band_noise  : {band_noise.std(dim=(-2,-1)).mean():.6f}")
    print(f"  syn  total       : {syn_noise.std(dim=(-2,-1)).mean():.6f}")

    # ── 分布图 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(real_noise[0].flatten().numpy(), bins=200,
                 alpha=0.6, label="real", density=True)
    axes[0].hist(syn_noise[0].flatten().numpy(),  bins=200,
                 alpha=0.6, label="syn",  density=True)
    axes[0].set_title("R channel noise histogram")
    axes[0].legend()

    real_row = real_noise.mean(dim=-1)[0].numpy()
    syn_row  = (gen_noise + band_noise).mean(dim=-1)[0].numpy()
    axes[1].plot(np.abs(np.fft.rfft(real_row, axis=-1)).mean(0), label="real row band")
    axes[1].plot(np.abs(np.fft.rfft(syn_row,  axis=-1)).mean(0), label="syn  row band")
    axes[1].set_title("Row band power spectrum (R)")
    axes[1].legend()

    real_col = real_noise.mean(dim=-2)[0].numpy()
    syn_col  = (gen_noise + band_noise).mean(dim=-2)[0].numpy()
    axes[2].plot(np.abs(np.fft.rfft(real_col, axis=-1)).mean(0), label="real col band")
    axes[2].plot(np.abs(np.fft.rfft(syn_col,  axis=-1)).mean(0), label="syn  col band")
    axes[2].set_title("Col band power spectrum (R)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("noise_gap_diagnosis.png", dpi=150)
    print("\nSaved: noise_gap_diagnosis.png")



# ── 入口 ──────────────────────────────────────────────────────
# print("\nnoisy file size:", os.path.getsize("synthetic_noisy/noisy_000_iso6400.npy"))
# print("clean file size:", os.path.getsize("synthetic_noisy/clean_000_iso6400.npy"))

# dataset = NoiseSynthesisDataset(
#     clean_raw_dir="../../data/xml196414/SID/Sony_npy/long",
#     benchmark_dir="../../data/xml196414/SID/dev_phase_release/sonyzve10m2",
#     model_path="checkpoints/PNNP_noise/ppm_generator_sonyzve10m2_final_mixed.pth",
#     camera_config="sonyzve10m2",
# )

# diagnose_noise_gap(
#     "synthetic_noisy/noisy_000_iso6400.npy",
#     "synthetic_noisy/clean_000_iso6400.npy",
#     dataset,
#     iso=6400,
#     top=0,
#     left=0,
# )

import numpy as np
import torch
import random
from datasets.pnnp_train_dataset import NoiseSynthesisDataset

# ── 加载官方合成的参考数据 ──────────────────────────────────
ref_noisy = np.load("synthetic_noisy/noisy_000_iso6400.npy")  # [4,512,512]
ref_clean  = np.load("synthetic_noisy/clean_000_iso6400.npy")
ref_noise  = ref_noisy - ref_clean

print("=== 官方 SynthTrainDataset ===")
print(f"noisy  min/max/std: {ref_noisy.min():.4f} / {ref_noisy.max():.4f} / {ref_noisy.std():.6f}")
print(f"clean  min/max/std: {ref_clean.min():.4f} / {ref_clean.max():.4f} / {ref_clean.std():.6f}")
print(f"noise  std: {ref_noise.std():.6f}")

# ── 用你的 NoiseSynthesisDataset 生成一个样本 ───────────────
dataset = NoiseSynthesisDataset(
    clean_raw_dir="../../data/xml196414/SID/Sony_npy/long",
    benchmark_dir="../../data/xml196414/SID/dev_phase_release/sonyzve10m2",
    model_path="checkpoints/PNNP_noise/ppm_generator_sonyzve10m2_final_mixed.pth",
    camera_config="sonyzve10m2",
    dgain_range=(100, 200),   # 当前设置
    inp_clip_low=float("-inf"),   # ← 对齐官方 inp_clip_low=False
    inp_clip_high=1.0            # ← 对齐官方 inp_clip_high=True
)

sample = dataset[0]
my_noisy = sample["noisy"][0]
my_clean  = sample["clean"][0]
my_noise  = my_noisy - my_clean

print("\n=== 你的 NoiseSynthesisDataset ===")
print(f"noisy  min/max/std: {my_noisy.min():.4f} / {my_noisy.max():.4f} / {my_noisy.std():.6f}")
print(f"clean  min/max/std: {my_clean.min():.4f} / {my_clean.max():.4f} / {my_clean.std():.6f}")
print(f"noise  std: {my_noise.std():.6f}")
print(f"dgain used: {sample['dgain'][0].item()}")

print("\n=== 差距 ===")
print(f"noise std 比值 (官方/你的): {ref_noise.std() / my_noise.std():.1f}x")