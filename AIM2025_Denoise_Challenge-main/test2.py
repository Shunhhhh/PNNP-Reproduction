import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datasets.synth_train_dataset import SynthTrainDataset
from datasets.pnnp_train_dataset import NoiseSynthesisDataset
# =========================================================
# 配置
# =========================================================
BENCHMARK_DIR = "../../data/xml196414/SID/dev_phase_release/sonyzve10m2"
CLEAN_RAW_DIR = "../../data/xml196414/SID/Sony_npy/long"
MODEL_PATH    = "./checkpoints/PNNP_noise/ppm_generator_sonyzve10m2.pth"
SAVE_DIR      = "./vis_results"
N_SAMPLES     = 4  
camera_config = {
    "sonyzve10m2": {
        "valid_roi": [0, 0, 4128, 6192]
    }
}

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 构建数据集
# =========================================================

dataset = NoiseSynthesisDataset(
    clean_raw_dir = CLEAN_RAW_DIR,
    benchmark_dir = BENCHMARK_DIR,
    model_path    = MODEL_PATH,
    camera_config = camera_config,
    iso_list      = [800, 1600, 3200],
    dgain_range   = (10, 200),
    patch_size    = 512,
    n_crop_per_img= 1,
    white_level   = 16383.0,
    black_level   = 512.0,
)

# model = dataset.model
# model.eval()
# with torch.no_grad():
#     for iso in [800, 1600, 3200, 6400]:
#         iso_t = torch.tensor([[float(iso)]])
#         iso_input = torch.log2(iso_t / 100.0)
#         gain = model.iso_gain(iso_input)
#         print(f"ISO={iso} | log2(iso/100)={iso_input.item():.3f} | gain={gain.item():.6f}")


# dataset = SynthTrainDataset(
#     clean_img_dir = CLEAN_RAW_DIR,
#     benchmark_dir = BENCHMARK_DIR,
#     # model_path    = MODEL_PATH,
#     camera_config = camera_config,
#     iso_list      = [800, 1600, 3200],
#     dgain_range   = (10, 200),
#     patch_size    = 512,
#     n_crop_per_img= 1,
#     # white_level   = 16383.0,
#     # black_level   = 512.0,
# )

# =========================================================
# ISP：packed RAW [4,H,W] -> RGB [H,W,3]
# =========================================================
def simple_isp(packed, wb=None, ccm=None, gamma=2.2):
    """
    packed: torch.Tensor [4, H, W]，归一化到[0,1]
            通道顺序: R, Gr, Gb, B
    wb:     white balance [4]，对应 R Gr Gb B，None则不做
    ccm:    color correction matrix [3,3]，None则不做
    返回:   np.ndarray [H, W, 3]，uint8
    """
    packed = packed.float()
    C, H, W = packed.shape

    # ── 1. White Balance ──────────────────────────────────
    if wb is not None:
        wb = torch.tensor(wb, dtype=torch.float32).view(4, 1, 1)
        packed = packed * wb
        packed = torch.clamp(packed, 0, 1)

    # ── 2. Demosaic（简单双线性，把4通道还原成全分辨率）──
    # R=ch0, Gr=ch1, Gb=ch2, B=ch3
    # 还原为 [H*2, W*2] Bayer，再取 R Gr+Gb G B
    full = torch.zeros(H * 2, W * 2)
    full[0::2, 0::2] = packed[0]   # R
    full[0::2, 1::2] = packed[1]   # Gr
    full[1::2, 0::2] = packed[2]   # Gb
    full[1::2, 1::2] = packed[3]   # B

    # 简单 demosaic：直接用4通道平均作为 RGB
    R  = packed[0]
    G  = (packed[1] + packed[2]) / 2.0
    B  = packed[3]
    rgb = torch.stack([R, G, B], dim=0)  # [3, H, W]

    # ── 3. CCM ───────────────────────────────────────────
    if ccm is not None:
        ccm = torch.tensor(ccm, dtype=torch.float32)  # [3,3]
        rgb_flat = rgb.view(3, -1)                     # [3, H*W]
        rgb_flat = ccm @ rgb_flat                      # [3, H*W]
        rgb = rgb_flat.view(3, H, W)
        rgb = torch.clamp(rgb, 0, 1)

    # ── 4. Gamma ─────────────────────────────────────────
    rgb = torch.pow(torch.clamp(rgb, 1e-8, 1.0), 1.0 / gamma)

    # ── 5. 转 uint8 ──────────────────────────────────────
    rgb = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return rgb


# =========================================================
# 获取 WB 和 CCM
# =========================================================
cam_config = camera_config["sonyzve10m2"]
ccm = cam_config.get("ccm", None)   # [9] 或 None
if ccm is not None:
    ccm = np.array(ccm).reshape(3, 3)

# Sony ZV-E10M2 典型日光 WB（R Gr Gb B）
# 如果 camera_config 里有就用，没有就用默认值
wb = cam_config.get("wb", [2.0, 1.0, 1.0, 1.6])


# =========================================================
# 生成并可视化
# =========================================================
for sample_idx in range(N_SAMPLES):
    data    = dataset[sample_idx]
    noisy   = data["noisy"][0]    # [4, H, W]
    clean   = data["clean"][0]    # [4, H, W]
    iso     = int(data["iso"].item())
    dgain   = float(data["dgain"][0].item())

    # ISP
    clean_rgb = simple_isp(clean, wb=wb, ccm=ccm)
    noisy_rgb = simple_isp(noisy, wb=wb, ccm=ccm)

    # 保存对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(clean_rgb)
    axes[0].set_title(f"Clean  |  ISO={iso}  dgain={dgain:.0f}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(noisy_rgb)
    axes[1].set_title(f"Noisy  |  ISO={iso}  dgain={dgain:.0f}", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"sample_{sample_idx:03d}_iso{iso}_dgain{dgain:.0f}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

print(f"\nDone. Results saved to {SAVE_DIR}/")