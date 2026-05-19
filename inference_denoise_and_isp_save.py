"""
PNNP合成数据集的去噪结果
"""
import os
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.ELD_models import UNetSeeInDark
from datasets.pnnp_train_dataset import NoiseSynthesisDataset
from utils.utils import tensor_dim5to4


def isp_process(packed, wb=None, ccm=None, gamma=2.2):
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


# =========================
# 保存图片
# =========================
def save_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image
    Image.fromarray(img).save(path)


# =========================
# 固定随机性（关键）
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# build dataset (NoiseSynthesisDataset)
# =========================
def build_dataset(args):

    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg_all = yaml.load(f, Loader=yaml.FullLoader)

    cam_cfg = cam_cfg_all[args.camera]

    benchmark_dir = os.path.join(args.benchmark_dir, args.camera)

    dataset = NoiseSynthesisDataset(
        model_path=f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}.pth",
        clean_raw_dir=args.clean_img_dir,
        benchmark_dir=benchmark_dir,
        camera_config=cam_cfg,

        # ===== inference关键 =====
        iso_list=[args.iso],
        dgain_range=[args.dgain],
        patch_size=args.patch_size,
        n_crop_per_img=1,

        inp_clip_low=False,
        inp_clip_high=True,
    )

    return dataset, cam_cfg


# =========================
# MAIN
# =========================
def main(args):
    camera_config = {
        "sonyzve10m2": {
            "valid_roi": [0, 0, 4128, 6192]
        }
    }

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== dataset =====
    dataset, cam_cfg = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # ===== model =====
    model = UNetSeeInDark().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loaded checkpoint:", args.ckpt)

    # ===== inference =====
    with torch.no_grad():

        for i, data in enumerate(tqdm(loader)):

            noisy = tensor_dim5to4(data["noisy"]).to(device)
            clean = tensor_dim5to4(data["clean"]).to(device)

            name = f"{i:06d}.png"

            # ===== forward =====
            denoised = model(noisy)

            noisy = torch.clamp(noisy, 0, 1)
            denoised = torch.clamp(denoised, 0, 1)
            clean = torch.clamp(clean, 0, 1)

            noisy = noisy[0].cpu()
            denoised = denoised[0].cpu()
            clean = clean[0].cpu()

            # ===== ISP =====
            cam_config = camera_config["sonyzve10m2"]
            ccm = cam_config.get("ccm", None)   # [9] 或 None
            if ccm is not None:
                ccm = np.array(ccm).reshape(3, 3)

            # Sony ZV-E10M2 典型日光 WB（R Gr Gb B）
            wb = cam_config.get("wb", [2.0, 1.0, 1.0, 1.6])

            noisy_rgb = isp_process(noisy, wb=wb, ccm=ccm)
            denoised_rgb = isp_process(denoised, wb=wb, ccm=ccm)
            clean_rgb = isp_process(clean, wb=wb, ccm=ccm)

            # ===== save =====
            save_img(os.path.join(args.out_dir, "noisy", name), noisy_rgb)
            save_img(os.path.join(args.out_dir, "denoised", name), denoised_rgb)
            save_img(os.path.join(args.out_dir, "clean", name), clean_rgb)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="checkpoints/PNNP_ELD/sonyzve10m2/best.pth")

    parser.add_argument("--camera", type=str, default="sonyzve10m2")

    parser.add_argument("--clean_img_dir", type=str,
                        default="../../data/xml196414/SID/Sony_npy/long")

    parser.add_argument("--benchmark_dir", type=str,
                        default="../../data/xml196414/SID/dev_phase_release")

    parser.add_argument("--out_dir", type=str, default="./vis_rgb")

    # ===== noise control =====
    parser.add_argument("--iso", type=int, default=800)
    parser.add_argument("--dgain", type=float, default=100.0)
    parser.add_argument("--patch_size", type=int, default=256)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    main(args)