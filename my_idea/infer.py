"""
PNNP训练的模型，在baseline合成噪声上去噪
支持按 ISO 分级统计 PSNR / SSIM
"""
import os
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from my_idea.conditional_denoiser import ConditionalNAFNet
from my_idea.dataset import NoiseSynthesisDataset
from datasets.synth_train_dataset import SynthTrainDataset
from utils.utils import tensor_dim5to4
from utils.utils import AverageMeter, psnr_ssim_metric_torch


# =========================
# 固定随机性
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# build dataset
# =========================
def build_dataset(args):

    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg_all = yaml.load(f, Loader=yaml.FullLoader)

    cam_cfg = cam_cfg_all[args.camera]

    dataset = SynthTrainDataset(
        clean_img_dir=args.clean_img_dir,
        benchmark_dir=args.benchmark_dir,
        camera_config={args.camera: cam_cfg},
        iso_list=args.iso_list,
        dgain_range=args.dgain_range,
        patch_size=args.patch_size,
        inp_clip_low=False,
        inp_clip_high=True,
    )

    return dataset, cam_cfg


# =========================
# 打印按 ISO 分级的统计表
# =========================
def print_iso_summary(iso_meters):
    """
    iso_meters: dict[int -> {"psnr": AverageMeter, "ssim": AverageMeter}]
    """
    print("\n" + "=" * 56)
    print(f"  {'ISO':>6}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'samples':>8}")
    print("-" * 56)
    for iso in sorted(iso_meters.keys()):
        pm = iso_meters[iso]["psnr"]
        sm = iso_meters[iso]["ssim"]
        print(f"  {iso:>6}  {pm.avg:>10.2f}  {sm.avg:>8.4f}  {pm.count:>8}")
    print("=" * 56)


# =========================
# MAIN
# =========================
def main(args):

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
    model = ConditionalNAFNet(img_channel=4, width=32,
                middle_blk_num=12,
                enc_blk_nums=[2,2,4,8],
                dec_blk_nums=[2,2,2,2],
                embed_dim=256).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loaded checkpoint:", args.ckpt)
    print(f"Test ISO list: {args.iso_list}")

    # ===== meters =====
    overall_psnr = AverageMeter("psnr", ":.2f")
    overall_ssim = AverageMeter("ssim", ":.4f")

    # 按 ISO 各维护一对 meter
    iso_meters = {
        iso: {
            "psnr": AverageMeter(f"psnr_iso{iso}", ":.2f"),
            "ssim": AverageMeter(f"ssim_iso{iso}", ":.4f"),
        }
        for iso in args.iso_list
    }

    # ===== inference =====
    with torch.no_grad():

        for i, data in enumerate(tqdm(loader)):

            noisy = tensor_dim5to4(data["noisy"]).to(device)
            clean = tensor_dim5to4(data["clean"]).to(device)

            # ===== forward =====
            denoised = model(noisy)

            noisy    = torch.clamp(noisy,    0, 1)
            denoised = torch.clamp(denoised, 0, 1)
            clean    = torch.clamp(clean,    0, 1)

            res = psnr_ssim_metric_torch(denoised, clean)

            overall_psnr.update(res["psnr"])
            overall_ssim.update(res["ssim"])

            # ── 按 ISO 分别累积 ──────────────────────────────
            # data["iso"] 形状为 [B, n_crop] 或 [B]，取第一个元素
            iso_val = int(data["iso"].reshape(-1)[0].item())
            if iso_val in iso_meters:
                iso_meters[iso_val]["psnr"].update(res["psnr"])
                iso_meters[iso_val]["ssim"].update(res["ssim"])

    # ===== 汇总输出 =====
    print_iso_summary(iso_meters)
    print(f"\n  Overall  PSNR: {overall_psnr.avg:.2f}  SSIM: {overall_ssim.avg:.4f}")
    print(f"  Total samples: {overall_psnr.count}\n")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/ConditionalDenoise/sonyzve10m2/best.pth")
    parser.add_argument("--camera", type=str, default="sonyzve10m2")
    parser.add_argument("--clean_img_dir", type=str,
                        default="../../data/xml196414/SID/Sony/long")
    parser.add_argument("--benchmark_dir", type=str,
                        default="../../data/xml196414/SID/dev_phase_release")

    # ===== noise control =====
    parser.add_argument("--iso_list", type=int, nargs="+",
                        default=[800, 1250, 1600, 3200, 6400])
    parser.add_argument("--dgain_range", type=list, default=[10, 200])
    parser.add_argument("--patch_size",  type=int,  default=256)
    parser.add_argument("--seed",        type=int,  default=0)

    args = parser.parse_args()

    main(args)