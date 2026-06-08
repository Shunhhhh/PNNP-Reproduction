"""
Self-supervised row-aware RAW denoising training script.

Main objective:
    L = L_blind + 0.2 L_r2r + 0.1 L_cons + 0.05 L_row + optional L_nll

This script expects:
    - SIDDNoisyRAWDataset returns var_map and row_profile.
    - ConditionalNAFNet supports:
        model(x, alpha, sigma2, row_std, var_map, row_profile)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import glob
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import *
from my_idea.conditional_denoiser_spatial import ConditionalNAFNet
from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset
from noise_param_refine import (
    NoiseParamRefineNet,
    refine_loss,
    extract_residual_features,
    residual_whitening_loss,
    row_whitening_loss,
    brightness_correlation_loss,
    param_regularization_loss,
)


TRAIN_SCENES = {1, 2, 4, 7, 8}
VAL_RATIO = 0.2


def scene_id_from_path(path: str) -> int:
    import re
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    match = re.match(r"^\d{4}_(\d{3})_", parent)
    if match:
        return int(match.group(1))
    return -1


def build_split_txts(noisy_img_dir: str, split_dir: str):
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    test_txt = os.path.join(split_dir, "test.txt")

    if os.path.exists(train_txt) and os.path.exists(val_txt) and os.path.exists(test_txt):
        train_lines = [line.strip() for line in open(train_txt) if line.strip()]
        val_lines = [line.strip() for line in open(val_txt) if line.strip()]
        test_lines = [line.strip() for line in open(test_txt) if line.strip()]
        if train_lines and val_lines and test_lines:
            print(
                f"[split] 复用已有 split: {split_dir} "
                f"(train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)})"
            )
            return train_txt, val_txt, test_txt
        print("[split] 已有 split 文件内容不完整，重新生成...")

    os.makedirs(split_dir, exist_ok=True)

    if not os.path.isdir(noisy_img_dir):
        raise FileNotFoundError(f"[split] noisy_img_dir 不存在或不是目录: {noisy_img_dir}")

    all_files = sorted(
        glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.MAT"), recursive=True)
        + glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.mat"), recursive=True)
        + glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.PNG"), recursive=True)
        + glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.png"), recursive=True)
    )

    if not all_files:
        top = sorted(os.listdir(noisy_img_dir))[:10]
        raise FileNotFoundError(
            f"[split] 在 {noisy_img_dir} 下未找到 NOISY 文件。\n"
            f"  目录前 10 条目: {top}\n"
            f"  请确认文件扩展名或调整 glob 模式。"
        )

    parse_failures = []
    train_val_dirs, test_dirs = set(), set()

    for file_path in all_files:
        scene_id = scene_id_from_path(file_path)
        if scene_id == -1:
            parse_failures.append(file_path)
            continue
        scene_dir = os.path.dirname(os.path.normpath(file_path))
        if scene_id in TRAIN_SCENES:
            train_val_dirs.add(scene_dir)
        else:
            test_dirs.add(scene_dir)

    if parse_failures:
        print(
            f"[split] 警告：{len(parse_failures)} 个文件无法解析 scene id，已跳过。\n"
            f"  示例（前 5 个）: {parse_failures[:5]}"
        )

    train_val_dirs = sorted(train_val_dirs)
    test_dirs = sorted(test_dirs)

    if not train_val_dirs and not test_dirs:
        raise RuntimeError(
            f"[split] 场景列表为空。共扫描 {len(all_files)} 个文件，"
            f"其中 {len(parse_failures)} 个解析失败。"
        )

    if not train_val_dirs:
        scene_ids_found = sorted(
            {
                scene_id_from_path(file_path)
                for file_path in all_files
                if scene_id_from_path(file_path) != -1
            }
        )
        raise RuntimeError(
            f"[split] 训练 scene {sorted(TRAIN_SCENES)} 未找到任何目录。\n"
            f"  数据集中实际存在的 scene id: {scene_ids_found}"
        )

    rng = random.Random(42)
    shuffled = train_val_dirs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_dirs = shuffled[:n_val]
    train_dirs = shuffled[n_val:]

    def write_txt(path, lines):
        with open(path, "w") as file:
            file.write("\n".join(lines) + "\n")

    write_txt(train_txt, sorted(train_dirs))
    write_txt(val_txt, sorted(val_dirs))
    write_txt(test_txt, sorted(test_dirs))

    print(
        f"[split] train={len(train_dirs)} dirs "
        f"val={len(val_dirs)} dirs "
        f"test={len(test_dirs)} dirs "
        f"(train/val 来自 scene {sorted(TRAIN_SCENES)}，val_ratio={VAL_RATIO})"
    )
    return train_txt, val_txt, test_txt


def build_dataloaders(args):
    train_set = SIDDNoisyRAWDataset(
        split_txt=args.train_txt,
        train=True,
        patch_size=args.train_patch_size,
        n_crop_per_img=args.n_crop_per_img,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        noise_low=args.noise_low,
        noise_high=args.noise_high,
        add_row_noise=args.add_row_noise,
        row_noise_scale=args.row_noise_scale,
        row_smooth_kernel=args.row_smooth_kernel,
    )
    val_set = SIDDNoisyRAWDataset(
        split_txt=args.val_txt,
        train=False,
        patch_size=args.train_patch_size,
        n_crop_per_img=16,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        noise_low=args.noise_low,
        noise_high=args.noise_high,
        add_row_noise=args.add_row_noise,
        row_noise_scale=args.row_noise_scale,
        row_smooth_kernel=args.row_smooth_kernel,
    )
    test_set = SIDDNoisyRAWDataset(
        split_txt=args.test_txt,
        train=False,
        patch_size=args.train_patch_size,
        n_crop_per_img=4,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        noise_low=args.noise_low,
        noise_high=args.noise_high,
        add_row_noise=False,
        row_noise_scale=args.row_noise_scale,
        row_smooth_kernel=args.row_smooth_kernel,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, args.num_workers)),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def extract_green_channels(data, device):
    """
    从 batch 里提取 G1/G2，用于修正网络输入。
    packed 通道顺序: [R, Gr, Gb, B] → index 1=Gr=G1, 2=Gb=G2
    """
    noisy_4d = tensor_dim5to4(data["noisy"]).to(device)   # [B*n_crop, 4, H, W]
    G1 = noisy_4d[:, 1:2, :, :]   # Gr [B*n_crop, 1, H, W]
    G2 = noisy_4d[:, 2:3, :, :]   # Gb [B*n_crop, 1, H, W]
    return G1, G2


class FFTLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return self.l1(pred_fft.abs(), target_fft.abs()) * self.weight


def make_blindspot_batch(x, mask_ratio=0.03):
    B, C, H, W = x.shape
    mask = torch.rand((B, C, H, W), device=x.device) < mask_ratio
    shifts = [
        torch.roll(x, shifts=1, dims=2),
        torch.roll(x, shifts=-1, dims=2),
        torch.roll(x, shifts=1, dims=3),
        torch.roll(x, shifts=-1, dims=3),
    ]
    rand_idx = torch.randint(0, len(shifts), (B, C, H, W), device=x.device)
    replacement = torch.zeros_like(x)
    for idx, shifted in enumerate(shifts):
        replacement = torch.where(rand_idx == idx, shifted, replacement)
    x_masked = torch.where(mask, replacement, x)
    return x_masked, mask.float()


def masked_l1(pred, target, mask, eps=1e-6):
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + eps)


def extract_row_profile(x):
    diff = x[:, 1:2] - x[:, 2:3]
    row_profile = diff.median(dim=3, keepdim=True).values
    row_profile = row_profile - row_profile.mean(dim=2, keepdim=True)
    return row_profile


def green_row_consistency_loss(pred):
    pred_row = extract_row_profile(pred)
    return torch.mean(torch.abs(pred_row))


def hetero_nll_loss(pred, noisy, var_map, eps=1e-8):
    var_map = torch.clamp(var_map, min=eps)
    residual = noisy - pred
    return (residual.pow(2) / var_map + torch.log(var_map)).mean()


def green_diff_std(x):
    return (x[:, 1:2] - x[:, 2:3]).std().item()


def green_diff_consistency_loss(pred, noisy):
    pred_diff = pred[:, 1:2] - pred[:, 2:3]
    noisy_diff = noisy[:, 1:2] - noisy[:, 2:3]
    return torch.mean(torch.abs(pred_diff - noisy_diff))


def compute_var_map(noisy, alpha, sigma2, row_std):
    """用修正后的参数重新计算 var_map。"""
    return (
        alpha[:, None, None, None] * noisy.clamp(min=0)
        + sigma2[:, None, None, None]
        + row_std[:, None, None, None] ** 2
    ).clamp(min=1e-10)


def train_one_ep(
    model,
    refine_net,
    train_loader,
    optimizer,
    refine_optimizer,
    criterion_l1,
    criterion_fft,
    device,
    args,
    train_refine: bool = True,
    train_denoiser: bool = True,
):
    if train_denoiser:
        model.train()
    else:
        model.eval()

    if train_refine:
        refine_net.train()
    else:
        refine_net.eval()

    loss_am         = AverageMeter("loss",         ":.4e")
    loss_refine_am  = AverageMeter("loss_refine",  ":.4e")
    loss_bs_am      = AverageMeter("loss_bs",      ":.4e")
    loss_r2r_am     = AverageMeter("loss_r2r",     ":.4e")
    loss_cons_am    = AverageMeter("loss_cons",    ":.4e")
    loss_row_am     = AverageMeter("loss_row",     ":.4e")
    loss_nll_am     = AverageMeter("loss_nll",     ":.4e")
    alpha_delta_am  = AverageMeter("alpha_delta",  ":.4f")
    sigma2_delta_am = AverageMeter("sigma2_delta", ":.4f")
    row_std_am      = AverageMeter("row_std",      ":.4e")
    psnr_noisy_am   = AverageMeter("psnr_noisy",   ":.2f")

    for data in tqdm(train_loader, desc="train", leave=False):
        noisy         = tensor_dim5to4(data["noisy"]).to(device)
        noisier       = torch.clamp(tensor_dim5to4(data["noisier"]).to(device), 0, 1)
        row_profile   = tensor_dim5to4(data["row_profile"]).to(device)
        alpha_init    = data["alpha"].reshape(-1).to(device)
        sigma2_init   = data["sigma2"].reshape(-1).to(device)
        row_std_batch = data["row_std"].reshape(-1).to(device)
        row_std_am.update(row_std_batch.mean().item())

        G1, G2 = extract_green_channels(data, device)

        # ── Step 1：refine_net ────────────────────────────────────────────────
        if train_refine:
            refine_optimizer.zero_grad()
            alpha_refined, sigma2_refined = refine_net.refine(G1, G2, alpha_init, sigma2_init)
            loss_r = refine_loss(
                G1, G2,
                alpha_refined, sigma2_refined,
                alpha_init, sigma2_init,
                w_whitening=args.refine_w_whitening,
                w_row=args.refine_w_row,
                w_brightness=args.refine_w_brightness,
                w_reg=args.refine_w_reg,
            )
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(refine_net.parameters(), 5.0)
            refine_optimizer.step()
            loss_refine_am.update(loss_r.item())
        else:
            with torch.no_grad():
                alpha_refined, sigma2_refined = refine_net.refine(G1, G2, alpha_init, sigma2_init)

        # ── 记录修正幅度 ──────────────────────────────────────────────────────
        with torch.no_grad():
            alpha_delta  = (alpha_refined  / alpha_init.clamp(min=1e-6)   - 1).abs().mean()
            sigma2_delta = (sigma2_refined / sigma2_init.clamp(min=1e-10) - 1).abs().mean()
            alpha_delta_am.update(alpha_delta.item())
            sigma2_delta_am.update(sigma2_delta.item())

        # ── Step 2：denoiser ──────────────────────────────────────────────────
        alpha_use  = alpha_refined.detach()
        sigma2_use = sigma2_refined.detach()

        with torch.no_grad():
            var_map = compute_var_map(noisy, alpha_use, sigma2_use, row_std_batch)

        if train_denoiser:
            masked_noisy, bs_mask = make_blindspot_batch(noisy, mask_ratio=args.bs_mask_ratio)
            optimizer.zero_grad()

            pred_clean = model(noisy,        alpha_use, sigma2_use, row_std_batch, var_map, row_profile)
            pred_blind = model(masked_noisy, alpha_use, sigma2_use, row_std_batch, var_map, row_profile)
            pred_r2r   = model(noisier,      alpha_use, sigma2_use, row_std_batch, var_map, row_profile)

            loss_bs    = masked_l1(
                torch.clamp(pred_blind, 0, 1),
                torch.clamp(noisy,      0, 1),
                bs_mask,
            )
            loss_r2r   = criterion_l1(pred_r2r, noisy) + criterion_fft(pred_r2r, noisy)
            loss_cons  = criterion_l1(
                torch.clamp(pred_r2r,            0, 1),
                torch.clamp(pred_clean.detach(), 0, 1),
            )
            loss_row   = green_row_consistency_loss(torch.clamp(pred_clean, 0, 1))
            loss_nll   = hetero_nll_loss(
                torch.clamp(pred_clean, 0, 1),
                torch.clamp(noisy,      0, 1),
                var_map,
            )
            loss_gdiff = green_diff_consistency_loss(
                torch.clamp(pred_clean, 0, 1),
                torch.clamp(noisy,      0, 1),
            )

            loss = (
                args.bs_weight         * loss_bs
                + args.r2r_weight      * loss_r2r
                + args.cons_weight     * loss_cons
                + args.row_loss_weight * loss_row
                + args.gdiff_weight    * loss_gdiff
                + args.nll_weight      * loss_nll
            )

            loss.backward()
            optimizer.step()

            loss_am.update(loss.item())
            loss_bs_am.update(loss_bs.item())
            loss_r2r_am.update(loss_r2r.item())
            loss_cons_am.update(loss_cons.item())
            loss_row_am.update(loss_row.item())
            loss_nll_am.update(loss_nll.item())

            with torch.no_grad():
                res_noisy = psnr_ssim_metric_torch(
                    torch.clamp(pred_clean, 0, 1),
                    torch.clamp(noisy,      0, 1),
                )
            psnr_noisy_am.update(res_noisy["psnr"])

    # ── 打印 ──────────────────────────────────────────────────────────────────
    refine_str  = f"refine={loss_refine_am.avg:.4e} " if train_refine   else "[refine frozen] "
    denoise_str = (
        f"loss={loss_am.avg:.4e} bs={loss_bs_am.avg:.4e} "
        f"r2r={loss_r2r_am.avg:.4e} cons={loss_cons_am.avg:.4e} "
        f"row={loss_row_am.avg:.4e} nll={loss_nll_am.avg:.4e} "
        if train_denoiser else "[denoiser frozen] "
    )
    print(
        f"  [TRAIN-DETAIL] {refine_str}{denoise_str}"
        f"row_std={row_std_am.avg:.4e} "
        f"alpha_delta={alpha_delta_am.avg:.3f} "
        f"sigma2_delta={sigma2_delta_am.avg:.3f}"
    )

    return (
        loss_refine_am.avg if not train_denoiser else loss_am.avg,
        psnr_noisy_am.avg,
    )

def validate(model, refine_net, val_loader, criterion_l1, criterion_fft, device):
    model.eval()
    refine_net.eval()

    loss_am = AverageMeter("loss", ":.4e")
    psnr_n2n_am = AverageMeter("psnr_n2n", ":.2f")
    psnr_gt_am = AverageMeter("psnr_gt", ":.2f")
    ssim_gt_am = AverageMeter("ssim_gt", ":.4f")
    std_ratio_am = AverageMeter("std_ratio", ":.4f")
    gdiff_ratio_am = AverageMeter("gdiff_ratio", ":.4f")
    has_clean = False

    with torch.no_grad():
        for data in tqdm(val_loader, desc="val", leave=False):
            noisy = tensor_dim5to4(data["noisy"]).to(device)
            noisier = tensor_dim5to4(data["noisier"]).to(device)
            row_profile = tensor_dim5to4(data["row_profile"]).to(device)
            alpha_init = data["alpha"].reshape(-1).to(device)
            sigma2_init = data["sigma2"].reshape(-1).to(device)
            row_std_batch = data["row_std"].reshape(-1).to(device)

            G1, G2 = extract_green_channels(data, device)
            alpha_use, sigma2_use = refine_net.refine(G1, G2, alpha_init, sigma2_init)
            var_map = compute_var_map(noisy, alpha_use, sigma2_use, row_std_batch)

            denoised_n2n = model(noisier, alpha_use, sigma2_use, row_std_batch, var_map, row_profile)
            denoised_ref = model(noisy, alpha_use, sigma2_use, row_std_batch, var_map, row_profile)

            loss = criterion_l1(denoised_n2n, noisy) + criterion_fft(denoised_n2n, noisy)
            loss_am.update(loss.item())

            res_n2n = psnr_ssim_metric_torch(
                torch.clamp(denoised_n2n, 0, 1),
                torch.clamp(noisy, 0, 1),
            )
            psnr_n2n_am.update(res_n2n["psnr"])

            noisy_std = noisy.std().item()
            denoised_std = torch.clamp(denoised_ref, 0, 1).std().item()
            std_ratio_am.update(denoised_std / max(noisy_std, 1e-12))
            gdiff_ratio_am.update(
                green_diff_std(torch.clamp(denoised_ref, 0, 1))
                / max(green_diff_std(torch.clamp(noisy, 0, 1)), 1e-12)
            )

            if "clean" in data:
                has_clean = True
                clean = tensor_dim5to4(data["clean"]).to(device)
                res_gt = psnr_ssim_metric_torch(
                    torch.clamp(denoised_ref, 0, 1),
                    torch.clamp(clean, 0, 1),
                )
                psnr_gt_am.update(res_gt["psnr"])
                ssim_gt_am.update(res_gt["ssim"])

    if has_clean:
        print(
            f"  [VAL] n2n_loss={loss_am.avg:.4e} "
            f"psnr(noisier->noisy)={psnr_n2n_am.avg:.2f} "
            f"psnr(noisy->GT)={psnr_gt_am.avg:.2f} "
            f"ssim(noisy->GT)={ssim_gt_am.avg:.4f} "
            f"std_ratio={std_ratio_am.avg:.4f} "
            f"gdiff_ratio={gdiff_ratio_am.avg:.4f}"
        )
    else:
        print(
            f"  [VAL] n2n_loss={loss_am.avg:.4e} "
            f"psnr(noisier->noisy)={psnr_n2n_am.avg:.2f} "
            f"std_ratio={std_ratio_am.avg:.4f} "
            f"gdiff_ratio={gdiff_ratio_am.avg:.4f}"
        )

    print(f"  alpha: init={alpha_init.mean():.4f} refined={alpha_use.mean():.4f}")
    print(f"  sigma2: init={sigma2_init.mean():.6f} refined={sigma2_use.mean():.6f}")

    return loss_am.avg, psnr_n2n_am.avg, psnr_gt_am.avg, has_clean


def run_test_inference(model, refine_net, test_loader, device, save_dir=None):
    model.eval()
    refine_net.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    noisy_mean_am = AverageMeter("noisy_mean", ":.4f")
    denoised_mean_am = AverageMeter("denoised_mean", ":.4f")
    noisy_std_am = AverageMeter("noisy_std", ":.4f")
    denoised_std_am = AverageMeter("denoised_std", ":.4f")
    gdiff_ratio_am = AverageMeter("gdiff_ratio", ":.4f")

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc="test-infer", leave=False)):
            noisy = tensor_dim5to4(data["noisy"]).to(device)
            row_profile = tensor_dim5to4(data["row_profile"]).to(device)
            alpha_init = data["alpha"].reshape(-1).to(device)
            sigma2_init = data["sigma2"].reshape(-1).to(device)
            row_std_batch = data["row_std"].reshape(-1).to(device)

            G1, G2 = extract_green_channels(data, device)
            alpha_use, sigma2_use = refine_net.refine(G1, G2, alpha_init, sigma2_init)
            var_map = compute_var_map(noisy, alpha_use, sigma2_use, row_std_batch)

            denoised = model(noisy, alpha_use, sigma2_use, row_std_batch, var_map, row_profile)
            denoised = torch.clamp(denoised, 0, 1)

            noisy_mean_am.update(noisy.mean().item())
            denoised_mean_am.update(denoised.mean().item())
            noisy_std_am.update(noisy.std().item())
            denoised_std_am.update(denoised.std().item())
            gdiff_ratio_am.update(
                green_diff_std(denoised) / max(green_diff_std(torch.clamp(noisy, 0, 1)), 1e-12)
            )

            if save_dir:
                torch.save(denoised.cpu(), os.path.join(save_dir, f"denoised_{idx:05d}.pt"))

    print(
        f"  [TEST-INFER]\n"
        f"  noisy:    mean={noisy_mean_am.avg:.4f} std={noisy_std_am.avg:.4f}\n"
        f"  denoised: mean={denoised_mean_am.avg:.4f} std={denoised_std_am.avg:.4f}\n"
        f"  std_ratio={denoised_std_am.avg / max(noisy_std_am.avg, 1e-12):.4f} "
        f"gdiff_ratio={gdiff_ratio_am.avg:.4f}"
    )
    return denoised_mean_am.avg


def main(args):
    criterion_l1 = nn.L1Loss()
    criterion_fft = FFTLoss(weight=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"训练/验证 scene : {sorted(TRAIN_SCENES)} (train {1 - VAL_RATIO:.0%} / val {VAL_RATIO:.0%})")
    print("测试 scene      : 其余所有 scene（纯推理，无 GT）")
    print(
        f"loss weights    : bs={args.bs_weight} r2r={args.r2r_weight} "
        f"cons={args.cons_weight} row={args.row_loss_weight} nll={args.nll_weight}"
    )
    print(
        f"row noise       : add={args.add_row_noise} "
        f"scale={args.row_noise_scale} smooth_kernel={args.row_smooth_kernel}"
    )
    print(
        f"refine_net      : lr={args.lr * 0.1:.2e} "
        f"delta_scale={args.refine_delta_scale} "
        f"w_whitening={args.refine_w_whitening} "
        f"w_row={args.refine_w_row} "
        f"w_brightness={args.refine_w_brightness} "
        f"w_reg={args.refine_w_reg}"
    )
    print(
        f"phase           : refine_pretrain={args.refine_pretrain_epochs} epochs  "
        f"denoise={args.denoise_epochs} epochs"
    )

    if args.train_txt is None or args.val_txt is None or args.test_txt is None:
        args.train_txt, args.val_txt, args.test_txt = build_split_txts(
            args.noisy_img_dir,
            split_dir=os.path.join("my_idea", "0531splits"),
        )
    else:
        for path in [args.train_txt, args.val_txt, args.test_txt]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"split 文件不存在: {path}")

    # ── 模型 ──────────────────────────────────────────────────────
    model = ConditionalNAFNet(
        img_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    refine_net = NoiseParamRefineNet(
        n_bins=8,
        hidden_dim=64,
        delta_scale=args.refine_delta_scale,
    ).to(device)

    # 预热：用随机数据触发一次 build
    with torch.no_grad():
        dummy_G1    = torch.zeros(1, 1, 64, 64, device=device)
        dummy_G2    = torch.zeros(1, 1, 64, 64, device=device)
        dummy_alpha  = torch.tensor([0.01],  device=device)
        dummy_sigma2 = torch.tensor([0.001], device=device)
        refine_net.refine(dummy_G1, dummy_G2, dummy_alpha, dummy_sigma2)

    # ── 优化器 & Scheduler ────────────────────────────────────────
    optimizer        = AdamW(model.parameters(),       lr=args.lr)
    refine_optimizer = AdamW(refine_net.parameters(),  lr=args.lr * 0.1)

    # 两个 scheduler 各自只负责自己 phase 的长度
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.denoise_epochs,
        eta_min=1e-5,
    )
    refine_scheduler = lr_scheduler.CosineAnnealingLR(
        refine_optimizer,
        max(args.refine_pretrain_epochs, args.denoise_epochs),  # 避免 T_max=0
        eta_min=1e-6,
    )

    train_loader, val_loader, test_loader = build_dataloaders(args)

    best_val_psnr_n2n = -float("inf")
    total_epochs = args.refine_pretrain_epochs + args.denoise_epochs

    for epoch in range(1, total_epochs + 1):

        # ── Phase 判断 ────────────────────────────────────────────
        # main() 里的 phase 判断改为
        if args.refine_pretrain_epochs > 0 and epoch <= args.refine_pretrain_epochs:
            phase_tag      = "REFINE-ONLY"
            train_refine   = True
            train_denoiser = False
        else:
            phase_tag      = "JOINT"        # 两个网络同时训练
            train_refine   = True
            train_denoiser = True

        train_loss, train_psnr_noisy = train_one_ep(
            model,
            refine_net,
            train_loader,
            optimizer,
            refine_optimizer,
            criterion_l1,
            criterion_fft,
            device,
            args,
            train_refine=train_refine,
            train_denoiser=train_denoiser,
        )

        # 各自的 scheduler 只在对应 phase step
        if train_refine:
            refine_scheduler.step()
        if train_denoiser:
            scheduler.step()

        # Phase 1 只有 refine loss 有意义，psnr 不打
        if train_denoiser:
            print(
                f"[{phase_tag} | Epoch {epoch:>4d}/{total_epochs}] "
                f"train_loss={train_loss:.4e} "
                f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}"
            )
        else:
            print(
                f"[{phase_tag} | Epoch {epoch:>4d}/{total_epochs}] "
                f"refine_loss={train_loss:.4e}"
            )

        # ── Phase 1 结束：保存 refine_net 权重 ────────────────────
        if epoch == args.refine_pretrain_epochs:
            ckpt_dir = f"./checkpoints/{args.task}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                refine_net.state_dict(),
                os.path.join(ckpt_dir, "refine_net_pretrained.pth"),
            )
            print(f"[Phase 1 done] refine_net 权重已保存 → {ckpt_dir}/refine_net_pretrained.pth")

        # ── val / test：只在 Phase 2 运行 ─────────────────────────
        if train_denoiser:
            if epoch % args.val_every == 0:
                val_loss, val_psnr_n2n, val_psnr_gt, has_clean = validate(
                    model,
                    refine_net,
                    val_loader,
                    criterion_l1,
                    criterion_fft,
                    device,
                )

                if val_psnr_n2n > best_val_psnr_n2n:
                    best_val_psnr_n2n = val_psnr_n2n
                    ckpt_dir = f"./checkpoints/{args.task}"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "refine_net": refine_net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "refine_optimizer": refine_optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_psnr_n2n": val_psnr_n2n,
                            "val_psnr_gt": val_psnr_gt,
                            "beta_min": args.beta_min,
                            "beta_max": args.beta_max,
                            "noise_low": args.noise_low,
                            "noise_high": args.noise_high,
                            "add_row_noise": args.add_row_noise,
                            "row_noise_scale": args.row_noise_scale,
                            "row_smooth_kernel": args.row_smooth_kernel,
                            "bs_mask_ratio": args.bs_mask_ratio,
                            "bs_weight": args.bs_weight,
                            "r2r_weight": args.r2r_weight,
                            "cons_weight": args.cons_weight,
                            "row_loss_weight": args.row_loss_weight,
                            "nll_weight": args.nll_weight,
                            "refine_delta_scale": args.refine_delta_scale,
                            "train_scenes": sorted(TRAIN_SCENES),
                        },
                        os.path.join(ckpt_dir, "best.pth"),
                    )
                    print("  -> Saved best model!")

            if epoch % args.test_every == 0:
                infer_save_dir = (
                    os.path.join(f"./outputs/{args.task}", f"epoch_{epoch:04d}")
                    if args.save_test_outputs
                    else None
                )
                run_test_inference(model, refine_net, test_loader, device, save_dir=infer_save_dir)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SIDD_2phase")
    parser.add_argument("--refine_pretrain_epochs", type=int, default=0,
                        help="Phase 1：只训练 refine_net 的 epoch 数")
    parser.add_argument("--denoise_epochs", type=int, default=200,
                        help="Phase 2：只训练 denoiser 的 epoch 数")
    parser.add_argument("--train_patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_crop_per_img", type=int, default=8)
    parser.add_argument("--beta_min", type=float, default=0.7)
    parser.add_argument("--beta_max", type=float, default=1.3)
    parser.add_argument("--noise_low", type=float, default=0.001)
    parser.add_argument("--noise_high", type=float, default=0.02)
    parser.add_argument("--add_row_noise", action="store_true")
    parser.add_argument("--row_noise_scale", type=float, default=1.0)
    parser.add_argument("--row_smooth_kernel", type=int, default=0)
    parser.add_argument("--gdiff_weight", type=float, default=0.1)
    parser.add_argument("--bs_mask_ratio", type=float, default=0.02)
    parser.add_argument("--bs_weight", type=float, default=1.0)
    parser.add_argument("--r2r_weight", type=float, default=0.5)
    parser.add_argument("--cons_weight", type=float, default=0.1)
    parser.add_argument("--row_loss_weight", type=float, default=0.05)
    parser.add_argument("--nll_weight", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--test_every", type=int, default=5)
    # 修正网络相关参数
    parser.add_argument("--refine_delta_scale", type=float, default=1.5,
                        help="修正网络 log 域修正幅度上限")
    parser.add_argument("--refine_w_whitening", type=float, default=1.0)
    parser.add_argument("--refine_w_row", type=float, default=0.5)
    parser.add_argument("--refine_w_brightness", type=float, default=0.5)
    parser.add_argument("--refine_w_reg", type=float, default=0.5)
    parser.add_argument("--noisy_img_dir", type=str,default="/data/xml196414/SIDD/SIDD_Medium/SIDD_Medium_Raw/Data",)

    parser.add_argument("--train_txt", type=str, default=None)
    parser.add_argument("--val_txt", type=str, default=None)
    parser.add_argument("--test_txt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_test_outputs", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    main(args)