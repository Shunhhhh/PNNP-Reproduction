"""
PNNP加噪 + ELD去噪（整合 Generalized Anscombe VST，per-sample per-channel 参数）

训练流程：
    noisy  --[VST forward, per-channel alpha]--> noisy_vst
    noisy_vst --[model]--> denoised_vst
    denoised_vst --[VST inverse]--> denoised
    loss = L1(denoised, clean)   # 在原始线性域算 loss

VST 参数来源（均由 dataset.__getitem__ 提供）：
    alpha      : (B*n, 4)   = K[iso][ch] / (wl-bl)，来自 sys_gain.npz
    sigma_read : (B*n,)     = band_params[nearest_iso]["pixel"]

逆变换：训练使用 inverse_closed_form（纯数值，无文件依赖，速度快）。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import random
import yaml
from tqdm import tqdm
import argparse

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from utils.utils import *
from models.ELD_models import UNetSeeInDark
from datasets.vst_pnnp_dataset import VSTSynthesisDataset


# VST 接口
from vst.genanscombe import (
    forward             as gat_forward,
    inverse_closed_form as gat_inverse,
)

# ===================== VST 工具 =====================

def vst_forward_per_sample(x: torch.Tensor,
                           alpha: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
    """
    逐样本、逐通道施加 VST 前向变换。

    Parameters
    ----------
    x     : (B, C, H, W)  线性域 noisy，值域 [0,1]，C=4（RGGB pack）
    alpha : (B, C)         per-sample per-channel 泊松增益 = K_c/(wl-bl)
    sigma : (B,)           per-sample 高斯读出噪声 σ（归一化域，各通道共用）

    Returns
    -------
    fz : (B, C, H, W)  VST 域数据（各通道方差 ≈ 1）
    """
    B, C, H, W = x.shape
    device = x.device
    result = torch.empty_like(x)
    # print("alpha shape:", alpha.shape)
    # print("sigma shape:", sigma.shape)
    # print("alpha check:", alpha)
    # print("sigma check:", sigma)

    for b in range(B):
        for c in range(C):
            sig = sigma[b, c].item()
            alp = alpha[b, c].item()
            arr = x[b, c].detach().cpu().numpy().astype(np.float64)   # (H, W)
            fz  = gat_forward(arr, sigma=sig, alpha=alp, g=0.0)
            result[b, c] = torch.from_numpy(fz.astype(np.float32)).to(device)

    return result


def vst_inverse_per_sample(fz: torch.Tensor,
                                alpha: torch.Tensor,
                                sigma: torch.Tensor) -> torch.Tensor:
    """
    可微的 GAT 闭合形式逆变换，梯度可以正常回传。
    fz    : (B, C, H, W)
    alpha : (B, C)
    sigma : (B, C)
    """
    # (B, C) -> (B, C, 1, 1) 方便广播
    a = alpha.view(alpha.shape[0], alpha.shape[1], 1, 1)
    s = sigma.view(sigma.shape[0], sigma.shape[1], 1, 1)

    z = (1.0 / a) * ((fz / 2.0) ** 2 + (s ** 2) / (4.0 * a ** 2) - 1.0 / 8.0)
    return z


# ===================== DATA =====================

def build_dataloaders(args):

    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg_all = yaml.load(f, Loader=yaml.FullLoader)

    cam_cfg = cam_cfg_all[args.camera]

    benchmark_dir = os.path.join(args.benchmark_dir, args.camera)

    ppm_model_paths = {
        800:  f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}_iso800.pth",
        1250: f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}_iso1250.pth",
        1600: f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}_iso1600.pth",
        3200: f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}_iso3200.pth",
        6400: f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}_iso6400.pth",
    }

    full_set = VSTSynthesisDataset(
        model_dir=ppm_model_paths,
        clean_raw_dir=args.clean_img_dir,
        benchmark_dir=benchmark_dir,
        camera_config=cam_cfg,
        iso_list=[800, 1250, 1600, 3200, 6400],
        dgain_range=args.train_dgain_range,
        patch_size=args.train_patch_size,
        inp_clip_low=False,
        inp_clip_high=True,
        n_crop_per_img=args.n_crop_per_img,
    )

    train_size = int(0.9 * len(full_set))
    val_size   = len(full_set) - train_size

    g_gen = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        full_set, [train_size, val_size], generator=g_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ===================== TRAIN =====================

def train_one_ep(model, train_loader, optimizer, criterion, device):
    model.train()

    loss_am = AverageMeter("loss", ":.4e")
    psnr_am = AverageMeter("psnr", ":.2f")

    for data in tqdm(train_loader):

        clean = tensor_dim5to4(data["clean"]).to(device)   # (B*n, 4, H, W)
        noisy = tensor_dim5to4(data["noisy"]).to(device)   # (B*n, 4, H, W)
        alpha = data["alpha"].flatten(0, 1).to(device)
        sigma = data["sigma"].flatten(0, 1).to(device)

        # Step 1: VST 前向（per-channel）
        noisy_vst = vst_forward_per_sample(noisy, alpha, sigma)
        # print("noisy       range:", noisy.min().item(), noisy.max().item())
        # print("noisy_vst   range:", noisy_vst.min().item(), noisy_vst.max().item())
        # Step 2: 模型去噪
        optimizer.zero_grad()
        denoised_vst = model(noisy_vst)
        # print("denoised_vst range:", denoised_vst.min().item(), denoised_vst.max().item())

        # Step 3: VST 逆变换
        denoised = vst_inverse_per_sample(denoised_vst, alpha, sigma)
        # print("denoised    range:", denoised.min().item(), denoised.max().item())
        # print("alpha range:", alpha.min().item(), alpha.max().item())
        # print("sigma range:", sigma.min().item(), sigma.max().item())

        # Step 4: 线性域 L1 loss
        loss = criterion(denoised, clean)
        loss.backward()
        optimizer.step()

        denoised_c = torch.clamp(denoised.detach(), 0, 1)
        clean_c    = torch.clamp(clean, 0, 1)
        res = psnr_ssim_metric_torch(denoised_c, clean_c)

        loss_am.update(loss.item())
        psnr_am.update(res["psnr"])

    return loss_am.avg, psnr_am.avg


# ===================== VALIDATE =====================

def validate(model, val_loader, device):
    model.eval()

    psnr_am = AverageMeter("psnr", ":.2f")
    ssim_am = AverageMeter("ssim", ":.4f")

    with torch.no_grad():
        for data in val_loader:

            clean = tensor_dim5to4(data["clean"]).to(device)
            noisy = tensor_dim5to4(data["noisy"]).to(device)

            alpha = data["alpha"].flatten(0, 1).to(device)
            sigma = data["sigma"].flatten(0, 1).to(device)

            # alpha, sigma = extract_vst_params(data, device)

            noisy_vst    = vst_forward_per_sample(noisy, alpha, sigma)
            denoised_vst = model(noisy_vst)
            denoised     = vst_inverse_per_sample(denoised_vst, alpha, sigma)

            denoised = torch.clamp(denoised, 0, 1)
            clean    = torch.clamp(clean, 0, 1)

            res = psnr_ssim_metric_torch(denoised, clean)
            psnr_am.update(res["psnr"])
            ssim_am.update(res["ssim"])

    return psnr_am.avg, ssim_am.avg


# ===================== MAIN =====================

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = build_dataloaders(args)

    model     = UNetSeeInDark().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-5)
    criterion = nn.L1Loss()

    best_psnr = 0

    print("START TRAIN SCRIPT")
    print("GPU init memory:", torch.cuda.memory_allocated() / 1024 ** 3)

    for epoch in range(1, args.n_epoch + 1):

        train_loss, train_psnr = train_one_ep(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()

        print(f"[Epoch {epoch}] loss={train_loss:.6f}, psnr={train_psnr:.4f}")

        if epoch % 5 == 0:

            val_psnr, val_ssim = validate(model, val_loader, device)
            print(f"[VAL Epoch {epoch}] psnr={val_psnr:.2f}, ssim={val_ssim:.4f}")

            if val_psnr > best_psnr:
                best_psnr = val_psnr

                save_dir = f"./checkpoints/{args.task}/{args.camera}"
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    {
                        "epoch":     epoch,
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(save_dir, "best.pth"),
                )
                print("Saved best model!")


# ===================== ENTRY =====================

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--task",               type=str,   default="PNNP_ELD_VST")
    parser.add_argument("--n_epoch",            type=int,   default=500)
    parser.add_argument("--train_patch_size",   type=int,   default=256)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--bs",                 type=int,   default=32)
    parser.add_argument("--n_crop_per_img",     type=int,   default=4)
    parser.add_argument("--train_dgain_range",  type=list,  default=[10.0, 200.0])
    parser.add_argument("--camera",             type=str,   default="sonyzve10m2")
    parser.add_argument("--clean_img_dir",      type=str,   default="../../data/xml196414/SID/Sony/long")
    parser.add_argument("--benchmark_dir",      type=str,   default="../../data/xml196414/SID/dev_phase_release")
    parser.add_argument("--seed",               type=int,   default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = False

    main(args)