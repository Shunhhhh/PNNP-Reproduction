"""
ConditionalDenoiser 训练脚本
Noisier2Noise + RAW Poisson-Gaussian
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..")
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import random
from tqdm import tqdm
import argparse

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from utils.utils import *

from my_idea.conditional_denoiser import ConditionalNAFNet
from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset


# =========================================================
# DATA
# =========================================================

def build_dataloaders(args):
    train_set = SIDDNoisyRAWDataset(
        split_txt=args.train_txt,
        train=True,
        patch_size=args.train_patch_size,
        n_crop_per_img=args.n_crop_per_img,
        beta=args.beta,
    )
    val_set = SIDDNoisyRAWDataset(
        split_txt=args.val_txt,
        train=False,
        patch_size=args.train_patch_size,
        n_crop_per_img=16,
        beta=args.beta,
    )
    # 不再需要 random_split，场景级别隔离已由 txt 保证
    train_loader = DataLoader(train_set, batch_size=args.bs,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# =========================================================
# TRAIN
# =========================================================

def train_one_ep(model, train_loader, optimizer, criterion, device):
    model.train()

    loss_am      = AverageMeter("loss", ":.4e")
    psnr_noisy_am = AverageMeter("psnr_noisy", ":.2f")
    psnr_gt_am   = AverageMeter("psnr_gt",    ":.2f")
    has_clean = False

    for data in tqdm(train_loader):
        noisy   = tensor_dim5to4(data["noisy"]).to(device)
        noisier = tensor_dim5to4(data["noisier"]).to(device)
        alpha_batch  = data["alpha"].reshape(-1).to(device)
        sigma2_batch = data["sigma2"].reshape(-1).to(device)

        optimizer.zero_grad()
        denoised = model(noisier, alpha_batch, sigma2_batch)
        loss = criterion(denoised, noisy)
        loss.backward()
        optimizer.step()

        loss_am.update(loss.item())

        # vs noisy（原来的）
        res_noisy = psnr_ssim_metric_torch(
            torch.clamp(denoised, 0, 1),
            torch.clamp(noisy, 0, 1),
        )
        psnr_noisy_am.update(res_noisy["psnr"])

        # vs GT（新增）
        if "clean" in data:
            has_clean = True
            clean = tensor_dim5to4(data["clean"]).to(device)
            res_gt = psnr_ssim_metric_torch(
                torch.clamp(denoised, 0, 1),
                torch.clamp(clean, 0, 1),
            )
            psnr_gt_am.update(res_gt["psnr"])

    if has_clean:
        return loss_am.avg, psnr_noisy_am.avg, psnr_gt_am.avg
    else:
        return loss_am.avg, psnr_noisy_am.avg, None

# =========================================================
# VALIDATE
# =========================================================

def validate(model, val_loader, criterion, device):
    model.eval()

    loss_am      = AverageMeter("loss",      ":.4e")
    psnr_n2n_am  = AverageMeter("psnr_n2n", ":.2f")  # noisier→denoised vs noisy
    psnr_gt_am   = AverageMeter("psnr_gt",  ":.2f")  # noisy→denoised vs GT
    ssim_gt_am   = AverageMeter("ssim_gt",  ":.4f")
    has_clean = False

    with torch.no_grad():
        for data in val_loader:
            noisy   = tensor_dim5to4(data["noisy"]).to(device)
            noisier = tensor_dim5to4(data["noisier"]).to(device)
            alpha_batch  = data["alpha"].reshape(-1).to(device)
            sigma2_batch = data["sigma2"].reshape(-1).to(device)

            # noisier → denoised，和 noisy 比（训练目标）
            denoised_n2n = model(noisier, alpha_batch, sigma2_batch)
            loss = criterion(denoised_n2n, noisy)
            loss_am.update(loss.item())

            res_n2n = psnr_ssim_metric_torch(
                torch.clamp(denoised_n2n, 0, 1),
                torch.clamp(noisy, 0, 1),
            )
            psnr_n2n_am.update(res_n2n["psnr"])

            # noisy → denoised，和 GT 比（真实质量）
            if "clean" in data:
                has_clean = True
                clean = tensor_dim5to4(data["clean"]).to(device)
                denoised_ref = model(noisy, alpha_batch, sigma2_batch)
                res_gt = psnr_ssim_metric_torch(
                    torch.clamp(denoised_ref, 0, 1),
                    torch.clamp(clean, 0, 1),
                )
                psnr_gt_am.update(res_gt["psnr"])
                ssim_gt_am.update(res_gt["ssim"])

    if has_clean:
        print(
            f"  [VAL] "
            f"n2n_loss={loss_am.avg:.4e}  "
            f"psnr(noisier→noisy)={psnr_n2n_am.avg:.2f}  "
            f"psnr(noisy→GT)={psnr_gt_am.avg:.2f}  "
            f"ssim(noisy→GT)={ssim_gt_am.avg:.4f}"
        )
    else:
        print(
            f"  [VAL] "
            f"n2n_loss={loss_am.avg:.4e}  "
            f"psnr(noisier→noisy)={psnr_n2n_am.avg:.2f}"
        )

    return loss_am.avg, psnr_n2n_am.avg, psnr_gt_am.avg, has_clean

# =========================================================
# MAIN
# =========================================================

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = ConditionalNAFNet(

        img_channel=4,

        width=32,

        middle_blk_num=12,

        enc_blk_nums=[2, 2, 4, 8],

        dec_blk_nums=[2, 2, 2, 2],

        embed_dim=256,

    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.n_epoch,
        eta_min=1e-5
    )

    criterion = nn.L1Loss()

    train_loader, val_loader = \
        build_dataloaders(args)

    best_val_loss = float("inf")
    best_val_psnr = -float("inf")

    for epoch in range(1, args.n_epoch + 1):

        train_loss, train_psnr_noisy, train_psnr_gt = train_one_ep(
            model, train_loader, optimizer, criterion, device,
        )
        scheduler.step()

        if train_psnr_gt is not None:
            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_loss:.4e}  "
                f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}  "
                f"train_psnr(vs_gt)={train_psnr_gt:.2f}"
            )
        else:
            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_loss:.4e}  "
                f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}"
            )

        if epoch % 5 == 0:

            val_loss, val_psnr_n2n, val_psnr_gt, has_clean = validate(
                model, val_loader, criterion, device,
            )

            # 依据 noisier→noisy 的 psnr 存最好权重
            should_save = (val_psnr_n2n > best_val_psnr)

            if should_save:
                best_val_psnr = val_psnr_n2n
                best_val_loss = val_loss

                os.makedirs(f"./checkpoints/{args.task}", exist_ok=True)
                torch.save(
                    {
                        "epoch":        epoch,
                        "model":        model.state_dict(),
                        "optimizer":    optimizer.state_dict(),
                        "val_loss":     val_loss,
                        "val_psnr_n2n": val_psnr_n2n,
                        "val_psnr_gt":  val_psnr_gt,
                        "beta":         args.beta,
                    },
                    f"./checkpoints/{args.task}/best.pth",
                )
                print("Saved best model!")

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SIDD_Noisier2Noise")
    parser.add_argument("--n_epoch", type=int, default=500)
    parser.add_argument("--train_patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--n_crop_per_img", type=int, default=36)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--noisy_img_dir", type=str, default="/data/xml196414/SIDD/SIDD_Medium/SIDD_Medium_Raw/Data")
    parser.add_argument("--train_txt", type=str, default="my_idea/splits/train.txt")
    parser.add_argument("--val_txt",   type=str, default="my_idea/splits/val.txt")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    main(args)