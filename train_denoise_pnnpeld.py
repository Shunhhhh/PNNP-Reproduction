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
from datasets.pnnp_train_dataset import NoiseSynthesisDataset


# ===================== DATA =====================
def build_dataloaders(args):

    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg_all = yaml.load(f, Loader=yaml.FullLoader)

    cam_cfg = cam_cfg_all[args.camera]

    benchmark_dir = os.path.join(args.benchmark_dir, args.camera)

    full_set = NoiseSynthesisDataset(
        model_path=f"./checkpoints/PNNP_noise/ppm_generator_{args.camera}.pth",
        clean_raw_dir=args.clean_img_dir,
        benchmark_dir=benchmark_dir,
        camera_config=cam_cfg,
        iso_list=[800, 1600, 3200],
        dgain_range=args.train_dgain_range,
        patch_size=args.train_patch_size,
        inp_clip_low=False,
        inp_clip_high=True,
        n_crop_per_img=args.n_crop_per_img,
    )

    train_size = int(0.9 * len(full_set))
    val_size = len(full_set) - train_size

    g = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(
        full_set, [train_size, val_size], generator=g
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


# ===================== TRAIN =====================
def train_one_ep(model, train_loader, optimizer, criterion, device):

    model.train()

    loss_am = AverageMeter("loss", ":.4e")
    psnr_am = AverageMeter("psnr", ":.2f")

    for i, data in enumerate(tqdm(train_loader)):

        clean = tensor_dim5to4(data["clean"]).to(device)
        noisy = tensor_dim5to4(data["noisy"]).to(device)


        optimizer.zero_grad()

        denoised = model(noisy)
        # denoised = torch.clamp(denoised, 0, 1)
        # clean    = torch.clamp(clean,    0, 1)

        # loss = criterion(denoised, clean)

        loss = criterion(denoised, clean)  # clean 已是[0,1]

        loss.backward()
        optimizer.step()

        denoised_clamped = torch.clamp(denoised, 0, 1)
        clean_clamped = torch.clamp(clean, 0, 1)
        res = psnr_ssim_metric_torch(denoised_clamped, clean_clamped)

        # denoised = torch.clamp(denoised, 0, 1)
        # clean = torch.clamp(clean, 0, 1)

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

            denoised = model(noisy)

            denoised = torch.clamp(denoised, 0, 1)
            clean = torch.clamp(clean, 0, 1)

            res = psnr_ssim_metric_torch(denoised, clean)

            psnr_am.update(res["psnr"])
            ssim_am.update(res["ssim"])

    return psnr_am.avg, ssim_am.avg


# ===================== MAIN =====================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # ===== model =====
    model = UNetSeeInDark().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-5)
    criterion = nn.L1Loss()

    train_loader, val_loader = build_dataloaders(args)

    best_psnr = 0

    print("START TRAIN SCRIPT")
    print("GPU init memory:", torch.cuda.memory_allocated() / 1024**3)

    # ===== train loop =====
    for epoch in range(1, args.n_epoch + 1):

        train_loss, train_psnr = train_one_ep(
            model, train_loader, optimizer, criterion, device
        )

        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"loss={train_loss}, psnr={train_psnr:.4f}"
        )

        # ===== validation =====
        if epoch % 5 == 0:

            val_psnr, val_ssim = validate(model, val_loader, device)

            print(
                f"[VAL Epoch {epoch}] "
                f"psnr={val_psnr:.2f}, ssim={val_ssim:.4f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr

                os.makedirs(f"./checkpoints/{args.task}/{args.camera}", exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    },
                    f"./checkpoints/{args.task}/{args.camera}/best.pth"
                )

                print("Saved best model!")


# ===================== ENTRY =====================
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="PNNP_ELD")
    parser.add_argument("--n_epoch", type=int, default=500)
    parser.add_argument("--train_patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--n_crop_per_img", type=int, default=4)
    parser.add_argument("--train_dgain_range", type=list, default=[10.0, 200.0])
    parser.add_argument("--camera", type=str, default="sonyzve10m2")
    parser.add_argument("--clean_img_dir", type=str, default="../../data/xml196414/SID/Sony_npy/long")
    parser.add_argument("--benchmark_dir", type=str, default="../../data/xml196414/SID/dev_phase_release")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = False

    main(args)