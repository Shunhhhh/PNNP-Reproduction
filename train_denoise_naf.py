"""
synth加噪+NAFNet去噪
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch.multiprocessing as mp
import numpy as np
import random
import yaml
from tqdm import tqdm
import argparse
import lpips
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.utils import *
from models.NAFNet_models import NAFNet
from datasets.synth_train_dataset import SynthTrainDataset


# ===================== DATA =====================
def build_dataloaders(args, rank):
    with open("./datasets/camera_config.yaml", "r") as f:
        cam_cfg = yaml.load(f, Loader=yaml.FullLoader)

    full_set = SynthTrainDataset(
        clean_img_dir=args.clean_img_dir,
        benchmark_dir=args.benchmark_dir,
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
    g = torch.Generator().manual_seed(42) # 固定划分种子
    train_set, val_set = torch.utils.data.random_split(full_set, [train_size, val_size], generator=g)

    train_sampler = DistributedSampler(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if rank == 0:
        print(f">>>>> train_set: {len(train_set)}")
        print(f">>>>> val_set: {len(val_set)}")

    return train_loader, val_loader, train_sampler


# ===================== TRAIN =====================
def train_one_ep(model, train_loader, optimizer, criterion, device, rank):
    model.train()

    loss_am = AverageMeter("train_loss", ":.4e")
    psnr_am = AverageMeter("train_psnr", ":.2f")

    for data in tqdm(train_loader, disable=(rank != 0)):
        clean = tensor_dim5to4(data["clean"]).to(device, non_blocking=True)
        noisy = tensor_dim5to4(data["noisy"]).to(device, non_blocking=True)

        optimizer.zero_grad()

        denoised = model(noisy)
        loss = criterion(denoised, clean)

        loss.backward()
        optimizer.step()

        denoised = torch.clamp(denoised, 0, 1)
        clean = torch.clamp(clean, 0, 1)

        res = psnr_ssim_metric_torch(denoised, clean)

        loss_am.update(loss.item())
        psnr_am.update(res["psnr"])

    return loss_am.avg, psnr_am.avg


# ===================== VALIDATE =====================
def validate(model, val_loader, device):
    model.eval()
    psnr_am = AverageMeter("val_psnr", ":.2f")
    ssim_am = AverageMeter("val_ssim", ":.4f")


    with torch.no_grad():
        for data in val_loader:
            clean = tensor_dim5to4(data["clean"]).to(device, non_blocking=True)
            noisy = tensor_dim5to4(data["noisy"]).to(device, non_blocking=True)

            denoised = model(noisy)
            denoised = torch.clamp(denoised, 0, 1)
            clean = torch.clamp(clean, 0, 1)

            res = psnr_ssim_metric_torch(denoised, clean)
            psnr_am.update(res["psnr"])
            ssim_am.update(res["ssim"])



    return psnr_am.avg, ssim_am.avg


# ===================== MAIN =====================
def main(args):

    # ===== DDP 初始化 =====
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print("Using DDP, device:", device)

    # ===== 模型 =====
    model = NAFNet(
        img_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )


    optimizer = AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1

    # ===== resume =====
    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            print(f"=> loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=device)

        model.module.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-5)
    criterion = nn.L1Loss()

    train_loader, val_loader, train_sampler = build_dataloaders(args, rank)

    if rank == 0:
        make_directory(f"./checkpoints/{args.task}")
        make_directory("./logs")
        logfile = f"./logs/{args.task}.log"
        open(logfile, "w").close()

    best_psnr = 0

    # ===== 训练循环 =====
    for epoch in range(start_epoch, args.n_epoch + 1):

        train_sampler.set_epoch(epoch)

        train_loss, train_psnr = train_one_ep(
            model, train_loader, optimizer, criterion, device, rank
        )
        scheduler.step()
        if rank == 0:
            log(
                f"epoch: {epoch}, "
                f"train loss: {train_loss:.4e}, "
                f"train psnr: {train_psnr:.2f}",
                log=logfile,
                notime=True,
            )

        # ===== 保存 =====
        if rank == 0 and epoch % args.save_freq == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                f"./checkpoints/{args.task}/epoch_{epoch}.pth"
            )

        # ===== 验证 =====
        if epoch % 5 == 0 and rank == 0:
            val_psnr, val_ssim = validate(model, val_loader, device)
            if rank == 0:
                log_str = (f"epoch: {epoch}, val psnr: {val_psnr:.2f}, "
                            f"val ssim: {val_ssim:.4f}")
                log(log_str, log=logfile, notime=True)

            print(f"[Epoch {epoch}] " + log_str)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    f"./checkpoints/{args.task}/best.pth"
                )
                if rank == 0:
                    log(
                        f"best epoch: {epoch}, val psnr: {val_psnr:.2f}, saved",
                        log=logfile,
                        notime=True,
                    )

    dist.destroy_process_group()


# ===================== ENTRY =====================
if __name__ == "__main__":
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NAFNet")
    parser.add_argument("--n_epoch", type=int, default=500)
    parser.add_argument("--train_patch_size", type=int, default=512) 
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=2) 
    parser.add_argument("--n_crop_per_img", type=int, default=4)
    parser.add_argument("--train_dgain_range", type=list, default=[10, 200])
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--clean_img_dir", type=str, default="../../data/xml196414/SID/Sony/long")
    parser.add_argument("--benchmark_dir", type=str, default="../../data/xml196414/SID/dev_phase_release")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = False

    main(args)