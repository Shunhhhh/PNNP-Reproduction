"""
ConditionalDenoiser training script.
Noisier2Noise + RAW Poisson-Gaussian + row-noise condition.
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
from my_idea.conditional_denoiser import ConditionalNAFNet
from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset


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
            {scene_id_from_path(file_path) for file_path in all_files if scene_id_from_path(file_path) != -1}
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
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class FFTLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return self.l1(pred_fft.abs(), target_fft.abs()) * self.weight


class SupervisedLoss(nn.Module):
    def __init__(self, fft_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.fft = FFTLoss(weight=fft_weight)

    def forward(self, pred, clean):
        return self.l1(pred, clean) + self.fft(pred, clean)


def train_one_ep(
    model,
    train_loader,
    optimizer,
    criterion_l1,
    criterion_fft,
    device,
    supervised_loss=None,
    sup_weight=0.5,
):
    model.train()

    loss_am = AverageMeter("loss", ":.4e")
    loss_n2n_am = AverageMeter("loss_n2n", ":.4e")
    loss_sup_am = AverageMeter("loss_sup", ":.4e")
    row_std_am = AverageMeter("row_std", ":.4e")
    psnr_noisy_am = AverageMeter("psnr_noisy", ":.2f")
    psnr_gt_am = AverageMeter("psnr_gt", ":.2f")
    psnr_ref_am = AverageMeter("psnr_ref", ":.2f")
    has_clean = False

    for data in tqdm(train_loader, desc="train", leave=False):
        noisy = tensor_dim5to4(data["noisy"]).to(device)
        noisier = tensor_dim5to4(data["noisier"]).to(device)
        noisier = torch.clamp(noisier, 0, 1)
        alpha_batch = data["alpha"].reshape(-1).to(device)
        sigma2_batch = data["sigma2"].reshape(-1).to(device)
        row_std_batch = data["row_std"].reshape(-1).to(device)
        row_std_am.update(row_std_batch.mean().item())

        has_clean_batch = "clean" in data and supervised_loss is not None and sup_weight > 0

        optimizer.zero_grad()

        if has_clean_batch:
            has_clean = True
            clean = tensor_dim5to4(data["clean"]).to(device)

            combined_input = torch.cat([noisier, noisy], dim=0)
            combined_alpha = torch.cat([alpha_batch, alpha_batch], dim=0)
            combined_sigma2 = torch.cat([sigma2_batch, sigma2_batch], dim=0)
            combined_row_std = torch.cat([row_std_batch, row_std_batch], dim=0)

            combined_out = model(
                combined_input,
                combined_alpha,
                combined_sigma2,
                combined_row_std,
            )
            batch_size = noisy.shape[0]
            denoised = combined_out[:batch_size]
            denoised_sup = combined_out[batch_size:]

            loss_n2n = criterion_l1(denoised, noisy) + criterion_fft(denoised, noisy)
            loss_sup = supervised_loss(torch.clamp(denoised_sup, 0, 1), clean)
            loss = loss_n2n + sup_weight * loss_sup

            loss_n2n_am.update(loss_n2n.item())
            loss_sup_am.update(loss_sup.item())

            with torch.no_grad():
                res_gt = psnr_ssim_metric_torch(torch.clamp(denoised, 0, 1), clean)
                res_ref = psnr_ssim_metric_torch(torch.clamp(denoised_sup, 0, 1), clean)
            psnr_gt_am.update(res_gt["psnr"])
            psnr_ref_am.update(res_ref["psnr"])
        else:
            denoised = model(noisier, alpha_batch, sigma2_batch, row_std_batch)
            loss_n2n = criterion_l1(denoised, noisy) + criterion_fft(denoised, noisy)
            loss = loss_n2n
            loss_n2n_am.update(loss_n2n.item())

        loss.backward()
        optimizer.step()
        loss_am.update(loss.item())

        with torch.no_grad():
            res_noisy = psnr_ssim_metric_torch(
                torch.clamp(denoised, 0, 1),
                torch.clamp(noisy, 0, 1),
            )
        psnr_noisy_am.update(res_noisy["psnr"])

    if has_clean:
        print(
            f"  [TRAIN-DETAIL] "
            f"loss_n2n={loss_n2n_am.avg:.4e} "
            f"loss_sup={loss_sup_am.avg:.4e} "
            f"row_std={row_std_am.avg:.4e} "
            f"psnr(noisier->GT)={psnr_gt_am.avg:.2f} "
            f"psnr(noisy->GT)={psnr_ref_am.avg:.2f}"
        )
        return loss_am.avg, psnr_noisy_am.avg, psnr_ref_am.avg

    print(
        f"  [TRAIN-DETAIL] "
        f"loss_n2n={loss_n2n_am.avg:.4e} "
        f"row_std={row_std_am.avg:.4e}"
    )
    return loss_am.avg, psnr_noisy_am.avg, None


def validate(model, val_loader, criterion_l1, criterion_fft, device):
    model.eval()

    loss_am = AverageMeter("loss", ":.4e")
    psnr_n2n_am = AverageMeter("psnr_n2n", ":.2f")
    psnr_gt_am = AverageMeter("psnr_gt", ":.2f")
    ssim_gt_am = AverageMeter("ssim_gt", ":.4f")
    row_std_am = AverageMeter("row_std", ":.4e")
    has_clean = False

    with torch.no_grad():
        for data in tqdm(val_loader, desc="val", leave=False):
            noisy = tensor_dim5to4(data["noisy"]).to(device)
            noisier = tensor_dim5to4(data["noisier"]).to(device)
            alpha_batch = data["alpha"].reshape(-1).to(device)
            sigma2_batch = data["sigma2"].reshape(-1).to(device)
            row_std_batch = data["row_std"].reshape(-1).to(device)
            row_std_am.update(row_std_batch.mean().item())

            denoised_n2n = model(noisier, alpha_batch, sigma2_batch, row_std_batch)
            loss = criterion_l1(denoised_n2n, noisy) + criterion_fft(denoised_n2n, noisy)
            loss_am.update(loss.item())

            res_n2n = psnr_ssim_metric_torch(
                torch.clamp(denoised_n2n, 0, 1),
                torch.clamp(noisy, 0, 1),
            )
            psnr_n2n_am.update(res_n2n["psnr"])

            if "clean" in data:
                has_clean = True
                clean = tensor_dim5to4(data["clean"]).to(device)
                denoised_ref = model(noisy, alpha_batch, sigma2_batch, row_std_batch)
                res_gt = psnr_ssim_metric_torch(
                    torch.clamp(denoised_ref, 0, 1),
                    torch.clamp(clean, 0, 1),
                )
                psnr_gt_am.update(res_gt["psnr"])
                ssim_gt_am.update(res_gt["ssim"])

    if has_clean:
        print(
            f"  [VAL] "
            f"n2n_loss={loss_am.avg:.4e} "
            f"row_std={row_std_am.avg:.4e} "
            f"psnr(noisier->noisy)={psnr_n2n_am.avg:.2f} "
            f"psnr(noisy->GT)={psnr_gt_am.avg:.2f} "
            f"ssim(noisy->GT)={ssim_gt_am.avg:.4f}"
        )
    else:
        print(
            f"  [VAL] "
            f"n2n_loss={loss_am.avg:.4e} "
            f"row_std={row_std_am.avg:.4e} "
            f"psnr(noisier->noisy)={psnr_n2n_am.avg:.2f}"
        )

    return loss_am.avg, psnr_n2n_am.avg, psnr_gt_am.avg, has_clean


def run_test_inference(model, test_loader, device, save_dir=None):
    model.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    noisy_mean_am = AverageMeter("noisy_mean", ":.4f")
    denoised_mean_am = AverageMeter("denoised_mean", ":.4f")
    noisy_std_am = AverageMeter("noisy_std", ":.4f")
    denoised_std_am = AverageMeter("denoised_std", ":.4f")
    row_std_am = AverageMeter("row_std", ":.4e")

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc="test-infer", leave=False)):
            noisy = tensor_dim5to4(data["noisy"]).to(device)
            alpha_batch = data["alpha"].reshape(-1).to(device)
            sigma2_batch = data["sigma2"].reshape(-1).to(device)
            row_std_batch = data["row_std"].reshape(-1).to(device)
            row_std_am.update(row_std_batch.mean().item())

            denoised = model(noisy, alpha_batch, sigma2_batch, row_std_batch)
            denoised = torch.clamp(denoised, 0, 1)

            noisy_mean_am.update(noisy.mean().item())
            denoised_mean_am.update(denoised.mean().item())
            noisy_std_am.update(noisy.std().item())
            denoised_std_am.update(denoised.std().item())

            if save_dir:
                torch.save(denoised.cpu(), os.path.join(save_dir, f"denoised_{idx:05d}.pt"))

    print(
        f"  [TEST-INFER] "
        f"row_std={row_std_am.avg:.4e}\n"
        f"  [TEST-INFER] "
        f"noisy: mean={noisy_mean_am.avg:.4f} std={noisy_std_am.avg:.4f}\n"
        f"  [TEST-INFER] "
        f"denoised: mean={denoised_mean_am.avg:.4f} std={denoised_std_am.avg:.4f}\n"
        f"  std_ratio={denoised_std_am.avg / noisy_std_am.avg:.4f}"
    )
    return denoised_mean_am.avg


def main(args):
    criterion_l1 = nn.L1Loss()
    criterion_fft = FFTLoss(weight=0.1)
    criterion_sup = SupervisedLoss(fft_weight=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"训练/验证 scene : {sorted(TRAIN_SCENES)} (train {1 - VAL_RATIO:.0%} / val {VAL_RATIO:.0%})")
    print("测试 scene      : 其余所有 scene（纯推理，无 GT）")
    print(
        f"row noise       : add={args.add_row_noise} "
        f"scale={args.row_noise_scale} smooth_kernel={args.row_smooth_kernel}"
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

    model = ConditionalNAFNet(
        img_channel=4,
        width=42,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch, eta_min=1e-5)

    train_loader, val_loader, test_loader = build_dataloaders(args)

    best_val_psnr_n2n = -float("inf")
    best_val_loss = float("inf")

    for epoch in range(1, args.n_epoch + 1):
        if epoch <= args.pretrain_epochs:
            sup_weight = 0.0
        else:
            ratio = (epoch - args.pretrain_epochs) / max(args.n_epoch - args.pretrain_epochs, 1)
            sup_weight = args.sup_weight_max * min(ratio * 3, 1.0)

        train_loss, train_psnr_noisy, train_psnr_gt = train_one_ep(
            model,
            train_loader,
            optimizer,
            criterion_l1,
            criterion_fft,
            device,
            supervised_loss=criterion_sup,
            sup_weight=sup_weight,
        )
        scheduler.step()

        print(
            f"[Epoch {epoch:>4d}] "
            f"sup_weight={sup_weight:.3f} "
            f"train_loss={train_loss:.4e} "
            + (
                f"train_psnr(noisy->GT)={train_psnr_gt:.2f}"
                if train_psnr_gt
                else f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}"
            )
        )

        if epoch % 5 == 0:
            val_loss, val_psnr_n2n, val_psnr_gt, has_clean = validate(
                model,
                val_loader,
                criterion_l1,
                criterion_fft,
                device,
            )

            should_save = val_psnr_n2n > best_val_psnr_n2n
            if should_save:
                best_val_psnr_n2n = val_psnr_n2n
                best_val_loss = val_loss

                ckpt_dir = f"./checkpoints/{args.task}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
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
                        "train_scenes": sorted(TRAIN_SCENES),
                    },
                    os.path.join(ckpt_dir, "best.pth"),
                )
                print("  -> Saved best model!")

        if epoch % 5 == 0:
            infer_save_dir = (
                os.path.join(f"./outputs/{args.task}", f"epoch_{epoch:04d}")
                if args.save_test_outputs
                else None
            )
            run_test_inference(model, test_loader, device, save_dir=infer_save_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SIDD_Noisier2Noise_row_noise")
    parser.add_argument("--n_epoch", type=int, default=300)
    parser.add_argument("--train_patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--n_crop_per_img", type=int, default=8)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--sup_weight_max", type=float, default=0.5)
    parser.add_argument("--beta_min", type=float, default=1.2)
    parser.add_argument("--beta_max", type=float, default=2.0)
    parser.add_argument("--noise_low", type=float, default=0.01)
    parser.add_argument("--noise_high", type=float, default=0.05)
    parser.add_argument("--add_row_noise", action="store_true")
    parser.add_argument("--row_noise_scale", type=float, default=1.0)
    parser.add_argument("--row_smooth_kernel", type=int, default=0)
    parser.add_argument(
        "--noisy_img_dir",
        type=str,
        default="/data/xml196414/SIDD/SIDD_Medium/SIDD_Medium_Raw/Data",
    )
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
