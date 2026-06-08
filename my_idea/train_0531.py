"""
ConditionalDenoiser 训练脚本
Noisier2Noise + RAW Poisson-Gaussian

数据划分策略:
  - Scene 1, 2, 8  : 噪声参数估计 + 合成 noisy-clean pair + 训练/验证
  - 其余所有 scene : 纯去噪推理测试（无 GT，仅输出 denoised 结果）
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
import glob

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from utils.utils import *

from my_idea.conditional_denoiser import ConditionalNAFNet
from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset


# =========================================================
# SCENE 划分常量
# =========================================================

TRAIN_SCENES = {1, 2, 8}   # 用于噪声估计 + 合成训练数据
VAL_RATIO    = 0.2          # scene 目录级别的 hold-out 比例


def scene_id_from_path(path: str) -> int:
    """
    SIDD 目录结构:
        .../Data/0001_001_S6_00100_00060_3200_L/0001_NOISY_RAW_010.MAT
    scene id 在父目录名的第二个字段（三位数字），如 001 → 1。
    """
    import re
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    m = re.match(r"^\d{4}_(\d{3})_", parent)
    if m:
        return int(m.group(1))
    return -1


# =========================================================
# SPLIT TXT 自动生成
# =========================================================

def build_split_txts(noisy_img_dir: str, split_dir: str):
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt   = os.path.join(split_dir, "val.txt")
    test_txt  = os.path.join(split_dir, "test.txt")

    if os.path.exists(train_txt) and os.path.exists(val_txt) and os.path.exists(test_txt):
        train_lines_check = [l.strip() for l in open(train_txt) if l.strip()]
        val_lines_check   = [l.strip() for l in open(val_txt)   if l.strip()]
        test_lines_check  = [l.strip() for l in open(test_txt)  if l.strip()]
        if train_lines_check and val_lines_check and test_lines_check:
            print(
                f"[split] 复用已有 split: {split_dir}  "
                f"(train={len(train_lines_check)}, "
                f"val={len(val_lines_check)}, "
                f"test={len(test_lines_check)})"
            )
            return train_txt, val_txt, test_txt
        else:
            print("[split] 已有 split 文件内容不完整，重新生成…")

    os.makedirs(split_dir, exist_ok=True)

    if not os.path.isdir(noisy_img_dir):
        raise FileNotFoundError(
            f"[split] noisy_img_dir 不存在或不是目录: {noisy_img_dir}"
        )

    all_files = sorted(
        glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.MAT"), recursive=True) +
        glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.mat"), recursive=True) +
        glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.PNG"), recursive=True) +
        glob.glob(os.path.join(noisy_img_dir, "**", "*NOISY*.png"), recursive=True)
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

    for f in all_files:
        sid = scene_id_from_path(f)
        if sid == -1:
            parse_failures.append(f)
            continue
        scene_dir = os.path.dirname(os.path.normpath(f))
        if sid in TRAIN_SCENES:
            train_val_dirs.add(scene_dir)
        else:
            test_dirs.add(scene_dir)

    if parse_failures:
        print(
            f"[split] 警告：{len(parse_failures)} 个文件无法解析 scene id，已跳过。\n"
            f"  示例（前 5 个）: {parse_failures[:5]}"
        )

    train_val_dirs = sorted(train_val_dirs)
    test_dirs      = sorted(test_dirs)

    if not train_val_dirs and not test_dirs:
        raise RuntimeError(
            f"[split] 场景列表为空。共扫描 {len(all_files)} 个文件，"
            f"其中 {len(parse_failures)} 个解析失败。\n"
            f"  解析失败示例（前 5 个）: {parse_failures[:5]}"
        )

    if not train_val_dirs:
        scene_ids_found = sorted({
            scene_id_from_path(f) for f in all_files if scene_id_from_path(f) != -1
        })
        raise RuntimeError(
            f"[split] 训练 scene {sorted(TRAIN_SCENES)} 未找到任何目录。\n"
            f"  数据集中实际存在的 scene id: {scene_ids_found}"
        )

    # scene 目录级别的 train/val 拆分，固定 seed 保证可复现
    rng = random.Random(42)
    shuffled = train_val_dirs[:]
    rng.shuffle(shuffled)
    n_val      = max(1, int(len(shuffled) * VAL_RATIO))
    val_dirs   = shuffled[:n_val]
    train_dirs = shuffled[n_val:]

    def write_txt(path, lines):
        with open(path, "w") as fp:
            fp.write("\n".join(lines) + "\n")

    write_txt(train_txt, sorted(train_dirs))
    write_txt(val_txt,   sorted(val_dirs))
    write_txt(test_txt,  sorted(test_dirs))

    print(
        f"[split] train={len(train_dirs)} dirs  "
        f"val={len(val_dirs)} dirs  "
        f"test={len(test_dirs)} dirs  "
        f"(train/val 来自 scene {sorted(TRAIN_SCENES)}，"
        f"val_ratio={VAL_RATIO})"
    )
    return train_txt, val_txt, test_txt

# =========================================================
# DATA
# =========================================================
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
    )


    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
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
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        loss_fft   = self.l1(pred_fft.abs(), target_fft.abs())
        return loss_fft * self.weight


class FFTLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        loss_fft   = self.l1(pred_fft.abs(), target_fft.abs())
        return loss_fft * self.weight

# =========================================================
# TRAIN
# =========================================================

def train_one_ep(model, train_loader, optimizer, criterion_l1, criterion_fft, device):
    model.train()

    loss_am       = AverageMeter("loss",       ":.4e")
    psnr_noisy_am = AverageMeter("psnr_noisy", ":.2f")
    psnr_gt_am    = AverageMeter("psnr_gt",    ":.2f")
    has_clean = False

    for data in tqdm(train_loader, desc="train", leave=False):
        noisy        = tensor_dim5to4(data["noisy"]).to(device)
        noisier = tensor_dim5to4(data["noisier"]).to(device)
        noisier = torch.clamp(noisier, 0, 1) 
        alpha_batch  = data["alpha"].reshape(-1).to(device)
        sigma2_batch = data["sigma2"].reshape(-1).to(device)

        optimizer.zero_grad()
        denoised = model(noisier, alpha_batch, sigma2_batch)
        loss = criterion_l1(denoised, noisy) + criterion_fft(denoised, noisy)
        loss.backward()
        optimizer.step()

        loss_am.update(loss.item())

        res_noisy = psnr_ssim_metric_torch(
            torch.clamp(denoised, 0, 1),
            torch.clamp(noisy,    0, 1),
        )
        psnr_noisy_am.update(res_noisy["psnr"])

        if "clean" in data:
            has_clean = True
            clean  = tensor_dim5to4(data["clean"]).to(device)
            res_gt = psnr_ssim_metric_torch(
                torch.clamp(denoised, 0, 1),
                torch.clamp(clean,    0, 1),
            )
            psnr_gt_am.update(res_gt["psnr"])

    if has_clean:
        return loss_am.avg, psnr_noisy_am.avg, psnr_gt_am.avg
    else:
        return loss_am.avg, psnr_noisy_am.avg, None
        
# =========================================================
# VALIDATE（scene 1/2/8 hold-out 20%，有合成 clean GT）
# =========================================================

def validate(model, val_loader, criterion_l1, criterion_fft, device):
    model.eval()

    loss_am     = AverageMeter("loss",     ":.4e")
    psnr_n2n_am = AverageMeter("psnr_n2n", ":.2f")
    psnr_gt_am  = AverageMeter("psnr_gt",  ":.2f")
    ssim_gt_am  = AverageMeter("ssim_gt",  ":.4f")
    has_clean = False

    with torch.no_grad():
        for data in tqdm(val_loader, desc="val", leave=False):
            noisy        = tensor_dim5to4(data["noisy"]).to(device)
            noisier      = tensor_dim5to4(data["noisier"]).to(device)
            alpha_batch  = data["alpha"].reshape(-1).to(device)
            sigma2_batch = data["sigma2"].reshape(-1).to(device)

            # 训练信号验证：noisier → denoised vs noisy
            denoised_n2n = model(noisier, alpha_batch, sigma2_batch)
            loss = criterion_l1(denoised_n2n, noisy) + criterion_fft(denoised_n2n, noisy)
            loss_am.update(loss.item())

            res_n2n = psnr_ssim_metric_torch(
                torch.clamp(denoised_n2n, 0, 1),
                torch.clamp(noisy,        0, 1),
            )
            psnr_n2n_am.update(res_n2n["psnr"])

            # GT 质量验证：noisy → denoised vs clean
            if "clean" in data:
                has_clean = True
                clean        = tensor_dim5to4(data["clean"]).to(device)
                denoised_ref = model(noisy, alpha_batch, sigma2_batch)
                res_gt = psnr_ssim_metric_torch(
                    torch.clamp(denoised_ref, 0, 1),
                    torch.clamp(clean,        0, 1),
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
# INFERENCE-ONLY TEST（其余 scene，无 GT）
# =========================================================

def run_test_inference(model, test_loader, device, save_dir=None):
    model.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    noisy_mean_am    = AverageMeter("noisy_mean",    ":.4f")
    denoised_mean_am = AverageMeter("denoised_mean", ":.4f")
    noisy_std_am    = AverageMeter("noisy_std",    ":.4f")
    denoised_std_am = AverageMeter("denoised_std", ":.4f")

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc="test-infer", leave=False)):
            noisy        = tensor_dim5to4(data["noisy"]).to(device)
            alpha_batch  = data["alpha"].reshape(-1).to(device)
            sigma2_batch = data["sigma2"].reshape(-1).to(device)

            denoised = model(noisy, alpha_batch, sigma2_batch)
            denoised = torch.clamp(denoised, 0, 1)

            noisy_mean_am.update(noisy.mean().item())
            denoised_mean_am.update(denoised.mean().item())
            noisy_std_am.update(noisy.std().item())
            denoised_std_am.update(denoised.std().item())

            if save_dir:
                torch.save(
                    denoised.cpu(),
                    os.path.join(save_dir, f"denoised_{idx:05d}.pt"),
                )

    print(
        f"  [TEST-INFER] "
        f"noisy:    mean={noisy_mean_am.avg:.4f}  std={noisy_std_am.avg:.4f}\n"
        f"  [TEST-INFER] "
        f"denoised: mean={denoised_mean_am.avg:.4f}  std={denoised_std_am.avg:.4f}\n"
        f"  std_ratio={denoised_std_am.avg/noisy_std_am.avg:.4f}  "
        f"(越小说明去噪越强，接近1说明接近恒等映射)"
    )
    return denoised_mean_am.avg


# =========================================================
# MAIN
# =========================================================

def main(args):
    criterion_l1  = nn.L1Loss()
    criterion_fft = FFTLoss(weight=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"训练/验证 scene : {sorted(TRAIN_SCENES)}  (train {1-VAL_RATIO:.0%} / val {VAL_RATIO:.0%})")
    print(f"测试 scene      : 其余所有 scene（纯推理，无 GT）")

    # ---- 自动生成 split txt ----
    if args.train_txt is None or args.val_txt is None or args.test_txt is None:
        args.train_txt, args.val_txt, args.test_txt = build_split_txts(
            args.noisy_img_dir,
            split_dir=os.path.join("my_idea", "0531splits"),
        )
    else:
        for p in [args.train_txt, args.val_txt, args.test_txt]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"split 文件不存在: {p}")

    # ---- 模型 ----
    model = ConditionalNAFNet(
        img_channel=4,
        width=48,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epoch, eta_min=1e-5
    )

    # ---- DataLoader ----
    train_loader, val_loader, test_loader = build_dataloaders(args)

    best_val_psnr = -float("inf")
    best_val_loss = float("inf")

    for epoch in range(1, args.n_epoch + 1):

        # ---- 训练（scene 1/2/8 的 80% 目录）----
        train_loss, train_psnr_noisy, train_psnr_gt = train_one_ep(
            model, train_loader, optimizer, criterion_l1, criterion_fft, device,
        )
        scheduler.step()

        if train_psnr_gt is not None:
            print(
                f"[Epoch {epoch:>4d}] "
                f"train_loss={train_loss:.4e}  "
                f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}  "
                f"train_psnr(vs_gt)={train_psnr_gt:.2f}"
            )
        else:
            print(
                f"[Epoch {epoch:>4d}] "
                f"train_loss={train_loss:.4e}  "
                f"train_psnr(vs_noisy)={train_psnr_noisy:.2f}"
            )

        # ---- 每 5 epoch：hold-out val 验证（scene 1/2/8 的 20% 目录）----
        if epoch % 5 == 0:
            val_loss, val_psnr_n2n, val_psnr_gt, has_clean = validate(
                model, val_loader, criterion_l1, criterion_fft, device,
            )

            if has_clean:
                should_save = (val_psnr_gt > best_val_psnr)
            else:
                should_save = (val_psnr_n2n > best_val_psnr)

            if should_save:
                best_val_psnr = val_psnr_n2n
                best_val_loss = val_loss

                ckpt_dir = f"./checkpoints/{args.task}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch":        epoch,
                        "model":        model.state_dict(),
                        "optimizer":    optimizer.state_dict(),
                        "val_loss":     val_loss,
                        "val_psnr_n2n": val_psnr_n2n,
                        "val_psnr_gt":  val_psnr_gt,
                        "beta_min":     args.beta_min,
                        "beta_max":     args.beta_max,
                        "noise_low":    args.noise_low,
                        "noise_high":   args.noise_high,
                        "train_scenes": sorted(TRAIN_SCENES),
                    },
                    os.path.join(ckpt_dir, "best.pth"),
                )
                print("  → Saved best model!")

        # ---- 每 5 epoch：对其余 scene 做纯推理（无 GT）----
        if epoch % 5 == 0:
            infer_save_dir = (
                os.path.join(f"./outputs/{args.task}", f"epoch_{epoch:04d}")
                if args.save_test_outputs else None
            )
            run_test_inference(model, test_loader, device, save_dir=infer_save_dir)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task",             type=str,   default="SIDD_Noisier2Noise0531")
    parser.add_argument("--n_epoch",          type=int,   default=200)
    parser.add_argument("--train_patch_size", type=int,   default=256)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--bs",               type=int,   default=1)
    parser.add_argument("--n_crop_per_img",   type=int,   default=36)
    parser.add_argument("--beta_min",    type=float, default=1.2)
    parser.add_argument("--beta_max",    type=float, default=2.0)
    parser.add_argument("--noise_low",   type=float, default=0.01,
                        help="noise_level 低端锚点，对应 beta_max")
    parser.add_argument("--noise_high",  type=float, default=0.05,
                        help="noise_level 高端锚点，对应 beta_min")
    parser.add_argument("--noisy_img_dir",    type=str,
                        default="/data/xml196414/SIDD/SIDD_Medium/SIDD_Medium_Raw/Data")
    parser.add_argument("--train_txt",        type=str,   default=None)
    parser.add_argument("--val_txt",          type=str,   default=None)
    parser.add_argument("--test_txt",         type=str,   default=None)
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--save_test_outputs", action="store_true",
                        help="是否将其余 scene 的去噪结果保存到 ./outputs/")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    main(args)