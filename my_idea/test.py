"""
ConditionalDenoiser inference script.

Supports the row-noise conditioned model trained with:
    model(x, alpha, sigma2, row_std)

It saves noisy / denoised / clean RGB previews and reports:
    - RAW / RGB PSNR, SSIM if clean exists
    - noisy / denoised std ratio
    - residual std
    - G1-G2 std before and after denoising
"""

import os
import argparse
import random
import sys
import csv

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from my_idea.conditional_denoiser import ConditionalNAFNet
from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset
from utils.utils import psnr_ssim_metric_torch


def isp_process(packed, wb=None, ccm=None, gamma=2.2):
    packed = packed.float()
    _, H, W = packed.shape

    if wb is not None:
        wb = torch.tensor(wb, dtype=torch.float32).view(4, 1, 1)
        packed = torch.clamp(packed * wb, 0, 1)

    R = packed[0]
    G = (packed[1] + packed[2]) / 2.0
    B = packed[3]
    rgb = torch.stack([R, G, B], dim=0)

    if ccm is not None:
        ccm = torch.tensor(ccm, dtype=torch.float32)
        rgb_flat = ccm @ rgb.view(3, -1)
        rgb = torch.clamp(rgb_flat.view(3, H, W), 0, 1)

    rgb = torch.pow(torch.clamp(rgb, 1e-8, 1.0), 1.0 / gamma)
    return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image

    Image.fromarray(img).save(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MetricAccumulator:
    def __init__(self):
        self.psnr_list = []
        self.ssim_list = []

    def update(self, psnr, ssim):
        self.psnr_list.append(psnr)
        self.ssim_list.append(ssim)

    def summary(self):
        if not self.psnr_list:
            return {}
        psnr = np.array(self.psnr_list)
        ssim = np.array(self.ssim_list)
        return {
            "n": len(psnr),
            "psnr_mean": float(psnr.mean()),
            "psnr_std": float(psnr.std()),
            "psnr_max": float(psnr.max()),
            "psnr_min": float(psnr.min()),
            "ssim_mean": float(ssim.mean()),
            "ssim_std": float(ssim.std()),
            "ssim_max": float(ssim.max()),
            "ssim_min": float(ssim.min()),
        }

    def has_data(self):
        return len(self.psnr_list) > 0


class ScalarAccumulator:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(float(value))

    def mean(self):
        return float(np.mean(self.values)) if self.values else float("nan")

    def std(self):
        return float(np.std(self.values)) if self.values else float("nan")


def green_diff_std(x):
    diff = x[:, 1:2] - x[:, 2:3]
    return diff.std().item()


def green_row_std(x):
    diff = x[:, 1:2] - x[:, 2:3]
    row_profile = diff.median(dim=3, keepdim=True).values
    row_profile = row_profile - row_profile.mean(dim=2, keepdim=True)
    return row_profile.std().item()


def denoise_patch_direct(model, noisy_4d, alpha_batch, sigma2_batch, row_std_batch):
    denoised = model(noisy_4d, alpha_batch, sigma2_batch, row_std_batch)
    return torch.clamp(denoised, 0, 1)


def denoise_patch_r2r(
    model,
    noisy_4d,
    alpha_batch,
    sigma2_batch,
    row_std_batch,
    beta,
):
    var_z = beta * (
        alpha_batch.view(-1, 1, 1, 1) * torch.clamp(noisy_4d, min=0)
        + sigma2_batch.view(-1, 1, 1, 1)
    )
    pixel_noise = torch.randn_like(noisy_4d) * torch.sqrt(torch.clamp(var_z, min=1e-10))

    row_noise = torch.randn(
        noisy_4d.shape[0],
        noisy_4d.shape[1],
        noisy_4d.shape[2],
        1,
        device=noisy_4d.device,
        dtype=noisy_4d.dtype,
    ) * row_std_batch.view(-1, 1, 1, 1)

    z_noisy = torch.clamp(noisy_4d + pixel_noise + row_noise, 0, 1)
    f_z = model(z_noisy, alpha_batch, sigma2_batch, row_std_batch)
    denoised = 2 * f_z - z_noisy
    return torch.clamp(denoised, 0, 1)


def denoise_patch(model, noisy_4d, alpha_batch, sigma2_batch, row_std_batch, args):
    if args.infer_mode == "direct":
        return denoise_patch_direct(model, noisy_4d, alpha_batch, sigma2_batch, row_std_batch)
    if args.infer_mode == "r2r":
        return denoise_patch_r2r(
            model,
            noisy_4d,
            alpha_batch,
            sigma2_batch,
            row_std_batch,
            beta=args.beta,
        )
    raise ValueError(f"Unknown infer_mode: {args.infer_mode}")


def load_model(args, device):
    model = ConditionalNAFNet(
        img_channel=4,
        width=42,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded checkpoint:", args.ckpt)
    if isinstance(ckpt, dict):
        print(
            f"  epoch={ckpt.get('epoch', '?')} "
            f"val_loss={ckpt.get('val_loss', float('nan')):.4e} "
            f"val_psnr_n2n={ckpt.get('val_psnr_n2n', float('nan'))}"
        )
        if "row_noise_scale" in ckpt:
            print(
                f"  ckpt row: add={ckpt.get('add_row_noise')} "
                f"scale={ckpt.get('row_noise_scale')} "
                f"smooth={ckpt.get('row_smooth_kernel')}"
            )

    return model


def main(args):
    camera_config = {
        "sonyzve10m2": {"wb": [2.0, 1.0, 1.0, 1.6], "ccm": None},
        "neutral": {"wb": [1.0, 1.0, 1.0, 1.0], "ccm": None},
    }

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Inference mode: {args.infer_mode}")

    test_set = SIDDNoisyRAWDataset(
        split_txt=args.test_txt,
        train=False,
        patch_size=args.patch_size,
        n_crop_per_img=args.n_crop_per_img,
        beta_min=args.beta,
        beta_max=args.beta,
        add_row_noise=False,
        row_noise_scale=args.row_noise_scale,
        row_smooth_kernel=args.row_smooth_kernel,
    )
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = load_model(args, device)

    cam_cfg = camera_config.get(args.camera, camera_config["neutral"])
    wb = cam_cfg.get("wb", [1.0, 1.0, 1.0, 1.0])
    ccm = cam_cfg.get("ccm", None)
    if ccm is not None:
        ccm = np.array(ccm).reshape(3, 3)

    raw_metrics = MetricAccumulator()
    rgb_metrics = MetricAccumulator()
    std_ratio_am = ScalarAccumulator()
    residual_std_am = ScalarAccumulator()
    gdiff_ratio_am = ScalarAccumulator()
    grow_ratio_am = ScalarAccumulator()
    row_std_am = ScalarAccumulator()
    per_sample_rows = []
    patch_counter = 0

    with torch.no_grad():
        for scene_idx, data in enumerate(tqdm(loader, desc="test")):
            n_crop = data["noisy"].shape[1]
            scene_name = os.path.basename(test_set.scene_dirs[scene_idx])
            parts = scene_name.split("_")
            lighting = parts[-1]
            camera = parts[2] if len(parts) > 2 else "unknown"

            for crop_id in range(n_crop):
                noisy_4d = data["noisy"][:, crop_id].to(device)
                alpha_batch = data["alpha"][:, crop_id].to(device)
                sigma2_batch = data["sigma2"][:, crop_id].to(device)
                row_std_batch = data["row_std"][:, crop_id].to(device)

                denoised_4d = denoise_patch(
                    model,
                    noisy_4d,
                    alpha_batch,
                    sigma2_batch,
                    row_std_batch,
                    args,
                )

                noisy_clamped = torch.clamp(noisy_4d, 0, 1)
                denoised_clamped = torch.clamp(denoised_4d, 0, 1)
                residual = noisy_clamped - denoised_clamped

                noisy_std = noisy_clamped.std().item()
                denoised_std = denoised_clamped.std().item()
                residual_std = residual.std().item()
                std_ratio = denoised_std / max(noisy_std, 1e-12)

                gdiff_noisy = green_diff_std(noisy_clamped)
                gdiff_denoised = green_diff_std(denoised_clamped)
                gdiff_ratio = gdiff_denoised / max(gdiff_noisy, 1e-12)

                grow_noisy = green_row_std(noisy_clamped)
                grow_denoised = green_row_std(denoised_clamped)
                grow_ratio = grow_denoised / max(grow_noisy, 1e-4)

                std_ratio_am.update(std_ratio)
                residual_std_am.update(residual_std)
                gdiff_ratio_am.update(gdiff_ratio)
                grow_ratio_am.update(grow_ratio)
                row_std_am.update(row_std_batch.mean().item())

                noisy_c = noisy_clamped[0].cpu()
                denoised_c = denoised_clamped[0].cpu()
                name = f"{patch_counter:06d}.png"

                row = {
                    "idx": patch_counter,
                    "scene_idx": scene_idx,
                    "scene_name": scene_name,
                    "lighting": lighting,
                    "camera": camera,
                    "crop_id": crop_id,
                    "alpha": alpha_batch[0].item(),
                    "sigma2": sigma2_batch[0].item(),
                    "row_std": row_std_batch[0].item(),
                    "noisy_std": noisy_std,
                    "denoised_std": denoised_std,
                    "std_ratio": std_ratio,
                    "residual_std": residual_std,
                    "gdiff_noisy": gdiff_noisy,
                    "gdiff_denoised": gdiff_denoised,
                    "gdiff_ratio": gdiff_ratio,
                    "green_row_noisy": grow_noisy,
                    "green_row_denoised": grow_denoised,
                    "green_row_ratio": grow_ratio,
                }

                if "clean" in data:
                    clean_4d = data["clean"][:, crop_id].to(device)
                    clean_c = torch.clamp(clean_4d, 0, 1)[0].cpu()

                    res_raw = psnr_ssim_metric_torch(
                        denoised_c.unsqueeze(0),
                        clean_c.unsqueeze(0),
                    )
                    raw_metrics.update(res_raw["psnr"], res_raw["ssim"])
                    row["psnr_raw"] = res_raw["psnr"]
                    row["ssim_raw"] = res_raw["ssim"]

                    denoised_rgb_01 = torch.from_numpy(
                        isp_process(denoised_c, wb=wb, ccm=ccm)
                    ).float() / 255.0
                    clean_rgb_01 = torch.from_numpy(
                        isp_process(clean_c, wb=wb, ccm=ccm)
                    ).float() / 255.0
                    res_rgb = psnr_ssim_metric_torch(
                        denoised_rgb_01.permute(2, 0, 1).unsqueeze(0),
                        clean_rgb_01.permute(2, 0, 1).unsqueeze(0),
                    )
                    rgb_metrics.update(res_rgb["psnr"], res_rgb["ssim"])
                    row["psnr_rgb"] = res_rgb["psnr"]
                    row["ssim_rgb"] = res_rgb["ssim"]

                    save_img(
                        os.path.join(args.out_dir, "clean", name),
                        (clean_rgb_01.numpy() * 255).astype(np.uint8),
                    )

                per_sample_rows.append(row)

                noisy_rgb = isp_process(noisy_c, wb=wb, ccm=ccm)
                denoised_rgb = isp_process(denoised_c, wb=wb, ccm=ccm)
                save_img(os.path.join(args.out_dir, "noisy", name), noisy_rgb)
                save_img(os.path.join(args.out_dir, "denoised", name), denoised_rgb)

                if args.save_residual:
                    residual_vis = residual[0].abs().mean(dim=0).cpu().numpy()
                    residual_vis = residual_vis / max(residual_vis.max(), 1e-12)
                    residual_vis = (residual_vis * 255).astype(np.uint8)
                    save_img(
                        os.path.join(args.out_dir, "residual", name),
                        np.stack([residual_vis] * 3, axis=-1),
                    )

                patch_counter += 1

    print(f"\nTotal patches processed: {patch_counter}")

    print("\n========== Denoising Strength ==========")
    print(f"  row_std           mean={row_std_am.mean():.4e} std={row_std_am.std():.4e}")
    print(f"  std_ratio         mean={std_ratio_am.mean():.4f} std={std_ratio_am.std():.4f}")
    print(f"  residual_std      mean={residual_std_am.mean():.4e} std={residual_std_am.std():.4e}")
    print(f"  G1-G2 std ratio   mean={gdiff_ratio_am.mean():.4f} std={gdiff_ratio_am.std():.4f}")
    print(f"  row std ratio     mean={grow_ratio_am.mean():.4f} std={grow_ratio_am.std():.4f}")
    print("========================================\n")

    print("========== Evaluation Summary ==========")
    if raw_metrics.has_data():
        summary = raw_metrics.summary()
        print(f"  [RAW domain] n={summary['n']}")
        print(
            f"    PSNR mean={summary['psnr_mean']:.2f} std={summary['psnr_std']:.2f} "
            f"max={summary['psnr_max']:.2f} min={summary['psnr_min']:.2f}"
        )
        print(
            f"    SSIM mean={summary['ssim_mean']:.4f} std={summary['ssim_std']:.4f} "
            f"max={summary['ssim_max']:.4f} min={summary['ssim_min']:.4f}"
        )

        summary_rgb = rgb_metrics.summary()
        print(f"  [sRGB domain] n={summary_rgb['n']}")
        print(
            f"    PSNR mean={summary_rgb['psnr_mean']:.2f} std={summary_rgb['psnr_std']:.2f} "
            f"max={summary_rgb['psnr_max']:.2f} min={summary_rgb['psnr_min']:.2f}"
        )
        print(
            f"    SSIM mean={summary_rgb['ssim_mean']:.4f} std={summary_rgb['ssim_std']:.4f} "
            f"max={summary_rgb['ssim_max']:.4f} min={summary_rgb['ssim_min']:.4f}"
        )
    else:
        print("  No clean reference found, skipping PSNR/SSIM.")
    print("========================================\n")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    fieldnames = sorted({key for row in per_sample_rows for key in row.keys()})
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"Per-sample metrics saved to: {csv_path}")

    df = pd.DataFrame(per_sample_rows)
    if not df.empty:
        df["noise_level"] = df["alpha"] + df["sigma2"]

        print("\n========== By Noise Level ==========")
        df["noise_tier"] = pd.qcut(
            df["noise_level"],
            q=3,
            labels=["low_noise", "mid_noise", "high_noise"],
            duplicates="drop",
        )
        columns = ["std_ratio", "residual_std", "gdiff_ratio", "green_row_ratio"]
        if "psnr_raw" in df.columns:
            columns += ["psnr_raw", "psnr_rgb"]
        grouped = df.groupby("noise_tier", observed=False)[columns].agg(["mean", "std", "count"])
        print(grouped.to_string())

        print("\n========== By Lighting ==========")
        grouped_lighting = df.groupby("lighting")[columns].agg(["mean", "std", "count"])
        print(grouped_lighting.to_string())

        print("\n========== By Camera ==========")
        grouped_camera = df.groupby("camera")[columns].agg(["mean", "std", "count"])
        print(grouped_camera.to_string())

        noise_csv_path = os.path.join(args.out_dir, "noise_vs_metrics.csv")
        df.to_csv(noise_csv_path, index=False)
        print(f"\nnoise_vs_metrics.csv saved to: {noise_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/SIDD_Noisier2Noise_row_noise/best.pth")
    parser.add_argument("--camera", type=str, default="sonyzve10m2")
    parser.add_argument("--test_txt", type=str, default="my_idea/0531splits/test.txt")
    parser.add_argument("--out_dir", type=str, default="./vis_rgb_row_noise")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--n_crop_per_img", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--infer_mode", type=str, default="direct", choices=["direct", "r2r"])
    parser.add_argument("--row_noise_scale", type=float, default=1.0)
    parser.add_argument("--row_smooth_kernel", type=int, default=0)
    parser.add_argument("--save_residual", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    main(parser.parse_args())
