# """
# ConditionalDenoiser 推理脚本
# - 对每个场景的所有 crop 都做推理
# - 保存 noisy / denoised / clean 可视化图片
# - 若有 clean，计算并汇总 PSNR / SSIM 指标
# """
# import os
# import argparse
# import random
# import numpy as np
# from tqdm import tqdm
# import sys
# import torch
# from torch.utils.data import DataLoader
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from my_idea.conditional_denoiser import ConditionalNAFNet
# from my_idea.SelfSupervisedDataset import SIDDNoisyRAWDataset
# from utils.utils import psnr_ssim_metric_torch


# # ===================== ISP =====================

# def isp_process(packed, wb=None, ccm=None, gamma=2.2):
#     packed = packed.float()
#     C, H, W = packed.shape

#     if wb is not None:
#         wb = torch.tensor(wb, dtype=torch.float32).view(4, 1, 1)
#         packed = torch.clamp(packed * wb, 0, 1)

#     R   = packed[0]
#     G   = (packed[1] + packed[2]) / 2.0
#     B   = packed[3]
#     rgb = torch.stack([R, G, B], dim=0)  # (3, H, W)

#     if ccm is not None:
#         ccm      = torch.tensor(ccm, dtype=torch.float32)
#         rgb_flat = ccm @ rgb.view(3, -1)
#         rgb      = torch.clamp(rgb_flat.view(3, H, W), 0, 1)

#     rgb = torch.pow(torch.clamp(rgb, 1e-8, 1.0), 1.0 / gamma)
#     return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# def save_img(path, img):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     from PIL import Image
#     Image.fromarray(img).save(path)


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # ===================== METRICS =====================

# class MetricAccumulator:
#     def __init__(self):
#         self.psnr_list = []
#         self.ssim_list = []

#     def update(self, psnr, ssim):
#         self.psnr_list.append(psnr)
#         self.ssim_list.append(ssim)

#     def summary(self):
#         if not self.psnr_list:
#             return {}
#         p = np.array(self.psnr_list)
#         s = np.array(self.ssim_list)
#         return {
#             "n":         len(p),
#             "psnr_mean": float(p.mean()),
#             "psnr_std":  float(p.std()),
#             "psnr_max":  float(p.max()),
#             "psnr_min":  float(p.min()),
#             "ssim_mean": float(s.mean()),
#             "ssim_std":  float(s.std()),
#             "ssim_max":  float(s.max()),
#             "ssim_min":  float(s.min()),
#         }

#     def has_data(self):
#         return len(self.psnr_list) > 0


# # ===================== DENOISE ONE PATCH =====================

# def denoise_patch(model, noisy_4d, alpha_batch, sigma2_batch, beta, device):
#     """
#     单张 patch 推理，使用 Noisier2Noise 校正步骤：
#         denoised = 2·f(y+z) - (y+z)
#     """
#     var_z   = beta * (
#         alpha_batch.view(-1, 1, 1, 1) * torch.clamp(noisy_4d, min=0)
#         + sigma2_batch.view(-1, 1, 1, 1)
#     )
#     z       = torch.randn_like(noisy_4d) \
#               * torch.sqrt(torch.clamp(var_z, min=1e-10))
#     z_noisy = torch.clamp(noisy_4d + z, 0, 1)

#     f_z      = model(z_noisy, alpha_batch, sigma2_batch)
#     denoised = torch.clamp(2 * f_z - z_noisy, 0, 1)
#     return denoised


# # ===================== MAIN =====================

# def main(args):
#     camera_config = {
#         "sonyzve10m2": {"wb": [2.0, 1.0, 1.0, 1.6], "ccm": None},
#         "neutral":     {"wb": [1.0, 1.0, 1.0, 1.0], "ccm": None},
#     }

#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     # ===== dataset =====
#     test_set = SIDDNoisyRAWDataset(
#         split_txt=args.test_txt,
#         train=False,
#         patch_size=args.patch_size,
#         n_crop_per_img=args.n_crop_per_img,
#         beta=args.beta,
#     )
#     loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

#     # ===== model =====
#     model = ConditionalNAFNet(
#         img_channel=4, width=32,
#         middle_blk_num=12,
#         enc_blk_nums=[2, 2, 4, 8],
#         dec_blk_nums=[2, 2, 2, 2],
#         embed_dim=256,
#     ).to(device)

#     ckpt = torch.load(args.ckpt, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     model.eval()
#     print("Loaded checkpoint:", args.ckpt)
#     if "val_psnr" in ckpt:
#         print(f"  ckpt val_psnr={ckpt['val_psnr']:.2f}  "
#               f"val_loss={ckpt.get('val_loss', float('nan')):.4e}  "
#               f"epoch={ckpt.get('epoch', '?')}")

#     cam_cfg = camera_config.get(args.camera, camera_config["neutral"])
#     wb  = cam_cfg.get("wb", [1.0, 1.0, 1.0, 1.0])
#     ccm = cam_cfg.get("ccm", None)
#     if ccm is not None:
#         ccm = np.array(ccm).reshape(3, 3)

#     raw_metrics = MetricAccumulator()
#     rgb_metrics = MetricAccumulator()
#     per_sample_rows = []

#     patch_counter = 0   # 全局 patch 编号，用于文件命名

#     # ===== inference =====
#     with torch.no_grad():
#         for scene_idx, data in enumerate(tqdm(loader)):

#             # data["noisy"]:  (1, n_crop, 4, H, W)
#             # data["alpha"]:  (1, n_crop)
#             # data["sigma2"]: (1, n_crop)
#             n_crop = data["noisy"].shape[1]

#             for crop_id in range(n_crop):

#                 # ── 取第 crop_id 个 patch ──
#                 noisy_4d     = data["noisy"][:, crop_id].to(device)    # (1,4,H,W)
#                 alpha_batch  = data["alpha"][:, crop_id].to(device)    # (1,)
#                 sigma2_batch = data["sigma2"][:, crop_id].to(device)   # (1,)

#                 # ── 推理（Noisier2Noise 校正）──
#                 denoised_4d = denoise_patch(
#                     model, noisy_4d, alpha_batch, sigma2_batch,
#                     beta=args.beta, device=device,
#                 )

#                 noisy_c    = torch.clamp(noisy_4d,    0, 1)[0].cpu()  # (4,H,W)
#                 denoised_c = torch.clamp(denoised_4d, 0, 1)[0].cpu()

#                 name = f"{patch_counter:06d}.png"   # scene_{scene_idx}_crop_{crop_id}
#                 row  = {
#                     "idx":       patch_counter,
#                     "scene_idx": scene_idx,
#                     "crop_id":   crop_id,
#                     "alpha":     alpha_batch[0].item(),
#                     "sigma2":    sigma2_batch[0].item(),
#                 }

#                 # ── clean 指标 ──
#                 if "clean" in data:
#                     clean_4d = data["clean"][:, crop_id].to(device)
#                     clean_c  = torch.clamp(clean_4d, 0, 1)[0].cpu()

#                     res_raw = psnr_ssim_metric_torch(
#                         denoised_c.unsqueeze(0),
#                         clean_c.unsqueeze(0),
#                     )
#                     raw_metrics.update(res_raw["psnr"], res_raw["ssim"])
#                     row["psnr_raw"] = res_raw["psnr"]
#                     row["ssim_raw"] = res_raw["ssim"]

#                     denoised_rgb_01 = torch.from_numpy(
#                         isp_process(denoised_c, wb=wb, ccm=ccm)
#                     ).float() / 255.0
#                     clean_rgb_01 = torch.from_numpy(
#                         isp_process(clean_c, wb=wb, ccm=ccm)
#                     ).float() / 255.0
#                     res_rgb = psnr_ssim_metric_torch(
#                         denoised_rgb_01.permute(2, 0, 1).unsqueeze(0),
#                         clean_rgb_01.permute(2, 0, 1).unsqueeze(0),
#                     )
#                     rgb_metrics.update(res_rgb["psnr"], res_rgb["ssim"])
#                     row["psnr_rgb"] = res_rgb["psnr"]
#                     row["ssim_rgb"] = res_rgb["ssim"]

#                     save_img(
#                         os.path.join(args.out_dir, "clean", name),
#                         (clean_rgb_01.numpy() * 255).astype(np.uint8),
#                     )

#                 per_sample_rows.append(row)

#                 # ── 保存可视化图片 ──
#                 noisy_rgb    = isp_process(noisy_c,    wb=wb, ccm=ccm)
#                 denoised_rgb = isp_process(denoised_c, wb=wb, ccm=ccm)
#                 save_img(os.path.join(args.out_dir, "noisy",    name), noisy_rgb)
#                 save_img(os.path.join(args.out_dir, "denoised", name), denoised_rgb)

#                 patch_counter += 1

#     total_patches = patch_counter
#     print(f"\nTotal patches processed: {total_patches}")

#     # ===== 汇总指标 =====
#     print("\n========== Evaluation Summary ==========")
#     if raw_metrics.has_data():
#         s = raw_metrics.summary()
#         print(f"  [RAW  domain]  n={s['n']}")
#         print(f"    PSNR  mean={s['psnr_mean']:.2f}  std={s['psnr_std']:.2f}"
#               f"  max={s['psnr_max']:.2f}  min={s['psnr_min']:.2f}")
#         print(f"    SSIM  mean={s['ssim_mean']:.4f}  std={s['ssim_std']:.4f}"
#               f"  max={s['ssim_max']:.4f}  min={s['ssim_min']:.4f}")

#         s2 = rgb_metrics.summary()
#         print(f"  [sRGB domain]  n={s2['n']}")
#         print(f"    PSNR  mean={s2['psnr_mean']:.2f}  std={s2['psnr_std']:.2f}"
#               f"  max={s2['psnr_max']:.2f}  min={s2['psnr_min']:.2f}")
#         print(f"    SSIM  mean={s2['ssim_mean']:.4f}  std={s2['ssim_std']:.4f}"
#               f"  max={s2['ssim_max']:.4f}  min={s2['ssim_min']:.4f}")
#     else:
#         print("  No clean reference found, skipping metrics.")
#     print("========================================\n")

#     # ===== 保存 CSV =====
#     if per_sample_rows and "psnr_raw" in per_sample_rows[0]:
#         import csv
#         csv_path = os.path.join(args.out_dir, "metrics.csv")
#         os.makedirs(args.out_dir, exist_ok=True)
#         fieldnames = ["idx", "scene_idx", "crop_id", "alpha", "sigma2",
#                       "psnr_raw", "ssim_raw", "psnr_rgb", "ssim_rgb"]
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(per_sample_rows)
#         print(f"Per-sample metrics saved to: {csv_path}")


# # ===================== ENTRY =====================

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt",           type=str,
#                         default="checkpoints/SIDD_Noisier2Noise/best.pth")
#     parser.add_argument("--camera",         type=str,  default="sonyzve10m2")
#     parser.add_argument("--test_txt",       type=str,
#                         default="my_idea/splits/test.txt")
#     parser.add_argument("--out_dir",        type=str,  default="./vis_rgb")
#     parser.add_argument("--patch_size",     type=int,  default=256)
#     parser.add_argument("--n_crop_per_img", type=int,  default=8)
#     parser.add_argument("--beta",           type=float, default=1.2)
#     parser.add_argument("--seed",           type=int,  default=0)

#     args = parser.parse_args()
#     main(args)

#     # df = pd.read_csv("vis_rgb/metrics.csv")
#     # # 按场景聚合，看每个场景的平均 PSNR
#     # print(df.groupby("scene_idx")["psnr_raw"].mean().sort_values())



"""
ConditionalDenoiser validation MAT 推理脚本
- noisy: ValidationNoisyBlocksRaw.mat, shape (40, 32, 256, 256), range [0, 1]
- gt:    ValidationGtBlocksRaw.mat,    shape (40, 32, 256, 256), range [0, 1]
- 模型输入为 packed RAW: (4, 128, 128)
- 保存 noisy / denoised / clean 可视化图片
- 保存 denoised raw mat: shape (40, 32, 256, 256)
- 计算并汇总 RAW packed 域和 sRGB 可视化域 PSNR / SSIM
"""

import os
import csv
import sys
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from my_idea.conditional_denoiser import ConditionalNAFNet
from utils.utils import psnr_ssim_metric_torch


# ===================== MAT IO =====================

def load_mat_array(path, key=None):
    mat = loadmat(path)

    if key is not None:
        if key not in mat:
            raise KeyError(f"{key} not found in {path}. Available keys: {list(mat.keys())}")
        arr = mat[key]
    else:
        candidates = [
            v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 4
        ]
        if not candidates:
            raise ValueError(f"No 4D array found in {path}. Available keys: {list(mat.keys())}")
        arr = candidates[0]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (40, 32, 256, 256):
        raise ValueError(f"Expected shape (40, 32, 256, 256), got {arr.shape} from {path}")

    return np.clip(arr, 0.0, 1.0)


# ===================== RAW PACK / UNPACK =====================

def pack_raw_block(raw_2d):
    """
    raw_2d: (H, W)
    return: (4, H/2, W/2), channel order [R, G1, G2, B] under RGGB assumption
    """
    return np.stack([
        raw_2d[0::2, 0::2],
        raw_2d[0::2, 1::2],
        raw_2d[1::2, 0::2],
        raw_2d[1::2, 1::2],
    ], axis=0).astype(np.float32)


def unpack_raw_block(packed):
    """
    packed: (4, H, W)
    return: (2H, 2W)
    """
    c, h, w = packed.shape
    assert c == 4

    raw = np.zeros((h * 2, w * 2), dtype=np.float32)
    raw[0::2, 0::2] = packed[0]
    raw[0::2, 1::2] = packed[1]
    raw[1::2, 0::2] = packed[2]
    raw[1::2, 1::2] = packed[3]
    return np.clip(raw, 0.0, 1.0)


# ===================== ISP =====================

def isp_process(packed, wb=None, ccm=None, gamma=2.2):
    packed = packed.float()
    _, h, w = packed.shape

    if wb is not None:
        wb = torch.tensor(wb, dtype=torch.float32).view(4, 1, 1)
        packed = torch.clamp(packed * wb, 0, 1)

    r = packed[0]
    g = (packed[1] + packed[2]) / 2.0
    b = packed[3]
    rgb = torch.stack([r, g, b], dim=0)

    if ccm is not None:
        ccm = torch.tensor(ccm, dtype=torch.float32)
        rgb_flat = ccm @ rgb.view(3, -1)
        rgb = torch.clamp(rgb_flat.view(3, h, w), 0, 1)

    rgb = torch.pow(torch.clamp(rgb, 1e-8, 1.0), 1.0 / gamma)
    return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image
    Image.fromarray(img).save(path)


# ===================== UTILS =====================

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
        p = np.array(self.psnr_list)
        s = np.array(self.ssim_list)
        return {
            "n": len(p),
            "psnr_mean": float(p.mean()),
            "psnr_std": float(p.std()),
            "psnr_max": float(p.max()),
            "psnr_min": float(p.min()),
            "ssim_mean": float(s.mean()),
            "ssim_std": float(s.std()),
            "ssim_max": float(s.max()),
            "ssim_min": float(s.min()),
        }


def estimate_noise_params(noisy, clean):
    """
    用 noisy-clean 估计 heteroscedastic noise:
        Var(noise) ~= alpha * clean + sigma2
    noisy/clean: torch tensor, shape (1, 4, H, W)
    """
    with torch.no_grad():
        x = torch.clamp(clean, 0, 1).reshape(-1)
        y = ((noisy - clean) ** 2).reshape(-1)

        x_mean = x.mean()
        y_mean = y.mean()
        var_x = torch.mean((x - x_mean) ** 2)

        if var_x.item() < 1e-12:
            alpha = torch.tensor(0.0, device=noisy.device)
        else:
            alpha = torch.mean((x - x_mean) * (y - y_mean)) / var_x
            alpha = torch.clamp(alpha, min=0.0)

        sigma2 = torch.clamp(y_mean - alpha * x_mean, min=1e-10)

    return alpha.view(1), sigma2.view(1)


# ===================== DENOISE =====================

def denoise_patch(model, noisy_4d, alpha_batch, sigma2_batch, beta):
    var_z = beta * (
        alpha_batch.view(-1, 1, 1, 1) * torch.clamp(noisy_4d, min=0)
        + sigma2_batch.view(-1, 1, 1, 1)
    )

    z = torch.randn_like(noisy_4d) * torch.sqrt(torch.clamp(var_z, min=1e-10))
    z_noisy = torch.clamp(noisy_4d + z, 0, 1)

    f_z = model(z_noisy, alpha_batch, sigma2_batch)
    denoised = torch.clamp(2 * f_z - z_noisy, 0, 1)
    return denoised


# ===================== MAIN =====================

def main(args):
    camera_config = {
        "sonyzve10m2": {"wb": [2.0, 1.0, 1.0, 1.6], "ccm": None},
        "neutral": {"wb": [1.0, 1.0, 1.0, 1.0], "ccm": None},
    }

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    noisy_blocks = load_mat_array(args.noisy_mat, args.noisy_key)
    gt_blocks = load_mat_array(args.gt_mat, args.gt_key)

    print("Loaded noisy:", noisy_blocks.shape, noisy_blocks.min(), noisy_blocks.max())
    print("Loaded gt:   ", gt_blocks.shape, gt_blocks.min(), gt_blocks.max())

    model = ConditionalNAFNet(
        img_channel=4,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        embed_dim=256,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Loaded checkpoint:", args.ckpt)
    if "val_psnr" in ckpt:
        print(
            f"  ckpt val_psnr={ckpt['val_psnr']:.2f}  "
            f"val_loss={ckpt.get('val_loss', float('nan')):.4e}  "
            f"epoch={ckpt.get('epoch', '?')}"
        )

    cam_cfg = camera_config.get(args.camera, camera_config["neutral"])
    wb = cam_cfg.get("wb", [1.0, 1.0, 1.0, 1.0])
    ccm = cam_cfg.get("ccm", None)
    if ccm is not None:
        ccm = np.array(ccm).reshape(3, 3)

    raw_metrics = MetricAccumulator()
    rgb_metrics = MetricAccumulator()
    per_sample_rows = []

    denoised_blocks = np.zeros_like(noisy_blocks, dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)

    patch_counter = 0

    with torch.no_grad():
        for scene_idx in tqdm(range(noisy_blocks.shape[0])):
            for block_idx in range(noisy_blocks.shape[1]):
                noisy_packed = pack_raw_block(noisy_blocks[scene_idx, block_idx])
                clean_packed = pack_raw_block(gt_blocks[scene_idx, block_idx])

                noisy_4d = torch.from_numpy(noisy_packed).unsqueeze(0).to(device)
                clean_4d = torch.from_numpy(clean_packed).unsqueeze(0).to(device)

                if args.alpha is None or args.sigma2 is None:
                    alpha_batch, sigma2_batch = estimate_noise_params(noisy_4d, clean_4d)
                else:
                    alpha_batch = torch.tensor([args.alpha], dtype=torch.float32, device=device)
                    sigma2_batch = torch.tensor([args.sigma2], dtype=torch.float32, device=device)

                denoised_4d = denoise_patch(
                    model=model,
                    noisy_4d=noisy_4d,
                    alpha_batch=alpha_batch,
                    sigma2_batch=sigma2_batch,
                    beta=args.beta,
                )

                noisy_c = torch.clamp(noisy_4d, 0, 1)[0].cpu()
                clean_c = torch.clamp(clean_4d, 0, 1)[0].cpu()
                denoised_c = torch.clamp(denoised_4d, 0, 1)[0].cpu()

                denoised_blocks[scene_idx, block_idx] = unpack_raw_block(denoised_c.numpy())

                name = f"scene_{scene_idx:02d}_block_{block_idx:02d}.png"

                res_raw = psnr_ssim_metric_torch(
                    denoised_c.unsqueeze(0),
                    clean_c.unsqueeze(0),
                )
                raw_metrics.update(res_raw["psnr"], res_raw["ssim"])

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

                per_sample_rows.append({
                    "idx": patch_counter,
                    "scene_idx": scene_idx,
                    "block_idx": block_idx,
                    "alpha": float(alpha_batch[0].item()),
                    "sigma2": float(sigma2_batch[0].item()),
                    "psnr_raw": res_raw["psnr"],
                    "ssim_raw": res_raw["ssim"],
                    "psnr_rgb": res_rgb["psnr"],
                    "ssim_rgb": res_rgb["ssim"],
                })

                noisy_rgb = isp_process(noisy_c, wb=wb, ccm=ccm)
                denoised_rgb = isp_process(denoised_c, wb=wb, ccm=ccm)
                clean_rgb = isp_process(clean_c, wb=wb, ccm=ccm)

                save_img(os.path.join(args.out_dir, "noisy", name), noisy_rgb)
                save_img(os.path.join(args.out_dir, "denoised", name), denoised_rgb)
                save_img(os.path.join(args.out_dir, "clean", name), clean_rgb)

                patch_counter += 1

    denoised_mat_path = os.path.join(args.out_dir, "ValidationDenoisedBlocksRaw.mat")
    savemat(denoised_mat_path, {"ValidationDenoisedBlocksRaw": denoised_blocks})
    print(f"\nDenoised MAT saved to: {denoised_mat_path}")
    print(f"Total blocks processed: {patch_counter}")

    print("\n========== Evaluation Summary ==========")

    s = raw_metrics.summary()
    print(f"  [RAW packed domain]  n={s['n']}")
    print(
        f"    PSNR  mean={s['psnr_mean']:.2f}  std={s['psnr_std']:.2f}  "
        f"max={s['psnr_max']:.2f}  min={s['psnr_min']:.2f}"
    )
    print(
        f"    SSIM  mean={s['ssim_mean']:.4f}  std={s['ssim_std']:.4f}  "
        f"max={s['ssim_max']:.4f}  min={s['ssim_min']:.4f}"
    )

    s2 = rgb_metrics.summary()
    print(f"  [sRGB domain]  n={s2['n']}")
    print(
        f"    PSNR  mean={s2['psnr_mean']:.2f}  std={s2['psnr_std']:.2f}  "
        f"max={s2['psnr_max']:.2f}  min={s2['psnr_min']:.2f}"
    )
    print(
        f"    SSIM  mean={s2['ssim_mean']:.4f}  std={s2['ssim_std']:.4f}  "
        f"max={s2['ssim_max']:.4f}  min={s2['ssim_min']:.4f}"
    )

    print("========================================\n")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    fieldnames = [
        "idx", "scene_idx", "block_idx", "alpha", "sigma2",
        "psnr_raw", "ssim_raw", "psnr_rgb", "ssim_rgb",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(f"Per-block metrics saved to: {csv_path}")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print("\nPer-scene RAW PSNR:")
        print(df.groupby("scene_idx")["psnr_raw"].mean().sort_values())
    except ImportError:
        pass


# ===================== ENTRY =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="checkpoints/SIDD_Noisier2Noise0531/best.pth")
    parser.add_argument("--noisy_mat", type=str, default="../../data/xml196414/SIDD/SIDD_Medium/ValidationNoisyBlocksRaw.mat")
    parser.add_argument("--gt_mat", type=str, default="../../data/xml196414/SIDD/SIDD_Medium/ValidationGtBlocksRaw.mat")
    parser.add_argument("--noisy_key", type=str, default=None)
    parser.add_argument("--gt_key", type=str, default=None)

    parser.add_argument("--camera", type=str, default="sonyzve10m2")
    parser.add_argument("--out_dir", type=str, default="./vis_validation_rgb")

    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--sigma2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)


# import pandas as pd

# df = pd.read_csv("vis_validation_rgb/metrics.csv")

# # 最差 10 张
# print("Worst RAW PSNR:")
# print(df.sort_values("psnr_raw").head(10)[
#     ["idx", "scene_idx", "block_idx", "psnr_raw", "ssim_raw", "psnr_rgb", "ssim_rgb"]
# ])

# # 最好 10 张
# print("Best RAW PSNR:")
# print(df.sort_values("psnr_raw", ascending=False).head(10)[
#     ["idx", "scene_idx", "block_idx", "psnr_raw", "ssim_raw", "psnr_rgb", "ssim_rgb"]
# ])