# import os
# import glob
# import torch
# import argparse
# import numpy as np
# from tqdm import tqdm

# from models.ELD_models import UNetSeeInDark
# from utils.utils import psnr_ssim_metric_torch


# # =========================================================
# # collect pairs
# # =========================================================

# def collect_pairs(data_dir):

#     noisy_paths = sorted(
#         glob.glob(
#             os.path.join(
#                 data_dir,
#                 "noisy_*.npy"
#             )
#         )
#     )

#     pairs = []

#     for noisy_path in noisy_paths:

#         noisy_name = os.path.basename(
#             noisy_path
#         )

#         clean_name = noisy_name.replace(
#             "noisy_",
#             "clean_"
#         )

#         clean_path = os.path.join(
#             data_dir,
#             clean_name
#         )

#         if not os.path.exists(clean_path):

#             print(
#                 f"[Skip] missing clean file: "
#                 f"{clean_name}"
#             )

#             continue

#         pairs.append(
#             (noisy_path, clean_path)
#         )

#     return pairs


# # =========================================================
# # load npy
# # =========================================================

# def load_raw(path):

#     raw = np.load(path).astype(np.float32)

#     # support:
#     # [4,H,W]
#     # [H,W,4]

#     if raw.ndim != 3:

#         raise ValueError(
#             f"Invalid shape: {raw.shape}"
#         )

#     if raw.shape[-1] == 4:

#         raw = np.transpose(
#             raw,
#             (2,0,1)
#         )

#     elif raw.shape[0] == 4:

#         pass

#     else:

#         raise ValueError(
#             f"Invalid RAW shape: {raw.shape}"
#         )

#     return raw


# # =========================================================
# # evaluate
# # =========================================================

# @torch.no_grad()
# def evaluate(model, pairs, device):

#     model.eval()

#     psnr_list = []
#     ssim_list = []

#     for noisy_path, clean_path in tqdm(pairs):

#         # =================================================
#         # load
#         # =================================================

#         noisy = load_raw(noisy_path)
#         clean = load_raw(clean_path)

#         noisy = torch.from_numpy(
#             noisy
#         ).unsqueeze(0).to(device)

#         clean = torch.from_numpy(
#             clean
#         ).unsqueeze(0).to(device)

#         # =================================================
#         # inference
#         # =================================================
#         # 在推理脚本里，模型 forward 之前加
#         # print(f"[INFER] noisy min={noisy.min():.4f} max={noisy.max():.4f} mean={noisy.mean():.4f} std={noisy.std():.4f}")
#         # print(f"[INFER] noisy shape={noisy.shape}")
#         denoised = model(noisy)

#         denoised = torch.clamp(
#             denoised,
#             0,
#             1
#         )

#         clean = torch.clamp(
#             clean,
#             0,
#             1
#         )

#         # =================================================
#         # metrics
#         # =================================================

#         res = psnr_ssim_metric_torch(
#             denoised,
#             clean
#         )

#         psnr = res["psnr"]
#         ssim = res["ssim"]

#         psnr_list.append(psnr)
#         ssim_list.append(ssim)

#         print(
#             f"{os.path.basename(noisy_path):35s} | "
#             f"PSNR: {psnr:.2f} | "
#             f"SSIM: {ssim:.4f}"
#         )

#     print("\n===================================")

#     print(
#         f"Average PSNR : "
#         f"{np.mean(psnr_list):.4f}"
#     )

#     print(
#         f"Average SSIM : "
#         f"{np.mean(ssim_list):.6f}"
#     )

#     print("===================================\n")


# # =========================================================
# # main
# # =========================================================

# def main(args):

#     device = torch.device(
#         "cuda" if torch.cuda.is_available()
#         else "cpu"
#     )

#     print("Using device:", device)

#     # =====================================================
#     # model
#     # =====================================================

#     model = UNetSeeInDark().to(device)

#     print("Loading checkpoint:")
#     print(args.ckpt)

#     ckpt = torch.load(
#         args.ckpt,
#         map_location=device
#     )

#     if "model" in ckpt:

#         model.load_state_dict(
#             ckpt["model"]
#         )

#     else:

#         model.load_state_dict(ckpt)

#     # =====================================================
#     # collect data
#     # =====================================================

#     pairs = collect_pairs(
#         args.data_dir
#     )

#     print(f"Found {len(pairs)} pairs")

#     if len(pairs) == 0:

#         print("No valid pairs found.")
#         return

#     # =====================================================
#     # evaluate
#     # =====================================================

#     evaluate(
#         model,
#         pairs,
#         device
#     )


# # =========================================================
# # entry
# # =========================================================

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         default="./checkpoints/PNNP_ELD/sonyzve10m2/best.pth"
#     )

#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default="./synthetic_noisy"
#     )

#     args = parser.parse_args()

#     main(args)



# # # 1. 检查数据值域
# # noisy = np.load("synthetic_noisy/noisy_000_iso6400.npy")
# # print(f"noisy min={noisy.min():.4f} max={noisy.max():.4f} mean={noisy.mean():.4f}")

# # clean = np.load("synthetic_noisy/clean_000_iso6400.npy")
# # print(f"clean shape={clean.shape}")
# # print(f"clean min={clean.min():.4f} max={clean.max():.4f} mean={clean.mean():.4f}")

# # # 2. 检查ckpt的key
# # ckpt = torch.load("./checkpoints/PNNP_ELD/sonyzve10m2/best.pth", map_location="cpu")
# # print(ckpt.keys())

# # device = torch.device(
# #     "cuda" if torch.cuda.is_available()
# #     else "cpu"
# # )

# # model = UNetSeeInDark().to(device)
# # model.load_state_dict(ckpt["model"])
# # # 3. 检查模型输出
# # model.eval()
# # with torch.no_grad():
# #     out = model(torch.from_numpy(noisy).unsqueeze(0).to(device))
# #     print(f"output min={out.min():.4f} max={out.max():.4f}")


import os
import argparse
import random
import imageio
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.real_eval_dataset import PairedEvalDataset
from models.ELD_models import UNetSeeInDark
from utils.utils import *

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def build_model(args):
    model = UNetSeeInDark().to(args.device)
    model.load_state_dict(
        torch.load(args.checkpoint_dir, map_location="cpu", weights_only=False)["model"],
        strict=True
    )
    model.eval()
    return model


@torch.no_grad()
def infer_and_save_for_submission(model, args):
    if os.path.exists(args.save_dir):
        import shutil
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    with open(args.camera_config_dir, "r") as f:
        cam_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 全局指标
    all_psnr = []
    all_ssim = []

    for cam_model in args.camera_models:
        os.makedirs(os.path.join(args.save_dir, cam_model))

        eval_set = PairedEvalDataset(
            benchmark_dir=args.benchmark_dir,
            camera_model=cam_model,
            camera_config=cam_cfg[cam_model],
            inp_clip_low=False,
            inp_clip_high=True,
            iso_list=[800, 1600, 3200],
            load_gt=True,   # 改为 True 加载 GT
        )
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)

        cam_psnr = []
        cam_ssim = []

        print(f"\n>>> Camera: {cam_model}")

        for _, data in enumerate(tqdm(eval_loader)):
            noisy    = data["noisy"].to(args.device)
            img_name = data["img_name"][0]
            denoised = model(noisy)   # [1, 4, H, W]

            # ── 指标计算（在保存前，float域）────────────────────
            if "clean" in data:
                clean_gt = data["clean"].to(args.device)   # [1, 4, H, W]

                denoised_clamped = torch.clamp(denoised, 0, 1)
                clean_clamped    = torch.clamp(clean_gt, 0, 1)

                res  = psnr_ssim_metric_torch(denoised_clamped, clean_clamped)
                psnr = res["psnr"]
                ssim = res["ssim"]

                cam_psnr.append(psnr)
                cam_ssim.append(ssim)
                all_psnr.append(psnr)
                all_ssim.append(ssim)

                tqdm.write(f"  {img_name:30s} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

            # ── 保存 ────────────────────────────────────────────
            denoised_np = denoised.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            denoised_np = np.clip(denoised_np, 0, 1)
            denoised_np = center_crop_numpy_img(denoised_np, args.eval_crop_size)
            denoised_np = np.uint16(denoised_np * 65535)
            np.save(os.path.join(args.save_dir, cam_model, f"{img_name}.npy"), denoised_np)

        # ── 每个相机的平均指标 ───────────────────────────────────
        if cam_psnr:
            print(f"\n  [{cam_model}] Avg PSNR: {np.mean(cam_psnr):.4f} | Avg SSIM: {np.mean(cam_ssim):.6f}")

    # ── 全局平均指标 ─────────────────────────────────────────────
    if all_psnr:
        print("\n" + "="*50)
        print(f"Overall Avg PSNR : {np.mean(all_psnr):.4f}")
        print(f"Overall Avg SSIM : {np.mean(all_ssim):.6f}")
        print("="*50)


def main(args):
    print(f"Will process cameras: {args.camera_models}")
    model = build_model(args).to(args.device)
    infer_and_save_for_submission(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_config_dir", type=str, default="./datasets/camera_config.yaml")
    parser.add_argument("--benchmark_dir", type=str, default="../../data/xml196414/SID/dev_phase_release")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/PNNP_ELD/sonyzve10m2/best.pth")
    parser.add_argument("--device", type=str, default="cuda")

    ## DO NOT change below setups
    parser.add_argument("--save_dir", type=str, default="./saved_res_for_submission")
    parser.add_argument("--camera_models", type=str, nargs="+", default=["canon70d", "sonya6700", "sonya7r4", "sonyzve10m2"])
    parser.add_argument("--eval_crop_size", type=int, default=512)
    _args = parser.parse_args() 

    main(_args)