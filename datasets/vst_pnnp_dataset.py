import os
import glob
import rawpy
import random
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from train_syn_pnnp import PPMGenerator

SIGMA_LIST = {
    800: [0.00037831, 0.00037627, 0.00038227, 0.00035283],
    1250:[0.00055401, 0.00055769, 0.00056314, 0.00051816],
    1600:[0.00068649, 0.00068999, 0.00069709, 0.00064069],
    3200:[0.00130139, 0.00130641, 0.00132486, 0.00121729],
    6400:[0.00248537, 0.00249739, 0.00253268, 0.0023103 ]
}

ALPHA_LIST = {
    800: [3.77602386, 4.30969631, 4.28895058, 3.77481009],
    1250:[6.07181915, 7.10341625, 7.09160789, 6.13173138],
    1600:[7.33922255, 7.68992223, 7.66910537, 6.49189788],
    3200:[15.22261798, 17.16000776, 17.17242262, 14.76595408],
    6400:[29.28177504, 31.92062629, 31.92748817, 27.11866556]

}



class VSTSynthesisDataset(Dataset):

    def __init__(
        self,
        clean_raw_dir,
        benchmark_dir,
        model_dir,
        camera_config,
        iso_list=[800, 1250, 1600, 3200, 6400],
        dgain_range=(10, 200),
        patch_size=512,
        inp_clip_low=False,
        inp_clip_high=True,
        n_crop_per_img=8,
    ):
        # ── 直接搜 ARW，和 SynthTrainDataset 保持一致 ──
        self.clean_paths = sorted(
            glob.glob(os.path.join(clean_raw_dir, "*.ARW"))
        )
        print("Found clean ARW:", len(self.clean_paths))

        self.iso_list       = iso_list
        self.dgain_range    = dgain_range
        self.patch_size     = patch_size
        self.clip_low       = 0   if inp_clip_low  else float("-inf")
        self.clip_high      = 1   if inp_clip_high else float("inf")
        self.n_crop_per_img = n_crop_per_img
        self.camera_config  = camera_config

        self.transforms = transforms.Compose([
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
        ])

        # =====================================================
        # Load shading maps（ADU域，和 SynthTrainDataset 一致）
        # =====================================================
        self.shadings = {}
        calib_path = os.path.join(benchmark_dir, "calib_res")

        for f in glob.glob(os.path.join(calib_path, "dark_shading_iso*.npy")):
            iso_val = int(
                os.path.basename(f).split("iso")[-1].split(".")[0]
            )
            self.shadings[iso_val] = np.load(f).astype(np.float32)  # [H, W]，ADU

        # =====================================================
        # Band noise calibration
        # =====================================================
        self.band_params = self._calibrate_band_noise(benchmark_dir)

        # =====================================================
        # Load per-ISO PPMGenerator models
        # =====================================================
        self.models = {}
        print("\nLoading per-ISO PPMGenerators")
        print("--------------------------------")
        for iso_val in self.iso_list:
            ckpt_path = model_dir[iso_val]
            print(f"Loading: {ckpt_path}")
            model = PPMGenerator(in_channels=4, nf=16)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.eval()
            self.models[iso_val] = model
        print("All ISO models loaded.")

    # =========================================================
    # pack_raw：和 SynthTrainDataset 完全一致
    # =========================================================
    def pack_raw(self, img, wl, bl, norm=False, clip=False):
        out = np.stack([
            img[0::2, 0::2],
            img[0::2, 1::2],
            img[1::2, 0::2],
            img[1::2, 1::2],
        ], axis=-1)                          # [H/2, W/2, 4]
        out = (out - bl) / (wl - bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def random_crop(self, img, psize, n_crop=1):
        """img: [H, W, C] numpy"""
        res = []
        for _ in range(n_crop):
            hs = np.random.randint(0, img.shape[0] - psize + 1)
            ws = np.random.randint(0, img.shape[1] - psize + 1)
            res.append(img[hs:hs + psize, ws:ws + psize, :])
        return np.stack(res, axis=0)         # [n, psize, psize, C]

    # =========================================================
    # Band noise calibration（从 dark_frame_npz 读取）
    # =========================================================
    def _calibrate_band_noise(self, benchmark_dir):

        dark_paths = glob.glob(
            os.path.join(benchmark_dir, "dark_frame_npz/**/*.npz"),
            recursive=True
        )

        iso_to_row, iso_to_col, iso_to_pixel = {}, {}, {}

        for path in dark_paths:
            data    = np.load(path)
            raw_adu = data["raw"].astype(np.float32)   # ADU, full sensor
            iso_val = self._get_iso(path)

            # 找最近的 shading iso
            target_iso       = min(self.shadings.keys(), key=lambda x: abs(x - iso_val))
            shading_full_adu = self.shadings[target_iso]   # [H_full, W_full]

            # 元信息
            y1 = (int(data["top_margin"])  // 2) * 2
            x1 = (int(data["left_margin"]) // 2) * 2
            h  = int(data["height"])
            w  = int(data["width"])

            bl = float(data["black_level"])
            wl = float(data["white_level"])

            # 裁剪到有效区域，减 bl 和 shading，归一化
            raw_roi     = raw_adu[y1:y1 + h, x1:x1 + w]
            shading_roi = shading_full_adu[y1:y1 + h, x1:x1 + w]
            noise       = (raw_roi - bl - shading_roi) / (wl - bl + 1e-8)

            # pack 成 [4, H/2, W/2]
            noise_packed = np.stack([
                noise[0::2, 0::2],
                noise[0::2, 1::2],
                noise[1::2, 0::2],
                noise[1::2, 1::2],
            ], axis=0)
            noise_t = torch.from_numpy(noise_packed)   # [4, H/2, W/2]

            row_band      = noise_t.mean(dim=-1, keepdim=True)
            row_band_zero = row_band - row_band.mean(dim=-2, keepdim=True)
            sigma_row     = row_band_zero.std().item()

            residual      = noise_t - row_band
            col_band      = residual.mean(dim=-2, keepdim=True)
            col_band_zero = col_band - col_band.mean(dim=-1, keepdim=True)
            sigma_col     = col_band_zero.std().item()

            pixel_noise   = residual - col_band
            sigma_pixel   = pixel_noise.std().item()

            iso_to_row.setdefault(iso_val, []).append(sigma_row)
            iso_to_col.setdefault(iso_val, []).append(sigma_col)
            iso_to_pixel.setdefault(iso_val, []).append(sigma_pixel)

        band_params = {}
        print("\nBand Noise Calibration")
        print("----------------------")
        for iso_val in sorted(iso_to_row.keys()):
            r = float(np.mean(iso_to_row[iso_val]))
            c = float(np.mean(iso_to_col[iso_val]))
            p = float(np.mean(iso_to_pixel[iso_val]))
            band_params[iso_val] = {"row": r, "col": c, "pixel": p}
            print(f"ISO {iso_val:5d} | row={r:.6e} | col={c:.6e} | pixel={p:.6e}")

        return band_params

    def _get_iso(self, path):
        for p in path.split(os.sep):
            if "iso" in p.lower():
                return int("".join(filter(str.isdigit, p)))
        return 800

    # =========================================================
    # Length
    # =========================================================
    def __len__(self):
        return len(self.clean_paths) * len(self.iso_list)

    # =========================================================
    # Main pipeline
    # =========================================================
    @torch.no_grad()
    def __getitem__(self, idx):
        img_idx = idx % len(self.clean_paths)
        iso     = self.iso_list[idx // len(self.clean_paths)]
        # print(f"idx={idx}, img_idx={img_idx}, len={len(self.clean_paths)}")

        ppm_model = self.models[iso] 
        
        # ── STEP 1: 读 ARW，用图像自身的 bl/wl 归一化──
        raw_file = rawpy.imread(self.clean_paths[img_idx])
        wl = float(raw_file.white_level)
        bl = float(np.mean(raw_file.black_level_per_channel))

        raw_np = np.array(raw_file.raw_image_visible).astype(np.float32)
        clean  = self.pack_raw(raw_np, wl=wl, bl=bl, norm=True, clip=True)  # [H/2, W/2, 4]


        alpha = np.array(ALPHA_LIST[iso], dtype=np.float32)
        sigma = np.array(SIGMA_LIST[iso], dtype=np.float32)

        # alpha = alpha / (wl - bl)

        alpha = torch.tensor(alpha, dtype=torch.float32).unsqueeze(0)
        alpha = alpha.repeat(self.n_crop_per_img, 1)

        sigma = torch.tensor(sigma, dtype=torch.float32).unsqueeze(0)
        sigma = sigma.repeat(self.n_crop_per_img, 1)


        # ── crop + augment ──
        clean_crops = self.random_crop(clean, psize=self.patch_size, n_crop=self.n_crop_per_img)
        clean_crops = torch.FloatTensor(clean_crops).permute(0, 3, 1, 2)   # [n, 4, H, W]
        clean_crops = self.transforms(clean_crops)

        # ── band noise sigma（找最近 ISO）──
        nearest_iso = min(self.band_params.keys(), key=lambda x: abs(x - iso))
        b_sigma_row = self.band_params[nearest_iso]["row"]
        b_sigma_col = self.band_params[nearest_iso]["col"]

        # ── STEP 2: 每个 crop 加噪 ──
        all_noisy, all_dgain = [], []

        for i in range(self.n_crop_per_img):

            clean_crop = clean_crops[i]                  # [4, H, W]，[0,1]

            dgain = float(np.random.randint(*self.dgain_range))

            # 信号缩小 dgain 倍，模拟短曝光（ADU域）
            img_adu = clean_crop * (wl - bl) / dgain

            # PPM 生成 pixel-wise noise
            n1 = torch.randn_like(clean_crop).unsqueeze(0)
            n2 = torch.randn_like(clean_crop).unsqueeze(0)
            iso_tensor = torch.tensor([[float(iso)]], dtype=torch.float32)

            gen_noise     = ppm_model(n1, n2, iso_tensor).squeeze(0)
            gen_noise_adu = gen_noise * (wl - bl)

            # Band noise（归一化域 sigma → ADU）
            row_noise = torch.randn(clean_crop.shape[0], clean_crop.shape[1], 1) * b_sigma_row
            col_noise = torch.randn(clean_crop.shape[0], 1, clean_crop.shape[2]) * b_sigma_col
            band_noise_adu = (row_noise + col_noise) * (wl - bl)

            # 合成 noisy，归一化回 [0,1] 并乘 dgain 还原亮度
            noisy_adu = img_adu + gen_noise_adu + band_noise_adu
            noisy     = noisy_adu / (wl - bl) * dgain

            all_noisy.append(noisy.cpu())
            all_dgain.append(dgain)

        return {
            "cam_model": self.camera_config,
            "iso":   torch.ones((1,)) * iso,
            "dgain": torch.tensor(all_dgain, dtype=torch.float32),
            "noisy": torch.clamp(torch.stack(all_noisy), self.clip_low, self.clip_high),
            "clean": torch.clamp(clean_crops, 0, 1),
            "alpha": alpha,           
            "sigma": sigma,  
            
        }