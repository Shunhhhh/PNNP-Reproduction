import os
import glob
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset

from my_idea.tools import estimate_noise_params


def augment(x, mode):
    if mode == 0:
        return x
    elif mode == 1:
        return torch.flip(x, [1])
    elif mode == 2:
        return torch.flip(x, [2])
    elif mode == 3:
        return torch.rot90(x, 1, [1, 2])
    elif mode == 4:
        return torch.rot90(x, 2, [1, 2])
    elif mode == 5:
        return torch.rot90(x, 3, [1, 2])
    elif mode == 6:
        return torch.flip(torch.rot90(x, 1, [1, 2]), [1])
    elif mode == 7:
        return torch.flip(torch.rot90(x, 1, [1, 2]), [2])


class SIDDNoisyRAWDataset(Dataset):

    def __init__(
        self,
        split_txt=None,
        patch_size=256,
        n_crop_per_img=4,
        train=True,
        inp_clip_low=False,
        inp_clip_high=True,
        sanity_check_n=10,
        r2_threshold=0.3,
        alpha_min=1e-6,
        alpha_max=5e-2,
        sigma2_min=1e-10,
        sigma2_max=5e-3,
        sat_threshold=0.85,
        dark_threshold=0.05,
        beta_min=0.5,
        beta_max=2.0,
        noise_low=0.001,
        noise_high=0.015,
        add_row_noise=True,
        row_noise_scale=1.0,
        row_smooth_kernel=0,
        debug_r2=False,
    ):

        if split_txt is not None:
            if not os.path.exists(split_txt):
                raise FileNotFoundError(f"split_txt 不存在: {split_txt}")

            with open(split_txt) as f:
                self.scene_dirs = [line.strip() for line in f if line.strip()]

        else:
            raise ValueError("split_txt 必须提供")

        self.patch_size = patch_size
        self.n_crop_per_img = n_crop_per_img
        self.train = train

        self.clip_low = 0 if inp_clip_low else float("-inf")
        self.clip_high = 1 if inp_clip_high else float("inf")

        self.r2_threshold = r2_threshold

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.sigma2_min = sigma2_min
        self.sigma2_max = sigma2_max

        self.beta_min = beta_min
        self.beta_max = beta_max

        self.noise_low = noise_low
        self.noise_high = noise_high

        self.sat_threshold = sat_threshold
        self.dark_threshold = dark_threshold

        self.add_row_noise = add_row_noise
        self.row_noise_scale = row_noise_scale
        self.row_smooth_kernel = row_smooth_kernel

        self.debug_r2 = debug_r2

        self.CAMERA_BAYER = {
            "GP": "bggr",
            "IP": "rggb",
            "S6": "grbg",
            "N6": "bggr",
            "G4": "bggr",
        }

        print("Pre-computing global noise params per scene...")

        self.global_alpha, self.global_sigma2 = (
            self._precompute_global_params()
        )

    def _read_mat(self, mat_path):

        with h5py.File(mat_path, "r") as f:
            return np.array(f["x"], dtype=np.float32)

    def _parse_iso(self, scene_dir):

        parts = os.path.basename(scene_dir).split("_")

        try:
            return int(parts[3])
        except:
            return 1600

    def _parse_camera(self, scene_dir):

        name = os.path.basename(scene_dir)
        parts = name.split("_")

        for part in parts:
            if part in self.CAMERA_BAYER:
                return part

        raise ValueError(f"无法解析相机型号: {name}")

    def _get_bayer_pattern(self, scene_dir):

        camera = self._parse_camera(scene_dir)
        return self.CAMERA_BAYER[camera]

    def _find_mat_files(self, scene_dir):

        noisy_files = sorted(
            glob.glob(os.path.join(scene_dir, "*NOISY*.MAT"))
        )

        gt_files = sorted(
            glob.glob(os.path.join(scene_dir, "*GT*.MAT"))
        )

        noisy_path = noisy_files[0]
        gt_path = gt_files[0] if gt_files else None

        return noisy_path, gt_path

    def pack_raw(self, img, pattern):

        pattern = pattern.lower()

        if pattern == "rggb":
            R = img[0::2, 0::2]
            Gr = img[0::2, 1::2]
            Gb = img[1::2, 0::2]
            B = img[1::2, 1::2]

        elif pattern == "bggr":
            B = img[0::2, 0::2]
            Gb = img[0::2, 1::2]
            Gr = img[1::2, 0::2]
            R = img[1::2, 1::2]

        elif pattern == "grbg":
            Gr = img[0::2, 0::2]
            R = img[0::2, 1::2]
            B = img[1::2, 0::2]
            Gb = img[1::2, 1::2]

        else:
            raise ValueError(pattern)

        return np.stack([R, Gr, Gb, B], axis=-1).astype(np.float32)

    # def _remove_row_noise_and_pack(self, raw, pattern):

    #     row_profile_packed, raw_detrended, row_meta = (
    #         estimate_row_noise_from_raw(
    #             raw,
    #             pattern,
    #             smooth_kernel=self.row_smooth_kernel,
    #         )
    #     )

        packed_original = self.pack_raw(raw, pattern)
        packed_detrended = self.pack_raw(raw_detrended, pattern)

        return (
            packed_original,
            packed_detrended,
            row_profile_packed,
            row_meta,
        )

    def _estimate(self, G1, G2):

        return estimate_noise_params(
            G1,
            G2,
            min_mean_range=0.001,
        )

    def _clamp_params(self, alpha, sigma2):

        alpha = float(
            np.clip(alpha, self.alpha_min, self.alpha_max)
        )

        sigma2 = float(
            np.clip(sigma2, self.sigma2_min, self.sigma2_max)
        )

        return alpha, sigma2

    def _precompute_global_params(self):
        global_alpha  = []
        global_sigma2 = []

        for scene_dir in self.scene_dirs:
            noisy_path, _ = self._find_mat_files(scene_dir)
            raw     = self._read_mat(noisy_path)
            pattern = self._get_bayer_pattern(scene_dir)

            packed = self.pack_raw(raw, pattern)   # 直接 pack，不去行噪声
            G1 = packed[:, :, 1].astype(np.float64)
            G2 = packed[:, :, 2].astype(np.float64)

            alpha, sigma2, meta = self._estimate(G1, G2)
            alpha, sigma2 = self._clamp_params(alpha, sigma2)

            global_alpha.append(alpha)
            global_sigma2.append(sigma2)

        return global_alpha, global_sigma2



    def _compute_beta(self, alpha, sigma2):

        noise_level = float(alpha + sigma2)

        noise_level = float(
            np.clip(
                noise_level,
                self.noise_low,
                self.noise_high,
            )
        )

        ratio = (
            (noise_level - self.noise_low)
            / (self.noise_high - self.noise_low)
        )

        beta = (
            self.beta_max
            - ratio * (self.beta_max - self.beta_min)
        )

        return float(
            np.clip(
                beta,
                self.beta_min,
                self.beta_max,
            )
        )

    def _add_extra_noise(
        self,
        noisy,
        alpha,
        sigma2,
        row_std=None,
        generator=None,
    ):

        beta = self._compute_beta(alpha, sigma2)

        # strong beta augmentation
        if self.train:
            beta *= np.random.uniform(0.3, 2.0)

        var_z = beta * (
            alpha * torch.clamp(noisy, min=0)
            + sigma2
        )

        std_z = torch.sqrt(
            torch.clamp(var_z, min=1e-10)
        )

        pixel_noise = torch.randn(
            noisy.shape,
            generator=generator,
            device=noisy.device,
            dtype=noisy.dtype,
        )

        extra_noise = pixel_noise * std_z

        if (
            self.add_row_noise
            and row_std is not None
            and row_std > 0
        ):

            row_noise = torch.randn(
                (noisy.shape[0], noisy.shape[1], 1),
                generator=generator,
                device=noisy.device,
                dtype=noisy.dtype,
            ) * row_std

            extra_noise = extra_noise + row_noise

        return noisy + extra_noise

    def _make_var_map(
        self,
        crop_t,
        alpha,
        sigma2,
        row_std,
    ):

        var_map = (
            alpha * torch.clamp(crop_t, min=0)
            + sigma2
            + row_std ** 2
        )

        return torch.clamp(var_map, min=1e-10)

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir  = self.scene_dirs[idx]
        camera     = self._parse_camera(scene_dir)
        noisy_path, clean_path = self._find_mat_files(scene_dir)
        pattern    = self._get_bayer_pattern(scene_dir)
        raw_noisy  = self._read_mat(noisy_path)

        # 直接 pack，不去行噪声，row_profile 全部置零
        packed_noisy = self.pack_raw(raw_noisy, pattern)
        H, W = packed_noisy.shape[:2]

        # row_profile 不再使用，统一填零占位（保持接口不变）
        row_profile_zeros = np.zeros(
            (H, 1), dtype=np.float32
        )

        has_clean = clean_path is not None
        if has_clean:
            raw_clean  = self._read_mat(clean_path)
            packed_clean = self.pack_raw(raw_clean, pattern)

        alpha_global  = self.global_alpha[idx]
        sigma2_global = self.global_sigma2[idx]

        all_noisy, all_noisier, all_clean  = [], [], []
        all_alpha, all_sigma2              = [], []
        all_var_map                        = []
        all_row_profile                    = []
        all_row_std, all_row_var, all_r2   = [], [], []

        for crop_id in range(self.n_crop_per_img):
            hs = np.random.randint(0, H - self.patch_size + 1)
            ws = np.random.randint(0, W - self.patch_size + 1)

            crop_noisy = packed_noisy[
                hs:hs + self.patch_size,
                ws:ws + self.patch_size, :
            ]

            G1 = crop_noisy[:, :, 1].astype(np.float64)
            G2 = crop_noisy[:, :, 2].astype(np.float64)

            alpha_patch, sigma2_patch, meta = self._estimate(G1, G2)
            r2 = meta.get("r_squared", 0.0)

            if r2 >= self.r2_threshold:
                alpha_use, sigma2_use = self._clamp_params(alpha_patch, sigma2_patch)
            else:
                alpha_use  = alpha_global
                sigma2_use = sigma2_global

            # row_std 不再从行噪声估计，固定为 0
            row_std_use = 0.0

            # row_profile 填零，shape [4, patch_size, 1]
            row_profile_t = torch.zeros(
                4, self.patch_size, 1, dtype=torch.float32
            )

            crop_t = torch.FloatTensor(crop_noisy).permute(2, 0, 1)

            # ── STRONG RAW AUGMENTATION ──────────────────────────────
            if self.train:
                gain = np.random.uniform(0.3, 1.8)
                crop_t    = torch.clamp(crop_t * gain, 0, 1)
                alpha_use  *= np.random.uniform(0.5, 1.5) * gain
                sigma2_use *= np.random.uniform(0.3, 2.0) * gain * gain
                # row_std_use 为 0，不需要 jitter

            var_map_t = self._make_var_map(crop_t, alpha_use, sigma2_use, row_std_use)

            if self.train:
                aug_mode  = np.random.randint(0, 8)
                crop_t    = augment(crop_t,    aug_mode)
                var_map_t = augment(var_map_t, aug_mode)
                # row_profile 全零，augment 可跳过

            # stripe noise augmentation
            if self.train and np.random.rand() < 0.5:
                stripe = torch.randn(1, crop_t.shape[1], 1,
                                    dtype=crop_t.dtype) * np.random.uniform(0.001, 0.02)
                crop_t = crop_t + stripe

            # dead rows augmentation
            if self.train and np.random.rand() < 0.3:
                for _ in range(np.random.randint(1, 8)):
                    r = np.random.randint(0, crop_t.shape[1])
                    crop_t[:, r:r+1, :] *= np.random.uniform(0.0, 0.5)

            # hot pixel augmentation
            if self.train:
                mask    = torch.rand_like(crop_t) < 0.0005
                hot     = torch.rand_like(crop_t) * 2.0
                crop_t  = torch.where(mask, hot, crop_t)

            # black level drift
            if self.train and np.random.rand() < 0.3:
                crop_t = crop_t + np.random.uniform(-0.02, 0.02)

            crop_t    = torch.clamp(crop_t, 0, 1)
            noisier_t = self._add_extra_noise(
                crop_t, alpha_use, sigma2_use, row_std=None
            )

            all_noisy.append(crop_t)
            all_noisier.append(noisier_t)
            all_alpha.append(alpha_use)
            all_sigma2.append(sigma2_use)
            all_var_map.append(var_map_t)
            all_row_profile.append(row_profile_t)
            all_row_std.append(row_std_use)
            all_row_var.append(row_std_use ** 2)
            all_r2.append(r2)

            if has_clean:
                crop_clean = packed_clean[
                    hs:hs + self.patch_size,
                    ws:ws + self.patch_size, :
                ]
                crop_c = torch.FloatTensor(crop_clean).permute(2, 0, 1)
                if self.train:
                    crop_c = torch.clamp(crop_c * gain, 0, 1)
                    crop_c = augment(crop_c, aug_mode)
                all_clean.append(crop_c)

        result = {
            "noisy":       torch.stack(all_noisy),
            "noisier":     torch.stack(all_noisier),
            "alpha":       torch.tensor(all_alpha,  dtype=torch.float32),
            "sigma2":      torch.tensor(all_sigma2, dtype=torch.float32),
            "var_map":     torch.stack(all_var_map),
            "iso":         torch.tensor(self._parse_iso(scene_dir), dtype=torch.float32),
            "camera":      camera,
            "row_profile": torch.stack(all_row_profile),
            "row_std":     torch.tensor(all_row_std, dtype=torch.float32),
            "row_var":     torch.tensor(all_row_var, dtype=torch.float32),
            "r2":          torch.tensor(all_r2,      dtype=torch.float32),
        }

        if has_clean:
            result["clean"] = torch.stack(all_clean)

        return result