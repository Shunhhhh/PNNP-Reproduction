import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from train_syn_pnnp import PPMGenerator


class NoiseSynthesisDataset(Dataset):

    def __init__(
        self,
        clean_raw_dir,
        benchmark_dir,
        model_path,
        camera_config,
        iso_list=[800, 1250, 1600, 3200, 6400],
        dgain_range=(10, 200),
        patch_size=512,
        inp_clip_low=0.0,
        inp_clip_high=1.0,
        n_crop_per_img=8,
        white_level=16383.0,
        black_level=512.0,
    ):
        self.clean_paths = sorted(
            glob.glob(os.path.join(clean_raw_dir, "*.npy"))
        )
        print("Found clean RAW:", len(self.clean_paths))

        self.iso_list = iso_list
        self.dgain_range = dgain_range
        self.patch_size = patch_size
        self.inp_clip_low = inp_clip_low
        self.inp_clip_high = inp_clip_high
        self.n_crop_per_img = n_crop_per_img
        self.camera_config = camera_config
        self.wl = white_level
        self.bl = black_level

        # =========================
        # Load PNNP generator
        # =========================
        self.model = PPMGenerator(in_channels=4, nf=16)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # =========================
        # Load shading maps
        # =========================
        self.shadings = {}
        calib_path = os.path.join(benchmark_dir, "calib_res")

        for f in glob.glob(os.path.join(calib_path, "dark_shading_iso*.npy")):
            iso = int(os.path.basename(f).split("iso")[-1].split(".")[0])
            shading_np = np.load(f).astype(np.float32)
            shading_torch = torch.from_numpy(shading_np)
            self.shadings[iso] = self.pack_bayer(shading_torch)  # [4, H/2, W/2], ADU

        # =========================
        # Band noise calibration
        # =========================
        self.band_params = self._calibrate_band_noise(benchmark_dir)


        self.sys_gains = {}
        sys_gain_path = os.path.join(benchmark_dir, "calib_res", "sys_gain.npz")
        
        if os.path.exists(sys_gain_path):
            sys_gain_npz = np.load(sys_gain_path)
            for iso in self.iso_list:
                key = f"iso{iso}"
                if key in sys_gain_npz:
                    self.sys_gains[iso] = sys_gain_npz[key].item() if sys_gain_npz[key].ndim == 0 else float(sys_gain_npz[key].mean())
                else:
                    # 找最近的ISO
                    available = [int(k.replace("iso","")) for k in sys_gain_npz.keys()]
                    nearest = min(available, key=lambda x: abs(x - iso))
                    self.sys_gains[iso] = float(sys_gain_npz[f"iso{nearest}"])
                    print(f"sys_gain: ISO {iso} not found, using ISO {nearest}")
            print("Loaded sys_gain:", self.sys_gains)
        else:
            # sys_gain.npz 不存在，用默认值
            print(f"Warning: {sys_gain_path} not found, using default sys_gain=1.0")
            for iso in self.iso_list:
                self.sys_gains[iso] = 1.0

    # =========================================================
    # Bayer pack
    # =========================================================
    def pack_bayer(self, raw):
        if raw.ndim == 3:
            raw = raw.squeeze(0)

        H, W = raw.shape
        H = (H // 2) * 2
        W = (W // 2) * 2
        raw = raw[:H, :W]

        return torch.stack([
            raw[0:H:2, 0:W:2],
            raw[0:H:2, 1:W:2],
            raw[1:H:2, 0:W:2],
            raw[1:H:2, 1:W:2],
        ], dim=0)

    # =========================================================
    # Normalize: ADU -> [0, 1]
    # =========================================================
    def normalize_raw(self, raw_adu):
        return ((raw_adu - self.bl) / (self.wl - self.bl + 1e-8))

    def _get_iso(self, path):
        for p in path.split(os.sep):
            if "iso" in p.lower():
                return int("".join(filter(str.isdigit, p)))
        return 800

    # =========================================================
    # Band noise calibration（归一化域）
    # =========================================================
    def _calibrate_band_noise(self, benchmark_dir):
        dark_paths = glob.glob(
            os.path.join(benchmark_dir, "dark_frame_npz/**/*.npz"),
            recursive=True
        )

        iso_to_row = {}
        iso_to_col = {}

        for path in dark_paths:
            data = np.load(path)
            raw = torch.from_numpy(data["raw"].astype(np.float32))
            iso = self._get_iso(path)

            target_iso = min(self.shadings.keys(), key=lambda x: abs(x - iso))
            shading_full_adu = self.shadings[target_iso]  # [4, H/2, W/2], ADU

            y1 = (data["top_margin"] // 2) * 2
            x1 = (data["left_margin"] // 2) * 2
            h = data["height"]
            w = data["width"]

            shading = shading_full_adu[
                :,
                y1 // 2: y1 // 2 + h // 2,
                x1 // 2: x1 // 2 + w // 2
            ]  # [4, H/2, W/2], ADU

            # ── pipeline 对齐：ADU域减bl和shading ──────────────────
            noise = self.pack_bayer(raw)                        # [4, H/2, W/2], ADU
            noise = (noise - self.bl) - shading                 # ADU残差，与pipeline第1步一致
            noise = noise / (self.wl - self.bl + 1e-8)  # 归一化+RES_SCALE，与pipeline第4步一致

            # 行带
            row_band = noise.mean(dim=-1)
            row_band = row_band - row_band.mean(dim=-1, keepdim=True)
            sigma_row = row_band.flatten(1).std(dim=1).mean().item()

            # 列带
            residual = noise - row_band.unsqueeze(-1)
            col_band = residual.mean(dim=-2)
            col_band = col_band - col_band.mean(dim=-1, keepdim=True)
            sigma_col = col_band.flatten(1).std(dim=1).mean().item()

            iso_to_row.setdefault(iso, []).append(sigma_row)
            iso_to_col.setdefault(iso, []).append(sigma_col)

        band_params = {}
        print("\nBand Noise Calibration")
        print("----------------------")
        for iso in sorted(iso_to_row.keys()):
            row_sigma = np.mean(iso_to_row[iso])
            col_sigma = np.mean(iso_to_col[iso])
            band_params[iso] = {"row": row_sigma, "col": col_sigma}
            print(f"ISO {iso:5d} | row={row_sigma:.8e} | col={col_sigma:.8e}")

        return band_params

    # =========================================================
    # Dataset length
    # =========================================================
    def __len__(self):
        return len(self.clean_paths) * len(self.iso_list)

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    @torch.no_grad()
    def __getitem__(self, idx):

        img_idx = idx // len(self.iso_list)
        iso     = self.iso_list[idx % len(self.iso_list)]

        # ── 1. 加载 clean（ADU），归一化到[0,1]作为GT ─────────────
        raw_np  = np.load(self.clean_paths[img_idx]).astype(np.float32)
        raw     = torch.from_numpy(raw_np)
        H, W    = raw.shape
        th      = self.patch_size * 2
        tw      = self.patch_size * 2

        target_iso       = min(self.shadings.keys(), key=lambda x: abs(x - iso))
        shading_full_adu = self.shadings[target_iso]   # [4,H/2,W/2], ADU已减bl

        b_sigma_row = self.band_params[target_iso]["row"]
        b_sigma_col = self.band_params[target_iso]["col"]
        sys_gain    = self.sys_gains[target_iso]

        all_noisy, all_clean, all_dgain = [], [], []

        for _ in range(self.n_crop_per_img):

            # ── 2. crop ───────────────────────────────────────────
            top  = (np.random.randint(0, max(1, H - th + 1)) // 2) * 2
            left = (np.random.randint(0, max(1, W - tw + 1)) // 2) * 2
            raw_crop  = raw[top:top + th, left:left + tw]

            # ── 3. Bayer pack，归一化，作为GT ──────────────────────
            clean_adu = self.pack_bayer(raw_crop)               # [4,P,P], ADU
            clean     = self.normalize_raw(clean_adu)           # [4,P,P], [0,1]
            clean     = torch.clamp(clean, 0.0, 1.0)

            # ── 4. dgain（对应官方随机选曝光比）────────────────────
            dgain = float(np.random.randint(*self.dgain_range)) # 整数，如10~200

            # ── 5. 对照官方：clean反归一化再除dgain得到短曝光ADU ───
            # 官方：img = clip(clean,0,1) * (wl-bl) / dgain
            img_adu = torch.clamp(clean, 0.0, 1.0) * (self.wl - self.bl) / dgain

            # ── 7. shading 裁剪（ADU，已减bl）─────────────────────
            shading_adu = shading_full_adu[
                :,
                top  // 2: top  // 2 + self.patch_size,
                left // 2: left // 2 + self.patch_size
            ]                                                   # [4,P,P], ADU

            # ── 8. PPMGenerator 生成信号无关像素噪声 ───────────────
            n1 = torch.randn_like(clean).unsqueeze(0)
            n2 = torch.randn_like(clean).unsqueeze(0)
            iso_tensor = torch.tensor([[iso]]).float()
            gen_noise  = self.model(n1, n2, iso_tensor).squeeze(0)  # RES_SCALE域
            gen_noise_adu = gen_noise * (self.wl - self.bl) / np.sqrt(dgain)  # → ADU

            # ── 9. Band noise（ADU域）──────────────────────────────
            row_noise     = torch.randn(clean.shape[0], clean.shape[1], 1) * b_sigma_row
            col_noise     = torch.randn(clean.shape[0], 1, clean.shape[2]) * b_sigma_col
            band_noise_adu = (row_noise + col_noise) * (self.wl - self.bl)

            # ── 10. 全ADU域合成，对照官方：img+shot+dark ───────────
            # 官方 dark_frame = raw - shading - bl（ADU）
            # 你的等价替换：gen_noise_adu + band_noise_adu
            signal_std  = img_adu.std().item()
            
            shading_std = shading_adu.std().item()
            gen_std     = gen_noise_adu.std().item()
            band_std    = band_noise_adu.std().item()

            # print(f"ISO={iso:5d} dgain={dgain:6.0f} | "
            #     f"signal={signal_std:.4f} | "
            #     f"shading={shading_std:.4f} | "
            #     f"gen_noise={gen_std:.4f} | "
            #     f"band_noise={band_std:.4f} | "
            #     f"SNR={signal_std/(gen_std+band_std+1e-8):.2f}")

            noisy_adu = img_adu + shading_adu + gen_noise_adu + band_noise_adu
            noisy = noisy_adu / (self.wl - self.bl) * dgain
            noisy = torch.clamp(noisy, max=1.0)

            # print(f"  → noisy std={noisy.std():.4f} clean std={clean.std():.4f} "
            #     f"noise_std={(noisy-clean).std():.4f}")
            noisy_adu = img_adu + shading_adu + gen_noise_adu + band_noise_adu

            # ── 11. 归一化，对照官方：img/(wl-bl)*dgain ────────────
            noisy = noisy_adu / (self.wl - self.bl) * dgain
            noisy = torch.clamp(noisy, max=1.0)

            all_noisy.append(noisy.cpu())
            all_clean.append(clean.cpu())
            all_dgain.append(dgain)   

        return {
            "cam_model": self.camera_config,
            "iso":       torch.ones((1,)) * iso,
            "dgain":     torch.tensor(all_dgain, dtype=torch.float32),
            "noisy":     torch.stack(all_noisy),
            "clean":     torch.stack(all_clean),
        }