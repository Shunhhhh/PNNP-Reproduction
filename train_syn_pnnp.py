"""
训练PNNP加噪模型
"""
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import WeightedRandomSampler

def high_bit_reconstruction(patch, wl, bl, bit_depth=None):
    y = patch.float()
    # print(f"[DEBUG] y.shape={y.shape}, wl.shape={wl.shape}, bl.shape={bl.shape}")  # 加这行
    std = 0.5 / (wl.view(-1, 1, 1, 1) - bl.view(-1, 1, 1, 1) + 1e-8)
    bell_shaped_noise = torch.randn_like(y) * std
    high_bit_values = y + bell_shaped_noise
    return high_bit_values

# =========================================================
# Physics Pipeline (PNNP style)
# =========================================================
def gpu_physics_pipeline(raw, shading, wl, bl):
    if raw.ndim == 2: raw = raw.unsqueeze(0)
    if shading.ndim == 2: shading = shading.unsqueeze(0)

    x       = raw.float()
    shading = shading.float()

    wl_v = wl.view(-1, 1, 1).float()
    bl_v = bl.view(-1, 1, 1).float()

    # 1. ADU 域：减 bl，再减已减过 bl 的 shading
    x = (x - bl_v) - shading
    # print(f"[DEBUG] ADU残差 std={x.std():.4f}, mean={x.mean():.4f}")

    # 2. Bayer Pack: [B, H, W] -> [B, 4, H/2, W/2]
    H = (x.shape[-2] // 2) * 2
    W = (x.shape[-1] // 2) * 2
    x = x[:, :H, :W]
    out = torch.stack([
        x[:, 0:H:2, 0:W:2],
        x[:, 0:H:2, 1:W:2],
        x[:, 1:H:2, 0:W:2],
        x[:, 1:H:2, 1:W:2],
    ], dim=1)

    # 3. Band-wise 分离（ADU 域）
    row_band    = out.mean(dim=-1, keepdim=True)
    residual    = out - row_band
    col_band    = residual.mean(dim=-2, keepdim=True)
    pixel_noise = residual - col_band
    # print(f"[DEBUG] pixel_noise ADU std={pixel_noise.std():.4f}")

    # 4. 归一化
    pixel_noise = pixel_noise / (wl_v.view(-1, 1, 1, 1) - bl_v.view(-1, 1, 1, 1) + 1e-8)
    # print(f"[DEBUG] pixel_noise 归一化后 std={pixel_noise.std():.6f}")

    # 5. Dithering 高位重构
    pixel_noise = high_bit_reconstruction(pixel_noise, wl, bl, bit_depth=14)

    return pixel_noise.float()


# =========================================================
# ResBlock
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, c=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.SiLU(),
            nn.Conv2d(c, c, 1),
            nn.SiLU()
        )

    def forward(self, x):
        return x + self.net(x)


# =========================================================
# PPM Generator
# =========================================================
class PPMGenerator(nn.Module):

    def __init__(self, in_channels=4, nf=16,
                 alpha=1.48888310e-08,
                 beta=1.17790606e-05):

        super().__init__()

        self.in_proj = nn.Conv2d(in_channels, nf, 1)

        # calibration fitting
        self.alpha = alpha
        self.beta = beta

        # ISO-dependent branch
        self.dep = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.SiLU(),

            ResBlock(nf),
            ResBlock(nf),

            nn.Conv2d(nf, nf, 1),
            nn.SiLU()
        )

        # ISO-agnostic branch
        self.agn = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.SiLU(),

            ResBlock(nf),
            ResBlock(nf),

            nn.Conv2d(nf, nf, 1),
            nn.SiLU()
        )

        self.out_proj = nn.Conv2d(nf, in_channels, 1)

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.normal_(m.weight, std=1e-2)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, n1, n2, iso):

        # =====================================================
        # ISO gain from calibration
        # =====================================================

        gain = self.alpha * iso.float() + self.beta

        gain = gain.view(-1, 1, 1, 1)

        # =====================================================
        # Features
        # =====================================================

        n1_feat = self.in_proj(n1)
        n2_feat = self.in_proj(n2)

        # =====================================================
        # Branches
        # =====================================================

        npre = self.dep(n1_feat)

        # agnostic branch 必须很弱
        nfol = 0.02 * self.agn(n2_feat)

        # =====================================================
        # Physics prior
        # =====================================================
        x = npre * gain + nfol
        x = self.out_proj(x)

        return x

def compare_distribution(pred, gt, iso_val, channel=0):
    import matplotlib.pyplot as plt
    import numpy as np

    p = pred[:, channel].flatten().cpu().numpy()
    g = gt[:, channel].flatten().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图
    axes[0].hist(p, bins=200, alpha=0.5, label='pred', density=True)
    axes[0].hist(g, bins=200, alpha=0.5, label='gt',   density=True)
    axes[0].set_title(f'ISO {iso_val} | Channel {channel} | Histogram')
    axes[0].legend()
    xlim = max(np.abs(p).max(), np.abs(g).max())
    axes[0].set_xlim(-xlim * 1.1, xlim * 1.1)

    # Q-Q plot：用相同的分位数点采样，保证对齐
    quantiles = np.linspace(0, 100, 1000)
    p_q = np.percentile(p, quantiles)
    g_q = np.percentile(g, quantiles)

    axes[1].scatter(g_q, p_q, s=2, alpha=0.5)
    # 用 1%~99% 分位数范围设置轴范围，避免离群点拉伸
    lim_lo = min(np.percentile(g_q, 1),  np.percentile(p_q, 1))
    lim_hi = max(np.percentile(g_q, 99), np.percentile(p_q, 99))
    axes[1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', label='ideal')
    axes[1].set_xlim(lim_lo, lim_hi)
    axes[1].set_ylim(lim_lo, lim_hi)

    plt.tight_layout()
    plt.savefig(f'dist_iso{iso_val}.png')

# =========================================================
# Dataset
# =========================================================
class PPMTrainingDataset(Dataset):

    def __init__(self, benchmark_dir, patch_size=1024):

        self.dark_paths = glob.glob(
            os.path.join(
                benchmark_dir,
                'dark_frame_npz/**/*.npz'
            ), recursive=True
        )

        self.patch_size = patch_size
        self.shadings = {}


        calib_path = os.path.join(benchmark_dir, 'calib_res')

        files = glob.glob(os.path.join(calib_path, 'dark_shading_iso*.npy'))
        print(f"Found {len(files)} calib files.")

        for f in glob.glob(os.path.join(calib_path, 'dark_shading_iso*.npy')):
            iso = int(os.path.basename(f).split('iso')[-1].split('.')[0])
            ### 裁剪

            shading = np.load(f).astype(np.float32)
            self.shadings[iso] = torch.from_numpy(shading)
        
        self.alpha, self.beta = self._calibrate_banding()

    def _calibrate_banding(self):

        iso_to_sigmas = {}

        ref_wl = 16383.0
        ref_bl = 512.0

        for path in self.dark_paths:

            # =====================================================
            # Load dark frame
            # =====================================================
            data = np.load(path)

            raw = torch.from_numpy(
                data["raw"].astype(np.float32)
            )

            iso = self._get_iso(path)

            # =====================================================
            # Get shading
            # =====================================================
            target_iso = min(
                self.shadings.keys(),
                key=lambda x: abs(x - iso)
            )

            shading_full = self.shadings[target_iso]

            y1 = (data["top_margin"] // 2) * 2
            x1 = (data["left_margin"] // 2) * 2
            h = data["height"]
            w = data["width"]

            shading = shading_full[
                y1:y1+h,
                x1:x1+w
            ]

            # print(f"[DEBUG] raw   min={raw.min():.2f} max={raw.max():.2f} mean={raw.mean():.2f}")
            # print(f"[DEBUG] shading min={shading.min():.2f} max={shading.max():.2f} mean={shading.mean():.2f}")

            # =====================================================
            # Residual noise
            # =====================================================
            noise = (raw - ref_bl) - shading ##########################################

            # =====================================================
            # Normalize
            # =====================================================
            noise = noise / (ref_wl - ref_bl + 1e-8)

            # =====================================================
            # Bayer pack
            # =====================================================
            H = (noise.shape[0] // 2) * 2
            W = (noise.shape[1] // 2) * 2

            noise = noise[:H, :W]

            noise = torch.stack([
                noise[0:H:2, 0:W:2],
                noise[0:H:2, 1:W:2],
                noise[1:H:2, 0:W:2],
                noise[1:H:2, 1:W:2],
            ], dim=0)

            # =====================================================
            # Row band estimation
            # =====================================================
            row_band = noise.mean(dim=-1)
            row_band = row_band - row_band.mean(dim=-1, keepdim=True)
            sigma_row = row_band.flatten(1).std(dim=1).mean()
            # =====================================================
            # Save
            # =====================================================
            if iso not in iso_to_sigmas:
                iso_to_sigmas[iso] = []

            iso_to_sigmas[iso].append(sigma_row)

        # =========================================================
        # Average sigma per ISO
        # =========================================================

        isos = []
        sigmas = []

        for iso in sorted(iso_to_sigmas.keys()):

            mean_sigma = np.mean(
                iso_to_sigmas[iso]
            )

            isos.append(iso)
            sigmas.append(mean_sigma)

        # =========================================================
        # Linear fit
        # =========================================================

        alpha, beta = np.polyfit(
            isos,
            sigmas,
            1
        )

        print("\nBand Noise Calibration")
        print("----------------------")

        for iso, sigma in zip(isos, sigmas):
            print(f"ISO {iso:5d} | sigma_row = {sigma:.8f}")

        print(f"\nFitted:")
        print(f"alpha = {alpha:.8e}")
        print(f"beta  = {beta:.8e}")

        return alpha, beta


    def _get_iso(self, path):
        for p in path.split(os.sep):
            if 'iso' in p.lower():
                return int(''.join(filter(str.isdigit, p)))
        return 800

    def __len__(self):
        return len(self.dark_paths)

    def __getitem__(self, idx):
        data = np.load(self.dark_paths[idx])
        raw = torch.from_numpy(data["raw"].astype(np.float32))
        iso = self._get_iso(self.dark_paths[idx])
        y1 = (data["top_margin"] // 2) * 2
        x1 = (data["left_margin"] // 2) * 2
        h = data["height"]
        w = data["width"]

        # =====================================================
        # shading
        # =====================================================
        target_iso = min(self.shadings.keys(), key=lambda x: abs(x - iso))
        shading_full = self.shadings[target_iso]
        shading = shading_full[y1:y1+h, x1:x1+w]

        # =====================================================
        # RAW domain crop
        # =====================================================
        th = self.patch_size * 2
        tw = self.patch_size * 2
        i = np.random.randint(0, max(1, h - th + 1))
        j = np.random.randint(0, max(1, w - tw + 1))
        i = (i // 2) * 2
        j = (j // 2) * 2
        raw = raw[i:i+th, j:j+tw]
        shading = shading[i:i+th, j:j+tw]
        return {
            "raw": raw,
            "shading": shading,
            "iso": torch.tensor(iso).float(),
            "wl": torch.tensor(data.get("white_level", 16380.0)).float(),
            "bl": torch.tensor(data.get("black_level", 512.0)).float(),
            "alpha": torch.tensor(self.alpha).float(),
            "beta": torch.tensor(self.beta).float()
        }

@torch.no_grad()
def full_evaluation(model, dataset, device, iso_list=None):
    model.eval()
    if iso_list is None:
        iso_list = sorted(dataset.shadings.keys())
 
    for iso_val in iso_list:
        indices = [i for i, p in enumerate(dataset.dark_paths)
                   if dataset._get_iso(p) == iso_val][:4]
        if not indices:
            continue
 
        preds, gts = [], []
        for idx in indices:
            data = dataset[idx]
            raw     = data["raw"].unsqueeze(0).to(device)
            shading = data["shading"].unsqueeze(0).to(device)
            wl      = data["wl"].unsqueeze(0).to(device)
            bl      = data["bl"].unsqueeze(0).to(device)
            iso     = data["iso"].unsqueeze(0).to(device)
 
            gt   = gpu_physics_pipeline(raw, shading, wl, bl)
            n1   = torch.randn_like(gt)
            n2   = torch.randn_like(gt)
            pred = model(n1, n2, iso)
 
            preds.append(pred)
            gts.append(gt)
 
        pred_all = torch.cat(preds)
        gt_all   = torch.cat(gts)
 
        print(f"\n{'='*40}")
        print(f"ISO {iso_val} | PredStd={pred_all.std():.6f} | GTStd={gt_all.std():.6f}")
        compare_distribution(pred_all, gt_all, iso_val)
    

def check_model_std(model, dataset, device, num_samples=5):
    model.eval()
    print("\n" + "="*30)
    print("      NOISE STD CHECK")
    print("="*30)
    
    with torch.no_grad():
        for i in range(num_samples):
            data = dataset[i]
            
            raw = data["raw"].unsqueeze(0).to(device)
            shading = data["shading"].unsqueeze(0).to(device)
            wl = torch.tensor([data["wl"]]).to(device)
            bl = torch.tensor([data["bl"]]).to(device)
            iso_val = data["iso"]
            
            # GT
            target_noise = gpu_physics_pipeline(raw, shading, wl, bl)
            
            noise_in1 = torch.randn_like(target_noise)
            noise_in2 = torch.randn_like(target_noise)
            iso_tensor = torch.tensor([[iso_val]], device=device).float()
            
            pred = model(noise_in1, noise_in2, iso_tensor)
            
            print(f"Sample {i} (ISO {iso_val:5d}) | PredStd: {pred.std():.6f} | GTStd: {target_noise.std():.6f}")
    model.train()




class DDLLoss(nn.Module):
    def __init__(self, num_samples=2048):
        super().__init__()
        self.num_samples = num_samples

    def get_quantile(self, x, p):
        B, C, N = x.shape
        x_sorted, _ = torch.sort(x, dim=-1)
        idx_cont = p.expand(B, C, -1) * (N - 1)
        idx_low = idx_cont.long().clamp(0, N - 2)
        idx_high = (idx_low + 1).clamp(0, N - 1)
        weight = idx_cont - idx_low.float()
        return torch.gather(x_sorted, -1, idx_low) + weight * (torch.gather(x_sorted, -1, idx_high) - torch.gather(x_sorted, -1, idx_low))

    def get_cdf(self, x, v_query):
        B, C, N = x.shape
        x_sorted, _ = torch.sort(x, dim=-1)
        v_query = v_query.expand(B, C, -1)
        idx = torch.searchsorted(x_sorted, v_query).clamp(1, N - 1)
        x_l, x_r = torch.gather(x_sorted, -1, idx-1), torch.gather(x_sorted, -1, idx)
        return (idx.float() - (x_r - v_query) / (x_r - x_l + 1e-6)) / N

    def forward(self, pred, gt):
        B, C, H, W = pred.shape
        pred, gt = pred.view(B, C, -1), gt.view(B, C, -1)
        p_query = torch.linspace(0.01, 0.99, self.num_samples, device=pred.device).view(1, 1, -1)
        
        l_quantile = F.l1_loss(self.get_quantile(pred, p_query), self.get_quantile(gt, p_query))
        
        v_query = gt.mean(dim=-1, keepdim=True) + torch.randn(B, C, self.num_samples, device=pred.device) * (gt.std(dim=-1, keepdim=True) * 3 + 0.01)
        l_cdf = F.l1_loss(self.get_cdf(pred, v_query), self.get_cdf(gt, v_query))
        
        return l_quantile + l_cdf

# =========================
# 5. 主训练循环
# =========================

def train_noise_generator(benchmark_dir, camera, is_training, steps_per_iso=2000):
    device = torch.device("cuda")
    model = PPMGenerator().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=steps_per_iso, eta_min=1e-5)
    ddl_loss = DDLLoss(num_samples=2048)

    full_dataset = PPMTrainingDataset(benchmark_dir=benchmark_dir)
    all_isos = sorted(list(full_dataset.shadings.keys()))
    
    print(f"Starting PNNP Training for {camera}...")

    # # # 先打印一下各ISO的实际gain值
    # model.load_state_dict(torch.load(f"checkpoints/PNNP_noise/ppm_generator_{camera}.pth"))
    # model.eval()
    # with torch.no_grad():
    #     print("=== Scale vs GT std ===")
    #     for iso_val in sorted(full_dataset.shadings.keys()):
    #         iso_t = torch.tensor([[float(iso_val)]], device=device)
    #         iso_log = torch.log2(iso_t / 100.0)
            
    #         # 用你实际的 forward 里的 gain 计算方式
    #         w = F.softplus(model.iso_gain[0].weight)
    #         b = model.iso_gain[0].bias
    #         scale = F.softplus(F.linear(iso_log, w, b)).item()
            
    #         # gt std
    #         indices = [i for i, p in enumerate(full_dataset.dark_paths)
    #                 if full_dataset._get_iso(p) == iso_val][:2]
    #         gt_stds = []
    #         for idx in indices:
    #             data = full_dataset[idx]
    #             raw = data["raw"].unsqueeze(0).to(device)
    #             shading = data["shading"].unsqueeze(0).to(device)
    #             wl = data["wl"].unsqueeze(0).to(device)
    #             bl = data["bl"].unsqueeze(0).to(device)
    #             gt = gpu_physics_pipeline(raw, shading, wl, bl)
    #             gt_stds.append(gt.std().item())
            
    #         print(f"ISO {iso_val:5d} | scale={scale:.6f} | gt_std={np.mean(gt_stds):.6f} | ratio={scale/np.mean(gt_stds):.2f}")

        



    model.train()
    for iso_val in all_isos:

        print(f"\n>>> Optimizing ISO {iso_val}")

        # =====================================================
        # 每个ISO重新初始化模型
        # =====================================================

        model = PPMGenerator().to(device)
        model.train()

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-2
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_iso,
            eta_min=1e-5
        )

        indices = [
            i for i, p in enumerate(full_dataset.dark_paths)
            if full_dataset._get_iso(p) == iso_val
        ]

        if not indices:
            continue

        iso_loader = DataLoader(
            Subset(full_dataset, indices),
            batch_size=4,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        def inf_train_gen(loader):
            while True:
                for batch in loader:
                    yield batch

        data_gen = inf_train_gen(iso_loader)

        # =====================================================
        # train
        # =====================================================

        for step in range(1, steps_per_iso + 1):

            data = next(data_gen)

            raw_gpu = data["raw"].to(device, non_blocking=True)
            shading_gpu = data["shading"].to(device, non_blocking=True)
            iso = data["iso"].to(device, non_blocking=True)
            wl = data["wl"].to(device, non_blocking=True)
            bl = data["bl"].to(device, non_blocking=True)

            target_noise = gpu_physics_pipeline(
                raw_gpu,
                shading_gpu,
                wl,
                bl
            )

            noise_in1 = torch.randn_like(target_noise)
            noise_in2 = torch.randn_like(target_noise)

            generated_noise = model(
                noise_in1,
                noise_in2,
                iso
            )

            loss = ddl_loss(
                generated_noise,
                target_noise
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.1
            )

            optimizer.step()
            scheduler.step()

            if step % 200 == 0:

                print(
                    f"Step {step:4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"PredStd: {generated_noise.std().item():.5f} | "
                    f"GTStd: {target_noise.std().item():.5f}"
                )

        # =====================================================
        # save per ISO
        # =====================================================

        os.makedirs(
            "checkpoints/PNNP_noise",
            exist_ok=True
        )

        save_path = (
            f"checkpoints/PNNP_noise/"
            f"ppm_generator_{camera}_iso{iso_val}.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        print(f"\nSaved: {save_path}")

        # =====================================================
        # evaluate current ISO only
        # =====================================================

        full_evaluation(
            model,
            full_dataset,
            device,
            [iso_val]
        )


if __name__ == "__main__":
    train_noise_generator(
        benchmark_dir="../../data/xml196414/SID/dev_phase_release/sonyzve10m2", 
        camera="sonyzve10m2",
        is_training=True
    )