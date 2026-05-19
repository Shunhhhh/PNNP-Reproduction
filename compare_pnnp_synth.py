"""
对比PNNP、synth两种加噪
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.pnnp_train_dataset import NoiseSynthesisDataset
from datasets.synth_train_dataset import SynthTrainDataset


def compare_noise_pipelines(clean_crop, iso, dgain, 
                             pnnp_dataset, synth_dataset,
                             cam_model, device='cpu'):
    """
    clean_crop: [4, H, W] 归一化的clean patch
    """
    # =====================
    # PNNP pipeline
    # =====================
    wl = pnnp_dataset.wl
    bl = pnnp_dataset.bl
    
    img_adu = clean_crop * (wl - bl) / dgain
    
    n1 = torch.randn_like(clean_crop).unsqueeze(0)
    n2 = torch.randn_like(clean_crop).unsqueeze(0)
    iso_tensor = torch.tensor([[iso]]).float()
    gen_noise = pnnp_dataset.model(n1, n2, iso_tensor).squeeze(0)
    gen_noise_adu = gen_noise * (wl - bl)
    
    b_sigma_row = pnnp_dataset.band_params[iso]["row"]
    b_sigma_col = pnnp_dataset.band_params[iso]["col"]
    row_noise = torch.randn(clean_crop.shape[0], clean_crop.shape[1], 1) * b_sigma_row
    col_noise = torch.randn(clean_crop.shape[0], 1, clean_crop.shape[2]) * b_sigma_col
    band_noise_adu = (row_noise + col_noise) * (wl - bl)
    
    noisy_pnnp_adu = img_adu + gen_noise_adu + band_noise_adu
    noisy_pnnp = torch.clamp(noisy_pnnp_adu / (wl - bl) * dgain, 0.0, 1.0)
    noise_pnnp = noisy_pnnp - clean_crop

    # =====================
    # SynthTrainDataset pipeline
    # =====================
    sys_gain = synth_dataset.sys_gain[f"{cam_model}_iso{iso}"]
    dark_wl = 16383.0
    dark_bl = 512.0
    
    # 随机取一个dark frame crop
    import random
    dark_path = random.choice(synth_dataset.dark_frame_dirs[f"{cam_model}_iso{iso}"])
    import rawpy
    dark_raw = np.array(rawpy.imread(dark_path).raw_image).astype(np.float32)
    dark_shading = synth_dataset.dark_shadings[f"{cam_model}_iso{iso}"]
    dark_frame = dark_raw - dark_shading - dark_bl
    h_start, w_start, h_end, w_end = synth_dataset.cam_cfg[cam_model]["valid_roi"]
    dark_frame = dark_frame[h_start:h_end, w_start:w_end]
    dark_frame = synth_dataset.pack_raw(dark_frame, wl=dark_wl, bl=dark_bl, norm=False, clip=False)
    H, W = dark_frame.shape[:2]
    P = clean_crop.shape[1]
    hs = np.random.randint(0, H - P + 1)
    ws = np.random.randint(0, W - P + 1)
    dark_crop = dark_frame[hs:hs+P, ws:ws+P, :]  # [H, W, 4]
    
    clean_np = clean_crop.permute(1, 2, 0).numpy()
    noisy_synth = synth_dataset.noise_synthesis(
        clean=clean_np,
        dark_frame=dark_crop,
        dgain=dgain,
        sys_gain=sys_gain,
        wl=dark_wl,
        bl=dark_bl,
    )
    noisy_synth = torch.FloatTensor(noisy_synth).permute(2, 0, 1)
    noisy_synth = torch.clamp(noisy_synth, 0.0, 1.0)
    noise_synth = noisy_synth - clean_crop

    noisy_pnnp = noisy_pnnp.detach()
    noise_pnnp = noise_pnnp.detach()
    clean_crop = clean_crop.detach()
    noisy_synth = noisy_synth.detach()
    noise_synth = noise_synth.detach()

    # =====================
    # 统计对比
    # =====================
    print(f"\nISO={iso} dgain={dgain}")
    print(f"{'':20s} {'PNNP':>10s} {'Synth':>10s}")
    print(f"{'noise std':20s} {noise_pnnp.std().item():>10.5f} {noise_synth.std().item():>10.5f}")
    print(f"{'noise mean':20s} {noise_pnnp.mean().item():>10.5f} {noise_synth.mean().item():>10.5f}")
    print(f"{'noise kurtosis':20s} {_kurtosis(noise_pnnp):>10.3f} {_kurtosis(noise_synth):>10.3f}")
    print(f"{'noisy min':20s} {noisy_pnnp.min().item():>10.5f} {noisy_synth.min().item():>10.5f}")
    print(f"{'noisy max':20s} {noisy_pnnp.max().item():>10.5f} {noisy_synth.max().item():>10.5f}")

    # =====================
    # 可视化对比
    # =====================
    ch = 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(clean_crop[ch].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('clean')
    axes[0, 1].imshow(noisy_pnnp[ch].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'PNNP noisy')
    axes[0, 2].imshow((noise_pnnp[ch].numpy() * 10 + 0.5), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('PNNP noise x10')
    axes[0, 3].hist(noise_pnnp[ch].flatten().numpy(), bins=200, density=True, alpha=0.7, label='PNNP')
    axes[0, 3].hist(noise_synth[ch].flatten().numpy(), bins=200, density=True, alpha=0.7, label='Synth')
    axes[0, 3].legend()
    axes[0, 3].set_title('noise distribution')
    
    axes[1, 0].imshow(clean_crop[ch].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('clean')
    axes[1, 1].imshow(noisy_synth[ch].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Synth noisy')
    axes[1, 2].imshow((noise_synth[ch].numpy() * 10 + 0.5), cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Synth noise x10')
    
    # Q-Q plot
    p_flat = np.sort(noise_pnnp[ch].flatten().numpy())
    s_flat = np.sort(noise_synth[ch].flatten().numpy())
    n = min(len(p_flat), len(s_flat))
    idx = np.linspace(0, len(p_flat)-1, n).astype(int)
    axes[1, 3].scatter(s_flat[::100], p_flat[idx][::100], s=1, alpha=0.3)
    axes[1, 3].plot([s_flat.min(), s_flat.max()],
                    [s_flat.min(), s_flat.max()], 'r--')
    axes[1, 3].set_xlabel('Synth quantile')
    axes[1, 3].set_ylabel('PNNP quantile')
    axes[1, 3].set_title('Q-Q: PNNP vs Synth')
    
    plt.suptitle(f'ISO={iso} dgain={dgain}')
    plt.tight_layout()
    plt.savefig(f'compare_iso{iso}_dgain{dgain}.png')
    print(f"Saved compare_iso{iso}_dgain{dgain}.png")


def _kurtosis(x):
    x = x - x.mean()
    return ((x**4).mean() / (x**2).mean()**2).item()


pnnp_ds = NoiseSynthesisDataset(
    clean_raw_dir="../../data/xml196414/SID/Sony_npy/long",
    benchmark_dir="../../data/xml196414/SID/dev_phase_release/sonyzve10m2",
    model_path="checkpoints/PNNP_noise/ppm_generator_sonyzve10m2.pth",
    camera_config="sonyzve10m2",
)

synth_ds = SynthTrainDataset(
    clean_img_dir="../../data/xml196414/SID/Sony/long",
    benchmark_dir="../../data/xml196414/SID/dev_phase_release",
    camera_config={
    "sonyzve10m2": {
        "valid_roi": [0, 0, 4128, 6192]
    }
},
)

# 用pnnp_dataset里的一个clean patch
data = pnnp_ds[0]
clean_crop = data["clean"][0]  # [4, H, W]

for iso, dgain in [(800, 20), (800, 100), (3200, 20), (3200, 100), (6400, 150)]:
    compare_noise_pipelines(
        clean_crop=clean_crop,
        iso=iso,
        dgain=dgain,
        pnnp_dataset=pnnp_ds,
        synth_dataset=synth_ds,
        cam_model="sonyzve10m2"
    )