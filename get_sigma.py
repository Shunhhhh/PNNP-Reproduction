import numpy as np
import glob
import rawpy

def estimate_read_noise(dark_frame_paths, bl, wl):
    """
    用多帧暗帧估计 per-channel 读出噪声 sigma。
    
    dark_frame_paths : list of .npz 路径（同一 ISO 下）
    返回 sigma_per_ch : shape (4,)，归一化域
    """
    frames = []
    for path in dark_frame_paths:
        data = np.load(path)
        raw  = data["raw"].astype(np.float64)
        # pack 成 4 通道
        packed = np.stack([
            raw[0::2, 0::2],
            raw[0::2, 1::2],
            raw[1::2, 0::2],
            raw[1::2, 1::2],
        ], axis=0)  # (4, H/2, W/2)，ADU域
        frames.append(packed)
    
    frames = np.stack(frames, axis=0)   # (N, 4, H/2, W/2)
    
    # 多帧均值 = 暗电流（固定图案）
    mean_frame = frames.mean(axis=0)    # (4, H/2, W/2)
    
    # 每帧减均值 = 读出噪声（i.i.d. 高斯）
    residuals = frames - mean_frame     # (N, 4, H/2, W/2)
    
    # per-channel std，ADU域
    sigma_adu = residuals.std(axis=(0, 2, 3))   # (4,)
    
    # 归一化到 [0,1] 域
    sigma_norm = sigma_adu / (wl - bl)
    
    return sigma_norm

import numpy as np


import numpy as np


def load_alpha_from_sys_gain(npz_path, iso):
    """
    从 sys_gain.npz 中读取指定 ISO 对应的 alpha

    Parameters
    ----------
    npz_path : str
        sys_gain.npz 路径

    iso : int or str
        ISO值，例如:
            800
            1600
            "iso1600"

    Returns
    -------
    alpha : float or ndarray
    """

    data = np.load(npz_path)

    print("Keys in npz:", data.files)

    # 自动转换
    if isinstance(iso, int):
        iso_key = f"iso{iso}"
    else:
        iso_key = iso

    if iso_key not in data.files:
        raise KeyError(
            f"{iso_key} not found.\n"
            f"Available keys: {data.files}"
        )

    alpha = data[iso_key]

    print(f"Loaded alpha from '{iso_key}'")
    print("alpha =", alpha)

    return alpha






# # 随便取一张 ARW 读 wl/bl
# raw_file = rawpy.imread("../../data/xml196414/SID/dev_phase_release/sonyzve10m2/test_data/paired_input/scene3_iso800_dgain200.ARW")
# wl = float(raw_file.white_level)
# bl = float(np.mean(raw_file.black_level_per_channel))
# print("wl:", wl, "bl", bl)

# # 取某个 ISO 的所有暗帧
# paths = glob.glob("../../data/xml196414/SID/dev_phase_release/sonyzve10m2/dark_frame_npz/iso6400/*.npz")
# sigma = estimate_read_noise(paths, bl, wl)
# print("sigma per channel:", sigma)
# print("mean sigma:", sigma.mean())


alpha = load_alpha_from_sys_gain(
    "../../data/xml196414/SID/dev_phase_release/sonyzve10m2/calib_res/sys_gain.npz",
    6400
)
