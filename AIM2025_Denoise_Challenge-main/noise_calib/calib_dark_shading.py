import os
import numpy as np
import rawpy
from glob import glob
from tqdm import tqdm
import random

from matplotlib import pyplot as plt
from scipy.stats import linregress


def load_one_raw(img_name):
    raw_file = rawpy.imread(img_name)
    black_level = np.mean(raw_file.black_level_per_channel)
    white_level = float(raw_file.white_level)
    bayer_img = np.float32(raw_file.raw_image)
    return bayer_img, black_level, white_level


def center_crop(img, crop_size=512):
    h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


def vis_bayer(bayer):
    r = bayer[0::2, 0::2]
    g = (bayer[0::2, 1::2] + bayer[1::2, 0::2]) / 2
    b = bayer[1::2, 1::2]
    return np.stack((r, g, b), axis=-1)


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def per_pixel_r2_value(ds_k, ds_b, legal_iso, ds_data):
    """ds_k and ds_b should be [h, w], legal_iso should be [n_iso], ds_data should be [n_iso, h, w]"""

    ds_k, ds_b = ds_k[None, :, :], ds_b[None, :, :]  # [1, h, w]
    legal_iso = (
        np.array(legal_iso)[:, None, None].repeat(ds_data.shape[1], axis=1).repeat(ds_data.shape[2], axis=2)
    )  # [n_iso, h, w]

    y_hat = ds_k * legal_iso + ds_b  # [n_iso, h, w]
    y_bar = np.sum(ds_data, axis=0, keepdims=True) / ds_data.shape[0]  # [1, h, w]
    ss_reg = np.sum((ds_data - y_hat) ** 2, axis=0)  # [h, w]
    ss_tot = np.sum((ds_data - np.repeat(y_bar, ds_data.shape[0], axis=0)) ** 2, axis=0)  # [h, w]
    res = 1 - np.divide(ss_reg, ss_tot)
    return res


def compute_dark_shading_for_one_iso(all_file, iso, plt_save_dir):
    nimg_for_plot = [int(x) for x in np.linspace(0, len(all_file), 50)[1:-1]]
    var_for_plot = []
    fail_count = 0

    ## calculate dark shading
    for data_id, img_dir in enumerate(tqdm(all_file)):
        bayer, bl, wl = load_one_raw(img_dir)
        bayer = np.float32(bayer - bl)
        all_bayer = bayer if data_id == 0 else all_bayer + bayer
        ## for vis
        if data_id in nimg_for_plot:
            var_for_plot.append(np.var(all_bayer / data_id))
    
    dark_shading = all_bayer / (len(all_file) - fail_count)

    ## vis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(nimg_for_plot[: len(var_for_plot)], var_for_plot)
    ax[0].set_title("pixel_var VS #shot")
    ax[0].set_xlabel("#shot")
    ax[0].set_ylabel("pixel var")

    print("ds stats:", dark_shading.max(), dark_shading.min(), dark_shading.mean(), dark_shading.std())
    ds_vis = vis_bayer(dark_shading)
    ds_vis = np.clip(ds_vis, -3 * dark_shading.std(), 3 * dark_shading.std())
    ax[1].imshow(ds_vis)
    ax[1].set_title(f"ds of iso{iso}")

    plt.tight_layout()
    plt.savefig(plt_save_dir)
    plt.close()

    return dark_shading


##------------------------------------------------------------------------------------


if __name__ == "__main__":

    camera_type = "sonyzve10m2"
    suffix = "ARW"
    all_iso = [800, 1250, 1600, 3200, 6400]
    root_dir = f"./AIM_challenge/{camera_type}"

    make_dir(os.path.join(root_dir, "calib_res"))
    make_dir(os.path.join(root_dir, "calib_res/vis"))

    for iso in all_iso:
        all_file = glob(os.path.join(root_dir, f"dark_frame/iso{iso}/*.{suffix}"))
        dark_shading = compute_dark_shading_for_one_iso(
            all_file,
            iso,
            plt_save_dir=os.path.join(root_dir, f"calib_res/vis/dark_shading_iso{iso}.png"),
        )
        np.save(os.path.join(root_dir, f"calib_res/dark_shading_iso{iso}.npy"), dark_shading)
