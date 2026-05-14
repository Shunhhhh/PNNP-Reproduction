import os
import numpy as np
import rawpy
from glob import glob
from tqdm import tqdm

from matplotlib import pyplot as plt
from scipy.stats import linregress


def load_one_raw(img_name):
    raw_file = rawpy.imread(img_name)
    black_level = np.mean(raw_file.black_level_per_channel)
    white_level = float(raw_file.white_level)
    bayer_img = raw_file.raw_image_visible
    bayer_img = np.float32(bayer_img)
    return bayer_img, black_level, white_level


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def center_crop(img, crop_size=512):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


##------------------------------------------------------------------------------------


if __name__ == "__main__":

    camera_type = "sonyzve10m2"
    suffix = "ARW"
    all_iso = [800, 1250, 1600, 3200, 6400]
    root_dir = f"./AIM_challenge/{camera_type}"

    make_dir(os.path.join(root_dir, 'calib_res'))
    make_dir(os.path.join(root_dir, "calib_res/vis"))

    all_K = {}

    for iso_id, curr_iso in enumerate(all_iso):
        print(f">>>>> processing iso: {curr_iso}")
        all_file = sorted(glob(os.path.join(root_dir, f"flat_field/iso{curr_iso}/**/*.{suffix}")))
        

        stats_res = {"mu": [], "var": []}
        for img_dir in tqdm(all_file):
            bayer, bl, wl = load_one_raw(img_dir)  ## no need to subtract bl as it won't affect the slope
            rggb = np.stack([bayer[0::2, 0::2], bayer[0::2, 1::2], bayer[1::2, 0::2], bayer[1::2, 1::2]], axis=2)
            rggb = center_crop(rggb, crop_size=96 if curr_iso < 320 else 512)
            stats_res["mu"].append(np.mean(rggb, axis=(0, 1)))
            stats_res["var"].append(np.var(rggb, axis=(0, 1)))

        ## calib and plot
        color = ["r", "g", "cyan", "b"]
        all_channels = ["R", "Gr", "Gb", "B"]
        plt.figure(figsize=(6, 6))
        for i in range(4):
            med, var = np.array(stats_res["mu"])[:, i], np.array(stats_res["var"])[:, i]
            ## slope is the system gain K, and we want to r^2 > 0.99 to be a good fit
            slope, intercept, r_value, p_value, std_err = linregress(med, var)
            if i == 0:
                all_K[f"iso{curr_iso}"] = [slope]
            else:
                all_K[f"iso{curr_iso}"].append(slope)

            ## plot
            plt.plot(med, var, "o", color=color[i])
            plt.plot(
                med,
                slope * np.array(med) + intercept,
                "--",
                color=color[i],
                label=f"gain {all_channels[i]}: {slope:.5f}, $r^2$-score: {r_value**2:.5f}",
            )
        plt.title(f"iso: {curr_iso}")
        plt.xlabel("pixel value mean")
        plt.ylabel("pixel value var")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"calib_res/vis/sys_gain_iso{curr_iso}.png"))
        plt.close()

    np.savez(os.path.join(root_dir, "calib_res/sys_gain.npz"), **all_K)
