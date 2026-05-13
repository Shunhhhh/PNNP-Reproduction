#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm


# =========================
# 1. 读取 SID txt
# =========================
def load_sid_list(txt_path):
    data = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            short_path = parts[0]
            long_path = parts[1]

            iso = int(parts[2].replace("ISO", ""))  # ISO200 → 200

            data.append((short_path, long_path, iso))

    return data


def extract_mean_var_sid(raw, num_bins=50):

    # RGGB
    channels = [
        raw[0::2, 0::2],
        raw[0::2, 1::2],
        raw[1::2, 0::2],
        raw[1::2, 1::2],
    ]

    means, vars_ = [], []

    for ch in channels:

        flat = ch.flatten()

        # 去掉极端值
        flat = flat[(flat > 0.01) & (flat < 0.9)]

        if len(flat) < 100:
            continue

        # 按强度排序
        flat = np.sort(flat)

        bins = np.array_split(flat, num_bins)

        for b in bins:
            if len(b) < 20:
                continue

            means.append(np.mean(b))
            vars_.append(np.var(b))

    return np.array(means), np.array(vars_)

# =========================
# 3. 拟合 k, sigma
# =========================
def fit_k_sigma(means, vars_):
    if len(means) < 50:
        return None, None

    coeff = np.polyfit(means, vars_, 1)
    return coeff[0], coeff[1]


# =========================
# 4. 主流程（按 ISO 统计）
# =========================
def calibrate_sid_from_txt(root, txt_path):

    results = {}

    data_list = load_sid_list(txt_path)

    print(f"Found {len(data_list)} samples")

    for short_rel, long_rel, iso in tqdm(data_list):

        short_path = os.path.join(root, short_rel)

        if not os.path.exists(short_path):
            continue

        try:
            raw = np.load(short_path).astype(np.float32)

            # ⭐ 归一化（SID Sony是14bit）
            if raw.max() > 1.5:
                raw /= 16383.0

            means, vars_ = extract_mean_var_sid(raw)

            if len(means) < 10:
                continue

            k, sigma = fit_k_sigma(means, vars_)
            if k is None:
                continue

            if iso not in results:
                results[iso] = {"k": [], "sigma": []}

            results[iso]["k"].append(k)
            results[iso]["sigma"].append(sigma)

        except Exception:
            continue

    return results


# =========================
# 5. ISO → K/B 拟合
# =========================
def fit_iso_model(results):

    print("\n==============================")
    print("KSigma ISO Model (SID)")
    print("==============================")


    iso_list, k_list, sigma_list = [], [], []

    for iso in sorted(results.keys()):

        k_vals = np.array(results[iso]["k"])
        s_vals = np.array(results[iso]["sigma"])

        # ⭐ 去异常值
        if len(k_vals) > 5:
            k_vals = k_vals[np.abs(k_vals - np.median(k_vals)) < 2*np.std(k_vals)]
            s_vals = s_vals[np.abs(s_vals - np.median(s_vals)) < 2*np.std(s_vals)]

        if len(k_vals) == 0:
            continue

        k_m = np.mean(k_vals)
        s_m = np.mean(s_vals)

        print(f"ISO {iso}: k={k_m:.6e}, sigma={s_m:.6e}")

        iso_list.append(iso)
        k_list.append(k_m)
        sigma_list.append(s_m)

    if len(iso_list) < 2:
        print("❌ ISO数量不足，无法拟合曲线")
        return

    iso_arr = np.array(iso_list)
    k_arr = np.array(k_list)
    sigma_arr = np.array(sigma_list)

    # 排序
    idx = np.argsort(iso_arr)
    iso_arr = iso_arr[idx]
    k_arr = k_arr[idx]
    sigma_arr = sigma_arr[idx]

    # ⭐ 单调性约束（非常关键）
    k_arr = np.maximum.accumulate(k_arr)
    sigma_arr = np.maximum(sigma_arr, 1e-8)

    # ⭐ 拟合
    K_coeff = np.polyfit(iso_arr, k_arr, 1)
    B_coeff = np.polyfit(iso_arr, sigma_arr, 2)

    print("\n--- KSigma Model ---")
    print(f"K_coeff=[{K_coeff[0]:.10f}, {K_coeff[1]:.10f}],")
    print(f"B_coeff=[{B_coeff[0]:.10e}, {B_coeff[1]:.10e}, {B_coeff[2]:.10e}],")
    print("anchor=1600,")
    print("==============================\n")


# =========================
# 6. 主函数
# =========================
if __name__ == "__main__":

    root = "../../data/xml196414/SID"
    txt_path = "../../data/xml196414/SID/Sony_npy/Sony_train_list.txt"

    results = calibrate_sid_from_txt(root, txt_path)

    print("有效ISO:", sorted(results.keys()))

    fit_iso_model(results)