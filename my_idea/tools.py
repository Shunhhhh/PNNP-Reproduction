import numpy as np


# def estimate_row_noise_from_raw(raw, pattern, smooth_kernel=0):
#     """
#     在 pack 之前，对每个 Bayer 通道分别估计行噪声。
#     Gr 只在偶数行，Gb 只在奇数行，各自的行中值只含本通道场景内容。
#     利用 Gr/Gb 差异消除场景，和原来 G1-G2 逻辑等价，但行索引完全对应。
#     """
#     pattern = pattern.lower()

#     if pattern in ("rggb", "grbg"):
#         gr_rows = raw[0::2, :]
#         gb_rows = raw[1::2, :]
#     else:  # bggr, gbrg
#         gb_rows = raw[0::2, :]
#         gr_rows = raw[1::2, :]

#     gr_median = np.median(gr_rows, axis=1)  # [H/2]
#     gb_median = np.median(gb_rows, axis=1)  # [H/2]

#     diff_median = gr_median - gb_median     # [H/2]
#     diff_median -= np.mean(diff_median)

#     if smooth_kernel and smooth_kernel > 1:
#         kernel = int(smooth_kernel)
#         if kernel % 2 == 0:
#             kernel += 1
#         pad = kernel // 2
#         padded = np.pad(diff_median, (pad, pad), mode="reflect")
#         diff_median = np.convolve(
#             padded, np.ones(kernel) / kernel, mode="valid"
#         )

#     # Var[diff] = Var[gr_noise] + Var[gb_noise] = 2 * Var[single]
#     row_var_single = float(np.var(diff_median) / 2.0)
#     row_std = float(np.sqrt(max(row_var_single, 1e-12)))

#     # row_profile: packed 行域，shape [H/2, 1]
#     row_profile_packed = (diff_median / 2.0)[:, np.newaxis].astype(np.float32)

#     # RAW 域去噪：Gr 行减去 half，Gb 行加上 half（符号相反保持一致性）
#     raw_detrended = raw.astype(np.float64, copy=True)
#     half = diff_median / 2.0
#     if pattern in ("rggb", "grbg"):
#         raw_detrended[0::2, :] -= half[:, np.newaxis]  # Gr 行
#         raw_detrended[1::2, :] += half[:, np.newaxis]  # Gb 行
#     else:
#         raw_detrended[1::2, :] -= half[:, np.newaxis]
#         raw_detrended[0::2, :] += half[:, np.newaxis]

#     meta = {
#         "row_var": row_var_single,
#         "row_std": row_std,
#     }
#     return row_profile_packed, raw_detrended.astype(np.float32), meta


def estimate_noise_params(
    G1,
    G2,
    patch_size=16,
    min_valid_patches=5,
    n_bins=5,
    flat_percentile=90,
    brightness_percentile=(5, 95),
    lower_q=0.1,
    min_mean_range=0.001,
):
    """
    Estimate Poisson-Gaussian noise parameters (alpha, sigma2) from G1/G2.

    Expects G1/G2 to have row noise already removed (via
    estimate_row_noise_from_raw before pack_raw). This function only handles
    shot noise (alpha) and read noise (sigma2) estimation.
    """
    H, W = G1.shape
    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size

    
    G1 = G1[:H_crop, :W_crop].astype(np.float64, copy=False)
    G2 = G2[:H_crop, :W_crop].astype(np.float64, copy=False)

    diff_img = G1 - G2

    noise_var = float(np.var(diff_img) / 2.0)
    noise_std = float(np.sqrt(max(noise_var, 1e-12)))
    mean_brightness = float(np.mean(0.5 * (G1 + G2)))
    mean_brightness = max(mean_brightness, 1e-4)

    alpha_simple = float(np.clip(noise_var / mean_brightness, 1e-6, 5e-2))
    sigma2_simple = float(
        np.clip(noise_var - alpha_simple * mean_brightness, 1e-10, 5e-3)
    )

    fallback_meta = {
        "reliable": False,
        "reason": "unknown",
        "n_patches": 0,
        "r_squared": 0.0,
        "alpha_simple": alpha_simple,
        "sigma2_simple": sigma2_simple,
        "noise_std": noise_std,
        "diff_std": float(np.std(diff_img)),
        "g1_max": float(G1.max()) if G1.size else 0.0,
    }

    g1_max = float(G1.max())
    signal_img = 0.5 * (G1 + G2)
    bright_p95 = float(np.percentile(signal_img, 95))
    
    if g1_max < 0.08 and bright_p95 < 0.03:
        meta = {
            **fallback_meta,

            # 这里不要再当成失败
            "reliable": True,
            "reason": "dark_scene_global",

            # 表示已经成功使用 global estimator
            "r_squared": 1.0,

            # 返回最终采用参数
            "alpha": alpha_simple,
            "sigma2": sigma2_simple,

            # debug
            "g1_max": g1_max,
            "bright_p95": bright_p95,

            # 标记来源
            "used_global_estimator": True,
        }

        return alpha_simple, sigma2_simple, meta

    if H_crop < patch_size or W_crop < patch_size:
        fallback_meta["reason"] = "image_too_small"
        return alpha_simple, sigma2_simple, fallback_meta

    def to_patches(img):
        ph = H_crop // patch_size
        pw = W_crop // patch_size
        return (
            img.reshape(ph, patch_size, pw, patch_size)
            .transpose(0, 2, 1, 3)
            .reshape(-1, patch_size * patch_size)
        )

    mean_img = 0.5 * (G1 + G2)
    mean_patches = to_patches(mean_img)
    diff_patches = to_patches(diff_img)

    local_mean = mean_patches.mean(axis=1)
    local_flatness = mean_patches.var(axis=1, ddof=1)
    local_var = diff_patches.var(axis=1, ddof=1) / 2.0

    flat_threshold = np.percentile(local_flatness, flat_percentile)
    flat_mask = local_flatness < flat_threshold
    p_low = np.percentile(local_mean, brightness_percentile[0])
    p_high = np.percentile(local_mean, brightness_percentile[1])
    brightness_mask = local_mean > p_low
    valid_mask = flat_mask & brightness_mask

    mean_arr = local_mean[valid_mask]
    var_arr = local_var[valid_mask]
    flatness_arr = local_flatness[valid_mask]
    fallback_meta["n_patches"] = int(len(mean_arr))

    if len(mean_arr) < min_valid_patches:
        fallback_meta["reason"] = "too_few_patches"
        return alpha_simple, sigma2_simple, fallback_meta

    if float(mean_arr.max() - mean_arr.min()) < min_mean_range:
        fallback_meta["reason"] = "mean_range_too_narrow"
        fallback_meta["mean_range"] = [float(mean_arr.min()), float(mean_arr.max())]
        return alpha_simple, sigma2_simple, fallback_meta

    bins = np.linspace(mean_arr.min(), mean_arr.max(), n_bins + 1)

    v_low = np.percentile(var_arr, 0.5)
    v_high = np.percentile(var_arr, 99.5)
    f_high = np.percentile(flatness_arr, 95)
    keep = (var_arr >= v_low) & (var_arr <= v_high) & (flatness_arr <= f_high)

    mean_arr = mean_arr[keep]
    var_arr = var_arr[keep]
    fallback_meta["n_patches"] = int(len(mean_arr))

    if len(mean_arr) < min_valid_patches:
        fallback_meta["reason"] = "too_few_after_clipping"
        return alpha_simple, sigma2_simple, fallback_meta

    if float(mean_arr.max() - mean_arr.min()) < 1e-8:
        fallback_meta["reason"] = "degenerate_mean_range"
        return alpha_simple, sigma2_simple, fallback_meta

    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(mean_arr, quantiles)
    bins = np.unique(bins)

    # 如果去重后 bins 不足 2 个，直接回退
    if len(bins) < 2:
        fallback_meta["reason"] = "bins_collapsed"
        return alpha_simple, sigma2_simple, fallback_meta

    x_bins, y_bins, w_bins = [], [], []
    n_actual_bins = len(bins) - 1   # 用实际的 bin 数量

    for bin_idx in range(n_actual_bins):
        if bin_idx == n_actual_bins - 1:
            mask = (mean_arr >= bins[bin_idx]) & (mean_arr <= bins[bin_idx + 1])
        else:
            mask = (mean_arr >= bins[bin_idx]) & (mean_arr < bins[bin_idx + 1])
        count = int(mask.sum())
        if count < 3:
            continue
        x_bins.append(float(np.median(mean_arr[mask])))
        y_bins.append(float(np.mean(var_arr[mask])))
        w_bins.append(float(count))

    # 后面 len(x_bins) < 5 的判断也要对应降低门槛
    if len(x_bins) < min(3, n_actual_bins):
        fallback_meta["reason"] = "too_few_bins"
        fallback_meta["mean_range"] = [float(mean_arr.min()), float(mean_arr.max())]
        fallback_meta["n_bins_used"] = int(len(x_bins))
        return alpha_simple, sigma2_simple, fallback_meta

    x = np.array(x_bins, dtype=np.float64)
    y = np.array(y_bins, dtype=np.float64)
    weights = np.array(w_bins, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)

    x_mean = np.sum(weights * x)
    y_mean = np.sum(weights * y)
    denom = np.sum(weights * (x - x_mean) ** 2)

    if denom < 1e-12:
        fallback_meta["reason"] = "degenerate_x"
        return alpha_simple, sigma2_simple, fallback_meta

    slope = np.sum(weights * (x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean

    y_fit = slope * x + intercept
    ss_res = np.sum(weights * (y - y_fit) ** 2)
    ss_tot = np.sum(weights * (y - y_mean) ** 2)
    r_squared = 0.0 if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)

    reliable = True
    reason = "ok"
    if slope <= 0:
        reliable = False
        reason = "negative_slope"
    elif intercept < -1e-6:
        reliable = False
        reason = "negative_intercept"
    elif r_squared < 0.3:
        reliable = False
        reason = "low_r2"

    meta = {
        "reliable": reliable,
        "reason": reason,
        "n_patches": int(len(mean_arr)),
        "r_squared": r_squared,
        "slope_raw": float(slope),
        "intercept_raw": float(intercept),
        "alpha": max(float(slope), 1e-6),
        "sigma2": max(float(intercept), 1e-8),
        "alpha_simple": alpha_simple,
        "sigma2_simple": sigma2_simple,
        "noise_std": noise_std,
        "mean_range": [float(mean_arr.min()), float(mean_arr.max())],
        "flat_threshold": float(flat_threshold),
        "diff_std": float(np.std(diff_img)),
        "g1_max": float(G1.max()),
        "n_bins_used": int(len(x_bins)),
        "lower_q": float(lower_q),
    }

    if not reliable:
        meta["alpha"] = alpha_simple
        meta["sigma2"] = sigma2_simple
        return alpha_simple, sigma2_simple, meta

    return meta["alpha"], meta["sigma2"], meta