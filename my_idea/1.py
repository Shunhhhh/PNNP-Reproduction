# # 临时诊断脚本，单独看这两个场景
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os
# sys.path.insert(
#     0,
#     os.path.join(os.path.dirname(__file__), "..")
# )
# from my_idea.tools import estimate_noise_params

# import numpy as np

# def make_synthetic_correct(alpha, sigma2, shape=(512, 512)):
#     # 用结构化信号，保证亮度跨度大
#     # 方法：用梯度图像，而不是随机均匀噪声
#     rows = np.linspace(0.02, 0.98, shape[0])
#     cols = np.linspace(0.02, 0.98, shape[1])
#     mu = np.outer(rows, np.ones(shape[1])) * 0.5 + \
#          np.outer(np.ones(shape[0]), cols) * 0.5
#     # mu 是一个从 0.02 到 0.98 的平滑梯度，patch间亮度差异大

#     def shoot(mu):
#         photons = np.random.poisson(mu / alpha).astype(np.float64)
#         gaussian = np.random.normal(0, np.sqrt(sigma2), shape)
#         return photons * alpha + gaussian

#     G1 = shoot(mu)
#     G2 = shoot(mu)
#     return G1, G2

# true_alpha  = 1e-3
# true_sigma2 = 8e-4

# G1, G2 = make_synthetic_correct(true_alpha, true_sigma2)
# est_alpha, est_sigma2, info = estimate_noise_params(
#     G1, G2,
#     patch_size=16,
#     n_bins=30,
#     flat_percentile=50,
#     brightness_percentile=(5, 95),
#     lower_q=0.3,
# )

# print(f"alpha:  真值={true_alpha:.4f}  估计={est_alpha:.4f}  误差={abs(est_alpha-true_alpha)/true_alpha*100:.1f}%")
# print(f"sigma2: 真值={true_sigma2:.6f}  估计={est_sigma2:.6f}")
# print(f"R²={info['r_squared']:.4f}  reliable={info['reliable']}")



# import scipy.io as sio

# mat = sio.loadmat('../../../data/xml196414/SIDD/SIDD_Medium/ValidationNoisyBlocksRaw.mat')

# data = mat['ValidationNoisyBlocksRaw']

# print(type(data))
# print(data.shape)
# print(data.dtype)

# print(data.min(), data.max())



import scipy.io as sio
import glob

# 随便找一个场景的 metadata 文件看结构
meta_path = "../../data/xml196414/SIDD/SIDD_Medium/SIDD_Medium_Raw/Data/0200_010_GP_01600_03200_5500_N/0200_METADATA_RAW_011.MAT"

meta = sio.loadmat(meta_path)
print(meta.keys())

# 看一下每个字段的内容
for k, v in meta.items():
    if not k.startswith("_"):
        print(f"{k}: {type(v)}, shape={getattr(v, 'shape', 'N/A')}, value={v}")

# from scipy.io import loadmat
# import numpy as np

# path = "../../data/xml196414/SIDD/SIDD_Benchmark/BenchmarkNoisyBlocksRaw.mat"
# # 1. 先看完整结构（包含嵌套 struct）
# data = loadmat(path, squeeze_me=True, struct_as_record=False)


# # 取出变量
# x = data['BenchmarkNoisyBlocksRaw']

# # 查看信息
# print('shape:', x.shape)
# print('dtype:', x.dtype)

# # 最大最小值
# print('min:', np.min(x))
# print('max:', np.max(x))

# blocks = mat['BenchmarkBlocks32']

# print(blocks)

# for k, v in mat.items():
#     if k.startswith("__"):
#         continue
#     print("key:", k)
#     print("type:", type(v))
#     if isinstance(v, np.ndarray):
#         print("shape:", v.shape)
#         print("dtype:", v.dtype)
#     print()