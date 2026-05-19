## AIM2025 Raw Denoising + PNNP

### PNNP复现
1. 运行train_syn_pnnp.py，训练加噪，训练好的checkpoints按照不同ISO存储在./checkpoints/PNNP_noise，dist_iso{iso_value}.png可视化不同iso下预测噪声与gt对比图
2. datasets/pnnnp_train_dataset定义合成的加噪数据集
3. 运行train_denoise_pnnpeld.py，训练去噪，结果存在./checkpoints/PNNP_ELD
4. 运行infer_pnnp_base.py验证，在官方加噪方法得到的合成数据集上，使用PNNP训练权重去噪