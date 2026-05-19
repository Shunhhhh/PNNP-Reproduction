## AIM2025 Raw Denoising + PNNP

### PNNP复现
1. 运行`train_syn_pnnp.py`，训练加噪，训练好的checkpoints按照不同ISO存储在`./checkpoints/PNNP_noise`，`dist_iso{iso_value}.png`可视化不同iso下预测噪声与gt对比图
2. `datasets/pnnnp_train_dataset`定义合成的加噪数据集
3. 运行`train_denoise_pnnpeld.py`，训练去噪，结果存在`./checkpoints/PNNP_ELD`
4. 运行`infer_pnnp_base.py`验证，在官方加噪方法得到的合成数据集上，使用PNNP训练权重去噪

### AIM2025 Raw Denoising
1. 在官方加噪方法上，使用ELD去噪：运行
```
torchrun --nproc_per_node=2 train_denoise_eld.py
```
2. 在官方加噪方法上，使用NAFNet去噪：运行
```
torchrun --nproc_per_node=2 train_denoise_naf.py
```