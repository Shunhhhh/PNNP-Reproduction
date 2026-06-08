import numpy as np
data = np.load("../../data/xml196414/SID/dev_phase_release/sonyzve10m2/calib_res/sys_gain.npz")
print(list(data.keys()))
for k, v in data.items():
    print(k, v.shape, v.dtype, v.flat[:5])