import os
import numpy as np
import torch
from tqdm import tqdm

from datasets.synth_train_dataset import SynthTrainDataset

# =========================================================
# camera config
# =========================================================

camera_config = {
    "sonyzve10m2": {
        "valid_roi": [0, 0, 4128, 6192]
    }
}

# =========================================================
# dataset
# =========================================================

dataset = SynthTrainDataset(

    clean_img_dir=
    "../../data/xml196414/SID/Sony/long",

    benchmark_dir=
    "../../data/xml196414/SID/dev_phase_release",

    camera_config=
    camera_config,

    iso_list=[800, 1600, 3200, 6400],

    patch_size=512,

    n_crop_per_img=1
)

# =========================================================
# output dir
# =========================================================

save_dir = "./synthetic_noisy"

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# generate 50 noisy images
# =========================================================

count = 0

for idx in tqdm(range(len(dataset))):

    data = dataset[idx]

    noisy = data["noisy"]      # [1,4,H,W]
    clean = data["clean"]

    iso = int(data["iso"].item())

    noisy = noisy[0].numpy()
    clean = clean[0].numpy()

    np.save(
        os.path.join(
            save_dir,
            f"noisy_{count:03d}_iso{iso}.npy"
        ),
        noisy
    )

    np.save(
        os.path.join(
            save_dir,
            f"clean_{count:03d}_iso{iso}.npy"
        ),
        clean
    )

    count += 1

    if count >= 50:
        break

print("Done.")