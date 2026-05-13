import os
import torch
import numpy as np
import rawpy
import random
from glob import glob
from torchvision import transforms


class SynthTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean_img_dir,
        benchmark_dir,
        camera_config,
        iso_list=[800, 1250, 1600, 3200, 6400],
        dgain_range=(10, 200),
        patch_size=512,
        inp_clip_low=False,
        inp_clip_high=True,
        n_crop_per_img=8,
    ):
        super().__init__()
        self.clip_low = 0 if inp_clip_low else float("-inf")
        self.clip_high = 1 if inp_clip_high else float("inf")
        self.psize = patch_size
        self.dgain_range = dgain_range
        self.iso_list = iso_list
        self.cam_cfg = camera_config
        self.n_crop_per_img = n_crop_per_img

        self.transforms = transforms.Compose(
            [
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
            ]
        )

        ## load clean images as gt & base for adding noise on
        self.clean_imgs = sorted(glob(os.path.join(clean_img_dir, "*.ARW")))

        ## load necessary calibration data
        self.sys_gain, self.dark_frame_dirs, self.dark_shadings = {}, {}, {}
        print(">>>>> Will synthesize training data for the following camera models: ", list(self.cam_cfg.keys()))
        for cam_model in self.cam_cfg.keys():
            ## load flat-field calibrated system gain for shot noise synthesis
            curr_calib_gain = np.load(os.path.join(benchmark_dir, cam_model, f"calib_res/sys_gain.npz"))
            for iso in self.iso_list:
                self.sys_gain[f"{cam_model}_iso{iso}"] = curr_calib_gain[f"iso{iso}"]

            ## load dark frames for signal-independent noise synthesis
            for iso in self.iso_list:
                curr_dark_frame_dirs = sorted(glob(os.path.join(benchmark_dir, cam_model, f"dark_frame/iso{iso}/*")))
                self.dark_frame_dirs[f"{cam_model}_iso{iso}"] = curr_dark_frame_dirs

            ## load dark shadings, which already subtracted black level
            for iso in self.iso_list:
                curr_dark_shadings = np.load(
                    os.path.join(benchmark_dir, cam_model, f"calib_res/dark_shading_iso{iso}.npy")
                )
                self.dark_shadings[f"{cam_model}_iso{iso}"] = curr_dark_shadings

    def __len__(self):
        return len(self.clean_imgs)

    def pack_raw(self, img, wl, bl, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - bl) / (wl - bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def random_crop(self, img, psize=512, n_crop=8):
        res = []
        for i in range(n_crop):
            hs = np.random.randint(0, img.shape[0] - psize + 1)
            ws = np.random.randint(0, img.shape[1] - psize + 1)
            res.append(img[hs : hs + psize, ws : ws + psize, :])
        return np.stack(res, axis=0)  # [n_crop, psize, psize, c]

    def noise_synthesis(self, clean, dark_frame, dgain, sys_gain, wl, bl):
        img = np.clip(clean, 0, 1) * (wl - bl) / dgain  # 从0~1恢复回真实RAW亮度
        short_noisy = np.random.poisson(img / sys_gain) * sys_gain - img # 泊松噪声
        img = img + short_noisy + dark_frame ######## 
        img = img / (wl - bl) * dgain # 归一化
        return img

    def __getitem__(self, idx):
        ## randomly choose an ISO and a camera model
        iso = random.choice(self.iso_list)
        cam_model = random.choice(list(self.cam_cfg.keys()))
        h_start, w_start, h_end, w_end = self.cam_cfg[cam_model]["valid_roi"]  ## 有效成像区域

        ##---- STEP 1: load a clean image, pack to rggb, and crop
        # packraw（norm） ,crop
        clean_raw_file = rawpy.imread(self.clean_imgs[idx])
        clean_wl, clean_bl = float(clean_raw_file.white_level), np.mean(clean_raw_file.black_level_per_channel)
        clean = np.array(clean_raw_file.raw_image_visible).astype(np.float32)
        clean = self.pack_raw(clean, wl=clean_wl, bl=clean_bl, norm=True, clip=True)  # [h, w, c]

        clean_crops = self.random_crop(clean, psize=self.psize, n_crop=self.n_crop_per_img)  ## [n, h, w, c]
        clean_crops = torch.FloatTensor(clean_crops).permute(0, 3, 1, 2)  # [n, c, h, w]
        clean_crops = self.transforms(clean_crops)  # [n, c, h, w]

        ##---- STEP 2: load a random noise frame, subtract bl and dark shading to get signal-independent noise, and crop
        # darkframe - darkshading - bl ,crop, packraw(no norm)
        dark_raw_file = rawpy.imread(random.choice(self.dark_frame_dirs[f"{cam_model}_iso{iso}"]))
        dark_wl, dark_bl = float(dark_raw_file.white_level), np.mean(dark_raw_file.black_level_per_channel)
        dark_frame = np.array(dark_raw_file.raw_image).astype(np.float32)

        dark_frame = (
            dark_frame - self.dark_shadings[f"{cam_model}_iso{iso}"] - dark_bl
        )  # pre-subtract and no need to do so on noise synthsis

        dark_frame = dark_frame[h_start:h_end, w_start:w_end]
        dark_frame = self.pack_raw(dark_frame, wl=dark_wl, bl=dark_bl, norm=False, clip=False)  # [h, w, c]

        ##---- STEP 3: add shot noise
        noisy_crops = torch.empty_like(clean_crops)
        all_dgain = torch.empty(self.n_crop_per_img)
        for i in range(self.n_crop_per_img):
            curr_dgain = random.choice(self.dgain_range)
            all_dgain[i] = curr_dgain # 给每个 crop 随机选择一个数字增益
            curr_dark_frame = self.random_crop(dark_frame, psize=self.psize, n_crop=1).squeeze(0)
            curr_noisy = self.noise_synthesis(
                clean=clean_crops[i].permute(1, 2, 0).numpy(),
                dark_frame=curr_dark_frame,
                dgain=curr_dgain,
                sys_gain=self.sys_gain[f"{cam_model}_iso{iso}"],
                wl=dark_wl,
                bl=dark_bl,
            )  # [h, w, c]
            noisy_crops[i] = torch.FloatTensor(curr_noisy).permute(2, 0, 1)

        data = {
            "cam_model": cam_model,
            "dgain": all_dgain,
            "iso": torch.ones((1,)) * iso,
            "noisy": torch.clamp(noisy_crops, self.clip_low, self.clip_high),
            "clean": torch.clamp(clean_crops, 0, 1),
        }

        return data

