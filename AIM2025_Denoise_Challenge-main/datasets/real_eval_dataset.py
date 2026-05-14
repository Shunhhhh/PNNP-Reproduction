import os
import torch
import rawpy
import numpy as np
from glob import glob


class PairedEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        benchmark_dir,
        camera_model,
        camera_config,
        inp_clip_low,
        inp_clip_high,
        iso_list=[800, 1600, 3200],
        load_gt=False,
    ):
        """
        Args:
            data_dir (str): path to data
            camera_model (str): camera model name
            camera_config (yaml.dict): camera config loaded as yaml file
            inp_clip_low (bool): clip low or not, this only applies to the input noisy image, and gt will always clip to [0, 1]
            inp_clip_high (bool): clip high or not, this only applies to the input noisy image, and gt will always clip to [0, 1]
            iso_list (list): available ISOs for the indoor paired data is [800, 1600, 3200]
            load_gt (bool): load gt or not, since gt is not released, this should be set to false, which will return a dummy all-zero tensor
        """
        super().__init__()
        ## common setup
        self.h_start, self.w_start, self.h_end, self.w_end = camera_config["valid_roi"]
        self.ccm = np.array(camera_config["ccm"]).reshape(3, 3)
        self.camera_model = camera_model
        self.clip_low = 0 if inp_clip_low else float("-inf")
        self.clip_high = 1 if inp_clip_high else float("inf")
        self.eval_iso = iso_list
        self.load_gt = load_gt
        suffix = 'CR2' if 'canon' in camera_model else 'ARW'

        ## load dark shading, which already subtracted black level
        self.dark_shadings = {
            iso: np.load(os.path.join(benchmark_dir, camera_model, f"calib_res/dark_shading_iso{iso}.npy"))
            for iso in self.eval_iso
        }

        ## prepare data
        self.all_data_info = []
        for noisy_dir in glob(os.path.join(benchmark_dir, camera_model, f"test_data/paired_input/*.{suffix}")):
            scene_id, iso, dgain = os.path.basename(noisy_dir).split(".")[0].split("_")
            curr_data = {
                "iso": int(iso[3:]),
                "dgain": float(dgain[5:]),
                "clean_dir": (
                    os.path.join(benchmark_dir, camera_model, f"test_data/paired_gt/{scene_id}_{iso}_gt.{suffix}")
                    if self.load_gt
                    else ""
                ),
                "noisy_dir": noisy_dir,
            }
            self.all_data_info.append(curr_data)

    def __len__(self):
        return len(self.all_data_info)

    def pack_raw(self, img, wl, bl, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - bl) / (wl - bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def __getitem__(self, idx):
        data_info = self.all_data_info[idx]

        ## load noisy image
        rf_noisy = rawpy.imread(data_info["noisy_dir"])
        wb = np.array(rf_noisy.camera_whitebalance)
        wb = wb / wb[1]
        wl, bl = float(rf_noisy.white_level), np.mean(rf_noisy.black_level_per_channel)
        noisy = np.array(rf_noisy.raw_image).astype(np.float32)
        noisy = noisy[self.h_start : self.h_end, self.w_start : self.w_end]

        ## subtract dark shading
        ds = self.dark_shadings[data_info["iso"]]
        ds = ds[self.h_start : self.h_end, self.w_start : self.w_end]
        noisy = noisy - ds

        ## pack to 4-chans and align brightness
        noisy = self.pack_raw(noisy, wl=wl, bl=bl, norm=True, clip=False)  ## [h, w, c]
        noisy = torch.FloatTensor(noisy).permute(2, 0, 1)
        noisy *= data_info["dgain"]

        ## load clean gt img
        if self.load_gt:
            rf_clean = rawpy.imread(data_info["clean_dir"])
            clean = np.array(rf_clean.raw_image).astype(np.float32)
            clean = clean[self.h_start : self.h_end, self.w_start : self.w_end]
            clean = self.pack_raw(clean, wl=wl, bl=bl, norm=True, clip=True)
            clean = torch.FloatTensor(clean).permute(2, 0, 1)
        else:
            clean = torch.zeros((1,))

        data = {
            "ccm": torch.FloatTensor(self.ccm),
            "wb": torch.FloatTensor(wb),
            "noisy": torch.clamp(noisy, self.clip_low, self.clip_high),
            "clean": torch.clamp(clean, 0, 1),
            "img_name": os.path.basename(data_info["noisy_dir"]).split(".")[0],
            "camera_model": self.camera_model,
        }
        data.update(data_info)
        return data


class InTheWildEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        benchmark_dir,
        camera_model,
        camera_config,
        inp_clip_low,
        inp_clip_high,
        iso_list=[800, 1250, 1600, 3200, 6400],
    ):
        """
        Args:
            data_dir (str): path to data
            camera_model (str): camera model name
            camera_config (yaml.dict): camera config loaded as yaml file
            inp_clip_low (bool): clip low or not, this only applies to the input noisy image, and gt will always clip to [0, 1]
            inp_clip_high (bool): clip high or not, this only applies to the input noisy image, and gt will always clip to [0, 1]
            iso_list (list): available ISOs for the indoor paired data is [800, 1600, 3200]
        """
        super().__init__()
        ## common setup
        self.h_start, self.w_start, self.h_end, self.w_end = camera_config["valid_roi"]
        self.ccm = np.array(camera_config["ccm"]).reshape(3, 3)
        self.clip_low = 0 if inp_clip_low else float("-inf")
        self.clip_high = 1 if inp_clip_high else float("inf")
        self.camera_model = camera_model

        ## load dark shading, which already subtracted black level
        self.dark_shadings = {
            iso: np.load(os.path.join(benchmark_dir, camera_model, f"calib_res/dark_shading_iso{iso}.npy"))
            for iso in iso_list
        }

        ## prepare data
        self.all_data_dirs = glob(
            os.path.join(benchmark_dir, camera_model, f"test_data/in_the_wild/*.{camera_config['suffix']}")
        )

    def __len__(self):
        return len(self.all_data_dirs)

    def pack_raw(self, img, wl, bl, norm=False, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - bl) / (wl - bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def __getitem__(self, idx):
        ## load metadata
        data_dir = self.all_data_dirs[idx]
        img_name = os.path.basename(data_dir).split(".")[0]
        _, iso, dgain = img_name.split("_")
        iso, dgain = int(iso[3:]), float(dgain[5:])

        ## load images
        rf_noisy = rawpy.imread(data_dir)
        wb = np.array(rf_noisy.camera_whitebalance)
        wb = wb / wb[1]
        wl, bl = float(rf_noisy.white_level), np.mean(rf_noisy.black_level_per_channel)
        noisy = np.array(rf_noisy.raw_image).astype(np.float32)
        noisy = noisy[self.h_start : self.h_end, self.w_start : self.w_end]

        ## subtract dark shading
        ds = self.dark_shadings[iso]
        ds = ds[self.h_start : self.h_end, self.w_start : self.w_end]
        noisy = noisy - ds

        ## pack to 4-chans and align brightness
        noisy = self.pack_raw(noisy, wl=wl, bl=bl, norm=True, clip=False)  ## [h, w, c]
        noisy = torch.FloatTensor(noisy).permute(2, 0, 1)
        noisy *= dgain

        data = {
            "ccm": torch.FloatTensor(self.ccm),
            "wb": torch.FloatTensor(wb),
            "noisy": torch.clamp(noisy, self.clip_low, self.clip_high),
            "dgain": torch.ones(1) * dgain,
            "iso": torch.ones(1) * iso,
            "img_name": img_name,
            "camera_model": self.camera_model,
        }
        return data

