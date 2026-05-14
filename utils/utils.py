import os
import torch
import numpy as np
import time
import random
import cv2
import shutil
import pyiqa

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def simple_ISP(raw, wb, ccm, gamma):
    raw = np.uint16(np.clip(raw, 0, 1) * 65535)
    ## demosicing
    rgb = cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB_EA)
    rgb = np.float32(rgb) / 65535
    ## wb
    rgb = rgb * wb[None, None, :3]
    rgb = np.clip(rgb, 0, 1)
    ## ccm
    rgb = rgb @ ccm.T
    rgb = np.clip(rgb, 0, 1)
    ## gamma
    rgb = rgb ** (1 / gamma)
    rgb = np.clip(rgb, 0, 1)
    return np.uint8(rgb * 255)


def center_crop_batch(img, psize):
    b, c, h, w = img.shape
    crop_h, crop_w = psize, psize
    crop_top = int(round((h - crop_h) / 2.0))
    crop_left = int(round((w - crop_w) / 2.0))
    return img[:, :, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]


def center_crop_numpy_img(img, psize):
    h, w = img.shape[:2]
    crop_h, crop_w = psize, psize
    crop_top = int(round((h - crop_h) / 2.0))
    crop_left = int(round((w - crop_w) / 2.0))
    return img[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w, ...]


def psnr_ssim_metric_torch(pred, gt):
    ## psnr and ssim
    all_psnr, all_ssim = [], []
    for i in range(gt.shape[0]):
        img1_i = gt[i].detach().cpu().permute(1, 2, 0).numpy()
        img2_i = pred[i].detach().cpu().permute(1, 2, 0).numpy()
        img1_i, img2_i = np.clip(img1_i, 0, 1), np.clip(img2_i, 0, 1)
        psnr = compare_psnr(img1_i, img2_i, data_range=1.0)
        ssim = compare_ssim(img1_i, img2_i, data_range=1.0, channel_axis=-1)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    return_dict = {"psnr": np.mean(all_psnr), "ssim": np.mean(all_ssim)}
    return return_dict


def vis_rggb(raw, wb=None, gamma=2.2, format="rggb", uint8=True):
    """input is [H, W, 4] numpy 0-1 packed_raw, wb and ccm comes in rgbg format"""
    raw = np.clip(raw, 0, 1)
    if format == "rggb":
        raw = np.stack([raw[:, :, 0], raw[:, :, 1], raw[:, :, 3], raw[:, :, 2]], axis=-1)
    elif format == "rgbg":
        raw = raw
    ## white-balance
    res = raw * wb[None, None, :]
    res = np.clip(res, 0, 1)
    ## gamma
    res = res ** (1 / gamma)
    res = np.clip(res, 0, 1)
    return np.uint8(res * 255) if uint8 else res


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type="mean"):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is None:
            fmtstr = ""
        elif self.summary_type == "mean":
            fmtstr = "{name} {avg:.5e}"
        elif self.summary_type == "sum":
            fmtstr = "{name} {sum:.5e}"
        elif self.summary_type == "count":
            fmtstr = "{name} {count:.5e}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_and_remake_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def log(string, log=None, str=False, end="\n", notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log, "a+") as f:
            f.write(log_string + "\n")
    else:
        pass
    if str:
        return string + end


def collate_fn_replace_corrupted(batch, dataset):
    original_batch_len = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        batch.extend([dataset[random.randint(0, len(dataset) - 1)] for _ in range(diff)])
        return collate_fn_replace_corrupted(batch, dataset)
    return torch.utils.data.dataloader.default_collate(batch)


def tensor_dim5to4(tensor):
    batchsize, crops, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize * crops, c, h, w)
    return tensor
