import os
import argparse
import random
import imageio
import torch
import yaml
import pyiqa
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F

from datasets.real_eval_dataset import PairedEvalDataset
from models.ELD_models import UNetSeeInDark

from utils.utils import *


def build_model(args):
    model = UNetSeeInDark().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_and_save_for_submission(model, args):
    ## make directory for saving
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    ## load camera config file
    with open(args.camera_config_dir, "r") as f:
        cam_cfg = yaml.load(f, Loader=yaml.FullLoader)

    ## build dataloader, inference, and save, done in a camera-wise manner
    for cam_model in args.camera_models:
        os.makedirs(os.path.join(args.save_dir, cam_model))

        ## --- data loader
        eval_set = PairedEvalDataset(
            benchmark_dir=args.benchmark_dir,
            camera_model=cam_model,
            camera_config=cam_cfg[cam_model],
            inp_clip_low=False,
            inp_clip_high=True,
            iso_list=[800, 1600, 3200],
            load_gt=False,
        )
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=2)

        ## --- inference
        for _, data in enumerate(tqdm(eval_loader)):
            noisy = data["noisy"].to(args.device)
            img_name = data["img_name"][0]
            denoised = model(noisy)  ## if the output is not [1, 4, H, W], please change it to [1, 4, H, W]

            # ## processing for saving
            denoised = denoised.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            denoised = np.clip(denoised, 0, 1)
            denoised = center_crop_numpy_img(denoised, args.eval_crop_size)
            denoised = np.uint16(denoised * 65535)
            np.save(os.path.join(args.save_dir, cam_model, f"{img_name}.npy"), denoised)


def main(args):
    model = build_model(args).to(args.device)
    infer_and_save_for_submission(model, args)


##--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## dir
    parser.add_argument("--camera_config_dir", type=str, default="./datasets/camera_config.yaml")
    parser.add_argument("--benchmark_dir", type=str, default="../../data/xml196414/SID/dev_phase_release")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/PNNP_ELD/sonyzve10m2/best.pth")
    parser.add_argument("--device", type=str, default="cuda")

    ## DO NOT change below setups
    parser.add_argument("--save_dir", type=str, default="./saved_res_for_submission")
    parser.add_argument("--camera_models", type=str, nargs="+", default=["canon70d", "sonya6700", "sonya7r4", "sonyzve10m2"])
    parser.add_argument("--eval_crop_size", type=int, default=512)
    _args = parser.parse_args()

    main(_args)
