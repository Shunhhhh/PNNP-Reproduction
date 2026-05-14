import os
import glob
import rawpy
import numpy as np

input_dir = "../../data/xml196414/SID/dev_phase_release/sonyzve10m2/dark_frame"
output_dir = input_dir + "_npz"

os.makedirs(output_dir, exist_ok=True)

paths = glob.glob(os.path.join(input_dir, "**/*.ARW"), recursive=True)

for path in paths:
    try:
        with rawpy.imread(path) as raw:
            # ✅ 不归一化，保留 ADU 原始值
            visible = raw.raw_image_visible.astype(np.float32)

            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
            print("black:", black)
            print("white:", white)

            top    = raw.sizes.top_margin
            left   = raw.sizes.left_margin
            height = raw.sizes.height
            width  = raw.sizes.width

        rel_path  = os.path.relpath(path, input_dir)
        save_path = os.path.join(output_dir, rel_path).replace(".ARW", ".npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.savez(
            save_path,
            raw=visible,          # ADU 原始值，未归一化
            black_level=black,
            white_level=white,
            top_margin=top,
            left_margin=left,
            height=height,
            width=width,
        )

    except Exception as e:
        print(f"Error processing {path}: {e}")

print("Done!")