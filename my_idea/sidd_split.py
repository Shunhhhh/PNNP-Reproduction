"""
SIDD 数据集划分脚本（改进版）
- 按场景ID分组，同一场景的所有ISO/曝光变体保持在同一split
- 在组内按相机型号和ISO做分层采样，保证各split分布均衡
"""

import os
import glob
import random
import argparse
from collections import defaultdict


def parse_scene_meta(scene_name):
    """
    解析场景目录名
    格式：{idx}_{instance}_{camera}_{ISO}_{shutter}_{brightness}_{light}_N
    例如：0001_001_S6_00100_00060_3200_L_N
    """
    parts = scene_name.split("_")
    return {
        "scene_id":  parts[0] if len(parts) > 0 else "unknown",
        "camera":    parts[2] if len(parts) > 2 else "unknown",
        "iso":       parts[3] if len(parts) > 3 else "unknown",
        "raw_name":  scene_name,
    }


def split_sidd(sidd_dir, out_dir, train_ratio, val_ratio, test_ratio, seed):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 找所有场景目录
    all_dirs = sorted([
        os.path.basename(d)
        for d in glob.glob(os.path.join(sidd_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d).endswith("_N")
    ])

    if not all_dirs:
        raise FileNotFoundError(f"在 {sidd_dir} 下没有找到以 _N 结尾的场景目录")

    print(f"Found {len(all_dirs)} scene dirs in {sidd_dir}")

    # ── 按 scene_id 分组 ──────────────────────────────────────────
    # 同一 scene_id 的所有变体（不同ISO、曝光）必须进同一个split
    groups = defaultdict(list)
    for d in all_dirs:
        meta = parse_scene_meta(d)
        groups[meta["scene_id"]].append(d)

    print(f"Unique scene IDs: {len(groups)}")
    for sid, dirs in sorted(groups.items())[:5]:
        print(f"  scene {sid}: {dirs}")
    print("  ...")

    # ── 按相机分层，保证各split相机分布均衡 ──────────────────────
    # 先按相机分桶
    camera_buckets = defaultdict(list)
    for scene_id, dirs in groups.items():
        camera = parse_scene_meta(dirs[0])["camera"]
        camera_buckets[camera].append(scene_id)

    random.seed(seed)

    train_ids, val_ids, test_ids = [], [], []

    for camera, scene_ids in camera_buckets.items():
        ids = sorted(scene_ids)
        random.shuffle(ids)

        n = len(ids)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(0, int(n * val_ratio))
        # test 拿剩余，保证不丢场景
        n_test  = n - n_train - n_val

        # 如果相机场景太少（比如只有1-2个），优先保证train有数据
        if n_test < 0:
            n_val  = max(0, n_val + n_test)
            n_test = 0

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

        print(f"  Camera {camera}: total={n} "
              f"train={n_train} val={n_val} test={n_test}")

    # scene_id → 所有目录
    train_dirs = [d for sid in train_ids for d in groups[sid]]
    val_dirs   = [d for sid in val_ids   for d in groups[sid]]
    test_dirs  = [d for sid in test_ids  for d in groups[sid]]

    print(f"\nFinal split (dirs): "
          f"train={len(train_dirs)} val={len(val_dirs)} test={len(test_dirs)}")

    # ── 写文件 ────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    def write_list(filename, dir_list, split_name):
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            for d in sorted(dir_list):
                f.write(os.path.join(sidd_dir, d) + "\n")
        print(f"  [{split_name:5s}] {len(dir_list):4d} dirs → {path}")

    write_list("train.txt", train_dirs, "train")
    write_list("val.txt",   val_dirs,   "val")
    write_list("test.txt",  test_dirs,  "test")

    # ── 验证：scene_id 没有跨 split ───────────────────────────────
    train_scene_ids = set(train_ids)
    val_scene_ids   = set(val_ids)
    test_scene_ids  = set(test_ids)

    assert len(train_scene_ids & val_scene_ids)  == 0, "train/val 有重叠 scene_id！"
    assert len(train_scene_ids & test_scene_ids) == 0, "train/test 有重叠 scene_id！"
    assert len(val_scene_ids   & test_scene_ids) == 0, "val/test 有重叠 scene_id！"
    print("\n✓ 验证通过：所有 scene_id 严格不跨 split")

    # ── 打印分布统计 ───────────────────────────────────────────────
    print("\n── 各 split 分布统计 ──")
    for split_name, dir_list in [
        ("train", train_dirs),
        ("val",   val_dirs),
        ("test",  test_dirs),
    ]:
        cam_count = defaultdict(int)
        iso_count = defaultdict(int)
        for d in dir_list:
            meta = parse_scene_meta(d)
            cam_count[meta["camera"]] += 1
            iso_count[meta["iso"]]    += 1

        cam_str = "  ".join(f"{k}×{v}" for k, v in sorted(cam_count.items()))
        iso_str = "  ".join(f"ISO{k}×{v}" for k, v in sorted(iso_count.items()))
        print(f"  [{split_name:5s}] 相机: {cam_str}")
        print(f"         ISO:  {iso_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidd_dir",     type=str,   default="/data/xml196414/SIDD/SIDD_Small_Raw_Only/Data")
    parser.add_argument("--out_dir",      type=str,   default="./splits")
    parser.add_argument("--train_ratio",  type=float, default=0.8)
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--test_ratio",   type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    split_sidd(
        sidd_dir=args.sidd_dir,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )