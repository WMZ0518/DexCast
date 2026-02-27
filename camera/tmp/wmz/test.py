import os
import re
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from foundationpose_gdsam_tracker import FoundationPoseGDSAMTracker


def _natural_key(p: str):
    stem = Path(p).stem
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else stem


def find_rgb_depth_pairs(scene_dir: str):
    scene_dir = str(Path(scene_dir).resolve())

    # RGB 候选目录
    rgb_dirs = [
        os.path.join(scene_dir, "rgb"),
        os.path.join(scene_dir, "color"),
        os.path.join(scene_dir, "images"),
        os.path.join(scene_dir, "imgs"),
    ]
    rgb_dir = next((d for d in rgb_dirs if os.path.isdir(d)), None)
    if rgb_dir is None:
        raise FileNotFoundError(f"找不到 RGB 目录（尝试过 {rgb_dirs}）")

    # Depth 候选目录
    depth_dirs = [
        os.path.join(scene_dir, "depth"),
        os.path.join(scene_dir, "depths"),
    ]
    depth_dir = next((d for d in depth_dirs if os.path.isdir(d)), None)
    if depth_dir is None:
        raise FileNotFoundError(f"找不到 Depth 目录（尝试过 {depth_dirs}）")

    rgb_files = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        rgb_files += glob.glob(os.path.join(rgb_dir, f"*.{ext}"))
    rgb_files = sorted(rgb_files, key=_natural_key)
    if len(rgb_files) == 0:
        raise FileNotFoundError(f"RGB 目录为空：{rgb_dir}")

    depth_files = []
    for ext in ("png", "npy", "exr", "tiff", "tif"):
        depth_files += glob.glob(os.path.join(depth_dir, f"*.{ext}"))
    depth_files = sorted(depth_files, key=_natural_key)
    if len(depth_files) == 0:
        raise FileNotFoundError(f"Depth 目录为空：{depth_dir}")

    # 尽量按文件名 stem 对齐（更稳）
    depth_map = {Path(p).stem: p for p in depth_files}
    pairs = []
    missing = 0
    for rp in rgb_files:
        stem = Path(rp).stem
        if stem in depth_map:
            pairs.append((rp, depth_map[stem]))
        else:
            missing += 1

    if len(pairs) == 0:
        # fallback：直接按顺序截断配对
        n = min(len(rgb_files), len(depth_files))
        pairs = list(zip(rgb_files[:n], depth_files[:n]))
        print("[WARN] stem 无法对齐，已改为按顺序配对。")

    if missing > 0:
        print(f"[WARN] 有 {missing} 张 RGB 没找到同 stem 的 depth，已跳过。")

    return pairs


def load_K(scene_dir: str) -> np.ndarray:
    scene_dir = str(Path(scene_dir).resolve())

    # 1) npy
    npy_candidates = [
        os.path.join(scene_dir, "intrinsics.npy"),
        os.path.join(scene_dir, "K.npy"),
    ]
    for p in npy_candidates:
        if os.path.exists(p):
            K = np.load(p)
            K = np.asarray(K, dtype=np.float32)
            if K.shape == (3, 3):
                return K

    # 2) txt
    txt_candidates = [
        os.path.join(scene_dir, "cam_K.txt"),
        os.path.join(scene_dir, "K.txt"),
        os.path.join(scene_dir, "intrinsics.txt"),
        os.path.join(scene_dir, "camera_K.txt"),
    ]
    for p in txt_candidates:
        if os.path.exists(p):
            nums = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    nums += re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            vals = [float(x) for x in nums]
            if len(vals) >= 9:
                K = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
                return K
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
                K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float32)
                return K

    # 3) json
    json_candidates = [
        os.path.join(scene_dir, "intrinsics.json"),
        os.path.join(scene_dir, "meta.json"),
        os.path.join(scene_dir, "camera.json"),
    ]
    for p in json_candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 常见键
            for key in ("K", "camera_matrix", "intrinsic_matrix", "intrinsics"):
                if key in data:
                    arr = np.asarray(data[key], dtype=np.float32)
                    if arr.shape == (3, 3):
                        return arr
                    if arr.size >= 9:
                        return arr.reshape(3, 3)

            # fx/fy/cx/cy
            if all(k in data for k in ("fx", "fy", "cx", "cy")):
                fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
                return np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]], dtype=np.float32)

    raise FileNotFoundError(
        "无法在 scene_dir 中找到相机内参 K。请提供 cam_K.txt / K.txt / intrinsics.json / meta.json / intrinsics.npy 之一。"
    )


def read_color(color_path: str) -> np.ndarray:
    img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读 RGB 失败：{color_path}")
    return img  # BGR uint8


def read_depth(depth_path: str) -> np.ndarray:
    ext = Path(depth_path).suffix.lower()
    if ext == ".npy":
        d = np.load(depth_path)
    else:
        # png/tiff/exr 等
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"读 Depth 失败：{depth_path}")

    # 保证 float32
    if d.dtype == np.uint16:
        # 常见：单位 mm
        d = d.astype(np.float32) / 1000.0
    else:
        d = d.astype(np.float32)

    # 有些深度会有 3 通道或 1 通道
    if d.ndim == 3:
        d = d[:, :, 0]

    return d


def main():
    parser = argparse.ArgumentParser("Test FoundationPoseGDSAMTracker with local camera data")
    parser.add_argument("--scene_dir", type=str, required=True, help="包含 rgb/ 与 depth/ 的场景目录")
    parser.add_argument("--mesh_file", type=str, required=True, help="物体 mesh 路径")
    parser.add_argument("--text_prompt", type=str, required=True, help='例如 "mango"')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repo_root", type=str, default=None, help="项目 repo root（默认=当前脚本同级）")
    parser.add_argument("--max_frames", type=int, default=-1, help="只跑前 N 帧，-1 表示全跑")
    parser.add_argument("--debug", type=int, default=3)
    args = parser.parse_args()

    scene_dir = str(Path(args.scene_dir).resolve())
    mesh_file = str(Path(args.mesh_file).resolve())

    K = load_K(scene_dir)
    pairs = find_rgb_depth_pairs(scene_dir)

    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    # mask 默认放 scene_dir/mask
    mask_dir = str(Path(scene_dir) / "mask")

    tracker = FoundationPoseGDSAMTracker(
        mesh_file=mesh_file,
        K=K,
        test_scene_dir=scene_dir,
        repo_root=args.repo_root,
        device=args.device,
        debug=args.debug,
        mask_dir=mask_dir,
        reuse_mask=True,  # 若已有 mask_binary.png 直接复用
    )

    print(f"[INFO] frames = {len(pairs)}")
    print(f"[INFO] K =\n{K}")

    for i, (rgb_path, depth_path) in enumerate(pairs):
        color = read_color(rgb_path)
        depth = read_depth(depth_path)

        if i == 0:
            ok, pose = tracker.init(color, depth, text_prompt=args.text_prompt)
            print(f"[INIT] ok={ok}, pose=\n{pose}")
            if not ok:
                break
        else:
            ok, pose = tracker.update(color, depth)
            print(f"[UPD ] frame={i:06d} ok={ok}")
            if not ok:
                break

    print("[DONE] 结果已写入 debug_dir（见 tracker 自动生成的 debug/... 目录结构）")


if __name__ == "__main__":
    main()
