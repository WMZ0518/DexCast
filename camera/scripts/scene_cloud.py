import argparse
from pathlib import Path
import importlib.util
import numpy as np
import torch


# =========================
# 绝对路径：按你给的路径写死
# =========================
BASE_CLOUD_PY = "/home/zyh/wmz/camera/scripts/base_cloud.py"
GENERATE_PC_PY = "/home/zyh/wmz/cordvip_my/generate_pc/data_utils/generate_pc.py"


def import_from_path(module_name: str, file_path: str):
    """从绝对路径导入一个 .py 作为模块（不依赖 PYTHONPATH）"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 导入你的两个脚本模块
base_cloud = import_from_path("base_cloud", BASE_CLOUD_PY)
generate_pc = import_from_path("generate_pc", GENERATE_PC_PY)

# 从模块里拿函数/类
track_pose_stream = base_cloud.track_pose_stream
save_ply_xyzrgb = base_cloud.save_ply_xyzrgb
save_pointcloud_png = base_cloud.save_pointcloud_png

HandPCGenerator = generate_pc.HandPCGenerator  # 你需要已把 HandPCGenerator 加进 generate_pc.py


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_numpy_xyz(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy().astype(np.float32)


# =========================
# 写死 T_cam_to_base（按你之前 main.py 里那套）
# =========================
def get_T_cam_to_base() -> np.ndarray:
    MAIN_MAT = np.array(
        [
            [-0.99857,  0.05338,  0.00201, 544.02],
            [ 0.03946,  0.76244, -0.64586, 552.76],
            [-0.03601, -0.64486, -0.76345, 628.56],
            [ 0.0,      0.0,      0.0,       1.0 ],
        ],
        dtype=np.float32,
    )
    MAIN_MAT[:3, 3] /= 1000.0  # mm -> m
    return MAIN_MAT


def main():
    parser = argparse.ArgumentParser()

    # object / tracker
    parser.add_argument("--text_prompt", default="yellow", type=str)
    parser.add_argument("--mesh_file", required=True, type=str)
    parser.add_argument("--device_index", default=1, type=int)
    parser.add_argument("--server_url", default="tcp://127.0.0.1:5555", type=str)
    parser.add_argument("--mesh_n_points", default=1024, type=int)

    # hand / demo
    parser.add_argument("--demo_dir", required=True, type=str)
    parser.add_argument("--robot_name", default="ur5_xhand", type=str)
    parser.add_argument("--hand_n_points", default=1024, type=int)
    parser.add_argument("--hand_device", default="cuda", type=str)
    parser.add_argument("--fallback_commanded", action="store_true")

    # output
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--max_frames", default=None, type=int)

    args = parser.parse_args()

    out_root = ensure_dir(Path(args.output_dir))
    ply_dir = ensure_dir(out_root / "ply")
    png_dir = ensure_dir(out_root / "pc_vis")

    T_cam_to_base = get_T_cam_to_base()

    # init hand generator
    hand_gen = HandPCGenerator(
        robot_name=args.robot_name,
        num_points=args.hand_n_points,
        device=args.hand_device,
    )
    pkl_list = hand_gen.list_pkl_files(args.demo_dir)
    if len(pkl_list) == 0:
        raise RuntimeError(f"No digit-named pkl files found in: {args.demo_dir}")

    # colors (uint8)
    obj_rgb_color = np.array([255, 220, 0], dtype=np.uint8)   # object: yellow-ish
    hand_rgb_color = np.array([80, 160, 255], dtype=np.uint8) # hand: blue-ish

    for frame_id, pose_base, obj_pts_world in track_pose_stream(
        text_prompt=args.text_prompt,
        mesh_file=args.mesh_file,
        device_index=args.device_index,
        server_url=args.server_url,
        T_cam_to_base=T_cam_to_base,
        mesh_n_points=args.mesh_n_points,
        max_frames=args.max_frames,
    ):
        # align hand frame by index (clamp)
        idx = min(frame_id - 1, len(pkl_list) - 1)
        pkl_path = pkl_list[idx]

        # hand pc
        hand_pc_t, hand_meta = hand_gen.hand_pc_from_pkl(
            pkl_path,
            num_points=args.hand_n_points,
            fallback=args.fallback_commanded,
        )
        hand_pts = to_numpy_xyz(hand_pc_t)

        # ===== 可选：如果发现手不在 base/world，这里再补一个变换 =====
        # hand_pts = base_cloud.transform_points(T_hand_to_base, hand_pts)

        obj_pts = np.asarray(obj_pts_world, dtype=np.float32).reshape(-1, 3)
        all_pts = np.vstack([obj_pts, hand_pts])

        obj_rgb = np.tile(obj_rgb_color[None, :], (obj_pts.shape[0], 1))
        hand_rgb = np.tile(hand_rgb_color[None, :], (hand_pts.shape[0], 1))
        all_rgb = np.vstack([obj_rgb, hand_rgb])

        ply_path = ply_dir / f"scene_{frame_id:06d}.ply"
        save_ply_xyzrgb(ply_path, all_pts, all_rgb)

        if args.save_png:
            png_path = png_dir / f"scene_{frame_id:06d}.png"
            save_pointcloud_png(
                png_path,
                all_pts,
                rgb=all_rgb,
                max_points=20000,
                elev=22.0,
                azim=35.0,
                draw_world_axes=True,
            )

        print(f"[{frame_id:06d}] saved: {ply_path} (hand_pkl={pkl_path.name})")


if __name__ == "__main__":
    main()
