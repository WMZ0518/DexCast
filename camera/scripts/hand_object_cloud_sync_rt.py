#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from looptick import LoopTick

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make sure local modules can be found ---
CAMERA_ROOT = "/home/zyh/wmz/camera"
GEN_PC_DIR = "/home/zyh/wmz/cordvip_my/generate_pc/data_utils"
SYNAPATH_ROOT = "/home/zyh/wmz/Synapath_Python_Projects/vrRemoteControl"

for p in (CAMERA_ROOT, GEN_PC_DIR, SYNAPATH_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# from camera scripts
from base_cloud import cam2base, sample_points_from_mesh, transform_points

# from generate_pc (hand point cloud)
from generate_pc import HandPCGenerator, reorder_q_for_hand_model

from fdt_client.rgbd_cam import OrbbecRGBDCamera
from fdt_client.client import RemoteFoundationPose
from fdt_client.vis import draw_3d_box_client, draw_dict_to_img

from robots.universalRobotics.ur5RobotAgent import ur5RobotAgent
from robots.xhand.xhand_agent import XHandAgent


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_points(pts_xyz: np.ndarray) -> np.ndarray:
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
    if pts_xyz.ndim == 3 and pts_xyz.shape[0] == 1:
        pts_xyz = pts_xyz[0]
    if pts_xyz.ndim != 2:
        raise ValueError(f"点云维度不支持: {pts_xyz.shape}")
    if pts_xyz.shape[1] == 4:
        pts_xyz = pts_xyz[:, :3]
    if pts_xyz.shape[1] != 3:
        raise ValueError(f"点云最后维度应为3，当前: {pts_xyz.shape}")
    return pts_xyz.reshape(-1, 3)


def save_ply_xyzrgb(path: Path, pts_xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    pts_xyz = _normalize_points(pts_xyz)
    n = pts_xyz.shape[0]
    if n == 0:
        print(f"[WARN] 点云为空，跳过保存: {path}")
        return

    if rgb is None:
        rgb = np.full((n, 3), 180, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        if rgb.shape[0] != n:
            raise ValueError("rgb 点数必须与 pts_xyz 一致")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]) + "\n"

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n):
            x, y, z = pts_xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        f.flush()
        import os
        os.fsync(f.fileno())

    import os
    os.replace(tmp_path, path)


def save_pointcloud_png_multi(
    path: Path,
    pts_list: list[np.ndarray],
    rgb_list: list[np.ndarray],
    max_points: int = 20000,
    elev: float = 22.0,
    azim: float = 35.0,
    dpi: int = 150,
    figsize=(6, 4),
    draw_world_axes: bool = True,
    world_axis_len: float | None = None,
) -> None:
    path = Path(path)
    if len(pts_list) == 0:
        return

    norm_pts = []
    norm_rgb = []
    for pts, rgb in zip(pts_list, rgb_list):
        pts_n = _normalize_points(pts)
        if pts_n.shape[0] == 0:
            continue
        rgb_n = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        if rgb_n.shape[0] != pts_n.shape[0]:
            raise ValueError("rgb 点数必须与 pts_xyz 一致")
        norm_pts.append(pts_n)
        norm_rgb.append(rgb_n)

    if len(norm_pts) == 0:
        return

    pts_all = np.concatenate(norm_pts, axis=0)

    viz_pts = []
    viz_rgb = []
    for pts_n, rgb_n in zip(norm_pts, norm_rgb):
        n = pts_n.shape[0]
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            pts_v = pts_n[idx]
            rgb_v = rgb_n[idx]
        else:
            pts_v = pts_n
            rgb_v = rgb_n
        viz_pts.append(pts_v)
        viz_rgb.append(rgb_v)

    mins = pts_all.min(axis=0)
    maxs = pts_all.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    radius = float(np.max(span) / 2.0 + 1e-6)

    if world_axis_len is None:
        world_axis_len = max(0.05, 2 * (2.0 * radius))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    for pts_v, rgb_v in zip(viz_pts, viz_rgb):
        colors = (rgb_v.astype(np.float32) / 255.0) if rgb_v is not None else None
        ax.scatter(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2], s=0.3, c=colors, depthshade=False)

    if draw_world_axes:
        O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        L = float(world_axis_len)
        ax.plot([O[0], O[0] + L], [O[1], O[1]], [O[2], O[2]], linewidth=2)
        ax.plot([O[0], O[0]], [O[1], O[1] + L], [O[2], O[2]], linewidth=2)
        ax.plot([O[0], O[0]], [O[1], O[1]], [O[2], O[2] + L], linewidth=2)
        ax.text(O[0], O[1], O[2], "O_world", fontsize=10)
        ax.text(O[0] + L, O[1], O[2], "X", fontsize=10)
        ax.text(O[0], O[1] + L, O[2], "Y", fontsize=10)
        ax.text(O[0], O[1], O[2] + L, "Z", fontsize=10)

    ax.view_init(elev=elev, azim=azim)

    box_center = (center + np.array([0.0, 0.0, 0.0], dtype=np.float32)) / 2.0
    box_radius = max(radius, float(np.linalg.norm(center)) * 0.6, world_axis_len)

    ax.set_xlim(box_center[0] - box_radius, box_center[0] + box_radius)
    ax.set_ylim(box_center[1] - box_radius, box_center[1] + box_radius)
    ax.set_zlim(box_center[2] - box_radius, box_center[2] + box_radius)

    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def default_T_cam_to_base() -> np.ndarray:
    # 与 base_cloud.py 中 MAIN_MAT_converted 一致
    T = np.array(
        [
            [-0.99857, 0.05338, 0.00201, 544.02],
            [0.03946, 0.76244, -0.64586, 552.76],
            [-0.03601, -0.64486, -0.76345, 628.56],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T[:3, 3] /= 1000.0  # mm -> m
    return T


def _read_hand_joints(xhand: XHandAgent) -> Optional[list[float]]:
    states = xhand.get_joint_states()
    if states is None:
        return None
    try:
        joints = [float(states[i]) for i in range(12)]
    except Exception:
        return None
    return joints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--robot_name", type=str, default="ur5_xhand")
    parser.add_argument("--hand_points", type=int, default=1024)
    parser.add_argument("--hand_device", type=str, default="cuda")
    parser.add_argument("--mesh_points", type=int, default=1024)
    parser.add_argument("--device_index", type=int, default=1)
    parser.add_argument("--server_url", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--vis", action="store_true", help="打开相机可视化窗口")

    parser.add_argument("--ur_ip", type=str, default="192.168.58.100")
    parser.add_argument("--xhand_id", type=int, default=0)
    parser.add_argument("--xhand_mode", type=int, default=3)
    parser.add_argument("--xhand_protocol", type=str, default="RS485")
    parser.add_argument("--xhand_serial", type=str, default=None)
    parser.add_argument("--xhand_baud", type=int, default=3000000)
    parser.add_argument("--xhand_no_autofind", action="store_true")

    args = parser.parse_args()

    T_cam_to_base = default_T_cam_to_base()

    hand_gen = HandPCGenerator(
        robot_name=args.robot_name,
        num_points=args.hand_points,
        device=args.hand_device,
    )
    device = hand_gen._device

    # connect robot
    ur5 = ur5RobotAgent(ip_address=args.ur_ip)

    xhand = XHandAgent(
        hand_id=args.xhand_id,
        mode=args.xhand_mode,
        serial_port=args.xhand_serial,
        baud_rate=args.xhand_baud,
        auto_find=not args.xhand_no_autofind,
    )
    if not xhand.connect(protocol=args.xhand_protocol):
        raise RuntimeError("XHand 连接失败")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_root = ensure_dir(Path(args.output_dir) / run_tag)
    obj_dir = ensure_dir(out_root / "object_ply")
    hand_dir = ensure_dir(out_root / "hand_ply")
    pc_vis_dir = ensure_dir(out_root / "pc_vis")
    meta_dir = ensure_dir(out_root / "meta")

    obj_rgb = np.array([[255, 220, 0]], dtype=np.uint8)
    hand_rgb = np.array([[0, 200, 255]], dtype=np.uint8)

    print(f"输出目录: {out_root}")

    tracker = RemoteFoundationPose(args.server_url)
    cam = OrbbecRGBDCamera(device_index=args.device_index)

    mesh_points_obj = sample_points_from_mesh(args.mesh_file, n_points=args.mesh_points)

    loop = LoopTick()
    frame_id = 0
    saved_id = 0

    try:
        cam.start()
        intrinsic = cam.get_intrinsic()

        while True:
            frame_id += 1
            if args.max_frames is not None and saved_id >= args.max_frames:
                break

            color_image, depth_image, _ = cam.get_frames()
            if color_image is None or depth_image is None:
                continue

            if not tracker.initialized and intrinsic is not None:
                ok = tracker.init(
                    text_prompt=args.text_prompt,
                    cam_K=intrinsic,
                    mesh_file=args.mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )
                if not ok:
                    time.sleep(0.05)
                    continue

            ret, pose_cam = tracker.update(color_image, depth_image)
            if not ret:
                continue

            ur_joints = list(ur5.get_joint_angles())
            hand_joints = _read_hand_joints(xhand)
            if hand_joints is None:
                continue

            q = torch.tensor(ur_joints + hand_joints, dtype=torch.float32, device=device)
            q = reorder_q_for_hand_model(q, args.robot_name)

            pose_cam = np.array(pose_cam, dtype=np.float32).reshape(4, 4)
            pose_base = cam2base(pose_cam, T_cam_to_base)
            pose_base = np.array(pose_base, dtype=np.float32).reshape(4, 4)

            obj_points = transform_points(pose_base, mesh_points_obj)

            with torch.no_grad():
                hand_pc, _, hand_transform = hand_gen.hand.get_sampled_pc(q=q, num_points=args.hand_points)

            hand_pc_np = hand_pc.detach().to("cpu").numpy().astype(np.float32)
            obj_points = np.asarray(obj_points, dtype=np.float32)

            obj_rgb_full = np.tile(obj_rgb, (obj_points.shape[0], 1))
            hand_rgb_full = np.tile(hand_rgb, (hand_pc_np.shape[0], 1))

            saved_id += 1
            obj_ply = obj_dir / f"object_{saved_id:06d}.ply"
            hand_ply = hand_dir / f"hand_{saved_id:06d}.ply"

            save_ply_xyzrgb(obj_ply, obj_points, obj_rgb_full)
            save_ply_xyzrgb(hand_ply, hand_pc_np, hand_rgb_full)

            pcimg_path = pc_vis_dir / f"pc_{saved_id:06d}.png"
            save_pointcloud_png_multi(
                pcimg_path,
                pts_list=[obj_points, hand_pc_np],
                rgb_list=[obj_rgb_full, hand_rgb_full],
                max_points=20000,
                elev=22.0,
                azim=35.0,
                draw_world_axes=True,
            )

            meta_path = meta_dir / f"frame_{saved_id:06d}.npz"
            np.savez(
                meta_path,
                frame_id=frame_id,
                saved_id=saved_id,
                pose_base=pose_base,
                q=q.detach().to("cpu").numpy(),
                ur_joints=np.array(ur_joints, dtype=np.float32),
                hand_joints=np.array(hand_joints, dtype=np.float32),
                sampled_transform=hand_transform.detach().to("cpu").numpy(),
            )

            if saved_id % 10 == 0:
                print(f"已保存帧 {saved_id}")

            if args.vis:
                ns = loop.tick()
                hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
                result = {"Hz": f"{hz:.2f}"}
                vis_image = color_image.copy()
                vis_image = draw_dict_to_img(vis_image, result, font_size=1)
                if hasattr(tracker, "bbox_corners") and tracker.bbox_corners is not None:
                    vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)

                cv2.imshow("vis_image", vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        try:
            tracker.release()
        except Exception:
            pass
        try:
            xhand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
