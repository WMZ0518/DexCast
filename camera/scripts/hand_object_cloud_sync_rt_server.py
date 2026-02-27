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
import zmq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make sure local modules can be found ---
CAMERA_ROOT = "/home/zyh/wmz/camera"
GEN_PC_DIR = "/home/zyh/wmz/cordvip/generate_pc/data_utils"
SYNAPATH_ROOT = "/home/zyh/wmz/Synapath_Python_Projects/vrRemoteControl"

for p in (CAMERA_ROOT, GEN_PC_DIR, SYNAPATH_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from base_cloud import cam2base, sample_points_from_mesh, transform_points
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


def _apply_xyz_offset(pts_xyz: np.ndarray, offset_xyz: np.ndarray) -> np.ndarray:
    pts_xyz = _normalize_points(pts_xyz)
    if offset_xyz is None:
        return pts_xyz
    offset_xyz = np.asarray(offset_xyz, dtype=np.float32).reshape(1, 3)
    return pts_xyz + offset_xyz


def save_ply_xyz(path: Path, pts_xyz: np.ndarray) -> None:
    path = Path(path)
    pts_xyz = _normalize_points(pts_xyz)
    n = pts_xyz.shape[0]
    if n == 0:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n):
            x, y, z = pts_xyz[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    import os
    os.replace(tmp_path, path)


def save_pointcloud_png_multi(
    path: Path,
    pts_list: list[np.ndarray],
    max_points: int = 20000,
    elev: float = 22.0,
    azim: float = 35.0,
    dpi: int = 150,
    figsize=(6, 4),
    draw_world_axes: bool = True,
    world_axis_len: float | None = None,
    draw_xy_grid: bool = True,
    xy_grid_step: float | None = None,
) -> None:
    path = Path(path)
    if len(pts_list) == 0:
        return

    norm_pts = []
    for pts in pts_list:
        pts_n = _normalize_points(pts)
        if pts_n.shape[0] == 0:
            continue
        norm_pts.append(pts_n)

    if len(norm_pts) == 0:
        return

    pts_all = np.concatenate(norm_pts, axis=0)

    viz_pts = []
    for pts_n in norm_pts:
        n = pts_n.shape[0]
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            pts_v = pts_n[idx]
        else:
            pts_v = pts_n
        viz_pts.append(pts_v)

    mins = pts_all.min(axis=0)
    maxs = pts_all.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    radius = float(np.max(span) / 2.0 + 1e-6)

    if world_axis_len is None:
        world_axis_len = max(0.05, 2 * (2.0 * radius))

    if xy_grid_step is None:
        xy_grid_step = max(0.01, box_safe_step := (max(radius, 0.05) / 6.0))


    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # color: hand cyan, obj yellow
    for i, pts_v in enumerate(viz_pts):
        if i == 0:
            color = (0.0, 0.78, 1.0)
        else:
            color = (1.0, 0.86, 0.0)
        ax.scatter(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2], s=0.3, c=[color], depthshade=False)
    
    if draw_xy_grid:
        grid_half = max(radius * 1.2, world_axis_len * 0.5)
        n = int(np.ceil(grid_half / xy_grid_step))
        x_vals = np.arange(-n, n + 1, dtype=np.float32) * float(xy_grid_step)
        y_vals = np.arange(-n, n + 1, dtype=np.float32) * float(xy_grid_step)
        z0 = 0.0
        for xv in x_vals:
            ax.plot([xv, xv], [y_vals[0], y_vals[-1]], [z0, z0], color=(0.65, 0.65, 0.65), linewidth=0.5, alpha=0.35)
        for yv in y_vals:
            ax.plot([x_vals[0], x_vals[-1]], [yv, yv], [z0, z0], color=(0.65, 0.65, 0.65), linewidth=0.5, alpha=0.35)


    if draw_world_axes:
        O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        L = float(world_axis_len)
        ax.plot([O[0], O[0] + L], [O[1], O[1]], [O[2], O[2]], linewidth=2)
        ax.plot([O[0], O[0]], [O[1], O[1] + L], [O[2], O[2]], linewidth=2)
        ax.plot([O[0], O[0]], [O[1], O[1]], [O[2], O[2] + L], linewidth=2)
        ax.text(O[0], O[1], O[2], "O_world", fontsize=10)
        ax.text(O[0] + L, O[1], O[2], f"X {L:.3f}", fontsize=9)
        ax.text(O[0], O[1] + L, O[2], f"Y {L:.3f}", fontsize=9)
        ax.text(O[0], O[1], O[2] + L, f"Z {L:.3f}", fontsize=9)

        # Tick labels to help estimate offsets
        tick_fracs = (0.0, 0.5, 1.0)
        tick_len = max(L * 0.015, 1e-3)
        for frac in tick_fracs:
            v = L * float(frac)
            label = f"{v:.3f}"
            ax.plot([v, v], [0.0, 0.0], [0.0, tick_len], linewidth=1)
            ax.text(v, 0.0, tick_len * 1.5, label, fontsize=8)
            ax.plot([0.0, 0.0], [v, v], [0.0, tick_len], linewidth=1)
            ax.text(0.0, v, tick_len * 1.5, label, fontsize=8)
            ax.plot([0.0, tick_len], [0.0, 0.0], [v, v], linewidth=1)
            ax.text(tick_len * 1.5, 0.0, v, label, fontsize=8)

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
    T = np.array(
        [
            [-0.99857, 0.05338, 0.00201, 544.02],
            [0.03946, 0.76244, -0.64586, 552.76],
            [-0.03601, -0.64486, -0.76345, 628.56],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T[:3, 3] /= 1000.0
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
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--server_url", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument("--ur_ip", type=str, default="192.168.58.100")
    parser.add_argument("--xhand_id", type=int, default=0)
    parser.add_argument("--xhand_mode", type=int, default=3)
    parser.add_argument("--xhand_protocol", type=str, default="RS485")
    parser.add_argument("--xhand_serial", type=str, default=None)
    parser.add_argument("--xhand_baud", type=int, default=3000000)
    parser.add_argument("--xhand_no_autofind", action="store_true")
    parser.add_argument("--no_ur", action="store_true")
    parser.add_argument("--no_xhand", action="store_true")
    parser.add_argument("--state_connect", type=str, default=None)
    parser.add_argument("--state_topic", type=str, default="state")
    parser.add_argument("--state_req_connect", type=str, default=None,
                        help="若设置，采集端将通过REQ请求执行端状态（不使用state_sub）。")
    parser.add_argument("--state_req_timeout_ms", type=int, default=1000)

    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:7777")
    parser.add_argument("--req_bind", type=str, default=None,
                        help="若设置，将开启请求模式(REP)。推理端发送请求后返回最新点云/状态。")
    parser.add_argument("--topic", type=str, default="pc")
    parser.add_argument("--send_hz", type=float, default=10.0)
    parser.add_argument("--vis", action="store_true", help="打开相机可视化窗口")
    parser.add_argument("--save", action="store_true", default=True, help="保存点云与元数据")
    parser.add_argument("--no_save", action="store_true", help="不保存点云与元数据")
    parser.add_argument("--output_dir", type=str, default="/home/zyh/wmz/our_data/pc_stream")
    parser.add_argument("--save_png", action="store_true", default=True, help="保存点云渲染图片")
    parser.add_argument("--no_save_png", action="store_true", help="不保存点云渲染图片")
    parser.add_argument("--save_png_interval_sec", type=float, default=5.0,
                        help=">0 时启用PNG间隔保存：每隔该秒数保存一张；<=0 表示每帧都保存。")
    parser.add_argument("--no_xy_grid", action="store_true", help="PNG中关闭XY平面网格")
    parser.add_argument("--pc_offset_xyz", type=str, default="0,0,0",
                        help="点云XYZ整体偏置，格式: x,y,z（单位与点云一致）")
    parser.add_argument("--pc_offset_hand_xyz", type=str, default=None,
                        help="手点云XYZ偏置，格式: x,y,z（单位与点云一致），优先生效")
    parser.add_argument("--pc_offset_obj_xyz", type=str, default=None,
                        help="物体点云XYZ偏置，格式: x,y,z（单位与点云一致），优先生效")

    args = parser.parse_args()

    if args.no_save:
        args.save = False
    if args.no_save_png:
        args.save_png = False

    def _parse_offset(label: str, value: str | None) -> np.ndarray:
        if value is None:
            return None
        _items = [v.strip() for v in value.split(",")]
        if len(_items) != 3:
            raise ValueError(f"{label} 格式错误: {value}，应为 x,y,z")
        try:
            return np.array([float(v) for v in _items], dtype=np.float32)
        except Exception:
            raise ValueError(f"{label} 格式错误: {value}，应为 x,y,z")

    pc_offset_xyz = _parse_offset("--pc_offset_xyz", args.pc_offset_xyz)
    pc_offset_hand = _parse_offset("--pc_offset_hand_xyz", args.pc_offset_hand_xyz)
    pc_offset_obj = _parse_offset("--pc_offset_obj_xyz", args.pc_offset_obj_xyz)
    if pc_offset_hand is None:
        pc_offset_hand = pc_offset_xyz
    if pc_offset_obj is None:
        pc_offset_obj = pc_offset_xyz

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.bind)

    rep = None
    rep_poller = None
    if args.req_bind:
        rep = ctx.socket(zmq.REP)
        rep.bind(args.req_bind)
        rep_poller = zmq.Poller()
        rep_poller.register(rep, zmq.POLLIN)

    state_req = None
    state_req_poller = None
    if args.state_req_connect:
        state_req = ctx.socket(zmq.REQ)
        state_req.connect(args.state_req_connect)
        state_req_poller = zmq.Poller()
        state_req_poller.register(state_req, zmq.POLLIN)
        def _reset_state_req() -> None:
            nonlocal state_req, state_req_poller
            try:
                if state_req is not None:
                    state_req.close(0)
            except Exception:
                pass
            state_req = ctx.socket(zmq.REQ)
            state_req.setsockopt(zmq.LINGER, 0)
            try:
                state_req.setsockopt(zmq.REQ_RELAXED, 1)
                state_req.setsockopt(zmq.REQ_CORRELATE, 1)
            except Exception:
                pass
            state_req.connect(args.state_req_connect)
            state_req_poller = zmq.Poller()
            state_req_poller.register(state_req, zmq.POLLIN)

    T_cam_to_base = default_T_cam_to_base()

    hand_gen = HandPCGenerator(
        robot_name=args.robot_name,
        num_points=args.hand_points,
        device=args.hand_device,
    )
    device = hand_gen._device

    ur5 = None
    xhand = None
    if not args.no_ur:
        ur5 = ur5RobotAgent(ip_address=args.ur_ip)

    if not args.no_xhand:
        xhand = XHandAgent(
            hand_id=args.xhand_id,
            mode=args.xhand_mode,
            serial_port=args.xhand_serial,
            baud_rate=args.xhand_baud,
            auto_find=not args.xhand_no_autofind,
        )
        if not xhand.connect(protocol=args.xhand_protocol):
            raise RuntimeError("XHand 连接失败")

    tracker = RemoteFoundationPose(args.server_url)
    cam = OrbbecRGBDCamera(device_index=args.device_index)

    mesh_points_obj = sample_points_from_mesh(args.mesh_file, n_points=args.mesh_points)

    frame_id = 0
    last_send = 0.0
    last_log = 0.0
    last_state_recv_log = 0.0
    last_vis_time = None
    last_png_save_ts = 0.0
    period = 1.0 / max(args.send_hz, 1e-3)
    last_payload = None

    state_sub = None
    latest_state = {"ur_joints": None, "hand_joints": None, "ts": None, "state_idx": None}
    last_state_log = 0.0
    if args.state_req_connect is None and args.state_connect is not None:
        state_sub = ctx.socket(zmq.SUB)
        state_sub.connect(args.state_connect)
        state_sub.setsockopt(zmq.SUBSCRIBE, args.state_topic.encode("utf-8"))

    if args.save:
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        out_root = ensure_dir(Path(args.output_dir) / run_tag)
        obj_dir = ensure_dir(out_root / "object_ply")
        hand_dir = ensure_dir(out_root / "hand_ply")
        meta_dir = ensure_dir(out_root / "meta")
        pc_vis_dir = ensure_dir(out_root / "pc_vis")
        print(f"[SAVE] 输出目录: {out_root}")

    try:
        cam.start()
        intrinsic = cam.get_intrinsic()
        request_mode = rep is not None
        pending_request = False
        tracker_ready = False
        def _reply(payload: dict) -> bool:
            nonlocal pending_request
            if not request_mode or rep is None:
                return False
            if not pending_request:
                return False
            rep.send(zmq.utils.jsonapi.dumps(payload))
            pending_request = False
            return True
        vis_ready = False
        if request_mode:
            print("[READY] 请求模式已启用，等待推理端请求...")

        while True:
            if request_mode:
                socks = dict(rep_poller.poll(timeout=0))
                if rep in socks:
                    try:
                        _ = rep.recv_multipart()
                    except Exception:
                        _ = rep.recv()
                    if pending_request:
                        _reply({"error": "busy"})
                        print("[REQ] 收到请求但仍在处理上一请求，已返回 error=busy")
                    else:
                        pending_request = True
                        print("[REQ] 收到推理端请求，准备采集最新帧...")

            frame_id += 1
            if args.max_frames is not None and frame_id > args.max_frames:
                break

            color_image, depth_image, _ = cam.get_frames()
            if color_image is None or depth_image is None:
                if _reply({
                    "error": "no_frame",
                    "frame_id": frame_id,
                }):
                    print("[REQ] 相机帧为空，已返回 error=no_frame")
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
                    if _reply({
                        "error": "tracker_init_failed",
                        "frame_id": frame_id,
                    }):
                        print("[REQ] tracker 初始化失败，已返回 error=tracker_init_failed")
                    time.sleep(0.05)
                    continue
                tracker_ready = True
                if request_mode:
                    print("[TRACK] 初始化成功，等待推理端请求...")
                if args.vis and not vis_ready:
                    cv2.namedWindow("vis_image", cv2.WINDOW_NORMAL)
                    cv2.imshow("vis_image", color_image)
                    cv2.waitKey(1)
                    vis_ready = True

            ret, pose_cam = tracker.update(color_image, depth_image)
            if not ret:
                if _reply({
                    "error": "tracker_update_failed",
                    "frame_id": frame_id,
                }):
                    print("[REQ] tracker 更新失败，已返回 error=tracker_update_failed")
                continue

            now = time.time()
            auto_save_due = False
            if args.save and args.save_png:
                if args.save_png_interval_sec > 0:
                    auto_save_due = (now - last_png_save_ts) >= args.save_png_interval_sec
                else:
                    auto_save_due = True

            if request_mode and not pending_request and not auto_save_due:
                # 持续跟踪 + 可视化，但不请求状态/点云发送
                if args.vis:
                    result = {"TRACK": "ON"}
                    vis_image = color_image.copy()
                    vis_image = draw_dict_to_img(vis_image, result, font_size=1)
                    if hasattr(tracker, "bbox_corners") and tracker.bbox_corners is not None:
                        vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)
                    cv2.imshow("vis_image", vis_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                continue

            if state_req is not None:
                # 请求执行端最新关节状态
                try:
                    state_req.send(b"state")
                    socks = dict(state_req_poller.poll(timeout=args.state_req_timeout_ms))
                    if state_req in socks:
                        reply = state_req.recv()
                        latest_state = zmq.utils.jsonapi.loads(reply)
                        latest_state["ts"] = latest_state.get("ts", time.time())
                    else:
                        latest_state = {"ur_joints": None, "hand_joints": None, "ts": None, "state_idx": None}
                        print("[STATE-REQ] 超时，重置请求socket")
                        _reset_state_req()
                except Exception:
                    latest_state = {"ur_joints": None, "hand_joints": None, "ts": None, "state_idx": None}
                    print("[STATE-REQ] 异常，重置请求socket")
                    _reset_state_req()

                ur_joints = latest_state.get("ur_joints", None)
                hand_joints = latest_state.get("hand_joints", None)
                state_ts = latest_state.get("ts", None)
                state_idx = latest_state.get("state_idx", None)
                if request_mode:
                    if state_idx is not None:
                        print(f"[STATE-REQ] 已获取关节状态 idx={state_idx}")
                    else:
                        print("[STATE-REQ] 获取关节状态失败")
            elif state_sub is not None:
                try:
                    while True:
                        _t, _payload = state_sub.recv_multipart(flags=zmq.NOBLOCK)
                        latest_state = zmq.utils.jsonapi.loads(_payload)
                        latest_state["ts"] = time.time()
                except zmq.Again:
                    pass
                ur_joints = latest_state.get("ur_joints", None)
                hand_joints = latest_state.get("hand_joints", None)
                state_ts = latest_state.get("ts", None)
                state_idx = latest_state.get("state_idx", None)
                if (not request_mode) and state_ts is not None and (time.time() - last_state_recv_log >= 1.0):
                    age = time.time() - state_ts
                    if state_idx is not None:
                        print(f"[STATE] 已接收关节状态 idx={state_idx} age={age:.2f}s")
                    else:
                        print(f"[STATE] 已接收关节状态 age={age:.2f}s")
                    last_state_recv_log = time.time()

                if (not request_mode) and (ur_joints is None or hand_joints is None) and (time.time() - last_state_log >= 1.0):
                    print("[STATE] 等待执行端关节状态中...")
                    last_state_log = time.time()
            else:
                ur_joints = list(ur5.get_joint_angles()) if ur5 is not None else None
                hand_joints = _read_hand_joints(xhand) if xhand is not None else None
                state_ts = None
                state_idx = None

            # 可视化窗口（先画出来，便于调试）
            if args.vis:
                result = {"Hz": f"{(1.0 / max(now - last_vis_time, 1e-3)):.2f}"} if last_vis_time else {"Hz": "0.00"}
                vis_image = color_image.copy()
                vis_image = draw_dict_to_img(vis_image, result, font_size=1)
                if hasattr(tracker, "bbox_corners") and tracker.bbox_corners is not None:
                    vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)
                cv2.imshow("vis_image", vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if ur_joints is None or hand_joints is None:
                if _reply({
                    "error": "state_unavailable",
                    "frame_id": frame_id,
                }):
                    print("[REQ] 状态不可用，已返回 error=state_unavailable")
                continue

            q = torch.tensor(list(ur_joints) + list(hand_joints), dtype=torch.float32, device=device)
            q = reorder_q_for_hand_model(q, args.robot_name)

            pose_cam = np.array(pose_cam, dtype=np.float32).reshape(4, 4)
            pose_base = cam2base(pose_cam, T_cam_to_base)
            pose_base = np.array(pose_base, dtype=np.float32).reshape(4, 4)

            obj_points = transform_points(pose_base, mesh_points_obj)
            

            with torch.no_grad():
                hand_pc, _, _ = hand_gen.hand.get_sampled_pc(q=q, num_points=args.hand_points)

            hand_pc_np = hand_pc.detach().to("cpu").numpy().astype(np.float32)
            obj_points = np.asarray(obj_points, dtype=np.float32)

            if pc_offset_hand is not None and np.any(pc_offset_hand != 0.0):
                hand_pc_np = _apply_xyz_offset(hand_pc_np, pc_offset_hand)
            if pc_offset_obj is not None and np.any(pc_offset_obj != 0.0):
                obj_points = _apply_xyz_offset(obj_points, pc_offset_obj)

            # 发送频率控制（请求模式下不节流）
            if not request_mode:
                if now - last_send < period:
                    continue
            last_send = now

            payload = {
                "ts": now,
                "frame_id": frame_id,
                "hand_pc": _normalize_points(hand_pc_np).tolist(),
                "obj_pc": _normalize_points(obj_points).tolist(),
                "ur_joints": np.asarray(ur_joints, dtype=np.float32).tolist(),
                "hand_joints": np.asarray(hand_joints, dtype=np.float32).tolist(),
                "state_ts": state_ts,
                "state_idx": state_idx,
            }

            last_payload = payload

            send_pub = not (request_mode and not pending_request)
            if send_pub:
                pub.send_multipart([
                    args.topic.encode("utf-8"),
                    zmq.utils.jsonapi.dumps(payload),
                ])

            if request_mode:
                reply = last_payload if last_payload is not None else {}
                if _reply(reply):
                    print(f"[REQ] 已发送回复 frame={frame_id} state_idx={state_idx}")

            if args.save:
                obj_ply = obj_dir / f"object_{frame_id:06d}.ply"
                hand_ply = hand_dir / f"hand_{frame_id:06d}.ply"
                save_ply_xyz(obj_ply, obj_points)
                save_ply_xyz(hand_ply, hand_pc_np)
                meta_path = meta_dir / f"frame_{frame_id:06d}.npz"
                np.savez(
                    meta_path,
                    frame_id=frame_id,
                    ts=now,
                    ur_joints=np.asarray(ur_joints, dtype=np.float32),
                    hand_joints=np.asarray(hand_joints, dtype=np.float32),
                    pose_base=pose_base,
                )
                if args.save_png:
                    do_save_png = True
                    if args.save_png_interval_sec > 0:
                        do_save_png = (now - last_png_save_ts) >= args.save_png_interval_sec
                    if do_save_png:
                        pcimg_path = pc_vis_dir / f"pc_{frame_id:06d}.png"
                        save_pointcloud_png_multi(
                            pcimg_path,
                            pts_list=[hand_pc_np, obj_points],
                            max_points=20000,
                            elev=22.0,
                            azim=35.0,
                            draw_world_axes=True,
                            draw_xy_grid=(not args.no_xy_grid),
                        )
                        last_png_save_ts = now
                        print(
                            f"[SAVE_PNG] frame={frame_id} path={pcimg_path} "
                            f"hand_pts={hand_pc_np.shape[0]} obj_pts={obj_points.shape[0]}"
                        )

            # 终端状态输出（每秒一次）
            if (not request_mode) and (now - last_log >= 1.0):
                hz = 0.0 if last_vis_time is None else 1.0 / max(now - last_vis_time, 1e-3)
                last_vis_time = now
                state_age = (time.time() - state_ts) if state_ts is not None else -1.0
                print(
                    f"[PUB] frame={frame_id} hand={len(payload['hand_pc'])} "
                    f"obj={len(payload['obj_pc'])} hz={hz:.2f} state_age={state_age:.2f}"
                )
                last_log = now

            # 可视化窗口（这里不再重复绘制）

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
            if xhand is not None:
                xhand.disconnect()
        except Exception:
            pass
        try:
            if state_sub is not None:
                state_sub.close(0)
        except Exception:
            pass
        try:
            if rep is not None:
                rep.close(0)
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
