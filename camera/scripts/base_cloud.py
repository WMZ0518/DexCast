# main.py

from pathlib import Path
import time
import cv2
import sys
# sys.path.append('/home/zyh/wmz/camera/venv/lib64/python3.10/site-packages')

from looptick import LoopTick
import numpy as np
from rich import print as rprint

# 动态设置 PYTHONPATH
fdt_parent_dir = "/home/zyh/wmz/camera"
if fdt_parent_dir not in sys.path:
    sys.path.insert(0, fdt_parent_dir)

from fdt_client.logger import logger
from fdt_client.rgbd_cam import OrbbecRGBDCamera
from fdt_client.client import RemoteFoundationPose
from fdt_client.vis import draw_3d_box_client, draw_dict_to_img

# === mesh 采样依赖 ===
import trimesh

# === 新增：点云图像离屏渲染（不依赖OpenGL）===
import matplotlib
matplotlib.use("Agg")  # 关键：无窗口离屏渲染
import matplotlib.pyplot as plt


loop = LoopTick()

# =========================
# 工具函数
# =========================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sample_points_from_mesh(mesh_path: Path | str, n_points: int = 50000) -> np.ndarray:
    """
    从 mesh 表面均匀采样点，返回 (N,3) 物体坐标系点云（单位与 mesh 一致）
    """
    mesh_path = str(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if mesh.is_empty:
        raise ValueError(f"Mesh 为空: {mesh_path}")

    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    T: (4,4), pts: (N,3) -> (N,3)
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    out = (T @ pts_h.T).T[:, :3]
    return out.astype(np.float32)


import os

def save_ply_xyzrgb(path: Path, pts_xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """
    原子写入 ASCII PLY（x y z r g b）
    - 写入 path.tmp，flush+fsync 后 os.replace 到最终 path
    """
    path = Path(path)
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    n = pts_xyz.shape[0]
    if n == 0:
        logger.warning(f"点云为空，跳过保存: {path}")
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
        "end_header"
    ]) + "\n"

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n):
            x, y, z = pts_xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def save_pointcloud_png(
    path: Path,
    pts_xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    max_points: int = 20000,
    elev: float = 22.0,
    azim: float = 35.0,
    dpi: int = 150,
    figsize=(6, 4),
    draw_world_axes: bool = True,
    world_axis_len: float | None = None,   # 若 None：自动按点云尺度设置
) -> None:
    """
    将点云渲染成PNG（离屏，无需OpenGL）
    - 渲染空间：pts_xyz 所在坐标系（你这里传入的是 points_world，所以就是世界/基座系）
    - draw_world_axes=True 时：在 (0,0,0) 画世界原点与 XYZ 轴，并标注 O_world
    """
    path = Path(path)
    pts_xyz = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    n = pts_xyz.shape[0]
    if n == 0:
        logger.warning(f"点云为空，跳过保存图片: {path}")
        return

    # 下采样仅用于渲染加速，不影响PLY
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        pts_viz = pts_xyz[idx]
        rgb_viz = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)[idx] if rgb is not None else None
    else:
        pts_viz = pts_xyz
        rgb_viz = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3) if rgb is not None else None

    colors = (rgb_viz.astype(np.float32) / 255.0) if rgb_viz is not None else None

    # === 计算显示范围（等比例盒）===
    mins = pts_xyz.min(axis=0)   # 注意：用全量点云计算范围更稳定
    maxs = pts_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    radius = float(np.max(span) / 2.0 + 1e-6)

    # 世界轴长度：默认取点云尺度的 25%（也可以固定成 0.2m 之类）
    if world_axis_len is None:
        world_axis_len = max(0.05, 2 * (2.0 * radius))  # 至少 5cm

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # 点云
    ax.scatter(
        pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
        s=0.3, c=colors, depthshade=False
    )

    # === 世界/基座坐标系原点与三轴 ===
    if draw_world_axes:
        O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        L = float(world_axis_len)

        # X轴（红）
        ax.plot([O[0], O[0] + L], [O[1], O[1]], [O[2], O[2]], linewidth=2)
        # Y轴（绿）
        ax.plot([O[0], O[0]], [O[1], O[1] + L], [O[2], O[2]], linewidth=2)
        # Z轴（蓝）
        ax.plot([O[0], O[0]], [O[1], O[1]], [O[2], O[2] + L], linewidth=2)

        # 标注（避免太贴近原点，看不见）
        ax.text(O[0], O[1], O[2], "O_world", fontsize=10)
        ax.text(O[0] + L, O[1], O[2], "X", fontsize=10)
        ax.text(O[0], O[1] + L, O[2], "Y", fontsize=10)
        ax.text(O[0], O[1], O[2] + L, "Z", fontsize=10)

    # 视角
    ax.view_init(elev=elev, azim=azim)

    # 轴范围：把点云中心放中间，并保证原点也尽量落在视野里
    # 为了让原点可见，我们把显示盒中心“夹在点云中心和原点之间”
    # （如果点云离原点很远，这一步很关键）
    box_center = (center + np.array([0.0, 0.0, 0.0], dtype=np.float32)) / 2.0
    box_radius = max(radius, float(np.linalg.norm(center)) * 0.6, world_axis_len)  # 保证能看到原点+轴

    ax.set_xlim(box_center[0] - box_radius, box_center[0] + box_radius)
    ax.set_ylim(box_center[1] - box_radius, box_center[1] + box_radius)
    ax.set_zlim(box_center[2] - box_radius, box_center[2] + box_radius)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def track_pose(
    text_prompt: str,
    mesh_file: Path | str,
    device_index: int = 1,
    server_url: str = "tcp://127.0.0.1:5555",
    T_cam_to_base=np.eye(4),
    vis: bool = False,
    root_output_dir: str | Path = "logs/mesh_cloud"  # 新增：统一母文件夹参数
):
    tracker = RemoteFoundationPose(server_url)
    gemini_2 = OrbbecRGBDCamera(device_index=device_index)

    # =========================
    # 输出目录：使用传入的 root_output_dir
    # =========================
    RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
    # 统一母文件夹逻辑
    PARENT_DIR = Path(root_output_dir)
    OUT_DIR = ensure_dir(PARENT_DIR / RUN_TAG) 
    
    PLY_DIR = ensure_dir(OUT_DIR / "ply")
    PCIMG_DIR = ensure_dir(OUT_DIR / "pc_vis")

    SAVE_IMAGE_EVERY_N_FRAMES = 1   # 点云可视化图片：每30帧保存一次
    SAVE_CLOUD_EVERY_N_FRAMES = 1    # 点云PLY：每帧保存（必须）

    # =========================
    # 预采样 mesh 点云（只做一次）
    # =========================
    mesh_file = Path(mesh_file)
    try:
        MESH_N_POINTS = 1024
        mesh_points_obj = sample_points_from_mesh(mesh_file, n_points=MESH_N_POINTS)

        # 固定颜色（偏黄）
        mesh_rgb = np.tile(np.array([[255, 220, 0]], dtype=np.uint8), (mesh_points_obj.shape[0], 1))
        logger.info(f"Mesh 点云采样完成: {mesh_points_obj.shape[0]} points, mesh={mesh_file}")
    except Exception as e:
        logger.exception(f"Mesh 点云采样失败: {e}")
        raise

    frame_id = 0
    last_cloud_saved = 0
    last_pcimg_saved = 0

    try:
        gemini_2.start()

        intrinsic = gemini_2.get_intrinsic()
        rprint(intrinsic)

        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        is_init = False

        while True:
            frame_id += 1

            color_image, depth_image, depth_data = gemini_2.get_frames()
            if color_image is None or depth_image is None:
                print("图像帧为空，请检查设备是否正常连接")
                continue

            if not tracker.initialized and intrinsic is not None:
                is_init = tracker.init(
                    text_prompt=text_prompt,
                    cam_K=intrinsic,
                    mesh_file=mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )
                if not is_init:
                    time.sleep(0.1)
                    continue

            ret, pose_cam = tracker.update(color_image, depth_image)
            pose_cam = np.array(pose_cam).reshape(4, 4)

            xyz_arm = []
            xyz_cam = []
            pose_base = None
            if ret:
                pose_base = cam2base(pose_cam, T_cam_to_base)
                pose_base = np.array(pose_base).reshape(4, 4)

                xyz_arm = pose_base[:3, 3]
                xyz_cam = pose_cam[:3, 3]

            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
            print(f"\rHz: {hz:.2f}, xyz_cam: {xyz_cam}, xyz_base(world): {xyz_arm}" + "" * 10, end="", flush=True)

            # =========================
            # 每帧：保存 PLY（世界/基座坐标系）
            # 每30帧：保存 点云渲染图（PNG）
            # =========================
            if ret and pose_base is not None:
                try:
                    points_world = transform_points(pose_base, mesh_points_obj)

                    # 1) PLY 每帧保存
                    if (frame_id - last_cloud_saved) >= SAVE_CLOUD_EVERY_N_FRAMES:
                        last_cloud_saved = frame_id
                        ply_path = PLY_DIR / f"cloud_{frame_id:06d}.ply"
                        save_ply_xyzrgb(ply_path, points_world, mesh_rgb)

                    # 2) 点云PNG 每30帧保存一次（与PLY分开目录）
                    if (frame_id - last_pcimg_saved) >= SAVE_IMAGE_EVERY_N_FRAMES:
                        last_pcimg_saved = frame_id
                        pcimg_path = PCIMG_DIR / f"pc_{frame_id:06d}.png"
                        save_pointcloud_png(
                            pcimg_path,
                            points_world,
                            rgb=mesh_rgb,
                            max_points=20000,   # 只影响渲染速度，不影响PLY保存
                            elev=22.0,
                            azim=35.0
                        )

                        # 如果你还想额外保存"相机图+3D框"，可以取消下面注释，并另建目录
                        # rgb_dir = ensure_dir(OUT_DIR / "rgb_vis")
                        # vis_image = color_image.copy()
                        # info = {"Hz": f"{hz:.2f}", "xyz_cam": xyz_cam, "xyz_base": xyz_arm}
                        # vis_image = draw_dict_to_img(vis_image, info, font_size=1)
                        # if hasattr(tracker, "bbox_corners") and tracker.bbox_corners is not None:
                        #     vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)
                        # cv2.imwrite(str(rgb_dir / f"rgb_{frame_id:06d}.png"), vis_image)

                except Exception as e:
                    logger.exception(f"保存点云/点云图片失败: {e}")

            # 实时窗口（可选，还是显示相机图叠框，便于你调试追踪）
            if ret and vis:
                result = {"Hz": f"{hz:.2f}", "xyz_cam": xyz_cam, "xyz_base": xyz_arm}
                vis_image = color_image.copy()
                vis_image = draw_dict_to_img(vis_image, result, font_size=1)
                if hasattr(tracker, "bbox_corners") and tracker.bbox_corners is not None:
                    vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)

                cv2.imshow("vis_image", vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except Exception as e:
        logger.exception(f"程序运行出错: {e}")

    except KeyboardInterrupt:
        logger.warning("用户中断程序")

    finally:
        cv2.destroyAllWindows()
        gemini_2.stop()
        tracker.release()
        print(f"\n输出目录: {OUT_DIR.resolve()}")
        print(f"PLY 目录: {PLY_DIR.resolve()}")
        print(f"点云PNG目录: {PCIMG_DIR.resolve()}")
def cam2base(pose_cam, T_cam_to_base):
    """
    pose_base = T_cam_to_base @ pose_cam
    """
    T_cam_to_base = np.asarray(T_cam_to_base).reshape(4, 4)

    if isinstance(pose_cam, (list, tuple)):
        pose_cam = np.array(pose_cam).reshape(4, 4)
    elif isinstance(pose_cam, np.ndarray):
        if pose_cam.shape == (4, 4):
            pass
        elif pose_cam.size == 16:
            pose_cam = pose_cam.reshape(4, 4)
        else:
            raise ValueError(f"pose_cam 应该是16个元素的一维数组或4x4矩阵，当前形状为: {pose_cam.shape}")
    else:
        raise TypeError(f"pose_cam 类型不支持: {type(pose_cam)}")

    return T_cam_to_base @ pose_cam


##========================================================================
# ====== add: stream object pose / point cloud ======
from typing import Iterator, Optional, Tuple

def track_pose_stream(
    text_prompt: str,
    mesh_file: Path | str,
    device_index: int = 1,
    server_url: str = "tcp://127.0.0.1:5555",
    T_cam_to_base=np.eye(4),
    mesh_n_points: int = 1024,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    逐帧产出:
      frame_id: int
      pose_base: (4,4) np.ndarray
      points_world: (N,3) np.ndarray   # mesh点云已变换到 base/world
    """
    tracker = RemoteFoundationPose(server_url)
    cam = OrbbecRGBDCamera(device_index=device_index)

    mesh_file = Path(mesh_file)
    mesh_points_obj = sample_points_from_mesh(mesh_file, n_points=mesh_n_points)

    frame_id = 0
    try:
        cam.start()
        intrinsic = cam.get_intrinsic()

        while True:
            frame_id += 1
            if max_frames is not None and frame_id > max_frames:
                break

            color_image, depth_image, depth_data = cam.get_frames()
            if color_image is None or depth_image is None:
                continue

            if not tracker.initialized and intrinsic is not None:
                ok = tracker.init(
                    text_prompt=text_prompt,
                    cam_K=intrinsic,
                    mesh_file=mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )
                if not ok:
                    time.sleep(0.05)
                    continue

            ret, pose_cam = tracker.update(color_image, depth_image)
            if not ret:
                continue

            pose_cam = np.array(pose_cam, dtype=np.float32).reshape(4, 4)
            pose_base = cam2base(pose_cam, T_cam_to_base)
            pose_base = np.array(pose_base, dtype=np.float32).reshape(4, 4)

            points_world = transform_points(pose_base, mesh_points_obj)
            yield frame_id, pose_base, points_world

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            tracker.release()
        except Exception:
            pass


##========================================================================

if __name__ == "__main__":
    text_prompt = "yellow"
    mesh_file = "/home/zyh/wmz/our_data/processed/mangguo/1764919025923146/scaled_mesh.obj"

    # --- 在这里指定你的母文件夹 ---
    output_path = "/home/zyh/wmz/camera/output_data" 
    
    device_index = 1
    server_url = "tcp://127.0.0.1:5555"
    vis = True

    MAIN_MAT = [
        [-0.99857, 0.05338, 0.00201, 544.02],
        [0.03946, 0.76244, -0.64586, 552.76],
        [-0.03601, -0.64486, -0.76345, 628.56],
        [0.0, 0.0, 0.0, 1.0],
    ]
    MAIN_MAT = np.array(MAIN_MAT)
    MAIN_MAT_converted = MAIN_MAT.copy()
    MAIN_MAT_converted[:3, 3] /= 1000.0  # mm -> m

    # 传入 root_output_dir
    track_pose(
        text_prompt, 
        mesh_file, 
        device_index, 
        server_url, 
        MAIN_MAT_converted, 
        vis,
        root_output_dir=output_path # 传入自定义路径
    )