import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
import imageio
import open3d as o3d  # 需要引入 open3d
from rich import print as rprint
import nvdiffrast.torch as dr
# 配置环境
from logger import logger
import config
config.setup_env()

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # 修改：向上3级到项目根目录
FP_DIR = REPO_ROOT / "FoundationPose"
GSA_DIR = REPO_ROOT / "GroundedSegmentAnything"
# 让 Python 能找到 FoundationPose 下的 estimater.py / datareader.py
sys.path.insert(0, str(FP_DIR))
# 让 Python 能找到 GroundingDINO / segment_anything
sys.path.insert(0, str(GSA_DIR))
sys.path.insert(0, str(GSA_DIR / "GroundingDINO"))
sys.path.insert(0, str(GSA_DIR / "segment_anything"))

# FoundationPose 模块
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from detector import GDSAMDetector
# 假设 Utils 包含这些工具函数，如果没有请确保从 FoundationPose/estimater.py 或其他地方引入
from Utils import draw_posed_3d_box, draw_xyz_axis 
from estimater import depth2xyzmap  # 需要引入这个转换函数

def toOpen3dCloud(points, colors=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return cloud

class FoundationPoseGDSAMTracker:
    def __init__(
        self, 
        text_prompt: str, 
        mesh_file: str, 
        K: np.ndarray, 
        show_vis=False, 
        save_vis=False, 
        save_3d=False,
        save_pose_txt=True,
        debug_dir="./debug_vis",
        device="cuda"
    ):
        self.text_prompt = text_prompt
        self.mesh_file = mesh_file
        self.show_vis = show_vis
        self.save_vis = save_vis
        self.save_3d = save_3d
        self.save_pose_txt = save_pose_txt
        self.debug_dir = Path(debug_dir)
        self.device = device
        
        # 处理 K 矩阵 (兼容 4x4 或 3x3)
        if K.shape == (4, 4):
            self.K = K[:3, :3]
        else:
            self.K = K

        # 初始化目录
        if self.save_vis or self.save_3d:
            if self.debug_dir.exists():
                shutil.rmtree(self.debug_dir)

        # 加载 Mesh 和 Bounds
        if not os.path.exists(mesh_file):   # 检查mesh文件是否存在
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_file}")
        
        self.mesh = trimesh.load(mesh_file)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

        # 初始化 FoundationPose 组件
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=str(self.debug_dir),
            debug=0, # 内部 debug 关闭，由外层控制
            glctx=self.glctx
        )
        
        # 懒加载 Detector
        self.detector = None
        self.frame_idx = 0

    def _ensure_detector(self):
        if self.detector is None:
            print("Loading GDSAM Detector...")
            self.detector = GDSAMDetector(
                config.WEIGHTS["dino_config"],
                config.WEIGHTS["dino_checkpoint"],
                config.WEIGHTS["sam_checkpoint"],
                device=self.device
            )

    def init(self, color: np.ndarray, depth: np.ndarray):
        """
        第一帧初始化：使用 DINO+SAM 生成 mask 并注册
        返回: (success, pose_4x4)
        """
        self._ensure_detector()
        self.frame_idx = 0

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        imageio.imwrite(self.debug_dir / "init_depth.png", (depth * 255).astype(np.uint8))
        # imageio.imwrite(self.debug_dir / "init_depth.png", depth)
        imageio.imwrite(self.debug_dir / "init_color.png", color_rgb)
        
        # 1. 生成 Mask
        logger.info("正在生成 Mask...")
        mask = self.detector.detect(color_rgb, self.text_prompt)
        
        if not np.any(mask):
            logger.error("未检测到任何目标物体，初始化失败。")
            return False, np.eye(4)
            
        imageio.imwrite(self.debug_dir / "init_mask.png", (mask * 255).astype(np.uint8))

        # 2. 注册 (Register)
        logger.info("正在注册 FoundationPose 第一帧 ...")
        pose = self.est.register(K=self.K, rgb=color_rgb, depth=depth, ob_mask=mask, iteration=5)
        logger.info("第一帧注册完成")
        rprint(pose)

        # === 补充 save_3d 逻辑 ===
        if self.save_3d:
            self._save_debug_3d(pose, depth, color_rgb)

        self._vis_post_process(color_rgb, pose, "init")
        return True, pose

    def update(self, color: np.ndarray, depth: np.ndarray):
        """
        后续帧追踪
        """
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        self.frame_idx += 1
        
        try:
            pose = self.est.track_one(rgb=color_rgb, depth=depth, K=self.K, iteration=2)
            self._vis_post_process(color_rgb, pose, f"frame_{self.frame_idx:04d}")
            return True, pose
        except Exception as e:
            print(f"Tracking failed: {e}")
            return False, np.eye(4)

    def _save_debug_3d(self, pose, depth, color):
        """保存 3D 调试文件 (model_tf.obj 和 scene_complete.ply)"""
        logger.info(f"Saving 3D debug files to {self.debug_dir}...")
        
        # 1. 保存变换后的物体模型 (model_tf.obj)
        m = self.mesh.copy()
        m.apply_transform(pose)
        m.export(self.debug_dir / "model_tf.obj")
        
        # 2. 保存完整场景点云 (scene_complete.ply)
        xyz_map = depth2xyzmap(depth, self.K)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(str(self.debug_dir / "scene_complete.ply"), pcd)

    def _vis_post_process(self, color, pose, tag):
        """可视化处理内部函数"""
        # 保存位姿 txt (如果开启)
        if self.save_pose_txt:
            pose_out_dir = self.debug_dir / "ob_in_cam"
            pose_out_dir.mkdir(exist_ok=True)
            np.savetxt(pose_out_dir / f"{tag}.txt", pose.reshape(4, 4))

        if not (self.show_vis or self.save_vis):
            return

        # 计算物体中心 pose (用于可视化 bbox)
        center_pose = pose @ np.linalg.inv(self.to_origin)
        
        # 绘制
        vis_img = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
        vis_img = draw_xyz_axis(
            vis_img, ob_in_cam=center_pose, scale=0.1, K=self.K, 
            thickness=3, transparency=0, is_input_rgb=True
        )

        if self.save_vis:
            vis_out_dir = self.debug_dir / "track_vis"
            vis_out_dir.mkdir(exist_ok=True)
            out_path = vis_out_dir / f"{tag}.png"
            imageio.imwrite(out_path, vis_img)
        
        if self.show_vis:
            cv2.imshow("Tracking", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def get_bbox_corners(self):
        """
        [新增接口] 获取 Mesh 的 8 个顶点坐标
        用于发送给客户端，让客户端自己画框
        """
        # trimesh 的 bounds 是 [min, max]，我们需要把它变成 8 个角点
        # trimesh.bounds.corners 属性可以直接获取8个点
        # 但是 FoundationPose 内部通常把物体中心归零了，需要注意 offset
        # 这里返回原始 mesh 坐标系下的 corners 即可，因为 pose 包含了 transform
        return trimesh.bounds.corners(self.mesh.bounds)

