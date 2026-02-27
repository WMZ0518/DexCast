# foundationpose_gdsam_tracker.py
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import cv2
import imageio
import trimesh
from PIL import Image
import matplotlib.pyplot as plt

from fp_gdsam_utils import make_debug_dir, show_mask_rgba, show_box_xyxy


class FoundationPoseGDSAMTracker:
    """
    将 GroundingDINO + SAM（或 SAM-HQ） + FoundationPose 封装成:
      - init(bgr, depth, text_prompt): 首帧生成 union mask + register
      - update(bgr, depth): 后续帧 track_one
      - save_pose/save_vis: 专门负责落盘
    """

    # --------------------------- 初始化 ---------------------------
    def __init__(
        self,
        mesh_file: str,
        K: np.ndarray,                           # 3x3 内参
        test_scene_dir: str,                     # 用于 debug_dir 映射与落盘
        repo_root: Optional[str] = None,         # 默认=本文件所在目录
        device: str = "cuda",

        # ---- GroundingDINO ----
        dino_config: str = "GroundedSegmentAnything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        dino_checkpoint: str = "GroundedSegmentAnything/groundingdino_swint_ogc.pth",
        bert_base_uncased_path: Optional[str] = None,

        # ---- SAM ----
        sam_version: str = "vit_h",
        sam_checkpoint: str = "GroundedSegmentAnything/sam_vit_h_4b8939.pth",
        use_sam_hq: bool = False,
        sam_hq_checkpoint: Optional[str] = None,

        # ---- 默认阈值（不在 init/update 传）----
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,

        # ---- FoundationPose ----
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 3,

        # ---- mask/debug 落盘 ----
        mask_dir: Optional[str] = None,
        reuse_mask: bool = True,
        debug_dir: Optional[str] = None,
    ):
        logging.getLogger().setLevel(logging.WARNING)

        # 设备与参数
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.debug = int(debug)

        # 关键输入：K 必须 float32（避免 torch matmul dtype 错误）
        self.K = K.astype(np.float32, copy=False)

        self.mesh_file = str(Path(mesh_file).resolve())
        self.test_scene_dir = str(Path(test_scene_dir).resolve())

        # repo root：用于拼接默认的 config/ckpt 相对路径
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent

        # GroundingDINO 路径 resolve
        self.dino_config = self._resolve_path(dino_config)
        self.dino_checkpoint = self._resolve_path(dino_checkpoint)
        self.bert_base_uncased_path = bert_base_uncased_path

        # SAM 路径 resolve
        self.sam_version = sam_version
        self.sam_checkpoint = self._resolve_path(sam_checkpoint)
        self.use_sam_hq = bool(use_sam_hq)
        self.sam_hq_checkpoint = self._resolve_path(sam_hq_checkpoint) if sam_hq_checkpoint else None

        # mask 输出目录
        if mask_dir is None:
            self.mask_dir = str((Path(self.test_scene_dir) / "mask").resolve())
        else:
            self.mask_dir = str(Path(mask_dir).resolve())
        os.makedirs(self.mask_dir, exist_ok=True)
        self.reuse_mask = bool(reuse_mask)
        self.mask_path = os.path.join(self.mask_dir, "mask_binary.png")

        # debug_dir：若不传，则 init 时按 prompt/scene 自动创建
        self.debug_dir = str(Path(debug_dir).resolve()) if debug_dir else None
        if self.debug_dir:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)

        # 延迟加载对象
        self._dino_model = None
        self._sam_predictor = None
        self._T = None
        self._get_phrases_from_posmap = None

        self._mesh = None
        self._to_origin = None
        self._bbox = None
        self._est = None

        # 状态
        self._initialized = False
        self._frame_idx = 0
        self._last_pose = None
        self._text_prompt = None

        # 加载依赖：保证 sys.path 可找到 FoundationPose / GroundedSegmentAnything
        self._setup_sys_path()

        # import FoundationPose 内部模块（保存引用，避免到处 import）
        from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
        from datareader import set_logging_format, set_seed
        import nvdiffrast.torch as dr

        self._FP_ScorePredictor = ScorePredictor
        self._FP_PoseRefinePredictor = PoseRefinePredictor
        self._FP_FoundationPose = FoundationPose
        self._FP_set_logging_format = set_logging_format
        self._FP_set_seed = set_seed
        self._dr = dr

        # 可视化绘制函数（来自 FoundationPose）
        from estimater import draw_posed_3d_box, draw_xyz_axis
        self._draw_posed_3d_box = draw_posed_3d_box
        self._draw_xyz_axis = draw_xyz_axis

        self._FP_set_logging_format()
        self._FP_set_seed(0)

        # torch patch 标记
        self._torch_patched = False

    def _resolve_path(self, p: Optional[str]) -> Optional[str]:
        """把相对路径拼到 repo_root，下游用绝对路径更稳"""
        if p is None:
            return None
        return str((self.repo_root / p).resolve()) if (not os.path.isabs(p)) else p

    def _setup_sys_path(self):
        """确保能 import 到 FoundationPose / GroundedSegmentAnything 及其子模块"""
        fp_dir = (self.repo_root / "FoundationPose").resolve()
        gsa_dir = (self.repo_root / "GroundedSegmentAnything").resolve()

        sys.path.insert(0, str(fp_dir))
        sys.path.insert(0, str(gsa_dir))
        sys.path.insert(0, str(gsa_dir / "GroundingDINO"))
        sys.path.insert(0, str(gsa_dir / "segment_anything"))

    # --------------------------- dtype / torch 清洗逻辑（保留原功能） ---------------------------
    def _patch_torch_to_float32_once(self):
        """
        目标：阻断 numpy float64 -> torch.double 的链路。
        你之前已经做了：
          - torch.set_default_dtype(float32)
          - patch from_numpy / tensor / as_tensor
          - patch set_default_tensor_type（防止改成 DoubleTensor）
        这里做同样事情，但整理到一个函数里，且只执行一次。
        """
        if self._torch_patched:
            return
        self._torch_patched = True

        torch.set_default_dtype(torch.float32)

        # 默认 Tensor 类型固定为 FloatTensor
        if str(self.device).startswith("cuda"):
            default_tensor_type = torch.cuda.FloatTensor
        else:
            default_tensor_type = torch.FloatTensor

        orig_set_default_tensor_type = torch.set_default_tensor_type

        def patched_set_default_tensor_type(t):
            # 如果有人试图设成 DoubleTensor，则强制改成 FloatTensor
            try:
                if isinstance(t, str):
                    if "Double" in t or "double" in t:
                        t = t.replace("Double", "Float").replace("double", "float")
                        return orig_set_default_tensor_type(t)
                else:
                    name = getattr(t, "__name__", "")
                    if "Double" in name:
                        return orig_set_default_tensor_type(default_tensor_type)
            except Exception:
                pass
            return orig_set_default_tensor_type(t)

        # 注意：PyTorch 2.1+ 这个接口 deprecated，但你工程里也在用，因此保持一致
        torch.set_default_tensor_type(default_tensor_type)
        torch.set_default_tensor_type = patched_set_default_tensor_type

        # patch torch.from_numpy：把 torch.double 立刻转成 float
        orig_from_numpy = torch.from_numpy

        def patched_from_numpy(nd):
            t = orig_from_numpy(nd)
            return t.float() if t.dtype == torch.float64 else t

        torch.from_numpy = patched_from_numpy

        # patch torch.tensor
        orig_tensor = torch.tensor

        def patched_tensor(data, *args, **kwargs):
            t = orig_tensor(data, *args, **kwargs)
            return t.float() if (isinstance(t, torch.Tensor) and t.dtype == torch.float64) else t

        torch.tensor = patched_tensor

        # patch torch.as_tensor
        orig_as_tensor = torch.as_tensor

        def patched_as_tensor(data, *args, **kwargs):
            t = orig_as_tensor(data, *args, **kwargs)
            return t.float() if (isinstance(t, torch.Tensor) and t.dtype == torch.float64) else t

        torch.as_tensor = patched_as_tensor

    def _force_float32_recursive(self, obj):
        """递归把结构里的 torch.float64 Tensor 转为 float32"""
        if isinstance(obj, torch.Tensor):
            return obj.float() if obj.dtype == torch.float64 else obj
        if isinstance(obj, dict):
            return {k: self._force_float32_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._force_float32_recursive(v) for v in obj)
        return obj

    def _sanitize_estimator_tensors(self):
        """扫描 self._est.__dict__，将其中 float64 tensor 递归转成 float32"""
        if self._est is None:
            return
        for k, v in list(self._est.__dict__.items()):
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                self._est.__dict__[k] = v.float()
            elif isinstance(v, (dict, list, tuple)):
                self._est.__dict__[k] = self._force_float32_recursive(v)

    # --------------------------- FoundationPose 初始化 ---------------------------
    def _init_foundationpose_if_needed(self):
        if self._est is not None:
            return

        # 1) 先 patch torch，防止 float64 在内部流转成 torch.double
        self._patch_torch_to_float32_once()

        # 2) mesh / bbox / to_origin：尽量确保我们生成的关键数据是 float32
        mesh = trimesh.load(self.mesh_file, process=False)

        model_pts = np.asarray(mesh.vertices, dtype=np.float32)
        model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        to_origin = np.asarray(to_origin, dtype=np.float32)
        extents = np.asarray(extents, dtype=np.float32)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3).astype(np.float32)

        scorer = self._FP_ScorePredictor()
        refiner = self._FP_PoseRefinePredictor()
        glctx = self._dr.RasterizeCudaContext()

        est = self._FP_FoundationPose(
            model_pts=model_pts,            # float32
            model_normals=model_normals,    # float32
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir if self.debug_dir else "",
            debug=self.debug,
            glctx=glctx,
        )

        self._mesh = mesh
        self._to_origin = to_origin
        self._bbox = bbox
        self._est = est

        # 3) 生成后立刻清洗一次 est 缓存张量
        self._sanitize_estimator_tensors()

        # 4) 包裹关键方法：每次调用后都清洗一次（保留你原来的逻辑）
        def wrap_method(name: str):
            if not hasattr(self._est, name):
                return
            orig = getattr(self._est, name)

            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                self._sanitize_estimator_tensors()
                return out

            setattr(self._est, name, wrapped)

        for meth in ("reset_object", "make_rotation_grid", "register", "track_one"):
            wrap_method(meth)

    # --------------------------- GroundingDINO + SAM 加载与推理 ---------------------------
    def _load_dino_and_sam_if_needed(self):
        """只在第一次需要生成 mask 时加载 DINO + SAM"""
        if self._dino_model is not None and self._sam_predictor is not None:
            return

        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

        self._T = T
        self._get_phrases_from_posmap = get_phrases_from_posmap

        args = SLConfig.fromfile(self.dino_config)
        args.device = self.device
        args.bert_base_uncased_path = self.bert_base_uncased_path

        model = build_model(args)
        ckpt = torch.load(self.dino_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        model.eval()
        self._dino_model = model

        from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

        if self.use_sam_hq:
            if self.sam_hq_checkpoint is None:
                raise ValueError("use_sam_hq=True 但 sam_hq_checkpoint=None")
            sam = sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device)
        else:
            sam = sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device)

        self._sam_predictor = SamPredictor(sam)

    def _load_image_for_dino_from_array(self, rgb_u8: np.ndarray):
        """
        输入：HxWx3 RGB uint8
        输出：PIL + 归一化后的 tensor（3xhxw）
        """
        image_pil = Image.fromarray(rgb_u8)
        transform = self._T.Compose(
            [
                self._T.RandomResize([800], max_size=1333),
                self._T.ToTensor(),
                self._T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)
        return image_pil, image_tensor

    @torch.no_grad()
    def _get_grounding_output(self, image_tensor, caption: str):
        """
        DINO 输出：boxes_filt（cxcywh 归一化） + pred_phrases
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        model = self._dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)

        outputs = model(image_tensor[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        filt_mask = logits.max(dim=1)[0] > self.box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        for logit, _box in zip(logits_filt, boxes_filt):
            phrase = self._get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer)
            pred_phrases.append(phrase + f"({str(logit.max().item())[:4]})")

        return boxes_filt, pred_phrases

    def _generate_union_mask(self, rgb_u8: np.ndarray, save_dir: Optional[str]) -> np.ndarray:
        self._load_dino_and_sam_if_needed()

        H, W = rgb_u8.shape[:2]
        _, image_tensor = self._load_image_for_dino_from_array(rgb_u8)

        boxes_filt, pred_phrases = self._get_grounding_output(image_tensor, self._text_prompt)

        predictor = self._sam_predictor
        predictor.set_image(rgb_u8)  # SAM 要 RGB

        # ========= 关键：统一 boxes 的 device/类型 =========
        # boxes_filt 是 CPU（你在 _get_grounding_output 里 .cpu() 了）
        boxes_xyxy = boxes_filt.clone().to("cpu")  # 保证在 CPU
        scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)  # CPU tensor

        # cxcywh(归一化)->xyxy(像素)
        boxes_xyxy = boxes_xyxy * scale
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        # apply_boxes_torch 期望 boxes 在 CPU（transform 通常在 CPU 做）
        boxes_xyxy_cpu = boxes_xyxy

        # 这里返回的 transformed_boxes 默认也是 CPU tensor
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_xyxy_cpu, rgb_u8.shape[:2]
        )

        # predict_torch 需要 boxes 在 SAM model 的 device 上
        sam_dev = next(predictor.model.parameters()).device  # 更可靠：跟随 SAM 模型
        transformed_boxes = transformed_boxes.to(device=sam_dev)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # masks: (N,1,H,W) bool on sam_dev

        if masks is None or masks.shape[0] == 0:
            raise RuntimeError("未检测到任何 mask；请检查 text_prompt 或阈值。")

        union = torch.any(masks.squeeze(1), dim=0)
        union_np = union.detach().cpu().numpy().astype(bool)
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(rgb_u8).save(os.path.join(save_dir, "raw_image.jpg"))
            Image.fromarray((union_np.astype(np.uint8) * 255)).save(os.path.join(save_dir, "mask_binary.png"))

            # 叠加可视化：mask + box + label
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_u8)
            for m in masks:
                show_mask_rgba(m.squeeze(0).detach().cpu().numpy().astype(np.uint8), plt.gca(), random_color=True)
            for b, lab in zip(boxes_xyxy_cpu, pred_phrases):
                show_box_xyxy(b.numpy(), plt.gca(), lab)
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, "grounded_sam_output.jpg"),
                        bbox_inches="tight", dpi=300, pad_inches=0.0)
            plt.close()

            # 保存 meta
            inst = [{"value": 0, "label": "background"}]
            v = 0
            for lab, b in zip(pred_phrases, boxes_xyxy_cpu):
                v += 1
                name, logit_f = lab, None
                if "(" in lab and lab.endswith(")"):
                    name = lab.split("(")[0].strip()
                    try:
                        logit_f = float(lab.split("(")[1][:-1])
                    except Exception:
                        logit_f = None
                inst.append({"value": v, "label": name, "logit": logit_f, "box_xyxy": b.numpy().tolist()})
            with open(os.path.join(save_dir, "mask_meta.json"), "w", encoding="utf-8") as f:
                json.dump(inst, f, ensure_ascii=False, indent=2)

        return union_np

    # --------------------------- 对外 API：init / update ---------------------------
    def init(self, color_bgr_u8: np.ndarray, depth: np.ndarray, text_prompt: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        第0帧初始化：生成 mask + register
        输入：
          - color_bgr_u8: BGR uint8（通常来自 cv2 / reader）
          - depth: 深度图（建议 float32；这里会强转 float32）
          - text_prompt: 例如 "mango"
        输出：
          - (success, pose_4x4)
        """
        try:
            self._text_prompt = text_prompt

            # 1) 创建 debug_dir（按 prompt/scene 分目录），并清空重建子目录
            if self.debug_dir is None:
                debug_dir_path = make_debug_dir(self.repo_root, Path(self.test_scene_dir), text_prompt)
                shutil.rmtree(debug_dir_path, ignore_errors=True)
                (debug_dir_path / "track_vis").mkdir(parents=True, exist_ok=True)
                (debug_dir_path / "ob_in_cam").mkdir(parents=True, exist_ok=True)
                self.debug_dir = str(debug_dir_path)

            # 2) 初始化 FoundationPose（需要 debug_dir 写进去）
            self._init_foundationpose_if_needed()
            self._est.debug_dir = self.debug_dir

            print(f"[SAVE_DIR] debug_dir = {self.debug_dir}")
            print(f"[SAVE_DIR] poses -> {os.path.join(self.debug_dir, 'ob_in_cam')}")
            print(f"[SAVE_DIR] vis   -> {os.path.join(self.debug_dir, 'track_vis')}")
            print(f"[SAVE_DIR] mask  -> {self.mask_dir}")

            # 3) mask：优先复用已有 mask_binary.png（避免重复跑 DINO/SAM）
            if self.reuse_mask and os.path.exists(self.mask_path):
                union_mask = (np.array(Image.open(self.mask_path).convert("L")) > 0)
            else:
                rgb = cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2RGB)
                union_mask = self._generate_union_mask(rgb_u8=rgb, save_dir=self.mask_dir)

            # 4) dtype 检查（你调试时很关键，保留）
            print("[dtype] K:", self.K.dtype)
            print("[dtype] depth:", depth.dtype, "min/max:", float(np.nanmin(depth)), float(np.nanmax(depth)))
            print("[dtype] mesh.v:", self._mesh.vertices.dtype)
            print("[dtype] mesh.n:", self._mesh.vertex_normals.dtype)
            print("[dtype] to_origin:", self._to_origin.dtype)
            print("[check] rot_grid dtype:",
                  getattr(self._est, "rot_grid", None).dtype if hasattr(self._est, "rot_grid") else None)

            # 5) 强制 depth/K 为 float32（避免 float/double matmul）
            depth = depth.astype(np.float32, copy=False)
            self.K = self.K.astype(np.float32, copy=False)

            # 6) 再清洗一次 estimator 缓存张量（保留你原逻辑）
            self._sanitize_estimator_tensors()

            # 7) register：注意 rgb 要 RGB（FoundationPose 原脚本就是 RGB）
            pose = self._est.register(
                K=self.K,
                rgb=cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2RGB),
                depth=depth,
                ob_mask=union_mask.astype(bool),
                iteration=self.est_refine_iter,
            )

            self._initialized = True
            self._frame_idx = 0
            self._last_pose = pose.copy()

            # 8) 保存首帧结果
            self.save_pose(self._frame_idx, pose)
            if self.debug >= 1:
                vis = self._make_vis(color_bgr_u8, pose)
                self.save_vis(self._frame_idx, vis)

            # 9) 可选：导出首帧变换后的模型
            if self.debug >= 3:
                m = self._mesh.copy()
                m.apply_transform(pose)
                m.export(os.path.join(self.debug_dir, "model_tf.obj"))

            return True, pose

        except Exception as e:
            print(f"[FoundationPoseGDSAMTracker] init 失败: {e}")
            return False, None

    def update(self, color_bgr_u8: np.ndarray, depth: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        后续帧：track_one
        输入：
          - color_bgr_u8: BGR uint8
          - depth: 深度图（这里同样强转 float32 更稳）
        输出：
          - (success, pose_4x4)
        """
        if not self._initialized:
            return False, None

        try:
            self._frame_idx += 1

            # 与 init 保持一致：避免 depth 是 float64/uint16 时混 dtype
            depth = depth.astype(np.float32, copy=False)

            pose = self._est.track_one(
                rgb=cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2RGB),
                depth=depth,
                K=self.K,
                iteration=self.track_refine_iter,
            )

            self._last_pose = pose.copy()

            self.save_pose(self._frame_idx, pose)
            if self.debug >= 1:
                vis = self._make_vis(color_bgr_u8, pose)
                if self.debug >= 2:
                    self.save_vis(self._frame_idx, vis)

            return True, pose

        except Exception as e:
            print(f"[FoundationPoseGDSAMTracker] update 失败: {e}")
            return False, None

    # --------------------------- 可视化与保存（单独函数，方便你控制落盘策略） ---------------------------
    def _make_vis(self, color_bgr_u8: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
        """
        生成可视化：3D bbox + 坐标轴
        注意：
          - FoundationPose 的 draw_xyz_axis 里 is_input_rgb=True，所以这里先转 RGB 再画
          - 返回的是 RGB 图像（用于 imageio.imwrite）
        """
        rgb = cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2RGB)
        center_pose = pose_4x4 @ np.linalg.inv(self._to_origin)

        vis = self._draw_posed_3d_box(self.K, img=rgb, ob_in_cam=center_pose, bbox=self._bbox)
        vis = self._draw_xyz_axis(
            vis, ob_in_cam=center_pose, scale=0.1, K=self.K,
            thickness=3, transparency=0, is_input_rgb=True
        )
        return vis  # RGB

    def save_pose(self, frame_idx: int, pose_4x4: np.ndarray):
        """保存位姿到 debug_dir/ob_in_cam/xxxxxx.txt"""
        out_dir = os.path.join(self.debug_dir, "ob_in_cam")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{frame_idx:06d}.txt")
        np.savetxt(out_path, pose_4x4.reshape(4, 4))
        print(f"[SAVE] pose -> {out_path}")

    def save_vis(self, frame_idx: int, vis_rgb: np.ndarray):
        """保存可视化到 debug_dir/track_vis/xxxxxx.png（输入必须是 RGB）"""
        out_dir = os.path.join(self.debug_dir, "track_vis")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{frame_idx:06d}.png")
        imageio.imwrite(out_path, vis_rgb)
        print(f"[SAVE] vis  -> {out_path}")

    def get_last_pose(self) -> Optional[np.ndarray]:
        """获取最近一次 pose（拷贝，避免外部改写内部状态）"""
        return None if self._last_pose is None else self._last_pose.copy()
