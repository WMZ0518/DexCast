import argparse
import os
import sys
import json
import time
import logging

import numpy as np
import torch
import cv2
import imageio
import trimesh
from PIL import Image
import matplotlib.pyplot as plt

# ========== 你的工程依赖：请确保这两个目录存在 ==========
import sys
from pathlib import Path
import shutil
import hashlib

def sanitize_name(s: str) -> str:
    s = s.strip().lower()
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s) or "prompt"

def make_debug_dir(repo_root: Path, scene_dir: Path, prompt: str) -> Path:
    repo_root = repo_root.resolve()
    scene_dir = scene_dir.resolve()
    prompt_name = sanitize_name(prompt)

    try:
        rel = scene_dir.relative_to(repo_root)  # scene 在 repo 内：保留完整层级
        return repo_root / "debug" / rel / prompt_name
    except ValueError:
        # scene 不在 repo 内：放到 _external，并尽量保留可读性
        tail = Path(*scene_dir.parts[-2:]) if len(scene_dir.parts) >= 2 else Path(scene_dir.name)
        # 可选：再加一个短 hash，避免不同外部路径重名
        h = hashlib.sha1(str(scene_dir).encode("utf-8")).hexdigest()[:8]
        return repo_root / "debug" / "_external" / tail / f"{prompt_name}_{h}"


REPO_ROOT = Path(__file__).resolve().parent  # 脚本放 RepoRoot
FP_DIR = REPO_ROOT / "FoundationPose"
GSA_DIR = REPO_ROOT / "GroundedSegmentAnything"

# 让 Python 能找到 FoundationPose 下的 estimater.py / datareader.py
sys.path.insert(0, str(FP_DIR))

# 让 Python 能找到 GroundingDINO / segment_anything
sys.path.insert(0, str(GSA_DIR))
sys.path.insert(0, str(GSA_DIR / "GroundingDINO"))
sys.path.insert(0, str(GSA_DIR / "segment_anything"))


# -------- GroundingDINO --------
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# -------- segment anything --------
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

# -------- FoundationPose 工程内模块 --------
from estimater import *
from datareader import *


# -------------------- GroundingDINO + SAM 部分 --------------------
def load_image_for_dino(image_path: str):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image_tensor


from typing import Optional

def load_dino_model(model_config_path: str,model_checkpoint_path: str,bert_base_uncased_path: Optional[str],device: str):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model


@torch.no_grad()
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)

    outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4)

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def _show_mask_rgba(mask_hw: np.ndarray, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask_hw.shape
    mask_image = mask_hw.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def _show_box_xyxy(box_xyxy: np.ndarray, ax, label: str):
    x0, y0, x1, y1 = box_xyxy.tolist()
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def generate_union_mask_with_dino_sam(
    image_path: str,
    text_prompt: str,
    dino_model,
    predictor: SamPredictor,
    box_threshold: float,
    text_threshold: float,
    device: str,
    save_dir: Optional[str],
):
    """
    返回：union_mask_bool (H, W) 的 bool numpy
    同时可选保存：raw_image / 叠加可视化 / 二值mask(png) / 实例信息(json)
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    # DINO 输入
    image_pil, image_tensor = load_image_for_dino(image_path)
    W, H = image_pil.size  # PIL: (W,H)

    # DINO 推理
    boxes_filt, pred_phrases = get_grounding_output(
        dino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=device
    )

    # SAM 输入
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"cv2 读取失败：{image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # boxes: cxcywh(归一化) -> xyxy(像素)
    boxes_xyxy = boxes_filt.clone()
    for i in range(boxes_xyxy.size(0)):
        boxes_xyxy[i] = boxes_xyxy[i] * torch.Tensor([W, H, W, H])
        boxes_xyxy[i][:2] -= boxes_xyxy[i][2:] / 2
        boxes_xyxy[i][2:] += boxes_xyxy[i][:2]
    boxes_xyxy_cpu = boxes_xyxy.cpu()

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy_cpu, image_rgb.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # masks: (N, 1, H, W) bool

    if masks is None or masks.shape[0] == 0:
        raise RuntimeError("未检测到任何 mask；请检查 text_prompt 或阈值（box_threshold/text_threshold）。")

    # 合并多个实例 mask -> union
    union = torch.any(masks.squeeze(1), dim=0)  # (H,W) bool
    union_np = union.detach().cpu().numpy().astype(bool)

    if save_dir is not None:
        # 1) 原图
        image_pil.save(os.path.join(save_dir, "raw_image.jpg"))

        # 2) 二值 mask：0/255
        mask_png = (union_np.astype(np.uint8) * 255)
        Image.fromarray(mask_png).save(os.path.join(save_dir, "mask_binary.png"))

        # 3) 可视化叠加图（实例mask + box + label）
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        for m in masks:
            _show_mask_rgba(m.squeeze(0).detach().cpu().numpy().astype(np.uint8), plt.gca(), random_color=True)
        for b, lab in zip(boxes_xyxy_cpu, pred_phrases):
            _show_box_xyxy(b.numpy(), plt.gca(), lab)
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()

        # 4) 保存实例元信息（可选调试）
        inst = [{"value": 0, "label": "background"}]
        v = 0
        for lab, b in zip(pred_phrases, boxes_xyxy_cpu):
            v += 1
            if "(" in lab and lab.endswith(")"):
                name, logit = lab.split("(")
                logit = logit[:-1]
                try:
                    logit_f = float(logit)
                except:
                    logit_f = None
            else:
                name, logit_f = lab, None
            inst.append({"value": v, "label": name.strip(), "logit": logit_f, "box_xyxy": b.numpy().tolist()})
        with open(os.path.join(save_dir, "mask_meta.json"), "w") as f:
            json.dump(inst, f, ensure_ascii=False, indent=2)

    return union_np


def load_binary_mask(mask_path: str):
    m = Image.open(mask_path).convert("L")
    m = np.array(m)
    return (m > 0)


# -------------------- 合并后的主流程 --------------------
def main():
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser("DINO+SAM -> FoundationPose (all-in-one)")

    # ---- DINO/SAM 参数 ----
    parser.add_argument("--config", type=str, required=True, help="GroundingDINO config .py")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="GroundingDINO checkpoint .pth")
    parser.add_argument("--bert_base_uncased_path", type=str, default=None, help="可选：本地 bert_base_uncased 路径")

    parser.add_argument("--sam_version", type=str, default="vit_h", help="vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM checkpoint .pth")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="SAM-HQ checkpoint .pth")
    parser.add_argument("--use_sam_hq", action="store_true", help="使用 sam-hq")

    parser.add_argument("--text_prompt", type=str, required=True, help='例如 "mango"')
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda", help='例如 "cuda" 或 "cpu"')

    # ---- FoundationPose 参数 ----
    parser.add_argument("--mesh_file", type=str, required=True)
    parser.add_argument("--test_scene_dir", type=str, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=3)
    parser.add_argument("--debug_dir", type=str, default=None)

    # ---- mask 缓存/输出 ----
    parser.add_argument("--mask_dir", type=str, default=None, help="保存mask的目录；默认= test_scene_dir/mask")
    parser.add_argument("--reuse_mask", action="store_true", help="若 mask_binary.png 已存在则复用，不再重新生成")

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # mask_dir 仍然可以放 scene 内（你现在的逻辑没问题）
    if args.mask_dir is None:
        args.mask_dir = os.path.join(args.test_scene_dir, "mask")
    os.makedirs(args.mask_dir, exist_ok=True)

    # debug_dir：集中到 RepoRoot/debug 下，并映射 scene 层级
    repo_root = REPO_ROOT  # 你前面定义的 REPO_ROOT = Path(__file__).resolve().parent
    scene_dir = Path(args.test_scene_dir)

    if args.debug_dir is None:
        debug_dir_path = make_debug_dir(repo_root, scene_dir, args.text_prompt)
    else:
        debug_dir_path = Path(args.debug_dir).resolve()

    debug_dir = str(debug_dir_path)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    # 清空并重建子目录（跨平台安全）
    shutil.rmtree(debug_dir_path, ignore_errors=True)
    (debug_dir_path / "track_vis").mkdir(parents=True, exist_ok=True)
    (debug_dir_path / "ob_in_cam").mkdir(parents=True, exist_ok=True)

    # 读取 mesh + bbox
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # 初始化 FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    # 初始化 reader
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # 第0帧：准备 mask（生成或复用）
    mask_path = os.path.join(args.mask_dir, "mask_binary.png")
    if args.reuse_mask and os.path.exists(mask_path):
        ob_mask0 = load_binary_mask(mask_path)
    else:
        # 初始化 DINO + SAM
        dino_model = load_dino_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)

        if args.use_sam_hq:
            if args.sam_hq_checkpoint is None:
                raise ValueError("use_sam_hq=True 但未提供 --sam_hq_checkpoint")
            sam = sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(args.device)
        else:
            sam = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device)

        predictor = SamPredictor(sam)

        # 用 reader 的首帧文件路径生成 mask（更稳：避免你手写 ./rgb/000000.png 路径对不上）
        if not hasattr(reader, "color_files") or len(reader.color_files) == 0:
            raise RuntimeError("reader.color_files 为空，无法定位首帧图像路径。")

        frame0_path = reader.color_files[0]
        ob_mask0 = generate_union_mask_with_dino_sam(
            image_path=frame0_path,
            text_prompt=args.text_prompt,
            dino_model=dino_model,
            predictor=predictor,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
            save_dir=args.mask_dir,
        )

    # ---- 主循环：register + track ----
    time_start_all = time.perf_counter()
    track_times = []
    reg_time = 0.0

    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        t_frame_start = time.perf_counter()
        if i == 0:
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask0.astype(bool), iteration=args.est_refine_iter)
            reg_time = time.perf_counter() - t_frame_start
            print(f"\n{'*'*20} FRAME 0 REGISTER DONE {'*'*20}")
            print(f">> Registration Time: {reg_time:.4f}s")
            print(f"{'*'*63}\n")

            if args.debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f"{debug_dir}/model_tf.obj")
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
            t_track = time.perf_counter() - t_frame_start
            track_times.append(t_track)
            if i % 10 == 0:
                print(f"Processing Frame {i}/{len(reader.color_files)}, Curr Track Time: {t_track:.4f}s")

        # 保存位姿
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        # 可视化
        if args.debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                thickness=3, transparency=0, is_input_rgb=True
            )

        if args.debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)

    # ---- 性能统计 ----
    print("\n" + "█" * 40)
    print("      FINAL PERFORMANCE REPORT")
    print("█" * 40)
    print(f" Total Frames : {len(reader.color_files)}")
    print(f" Registration : {reg_time:.4f} sec")
    if len(track_times) > 0:
        avg_track = sum(track_times) / len(track_times)
        print(f" Avg Tracking : {avg_track:.4f} sec/frame")
        print(f" Tracking FPS : {1.0 / avg_track:.2f}")
    print(f" Total Time   : {time.perf_counter() - time_start_all:.2f} sec")
    print("█" * 40 + "\n")


if __name__ == "__main__":
    main()
