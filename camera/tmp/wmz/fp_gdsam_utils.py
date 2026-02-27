# fp_gdsam_utils.py
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def sanitize_name(s: str) -> str:
    """把 prompt/名字清洗成安全的目录名"""
    s = s.strip().lower()
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s) or "prompt"


def make_debug_dir(repo_root: Path, scene_dir: Path, prompt: str) -> Path:
    """
    生成 debug_dir：
    - 如果 scene 在 repo 内：保持相对路径层级
    - 如果 scene 在 repo 外：放到 debug/_external 下，并附加 hash 防止重名
    """
    repo_root = repo_root.resolve()
    scene_dir = scene_dir.resolve()
    prompt_name = sanitize_name(prompt)

    try:
        rel = scene_dir.relative_to(repo_root)
        return repo_root / "debug" / rel / prompt_name
    except ValueError:
        tail = Path(*scene_dir.parts[-2:]) if len(scene_dir.parts) >= 2 else Path(scene_dir.name)
        h = hashlib.sha1(str(scene_dir).encode("utf-8")).hexdigest()[:8]
        return repo_root / "debug" / "_external" / tail / f"{prompt_name}_{h}"


def show_mask_rgba(mask_hw: np.ndarray, ax, random_color: bool = False):
    """用于 SAM mask overlay 的可视化（matplotlib）"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask_hw.shape
    mask_image = mask_hw.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box_xyxy(box_xyxy: np.ndarray, ax, label: str):
    """用于 DINO box 可视化（matplotlib）"""
    x0, y0, x1, y1 = box_xyxy.tolist()
    w, h = x1 - x0, y1 - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)
