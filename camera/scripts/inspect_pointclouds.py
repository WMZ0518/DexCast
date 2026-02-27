#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_pointclouds.py
对 logs/mesh_cloud/<RUN_TAG>/ply/ 下生成的 ASCII PLY 点云做统计与 summary。

用法示例：
  python inspect_pointclouds.py logs/mesh_cloud/20260121_153000/ply
  python inspect_pointclouds.py logs/mesh_cloud/20260121_153000/ply --csv summary.csv
  python inspect_pointclouds.py logs/mesh_cloud/20260121_153000/ply --limit 200
  python inspect_pointclouds.py logs/mesh_cloud/20260121_153000/ply --check-first-shape
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np


PLY_NAME_RE = re.compile(r"cloud_world_(\d+)_")  # 从文件名提取 frame_id


@dataclass
class CloudStats:
    path: Path
    frame_id: Optional[int]
    n_points: int
    mins: np.ndarray      # (3,)
    maxs: np.ndarray      # (3,)
    centroid: np.ndarray  # (3,)
    extent: np.ndarray    # (3,) = max-min
    mean_dist_to_origin: float
    centroid_dist_to_origin: float
    approx_radius: float  # 以 centroid 为中心的点云半径近似（max distance）

    file_bytes: int
    ok: bool
    error: Optional[str] = None


def parse_frame_id(p: Path) -> Optional[int]:
    m = PLY_NAME_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_ascii_ply_xyz(ply_path: Path) -> np.ndarray:
    """
    读取 ASCII PLY，仅取 x y z 三列（忽略 rgb），返回 (N,3) float32。
    若文件损坏/格式不符，抛异常。
    """
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines or not lines[0].strip().startswith("ply"):
        raise ValueError("Not a PLY file or empty")

    # 解析 header
    n_verts = None
    header_end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            parts = s.split()
            if len(parts) == 3:
                n_verts = int(parts[2])
        if s == "end_header":
            header_end = i
            break

    if n_verts is None or header_end is None:
        raise ValueError("PLY header missing element vertex or end_header")

    data_lines = lines[header_end + 1 : header_end + 1 + n_verts]
    if len(data_lines) < n_verts:
        raise ValueError(f"PLY truncated: expected {n_verts} vertices, got {len(data_lines)}")

    # 解析 xyz（前三列）
    xyz = np.empty((n_verts, 3), dtype=np.float32)
    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Bad vertex line at {i}: {line[:80]}")
        xyz[i, 0] = float(parts[0])
        xyz[i, 1] = float(parts[1])
        xyz[i, 2] = float(parts[2])

    return xyz


def print_first_cloud_shape(ply_path: Path) -> None:
    """
    读取并打印一个点云文件的 shape/dtype/基本范围信息，用于快速核验 N×3。
    """
    xyz = load_ascii_ply_xyz(ply_path)
    print("\n" + "=" * 72)
    print("First Cloud Shape Check")
    print("=" * 72)
    print(f"File: {ply_path.name}")
    print(f"xyz dtype: {xyz.dtype}")
    print(f"xyz shape: {xyz.shape}   (expected: N x 3)")

    # 额外输出一些 sanity check
    if xyz.shape[0] > 0:
        print(f"First point: {xyz[0].tolist()}")
        print(f"Min xyz: {xyz.min(axis=0).tolist()}")
        print(f"Max xyz: {xyz.max(axis=0).tolist()}")


def compute_stats_for_cloud(ply_path: Path) -> CloudStats:
    frame_id = parse_frame_id(ply_path)
    file_bytes = ply_path.stat().st_size

    try:
        xyz = load_ascii_ply_xyz(ply_path)
        n = int(xyz.shape[0])
        if n == 0:
            raise ValueError("Empty point cloud")

        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        centroid = xyz.mean(axis=0)
        extent = maxs - mins

        dist_origin = np.linalg.norm(xyz, axis=1)
        mean_dist_to_origin = float(dist_origin.mean())
        centroid_dist_to_origin = float(np.linalg.norm(centroid))

        # 近似半径：点到 centroid 的最大距离
        approx_radius = float(np.linalg.norm(xyz - centroid[None, :], axis=1).max())

        return CloudStats(
            path=ply_path,
            frame_id=frame_id,
            n_points=n,
            mins=mins,
            maxs=maxs,
            centroid=centroid,
            extent=extent,
            mean_dist_to_origin=mean_dist_to_origin,
            centroid_dist_to_origin=centroid_dist_to_origin,
            approx_radius=approx_radius,
            file_bytes=file_bytes,
            ok=True,
        )

    except Exception as e:
        return CloudStats(
            path=ply_path,
            frame_id=frame_id,
            n_points=0,
            mins=np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            maxs=np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            centroid=np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            extent=np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            mean_dist_to_origin=float("nan"),
            centroid_dist_to_origin=float("nan"),
            approx_radius=float("nan"),
            file_bytes=file_bytes,
            ok=False,
            error=str(e),
        )


def summarize(all_stats: List[CloudStats]) -> str:
    ok_stats = [s for s in all_stats if s.ok]
    bad_stats = [s for s in all_stats if not s.ok]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Point Cloud Inspect Summary")
    lines.append("=" * 72)
    lines.append(f"Total files: {len(all_stats)}")
    lines.append(f"OK: {len(ok_stats)}   Failed: {len(bad_stats)}")

    if bad_stats:
        lines.append("\nFailed examples (up to 5):")
        for s in bad_stats[:5]:
            lines.append(f"  - {s.path.name}: {s.error}")

    if not ok_stats:
        lines.append("\nNo valid point clouds. Stop.")
        return "\n".join(lines)

    npts = np.array([s.n_points for s in ok_stats], dtype=np.int64)
    sizes = np.array([s.file_bytes for s in ok_stats], dtype=np.int64)

    lines.append("\nPoints per cloud:")
    lines.append(f"  min/median/mean/max = {npts.min()} / {int(np.median(npts))} / {npts.mean():.1f} / {npts.max()}")
    lines.append(f"  total points (sum) = {npts.sum()}")

    lines.append("\nFile size (bytes):")
    lines.append(f"  min/median/mean/max = {sizes.min()} / {int(np.median(sizes))} / {sizes.mean():.1f} / {sizes.max()}")

    # 全局包围盒
    gmin = np.min(np.stack([s.mins for s in ok_stats], axis=0), axis=0)
    gmax = np.max(np.stack([s.maxs for s in ok_stats], axis=0), axis=0)
    gextent = gmax - gmin
    lines.append("\nGlobal AABB (world frame):")
    lines.append(f"  min = [{gmin[0]:.4f}, {gmin[1]:.4f}, {gmin[2]:.4f}]")
    lines.append(f"  max = [{gmax[0]:.4f}, {gmax[1]:.4f}, {gmax[2]:.4f}]")
    lines.append(f"  extent = [{gextent[0]:.4f}, {gextent[1]:.4f}, {gextent[2]:.4f}]")

    # 与世界原点的距离分布
    cdist = np.array([s.centroid_dist_to_origin for s in ok_stats], dtype=np.float32)
    lines.append("\nCentroid distance to world origin |O_world| (meters):")
    lines.append(f"  min/median/mean/max = {cdist.min():.4f} / {np.median(cdist):.4f} / {cdist.mean():.4f} / {cdist.max():.4f}")

    # 最远/最近的点云
    nearest = ok_stats[int(np.argmin(cdist))]
    farthest = ok_stats[int(np.argmax(cdist))]
    lines.append("\nNearest / farthest cloud centroid:")
    lines.append(f"  nearest: {nearest.path.name}  |c|={nearest.centroid_dist_to_origin:.4f}  c={nearest.centroid}")
    lines.append(f"  farthest: {farthest.path.name} |c|={farthest.centroid_dist_to_origin:.4f} c={farthest.centroid}")

    # 缺帧检查（仅当能提取 frame_id）
    frame_ids = sorted([s.frame_id for s in ok_stats if s.frame_id is not None])
    if frame_ids:
        missing = []
        for a, b in zip(frame_ids[:-1], frame_ids[1:]):
            if b > a + 1:
                missing.extend(range(a + 1, b))
        lines.append("\nFrame ID check (from filename):")
        lines.append(f"  extracted frame_id count = {len(frame_ids)}")
        lines.append(f"  frame_id range = [{frame_ids[0]}, {frame_ids[-1]}]")
        if missing:
            lines.append(f"  missing frame_ids (first 20 shown): {missing[:20]}  (total missing={len(missing)})")
        else:
            lines.append("  missing frame_ids: none detected")
    else:
        lines.append("\nFrame ID check: frame_id not found in filenames (skip).")

    return "\n".join(lines)


def export_csv(stats: List[CloudStats], csv_path: Path) -> None:
    import csv
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "frame_id", "ok", "error",
            "n_points", "file_bytes",
            "min_x", "min_y", "min_z",
            "max_x", "max_y", "max_z",
            "centroid_x", "centroid_y", "centroid_z",
            "extent_x", "extent_y", "extent_z",
            "centroid_dist_to_origin", "mean_dist_to_origin", "approx_radius"
        ])
        for s in stats:
            w.writerow([
                s.path.name, s.frame_id, s.ok, s.error or "",
                s.n_points, s.file_bytes,
                float(s.mins[0]), float(s.mins[1]), float(s.mins[2]),
                float(s.maxs[0]), float(s.maxs[1]), float(s.maxs[2]),
                float(s.centroid[0]), float(s.centroid[1]), float(s.centroid[2]),
                float(s.extent[0]), float(s.extent[1]), float(s.extent[2]),
                float(s.centroid_dist_to_origin), float(s.mean_dist_to_origin), float(s.approx_radius)
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_dir", type=str, help="PLY directory, e.g. logs/mesh_cloud/<RUN_TAG>/ply")
    ap.add_argument("--pattern", type=str, default="cloud_world_*.ply", help="glob pattern")
    ap.add_argument("--limit", type=int, default=0, help="limit number of files (0 means all)")
    ap.add_argument("--csv", type=str, default="", help="export per-file stats to CSV")
    ap.add_argument(
        "--check-first-shape",
        default=True,   
        action="store_true",
        help="Read and print shape/dtype/min/max for the FIRST .ply file only (sanity check N×3)."
    )
    args = ap.parse_args()

    ply_dir = Path(args.ply_dir)
    if not ply_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ply_dir}")

    # 只取 .ply，忽略 .tmp
    files = sorted([p for p in ply_dir.glob(args.pattern) if p.suffix == ".ply"])
    if not files:
        raise FileNotFoundError(f"No .ply files found in: {ply_dir}")

    # 新增：检查第一个文件 shape
    if args.check_first_shape:
        print_first_cloud_shape(files[0])

    # summary 统计
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    all_stats: List[CloudStats] = []
    for p in files:
        all_stats.append(compute_stats_for_cloud(p))

    print(summarize(all_stats))

    if args.csv:
        export_csv(all_stats, Path(args.csv))
        print(f"\nCSV exported to: {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()

#python inspect_pointclouds.py /home/zyh/wmz/logs/mesh_cloud/20260121_191326/ply --csv /home/zyh/wmz/logs/mesh_cloud/20260121_191326/summary.csv
