import sys
from pathlib import Path

# ========== 路径配置 ==========
# 假设当前文件在工程根目录下
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FP_DIR = REPO_ROOT / "FoundationPose"
GSA_DIR = REPO_ROOT / "GroundedSegmentAnything"

# 模型权重路径 (请根据实际情况修改)
WEIGHTS = {
    "dino_config": GSA_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "dino_checkpoint": GSA_DIR / "groundingdino_swint_ogc.pth",
    "sam_checkpoint": GSA_DIR / "sam_vit_h_4b8939.pth",
    "bert_base_uncased": None, # 可选本地路径
}

# ========== 环境注入 ==========
# 自动将依赖加入 sys.path
def setup_env():
    sys.path.insert(0, str(FP_DIR))
    sys.path.insert(0, str(GSA_DIR))
    sys.path.insert(0, str(GSA_DIR / "GroundingDINO"))
    sys.path.insert(0, str(GSA_DIR / "segment_anything"))