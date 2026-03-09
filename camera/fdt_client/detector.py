import torch
import numpy as np
import cv2
from PIL import Image

import config
config.setup_env()

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor

class GDSAMDetector:
    def __init__(self, config_path, dino_ckpt, sam_ckpt, device="cuda"):
        self.device = device
        
        # --- Load DINO ---
        args = SLConfig.fromfile(str(config_path))
        args.device = device
        self.dino_model = build_model(args)
        checkpoint = torch.load(dino_ckpt, map_location="cpu")
        self.dino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.dino_model.eval()
        self.dino_model = self.dino_model.to(device)
        
        # --- Load SAM ---
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(device)
        self.sam_predictor = SamPredictor(sam)

    # def detect(self, image_rgb: np.ndarray, text_prompt: str, box_threshold=0.6, text_threshold=0.25):
    def detect(self, image_rgb: np.ndarray, text_prompt: str, box_threshold=0.4, text_threshold=0.2):

        """
        输入: RGB numpy图像
        输出: Union mask (bool numpy array)
        """
        # 1. DINO 推理
        image_pil = Image.fromarray(image_rgb)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)
        image_tensor = image_tensor.to(self.device)

        caption = text_prompt.lower().strip()
        if not caption.endswith("."): 
            caption += "."

        with torch.no_grad():
            outputs = self.dino_model(image_tensor[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        # 应用 box_threshold 和 text_threshold 过滤
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        # 如果没有检测到框，返回空mask
        if boxes_filt.shape[0] == 0:
            print(f"[Warning] No object found for prompt: {text_prompt}")
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        # 使用 text_threshold 过滤
        tokenlizer = self.dino_model.tokenizer
        tokenized = tokenlizer(caption)

        # 根据text_threshold过滤每个检测框的置信度
        pred_phrases = []
        filtered_boxes = []
        for logit, box in zip(logits_filt, boxes_filt):
            phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if len(phrase) > 0:  # 确保有匹配的短语
                pred_phrases.append(phrase)
                filtered_boxes.append(box)
        
        if len(filtered_boxes) == 0:
            print(f"[Warning] No object found for prompt: {text_prompt} after text threshold filtering")
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        
        # 转换为tensor
        boxes_filt = torch.stack(filtered_boxes)

        # 2. SAM 推理
        self.sam_predictor.set_image(image_rgb)
        
        H, W = image_rgb.shape[:2]
        boxes_xyxy = boxes_filt.clone()
        boxes_xyxy[:, [0, 2]] *= W
        boxes_xyxy[:, [1, 3]] *= H
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
        
        boxes_xyxy = boxes_xyxy.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_rgb.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # 合并 Mask
        union_mask = torch.any(masks.squeeze(1), dim=0).detach().cpu().numpy()
        return union_mask


if __name__ == "__main__":

    from pathlib import Path
    import imageio
    
    # 设置输入参数
    image_path = "vis/color_1768445468.57705.png"
    text_prompt = "yellow"
    output_dir = "./logs/detection_results"
    
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取输入图片
    print(f"正在读取图片: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"错误: 无法读取图片 - {image_path}")
        exit(1)
    
    # 转换颜色格式 (BGR to RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 初始化检测器
    print("正在初始化检测器...")
    detector = GDSAMDetector(
        config_path=config.WEIGHTS["dino_config"],
        dino_ckpt=config.WEIGHTS["dino_checkpoint"],
        sam_ckpt=config.WEIGHTS["sam_checkpoint"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"正在检测 '{text_prompt}' ...")
    mask = detector.detect(image_rgb, text_prompt)
    
    print(f"检测完成！Mask 形状: {mask.shape}")
    print(f"Mask 数据类型: {mask.dtype}")
    print(f"Mask 中 True 像素数量: {np.sum(mask)}")
    
    # 创建带掩码的可视化图像
    vis_image = image_rgb.copy()
    vis_image[mask] = (vis_image[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)  # 红色高亮
    
    # 保存结果
    image_name = Path(image_path).stem
    mask_output_path = output_dir / f"{image_name}_mask.png"
    vis_output_path = output_dir /f"{image_name}_visualization.png"
    original_output_path = output_dir / f"{image_name}_original.png"
    
    # 保存原始图片
    imageio.imwrite(original_output_path, image_rgb)
    
    # 保存二值掩码
    imageio.imwrite(mask_output_path, (mask * 255).astype(np.uint8))
    
    # 保存可视化结果
    imageio.imwrite(vis_output_path, vis_image)
    
    print(f"结果已保存 {output_dir}:")
    print(f"  - 原始图片: {original_output_path}")
    print(f"  - 检测掩码: {mask_output_path}")
    print(f"  - 可视化图: {vis_output_path}")