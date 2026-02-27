"""
Simple Reader for handling remote images as numpy arrays (简化版)
只做基本的图像处理，不涉及复杂矩阵变换
"""
import cv2
import numpy as np
from typing import Tuple


class SimpleImageReader:
    """
    简单的图像读取器，用于处理作为numpy数组的远程图像
    """
    def __init__(self, 
                 target_size: Tuple[int, int] = None,
                 zfar: float = np.inf):
        """
        Args:
            target_size: 目标尺寸 (width, height)，可选
            zfar: 最大有效深度值
        """
        self.target_size = target_size
        self.zfar = zfar

    def process_color(self, color_array: np.ndarray) -> np.ndarray:
        """
        处理彩色图像数组
        Args:
            color_array: RGB或BGR格式的图像数组 (H, W, 3)
        Returns:
            处理后的RGB图像数组 (H, W, 3)
        """
        # 确保输入是3维数组
        if len(color_array.shape) != 3:
            raise ValueError(f"Color array must be 3D, got shape {color_array.shape}")
        
        # 如果是BGR格式，转换为RGB
        if color_array.shape[-1] == 3:
            # 检测是否为BGR格式
            if np.mean(color_array[:, :, 0]) > np.mean(color_array[:, :, 2]):  # R channel has lower values in BGR
                color_rgb = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
            else:
                color_rgb = color_array
        else:
            raise ValueError(f"Color array must have 3 channels, got {color_array.shape[-1]}")
        
        # 如果指定了目标尺寸，则调整大小
        if self.target_size is not None:
            color_rgb = cv2.resize(color_rgb, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        return color_rgb

    def process_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """
        处理深度图像数组
        Args:
            depth_array: 深度图数组 (H, W)，单位通常是米
        Returns:
            处理后的深度图数组 (H, W)
        """
        # 确保输入是2维数组
        if len(depth_array.shape) != 2:
            if len(depth_array.shape) == 3 and depth_array.shape[-1] == 1:
                depth_array = depth_array.squeeze(-1)
            else:
                raise ValueError(f"Depth array must be 2D, got shape {depth_array.shape}")
        
        # 将深度值从毫米转换为米，确保结果仍为ndarray类型
        depth_array = depth_array.astype(np.float32) / 1000.0

        # 如果指定了目标尺寸，则调整大小
        if self.target_size is not None:
            depth_array = cv2.resize(depth_array, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 应用深度范围限制
        depth_array[(depth_array < 0.001) | (depth_array >= self.zfar)] = 0
        
        return depth_array

    def get_processed_pair(self, color_array: np.ndarray, depth_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取处理后的颜色和深度对
        Args:
            color_array: RGB或BGR格式的图像数组
            depth_array: 深度图数组
        Returns:
            (processed_color, processed_depth) 元组
        """
        processed_color = self.process_color(color_array)
        processed_depth = self.process_depth(depth_array)
        
        # 确保颜色和深度图尺寸一致
        if processed_color.shape[:2] != processed_depth.shape[:2]:
            # 调整到相同尺寸
            target_shape = processed_depth.shape[:2][::-1]  # (width, height)
            processed_color = cv2.resize(processed_color, target_shape, interpolation=cv2.INTER_NEAREST)
        
        return processed_color, processed_depth


def simple_process_image_pair(color_array: np.ndarray, 
                            depth_array: np.ndarray, 
                            target_size: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单的图像对处理函数，直接处理远程发送的图像数组
    
    Args:
        color_array: 彩色图像 numpy 数组
        depth_array: 深度图像 numpy 数组
        target_size: 目标尺寸 (width, height)，可选
    
    Returns:
        (processed_color, processed_depth) 处理后的图像对
    """
    reader = SimpleImageReader(target_size)
    return reader.get_processed_pair(color_array, depth_array)


# 示例使用方式
if __name__ == "__main__":
    # 假设你有从远程接收的 numpy 数组
    # color_remote = np.array(...)  # 形状 (H, W, 3)，RGB 或 BGR 格式
    # depth_remote = np.array(...)  # 形状 (H, W)，深度图
    
    # 示例：创建模拟数据
    color_remote = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_remote = np.random.rand(480, 640).astype(np.float32) * 10  # 模拟深度值 0-10 米
    
    # 使用简单处理函数
    processed_color, processed_depth = simple_process_image_pair(
        color_remote, 
        depth_remote, 
        target_size=(320, 240)  # 可选的目标尺寸
    )
    
    print(f"Original color shape: {color_remote.shape}")
    print(f"Processed color shape: {processed_color.shape}")
    print(f"Original depth shape: {depth_remote.shape}")
    print(f"Processed depth shape: {processed_depth.shape}")