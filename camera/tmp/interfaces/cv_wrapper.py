import cv2
import sys
import os
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytracking.evaluation import Tracker as PyTracker


class PyTrackingWrapper:
    """
    将PyTracking封装成类似OpenCV Tracker的API
    """
    
    def __init__(self, tracker_name="dimp", tracker_param="dimp50"):
        """
        初始化追踪器
        
        Args:
            tracker_name: 追踪器名称 (如: dimp, atom, kys等)
            tracker_param: 追踪器参数配置 (如: dimp50, dimp18, default等)
        """
        self.tracker_name = tracker_name
        self.tracker_param = tracker_param
        self.tracker = PyTracker(tracker_name, tracker_param)
        self.tracker_instance = None
        self.initialized = False
        
    def init(self, image, bbox):
        """
        初始化追踪器
        
        Args:
            image: 输入图像 (numpy array)
            bbox: 边界框 [x, y, width, height]
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 创建追踪器实例
            self.tracker_instance = self.tracker.create_tracker(self.tracker.get_parameters())
            
            # 初始化特征
            if hasattr(self.tracker_instance, 'initialize_features'):
                self.tracker_instance.initialize_features()
            
            # 准备初始化信息
            init_info = {'init_bbox': bbox}
            
            # 初始化追踪器
            self.tracker_instance.initialize(image, init_info)
            self.initialized = True
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def update(self, image):
        """
        更新追踪器
        
        Args:
            image: 输入图像 (numpy array)
            
        Returns:
            tuple: (success, bbox) 
                   success: 是否追踪成功 (bool)
                   bbox: 追踪到的边界框 [x, y, width, height]
        """
        if not self.initialized or self.tracker_instance is None:
            return False, None
            
        try:
            # 执行追踪
            output = self.tracker_instance.track(image, {})
            
            # 获取边界框
            if 'target_bbox' in output:
                bbox = output['target_bbox']
                # 确保bbox是正确的格式并转换为整数
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # 转换为整数并返回
                    int_bbox = [int(coord) for coord in bbox]
                    return True, int_bbox
                elif isinstance(bbox, np.ndarray) and bbox.shape[0] == 4:
                    # 转换为整数并返回
                    int_bbox = [int(coord) for coord in bbox.tolist()]
                    return True, int_bbox
            
            return False, None
        except Exception as e:
            print(f"追踪失败: {e}")
            return False, None


def create_pytracker(tracker_name="dimp", tracker_param="dimp50"):
    """
    创建PyTracking追踪器的工厂函数，模仿OpenCV的create函数
    
    Args:
        tracker_name: 追踪器名称
        tracker_param: 追踪器参数
        
    Returns:
        PyTrackingWrapper: 封装好的追踪器实例
    """
    return PyTrackingWrapper(tracker_name, tracker_param)


# 提供一些常用的追踪器创建函数，模仿OpenCV的API风格
def TrackerDiMP_create():
    """创建DiMP追踪器"""
    return PyTrackingWrapper("dimp", "dimp50")


def TrackerATOM_create():
    """创建ATOM追踪器"""
    return PyTrackingWrapper("atom", "default")


def TrackerKYS_create():
    """创建KYS追踪器"""
    return PyTrackingWrapper("kys", "default")


# 使用示例
if __name__ == "__main__":
    # 示例用法，类似于OpenCV的使用方式
    
    source="tmp/1.mp4"
    cap = cv2.VideoCapture(source)  # 或 0 表示摄像头

    # 读取第一帧
    ret, frame = cap.read()

    if not ret:
        print("无法读取视频源")
        sys.exit(1)

    # 手动选择跟踪目标区域 (x, y, w, h) - 与你的cvtracker.py保持一致
    bbox = cv2.selectROI("Frame", frame, False)
    
    # 创建追踪器
    tracker = TrackerDiMP_create()
    success = tracker.init(frame, bbox)
    
    if not success:
        print("追踪器初始化失败")
        sys.exit(1)
    
    # cv2.namedWindow("追踪", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 更新追踪器
        success, bbox = tracker.update(frame)
        print("追踪结果:", success, bbox)
        
        disp = frame.copy()
        if success and bbox is not None:
            # 绘制边界框
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(disp, "追踪失败", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        cv2.imshow("追踪", disp)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()