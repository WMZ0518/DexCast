# obcam.py
import numpy as np

import cv2
from loguru import logger
from rich import print as rprint
from looptick import LoopTick

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

from pyorbbecsdk import Context
from pyorbbecsdk import OBLogLevel
from pyorbbecsdk import OBAlignMode

# from .utils_ob import frame_to_bgr_image 
try:# 处理相对导入和绝对导入的兼容性
    from .utils_ob import frame_to_bgr_image  # 当作为模块导入时使用相对导入
except (ImportError, ValueError):
    try:
        from utils_ob import frame_to_bgr_image  # 当直接运行脚本时使用绝对导入
    except ImportError:
        raise ImportError("无法导入frame_to_bgr_image函数，请检查utils_ob.py文件是否存在且路径正确")



class OrbbecRGBDCamera:
    """
    用于同时捕获Orbbec相机的彩色图像和深度图像，并将其转换为OpenCV可以处理的格式
    """

    def __init__(self, 
                 color_width=1280, color_height=720, color_fps=30, 
                 depth_width=1280, depth_height=800, depth_fps=30, 
                 device_index=0):  
        """
        初始化ColorDepthCapture类
        
        参数:
        color_width: 彩色图像宽度
        color_height: 彩色图像高度
        color_fps: 彩色图像帧率
        depth_width: 深度图像宽度
        depth_height: 深度图像高度
        depth_fps: 深度图像帧率
        device_index: 相机设备索引，默认为0（第一个设备）
        """
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.device_index = device_index
        
        self.config = Config()
        self.config.set_align_mode(OBAlignMode.SW_MODE)
        
        # 获取指定索引的设备
        context = Context()
        context.set_logger_level(OBLogLevel.ERROR)
        device_list = context.query_devices()
        device_count = device_list.get_count()
        
        if device_count <= device_index:
            raise ValueError(f"请求的设备索引 {device_index} 超出可用设备范围 (共 {device_count} 个设备)")
        
        # 使用指定设备创建Pipeline
        device = device_list.get_device_by_index(device_index)
        self.pipeline = Pipeline(device)
        self.pipeline.enable_frame_sync()

        # 配置彩色图像流
        self._configure_color_stream()
        
        # 配置深度图像流
        self._configure_depth_stream()
        
    def _configure_color_stream(self):
        """
        配置彩色图像流
        """
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(
                    self.color_width, self.color_height, OBFormat.RGB, self.color_fps)
            except OBError as e:
                logger.error(f"无法设置指定的彩色图像配置: {e}")
                color_profile = profile_list.get_default_video_stream_profile()
                logger.info("使用默认彩色图像配置: ", color_profile)
            self.config.enable_stream(color_profile)
        except Exception as e:
            logger.error(f"配置彩色图像流时出错: {e}")
            raise
            
    def _configure_depth_stream(self):
        """
        配置深度图像流
        """
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile: VideoStreamProfile = profile_list.get_video_stream_profile(
                    self.depth_width, self.depth_height, OBFormat.Y16, self.depth_fps)
            except OBError as e:
                logger.error(f"无法设置指定的深度图像配置: {e}")
                depth_profile = profile_list.get_default_video_stream_profile()
                logger.info("使用默认深度图像配置: ", depth_profile)
            self.config.enable_stream(depth_profile)
        except Exception as e:
            logger.error(f"配置深度图像流时出错: {e}")
            raise
            
    def start(self):
        """
        启动图像采集管道
        """
        self.pipeline.start(self.config)
        
    def stop(self):
        """
        停止图像采集管道
        """
        self.pipeline.stop()
        
    def get_frames(self, timeout_ms=100):
        """
        获取一帧彩色图像和深度图像
        
        参数:
        timeout_ms: 等待帧的超时时间（毫秒）
        
        返回:
        color_image: 彩色图像（OpenCV格式BGR），如果失败则为None
        depth_image: 深度图像（OpenCV格式），如果失败则为None
        depth_data: 原始深度数据（numpy数组），如果失败则为None (注: uint16)
        """
        try:
            frames: FrameSet = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None, None, None
                
            # 获取彩色图像帧
            color_frame = frames.get_color_frame()
            color_image = None
            if color_frame is not None:
                color_image = frame_to_bgr_image(color_frame)
                
            # 获取深度图像帧
            depth_frame = frames.get_depth_frame()
            depth_image = None
            depth_data = None
            if depth_frame is not None:
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                
                # 获取深度数据
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_image = depth_data.astype(np.float32) * scale
                depth_image = depth_image.astype(np.uint16)  # 注意, 这是16位深度的图片
            if color_image is not None:
                color_image = np.ascontiguousarray(color_image)
            if depth_image is not None:
                depth_image = np.ascontiguousarray(depth_image)
            if depth_data is not None:
                depth_data = np.ascontiguousarray(depth_data)
            return color_image, depth_image, depth_data
        
        except Exception as e:
            logger.info(f"获取帧时出错: {e}")
            return None, None, None


    def _get_camera_params(self):
        """获取相机参数的辅助函数"""
        if self.pipeline is None:
            logger.error("[ERROR] Pipeline未初始化，请先调用init_pipeline()方法")
            return None
        
        self.camera_param = self.pipeline.get_camera_param()
        return self.camera_param

    def _create_intrinsic_matrix(self, fx, fy, cx, cy):
        """创建内参矩阵的辅助函数"""
        intrinsic = np.identity(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        return intrinsic


    def get_intrinsic(self):
        '''获取RGB相机内参'''
        camera_param = self._get_camera_params()
        if camera_param is None:
            return None
            
        # 获取彩色相机内参
        rgb_intrinsic = camera_param.rgb_intrinsic
        return self._create_intrinsic_matrix(
            rgb_intrinsic.fx, 
            rgb_intrinsic.fy, 
            rgb_intrinsic.cx, 
            rgb_intrinsic.cy
        )

    def get_depth_intrinsic(self):
        '''获取深度相机内参'''
        camera_param = self._get_camera_params()
        if camera_param is None:
            return None
            
        # 获取深度相机内参
        depth_intrinsic = camera_param.depth_intrinsic
        return self._create_intrinsic_matrix(
            depth_intrinsic.fx, 
            depth_intrinsic.fy, 
            depth_intrinsic.cx, 
            depth_intrinsic.cy
        )

    def _create_distortion_array(self, k1, k2, p1, p2, k3, k4, k5, k6):
        """创建畸变系数数组的辅助函数"""
        distortion = [k1, k2, p1, p2, k3, k4, k5, k6]
        return np.array(distortion, dtype=np.float64)
    
    def get_distortion(self):
        '''获取RGB相机的畸变系数'''
        camera_param = self._get_camera_params()
        if camera_param is None:
            return None
            
        # 获取RGB相机畸变参数
        d = camera_param.rgb_distortion
        return self._create_distortion_array(d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6)

    def get_depth_distortion(self):
        '''获取深度相机的畸变系数'''
        camera_param = self._get_camera_params()
        if camera_param is None:
            return None
            
        # 获取深度相机畸变参数
        d = camera_param.depth_distortion
        return self._create_distortion_array(d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6)



if __name__ == "__main__":

    # 创建采集器实例
    gemini_2 = OrbbecRGBDCamera(device_index=0)

    loop = LoopTick()

    is_recording = False

    try:
        # 启动采集
        gemini_2.start()

        intrinsic = gemini_2.get_intrinsic()
        distortion = gemini_2.get_depth_distortion()

        rprint(intrinsic)
        rprint(distortion)

        breakpoint()
        
        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        print("按 'r' 开始/停止录制")
        
        is_recording = False
        
        while True:
            # 获取图像帧
            color_image, depth_image, depth_data = gemini_2.get_frames()
            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
            print(hz)
            # 显示彩色图像
            if color_image is not None:
                cv2.imshow("Color Image", color_image)
                
            # 显示深度图像
            if depth_image is not None:
                depth_image = cv2.bitwise_not(depth_image)
                cv2.imshow("Depth Image", depth_image)
                
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                break
                    
    except Exception as e:
        print(f"程序运行出错: {e}")
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        gemini_2.stop()
        cv2.destroyAllWindows()
