# pytracking_client.py
import json
import os
from pathlib import Path
import uuid

from looptick import LoopTick
import numpy as np
import zmq
import cv2

from fdt_client.vis import draw_3d_box_client
from fdt_client.logger import logger   # 设置日志级别为DEBUG，这样就能看到debug信息了


class RemoteFoundationPose:
    def __init__(self, address="tcp://127.0.0.1:5555"):
        self.context = zmq.Context()
        self.zmq_socket = self.context.socket(zmq.REQ)
        self.zmq_socket.connect(address)
        self.initialized = False
        self.session_id = str(uuid.uuid4())  # 在客户端生成会话ID

        logger.info(f"连接到 {address}")
        logger.info(f"任务 Session ID: {self.session_id}")

        self.init_pose = []
        self.bbox_corners = []

    def _encode_frame(self, frame: np.ndarray):
        """编码图像为JPEG（二进制）或根据数据类型选择适当编码"""
        if frame.dtype == np.uint16:    
            # _, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # return buf.tobytes()  # 对于uint16深度图, 使用PNG进行无损压缩，保留更多细节
            return frame.tobytes()  # 对于uint16深度图, 直接返回原始数据以保证精度和最小延迟
        else:
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes()  # 对于普通彩色图像，使用JPEG

    def _encode_file(self, file_path:Path | str):
        """编码文件为二进制数据"""
        file_path = Path(file_path)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            logger.warning(f"文件不存在: {file_path}")
            return b""

    def init(
        self,
        text_prompt: str,
        cam_K: np.ndarray,
        mesh_file: Path | str,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
    ):
        logger.info(f"{color_frame.shape:}, {text_prompt:}, 开始初始化 Foundation Pose Tracker")

        h, w = depth_frame.shape[:2]

        meta = {
            "cmd": "init",
            "session_id": self.session_id,
            "width": w,   # 图像宽高, 服务端解析图片时需要
            "height": h,   
            "text_prompt": text_prompt,
            "cam_K": cam_K.tolist(),
        }
        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)
        mesh_file_bytes = self._encode_file(mesh_file)

        # multipart: [json, binary]
        self.zmq_socket.send_multipart([
                json.dumps(meta).encode("utf-8"), 
                mesh_file_bytes,
                color_frame_bytes, 
                depth_frame_bytes,
            ])
        reply = self.zmq_socket.recv_json()  # 接收回复

        # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = True
            logger.success(f"tracker 初始化成功: {reply}")
            self.init_pose = reply.get("pose")  # 在客户端其实没什么用
            self.bbox_corners = reply.get("bbox")
            return True
        else:
            logger.error(f"tracker 初始化失败: {reply}")
            return False

    def update(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> tuple:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        
        h, w = depth_frame.shape[:2]

        meta = {
            "cmd": "update",
            "session_id": self.session_id,
            "width": w,  # 图像宽高, 服务端解析图片时需要
            "height": h,
        }

        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)

        self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8"), color_frame_bytes, depth_frame_bytes])
        reply = self.zmq_socket.recv_json()  # 接收回复
        
        # return reply

        # TODO: 等待确定返回值
        # # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            pose = reply.get("pose")   # 确保bbox可以转换为tuple类型
            if isinstance(pose, (list, tuple)) and len(pose) == 16:
                logger.debug(f"任务 {self.session_id} 更新的 pose {pose}")
                return True, tuple(pose)  
            else:              
                logger.error(f"任务 {self.session_id} pose 格式错误: {pose}")
                return False, None   # 如果bbox不是期望的格式，则返回错误
        else:
            return False, None

    def release(self):
        """
        释放远程跟踪器资源
        """
        meta = {"cmd": "release", "session_id": self.session_id}
        self.zmq_socket.send_json(meta)
        reply = self.zmq_socket.recv_json()
        
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = False
            logger.info("tracker 已释放")
            return True
        else:
            logger.error("tracker 释放失败")
            return False    


# 测试用例
if __name__ == "__main__":
    
    from rich import print as rprint
    from rgbd_cam import OrbbecRGBDCamera


    tracker = RemoteFoundationPose("tcp://127.0.0.1:5555")
    tracker.release()

    # 创建采集器实例
    gemini_2 = OrbbecRGBDCamera(device_index=1)

    loop = LoopTick()

    is_recording = False

    try:
        # 启动采集
        gemini_2.start()

        # text_prompt = "mango"
        text_prompt = "yellow"
        intrinsic = gemini_2.get_intrinsic()
        mesh_file = "tmp/scaled_mesh.obj"

        rprint(intrinsic)
        
        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        is_init = False
        while True:
            # 获取图像帧
            color_image, depth_image, depth_data = gemini_2.get_frames()

            # 非空检查 (防止运行时报错 -215 Assertion failed)
            if color_image is None or depth_image is None:
                print("图像帧为空，请检查设备是否正常连接")
                continue

            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
            print(hz)

            # 显示彩色图像 和 深度图像
            # depth_image_show = cv2.bitwise_not(depth_image)
            # cv2.imshow("Color Image", color_image)
            # cv2.imshow("Depth Image", depth_image_show)

            if not tracker.initialized and intrinsic is not None:
                is_init = tracker.init(
                    text_prompt=text_prompt,
                    cam_K=intrinsic,
                    mesh_file=mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )

            # breakpoint()
            
            if is_init:
                ret, pose = tracker.update(color_image, depth_image)

                if ret:
                    bbox = tracker.bbox_corners
                    # 绘制3D框
                    color_image = draw_3d_box_client(color_image, pose, intrinsic, bbox)
                    cv2.imshow("Color Image", color_image)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                break
                    
    except Exception as e:
        print(f"程序运行出错: {e}")
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        cv2.destroyAllWindows()
        gemini_2.stop()
        tracker.release()