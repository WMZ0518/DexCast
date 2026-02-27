# pytracking_client.py
import json
import uuid
import sys

import zmq
import cv2
from loguru import logger# 设置日志级别为DEBUG，这样就能看到debug信息了

logger.remove()  # 移除默认的处理器
logger.add(sys.stderr, level="INFO")  # 添加新的处理器，级别为DEBUG
# logger.add(sys.stderr, level="DEBUG")  # 添加新的处理器，级别为DEBUG

class RemoteTracker:
    def __init__(self, address="tcp://127.0.0.1:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)
        self.initialized = False
        self.session_id = str(uuid.uuid4())  # 在客户端生成会话ID

        logger.info(f"连接到 {address}")
        logger.info(f"任务 Session ID: {self.session_id}")

    def _encode_frame(self, frame):
        """编码图像为JPEG（二进制）"""
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes()

    def init(self, frame, bbox):
        logger.info(f"frame.shape: {frame.shape}, bbox: {bbox}")

        meta = {"cmd": "init", "bbox": bbox, "session_id": self.session_id}
        frame_bytes = self._encode_frame(frame)

        # multipart: [json, binary]
        self.socket.send_multipart([json.dumps(meta).encode("utf-8"), frame_bytes])
        reply = self.socket.recv_json()  # 接收回复

        # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = True
            logger.success(f"tracker 初始化成功: {reply}")
            return True
        else:
            logger.error(f"tracker 初始化失败: {reply}")
            return False

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")

        meta = {"cmd": "update", "session_id": self.session_id}
        frame_bytes = self._encode_frame(frame)  # 编码帧
        self.socket.send_multipart([json.dumps(meta).encode("utf-8"), frame_bytes])  # 发送帧
        reply = self.socket.recv_json()  # 接收回复
        
        # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            bbox = reply.get("bbox")  # 确保bbox可以转换为tuple类型
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                logger.debug(f"任务 {self.session_id} 更新的 bbox {bbox}")
                return True, tuple(bbox)  
            else:                 # 如果bbox不是期望的格式，则返回错误
                logger.error(f"任务 {self.session_id} bbox 格式错误: {bbox}")
                return False, None
        else:
            return False, None

    def release(self):
        """
        释放远程跟踪器资源
        """
        meta = {"cmd": "release", "session_id": self.session_id}
        self.socket.send_json(meta)
        reply = self.socket.recv_json()
        
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = False
            logger.info("tracker 已释放")
            return True
        else:
            logger.error("tracker 释放失败")
            return False    


if __name__ == "__main__":

    tracker = RemoteTracker()
    tracker.release()

    cap = cv2.VideoCapture("tmp/2.mp4")

    logger.info(f"视频帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            if not tracker.initialized:
                tracker.init(frame, (651, 259, 114, 90))

            start_tick = cv2.getTickCount()
            
            success, bbox = tracker.update(frame)

            end_tick = cv2.getTickCount()
            elapsed_time = (end_tick - start_tick) / cv2.getTickFrequency() * 1000

            logger.info(f"追踪结果: {success} {bbox} 耗时: {elapsed_time:.3f} ms")

        except Exception as e:
            logger.error(e)

        except KeyboardInterrupt:
            tracker.release()
            break
