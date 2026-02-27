# pytracking_server.py
import json
import threading
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import zmq
import numpy as np
from loguru import logger

from interfaces.ob_wrapper import TrackerDiMP_create

logger.remove()  # 移除默认的处理器
logger.add(sys.stderr, level="INFO")  # 添加新的处理器，级别为DEBUG
# logger.add(sys.stderr, level="DEBUG")  # 添加新的处理器，级别为DEBUG

class SessionThread(threading.Thread):
    def __init__(self, session_id, frame, bbox, vis=False):
        super().__init__()
        self.session_id = session_id
        self.frame = frame
        self.bbox = bbox
        self.vis = vis
        self.tracker = None
        self.initialized = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.result = None
        self.result_available = threading.Event()

    def run(self):
        try:
            # 初始化跟踪器
            self.tracker = TrackerDiMP_create()
            self.tracker.init(self.frame, tuple(self.bbox))
            self.initialized = True
            
            # 通知主线程初始化完成
            self.result = {"status": "ok"}
            self.result_available.set()
            
            # 持续处理更新请求
            while not self.stop_event.is_set():
                time.sleep(0.001)  # 避免过度占用CPU
                
        except Exception as e:
            logger.error(f"会话 {self.session_id} 初始化跟踪器失败: {e}")
            self.result = {"status": "error", "msg": str(e)}
            self.result_available.set()

    def update(self, frame):
        if not self.initialized or self.tracker is None:
            return None

        try:
            # 调用实际的追踪器更新方法
            success, bbox = self.tracker.update(frame)

            # 根据vis参数决定是否显示调试窗口
            if self.vis:
                if frame is not None:
                    cv2.namedWindow(self.session_id, cv2.WINDOW_NORMAL)

                if success and bbox is not None:
                    # 绘制边界框
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "DiMP", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    
                    if self.vis:
                        cv2.imshow(self.session_id, frame)
                        cv2.waitKey(1)
                    return bbox
                
                else:
                    if self.vis:
                        cv2.putText(frame, "追踪失败", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        cv2.imshow(self.session_id, frame)
                        cv2.waitKey(1)
                    return None
            else:
                # 不显示调试窗口，直接返回结果
                if success and bbox is not None:
                    return bbox
                else:
                    return None
            
        except Exception as e:
            logger.info(f"会话 {self.session_id} 更新跟踪器失败: {e}")
            return None

    def stop(self):
        self.stop_event.set()
        # 只有在启用调试可视化时才需要销毁窗口
        if self.vis:
            cv2.destroyWindow(self.session_id)


class TrackerServer:
    def __init__(self, vis=False, address="tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = address
        self.socket.bind(self.address)
        
        # 存储多个会话线程，以会话ID为键
        self.sessions = {}
        self.sessions_lock = threading.Lock()
        self.vis = vis
        
        logger.info("服务端启动...")
        logger.info(f"监听地址: {self.address}")

    def decode_frame(self, buf):
        """解码JPEG为numpy图像"""
        np_arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def init_tracker(self, session_id, frame, bbox):
        """
        初始化追踪器线程
        """
        with self.sessions_lock:
            # 如果会话已存在，先停止它
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                self.sessions[session_id].join()
                
            # 创建并启动新的会话线程
            session_thread = SessionThread(session_id, frame, bbox, self.vis)
            session_thread.start()
            self.sessions[session_id] = session_thread

            logger.info(f"开始新任务: {bbox}, {session_id}")
            
            # 等待初始化结果
            session_thread.result_available.wait()
            return session_thread.result

    def update_tracker(self, session_id, frame):
        """
        更新追踪器
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return None
                
            session_thread = self.sessions[session_id]
            
        if not session_thread.initialized:
            return None
            
        return session_thread.update(frame)

    def release_tracker(self, session_id):
        """
        释放指定会话的跟踪器
        """
        with self.sessions_lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                self.sessions[session_id].join()
                del self.sessions[session_id]
                return True
        return False

    def handle_init_command(self, frame, msg):
        if frame is None:
            return {"status": "error", "msg": "failed to decode frame"}
        
        session_id = msg.get("session_id")
        if not session_id:
            return {"status": "error", "msg": "session_id is required"}
        
        result = self.init_tracker(session_id, frame, msg["bbox"])
        return result

    def handle_update_command(self, frame, msg):
        if frame is None:
            return {"status": "error", "msg": "failed to decode frame"}
        
        session_id = msg.get("session_id")
        if not session_id:
            return {"status": "error", "msg": "session_id is required"}
            
        bbox = self.update_tracker(session_id, frame)
        if bbox is not None:
            return {"status": "ok", "bbox": bbox}
        else:
            return {"status": "error", "msg": "tracker update failed"}

    def handle_release_command(self, msg):
        session_id = msg.get("session_id")
        if not session_id:
            return {"status": "error", "msg": "session_id is required"}
            
        success = self.release_tracker(session_id)
        if success:
            return {"status": "ok"}
        else:
            return {"status": "error", "msg": "failed to release tracker"}

    def run(self):
        try:
            while True:
                # multipart 接收: [json字符串, 图像二进制]
                parts = self.socket.recv_multipart()
                
                # 检查接收到的部分数量
                if len(parts) != 2:
                    logger.error(f"接收到错误的消息部分数量: {len(parts)}, 期望: 2")
                    self.socket.send_json({"status": "error", "msg": "invalid message format"})
                    continue
                    
                meta_str, frame_buf = parts
                logger.debug(f"接收到消息，元数据长度: {len(meta_str)}, 帧数据长度: {len(frame_buf)}")
                
                try:
                    msg = json.loads(meta_str.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解码失败: {e}")
                    self.socket.send_json({"status": "error", "msg": "invalid json format"})
                    continue
                    
                cmd = msg.get("cmd", "unknown")
                logger.debug(f"收到命令: {cmd}")

                if cmd == "init":
                    frame = self.decode_frame(frame_buf)
                    if frame is None:
                        logger.error("帧解码失败")
                        self.socket.send_json({"status": "error", "msg": "failed to decode frame"})
                        continue
                        
                    response = self.handle_init_command(frame, msg)
                    self.socket.send_json(response)

                elif cmd == "update":
                    frame = self.decode_frame(frame_buf)
                    if frame is None:
                        logger.error("帧解码失败")
                        self.socket.send_json({"status": "error", "msg": "failed to decode frame"})
                        continue
                        
                    response = self.handle_update_command(frame, msg)
                    self.socket.send_json(response)

                elif cmd == "release":
                    response = self.handle_release_command(msg)
                    self.socket.send_json(response)

                else:
                    logger.warning(f"未知命令: {cmd}")
                    self.socket.send_json({"status": "error", "msg": "unknown command"})
        except Exception as e:
            logger.error(f"服务器运行时发生异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理所有会话
            with self.sessions_lock:
                for session_id, session_thread in self.sessions.items():
                    session_thread.stop()
                    session_thread.join()
                self.sessions.clear()
                logger.info("服务器已关闭，所有会话已清理")


if __name__ == "__main__":
    # 可以通过命令行参数控制是否启用调试可视化
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true", help="启用调试可视化")
    # args = parser.parse_args()
    
    # server = TrackerServer(vis=args.debug)
    server = TrackerServer(vis=False)
    # server = TrackerServer(vis=True)
    server.run()