# main.py

from pathlib import Path
import time
import cv2
from looptick import LoopTick
import numpy as np
from rich import print as rprint

from fdt_client.logger import logger
from fdt_client.rgbd_cam import OrbbecRGBDCamera
from fdt_client.client import RemoteFoundationPose
from fdt_client.vis import draw_3d_box_client, draw_dict_to_img


loop = LoopTick()

def track_pose(
    text_prompt: str,
    mesh_file: Path | str,
    device_index: int = 1,
    server_url: str = "tcp://127.0.0.1:5555",
    T_cam_to_base = np.eye(4),   
    vis: bool = False
):
    tracker = RemoteFoundationPose(server_url)
    gemini_2 = OrbbecRGBDCamera(device_index=device_index)
    
    try:
        # 启动采集
        gemini_2.start()

        mesh_file = Path(mesh_file)
        intrinsic = gemini_2.get_intrinsic()

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

            # 初始化追踪器
            if not tracker.initialized and intrinsic is not None:
                is_init = tracker.init(
                    text_prompt=text_prompt,
                    cam_K=intrinsic,
                    mesh_file=mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )
                if not is_init:      # 未成功初始化则跳过
                    time.sleep(0.1)  # 防止 CPU 占满
                    continue
            
            # 更新追踪, 核心代码
            ret, pose_cam = tracker.update(color_image, depth_image)
            pose_cam = np.array(pose_cam).reshape(4, 4)  # 确保为 4x4 矩阵

            # 对 pose 结果做其他的后处理, 可以在此处之后进行二次开发
            ################################################
            xyz_arm = []
            xyz_cam  = []
            if ret:
                pose_base = cam2base(pose_cam, T_cam_to_base)
                pose_base = np.array(pose_base).reshape(4, 4)  # 确保为 4x4 矩阵

                xyz_arm = pose_base[:3, 3]
                xyz_cam = pose_cam[:3, 3]
            #################################################

            # 打印信息
            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
            print(f"\rHz: {hz:.2f}, xyz_cam: {xyz_cam}, xyz_base: {xyz_arm}" + ""*10, end='', flush=True)

            # 可视化 (可选)
            if ret and vis:  
                # 绘制 3D 边界框
                result = {
                    "Hz": f"{hz:.2f}",
                    "xyz_cam": xyz_cam, 
                    "xyz_base": xyz_arm,
                }
                vis_image = color_image
                vis_image = draw_dict_to_img(color_image, result, font_size=1)
                vis_image = draw_3d_box_client(vis_image, pose_cam, intrinsic, tracker.bbox_corners)
                
                cv2.imshow("vis_image", vis_image)

                # 显示原始彩色图像 和 深度图像
                # depth_image_show = cv2.bitwise_not(depth_image)
                # cv2.imshow("Color Image", color_image)
                # cv2.imshow("Depth Image", depth_image_show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                    break
            

    except Exception as e:
        logger.exception(f"程序运行出错: {e}")

    except KeyboardInterrupt:
        logger.warning("用户中断程序")

    finally:
        cv2.destroyAllWindows()
        gemini_2.stop()
        tracker.release()

def cam2base(pose_cam, T_cam_to_base):      
    """
    将相机坐标系下的物体位姿变换到机械臂基座坐标系。

    参数:
        pose_cam: 相机坐标系下的 4x4 齐次位姿矩阵。
                  支持 list/tuple (16元素) 或 np.ndarray ((16,) 或 (4,4))。
        T_cam_to_base: 从相机坐标系到基座坐标系的 4x4 齐次变换矩阵，
                       满足: pose_base = T_cam_to_base @ pose_cam。

    返回:
        np.ndarray: (4, 4) 物体在基座坐标系下的位姿矩阵。
    """
    # 确保输入是numpy数组
    T_cam_to_base = np.asarray(T_cam_to_base).reshape(4, 4)
   
    # 确保 pose 是 4x4 矩阵
    if isinstance(pose_cam, (list, tuple)):
        pose_cam = np.array(pose_cam).reshape(4, 4)
    elif isinstance(pose_cam, np.ndarray):
        if pose_cam.shape == (4, 4):
            pass  # 已经是正确的形状
        elif pose_cam.size == 16:
            pose_cam = pose_cam.reshape(4, 4)
        else:
            raise ValueError(f"pose_cam 应该是16个元素的一维数组或4x4矩阵，当前形状为: {pose_cam.shape}")
    else:
        raise TypeError(f"pose_cam 类型不支持: {type(pose_cam)}")

    pose_base =  T_cam_to_base @ pose_cam   # 应用 变换矩阵

    return pose_base


if __name__ == "__main__":

    # 要检测的物体
    text_prompt = "yellow"    # 物体提示词
    mesh_file = "../our_data/processed/mangguo/1764919025923146/scaled_mesh.obj"   # 模型文件路径

    # 运行参数
    device_index = 1          # 相机索引
    server_url = "tcp://127.0.0.1:5555"   # 如果不在本机推理, 切换成其他局域网地址
    # server_url = "tcp:// 192.168.1.124:5555"  

    # 是否可视化
    vis = True   
    # vis = False

    # 相机外参 (相机在机械臂基座坐标系中的位姿)
    MAIN_MAT = [
        [-0.99857, 0.05338, 0.00201, 544.02],
        [0.03946, 0.76244, -0.64586, 552.76],
        [-0.03601, -0.64486, -0.76345, 628.56],
        [0.0, 0.0, 0.0, 1.0],
    ]
    MAIN_MAT = np.array(MAIN_MAT)

    MAIN_MAT_converted = MAIN_MAT.copy()
    MAIN_MAT_converted[:3, 3] /= 1000.0  # 单位转米

    # 执行推理
    track_pose(text_prompt, mesh_file, device_index, server_url, MAIN_MAT_converted, vis)

