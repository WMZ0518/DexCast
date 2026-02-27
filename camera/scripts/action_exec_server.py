from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
import zmq

SYNAPATH_ROOT = "/home/zyh/wmz/Synapath_Python_Projects/vrRemoteControl"
if SYNAPATH_ROOT not in sys.path:
    sys.path.insert(0, SYNAPATH_ROOT)

from robots.universalRobotics.ur5RobotAgent import ur5RobotAgent
from robots.xhand.xhand_agent import XHandAgent


def _parse_reorder(reorder_str: Optional[str]) -> Optional[list[int]]:
    if not reorder_str:
        return None
    items = [s.strip() for s in reorder_str.split(",") if s.strip() != ""]
    return [int(x) for x in items]


def _parse_floats(list_str: Optional[str]) -> Optional[list[float]]:
    if not list_str:
        return None
    items = [s.strip() for s in list_str.split(",") if s.strip() != ""]
    return [float(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", type=str, default="tcp://127.0.0.1:7778")
    parser.add_argument("--bind", type=str, default=None)
    parser.add_argument("--topic", type=str, default="action")
    parser.add_argument("--state_bind", type=str, default="tcp://0.0.0.0:7779")
    parser.add_argument("--state_req_bind", type=str, default=None,
                        help="若设置，开启状态请求(REP)接口，采集端按需请求当前关节状态。")
    parser.add_argument("--state_topic", type=str, default="state")
    parser.add_argument("--exec_topic", type=str, default="exec")
    parser.add_argument("--state_hz", type=float, default=20.0)
    parser.add_argument(
        "--ur_home_joints",
        type=str,
        default="-3.263321, -1.480596, 1.341582, -1.519791, -1.595173, 3.048747",
    )
    parser.add_argument("--ur_home_unit", type=str, default="rad", choices=["rad", "deg"])
    parser.add_argument(
        "--xhand_home_deg",
        type=str,
        default="0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000",
    )
    parser.add_argument("--ur_ip", type=str, default="192.168.58.100")
    parser.add_argument("--xhand_id", type=int, default=0)
    parser.add_argument("--xhand_mode", type=int, default=3)
    parser.add_argument("--xhand_protocol", type=str, default="RS485")
    parser.add_argument("--xhand_serial", type=str, default=None)
    parser.add_argument("--xhand_baud", type=int, default=3000000)
    parser.add_argument("--xhand_no_autofind", action="store_true")
    parser.add_argument("--no_ur", action="store_true")
    parser.add_argument("--no_xhand", action="store_true")
    parser.add_argument("--robot_movej_vel", type=float, default=0.2)
    parser.add_argument("--robot_movej_acc", type=float, default=0.2)
    parser.add_argument(
        "--exec_stride",
        type=int,
        default=2,
        help="chunk执行步长。>1 表示只执行第0, stride, 2*stride...步",
    )
    parser.add_argument("--debug_home", action="store_true", help="打印回零/状态请求调试日志")
    args = parser.parse_args()

    ur5 = None
    if not args.no_ur:
        ur5 = ur5RobotAgent(ip_address=args.ur_ip)

    xhand = None
    if not args.no_xhand:
        xhand = XHandAgent(
            hand_id=args.xhand_id,
            mode=args.xhand_mode,
            serial_port=args.xhand_serial,
            baud_rate=args.xhand_baud,
            auto_find=not args.xhand_no_autofind,
        )
        if not xhand.connect(protocol=args.xhand_protocol):
            raise RuntimeError("XHand 连接失败")

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    if args.bind:
        sub.bind(args.bind)
    else:
        sub.connect(args.connect)
    sub.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))

    pub = ctx.socket(zmq.PUB)
    pub.bind(args.state_bind)

    rep = None
    if args.state_req_bind:
        rep = ctx.socket(zmq.REP)
        rep.bind(args.state_req_bind)

    last_action_id = -1
    ur_home_joints = _parse_floats(args.ur_home_joints)
    xhand_home_deg = _parse_floats(args.xhand_home_deg)

    if rep is not None:
        print("[EXEC] 等待动作信号中...（状态：PUB + REQ 可用）")
    else:
        print("[EXEC] 等待动作信号中...（状态：仅 PUB）")
    if args.exec_stride > 1:
        print(f"[EXEC] chunk执行步长启用 exec_stride={args.exec_stride}")

    try:
        last_state = 0.0
        state_idx = 0
        state_period = 1.0 / max(args.state_hz, 1e-3)
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        if rep is not None:
            poller.register(rep, zmq.POLLIN)
        while True:
            now = time.time()
            if now - last_state >= state_period:
                last_state = now
                ur_joints = None
                hand_joints = None
                if ur5 is not None:
                    try:
                        ur_joints = list(ur5.get_joint_angles())
                    except Exception:
                        ur_joints = None
                if xhand is not None:
                    try:
                        states = xhand.get_joint_states()
                        if states is not None:
                            hand_joints = [float(states[i]) for i in range(12)]
                    except Exception:
                        hand_joints = None

                state_idx += 1
                payload = {
                    "ts": now,
                    "state_idx": state_idx,
                    "ur_joints": ur_joints,
                    "hand_joints": hand_joints,
                    "last_action_id": last_action_id,
                }
                pub.send_multipart([
                    args.state_topic.encode("utf-8"),
                    zmq.utils.jsonapi.dumps(payload),
                ])

            socks = dict(poller.poll(timeout=50))
            if rep is not None and rep in socks:
                try:
                    _ = rep.recv_multipart()
                except Exception:
                    _ = rep.recv()
                if args.debug_home:
                    print("[STATE-REQ] received request")

                ur_joints = None
                hand_joints = None
                if ur5 is not None:
                    try:
                        ur_joints = list(ur5.get_joint_angles())
                    except Exception:
                        ur_joints = None
                if xhand is not None:
                    try:
                        states = xhand.get_joint_states()
                        if states is not None:
                            hand_joints = [float(states[i]) for i in range(12)]
                    except Exception:
                        hand_joints = None

                state_idx += 1
                reply = {
                    "ts": time.time(),
                    "state_idx": state_idx,
                    "ur_joints": ur_joints,
                    "hand_joints": hand_joints,
                    "last_action_id": last_action_id,
                }
                rep.send(zmq.utils.jsonapi.dumps(reply))
                if args.debug_home:
                    print(f"[STATE-REQ] replied idx={state_idx} last_action_id={last_action_id}")

            if sub not in socks:
                continue

            topic, payload = sub.recv_multipart()
            data = zmq.utils.jsonapi.loads(payload)

            action_id = data.get("action_id", None)
            cmd = data.get("cmd", None)
            if cmd == "home":
                ur_home = data.get("ur_home_joints", None)
                xhand_home = data.get("xhand_home_deg", None)
                ur_unit = data.get("ur_home_unit", args.ur_home_unit)
                if ur_home is None and xhand_home is None:
                    print(f"[EXEC] 回零指令缺少客户端姿态，已忽略 id={action_id}")
                    continue
                print(
                    f"[EXEC] 收到回零指令 id={action_id} "
                    f"ur_home={ur_home} xhand_home={xhand_home} unit={ur_unit}"
                )
                if ur5 is not None and ur_home:
                    ur5.moveJ(
                        desc_joint=list(ur_home),
                        vel=args.robot_movej_vel,
                        acc=args.robot_movej_acc,
                        asynchronous=False,
                        eula=(ur_unit == "deg"),
                    )
                if xhand is not None and xhand_home:
                    xhand.send_angles({i: float(xhand_home[i]) for i in range(min(len(xhand_home), 12))})
                exec_payload = {
                    "ts": time.time(),
                    "action_id": action_id,
                    "status": "done",
                    "cmd": "home",
                }
                pub.send_multipart([
                    args.exec_topic.encode("utf-8"),
                    zmq.utils.jsonapi.dumps(exec_payload),
                ])
                # Keep REQ state consistent with HOME completion so clients using
                # last_action_id as fallback can confirm done.
                last_action_id = action_id if action_id is not None else last_action_id
                if args.debug_home:
                    print(
                        f"[EXEC-DEBUG] home done: action_id={action_id} "
                        f"last_action_id={last_action_id} status={exec_payload.get('status')}"
                    )
                print(f"[EXEC] 回零完成 id={action_id}")
                continue

            action = np.asarray(data.get("action", []), dtype=np.float32)
            hand_dof = int(data.get("hand_dof", 12))
            arm_dof = int(data.get("arm_dof", 6))
            action_unit = data.get("action_unit", "rad")
            xhand_reorder = data.get("xhand_reorder", None)
            action_is_chunk = bool(data.get("action_is_chunk", False))

            if action_is_chunk:
                dof = hand_dof + arm_dof
                steps = len(action) // dof if dof > 0 else 0
                print(
                    f"[EXEC] 收到动作信号 id={action_id} len={len(action)}"
                    f" ({steps} * ({hand_dof} + {arm_dof}) = {steps * dof})"
                    f" chunk={action_is_chunk} -> 执行中..."
                )
            else:
                print(f"[EXEC] 收到动作信号 id={action_id} len={len(action)} chunk={action_is_chunk} -> 执行中...")

            if action.shape[0] < (hand_dof + arm_dof):
                print("[EXEC] 动作维度不足，跳过")
                continue

            if action_is_chunk:
                actions = action.reshape(-1, hand_dof + arm_dof)
            else:
                actions = action.reshape(1, -1)
                if actions.shape[1] < (hand_dof + arm_dof):
                    print("[EXEC] 动作维度不足，跳过")
                    continue

            for i, act in enumerate(actions):
                if args.exec_stride > 1 and (i % args.exec_stride) != 0:
                    continue
                hand_action = act[:hand_dof].copy()
                arm_action = act[hand_dof:hand_dof + arm_dof].copy()

                if ur5 is not None:
                    ur5.moveJ(
                        desc_joint=list(arm_action),
                        vel=args.robot_movej_vel,
                        acc=args.robot_movej_acc,
                        asynchronous=False,
                        eula=False,
                    )

                if xhand is not None:
                    if action_unit == "rad":
                        hand_deg = hand_action * 180.0 / np.pi
                    else:
                        hand_deg = hand_action

                    if isinstance(xhand_reorder, list):
                        hand_deg = hand_deg[xhand_reorder]

                    xhand.send_angles({i: float(hand_deg[i]) for i in range(min(len(hand_deg), 12))})

                if action_is_chunk:
                    time.sleep(max(0.0, 1.0 / 10.0))

            last_action_id = action_id if action_id is not None else last_action_id
            exec_payload = {
                "ts": time.time(),
                "action_id": action_id,
                "status": "done",
            }
            pub.send_multipart([
                args.exec_topic.encode("utf-8"),
                zmq.utils.jsonapi.dumps(exec_payload),
            ])
            print(f"[EXEC] 执行完毕 id={action_id}")

    finally:
        try:
            sub.close(0)
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        try:
            if rep is not None:
                rep.close(0)
        except Exception:
            pass
        try:
            if xhand is not None:
                xhand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
