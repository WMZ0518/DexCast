#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys

SYNAPATH_ROOT = "/home/zyh/wmz/Synapath_Python_Projects/vrRemoteControl"
if SYNAPATH_ROOT not in sys.path:
    sys.path.insert(0, SYNAPATH_ROOT)

from robots.universalRobotics.ur5RobotAgent import ur5RobotAgent
from robots.xhand.xhand_agent import XHandAgent


def _fmt_list(vals, precision=6):
    return ", ".join([f"{v:.{precision}f}" for v in vals])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ur_ip", type=str, default="192.168.58.100")
    parser.add_argument("--no_ur", action="store_true")
    parser.add_argument("--no_xhand", action="store_true")
    parser.add_argument("--xhand_id", type=int, default=0)
    parser.add_argument("--xhand_mode", type=int, default=3)
    parser.add_argument("--xhand_protocol", type=str, default="RS485")
    parser.add_argument("--xhand_serial", type=str, default=None)
    parser.add_argument("--xhand_baud", type=int, default=3000000)
    parser.add_argument("--xhand_no_autofind", action="store_true")
    args = parser.parse_args()

    ur5 = None
    xhand = None

    if not args.no_ur:
        ur5 = ur5RobotAgent(ip_address=args.ur_ip)

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

    try:
        if ur5 is not None:
            ur_joints = list(ur5.get_joint_angles())
            ur_deg = [v * 180.0 / math.pi for v in ur_joints]
            print("UR5 joints (rad):", _fmt_list(ur_joints))
            print("UR5 joints (deg):", _fmt_list(ur_deg))
            print(f"UR_HOME_JOINTS=\"{_fmt_list(ur_joints)}\"")
            print("UR_HOME_UNIT=rad")

        if xhand is not None:
            states = xhand.get_joint_states()
            if states is None:
                print("XHand joints: None")
            else:
                hand_deg = [float(states[i]) for i in range(12)]
                print("XHand joints (deg):", _fmt_list(hand_deg))
                print(f"XHAND_HOME_DEG=\"{_fmt_list(hand_deg)}\"")
    finally:
        try:
            if xhand is not None:
                xhand.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
