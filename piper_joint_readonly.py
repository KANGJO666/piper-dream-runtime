import can
import math
import time

CHANNELS = {
    "right": "can0",
    "left": "can1",
}

FRAME_JOINTS = {
    0x2A5: (0, 1),
    0x2A6: (2, 3),
    0x2A7: (4, 5),
}


def read_joints(channel, duration=2.0):
    values = [None] * 6
    timestamps = [None] * 6

    with can.Bus(
        channel=channel,
        interface="socketcan",
        receive_own_messages=False,
    ) as bus:

        end = time.monotonic() + duration

        while time.monotonic() < end:
            msg = bus.recv(timeout=0.1)

            if msg is None:
                continue

            pair = FRAME_JOINTS.get(msg.arbitration_id)

            if pair is None or len(msg.data) != 8:
                continue

            now = time.monotonic()

            for idx, raw in zip(pair, (msg.data[:4], msg.data[4:])):
                millidegrees = int.from_bytes(
                    raw, byteorder="big", signed=True
                )

                values[idx] = millidegrees * math.pi / 180000.0
                timestamps[idx] = now

    if any(v is None for v in values):
        raise RuntimeError(f"{channel}: 关节反馈不完整")

    if any(time.monotonic() - t > 0.5 for t in timestamps):
        raise RuntimeError(f"{channel}: 关节反馈可能已经过期")

    return values


def main():
    for side, channel in CHANNELS.items():
        print(f"\n===== {side.upper()} / {channel} =====")

        joints = read_joints(channel)

        for i, rad in enumerate(joints, 1):
            deg = math.degrees(rad)

            print(
                f"J{i}: {rad:+.6f} rad "
                f"({deg:+.3f} deg)"
            )

    print("\n完成：只读取 CAN 反馈，未发送控制指令。")


if __name__ == "__main__":
    main()
