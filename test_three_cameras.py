import os
import time
import pyrealsense2 as rs

CAMERAS = {
    "scene": os.environ.get("PIPER_SCENE_CAMERA_SERIAL"),
    "wrist_l": os.environ.get("PIPER_LEFT_WRIST_CAMERA_SERIAL"),
    "wrist_r": os.environ.get("PIPER_RIGHT_WRIST_CAMERA_SERIAL"),
}
missing = [role for role, serial in CAMERAS.items() if not serial]
if missing:
    raise SystemExit("missing camera serial environment variables: " + ", ".join(missing))

pipelines = {}

try:
    print("===== 启动三路 RGB 摄像头 =====")

    for role, serial in CAMERAS.items():
        pipe = rs.pipeline()
        cfg = rs.config()

        cfg.enable_device(serial)
        cfg.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.rgb8,
            15
        )

        pipe.start(cfg)
        pipelines[role] = pipe

        print(f"{role}: 已启动 ({serial})")

    print("\n===== 采集测试：每台 30 帧 =====")

    for role, pipe in pipelines.items():
        start = time.perf_counter()
        count = 0

        for _ in range(30):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            if frames.get_color_frame():
                count += 1

        elapsed = time.perf_counter() - start

        print(
            f"{role}: {count}/30 帧, "
            f"采集耗时 {elapsed:.2f} s"
        )

finally:
    print("\n===== 关闭相机视频流 =====")

    for role, pipe in pipelines.items():
        try:
            pipe.stop()
            print(f"{role}: 已关闭")
        except Exception as e:
            print(f"{role}: 关闭异常: {e}")
