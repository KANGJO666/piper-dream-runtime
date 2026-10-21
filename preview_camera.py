import os
import sys
from pathlib import Path

import pyrealsense2 as rs
import numpy as np

serial = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PIPER_CAMERA_SERIAL")
if not serial:
    raise SystemExit("usage: python preview_camera.py SERIAL (or set PIPER_CAMERA_SERIAL)")

pipeline = rs.pipeline()
config = rs.config()

config.enable_device(serial)
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.rgb8,
    15,
)

started = False

try:
    pipeline.start(config)
    started = True

    frames = None

    for _ in range(20):
        data = pipeline.wait_for_frames(timeout_ms=3000)
        color = data.get_color_frame()

        if color:
            frames = np.asanyarray(color.get_data()).copy()

    if frames is None:
        raise RuntimeError("未获取到 RGB 图像")

    from PIL import Image

    output_dir = Path(os.environ.get("PIPER_CAMERA_OUTPUT_DIR", Path.cwd()))
    output = output_dir / f"camera_{serial}.jpg"
    Image.fromarray(frames).save(output, quality=90)

    print("采集成功:", output)

finally:
    if started:
        pipeline.stop()
