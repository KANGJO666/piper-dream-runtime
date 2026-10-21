from robots.config import ArmConfig
from piper_driver import PiperDriver

for side, channel in (
    ("right", "can0"),
    ("left", "can1"),
):
    cfg = ArmConfig(
        can_channel=channel,
        home=[0.0] * 6,
    )

    arm = PiperDriver(cfg, side)

    try:
        _, gripper_mm, _ = arm.sample_raw()
        print(
            f"{side:5s} ({channel}): "
            f"{gripper_mm:.3f} mm"
        )
    finally:
        arm.close()
