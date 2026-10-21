from piper_driver import PiperDriver


class PiperReadOnlyDriver(PiperDriver):
    """Read-only Piper driver for Dream-1 dry runs."""

    def write(self, positions_rad, gripper):
        raise RuntimeError(
            "Read-only driver: motion commands are disabled"
        )

    def torque_off(self):
        raise RuntimeError(
            "Read-only driver: motor disable is not permitted"
        )
