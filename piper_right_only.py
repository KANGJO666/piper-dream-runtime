"""Dual observation, right-only actuation. Default: capture commands, no CAN sends."""
import json
import os
import time
from pathlib import Path
import numpy as np
from piper_driver import PiperDriver

ROOT = Path(__file__).resolve().parent

def validate_target(q, grip):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite joint angles in radians')
    if not np.isfinite(grip) or not 0 <= grip <= 1:
        raise ValueError('Expected normalized gripper opening in [0, 1]')
    return q

def run_folder():
    folder = Path(os.environ.get('PIPER_RIGHT_RUN_DIR') or
        ROOT / 'logs' / f'dreamcli_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}').resolve()
    folder.relative_to(ROOT)
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['PIPER_RIGHT_RUN_DIR'] = str(folder)
    return folder

class CaptureDriver(PiperDriver):
    def __init__(self, cfg, side):
        folder = run_folder()
        super().__init__(cfg, side)
        self.count = 0
        self.output = (folder / f'{side}_targets.jsonl').open('a')

    def write(self, positions_rad, gripper):
        q = validate_target(positions_rad, gripper)
        self.output.write(json.dumps({'monotonic_s': time.monotonic(),
            'disposition': 'left_discarded' if self.side == 'left' else 'right_preview_only',
            'joints_rad': q.tolist(), 'gripper': float(gripper)}) + '\n')
        self.output.flush()
        self.count += 1

    def close(self):
        try:
            super().close()
        finally:
            if hasattr(self, 'output') and not self.output.closed:
                self.output.close()
                print(f'RIGHT_ONLY_CAPTURE side={self.side} count={self.count}', flush=True)

    def torque_off(self):
        self.close()

class RightOnlyDriver:
    def __new__(cls, cfg, side):
        expected = {'left': 'can1', 'right': 'can0'}
        if side not in expected or cfg.can_channel != expected[side]:
            raise ValueError(f'Unexpected right-only binding: {side}/{cfg.can_channel}')
        if side == 'left' or os.environ.get('PIPER_RIGHT_EXECUTE') != '1':
            return CaptureDriver(cfg, side)
        from piper_control import PiperControlDriver
        return PiperControlDriver(cfg, side)
