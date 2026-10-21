"""Operator-run Piper targets. Construction/close never enable, zero, home, or transmit."""
import logging
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import can
import numpy as np
from piper_sdk.protocol.protocol_v2 import C_PiperParserV2
from piper_sdk.piper_msgs.msg_v2 import (
    ArmMsgType, PiperMessage, ArmMsgMotionCtrl_2, ArmMsgJointCtrl, ArmMsgGripperCtrl,
)
from piper_driver import PiperDriver

_log = logging.getLogger(__name__)


class PiperControlDriver(PiperDriver):
    def __init__(self, cfg, side):
        model = ET.parse(Path(__file__).resolve().parent / 'arms/piper_reference.urdf').getroot()
        limits = {j.get('name'): j.find('limit') for j in model.findall('joint') if j.get('type') == 'revolute'}
        self._limits = np.array([[float(limits[f'joint{i}'].get(k)) for k in ('lower', 'upper')] for i in range(1, 7)])
        self._parser = C_PiperParserV2()
        self._mode_initialized = False
        self._failed = False
        super().__init__(cfg, side)
        try:
            deadline = time.monotonic() + 0.75
            while len(self._motor_status) != 6 or self._arm_status is None:
                self.sample_raw()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Motor/status feedback missing; no command was sent')
        except Exception:
            self.close()
            raise

    def write(self, positions_rad, gripper):
        if gripper is None:
            raise ValueError('Gripper must be within [0, 1]')
        self._write(positions_rad, gripper)

    def write_joints(self, positions_rad):
        """Send only arm joints; do not move, enable, zero or retarget the gripper."""
        self._write(positions_rad, None)

    def _write(self, positions_rad, gripper):
        q = np.asarray(positions_rad, dtype=float)
        grip = None if gripper is None else float(gripper)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Expected six finite joint angles')
        jaw = None
        if grip is not None:
            if not math.isfinite(grip) or not 0 <= grip <= 1:
                raise ValueError('Gripper must be within [0, 1]')
            span = self.gripper_open_mm - self.gripper_closed_mm
            if not math.isfinite(span) or span <= 0:
                raise ValueError('Invalid gripper range')
            jaw = ArmMsgGripperCtrl(int(round((self.gripper_closed_mm + grip * span) * 1000)), 1000, 0x01, 0)
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError('Driver closed or latched after a failure; reopen explicitly')
            self._refresh_feedback()
            now = time.monotonic()
            if self._feedback_at is None or now - self._feedback_at > 0.15:
                # Give the receiver a turn after a scheduling pause; never send using stale data.
                with self._state:
                    self._state.wait_for(lambda: (self._refresh_feedback() or
                                         (self._feedback_at is not None and time.monotonic() - self._feedback_at <= 0.15)),
                                         timeout=0.02)
                now = time.monotonic()
                if self._feedback_at is None or now - self._feedback_at > 0.15:
                    raise TimeoutError(f'{self.side}/{self.channel}: joint/gripper feedback stale after receiver catch-up; refusing target')
            if self._arm_status is None or now - self._arm_status[1] > 0.25:
                raise TimeoutError('Controller status stale; refusing target')
            status = self._arm_status[0]
            # SDK teaching state 2 is STOP_RECORDING (already exited), not active teaching.
            if status[1] != 0 or status[3] not in (0, 2) or any(status[6:8]):
                raise RuntimeError(f'{self.side}/{self.channel}: controller not ready: arm_status={status[1]}, teach_status={status[3]}, error_bytes={list(status[6:8])}; refusing target')
            if status[0] not in (0, 1):
                raise RuntimeError('Controller is owned by another control mode')
            for i in range(1, 7):
                state = self._motor_status.get(i)
                if state is None or now - state[1] > 0.25:
                    raise TimeoutError(f'Motor {i} status stale; refusing target')
                if state[0] != 0x40:
                    raise RuntimeError(f'Motor {i} not enabled or faulted (0x{state[0]:02x}); enable is operator-only')
            outside = (q < self._limits[:, 0]) | (q > self._limits[:, 1])
            if np.any(outside):
                with self._state:
                    measured = np.asarray(self._joints, dtype=float)
                excess = np.maximum(self._limits[:, 0] - q, q - self._limits[:, 1])
                clipped = np.clip(q, self._limits[:, 0], self._limits[:, 1])
                rejected = bool(np.any(excess[outside] > 0.005))
                if not rejected:
                    # Allow at most one CAN count of solver/rounding residue, projected inside.
                    scale = 180000 / math.pi
                    within_count = np.abs(np.rint(q * scale) - np.rint(clipped * scale)) <= 1
                    # Startup blends from measured boundary residuals: accept only motion back toward that bound.
                    inward = (((measured < self._limits[:, 0]) & (q >= measured) & (q < self._limits[:, 0])) |
                              ((measured > self._limits[:, 1]) & (q <= measured) & (q > self._limits[:, 1])))
                    measured_excess = np.maximum(self._limits[:, 0] - measured, measured - self._limits[:, 1])
                    inward &= measured_excess <= 0.005
                    rejected = bool(np.any(~within_count[outside] & ~inward[outside] &
                                           (np.abs(q[outside] - measured[outside]) > 0.001)))
                if rejected:
                    bad = (np.flatnonzero(outside) + 1).tolist()
                    raise ValueError(f'{self.side}: target exceeds Piper URDF joint limits at joints {bad}; target={q.tolist()}, measured={measured.tolist()}')
                q = np.clip(q, self._limits[:, 0], self._limits[:, 1])
            scale = 180000 / math.pi
            # Round inward at non-representable URDF endpoints, so encoded angles stay legal too.
            raw = np.clip(np.rint(q * scale), np.ceil(self._limits[:, 0] * scale),
                          np.floor(self._limits[:, 1] * scale)).astype(np.int64)
            joints = ArmMsgJointCtrl(*map(int, raw))
            messages = []
            if not self._mode_initialized:
                messages.append(PiperMessage(type_=ArmMsgType.PiperMsgMotionCtrl_2,
                                             arm_motion_ctrl_2=ArmMsgMotionCtrl_2(0x01, 0x01, 100, 0)))
            messages.extend(PiperMessage(type_=kind, arm_joint_ctrl=joints) for kind in (
                ArmMsgType.PiperMsgJointCtrl_12, ArmMsgType.PiperMsgJointCtrl_34, ArmMsgType.PiperMsgJointCtrl_56))
            if jaw is not None:
                messages.append(PiperMessage(type_=ArmMsgType.PiperMsgGripperCtrl, arm_gripper_ctrl=jaw))
            frames = []
            for message in messages:
                frame = can.Message(is_extended_id=False)
                if not self._parser.EncodeMessage(message, frame):
                    raise ValueError('SDK could not encode target')
                if len(frame.data) != 8:
                    raise ValueError('SDK encoded a non-eight-byte Piper frame')
                frame.dlc = len(frame.data)
                frames.append(frame)
            try:
                for frame in frames:
                    self.bus.send(frame, timeout=0.02)
            except Exception:
                self._failed = True
                raise
            self._mode_initialized = True

    def torque_off(self):
        # Brakeless arms must not be disabled; close the link after home and leave controller hold on.
        self.close()
        _log.warning('brakeless Piper motors remain powered under controller hold; turn power off manually with the operator present')
