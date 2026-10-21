"""Passive Piper CAN feedback; live writes are implemented in piper_control.py."""

import math
import time
import threading
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
from can.interfaces.socketcan import SocketcanBus

import can
import numpy as np

from robots import config, followers


# J1/J2, J3/J4, J5/J6 feedback
JOINT_IDS = {
    0x2A5: (0, 1),
    0x2A6: (2, 3),
    0x2A7: (4, 5),
}

GRIPPER_ID = 0x2A8
RAD_PER_RAW = math.pi / 180000.0
# Operator specifies a 70 mm full opening with +/-0.5 mm endpoint tolerance.
GRIPPER_FEEDBACK_TOLERANCE_MM = 0.5


class PiperDriver(followers.Driver):
    drops_when_limp = False

    def __init__(self, cfg: config.ArmConfig, side: str):
        self.channel = cfg.can_channel
        self.side = side
        self._lock = threading.Lock()
        self._closed = False

        if self.channel not in ("can0", "can1"):
            raise ValueError(
                f"Unexpected CAN channel: {self.channel}"
            )

        self.bus = can.Bus(
            channel=self.channel,
            interface="socketcan",
            receive_own_messages=False,
            # Filter in the kernel: keep every joint, gripper, controller and motor-health frame.
            can_filters=[{'can_id': key, 'can_mask': 0x7FF, 'extended': False}
                         for key in (0x2A1, *JOINT_IDS, GRIPPER_ID, *range(0x261, 0x267))],
        )

        self._last_positions = None
        self._last_time = None

        self._motor_status = {}
        self._arm_status = None
        self._feedback_at = None

        # Must be measured before using Dream-1 read().
        self.gripper_closed_mm = 0.0
        self.gripper_open_mm = 70.0
        self._state = threading.Condition()
        self._rx_stop = threading.Event()
        self._rx_error = None
        self._joints = [None] * 6
        self._joint_stamps = [None] * 6
        self._gripper_mm = None
        self._gripper_stamp = None
        self._rx_thread = None
        self._rx_process = None
        self._snapshot = None
        self._snapshot_seq = 0
        self._snapshot_mtime = None
        try:
            if isinstance(self.bus, SocketcanBus):
                folder = Path(__file__).resolve().parent / '.runtime' / 'feedback'
                folder.mkdir(parents=True, exist_ok=True)
                self._snapshot = folder / f'{self.channel}-{os.getpid()}-{uuid.uuid4().hex}.json'
                # The child receives CAN; this socket only sends, with no duplicate receive queue.
                self.bus.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b'')
                self._rx_process = subprocess.Popen(
                    [sys.executable, '-B', str(Path(__file__).with_name('piper_feedback.py')),
                     self.channel, str(self._snapshot), str(os.getpid())],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, cwd=str(Path(__file__).resolve().parent),
                    start_new_session=True)
            else:
                self._rx_thread = threading.Thread(target=self._receive, name=f'piper-rx-{self.channel}', daemon=True)
                self._rx_thread.start()
        except Exception:
            self.bus.shutdown()
            raise

    def _refresh_feedback(self):
        if self._rx_process is None:
            return
        if self._rx_process.poll() is not None:
            self._rx_error = RuntimeError(f'receiver process exited with {self._rx_process.returncode}')
            self._feedback_at = None
            raise RuntimeError(f'{self.channel}: CAN receiver failed') from self._rx_error
        try:
            mtime = self._snapshot.stat().st_mtime_ns
            if mtime == self._snapshot_mtime:
                return
            data = json.loads(self._snapshot.read_text())
        except FileNotFoundError:
            return
        if data['pid'] != self._rx_process.pid or data['channel'] != self.channel:
            raise RuntimeError('Unexpected feedback snapshot owner')
        with self._state:
            if data['seq'] <= self._snapshot_seq:
                return
            if 'error' in data:
                self._rx_error = RuntimeError(data['error'])
                self._feedback_at = None
                raise RuntimeError(f'{self.channel}: CAN receiver failed') from self._rx_error
            self._snapshot_seq = data['seq']
            self._snapshot_mtime = mtime
            self._joints = data['joints']
            self._joint_stamps = data['joint_stamps']
            self._gripper_mm = data['gripper_mm']
            self._gripper_stamp = data['gripper_stamp']
            self._motor_status = {int(k): tuple(v) for k, v in data['motor_status'].items()}
            arm = data['arm_status']
            self._arm_status = (bytes(arm[0]), arm[1]) if arm is not None else None
            self._feedback_at = data['feedback_at']
            self._state.notify_all()

    def _receive(self):
        """Continuously receive all feedback, independent of policy/read cadence."""
        try:
            while not self._rx_stop.is_set():
                msg = self.bus.recv(timeout=0.02)
                if msg is None or msg.is_error_frame or msg.is_remote_frame or msg.is_extended_id or len(msg.data) != 8:
                    continue
                now = time.monotonic()
                # Preserve kernel arrival age: a queued old frame must not look new.
                age = max(0.0, time.time() - msg.timestamp) if msg.timestamp > 0 else 0.0
                stamp = now - age
                key = msg.arbitration_id
                with self._state:
                    if 0x261 <= key <= 0x266:
                        self._motor_status[key - 0x260] = (msg.data[5], stamp)
                    elif key == 0x2A1:
                        self._arm_status = (bytes(msg.data), stamp)
                    elif key in JOINT_IDS:
                        for index, raw in zip(JOINT_IDS[key], (msg.data[:4], msg.data[4:8])):
                            self._joints[index] = int.from_bytes(raw, 'big', signed=True) * RAD_PER_RAW
                            self._joint_stamps[index] = stamp
                    elif key == GRIPPER_ID:
                        self._gripper_mm = int.from_bytes(msg.data[:4], 'big', signed=True) / 1000.0
                        self._gripper_stamp = stamp
                    stamps = self._joint_stamps + [self._gripper_stamp]
                    if all(t is not None for t in stamps):
                        self._feedback_at = min(stamps)
                    self._state.notify_all()
        except Exception as exc:
            if not self._rx_stop.is_set():
                with self._state:
                    self._rx_error = exc
                    self._feedback_at = None
                    self._state.notify_all()

    def sample_raw(self):
        """Return a fresh cached sample; wait only if none is fresh."""
        deadline = time.monotonic() + 0.25
        with self._state:
            while True:
                if self._closed:
                    raise RuntimeError('CAN bus already closed')
                self._refresh_feedback()
                if self._rx_error is not None:
                    raise RuntimeError(f'{self.channel}: CAN receiver failed') from self._rx_error
                now = time.monotonic()
                if self._feedback_at is not None and now - self._feedback_at <= 0.15:
                    return np.array(self._joints, dtype=np.float64), self._gripper_mm, now
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(f'{self.channel}: missing or stale joint/gripper feedback')
                self._state.wait(min(remaining, 0.01) if self._rx_process is not None else remaining)

    def read_joints(self):
        """Fresh six-joint feedback; joint homing does not interpret gripper position."""
        return self.sample_raw()[0]

    def read(self) -> followers.Reading:
        positions, gripper_mm, now = self.sample_raw()

        if (
            self.gripper_closed_mm is None
            or self.gripper_open_mm is None
        ):
            raise RuntimeError(
                "Gripper calibration required before read()"
            )

        span = self.gripper_open_mm - self.gripper_closed_mm

        if not math.isfinite(span) or span <= 0:
            raise ValueError("Invalid gripper range")

        grip = (gripper_mm - self.gripper_closed_mm) / span

        feedback_min = self.gripper_closed_mm - GRIPPER_FEEDBACK_TOLERANCE_MM
        feedback_max = self.gripper_open_mm + GRIPPER_FEEDBACK_TOLERANCE_MM
        if not feedback_min <= gripper_mm <= feedback_max:
            raise ValueError(
                f'{self.side}/{self.channel}: Gripper outside calibrated range: '
                f'feedback={gripper_mm:.3f} mm, configured closed={self.gripper_closed_mm:.3f} mm, '
                f'open={self.gripper_open_mm:.3f} mm, '
                f'accepted feedback=[{feedback_min:.3f}, {feedback_max:.3f}] mm. '
                'Joint homing does not calibrate the gripper; verify physical endpoints before changing its range.'
            )

        grip = float(np.clip(grip, 0.0, 1.0))

        with self._lock:
            if self._last_positions is None:
                velocities = np.zeros(6, dtype=np.float64)
            else:
                dt = now - self._last_time

                if dt <= 0:
                    raise RuntimeError("Invalid timestamp")

                velocities = (
                    positions - self._last_positions
                ) / dt

            self._last_positions = positions.copy()
            self._last_time = now

        return followers.Reading(
            positions.copy(),
            velocities,
            grip,
        )

    def write(self, positions_rad, gripper):
        raise RuntimeError("Passive driver: use PiperControlDriver for operator-run motion")

    def disengage(self):
        return False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._rx_stop.set()
        with self._state:
            self._state.notify_all()
        try:
            if self._rx_process is not None:
                if self._rx_process.poll() is None:
                    self._rx_process.terminate()
                try:
                    self._rx_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._rx_process.kill()
                    self._rx_process.wait(timeout=1)
                self._snapshot.unlink(missing_ok=True)
                self._snapshot.with_suffix('.tmp').unlink(missing_ok=True)
        finally:
            self.bus.shutdown()
            if self._rx_thread is not None:
                self._rx_thread.join(timeout=0.5)
                if self._rx_thread.is_alive():
                    raise RuntimeError(f'{self.channel}: CAN receiver did not stop')

    def torque_off(self):
        raise RuntimeError(
            "READ-ONLY MODE: torque control disabled"
        )
