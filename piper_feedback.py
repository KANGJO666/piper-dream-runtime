"""Receive-only SocketCAN worker; publishes atomic snapshots inside the project."""
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

JOINT_IDS = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
FILTER_IDS = (0x2A1, *JOINT_IDS, 0x2A8, *range(0x261, 0x267))


def run_receiver(channel, destination, parent_pid, bus_factory=None):
    if channel not in ('can0', 'can1'):
        raise ValueError('Unexpected CAN channel')
    destination = Path(destination).resolve()
    root = Path(__file__).resolve().parent / '.runtime' / 'feedback'
    if destination.parent != root.resolve():
        raise ValueError('Feedback snapshots must stay inside the project')
    temporary = destination.with_suffix('.tmp')
    state = {'seq': 0, 'pid': os.getpid(), 'channel': channel,
             'joints': [None] * 6, 'joint_stamps': [None] * 6,
             'gripper_mm': None, 'gripper_stamp': None, 'feedback_at': None,
             'motor_status': {}, 'arm_status': None}
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def publish():
        state['seq'] += 1
        temporary.write_text(json.dumps(state, allow_nan=False))
        os.replace(temporary, destination)

    bus = None
    try:
        if bus_factory is None:
            import can
            bus_factory = can.Bus
        bus = bus_factory(channel=channel, interface='socketcan', receive_own_messages=False,
                          can_filters=[{'can_id': key, 'can_mask': 0x7FF, 'extended': False}
                                       for key in FILTER_IDS])
        last_publish = 0.0
        while not stopped and os.getppid() == parent_pid:
            msg = bus.recv(timeout=0.02)
            now = time.monotonic()
            if msg is not None and not (msg.is_error_frame or msg.is_remote_frame or msg.is_extended_id) and len(msg.data) == 8:
                age = max(0.0, time.time() - msg.timestamp) if msg.timestamp > 0 else 0.0
                stamp = now - age
                key = msg.arbitration_id
                if key in JOINT_IDS:
                    for index, raw in zip(JOINT_IDS[key], (msg.data[:4], msg.data[4:8])):
                        state['joints'][index] = int.from_bytes(raw, 'big', signed=True) * math.pi / 180000
                        state['joint_stamps'][index] = stamp
                elif key == 0x2A8:
                    state['gripper_mm'] = int.from_bytes(msg.data[:4], 'big', signed=True) / 1000
                    state['gripper_stamp'] = stamp
                elif key == 0x2A1:
                    state['arm_status'] = [list(msg.data), stamp]
                elif 0x261 <= key <= 0x266:
                    state['motor_status'][str(key - 0x260)] = [msg.data[5], stamp]
                stamps = state['joint_stamps'] + [state['gripper_stamp']]
                if all(t is not None for t in stamps):
                    state['feedback_at'] = min(stamps)
            if now - last_publish >= 0.01:
                publish()
                last_publish = now
    except BaseException as exc:
        state['error'] = repr(exc)
        state['feedback_at'] = None
        publish()
        raise
    finally:
        if bus is not None:
            bus.shutdown()
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    run_receiver(sys.argv[1], sys.argv[2], int(sys.argv[3]))
