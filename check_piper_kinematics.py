"""Passive CAN comparison; sends no frames and never opens an SDK controller."""
import json, math, runpy, sysconfig, time
from pathlib import Path
import can
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FK = runpy.run_path(str(Path(sysconfig.get_path('purelib')) / 'piper_sdk/kinematics/piper_fk.py'))['C_PiperForwardKinematics']

def urdf_pose(q):
    robot = ET.parse(ROOT / 'arms/piper_reference.urdf').getroot()
    transform = np.eye(4)
    for joint in robot.findall('joint'):
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        step = np.eye(4)
        step[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        step[:3, 3] = xyz
        transform = transform @ step
        if joint.get('type') == 'revolute':
            axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec(axis * q[int(joint.get('name')[5:]) - 1]).as_matrix()
            transform = transform @ step
    return transform

def pose_matrix(pose):
    t = np.eye(4)
    t[:3, 3] = np.array(pose[:3]) / 1000
    t[:3, :3] = Rotation.from_euler('xyz', pose[3:], degrees=True).as_matrix()
    return t

def error(a, b):
    return {'position_mm': round(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1000, 4),
            'rotation_deg': round(float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()) * 180 / math.pi, 4)}

def main():
    result = {}
    for side, channel in (('left', 'can1'), ('right', 'can0')):
        bus = can.Bus(channel=channel, interface='socketcan', receive_own_messages=False)
        frames = {}; stamps = {}; statuses = {}; samples = []
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=0.05)
                if msg is None or msg.is_error_frame or msg.is_remote_frame or msg.is_extended_id or len(msg.data) != 8:
                    continue
                now = time.monotonic(); key = msg.arbitration_id
                if key == 0x2A1: statuses['arm'] = list(msg.data)
                if 0x261 <= key <= 0x266: statuses[str(key - 0x260)] = int(msg.data[5])
                if key in (0x2A2, 0x2A3, 0x2A4, 0x2A5, 0x2A6, 0x2A7):
                    frames[key] = [int.from_bytes(msg.data[:4], 'big', signed=True), int.from_bytes(msg.data[4:8], 'big', signed=True)]
                    stamps[key] = now
                if len(frames) == 6 and max(stamps.values()) - min(stamps.values()) < 0.05:
                    q = np.array(sum((frames[k] for k in (0x2A5, 0x2A6, 0x2A7)), [])) * math.pi / 180000
                    pose = np.array(sum((frames[k] for k in (0x2A2, 0x2A3, 0x2A4)), [])) / 1000
                    actual = pose_matrix(pose)
                    nominal = urdf_pose(q)
                    samples.append({'q_rad': q.tolist(), 'controller_pose_mm_deg': pose.tolist(), 'urdf_error': error(actual, nominal),
                                    'sdk_offset0_error': error(actual, pose_matrix(FK(0).CalFK(q)[-1])),
                                    'sdk_offset1_error': error(actual, pose_matrix(FK(1).CalFK(q)[-1]))})
                    frames.clear(); stamps.clear()
        finally:
            bus.shutdown()
        result[side] = {'samples': len(samples), 'last': samples[-1] if samples else None, 'status_bytes': statuses}
    assert all(v['samples'] > 5 for v in result.values()), 'Missing synchronized passive feedback'
    path = ROOT / 'logs/kinematics_comparison.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
