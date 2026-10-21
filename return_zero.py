"""Operator-triggered joint-zero return. Default is preview only; no gripper commands or encoder reset."""
import argparse
import contextlib
import fcntl
import json
import math
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET

from run_piper import ROOT, environment

PLAN = ROOT / 'logs/return_zero_plan.json'


def plan_for(positions_rad, side='arm'):
    import numpy as np
    q = np.asarray(positions_rad, dtype=float)
    model = ET.parse(ROOT / 'arms/piper_reference.urdf').getroot()
    joints = {j.get('name'): j.find('limit') for j in model.findall('joint') if j.get('type') == 'revolute'}
    limits = np.array([[float(joints[f'joint{i}'].get(k)) for k in ('lower', 'upper')] for i in range(1, 7)])
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError(f'{side}: expected six finite starting joint angles; gripper is excluded')
    invalid_zero = (limits[:, 0] > 0) | (limits[:, 1] < 0)
    if np.any(invalid_zero):
        names = ', '.join(f'J{i+1}' for i in np.flatnonzero(invalid_zero))
        raise ValueError(f'{side}: zero target is outside URDF limits at {names}; nothing moved')
    outside = (q < limits[:, 0] - 0.005) | (q > limits[:, 1] + 0.005)
    if np.any(outside):
        details = []
        for i in np.flatnonzero(outside):
            excess = max(limits[i, 0] - q[i], q[i] - limits[i, 1])
            details.append(f'J{i+1} measured={math.degrees(q[i]):+.4f} deg, '
                           f'URDF=[{math.degrees(limits[i,0]):.4f}, {math.degrees(limits[i,1]):.4f}] deg, '
                           f'excess={math.degrees(excess):.4f} deg')
        raise ValueError(f'{side}: starting arm joint outside URDF limits: ' + '; '.join(details) +
                         f'; allowed boundary residual={math.degrees(0.005):.4f} deg. '
                         'Gripper is excluded; check joint calibration and CAN feedback source. No targets sent')
    measured = q.copy()
    # Accept only a small measured boundary residual; every commanded point stays inside limits.
    q = np.clip(q, limits[:, 0], limits[:, 1])
    # Quintic blend has peak derivative 1.875: cap each joint at 0.10 rad/s.
    duration = max(3.0, 1.875 * float(np.max(np.abs(q))) / 0.10)
    return {'measured_start_rad': measured.tolist(), 'boundary_projection_rad': (q-measured).tolist(), 'start_rad': q.tolist(), 'target_rad': [0.0] * 6, 'duration_s': duration}


def zero_one(arm, plan):
    import numpy as np
    q = np.array(plan['start_rad'])
    previous = q.copy()
    duration = plan['duration_s']
    started = time.monotonic()
    previous_tick = started
    while True:
        now = time.monotonic()
        if now - previous_tick > 0.15:
            raise TimeoutError('Return-zero scheduler stalled; no further targets sent')
        previous_tick = now
        measured = arm.read_joints()
        if np.max(np.abs(measured - previous)) > 0.15:
            raise RuntimeError('Return-zero tracking error exceeds 0.15 rad')
        t = min(1.0, (now - started) / duration)
        alpha = 10*t**3 - 15*t**4 + 6*t**5
        target = q * (1.0 - alpha)
        arm.write_joints(target)
        previous = target
        if t >= 1.0:
            break
        time.sleep(0.02)
    deadline = time.monotonic() + 5.0
    stable = 0
    while time.monotonic() < deadline:
        measured = arm.read_joints()
        stable = stable + 1 if np.max(np.abs(measured)) <= 0.02 else 0
        if stable >= 10:
            return
        arm.write_joints(np.zeros(6))
        time.sleep(0.02)
    raise TimeoutError('Zero target sent but measured joints did not settle within 0.02 rad')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-clear-path', action='store_true', help='Operator confirms empty grippers and collision-free full return paths for both arms')
    args = parser.parse_args()
    if args.execute and not args.confirm_clear_path:
        parser.error('--execute requires --confirm-clear-path after reviewing the preview')
    environment()
    from robots import arms_lock, config
    from robots.cli import paths
    from piper_readonly import PiperReadOnlyDriver
    from piper_control import PiperControlDriver
    import numpy as np
    app = config.load(ROOT / 'my_robot_dryrun_nominal.yaml')
    selected = ('left', 'right') if args.arm == 'both' else (args.arm,)
    arms = {}
    def interrupted(*_):
        raise KeyboardInterrupt()

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with contextlib.ExitStack() as stack:
        existing = Path.home() / '.dreamlabs' / arms_lock.LOCK_NAME
        if existing.exists():
            lock = stack.enter_context(existing.open('r'))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.enter_context(arms_lock.acquire(paths().lock, arms_lock.ROBOT_RUN))
        try:
            driver = PiperControlDriver if args.execute else PiperReadOnlyDriver
            for side in selected:
                arms[side] = driver(app.follower_l if side == 'left' else app.follower_r, side)
            plans = {side: plan_for(arm.read_joints(), f'{side}/{arm.channel}') for side, arm in arms.items()}
            print(json.dumps({'order': selected, 'gripper_control': 'untouched; no gripper commands', 'plans': plans}, indent=2), flush=True)
            if not args.execute:
                PLAN.write_text(json.dumps({'created': time.time(), 'plans': plans}, indent=2))
                print('PREVIEW_ONLY: no targets sent. Review the complete swept paths; no collision model is available.')
                return
            reviewed = json.loads(PLAN.read_text())
            if not 0 <= time.time() - reviewed['created'] <= 600 or set(reviewed['plans']) != set(plans):
                raise ValueError('Preview expired or arms changed; run the preview again')
            for side, plan in plans.items():
                old = np.asarray(reviewed['plans'][side]['start_rad'], dtype=float)
                if old.shape != (6,) or not np.isfinite(old).all() or np.max(np.abs(old - plan['start_rad'])) > 0.03:
                    raise ValueError('Pose changed since preview; preview again')
            for side in selected:
                # Re-read before each arm: another arm must not have been repositioned meanwhile.
                fresh = plan_for(arms[side].read_joints(), f'{side}/{arms[side].channel}')
                if np.max(np.abs(np.array(fresh['start_rad']) - plans[side]['start_rad'])) > 0.03:
                    raise ValueError('Pose changed before return; stopping')
                print(f'ZERO_MOVING {side}', flush=True)
                zero_one(arms[side], fresh)
                print(f'ZERO_REACHED {side} tolerance_rad=0.02', flush=True)
            print('ZERO_COMPLETE: selected arms verified; no gripper commands sent; arm motors remain powered', flush=True)
        finally:
            for arm in arms.values():
                arm.close()


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ZERO_ABORTED: {type(exc).__name__}: {exc}. No automatic retry or power-off.', flush=True)
        raise SystemExit(1)
