"""Piper entry point: default dry run; --execute is for the operator at the robot.

The live path reuses Dream's ControlLoop, without the CLI's home/torque-off lifecycle.
Stopping closes communications after the loop's hold; motors remain powered.
"""
import argparse
import contextlib
import fcntl
import asyncio
import dataclasses
import logging
import math
import os
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('piper.entry')


def environment():
    os.chdir(ROOT)
    folders = {'TMPDIR': '.runtime/tmp', 'TMP': '.runtime/tmp', 'TEMP': '.runtime/tmp',
               'XDG_CACHE_HOME': '.runtime/cache', 'MPLCONFIGDIR': '.runtime/matplotlib',
               'DREAMCLI_HOME': '.runtime/station', 'DREAM_LOG_DIR': 'logs',
               'JAX_COMPILATION_CACHE_DIR': '.runtime/cache/jax',
               'CUDA_CACHE_PATH': '.runtime/cache/cuda', 'NUMBA_CACHE_DIR': '.runtime/cache/numba'}
    for key, folder in folders.items():
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # Leave CPU time for the CAN receivers instead of spawning 48 workers per library.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import cv2
    cv2.setNumThreads(1)


async def drive(loop, stop):
    async with loop:
        running = asyncio.create_task(loop.run())
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((running, stopping), return_when=asyncio.FIRST_COMPLETED)
            if running in done:
                running.result()
        finally:
            for task in (running, stopping):
                task.cancel()
            await asyncio.gather(running, stopping, return_exceptions=True)


def preflight_feedback(app):
    """Read both followers before camera/API startup; this driver cannot send targets."""
    from piper_readonly import PiperReadOnlyDriver
    for side, cfg in (('left', app.follower_l), ('right', app.follower_r)):
        arm = PiperReadOnlyDriver(cfg, side)
        try:
            state = arm.read()
            LOG.info('START_FEEDBACK %s/%s: gripper=%.4f joints_rad=%s',
                     side, cfg.can_channel, state.gripper, state.positions_rad.tolist())
        finally:
            arm.close()


async def execute(app, stop):
    from robots import bodies, control, run
    readers = []
    arms = {}
    client = None
    recorder = None
    try:
        await asyncio.to_thread(preflight_feedback, app)
        readers = await asyncio.to_thread(run.make_cameras, app)
        kinematics = run.make_kinematics(app)
        policy = run.policy_config(app, None)
        # Let the loop retry a stuck request before its 12-second empty-buffer stop.
        policy['timeout_ms'] = int(app.action_timeout_s * 1000)
        client = await asyncio.to_thread(run.open_client, policy)
        if stop.is_set():
            return
        for reader in readers:
            reader.start()
        if any(reader.exposure is not None for reader in readers):
            await asyncio.sleep(run.EXPOSURE_SETTLE_S)
        for reader in readers:
            reader.set_exposure_hold(True)
        if stop.is_set():
            return
        body = bodies.for_config(app)
        for side, cfg in (('left', app.follower_l), ('right', app.follower_r)):
            arms[side] = body.open_arm(cfg, side)
        if stop.is_set():
            return
        recorder = run.make_recorder(app, app.checkpoint, run_number=1)
        loop = control.ControlLoop(followers=arms, cameras=readers, client=client, task=app.task,
                                   kinematics=kinematics, inference_log_path=app.inference_log_path,
                                   camera_owner=False, recorder=recorder, **run.loop_settings(app))
        await drive(loop, stop)
    finally:
        for reader in readers:
            control.run_teardown('stop camera', reader.stop)
        if client is not None:
            control.run_teardown('close API client', client.close)
        for side, arm in arms.items():
            control.run_teardown(f'close {side} CAN (keep powered)', arm.close)
        if recorder is not None:
            await asyncio.to_thread(recorder.wait)
        LOG.warning('Run ended: motors were NOT disabled. No home/park command was issued by this entry point.')


async def session(app, live, max_seconds=180.0):
    from robots import run
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be finite and greater than zero')
    stop = asyncio.Event()
    event_loop = asyncio.get_running_loop()
    reason = 'loop_returned'

    def request_stop(cause):
        nonlocal reason
        if not stop.is_set():
            reason = cause
            LOG.warning('RUN_STOP_REQUEST reason=%s; stopping policy, task success is unverified', cause)
            stop.set()

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    for sig in signals:
        event_loop.add_signal_handler(sig, request_stop, 'operator_signal')
    timer = event_loop.call_later(max_seconds, request_stop, 'time_limit')
    LOG.info('RUN_LIMIT seconds=%g including initialization; cleanup follows policy stop', max_seconds)
    try:
        if live:
            await execute(app, stop)
        else:
            await run.main(app, dry_run=True, stop=stop)
    except BaseException:
        reason = 'error'
        raise
    finally:
        timer.cancel()
        for sig in signals:
            event_loop.remove_signal_handler(sig)
        LOG.warning('RUN_ENDED reason=%s task_success=unverified; no automatic home', reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--max-seconds', type=float, default=180.0, help='Policy stop deadline including initialization; not task success')
    parser.add_argument('--execute', action='store_true', help='OPERATOR ONLY: send targets; motors must already be enabled')
    args = parser.parse_args()
    environment()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    from robots import arms_lock, config, run
    from robots.cli import paths
    app = config.load(ROOT / 'my_robot_dryrun_nominal.yaml')
    app = run.with_task(app, args.task)
    if args.execute:
        app = dataclasses.replace(app, custom=dataclasses.replace(app.custom,
            driver=str(ROOT / 'piper_control.py') + ':PiperControlDriver'),
            max_ee_speed_m_s=0.10, inference_log_path=str(ROOT / 'logs/piper_live_inference.jsonl'),
            trace_path=str(ROOT / 'logs/piper_live_trace.csv'),
            rollouts=dataclasses.replace(app.rollouts, enabled=True, dir=str(ROOT / 'logs/rollouts')))
        LOG.warning('LIVE operator run: nominal TCP, 0.10 m/s EEF cap; start here, stop powered, no automatic homing.')
    run.check_tool_point(app, skip=True)
    with contextlib.ExitStack() as stack:
        existing = Path.home() / '.dreamlabs' / arms_lock.LOCK_NAME
        if existing.exists():
            # Lock the existing station file read-only; never create/write it outside the project.
            default_lock = stack.enter_context(existing.open('r'))
            try:
                fcntl.flock(default_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another robot process holds the default station lock') from None
        stack.enter_context(arms_lock.acquire(paths().lock, arms_lock.ROBOT_RUN))
        asyncio.run(session(app, args.execute, args.max_seconds))


if __name__ == '__main__':
    main()
