"""Dream Piper entry: real dual observations; only can0 may transmit. Preview by default."""
import argparse
import asyncio
import contextlib
import dataclasses
import fcntl
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import run_piper
from piper_policy_adapter import policy_adapter_manifest, write_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Use only the right arm to stack the bowls into one stable stack. Keep the left arm stationary.')
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--execute', action='store_true', help='Operator only: enable right-arm command dispatch')
    args = parser.parse_args()
    run_piper.environment()
    from can.interfaces.socketcan import SocketcanBus
    from robots import arms_lock, config, run
    from robots.cli import paths

    folder = ROOT / 'logs' / ('right_' + __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    dataset_dir = ROOT / 'logs' / 'rollouts_right'
    existing_episodes = (
        {path for path in dataset_dir.iterdir() if path.is_dir()}
        if args.execute and dataset_dir.exists()
        else set()
    )
    os.environ['PIPER_RIGHT_EXECUTE'] = '1' if args.execute else '0'
    os.environ['PIPER_RIGHT_RUN_DIR'] = str(folder)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    app = run.with_task(config.load(ROOT / 'my_robot_right.yaml'), args.task)
    if args.execute:
        app = dataclasses.replace(
            app,
            rollouts=dataclasses.replace(
                app.rollouts,
                enabled=True,
                dir=str(dataset_dir),
                robot_name='piper-right',
            ),
        )
    app = dataclasses.replace(
        app,
        inference_log_path=str(folder / 'inference.jsonl'),
        trace_path=str(folder / 'trace.csv'),
    )
    adapter = policy_adapter_manifest(
        checkpoint=app.checkpoint,
        embodiment=app.custom.embodiment,
        active_arms=('right',),
        execute=args.execute,
        action_space=app.action_space,
        horizon=app.action_horizon,
        action_hz=app.control_hz,
    )
    adapter['run_log_dir'] = str(folder)
    adapter_path = folder / 'policy_adapter.json'
    write_json(adapter_path, adapter)
    dataset_adapter_path = None
    if args.execute:
        adapter['dataset_root'] = str(dataset_dir)
        write_json(dataset_dir / 'policy_adapter.json', adapter)
        dataset_adapter_path = dataset_dir / 'adapters' / f'{folder.name}.json'
        write_json(dataset_adapter_path, adapter)

    metrics = {
        'pid': os.getpid(),
        'execute': args.execute,
        'left_can_sends': 0,
        'right_can_sends': 0,
        'blocked_can_sends': 0,
        'real_observations': True,
        'log_dir': str(folder),
        'policy_adapter': str(adapter_path),
        'dataset_policy_adapter': None if dataset_adapter_path is None else str(dataset_adapter_path),
        'dataset_dir': str(dataset_dir) if args.execute else None,
        'task_success': 'unverified',
    }
    original_send = SocketcanBus.send

    def guarded_send(bus, *a, **kw):
        channel = getattr(bus, 'channel', None)
        if not args.execute or channel != 'can0':
            metrics['blocked_can_sends'] += 1
            raise RuntimeError(f'CAN transmission forbidden: {channel}')
        result = original_send(bus, *a, **kw)
        metrics['right_can_sends'] += 1
        return result

    (folder / 'status.json').write_text(json.dumps(metrics, indent=2))
    print('RIGHT_ONLY_READY ' + json.dumps(metrics), flush=True)
    try:
        with contextlib.ExitStack() as stack:
            existing = Path.home() / '.dreamlabs' / arms_lock.LOCK_NAME
            if existing.exists():
                lock = stack.enter_context(existing.open('r'))
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stack.enter_context(arms_lock.acquire(paths().lock, arms_lock.ROBOT_RUN))
            stack.enter_context(patch.object(SocketcanBus, 'send', guarded_send))
            asyncio.run(run_piper.session(app, True, args.max_seconds))
        metrics['status'] = 'session_ended'
    except BaseException as exc:
        metrics.update(status='failed', error=repr(exc))
        raise
    finally:
        if args.execute and dataset_dir.exists():
            # Bind the adapter contract to every new recording before the next run can reuse it.
            for rollout_path in dataset_dir.glob('*/rollout.json'):
                if rollout_path.parent in existing_episodes:
                    continue
                episode_adapter = rollout_path.parent / 'policy_adapter.json'
                if not episode_adapter.exists():
                    write_json(episode_adapter, adapter)
        (folder / 'status.json').write_text(json.dumps(metrics, indent=2))
        print('RIGHT_ONLY_RESULT ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
