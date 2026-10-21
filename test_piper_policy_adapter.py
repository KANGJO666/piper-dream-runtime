import json
import tempfile
import unittest
from pathlib import Path

from piper_policy_adapter import (
    CAMERA_MAP,
    SCHEMA_VERSION,
    STATE_NAMES,
    audit_episode,
    build_index,
    policy_adapter_manifest,
)


def make_episode(root: Path, *, status='complete', frames=4, task_success=None):
    path = root / 'episode'
    path.mkdir()
    rollout = {
        'status': status,
        'frames': frames,
        'task': 'stack bowls',
        'checkpoint': 'dreamlabs/dream-1-beta',
        'stop_reason': 'stopped',
        'duration_s': frames / 20.0,
    }
    if task_success is not None:
        rollout['task_success'] = task_success
    (path / 'rollout.json').write_text(json.dumps(rollout))
    (path / 'policy_adapter.json').write_text(json.dumps(policy_adapter_manifest(
        checkpoint='dreamlabs/dream-1-beta',
        embodiment='owner/piper-dual',
        active_arms=('right',),
        execute=True,
    )))
    features = {
        'observation.state': {'names': STATE_NAMES, 'shape': [14]},
        'action': {'names': STATE_NAMES, 'shape': [14]},
    }
    for camera in CAMERA_MAP.values():
        features[f'observation.images.{camera}'] = {'shape': [480, 640, 3]}
    (path / 'meta').mkdir()
    (path / 'meta' / 'info.json').write_text(json.dumps({'features': features}))
    parquet = path / 'data' / 'chunk-000' / 'episode_000000.parquet'
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    for camera in CAMERA_MAP.values():
        video = path / 'videos' / 'chunk-000' / f'observation.images.{camera}' / 'episode_000000.mp4'
        video.parent.mkdir(parents=True)
        video.touch()
    return path


class PiperPolicyAdapterTest(unittest.TestCase):
    def test_right_only_manifest_separates_observation_and_execution(self):
        manifest = policy_adapter_manifest(
            checkpoint='dreamlabs/dream-1-beta',
            embodiment='owner/piper-dual',
            active_arms=('right',),
            execute=True,
        )
        self.assertEqual(manifest['schema'], SCHEMA_VERSION)
        self.assertEqual(manifest['authority']['executed_arms'], ['right'])
        self.assertEqual(
            manifest['authority']['action_valid'],
            {'left': False, 'right': True},
        )
        self.assertEqual(manifest['model_boundary']['camera_roles']['overhead'], 'scene')
        self.assertEqual(manifest['dataset_boundary']['state_names'], STATE_NAMES)

    def test_preview_is_not_execution_evidence(self):
        manifest = policy_adapter_manifest(
            checkpoint='dreamlabs/dream-1-beta',
            embodiment='owner/piper-dual',
            active_arms=('right',),
            execute=False,
        )
        self.assertEqual(manifest['authority']['executed_arms'], [])
        self.assertEqual(
            manifest['authority']['action_valid'],
            {'left': False, 'right': False},
        )

    def test_complete_unlabelled_episode_needs_success_review(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp))
            report = audit_episode(episode)
        self.assertEqual(report['eligibility'], 'candidate_needs_task_success_label')
        self.assertFalse(report['eligible_for_training'])
        self.assertEqual(report['errors'], [])

    def test_success_label_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_episode(root, task_success='success')
            index = build_index(root)
        self.assertEqual(index['counts'], {'training_ready': 1})
        self.assertTrue(index['episodes'][0]['eligible_for_training'])

    def test_success_without_adapter_manifest_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp), task_success='success')
            (episode / 'policy_adapter.json').unlink()
            report = audit_episode(episode)
        self.assertFalse(report['eligible_for_training'])
        self.assertIsNone(report['policy_adapter'])

    def test_success_with_wrong_adapter_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp), task_success='success')
            path = episode / 'policy_adapter.json'
            manifest = json.loads(path.read_text())
            manifest['schema'] = 'wrong'
            path.write_text(json.dumps(manifest))
            report = audit_episode(episode)
        self.assertEqual(report['eligibility'], 'rejected')
        self.assertTrue(any('schema' in error for error in report['errors']))

    def test_success_without_executed_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp), task_success='success')
            path = episode / 'policy_adapter.json'
            manifest = json.loads(path.read_text())
            manifest['authority']['action_valid'] = {'left': False, 'right': False}
            path.write_text(json.dumps(manifest))
            report = audit_episode(episode)
        self.assertEqual(report['eligibility'], 'rejected')
        self.assertTrue(any('no executed arm' in error for error in report['errors']))

    def test_partial_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp), status='partial')
            report = audit_episode(episode)
        self.assertEqual(report['eligibility'], 'rejected')

    def test_failed_run_is_not_a_training_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = make_episode(Path(temp))
            rollout_path = episode / 'rollout.json'
            rollout = json.loads(rollout_path.read_text())
            rollout['stop_reason'] = 'ControlLoopError'
            rollout_path.write_text(json.dumps(rollout))
            report = audit_episode(episode)
        self.assertEqual(report['eligibility'], 'rejected')
        self.assertTrue(any('expected operator stop' in error for error in report['errors']))


if __name__ == '__main__':
    unittest.main()
