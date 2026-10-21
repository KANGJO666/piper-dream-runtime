"""Offline powered-lifecycle tests: no camera, API or CAN device is opened."""
import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from run_piper import drive, execute, session

class FakeLoop:
    def __init__(self, stop, fail=False):
        self.stop=stop;self.fail=fail;self.exited=False;self.cancelled=False
    async def __aenter__(self):return self
    async def __aexit__(self,*args):self.exited=True
    async def run(self):
        if self.fail:raise RuntimeError('injected loop failure')
        self.stop.set()
        try:await asyncio.Event().wait()
        finally:self.cancelled=True

class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_stop_cancels_loop_before_context_closes(self):
        stop=asyncio.Event();loop=FakeLoop(stop)
        await drive(loop,stop)
        self.assertTrue(loop.cancelled);self.assertTrue(loop.exited)
    async def test_loop_error_propagates_and_closes(self):
        loop=FakeLoop(asyncio.Event(),fail=True)
        with self.assertRaisesRegex(RuntimeError,'injected'):await drive(loop,loop.stop)
        self.assertTrue(loop.exited)
    async def execute_case(self,loop_error=False,second_arm_error=False,prestop=False):
        stop=asyncio.Event()
        if prestop:stop.set()
        loop=FakeLoop(stop,fail=loop_error)
        reader=Mock();reader.exposure=None
        client=Mock()
        opener=Mock(return_value=client)
        recorder=Mock()
        left=Mock();right=Mock()
        left.torque_off.side_effect=AssertionError('must not disable')
        right.torque_off.side_effect=AssertionError('must not disable')
        body=Mock();body.open_arm.side_effect=[left,RuntimeError('second arm failed') if second_arm_error else right]
        app=SimpleNamespace(follower_l=object(),follower_r=object(),task='test',inference_log_path=None,action_timeout_s=3.0,checkpoint='test/model')
        with ExitStack() as stack:
            for target,value in (
                ('run_piper.preflight_feedback',Mock()),
                ('robots.run.make_cameras',Mock(return_value=[reader])),
                ('robots.run.make_recorder',Mock(return_value=recorder)),
                ('robots.run.make_kinematics',Mock(return_value={})),
                ('robots.run.open_client',opener),
                ('robots.run.policy_config',Mock(return_value={})),
                ('robots.run.loop_settings',Mock(return_value={})),
                ('robots.bodies.for_config',Mock(return_value=body)),
                ('robots.control.ControlLoop',Mock(return_value=loop)),
                ('robots.home.return_to_home',Mock(side_effect=AssertionError('must not home')))):
                stack.enter_context(patch(target,value))
            if loop_error or second_arm_error:
                with self.assertRaises(RuntimeError):await execute(app,stop)
            else:await execute(app,stop)
        opener.assert_called_once_with({'timeout_ms':3000})
        reader.stop.assert_called_once();client.close.assert_called_once()
        left.torque_off.assert_not_called();right.torque_off.assert_not_called()
        if not prestop and not second_arm_error:recorder.wait.assert_called_once()
        else:recorder.wait.assert_not_called()
        if prestop:body.open_arm.assert_not_called()
        else:left.close.assert_called_once()
        if not second_arm_error and not prestop:right.close.assert_called_once()
    async def test_operator_stop_closes_without_home_or_disable(self):await self.execute_case()
    async def test_fault_closes_without_home_or_disable(self):await self.execute_case(loop_error=True)
    async def test_partial_open_failure_closes_first_arm(self):await self.execute_case(second_arm_error=True)
    async def test_stop_before_open_never_opens_arms(self):await self.execute_case(prestop=True)

class StartFeedback(unittest.IsolatedAsyncioTestCase):
    async def test_bad_feedback_stops_before_cameras_api_or_control(self):
        from run_piper import execute
        with patch('run_piper.preflight_feedback', side_effect=ValueError('right/can0: bad gripper range')), \
             patch('robots.run.make_cameras') as cameras, \
             patch('robots.run.open_client') as client, \
             patch('robots.bodies.for_config') as body:
            with self.assertRaisesRegex(ValueError, 'right/can0'):
                await execute(object(), asyncio.Event())
        cameras.assert_not_called();client.assert_not_called();body.assert_not_called()

    async def test_passive_preflight_reads_and_closes_both_arms(self):
        from run_piper import preflight_feedback
        cfg=SimpleNamespace(can_channel='can0')
        left=Mock();right=Mock()
        with patch('piper_readonly.PiperReadOnlyDriver', side_effect=[left,right]) as driver:
            preflight_feedback(SimpleNamespace(follower_l=cfg,follower_r=cfg))
        self.assertEqual(driver.call_count,2)
        for arm in (left,right):
            arm.read.assert_called_once();arm.close.assert_called_once();arm.write.assert_not_called()

    async def test_preflight_closes_failed_arm_and_does_not_open_next(self):
        from run_piper import preflight_feedback
        arm=Mock();arm.read.side_effect=ValueError('out of range')
        cfg=SimpleNamespace(can_channel='can1')
        with patch('piper_readonly.PiperReadOnlyDriver', return_value=arm) as driver:
            with self.assertRaisesRegex(ValueError,'out of range'):
                preflight_feedback(SimpleNamespace(follower_l=cfg,follower_r=cfg))
        driver.assert_called_once();arm.close.assert_called_once();arm.write.assert_not_called()


class TimeLimit(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_stops_live_and_reports_time_limit(self):
        async def wait_for_stop(app, stop):
            await asyncio.wait_for(stop.wait(), timeout=1)
        with patch('run_piper.execute', side_effect=wait_for_stop), self.assertLogs('piper.entry', level='WARNING') as logs:
            await session(object(), True, max_seconds=0.02)
        self.assertTrue(any('RUN_ENDED reason=time_limit task_success=unverified' in line for line in logs.output))

    async def test_failure_reports_error_and_propagates(self):
        with patch('run_piper.execute', side_effect=RuntimeError('injected failure')), self.assertLogs('piper.entry', level='WARNING') as logs:
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                await session(object(), True, max_seconds=1)
        self.assertTrue(any('RUN_ENDED reason=error' in line for line in logs.output))

    async def test_invalid_deadline_never_opens_hardware(self):
        with patch('run_piper.execute') as live:
            for seconds in (0, -1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    await session(object(), True, max_seconds=seconds)
            live.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
