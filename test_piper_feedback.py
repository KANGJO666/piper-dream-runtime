"""Exercise the real receiver process with synthetic frames and no hardware access."""
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

class FakeSource:
    def __init__(self, age=0):
        self.age = age
        self.index = 0
        self.sent = []
        self.socket = SimpleNamespace(setsockopt=Mock())
    def recv(self, timeout):
        time.sleep(0.001)
        keys = [0x2A1, *range(0x261, 0x267), 0x2A5, 0x2A6, 0x2A7, 0x2A8]
        key = keys[self.index % len(keys)]
        self.index += 1
        if key == 0x2A1:
            data = bytes([1,0,1,0,0,0,0,0])
        elif 0x261 <= key <= 0x266:
            data = bytes([0,0,0,0,0,0x40,0,0])
        else:
            pair = {0x2A5:(0,10000),0x2A6:(-10000,0),0x2A7:(0,0),0x2A8:(30000,0)}[key]
            data = b''.join(v.to_bytes(4,'big',signed=True) for v in pair)
        return SimpleNamespace(arbitration_id=key,data=data,timestamp=time.time()-self.age,
                               is_error_frame=False,is_remote_frame=False,is_extended_id=False)
    def send(self, frame, timeout):
        self.sent.append(frame)
    def shutdown(self):
        pass

class ProcessFeedback(unittest.TestCase):
    def open_arm(self, age=0):
        import piper_driver
        from piper_control import PiperControlDriver
        self.bus = FakeSource()
        original = subprocess.Popen
        def spawn(args, **kwargs):
            code = ("from test_piper_feedback import FakeSource; from piper_feedback import run_receiver; "
                    f"run_receiver({args[-3]!r}, {args[-2]!r}, {int(args[-1])}, "
                    f"bus_factory=lambda **kwargs: FakeSource(age={age!r}))")
            return original([sys.executable,'-B','-c',code], **kwargs)
        with patch.object(piper_driver.can,'Bus',return_value=self.bus), \
             patch.object(piper_driver,'SocketcanBus',FakeSource), \
             patch.object(piper_driver.subprocess,'Popen',side_effect=spawn):
            arm = PiperControlDriver(SimpleNamespace(can_channel='can0'),'right')
        self.addCleanup(arm.close)
        return arm
    def test_fresh_after_parent_stall_and_child_cleanup(self):
        arm = self.open_arm()
        self.assertEqual(self.bus.sent,[])
        self.assertIsNone(arm._rx_thread)
        child = arm._rx_process
        first = arm._snapshot_seq
        # Block parent Python threads long enough to exceed the feedback deadline.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1.0)
        try:
            until = time.monotonic()+0.35
            while time.monotonic()<until:
                pass
        finally:
            sys.setswitchinterval(previous)
        arm.write([0,0.2,-0.2,0,0,0],0.5)
        self.assertGreater(arm._snapshot_seq,first)
        self.assertLess(time.monotonic()-arm._feedback_at,0.15)
        self.assertEqual(len(self.bus.sent),5)
        self.assertEqual(len(arm._motor_status),6)
        self.assertAlmostEqual(arm.read().positions_rad[1],10000*math.pi/180000)
        destination=arm._snapshot
        arm.close()
        self.assertIsNotNone(child.poll())
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix('.tmp').exists())
    def test_receiver_death_blocks_all_targets(self):
        arm=self.open_arm()
        arm._rx_process.kill()
        arm._rx_process.wait(timeout=1)
        with self.assertRaisesRegex(RuntimeError,'CAN receiver failed'):
            arm.write([0,0.2,-0.2,0,0,0],0.5)
        self.assertEqual(self.bus.sent,[])
    def test_old_kernel_frames_remain_stale_in_process(self):
        with self.assertRaises(TimeoutError):
            self.open_arm(age=1.0)
        self.assertEqual(self.bus.sent,[])
    def test_worker_rejects_snapshot_outside_project(self):
        from piper_feedback import run_receiver
        with self.assertRaisesRegex(ValueError,'inside the project'):
            run_receiver('can0','/tmp/forbidden-piper-feedback.json',os.getpid(),FakeSource)
    def test_receiver_ignores_terminal_interrupt(self):
        arm = self.open_arm()
        child = arm._rx_process
        os.kill(child.pid, signal.SIGINT)
        time.sleep(0.05)
        self.assertIsNone(child.poll())

if __name__=='__main__':
    unittest.main()
