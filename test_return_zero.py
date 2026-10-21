"""Offline zero-return checks; no CAN, camera or API is opened."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from return_zero import plan_for, zero_one

class Clock:
    def __init__(self): self.now = 0.0
    def monotonic(self): return self.now
    def sleep(self, dt): self.now += dt

class Arm:
    def __init__(self, q, follow=True, fail=False):
        self.q = np.array(q, dtype=float); self.follow = follow; self.fail = fail; self.sent = []
    def read_joints(self): return self.q.copy()
    def write_joints(self, q):
        if self.fail: raise TimeoutError('feedback lost')
        self.sent.append(np.array(q))
        if self.follow: self.q = np.array(q)

class ZeroReturn(unittest.TestCase):
    def test_plan_rejects_invalid_feedback_and_limits(self):
        for q in ([0]*5, [float('nan')]*6, [0,-0.1,0,0,0,0], [0,0,0.1,0,0,0]):
            with self.assertRaises(ValueError): plan_for(q)
    def test_small_boundary_residual_projects_to_legal_target(self):
        plan=plan_for(Arm([0,-0.004,0.002,0,0,0]).q)
        self.assertEqual(plan['start_rad'],[0]*6)
        self.assertAlmostEqual(max(abs(x) for x in plan['boundary_projection_rad']),0.004)
        with self.assertRaises(ValueError): plan_for(Arm([0,-0.006,0,0,0,0]).q)
    def test_duration_bounds_quintic_peak_speed(self):
        plan = plan_for(Arm([0.5,0.8,-0.6,0.2,0.1,-0.3]).q)
        self.assertLessEqual(1.875*max(abs(q) for q in plan['start_rad'])/plan['duration_s'],0.100001)
    def test_reaches_joint_zero_without_any_gripper_interface(self):
        arm = Arm([0.2,0.3,-0.4,0.1,-0.1,0.2]); clock=Clock()
        with patch('return_zero.time',clock): zero_one(arm,plan_for(arm.q))
        self.assertLess(np.max(np.abs(arm.q)),1e-10)
        self.assertNotIn('gripper',plan_for(arm.q))
        steps=np.diff(np.array(arm.sent),axis=0)
        self.assertLessEqual(float(np.max(np.abs(steps))),0.1*0.02+1e-6)
    def test_feedback_fault_propagates_without_retry(self):
        arm=Arm([0,0.1,-0.1,0,0,0],fail=True)
        with patch('return_zero.time',Clock()):
            with self.assertRaisesRegex(TimeoutError,'feedback lost'): zero_one(arm,plan_for(arm.q))
        self.assertEqual(arm.sent,[])
    def test_measured_position_required_for_success(self):
        arm=Arm([0,0.05,-0.05,0,0,0],follow=False)
        with patch('return_zero.time',Clock()):
            with self.assertRaisesRegex(TimeoutError,'did not settle'): zero_one(arm,plan_for(arm.q))
    def test_tracking_error_aborts(self):
        arm=Arm([0,0.5,-0.5,0,0,0],follow=False)
        with patch('return_zero.time',Clock()):
            with self.assertRaisesRegex(RuntimeError,'tracking error'): zero_one(arm,plan_for(arm.q))

    def test_recorded_start_failure_identifies_side_joint_and_degrees(self):
        with self.assertRaisesRegex(ValueError,r'right/can0.*J3 measured=\+1.1610 deg.*Gripper is excluded'):
            plan_for(np.radians([0,0,1.161,0,0,0]),'right/can0')

    def test_zero_target_failure_is_distinct_from_start_failure(self):
        import xml.etree.ElementTree as ET
        from return_zero import ROOT
        model=ET.parse(ROOT/'arms/piper_reference.urdf')
        model.getroot().find("joint[@name='joint2']/limit").set('lower','0.01')
        with patch('return_zero.ET.parse',return_value=model):
            with self.assertRaisesRegex(ValueError,r'zero target.*J2'):
                plan_for([0,.1,-.1,0,0,0],'left/can1')

if __name__=='__main__': unittest.main()
