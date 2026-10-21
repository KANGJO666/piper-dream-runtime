"""Offline contracts: all CAN access is replaced by FakeBus before driver construction."""
import math, time, unittest
from unittest.mock import patch
from types import SimpleNamespace
import can
import numpy as np
from piper_control import PiperControlDriver
from piper_readonly import PiperReadOnlyDriver

class FakeBus:
    def __init__(self):
        self.sent = []; self.closed = False; self.fail = False; self.frames = []
        self.fill()
    def fill(self):
        self.frames += [can.Message(arbitration_id=0x260+i, is_extended_id=False, data=[0]*5+[0x40,0,0]) for i in range(1,7)]
        self.frames.append(can.Message(arbitration_id=0x2A1, is_extended_id=False, data=[1,0,1,0,0,0,0,0]))
        for key, pair in ((0x2A5,(0,10000)),(0x2A6,(-10000,0)),(0x2A7,(0,0)),(0x2A8,(30000,0))):
            self.frames.append(can.Message(arbitration_id=key, is_extended_id=False, data=b''.join(v.to_bytes(4,'big',signed=True) for v in pair)))
    def recv(self, timeout):
        time.sleep(0.001)
        if not self.frames:self.fill()
        return self.frames.pop(0)
    def send(self, frame, timeout):
        if self.fail:raise can.CanOperationError('injected send failure')
        self.sent.append(frame)
    def shutdown(self):self.closed = True

class Contracts(unittest.TestCase):
    def setUp(self):
        self.bus=FakeBus()
        self.patch=patch('piper_driver.can.Bus',return_value=self.bus);self.bus_factory=self.patch.start()
        self.arm=PiperControlDriver(SimpleNamespace(can_channel='can0'),'right')
        self.arm._rx_stop.set();self.arm._rx_thread.join(timeout=0.5)
        self.q=np.array([0.1,0.4,-0.5,0.1,-0.1,0.1])
    def tearDown(self):
        self.arm.close();self.patch.stop()
    def refused(self,exception=Exception):
        with self.assertRaises(exception):self.arm.write(self.q,0.5)
        self.assertEqual(self.bus.sent,[])
    def test_construct_and_close_never_send(self):
        self.assertEqual(self.bus.sent,[]);self.arm.close();self.assertEqual(self.bus.sent,[])
    def test_out_of_range_read_identifies_side_channel_and_endpoints(self):
        self.arm._gripper_mm=81.0
        self.arm._feedback_at=time.monotonic()
        with self.assertRaisesRegex(ValueError, r'right/can0: Gripper outside calibrated range: feedback=81.000 mm, configured closed=0.000 mm, open=70.000 mm'):
            self.arm.read()
        self.assertEqual(self.bus.sent,[])

    def test_operator_70mm_opening_and_half_mm_tolerance(self):
        for mm in (70.0,70.01,70.5):
            self.arm._gripper_mm=mm
            self.arm._feedback_at=time.monotonic()
            state=self.arm.read()
            self.assertEqual(state.gripper,1.0)
        self.assertEqual(self.bus.sent,[])
        self.arm.write(self.q,state.gripper)
        jaw=next(f for f in self.bus.sent if f.arbitration_id==0x159)
        self.assertEqual(int.from_bytes(jaw.data[:4],'big',signed=True),70000)

    def test_feedback_outside_calibrated_tolerance_is_refused(self):
        for mm in (70.501,73.85,80.01,81.27,100.0,16777.474,-0.501,float('nan'),float('inf')):
            self.arm._gripper_mm=mm
            self.arm._feedback_at=time.monotonic()
            with self.assertRaises(ValueError):self.arm.read()
        self.assertEqual(self.bus.sent,[])

    def test_closed_and_mid_feedback_keep_original_mapping(self):
        for mm in (0.0,0.56,1.68,35.0,70.0):
            self.arm._gripper_mm=mm
            self.arm._feedback_at=time.monotonic()
            self.assertAlmostEqual(self.arm.read().gripper,mm/70.0)
        self.assertEqual(self.bus.sent,[])

    def test_kernel_filters_preserve_all_feedback_health_ids(self):
        filters=self.bus_factory.call_args.kwargs['can_filters']
        self.assertEqual({f['can_id'] for f in filters},{0x2A1,0x2A5,0x2A6,0x2A7,0x2A8,*range(0x261,0x267)})
        self.assertTrue(all(f['can_mask']==0x7FF and f['extended'] is False for f in filters))
    def test_receiver_catches_up_before_any_target_is_sent(self):
        import threading
        self.arm._feedback_at=time.monotonic()-1
        sent_before=[]
        def receive_fresh():
            with self.arm._state:
                sent_before.append(len(self.bus.sent))
                self.arm._feedback_at=time.monotonic()
                self.arm._state.notify_all()
        timer=threading.Timer(0.005,receive_fresh);timer.start()
        try: self.arm.write(self.q,0.5)
        finally: timer.join()
        self.assertEqual(sent_before,[0]);self.assertEqual(len(self.bus.sent),5)
    def test_official_encoder_units_and_ids(self):
        self.arm.write(self.q,0.5)
        self.assertEqual([f.arbitration_id for f in self.bus.sent],[0x151,0x155,0x156,0x157,0x159])
        self.assertEqual(bytes(self.bus.sent[0].data),bytes([1,1,100,0,0,0,0,0]))
        for i,frame in enumerate(self.bus.sent[1:4]):
            got=[int.from_bytes(frame.data[:4],'big',signed=True),int.from_bytes(frame.data[4:],'big',signed=True)]
            self.assertEqual(got,np.rint(self.q[2*i:2*i+2]*180000/math.pi).astype(int).tolist())
        self.assertEqual(int.from_bytes(self.bus.sent[-1].data[:4],'big'),35000)
        self.assertEqual(bytes(self.bus.sent[-1].data[4:]),bytes([3,232,1,0]))
        self.assertTrue(all(not f.is_extended_id and f.dlc == len(f.data) == 8 for f in self.bus.sent))
        self.bus.sent.clear();self.arm.write(self.q,0.5)
        self.assertEqual([f.arbitration_id for f in self.bus.sent],[0x155,0x156,0x157,0x159])
    def test_invalid_or_outside_limits(self):
        for q in ([0]*5,[math.nan]*6,[0,-0.1,0,0,0,0],[0,0,0.1,0,0,0]):
            with self.assertRaises(ValueError):self.arm.write(q,0.5)
        for grip in (-0.1,1.1,math.nan):
            with self.assertRaises(ValueError):self.arm.write(self.q,grip)
        self.assertEqual(self.bus.sent,[])
    def test_measured_boundary_hold_projects_inside_limits(self):
        self.arm._joints=[0,-0.004,0.002,0,0,0]
        self.arm.write(np.array(self.arm._joints),0.5)
        for frame in self.bus.sent[1:4]:
            self.assertEqual(bytes(frame.data),bytes(8))
    def test_small_outside_policy_target_far_from_measurement_is_refused(self):
        self.q=np.array([0,-0.004,0,0,0,0]);self.refused(ValueError)
    def test_large_measured_excursion_is_still_refused(self):
        self.arm._joints=[0,-0.006,0,0,0,0]
        self.q=np.array(self.arm._joints);self.refused(ValueError)
    def test_near_boundary_hold_still_requires_fresh_feedback(self):
        self.arm._joints=[0,-0.004,0,0,0,0]
        self.arm._feedback_at=time.monotonic()-1
        self.q=np.array(self.arm._joints);self.refused(TimeoutError)
    def test_reported_sub_count_j3_overflow_encodes_exact_zero(self):
        self.arm._joints=[0.0017104226669544431,0.01984439359517553,-0.0023387411976724015,0,-0.016894787159305112,0]
        target=[0.0007813395248478511,0.01863609697226876,6.889335003559867e-06,-0.0001452072879496537,-0.013971381286693193,-0.0042482967182368644]
        self.arm.write(target,0.5)
        frame=next(f for f in self.bus.sent if f.arbitration_id==0x156)
        self.assertEqual(int.from_bytes(frame.data[:4],'big',signed=True),0)
    def test_two_counts_outside_target_still_rejected_when_not_measured_hold(self):
        self.q=np.array([0,0.1,2*math.pi/180000,0,0,0]);self.refused(ValueError)
    def test_full_chunk_worst_j3_residual_is_projected_inside(self):
        self.arm._joints=[0,0.02,-0.0023387411976724015,0,0,0]
        self.arm.write([0,0.02,9.59565868427837e-06,0,0,0],0.5)
        frame=next(f for f in self.bus.sent if f.arbitration_id==0x156)
        self.assertEqual(int.from_bytes(frame.data[:4],'big',signed=True),0)
    def test_recorded_startup_ramp_from_positive_j3_projects_inward(self):
        self.arm._joints=[-0.15001104920891264,0.006108652381980154,0.001762782544514273,0.2635098104661039,-0.019320794819577228,-0.26944392992288463]
        target=[-0.150365695705423,0.006899020254058219,0.0007563807853730786,0.26299175678540554,-0.018806666992368107,-0.26854865358720853]
        self.arm.write(target,0.02414486921529175)
        frame=next(f for f in self.bus.sent if f.arbitration_id==0x156)
        self.assertEqual(int.from_bytes(frame.data[:4],'big',signed=True),0)

    def test_lower_boundary_inward_ramp_projects_inside(self):
        self.arm._joints=[0,-0.004,0,0,0,0]
        self.arm.write([0,-0.001,0,0,0,0],0.5)
        frame=next(f for f in self.bus.sent if f.arbitration_id==0x155)
        self.assertEqual(int.from_bytes(frame.data[4:],'big',signed=True),0)

    def test_outward_or_untrusted_boundary_ramp_still_refused(self):
        self.arm._joints=[0,0.1,0.002,0,0,0]
        self.q=np.array([0,0.1,0.004,0,0,0]);self.refused(ValueError)
        self.arm._joints=[0,0.1,0.006,0,0,0]
        self.q=np.array([0,0.1,0.004,0,0,0]);self.refused(ValueError)

    def test_encoded_endpoints_never_round_outside_urdf(self):
        for joint in range(6):
            for bound in self.arm._limits[joint]:
                target=self.q.copy();target[joint]=bound
                self.bus.sent.clear();self.arm.write(target,0.5)
                raw=[]
                for frame in self.bus.sent:
                    if frame.arbitration_id in (0x155,0x156,0x157):
                        raw.extend(int.from_bytes(frame.data[j:j+4],'big',signed=True) for j in (0,4))
                actual=np.array(raw)*(math.pi/180000)
                self.assertTrue(np.all(actual>=self.arm._limits[:,0]))
                self.assertTrue(np.all(actual<=self.arm._limits[:,1]))
    def test_stale_joint_feedback(self):
        self.arm._feedback_at=time.monotonic()-1;self.refused(TimeoutError)
    def test_stale_controller(self):
        self.arm._arm_status=(self.arm._arm_status[0],time.monotonic()-1);self.refused(TimeoutError)
    def test_stale_motor(self):
        self.arm._motor_status[1]=(0x40,time.monotonic()-1);self.refused(TimeoutError)
    def test_disabled_or_faulted_motor(self):
        for state in (0,0x41,0xC0):
            self.arm._motor_status[1]=(state,time.monotonic());self.refused(RuntimeError)
    def test_controller_fault_or_teaching(self):
        for status in ([1,1,1,0,0,0,0,0],[2,0,1,0,0,0,0,0],[1,0,1,1,0,0,0,0],[1,0,1,0,0,0,1,0]):
            self.arm._arm_status=(bytes(status),time.monotonic());self.refused(RuntimeError)
    def test_exited_teaching_state_allows_normal_can_control(self):
        self.arm._arm_status=(bytes([1,0,1,2,0,0,0,0]),time.monotonic())
        self.arm.write(self.q,0.5)
        self.assertEqual(len(self.bus.sent),5)
    def test_active_or_unknown_teaching_states_still_refused(self):
        for teaching in (1,3,4,5,6,7,255):
            self.arm._arm_status=(bytes([1,0,1,teaching,0,0,0,0]),time.monotonic())
            self.refused(RuntimeError)
    def test_teaching_control_mode_not_allowed_even_when_recording_stopped(self):
        self.arm._arm_status=(bytes([2,0,1,2,0,0,0,0]),time.monotonic())
        self.refused(RuntimeError)
    def test_send_failure_latches(self):
        self.bus.fail=True
        with self.assertRaises(can.CanOperationError):self.arm.write(self.q,0.5)
        self.bus.fail=False;self.refused(RuntimeError)
    def test_close_and_torque_off_never_send(self):
        self.arm.torque_off()
        self.assertTrue(self.bus.closed);self.refused(RuntimeError)
    def test_readonly_still_never_sends(self):
        arm=PiperReadOnlyDriver(SimpleNamespace(can_channel='can1'),'left')
        arm.read()
        with self.assertRaises(RuntimeError):arm.write(self.q,0.5)
        arm.close();self.assertEqual(self.bus.sent,[])

    def test_wait_for_api_does_not_age_motor_status(self):
        arm=PiperControlDriver(SimpleNamespace(can_channel='can0'),'right')
        try:
            time.sleep(0.35)
            self.assertTrue(all(time.monotonic()-stamp<0.1 for state,stamp in arm._motor_status.values()))
            arm.write(self.q,0.5)
            self.assertEqual(len(self.bus.sent),5)
        finally:arm.close()
    def test_feedback_loss_still_blocks_after_receiver_stops(self):
        time.sleep(0.17)
        self.refused(TimeoutError)
    def test_kernel_timestamp_of_old_frames_is_not_refreshed(self):
        old=FakeBus()
        def stamp_old(timeout):
            msg=FakeBus.recv(old,timeout)
            msg.timestamp=time.time()-1
            return msg
        old.recv=stamp_old
        with patch('piper_driver.can.Bus',return_value=old):
            arm=PiperReadOnlyDriver(SimpleNamespace(can_channel='can0'),'right')
            try:
                with self.assertRaises(TimeoutError):arm.read()
            finally:arm.close()
        self.assertEqual(old.sent,[])

    def test_joint_only_never_sends_gripper_or_enable_reset_frames(self):
        self.arm.write_joints(self.q)
        self.assertEqual([f.arbitration_id for f in self.bus.sent],[0x151,0x155,0x156,0x157])
        self.bus.sent.clear()
        self.arm.write_joints(self.q)
        self.assertEqual([f.arbitration_id for f in self.bus.sent],[0x155,0x156,0x157])
        self.bus.sent.clear()
        self.arm.write(self.q,0.5)
        self.assertEqual([f.arbitration_id for f in self.bus.sent],[0x155,0x156,0x157,0x159])

    def test_joint_only_excludes_invalid_gripper_position_and_calibration(self):
        self.arm._gripper_mm=81.27
        self.assertTrue(np.array_equal(self.arm.read_joints(),self.arm._joints))
        with self.assertRaises(ValueError):self.arm.read()
        self.arm.gripper_closed_mm=None
        self.arm.gripper_open_mm=None
        self.arm.write_joints(self.q)
        self.assertFalse(any(f.arbitration_id==0x159 for f in self.bus.sent))

    def test_joint_only_keeps_joint_health_and_freshness_validation(self):
        with self.assertRaises(ValueError):self.arm.write_joints([0,0,0.1,0,0,0])
        self.arm._feedback_at=time.monotonic()-1
        with self.assertRaises(TimeoutError):self.arm.write_joints(self.q)
        self.arm._feedback_at=time.monotonic()
        self.arm._motor_status[1]=(0x41,time.monotonic())
        with self.assertRaises(RuntimeError):self.arm.write_joints(self.q)
        self.assertEqual(self.bus.sent,[])

    def test_regular_write_still_requires_explicit_gripper_target(self):
        with self.assertRaises(ValueError):self.arm.write(self.q,None)
        self.assertEqual(self.bus.sent,[])

if __name__=='__main__':unittest.main(verbosity=2)
