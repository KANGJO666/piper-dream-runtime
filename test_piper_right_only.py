"""No hardware is opened by these dispatch checks."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from piper_right_only import RightOnlyDriver, CaptureDriver, validate_target

class DispatchTest(unittest.TestCase):
    def test_left_is_always_capture(self):
        with patch.dict(os.environ, {'PIPER_RIGHT_EXECUTE': '1'}), patch('piper_right_only.CaptureDriver') as capture:
            RightOnlyDriver(SimpleNamespace(can_channel='can1'), 'left')
            capture.assert_called_once()

    def test_right_preview_is_capture(self):
        with patch.dict(os.environ, {'PIPER_RIGHT_EXECUTE': '0'}), patch('piper_right_only.CaptureDriver') as capture:
            RightOnlyDriver(SimpleNamespace(can_channel='can0'), 'right')
            capture.assert_called_once()

    def test_bindings_fail_closed(self):
        for side, channel in [('left','can0'), ('right','can1'), ('other','can0')]:
            with self.assertRaises(ValueError):
                RightOnlyDriver(SimpleNamespace(can_channel=channel), side)

    def test_capture_has_no_transmit(self):
        arm = object.__new__(CaptureDriver)
        arm.side, arm.count, arm.output, arm.bus = 'left', 0, Mock(), Mock()
        arm.write(np.zeros(6), .5)
        arm.bus.send.assert_not_called()
        self.assertEqual(arm.count, 1)

    def test_invalid_targets(self):
        for q, grip in [(np.zeros(7), 0), ([float('nan')]*6, .5), (np.zeros(6), 1.1)]:
            with self.assertRaises(ValueError): validate_target(q, grip)

if __name__ == '__main__': unittest.main()
