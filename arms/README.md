# Arm files

`piper.nominal.json` points to the bundled reference URDF and uses the nominal TCP shipped with the reference model. It is suitable for preview and kinematics checks only.

Run `dreamcli tcp` on the physical gripper and use the arm file produced by that calibration for real execution. Do not treat the nominal TCP as a measured physical calibration.
