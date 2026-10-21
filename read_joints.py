"""Print live Piper joint feedback without enabling or commanding motors."""
import argparse
import json
import math
import xml.etree.ElementTree as ET

from run_piper import ROOT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('left', 'right', 'both'), default='both')
    args = parser.parse_args()
    environment()

    from robots import config
    from piper_readonly import PiperReadOnlyDriver

    model = ET.parse(ROOT / 'arms/piper_reference.urdf').getroot()
    joints = {
        joint.get('name'): joint.find('limit')
        for joint in model.findall('joint')
        if joint.get('type') == 'revolute'
    }
    limits = [
        (float(joints[f'joint{i}'].get('lower')), float(joints[f'joint{i}'].get('upper')))
        for i in range(1, 7)
    ]
    app = config.load(ROOT / 'my_robot_dryrun_nominal.yaml')
    selected = ('left', 'right') if args.arm == 'both' else (args.arm,)
    samples = {}
    for side in selected:
        arm = PiperReadOnlyDriver(app.follower_l if side == 'left' else app.follower_r, side)
        try:
            reading = arm.read()
            measured = reading.positions_rad.tolist()
            samples[side] = {
                'channel': arm.channel,
                'joints_deg': [math.degrees(value) for value in measured],
                'limits_deg': [[math.degrees(lo), math.degrees(hi)] for lo, hi in limits],
                'in_range': [lo <= value <= hi for value, (lo, hi) in zip(measured, limits)],
                'gripper': reading.gripper,
            }
        finally:
            arm.close()
    print(json.dumps(samples, indent=2))
    print('READ_ONLY_COMPLETE: no motor command sent')


if __name__ == '__main__':
    main()
