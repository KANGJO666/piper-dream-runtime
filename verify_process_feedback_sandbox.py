"""Diagnostic only: synthetic arm feedback and captured targets; real cameras/API."""
import run_piper
run_piper.environment()
import asyncio, contextlib, dataclasses, fcntl, json, logging, os, subprocess, sys, time
from pathlib import Path
from unittest.mock import patch
from can.interfaces.socketcan import SocketcanBus
from robots import arms_lock, config, run
from robots.cli import paths
import piper_driver
from test_piper_feedback import FakeSource

folder=Path('logs/diagnostics/process_feedback_sandbox').resolve()
folder.mkdir(parents=True,exist_ok=True)
(folder/'README.txt').write_text('SYNTHETIC ARM FEEDBACK. All targets captured. No physical CAN is opened. Real cameras/API/IK/recording workload. NOT robot demonstrations or training data. This does not validate real arm calibration or task completion.\n')
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s: %(message)s')
metrics={'pid':os.getpid(),'physical_can_send_attempts':0,'captured_frames':{},'receiver_pids':[],
         'synthetic_feedback':True,'task_success':'unverified','max_feedback_age_ms':0}
buses=[]
original_spawn=subprocess.Popen
original_refresh=piper_driver.PiperDriver._refresh_feedback

def make_bus(**kwargs):
    bus=FakeSource();buses.append(bus);return bus

def spawn(args,*a,**kw):
    if isinstance(args,list) and len(args)>2 and str(args[2]).endswith('/piper_feedback.py'):
        code=('from test_piper_feedback import FakeSource; from piper_feedback import run_receiver; '
              f'run_receiver({args[-3]!r},{args[-2]!r},{int(args[-1])},bus_factory=lambda **kwargs:FakeSource())')
        process=original_spawn([sys.executable,'-B','-c',code],*a,**kw)
        metrics['receiver_pids'].append(process.pid)
        return process
    return original_spawn(args,*a,**kw)

def refresh(arm):
    result=original_refresh(arm)
    if arm._feedback_at is not None:
        metrics['max_feedback_age_ms']=max(metrics['max_feedback_age_ms'],(time.monotonic()-arm._feedback_at)*1000)
    return result

def forbid_send(*a,**kw):
    metrics['physical_can_send_attempts']+=1
    raise AssertionError('Physical CAN is forbidden in sandbox validation')

app=run.with_task(config.load('my_robot_dryrun_nominal.yaml'),
                  'Use both arms cooperatively to stack one bowl over the book.')
app=dataclasses.replace(app,custom=dataclasses.replace(app.custom,
    driver=str(run_piper.ROOT/'piper_control.py')+':PiperControlDriver'),
    max_ee_speed_m_s=.10,inference_log_path=str(folder/'inference.jsonl'),
    trace_path=str(folder/'trace.csv'),
    rollouts=dataclasses.replace(app.rollouts,enabled=True,dir=str(folder/'recording')))
started=time.monotonic()
print(f'SANDBOX_START pid={os.getpid()} synthetic_feedback=true physical_CAN=forbidden',flush=True)
try:
    with contextlib.ExitStack() as stack:
        existing=Path.home()/'.dreamlabs'/arms_lock.LOCK_NAME
        if existing.exists():
            lock=stack.enter_context(existing.open('r'));fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stack.enter_context(arms_lock.acquire(paths().lock,arms_lock.ROBOT_RUN))
        stack.enter_context(patch.object(piper_driver.can,'Bus',side_effect=make_bus))
        stack.enter_context(patch.object(piper_driver,'SocketcanBus',FakeSource))
        stack.enter_context(patch.object(piper_driver.subprocess,'Popen',side_effect=spawn))
        stack.enter_context(patch.object(piper_driver.PiperDriver,'_refresh_feedback',refresh))
        stack.enter_context(patch.object(SocketcanBus,'send',forbid_send))
        asyncio.run(run_piper.session(app,True,max_seconds=60))
    metrics['status']='completed_60_second_sandbox'
except BaseException as exc:
    metrics['status']='failed';metrics['error']=repr(exc)
    raise
finally:
    for bus in buses:
        for frame in bus.sent:
            key=hex(frame.arbitration_id)
            metrics['captured_frames'][key]=metrics['captured_frames'].get(key,0)+1
    metrics['duration_s']=time.monotonic()-started
    (folder/'result.json').write_text(json.dumps(metrics,indent=2))
    print('SANDBOX_RESULT '+json.dumps(metrics),flush=True)
