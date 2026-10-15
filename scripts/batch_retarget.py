"""Resumable video-to-robot batch orchestration. Heavy dependencies run in subprocesses."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import shutil
import tempfile

EVENTS = ('closure_start', 'grasp_established', 'lift_off', 'placement_contact',
          'opening_start', 'release_complete')
SCRIPTS = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Invalid demo ID: {value}')
    return value


def input_hashes(root, item):
    return {**{key: digest(root / item[key]) for key in ('video', 'calibration', 'config')},
            'tracker': digest(SCRIPTS/'process_pilot.py')}


def tracking_current(root, item):
    try:
        if item.get('tracking_inputs') != input_hashes(root, item):
            return False
        folder = root / 'data/processed' / item['id']
        return all(digest(folder / name) == value for name, value in item['tracking_outputs'].items())
    except (OSError, KeyError):
        return False


def validate_events(path, trajectory_path, report_path):
    payload = read(path)
    events = payload['events']
    times = []
    frames = []
    report = read(report_path)
    for name in EVENTS:
        event = events[name]
        t, frame = event['time_s'], event['frame']
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError(f'{name}: enter a finite nonnegative time_s')
        if type(frame) is not int or not 0 <= frame < report['decoded_frames']:
            raise ValueError(f'{name}: frame outside decoded video')
        if report['timestamps'] == 'frame_index_div_fps' and abs(t - frame/report['fps']) > 1/report['fps']:
            raise ValueError(f'{name}: frame/time mismatch')
        times.append(t); frames.append(frame)
    if any(b < a for a, b in zip(times, times[1:])) or any(b < a for a, b in zip(frames, frames[1:])):
        raise ValueError('Events must follow chronological order')
    if not 0 < times[1] - times[0] < 5 or not times[1] < times[3]:
        raise ValueError('Invalid closure or transfer duration')
    with Path(trajectory_path).open() as f:
        samples = list(csv.DictReader(f))
    ts = [float(row['seconds']) for row in samples]
    if len(ts) < 2 or any(not math.isfinite(t) for t in ts) or any(b <= a for a,b in zip(ts, ts[1:])):
        raise ValueError('Invalid tracked timestamps')
    if not ts[0] <= times[1] < times[3] <= ts[-1]:
        raise ValueError('Grasp-to-placement interval is outside tracked motion')
    return payload


def source_key(root, item):
    folder = root / 'data/processed' / item['id']
    return {'tracking_inputs': input_hashes(root, item),
            'trajectory': digest(folder/'marker_smoothed.csv'),
            'events': digest(folder/'human_events.json'),
            'controller': digest(SCRIPTS/'retarget_episode.py')}


def execute(command, log, input_text=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Running:', ' '.join(map(str, command)), flush=True)
    print('Log:', log, flush=True)
    with log.open('w') as f:
        return subprocess.run(list(map(str, command)), input=input_text, text=True,
                              stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    register = sub.add_parser('register')
    register.add_argument('--videos', nargs='+', required=True, help='Project-relative video paths; quote paths containing spaces')
    register.add_argument('--calibration', required=True)
    for name in ('track', 'retarget'):
        p = sub.add_parser(name)
        p.add_argument('--demo', nargs='+')
        p.add_argument('--retry', action='store_true', help='Retry unchanged failed jobs')
        if name == 'retarget':
            p.add_argument('--work-root', type=Path, default=Path('/content/humanoid_retarget_runs') if Path('/content').is_dir() else Path(tempfile.gettempdir())/'humanoid_retarget_runs')
    p = sub.add_parser('approve-events')
    p.add_argument('--demo', required=True)
    p = sub.add_parser('review')
    p.add_argument('--demo', required=True)
    p.add_argument('--decision', choices=['approve', 'reject'], required=True)
    p.add_argument('--note', required=True, help='Describe what you observed in replay.mp4')
    sub.add_parser('status')
    sub.add_parser('export-list')
    args = parser.parse_args()
    root = args.project.resolve()
    manifest = root / 'configs/retarget_batch.json'
    data = read(manifest) if manifest.exists() else {'version': 1, 'demos': {}}
    demos = data['demos']
    if args.command == 'register':
        calibration = (root / args.calibration).resolve()
        calibration.relative_to(root)
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
        for video in args.videos:
            path = (root / video).resolve()
            relative = str(path.relative_to(root))
            demo = safe_id(path.stem)
            if path.suffix.lower() not in ('.mov', '.mp4', '.m4v') or not path.is_file():
                raise ValueError(f'Not a video file: {path}')
            if demo in demos:
                if demos[demo]['video'] != relative or demos[demo]['calibration'] != str(calibration.relative_to(root)):
                    raise ValueError(f'ID already registered with different inputs: {demo}')
                continue
            if (root/'data/processed'/demo).exists() or (root/'data/robot'/demo).exists():
                raise ValueError(f'Existing unregistered demo data; use a new filename: {demo}')
            config = root / 'configs/batch' / (demo + '.json')
            if config.exists():
                raise FileExistsError(config)
            write(config, dict(video=relative, calibration=str(calibration.relative_to(root)),
                               output_hz=20, max_reprojection_px=2.0))
            demos[demo] = dict(id=demo, video=relative, calibration=str(calibration.relative_to(root)),
                               config=str(config.relative_to(root)), status='registered')
            write(manifest, data)
    elif args.command == 'status':
        for demo, item in demos.items():
            print(demo, item['status'], 'tracking_current='+str(tracking_current(root,item)), item.get('run',''))
    elif args.command == 'approve-events':
        item = demos[args.demo]
        if not tracking_current(root,item):
            raise ValueError('Run tracking first; its inputs or outputs are missing/stale')
        folder = root/'data/processed'/args.demo
        validate_events(folder/'human_events.json', folder/'marker_smoothed.csv', folder/'tracking_report.json')
        item['approved_events_hash'] = digest(folder/'human_events.json')
        item['status'] = 'ready_for_retargeting'
        write(manifest,data)
    elif args.command in ('track','retarget'):
        selected = args.demo or list(demos)
        for demo in selected:
            item = demos[demo]
            folder = root/'data/processed'/demo
            try:
                if args.command == 'track':
                    if tracking_current(root,item):
                        print(demo, 'tracking already complete'); continue
                    key = input_hashes(root,item)
                    if item.get('tracking_failed_inputs') == key and not args.retry:
                        print(demo, 'previous tracking failure; use --retry'); continue
                    log = root/'outputs/batch_logs'/f'{demo}_track_{stamp()}.log'
                    code = execute([sys.executable, SCRIPTS/'process_pilot.py', '--project',root,
                                    '--config',root/item['config'],'--output',folder], log)
                    if code:
                        item['tracking_failed_inputs'] = key
                        raise RuntimeError(f'Tracking exited {code}; see {log}')
                    item['tracking_inputs'] = key
                    item['tracking_outputs'] = {name:digest(folder/name) for name in
                        ('marker_raw.csv','marker_smoothed.csv','tracking_report.json')}
                    item.pop('approved_events_hash',None)
                    if not (folder/'human_events.json').exists():
                        write(folder/'human_events.json', {'video':item['video'], 'method':'manual visual annotation',
                              'frame_index_convention':'zero-based decoded frame index',
                              'events':{name:{'frame':None,'time_s':None} for name in EVENTS}})
                    item['status'] = 'needs_event_review'
                else:
                    if not tracking_current(root,item):
                        print(demo, 'waiting for current tracking'); continue
                    validate_events(folder/'human_events.json',folder/'marker_smoothed.csv',folder/'tracking_report.json')
                    if item.get('approved_events_hash') != digest(folder/'human_events.json'):
                        print(demo, 'waiting for approve-events'); continue
                    key = source_key(root,item)
                    if item.get('retarget_inputs') == key and item.get('run'):
                        old = root/item['run']/'report.json'
                        if old.exists() and read(old)['status'] == 'passed_automatic_checks':
                            print(demo, 'rollout already complete'); continue
                        if not args.retry:
                            print(demo,'previous failure; use --retry'); continue
                    run = root/'data/robot'/demo/stamp()
                    item.update(run=str(run.relative_to(root)), retarget_inputs=key, status='retargeting')
                    write(manifest,data)
                    local_run = args.work_root.resolve()/demo/run.name
                    local_log = args.work_root.resolve()/'logs'/f'{demo}_{run.name}.log'
                    item.update(local_run=str(local_run), local_log=str(local_log))
                    write(manifest,data)
                    print('Recording locally:', local_run, flush=True)
                    code = execute([sys.executable,SCRIPTS/'retarget_episode.py','--project',root,
                                    '--demo',demo,'--output',local_run],local_log,
                                   input_text='n\n')
                    # Copy only after the subprocess has closed HDF5/video writers.
                    # Keep local outputs available if copying fails or for further inspection.
                    print('Copying completed attempt to Drive:', run, flush=True)
                    if local_run.exists():
                        shutil.copytree(local_run, run)
                    drive_log = root/'outputs/batch_logs'/local_log.name
                    drive_log.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_log, drive_log)
                    if code or not (run/'report.json').exists() or read(run/'report.json')['status'] != 'passed_automatic_checks':
                        raise RuntimeError(f'Rollout failed; inspect log and {run}')
                    item['status'] = 'needs_visual_review'
                item.pop('error',None)
            except Exception as exc:
                item.update(status=args.command+'_failed',error=str(exc))
                print(demo,str(exc),flush=True)
            write(manifest,data)
    elif args.command == 'review':
        item = demos[args.demo]
        if not tracking_current(root,item) or item.get('retarget_inputs') != source_key(root,item):
            raise ValueError('Rollout inputs changed; rerun before review')
        report_path = root/item['run']/'report.json'
        report = read(report_path)
        approved = args.decision == 'approve'
        if approved and report['status'] != 'passed_automatic_checks':
            raise ValueError('Cannot approve a failed rollout')
        report.update(training_eligible=approved, visual_review_required=False,
                      visual_review={'decision':args.decision,'note':args.note,'reviewer':'user','time':stamp()})
        write(report_path,report)
        item['status'] = 'approved' if approved else 'rejected'
        write(manifest,data)
    elif args.command == 'export-list':
        episodes = []
        for demo,item in demos.items():
            if item['status'] != 'approved' or not tracking_current(root,item): continue
            if item.get('retarget_inputs') != source_key(root,item): continue
            run = root/item['run']
            report = read(run/'report.json')
            if report.get('training_eligible') and report['status'] == 'passed_automatic_checks' and (run/'episode.h5').exists():
                episodes.append({'demo':demo,'episode':str((run/'episode.h5').relative_to(root)),
                                 'report':str((run/'report.json').relative_to(root)), 'source':item['retarget_inputs']})
        path = root/'data/robot/approved_batch.json'
        write(path,{'episodes':episodes})
        print(f'{len(episodes)} approved episodes: {path}')


if __name__ == '__main__':
    main()
