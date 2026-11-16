#!/usr/bin/env python3
"""Read-only comparison monitor. All operational identities come from a private config.

Print one JSON snapshot; never update a ledger, signal a process, acquire a lease,
change a schedule, or write a checkpoint. SSH/transport failures belong to the
caller and must not be interpreted as a healthy snapshot.
"""
import argparse
import csv
import datetime as dt
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = 'tea-handnet-comparison-monitor-v1'
ACTIVE = {'reserved', 'conversion', 'validating_bank', 'resource_wait_or_guarded_training',
          'validating_model', 'validation'}
COMMAND_PHASES = {'conversion', 'resource_wait_or_guarded_training', 'validation'}
COMPLETE = 'complete_pending_independent_native_parity_backup_and_report'


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('nonfinite JSON constant: ' + value))), hashlib.sha256(raw).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite numerical output')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def duration_seconds(value):
    parts = re.findall(r'(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)', value)
    require(parts and not re.sub(r'(\d+(?:\.\d+)?)\s*(us|ms|min|s|h)', '', value).strip(),
            'unrecognized bounded service stop timeout')
    return sum(float(n) * {'us': 1e-6, 'ms': .001, 's': 1, 'min': 60, 'h': 3600}[unit]
               for n, unit in parts)


def validate_config(config):
    require(config.get('schema') == 'tea-handnet-comparison-runtime-v1', 'runtime config schema')
    for name in ('project_root', 'training_repo', 'proposal_dir', 'worker_path', 'python_executable'):
        value = config.get(name)
        require(isinstance(value, str) and Path(value).is_absolute() and str(Path(value)) == value,
                'absolute normalized config path required: ' + name)
    for name in ('hostname', 'username', 'service_name'):
        require(isinstance(config.get(name), str) and bool(config[name]), 'missing identity: ' + name)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:-]*\.service', config['service_name']),
            'invalid service name')
    root = Path(config['project_root']).resolve()
    for name in ('training_repo', 'proposal_dir'):
        require(Path(config[name]).resolve().is_relative_to(root), name + ' outside project root')
    m = config.get('monitor', {})
    require(m.get('schema') == SCHEMA, 'monitor config schema')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', m.get('run_id', '')), 'invalid run id')
    for key in ('worker_sha256', 'original_budget_sha256', 'runtime_inventory_sha256',
                'binary_sha256', 'coords_sha256', 'validation_manifest_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', m.get(key, '')), 'invalid expected digest: ' + key)
    for key in ('original_used_seconds', 'aggregate_ceiling_seconds', 'per_run_max_seconds',
                'service_min_stop_seconds', 'status_stale_seconds', 'progress_stale_seconds',
                'startup_grace_seconds'):
        require(number(m.get(key)) and m[key] >= 0, 'invalid numeric monitor field: ' + key)
    for key in ('original_settled_runs', 'original_final_trees', 'authorized_total_final_trees',
                'max_new_runs', 'samples', 'features', 'classes', 'decisions', 'validation_frames',
                'min_pixels', 'service_memory_max_bytes', 'disk_reserve_bytes'):
        require(type(m.get(key)) is int and m[key] > 0, 'invalid integer monitor field: ' + key)
    require(m['original_final_trees'] + m['max_new_runs'] == m['authorized_total_final_trees'],
            'authorized slot reconciliation mismatch')
    require(m['original_used_seconds'] < m['aggregate_ceiling_seconds'], 'no residual budget')
    require(type(m.get('launch_expected')) is bool, 'launch_expected must be explicit')
    require(isinstance(m.get('class_names'), list) and len(m['class_names']) == m['classes'], 'class map')
    require(m.get('foreground') == 'none' and m.get('fingertip_classes') == [0, 1, 2, 3, 4],
            'unsupported evaluation domain')
    require(m.get('validation_depths') == [12, 16, 20] and
            m.get('probability_thresholds') == [.25, .5], 'frozen six-setting grid required')
    return config


def settlements(document):
    seen = set()
    total = 0.
    for item in document['runs']:
        digest, seconds = item['spec_sha256'], item['seconds']
        require(re.fullmatch('[0-9a-f]{64}', digest) and digest not in seen,
                'duplicate or malformed settlement identity')
        require(number(seconds) and seconds >= 0, 'invalid settled seconds')
        seen.add(digest)
        total += seconds
    require(number(document['used_active_training_seconds']) and
            math.isclose(total, document['used_active_training_seconds'], rel_tol=0, abs_tol=1e-6),
            'settlement sum differs from ledger total')
    return total, seen


def accounting(original, residual, guards, expected, original_sha):
    """Original + residual settlements + each unsettled live spec, exactly once."""
    require(original_sha == expected['original_budget_sha256'], 'original budget bytes changed')
    old, old_specs = settlements(original)
    require(old == expected['original_used_seconds'] and
            len(old_specs) == expected['original_settled_runs'], 'original charge anchor mismatch')
    require(original['max_active_training_seconds'] == expected['aggregate_ceiling_seconds'] and
            original['max_final_trees'] == expected['original_final_trees'], 'original budget caps changed')
    new, new_specs = 0., set()
    if residual is not None:
        require(residual['original_budget_sha256'] == original_sha and
                residual['original_used_seconds'] == old, 'residual anchor mismatch')
        require(residual['max_active_training_seconds'] == expected['aggregate_ceiling_seconds'] - old,
                'residual ceiling changed or original time debited twice')
        require(residual.get('max_new_final_trees') == expected['max_new_runs'], 'residual slot cap')
        new, new_specs = settlements(residual)
        require(not (old_specs & new_specs) and len(new_specs) <= expected['max_new_runs'],
                'old charge copied into residual or excess new attempts')
    live = {}
    for guard in guards:
        require(not guard.get('accounting_failed'), 'guard accounting failure requires reconciliation')
        spec = guard.get('spec_sha256')
        elapsed = guard.get('active_training_seconds', 0)
        require(number(elapsed) and elapsed >= 0, 'invalid guard active time')
        if spec in old_specs or spec in new_specs:
            continue
        if guard.get('phase') == 'training':
            require(residual is not None and re.fullmatch('[0-9a-f]{64}', spec or ''),
                    'live guard has no valid residual ledger/spec')
            live[spec] = max(live.get(spec, 0.), elapsed)
        elif elapsed > 0:
            raise ValueError('unsettled terminal guard requires reconciliation')
    require(len(new_specs | live.keys()) <= expected['max_new_runs'], 'extra live/residual attempt')
    total = old + new + sum(live.values())
    require(total <= expected['aggregate_ceiling_seconds'] + 1e-6, 'aggregate training cap exceeded')
    return {'original_settled_seconds': old, 'residual_settled_seconds': new,
            'live_unsettled_seconds': sum(live.values()), 'aggregate_seconds': total,
            'remaining_seconds': expected['aggregate_ceiling_seconds'] - total,
            'old_settlements': len(old_specs), 'new_settlements': len(new_specs),
            'distinct_unsettled_specs': len(live)}


def process_snapshot(pid, proc=Path('/proc')):
    p = proc / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': fields[19], 'cpu_ticks': int(fields[11]) + int(fields[12]),
                'argv': (p / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'),
                'cgroup': (p / 'cgroup').read_text()}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': int(pid), 'exited': True}


def discover_processes(root, worker, proc=Path('/proc'), exclude_pid=None):
    found = []
    prefix = str(Path(root)) + '/'
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == exclude_pid:
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            if any(arg == str(worker) or arg.startswith(prefix) for arg in args):
                found.append(process_snapshot(p.name, proc))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


class LinuxProbe:
    def command(self, argv):
        result = subprocess.run(argv, text=True, capture_output=True, timeout=20)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}

    def identity(self):
        return {'hostname': socket.gethostname(), 'username': getpass.getuser(),
                'platform': sys.platform}

    def service(self, unit):
        keys = ['MainPID', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus',
                'MemoryMax', 'MemoryCurrent', 'KillMode', 'TimeoutStopUSec', 'ControlGroup',
                'NRestarts', 'ExecMainStartTimestamp']
        result = self.command(['systemctl', '--user', 'show', unit] +
                              [arg for key in keys for arg in ('-p', key)])
        require(result['returncode'] == 0, 'service query failed: ' + result['stderr'])
        return dict(line.split('=', 1) for line in result['stdout'].splitlines() if '=' in line)

    def processes(self, root, worker):
        return discover_processes(root, worker, exclude_pid=os.getpid())

    def process(self, pid):
        return process_snapshot(pid)

    def cgroup(self, group):
        # Never fall back to root-cgroup counters when a completed service is removed.
        if not group or group == '/':
            return {'status': 'not_applicable_no_service_cgroup'}
        base = Path('/sys/fs/cgroup').resolve()
        p = (base / group.lstrip('/')).resolve()
        require(p.is_relative_to(base) and p != base, 'invalid service cgroup path')
        if not p.is_dir():
            return {'status': 'removed'}
        result = {'status': 'present', 'path': str(p)}
        for name in ('memory.events', 'memory.pressure', 'memory.current', 'memory.max'):
            if (p / name).is_file():
                result[name] = (p / name).read_text()
        result['events'] = {k: int(v) for k, v in
                            (line.split() for line in result.get('memory.events', '').splitlines())}
        return result

    def resources(self, root):
        disk = shutil.disk_usage(root)
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                memory[key + '_bytes'] = int(value.split()[0]) * 1024
        gpu = self.command(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total',
                            '--format=csv,noheader,nounits'])
        require(gpu['returncode'] == 0, 'GPU telemetry failed: ' + gpu['stderr'])
        rows = []
        for row in csv.reader(gpu['stdout'].splitlines()):
            require(len(row) == 6, 'malformed GPU telemetry')
            rows.append(dict(zip(('index', 'name', 'utilization_percent', 'memory_used_mib',
                                  'memory_free_mib', 'memory_total_mib'),
                                 [int(row[0]), row[1].strip()] + [float(x) for x in row[2:]])))
        require(rows, 'empty GPU telemetry')
        return {'disk': {'free_bytes': disk.free, 'total_bytes': disk.total},
                'ram': memory, 'gpu': rows}


def log_snapshot(path, now):
    p = Path(path)
    if not p.is_file():
        return {'path': str(p), 'exists': False}
    with p.open('rb') as stream:
        stream.seek(max(0, p.stat().st_size - 65536))
        tail = stream.read().decode(errors='replace')
    conversions = re.findall(r'processed\s+(\d+)/(\d+)\s+frames,\s*samples=(\d+)', tail)
    evaluations = re.findall(r'evaluated\s*(\d+)/(\d+)\s+frames', tail)
    return {'path': str(p), 'exists': True, 'bytes': p.stat().st_size,
            'modified_unix': p.stat().st_mtime, 'age_seconds': max(0, now - p.stat().st_mtime),
            'conversion': list(map(int, conversions[-1])) if conversions else None,
            'evaluation': list(map(int, evaluations[-1])) if evaluations else None,
            'error_lines': [line for line in tail.splitlines() if re.search(
                r'Traceback|\b\w*Error\b|\bException\b|CUDA error|\bKilled\b|\b(?:nan|inf)\b', line,
                re.IGNORECASE)][-20:]}


def frame_id(frame):
    require(type(frame.get('gidx')) is int and frame.get('frame_id') == 'handnet:gidx:' + str(frame['gidx']),
            'raw global frame identity required')
    require(isinstance(frame.get('name'), str) and isinstance(frame.get('session'), str) and
            type(frame.get('idx')) is int, 'incomplete frame identity')
    return tuple(frame[k] for k in ('frame_id', 'name', 'gidx', 'session', 'idx'))


def evaluation_summary(result, expected_frames, m):
    finite(result)
    require(result.get('schema') == 'handnet-tea-evaluation-v1' and result.get('split') == 'validation',
            'wrong evaluation schema/split')
    require([frame_id(f) for f in result['frames']] == [frame_id(f) for f in expected_frames],
            'evaluation raw identities/order differ from frozen cohort')
    require(result['classes'] == m['class_names'] and result['foreground'] == m['foreground'] and
            result['pixel_limit_per_frame'] == 0, 'evaluation class/domain mismatch')
    local = result['localization']
    require(local['minimum_pixels'] == m['min_pixels'] and
            result['max_splits'] in m['validation_depths'] and
            local['probability_threshold'] in m['probability_thresholds'], 'evaluation grid mismatch')
    count = len(expected_frames)
    names = ('present', 'detected', 'absent', 'false_positive_on_absent',
             'sensor_pos_present', 'sensor_pos_detected')
    values = {}
    for key in names:
        a = local[key]
        require(isinstance(a, list) and len(a) == 5 and all(type(x) is int and 0 <= x <= count for x in a),
                'invalid integer localization counts: ' + key)
        values[key] = a
    predicted = [0] * 5
    for frame in result['frames']:
        fingers = frame['fingertips']
        require(isinstance(fingers, list) and len(fingers) == 5, 'five fingertip slots required')
        for i, center in enumerate(fingers):
            if center is None:
                continue
            require(type(center.get('pixels')) is int and center['pixels'] >= m['min_pixels'], 'bad component support')
            for name, size in (('xy', 2), ('xyz', 3)):
                require(isinstance(center.get(name), list) and len(center[name]) == size and
                        all(number(x) for x in center[name]), 'invalid center geometry')
            require(center['xyz'][2] > 0, 'nonpositive predicted depth')
            predicted[i] += 1
    for i in range(5):
        p, d, a, f, s, t = [values[k][i] for k in names]
        require(p + a == count and t <= s <= p and t <= d <= p and f <= a and predicted[i] == d + f,
                'inconsistent localization cardinalities')
    hit_arrays = {}
    for field, thresholds, denominator in (
            ('sensor_pos_recall_at_mm', ('10', '15', '20'), values['sensor_pos_detected']),
            ('label_centroid_recall_at_px', ('5', '10', '15'), values['detected'])):
        previous = [0] * 5
        for threshold in thresholds:
            a = local[field][threshold]['hits']
            require(isinstance(a, list) and len(a) == 5 and all(type(x) is int and 0 <= x <= count for x in a),
                    'invalid localization hit counts')
            require(all(x <= y <= z for x, y, z in zip(previous, a, denominator)),
                    'nonmonotonic/impossible localization hits')
            previous = a
            hit_arrays[(field, threshold)] = a
    scores = []
    for h, p, d, f in zip(hit_arrays[('sensor_pos_recall_at_mm', '15')],
                           values['sensor_pos_present'], values['sensor_pos_detected'],
                           values['false_positive_on_absent']):
        scores.append(2 * h / (p + d + f) if p + d + f else 0.)
    return {'splits': result['max_splits'], 'threshold': local['probability_threshold'],
            'frames': count, 'macro_f1_at_15mm': sum(scores) / 5,
            'independent_raw_ground_truth_metric_audit': 'not performed by this collector'}


def checkpoint_summary(bank, m, model_class=None):
    """Only immutable, finished depth files are loaded; never a temporary snapshot."""
    candidates = []
    for p in Path(bank).glob('Tree_*.depth*.tree'):
        match = re.fullmatch(r'Tree_0000\.depth(\d{2})\.tree', p.name)
        require(match is not None, 'unexpected checkpoint name')
        candidates.append((int(match[1]), p))
    candidates.sort()
    if not candidates:
        return {'count': 0, 'latest': None}
    require([d for d, _ in candidates] == list(range(candidates[-1][0] + 1)), 'checkpoint depth gap')
    depth, path = candidates[-1]
    require(depth < m['decisions'], 'checkpoint exceeds authorized depth')
    require(sha(Path(bank) / 'Coords.co') == m['coords_sha256'], 'checkpoint coordinates changed')
    if model_class is None:
        from tea_model import TeaTree
        model_class = TeaTree
    import numpy as np
    tree = model_class(path, Path(bank) / 'Coords.co')
    require(tree.splits == depth + 1 and tree.classes == m['classes'] and len(tree.coords) == m['features'],
            'checkpoint header/feature/class mismatch')
    for start in range(0, tree.hist_count, 65536):
        heaps = np.arange(start, min(tree.hist_count, start + 65536))
        require(np.all(tree.histograms[heaps].sum(axis=1) > 0), 'empty repaired posterior histogram')
        values = tree.reached_probabilities(heaps)
        require(np.isfinite(values).all() and np.allclose(values.sum(axis=1), 1, atol=2e-7, rtol=0),
                'nonfinite/unnormalized checkpoint posterior')
    return {'count': len(candidates), 'latest': {'name': path.name, 'sha256': sha(path),
            'bytes': path.stat().st_size, 'modified_unix': path.stat().st_mtime,
            'decisions': tree.splits, 'classes': tree.classes,
            'finite_normalized_probabilities': True}}


def completed_artifact_summary(control, bank, retained, status, m):
    validation = read_json(control / 'MODEL_VALIDATION.json')[0]
    require(validation.get('external_predictions_match') is True, 'external saved-label validation missing')
    checkpoints = validation['checkpoints']
    require(len(checkpoints) == m['decisions'] and
            [p['name'] for p in checkpoints] == [f'Tree_0000.depth{d:02d}.tree' for d in range(m['decisions'])],
            'completed checkpoint inventory incomplete')
    for checkpoint in checkpoints:
        require(sha(bank / checkpoint['name']) == checkpoint['sha256'], 'completed checkpoint changed')
    require(sha(bank / 'Tree_0000.tree') == validation['model_sha256'] == checkpoints[-1]['sha256'],
            'completed bank model changed')
    require(sha(bank / 'Coords.co') == validation['coords_sha256'] == m['coords_sha256'],
            'completed bank coordinates changed')
    copies = read_json(retained / 'RETAINED_FILES.json')[0]
    require(set(copies) == {'model.tree', 'Coords.co', 'training.json', 'frames.json'}, 'retained inventory mismatch')
    for name, digest in copies.items():
        require(sha(retained / name) == digest, 'retained model/metadata copy changed: ' + name)
    require(copies['model.tree'] == validation['model_sha256'] and copies['Coords.co'] == m['coords_sha256'],
            'retained model/coordinate validation mismatch')
    deployment = read_json(control / 'DEPLOYMENT_VALIDATION.json')[0]
    manifest = Path(deployment['path'])
    require(manifest.resolve() == (retained / 'selected/forest.json').resolve() and
            sha(manifest) == deployment['sha256'], 'selected deployment manifest changed')
    require(status['deployment']['sha256'] == deployment['sha256'] and
            status['deployment']['path'] == deployment['path'], 'status/deployment proof mismatch')
    selected = read_json(control / 'VALIDATION_SELECTION.json')[0]['selected']
    doc = read_json(manifest)[0]
    post = doc['postprocessing']
    require(post['foreground'] == m['foreground'] and post['min_pixels'] == m['min_pixels'] and
            post['fingertip_classes'] == m['fingertip_classes'] and
            post['probability_threshold'] == selected['threshold'], 'selected deployment operating point changed')
    for tree in doc['trees']:
        for key, digest in (('model', 'model_sha256'), ('coords', 'coords_sha256')):
            path = (manifest.parent / tree[key]).resolve()
            require(path.is_relative_to(manifest.parent.resolve()) and sha(path) == tree[digest],
                    'selected deployment asset changed')
    return {'all_checkpoint_hashes_match': True, 'retained_file_hashes_match': True,
            'selected_deployment_manifest_sha256': deployment['sha256'],
            'selected_deployment_assets_match': True,
            'independent_native_parity_and_raw_metric_audit': 'separate required gates'}


def completed_accounting(control, guard, m):
    require(guard and guard.get('phase') == 'training_complete_pending_validation' and
            guard.get('exit_code') == 0 and not guard.get('accounting_failed'),
            'completed run lacks successful terminal guard')
    spec, digest = read_json(control / 'spec.json')
    require(guard['spec_sha256'] == digest and Path(spec['budget_file']) == control / 'RESIDUAL_BUDGET.json',
            'completed guard/spec/residual binding mismatch')
    residual = read_json(control / 'RESIDUAL_BUDGET.json')[0]
    require(len(residual['runs']) == 1 and residual['runs'][0]['spec_sha256'] == digest and
            number(residual['runs'][0]['seconds']) and 0 < residual['runs'][0]['seconds'] <= m['per_run_max_seconds'] + 40,
            'completed run requires exactly one positive matching residual settlement')
    return {'guard_exit_code': 0, 'guard_spec_sha256': digest,
            'settled_seconds': residual['runs'][0]['seconds']}


def collect(config, config_sha, previous=None, probe=None, now=None):
    validate_config(config)
    now = time.time() if now is None else now
    probe = probe or LinuxProbe()
    m = config['monitor']
    root, repo = Path(config['project_root']), Path(config['training_repo'])
    current = root / 'runs/current'
    control, bank = root / 'runs' / m['run_id'], root / 'data' / m['run_id']
    errors, warnings = [], []
    report = {'schema': 'tea-handnet-comparison-snapshot-v1', 'captured_unix': now,
              'captured_at': dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
              'runtime_config_sha256': config_sha, 'collector_sha256': sha(Path(__file__)),
              'run_id': m['run_id'], 'paths': {'control': str(control), 'bank': str(bank),
              'status': str(control / 'STATUS.json'), 'guard': str(control / 'guard/status.json'),
              'original_budget': str(current / 'BUDGET.json'), 'residual_budget': str(control / 'RESIDUAL_BUDGET.json')},
              'errors': errors, 'warnings': warnings,
              'scope': 'Read-only observation and consistency checks; no independent raw-GT metric reconstruction.'}

    def checked(name, function):
        try:
            return function()
        except Exception as exc:
            errors.append({'check': name, 'error': repr(exc)})
            return None

    def verify_sources():
        matches = {}
        inventory, inventory_sha = read_json(current / 'runtime.json')
        require(inventory_sha == m['runtime_inventory_sha256'], 'frozen source inventory changed')
        for relative, expected in inventory['source_sha256'].items():
            p = repo / relative
            require(p.resolve().is_relative_to(repo.resolve()), 'source inventory escapes repo')
            matches[relative] = sha(p) == expected
        require(matches and all(matches.values()), 'frozen source file changed')
        require(sha(repo / 'Tea') == inventory['binary_sha256'] == m['binary_sha256'], 'training binary changed')
        require(sha(Path(config['worker_path'])) == m['worker_sha256'], 'published worker bytes changed')
        # Imports below are allowed only after the entire pinned inventory is checked.
        sys.path.insert(0, str(repo / 'scripts'))
        return {'inventory_sha256': inventory_sha, 'source_hash_matches': matches,
                'binary_matches': True, 'worker_matches': True}

    report['source_verification'] = checked('source_verification', verify_sources)
    identity = report['identity'] = checked('identity_query', probe.identity)
    if identity and (identity['hostname'] != config['hostname'] or identity['username'] != config['username'] or
                     not identity['platform'].startswith('linux')):
        errors.append({'check': 'identity', 'error': 'configured Linux host/user mismatch'})
    service = report['service'] = checked('service_query', lambda: probe.service(config['service_name']))
    processes = report['project_processes'] = checked('process_discovery', lambda: probe.processes(root, config['worker_path']))
    resources = report['resources'] = checked('resource_query', lambda: probe.resources(root))
    if resources and resources['disk']['free_bytes'] < m['disk_reserve_bytes']:
        errors.append({'check': 'disk_reserve', 'error': 'native disk reserve breached'})
    status = None
    if (control / 'STATUS.json').is_file():
        value = checked('worker_status', lambda: read_json(control / 'STATUS.json'))
        if value:
            status, report['status_sha256'] = value
    report['worker_status'] = status
    phase = report['phase'] = status.get('phase') if status else 'not_started'
    if status and phase not in ACTIVE | {COMPLETE, 'failed_preserved_no_retry'}:
        errors.append({'check': 'worker_phase', 'error': 'unknown worker phase; live/terminal identity cannot be inferred'})
    guard = None
    if (control / 'guard/status.json').is_file():
        value = checked('guard_status', lambda: read_json(control / 'guard/status.json'))
        if value:
            guard, report['guard_sha256'] = value
    report['guard_status'] = guard

    def budgets():
        original, original_sha = read_json(current / 'BUDGET.json')
        # A settlement can race a guard-status update; reread the ledger once around the pair.
        residual = None
        if (control / 'RESIDUAL_BUDGET.json').is_file():
            residual, first_sha = read_json(control / 'RESIDUAL_BUDGET.json')
            latest_guard = read_json(control / 'guard/status.json')[0] if (control / 'guard/status.json').is_file() else None
            residual2, second_sha = read_json(control / 'RESIDUAL_BUDGET.json')
            if first_sha != second_sha:
                residual = residual2
                latest_guard = read_json(control / 'guard/status.json')[0] if (control / 'guard/status.json').is_file() else None
        else:
            latest_guard = guard
        return accounting(original, residual, [latest_guard] if latest_guard else [], m, original_sha)

    report['accounting'] = checked('accounting', budgets)
    if phase == COMPLETE:
        report['completed_accounting'] = checked('completed_accounting', lambda: completed_accounting(control, guard, m))
    if not status:
        (errors if m['launch_expected'] else warnings).append({'check': 'run_presence', 'error': 'run STATUS.json absent'})
    else:
        checked('status_identity', lambda: require(status['run_id'] == m['run_id'] and
                status['worker_sha256'] == m['worker_sha256'] and
                status.get('runtime_config_sha256') == config_sha, 'status belongs to a different run/worker/config'))
        if phase.startswith('failed'):
            errors.append({'check': 'worker_failure', 'error': status.get('error', phase)})
        if phase in ACTIVE:
            age = now - status['updated_unix']
            if age < 0 or (phase in COMMAND_PHASES and age > m['status_stale_seconds']):
                errors.append({'check': 'status_freshness', 'error': 'active worker status is stale/future', 'age_seconds': age})
            def live_identity():
                worker = probe.process(status['pid'])
                require(not worker.get('exited') and worker['state'] != 'Z', 'worker process exited/zombie')
                ticks = status.get('start_ticks')
                require(ticks is not None and worker['start_ticks'] == str(ticks), 'worker PID/start identity mismatch')
                require(str(config['worker_path']) in worker['argv'], 'worker argv does not identify reviewed script')
                require(service and service.get('MainPID') == str(status['pid']) and service.get('ActiveState') == 'active',
                        'service MainPID/state mismatch')
                require(service.get('MemoryMax') == str(m['service_memory_max_bytes']) and
                        service.get('KillMode') == 'control-group' and
                        duration_seconds(service.get('TimeoutStopUSec', '')) >= m['service_min_stop_seconds'],
                        'service resource/shutdown contract mismatch')
                return worker
            report['worker_process'] = checked('live_worker_identity', live_identity)
            if phase not in COMMAND_PHASES and age > m['status_stale_seconds']:
                old_process = (previous or {}).get('worker_process') or {}
                current_process = report.get('worker_process') or {}
                cpu_progress = (current_process.get('start_ticks') == old_process.get('start_ticks') and
                                current_process.get('cpu_ticks', 0) > old_process.get('cpu_ticks', 0))
                limit = m.get('cpu_validation_stale_seconds', max(3600, m['progress_stale_seconds']))
                message = {'check': 'cpu_validation_freshness', 'age_seconds': age,
                           'cpu_ticks_advanced': cpu_progress,
                           'error': 'CPU validation/copy/export has no Runner heartbeat; assess process CPU and phase duration'}
                (errors if age > limit and not cpu_progress else warnings).append(message)
        elif phase.startswith('complete'):
            checked('completed_service', lambda: require(service and service.get('MainPID') == '0' and
                    service.get('ActiveState') == 'inactive' and service.get('ExecMainStatus') == '0' and
                    service.get('Result') == 'success', 'completed worker service not cleanly exited'))
    if service:
        cg = report['cgroup'] = checked('service_cgroup', lambda: probe.cgroup(service.get('ControlGroup', '')))
        if cg and (cg.get('events', {}).get('oom', 0) or cg.get('events', {}).get('oom_kill', 0)):
            errors.append({'check': 'service_oom', 'error': 'own service reports OOM events'})
        if phase in ACTIVE and cg and cg['status'] != 'present':
            errors.append({'check': 'service_cgroup', 'error': 'active service cgroup missing; root counters not substituted'})
    if guard and guard.get('phase') == 'failed':
        errors.append({'check': 'guard_failure', 'error': guard.get('error', 'guard failed')})
    expected_pids = {os.getpid()}
    if status:
        expected_pids.add(status['pid'])
        child = status.get('owned_child')
        if child and phase in COMMAND_PHASES:
            expected_pids.add(child['pid'])
            def child_identity():
                actual = probe.process(child['pid'])
                require(not actual.get('exited') and actual.get('state') != 'Z' and
                        actual['start_ticks'] == str(child['start_ticks']), 'owned child PID/start identity mismatch')
                require(actual['argv'] == child['argv'], 'owned child argv changed')
                return actual
            report['owned_child_process'] = checked('owned_child_identity', child_identity)
    if guard and guard.get('phase') in ('resource_wait', 'training'):
        expected_pids.add(guard['pid'])
        checked('guard_owner', lambda: require(status and status.get('owned_child', {}).get('pid') == guard['pid'],
                                              'live guard not owned by worker'))
        if now - guard['updated_at'] > m['status_stale_seconds']:
            errors.append({'check': 'guard_freshness', 'error': 'guard status stale'})
        def guard_spec():
            spec, digest = read_json(control / 'spec.json')
            require(digest == guard['spec_sha256'] and Path(spec['budget_file']) == control / 'RESIDUAL_BUDGET.json',
                    'guard spec/residual binding mismatch')
            require(Path(spec['checkpoint_dir']) == bank and Path(spec['output_dir']) == control / 'guard' and
                    Path(spec['cwd']) == repo and spec['argv'][0] == str(repo / 'Tea'), 'guard paths/Tea argv mismatch')
            require(number(spec['max_seconds']) and 0 < spec['max_seconds'] <= m['per_run_max_seconds'],
                    'guard spec active-time cap mismatch')
            return {'sha256': digest, 'max_seconds': spec['max_seconds'], 'budget_file': spec['budget_file']}
        report['guard_spec'] = checked('guard_spec', guard_spec)
        if guard['phase'] == 'training':
            expected_pids.add(guard.get('child_pid'))
            def tea_identity():
                spec, spec_sha = read_json(control / 'spec.json')
                require(spec_sha == guard['spec_sha256'] and Path(spec['budget_file']) == control / 'RESIDUAL_BUDGET.json',
                        'guard spec/budget binding mismatch')
                actual = probe.process(guard['child_pid'])
                require(not actual.get('exited') and actual.get('state') != 'Z' and
                        actual['start_ticks'] == str(guard['child_start_ticks']) and actual['argv'] == spec['argv'],
                        'Tea child PID/start/argv mismatch')
                require(guard['active_training_seconds'] <= guard['max_seconds'] + 40 and
                        guard['max_seconds'] <= m['per_run_max_seconds'], 'guard time cap exceeded')
                return actual
            report['tea_process'] = checked('tea_identity', tea_identity)
    if processes:
        unexpected = [p for p in processes if p['pid'] not in expected_pids and not p.get('exited')]
        report['unexpected_project_processes'] = unexpected
        if unexpected:
            errors.append({'check': 'replacement_discovery', 'error': 'unexpected/resumed project process requires reconciliation'})
    logs = report['logs'] = {}
    paths = {'conversion': control / 'conversion.log', 'guard_supervisor': control / 'guard-supervisor.log'}
    if status and status.get('setting'):
        setting = status['setting']
        paths['evaluation'] = control / f"validation-d{setting['splits']}-p{setting['threshold']}.log"
    if guard and guard.get('tea_log'):
        paths['tea'] = Path(guard['tea_log'])
    for name, path in paths.items():
        logs[name] = log_snapshot(path, now)
        if logs[name].get('error_lines'):
            errors.append({'check': 'numerical_log_errors', 'log': str(path), 'error': logs[name]['error_lines']})
    active_log = 'conversion' if phase == 'conversion' else 'evaluation' if phase == 'validation' else 'tea'
    if phase in COMMAND_PHASES and not (guard and guard.get('phase') == 'resource_wait'):
        log = logs.get(active_log, {})
        started = status.get('started_unix', now) if status else now
        if now - started > m['startup_grace_seconds'] and (not log.get('exists') or
                log.get('age_seconds', 0) > m['progress_stale_seconds']):
            errors.append({'check': 'progress_freshness', 'error': 'active command progress log missing/stale', 'kind': active_log})
    report['progress_assessment'] = 'expected_resource_wait' if guard and guard.get('phase') == 'resource_wait' else phase
    if report['source_verification']:
        report['checkpoints'] = checked('latest_checkpoint', lambda: checkpoint_summary(bank, m))
        if phase.startswith('complete'):
            checked('completed_checkpoint_count', lambda: require(report.get('checkpoints') and
                    report['checkpoints']['count'] == m['decisions'], 'completed run lacks all decision checkpoints'))
            report['completed_artifacts'] = checked('completed_artifacts', lambda: completed_artifact_summary(
                control, bank, root / 'models' / m['run_id'], status, m))
    if (bank / 'handnet_metadata.json').is_file():
        def dataset():
            info, digest = read_json(bank / 'handnet_metadata.json')
            require(info['complete'] and info['samples'] == m['samples'] and info['features'] == m['features'] and
                    info['classes'] == m['class_names'], 'completed bank metadata contract mismatch')
            require(info['coords_sha256'] == sha(bank / 'Coords.co') == m['coords_sha256'],
                    'completed bank coordinates changed')
            for index in range(m['features']):
                require((bank / f'F_{index:04d}.feat').stat().st_size == m['samples'], 'bank feature column size changed')
            for name, multiplier in (('Labels.lbl', 2), ('Pixels.pix', 4), ('Poses.pose', 4)):
                require((bank / name).stat().st_size == m['samples'] * multiplier, 'bank auxiliary size changed')
            return {'sha256': digest, 'samples': info['samples'], 'features': info['features'],
                    'frames': info['files_converted'], 'feature_columns': len(list(bank.glob('F_*.feat')))}
        report['dataset'] = checked('dataset', dataset)
    evaluations = report['evaluations'] = []
    def evals():
        cohort, cohort_sha = read_json(current / 'validation-selection/frames.json')
        require(cohort_sha == m['validation_manifest_sha256'] and len(cohort['frames']) == m['validation_frames'],
                'frozen validation cohort changed')
        require(len({frame_id(f)[0] for f in cohort['frames']}) == len(cohort['frames']), 'duplicate validation identity')
        for depth in m['validation_depths']:
            for threshold in m['probability_thresholds']:
                path = control / f'validation-d{depth}-p{threshold}.json'
                if not path.exists():
                    continue
                result, digest = read_json(path)
                summary = evaluation_summary(result, cohort['frames'], m)
                require(summary['splits'] == depth and summary['threshold'] == threshold, 'result filename/settings mismatch')
                manifest = root / 'models' / m['run_id'] / 'forest.json'
                require(result['manifest_sha256'] == sha(manifest), 'evaluation model manifest changed')
                evaluations.append({**summary, 'path': str(path), 'sha256': digest})
        if phase.startswith('complete'):
            require(len(evaluations) == len(m['validation_depths']) * len(m['probability_thresholds']), 'completed run lacks full grid')
            selected = max(evaluations, key=lambda e: (e['macro_f1_at_15mm'], -e['splits'], e['threshold']))
            chosen = read_json(control / 'VALIDATION_SELECTION.json')[0]['selected']
            require(chosen['splits'] == selected['splits'] and chosen['threshold'] == selected['threshold'] and
                    math.isclose(chosen['score'], selected['macro_f1_at_15mm'], rel_tol=0, abs_tol=1e-12), 'frozen grid winner/tie mismatch')
    checked('validation_outputs', evals)
    if previous:
        if previous.get('runtime_config_sha256') == config_sha and previous.get('run_id') == m['run_id']:
            changed = {key: report.get(key) != previous.get(key) for key in ('phase', 'checkpoints', 'evaluations')}
            changed['conversion'] = logs.get('conversion', {}).get('conversion') != previous.get('logs', {}).get('conversion', {}).get('conversion')
            report['progress_since_previous'] = changed
            if phase == previous.get('phase') == 'conversion' and not changed['conversion'] and now - previous['captured_unix'] >= m['progress_stale_seconds']:
                errors.append({'check': 'conversion_stall', 'error': 'processed frame/sample progress unchanged since prior check'})
        else:
            warnings.append({'check': 'previous_snapshot', 'error': 'different run/config; progress comparison skipped'})
    report['healthy'] = not errors
    report['observation_state'] = 'ok' if not errors else 'attention_required'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--previous-snapshot', type=Path)
    args = parser.parse_args()
    try:
        config, digest = read_json(args.runtime_config)
        previous = read_json(args.previous_snapshot)[0] if args.previous_snapshot else None
        report = collect(config, digest, previous)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report['healthy'] else 2
    except Exception as exc:
        print(json.dumps({'schema': 'tea-handnet-comparison-monitor-failure-v1',
                          'healthy': False, 'error': repr(exc), 'no_mutations': True}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
