"""Read-only monitor tests: temporary ledgers/process fixtures, never real jobs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/collect_handnet_comparison.py'
spec = importlib.util.spec_from_file_location('comparison_monitor_under_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return m.sha(path)


def item(number):
    return {'frame_id': 'handnet:gidx:' + str(number), 'gidx': number,
            'name': f'frame-{number}.mat', 'session': 'synthetic-session', 'idx': number}


class Probe:
    """No subprocesses, GPU queries, services or actual /proc accesses."""
    def __init__(self):
        self.identity_value = {'hostname': 'synthetic-host', 'username': 'synthetic-user', 'platform': 'linux'}
        self.service_value = {'MainPID': '321', 'ActiveState': 'active', 'Result': 'success',
                              'ExecMainStatus': '0', 'MemoryMax': str(24 * 1024**3),
                              'KillMode': 'control-group', 'TimeoutStopUSec': '1min',
                              'ControlGroup': '/synthetic-own-unit'}
        self.cg = {'status': 'present', 'events': {'oom': 0, 'oom_kill': 0}}
        self.process_values = {}
        self.extra = []
    def identity(self): return self.identity_value
    def service(self, unit): return self.service_value
    def cgroup(self, group): return self.cg
    def process(self, pid): return self.process_values.get(pid, {'pid': pid, 'exited': True})
    def processes(self, root, worker): return list(self.process_values.values()) + self.extra
    def resources(self, root):
        return {'disk': {'free_bytes': 100 * 1024**3, 'total_bytes': 200 * 1024**3},
                'ram': {'MemAvailable_bytes': 8 * 1024**3},
                'gpu': [{'utilization_percent': 99, 'memory_used_mib': 10000,
                         'memory_free_mib': 100, 'memory_total_mib': 10100}]}


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.current = self.root / 'runs/current'
        self.current.mkdir(parents=True)
        self.repo = self.root / 'training'
        (self.repo / 'scripts').mkdir(parents=True)
        (self.repo / 'scripts/frozen.py').write_text('# synthetic source fixture\n')
        (self.repo / 'Tea').write_bytes(b'synthetic binary; never executed')
        self.worker = self.root / 'operations/worker.py'
        self.worker.parent.mkdir()
        self.worker.write_text('# synthetic worker; never executed\n')
        self.proposal = self.root / 'proposal'
        self.proposal.mkdir()
        self.run_id = 'synthetic-comparison'
        self.control = self.root / 'runs' / self.run_id
        self.control.mkdir()
        self.bank = self.root / 'data' / self.run_id
        self.bank.mkdir(parents=True)
        self.old_specs = ['a' * 64, 'b' * 64]
        self.original = {'max_active_training_seconds': 172800, 'max_final_trees': 3,
                         'used_active_training_seconds': 100.,
                         'runs': [{'spec_sha256': self.old_specs[0], 'seconds': 40.},
                                  {'spec_sha256': self.old_specs[1], 'seconds': 60.}]}
        old_sha = write(self.current / 'BUDGET.json', self.original)
        inventory_sha = write(self.current / 'runtime.json', {
            'source_sha256': {'scripts/frozen.py': m.sha(self.repo / 'scripts/frozen.py')},
            'binary_sha256': m.sha(self.repo / 'Tea')})
        self.frames = [item(1), item(2)]
        val_sha = write(self.current / 'validation-selection/frames.json', {'frames': self.frames})
        expected = {'schema': m.SCHEMA, 'run_id': self.run_id,
                    'worker_sha256': m.sha(self.worker), 'original_budget_sha256': old_sha,
                    'original_used_seconds': 100., 'original_settled_runs': 2,
                    'original_final_trees': 3, 'authorized_total_final_trees': 4,
                    'aggregate_ceiling_seconds': 172800, 'per_run_max_seconds': 57600,
                    'max_new_runs': 1, 'runtime_inventory_sha256': inventory_sha,
                    'binary_sha256': m.sha(self.repo / 'Tea'), 'coords_sha256': 'c' * 64,
                    'validation_manifest_sha256': val_sha, 'samples': 400, 'features': 2,
                    'classes': 8, 'decisions': 20, 'validation_frames': 2,
                    'class_names': ['finger0', 'finger1', 'finger2', 'finger3', 'finger4', 'palm', 'wrist', 'background'],
                    'validation_depths': [12, 16, 20], 'probability_thresholds': [.25, .5],
                    'min_pixels': 3, 'foreground': 'none', 'fingertip_classes': [0, 1, 2, 3, 4],
                    'service_memory_max_bytes': 24 * 1024**3, 'service_min_stop_seconds': 60,
                    'disk_reserve_bytes': 16 * 1024**3, 'status_stale_seconds': 300,
                    'progress_stale_seconds': 1800, 'startup_grace_seconds': 120,
                    'launch_expected': True}
        self.config = {'schema': 'tea-handnet-comparison-runtime-v1',
                       'project_root': str(self.root), 'training_repo': str(self.repo),
                       'proposal_dir': str(self.proposal), 'worker_path': str(self.worker),
                       'python_executable': sys.executable, 'service_name': 'synthetic-comparison.service',
                       'hostname': 'synthetic-host', 'username': 'synthetic-user', 'monitor': expected}
        self.now = 2000000000.
        self.status = {'run_id': self.run_id, 'phase': 'conversion', 'pid': 321,
                       'start_ticks': '111', 'started_unix': self.now - 10,
                       'runtime_config_sha256': 'd' * 64,
                       'updated_unix': self.now, 'worker_sha256': expected['worker_sha256'],
                       'owned_child': {'pid': 322, 'start_ticks': '222', 'argv': ['python', 'converter.py']}}
        write(self.control / 'STATUS.json', self.status)
        self.residual = {'original_budget_sha256': old_sha, 'original_used_seconds': 100.,
                         'max_active_training_seconds': 172700., 'max_new_final_trees': 1,
                         'used_active_training_seconds': 0., 'runs': []}
        write(self.control / 'RESIDUAL_BUDGET.json', self.residual)
        (self.control / 'conversion.log').write_text('processed 100/200 frames, samples=135200\n')
        os.utime(self.control / 'conversion.log', (self.now, self.now))
        self.probe = Probe()
        self.probe.process_values = {
            321: {'pid': 321, 'state': 'S', 'start_ticks': '111', 'argv': ['python', str(self.worker)]},
            322: {'pid': 322, 'state': 'R', 'start_ticks': '222', 'argv': ['python', 'converter.py']}}

    def tearDown(self): self.temp.cleanup()

    def snapshot(self, previous=None):
        return m.collect(self.config, 'd' * 64, previous, self.probe, self.now)

    def test_conversion_progress_and_complete_read_only_inventory(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = self.snapshot()
        self.assertTrue(report['healthy'], report['errors'])
        self.assertEqual(report['logs']['conversion']['conversion'], [100, 200, 135200])
        self.assertEqual(report['accounting']['aggregate_seconds'], 100.)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_missing_run_is_failure_once_launch_expected(self):
        (self.control / 'STATUS.json').unlink()
        report = self.snapshot()
        self.assertFalse(report['healthy'])
        self.assertIn('run_presence', [e['check'] for e in report['errors']])

    def test_pid_reuse_wrong_service_and_source_changes_fail(self):
        self.probe.process_values[321]['start_ticks'] = 'reused'
        self.assertFalse(self.snapshot()['healthy'])
        self.probe.process_values[321]['start_ticks'] = '111'
        self.probe.service_value['KillMode'] = 'process'
        self.assertFalse(self.snapshot()['healthy'])
        self.probe.service_value['KillMode'] = 'control-group'
        (self.repo / 'scripts/frozen.py').write_text('changed source')
        self.assertFalse(self.snapshot()['healthy'])

    def test_runtime_config_binding_and_cpu_validation_without_runner_heartbeat(self):
        self.status['runtime_config_sha256'] = 'wrong'
        write(self.control / 'STATUS.json', self.status)
        self.assertIn('status_identity', [e['check'] for e in self.snapshot()['errors']])
        self.status.update(runtime_config_sha256='d'*64, phase='validating_bank', updated_unix=self.now-900)
        write(self.control / 'STATUS.json', self.status)
        self.probe.process_values.pop(322)
        # A real worker does synchronous CPU bank validation with no Runner tick.
        report = self.snapshot()
        self.assertTrue(report['healthy'], report['errors'])
        self.assertIn('cpu_validation_freshness', [e['check'] for e in report['warnings']])

    def test_completed_status_does_not_hide_missing_checkpoints_or_artifacts(self):
        self.status['phase'] = 'complete_pending_independent_native_parity_backup_and_report'
        write(self.control / 'STATUS.json', self.status)
        self.probe.process_values = {}
        self.probe.service_value.update(MainPID='0', ActiveState='inactive')
        self.probe.cg = {'status': 'not_applicable_no_service_cgroup'}
        report = self.snapshot()
        names = [e['check'] for e in report['errors']]
        self.assertIn('completed_checkpoint_count', names)
        self.assertIn('completed_artifacts', names)
        self.assertIn('completed_accounting', names)

    def test_completed_accounting_requires_guard_exact_settlement_and_clean_exit(self):
        spec = {'budget_file': str(self.control/'RESIDUAL_BUDGET.json')}
        digest = write(self.control/'spec.json', spec)
        guard = {'phase': 'training_complete_pending_validation', 'exit_code': 0,
                 'spec_sha256': digest}
        residual = dict(self.residual, used_active_training_seconds=5,
                        runs=[{'spec_sha256': digest, 'seconds': 5}])
        write(self.control/'RESIDUAL_BUDGET.json', residual)
        self.assertEqual(m.completed_accounting(self.control,guard,self.config['monitor'])['settled_seconds'],5)
        for bad in [None, dict(guard,exit_code=1),dict(guard,accounting_failed=True),dict(guard,phase='training')]:
            with self.assertRaises(ValueError):m.completed_accounting(self.control,bad,self.config['monitor'])
        (self.control/'RESIDUAL_BUDGET.json').unlink()
        with self.assertRaises(FileNotFoundError):m.completed_accounting(self.control,guard,self.config['monitor'])

    def test_unknown_worker_phase_is_never_healthy(self):
        self.status['phase'] = 'unrecognized_phase'
        write(self.control/'STATUS.json', self.status)
        self.assertIn('worker_phase',[e['check'] for e in self.snapshot()['errors']])

    def test_completed_retained_or_selected_artifact_corruption_refused(self):
        retained = self.root / 'models' / self.run_id
        selected = retained / 'selected'
        selected.mkdir(parents=True)
        (self.bank / 'Coords.co').write_bytes(b'synthetic coordinates')
        (self.bank / 'Tree_0000.depth00.tree').write_bytes(b'synthetic hash-only checkpoint')
        (self.bank / 'Tree_0000.tree').write_bytes((self.bank / 'Tree_0000.depth00.tree').read_bytes())
        checkpoint_hash = m.sha(self.bank / 'Tree_0000.tree')
        expected = dict(self.config['monitor'], decisions=1, coords_sha256=m.sha(self.bank/'Coords.co'))
        write(self.control / 'MODEL_VALIDATION.json', {'external_predictions_match': True,
              'checkpoints': [{'name': 'Tree_0000.depth00.tree', 'sha256': checkpoint_hash}],
              'model_sha256': checkpoint_hash, 'coords_sha256': expected['coords_sha256']})
        copies = {}
        for name, value in [('model.tree', b'synthetic hash-only checkpoint'),
                            ('Coords.co', b'synthetic coordinates'), ('training.json', b'{}'), ('frames.json', b'{}')]:
            (retained/name).write_bytes(value); copies[name] = m.sha(retained/name)
        write(retained/'RETAINED_FILES.json', copies)
        for name in ('model.tree', 'Coords.co'):
            (selected/name).write_bytes((retained/name).read_bytes())
        write(self.control/'VALIDATION_SELECTION.json', {'selected': {'threshold': .25}})
        final = selected/'forest.json'
        final_sha = write(final, {'postprocessing': {'foreground': 'none', 'min_pixels': 3,
                          'fingertip_classes': [0,1,2,3,4], 'probability_threshold': .25},
                          'trees': [{'model': 'model.tree', 'coords': 'Coords.co',
                                     'model_sha256': copies['model.tree'], 'coords_sha256': copies['Coords.co']}]})
        deployment = {'path': str(final), 'sha256': final_sha}
        write(self.control/'DEPLOYMENT_VALIDATION.json', deployment)
        self.assertTrue(m.completed_artifact_summary(self.control,self.bank,retained,
                        {'deployment': deployment},expected)['retained_file_hashes_match'])
        (retained/'model.tree').write_bytes(b'late corruption')
        with self.assertRaisesRegex(ValueError, 'retained model'):
            m.completed_artifact_summary(self.control,self.bank,retained,{'deployment': deployment},expected)

    def test_own_cgroup_oom_never_uses_root_fallback(self):
        self.probe.cg['events']['oom_kill'] = 1
        report = self.snapshot()
        self.assertIn('service_oom', [e['check'] for e in report['errors']])
        probe = m.LinuxProbe()
        self.assertEqual(probe.cgroup(''), {'status': 'not_applicable_no_service_cgroup'})
        self.assertEqual(probe.cgroup('/'), {'status': 'not_applicable_no_service_cgroup'})

    def test_expected_resource_wait_does_not_call_stale_training_a_stall(self):
        self.status.update(phase='resource_wait_or_guarded_training', started_unix=self.now-10000)
        write(self.control / 'STATUS.json', self.status)
        write(self.control / 'guard/status.json', {'pid': 322, 'phase': 'resource_wait',
              'updated_at': self.now, 'active_training_seconds': 0, 'spec_sha256': 'e' * 64,
              'wait_reason': 'cooperative GPU lease busy'})
        guard_spec = {'budget_file': str(self.control/'RESIDUAL_BUDGET.json'),
                      'checkpoint_dir': str(self.bank), 'output_dir': str(self.control/'guard'),
                      'cwd': str(self.repo), 'argv': [str(self.repo/'Tea')], 'max_seconds': 57600}
        digest = write(self.control/'spec.json', guard_spec)
        guard = m.read_json(self.control/'guard/status.json')[0]; guard['spec_sha256'] = digest
        write(self.control/'guard/status.json', guard)
        report = self.snapshot()
        self.assertTrue(report['healthy'], report['errors'])
        self.assertEqual(report['progress_assessment'], 'expected_resource_wait')

    def test_discovered_replacement_and_stale_conversion_are_failures(self):
        self.probe.extra = [{'pid': 999, 'argv': [str(self.repo/'Tea')], 'start_ticks': '999', 'state': 'R'}]
        self.assertIn('replacement_discovery', [e['check'] for e in self.snapshot()['errors']])
        self.probe.extra = []
        previous = self.snapshot()
        previous['captured_unix'] -= 1900
        self.assertIn('conversion_stall', [e['check'] for e in self.snapshot(previous)['errors']])

    def test_settlement_live_dedup_and_terminal_failure_reconciliation(self):
        mcfg = self.config['monitor']
        live = {'spec_sha256': 'e' * 64, 'phase': 'training', 'active_training_seconds': 5}
        report = m.accounting(self.original, self.residual, [live, live], mcfg, mcfg['original_budget_sha256'])
        self.assertEqual(report['aggregate_seconds'], 105.)
        settled = copy.deepcopy(self.residual)
        settled.update(runs=[{'spec_sha256': 'e' * 64, 'seconds': 5}], used_active_training_seconds=5)
        report = m.accounting(self.original, settled, [live], mcfg, mcfg['original_budget_sha256'])
        self.assertEqual(report['aggregate_seconds'], 105.)
        with self.assertRaisesRegex(ValueError, 'terminal guard'):
            m.accounting(self.original, self.residual, [dict(live, phase='failed')], mcfg, mcfg['original_budget_sha256'])
        with self.assertRaisesRegex(ValueError, 'accounting failure'):
            m.accounting(self.original, self.residual, [dict(live, accounting_failed=True)], mcfg, mcfg['original_budget_sha256'])

    def test_budget_mutation_duplicates_double_debit_and_extra_attempt_refused(self):
        e = self.config['monitor']
        for residual in [dict(self.residual, max_active_training_seconds=172600),
                         dict(self.residual, runs=[{'spec_sha256': self.old_specs[0], 'seconds': 1}], used_active_training_seconds=1),
                         dict(self.residual, runs=[{'spec_sha256': 'e'*64, 'seconds': 1}]*2, used_active_training_seconds=2)]:
            with self.assertRaises(ValueError): m.accounting(self.original, residual, [], e, e['original_budget_sha256'])
        with self.assertRaisesRegex(ValueError, 'budget bytes'):
            m.accounting(self.original, self.residual, [], e, '0'*64)

    def evaluation(self):
        center = {'pixels': 3, 'xy': [1., 1.], 'xyz': [1., 1., 400.]}
        frames = [dict(f, fingertips=[copy.deepcopy(center) for _ in range(5)]) for f in self.frames]
        recall = {t: {'hits': [1]*5} for t in ('5', '10', '15', '20')}
        return {'schema': 'handnet-tea-evaluation-v1', 'split': 'validation', 'frames': frames,
                'classes': self.config['monitor']['class_names'], 'foreground': 'none',
                'pixel_limit_per_frame': 0, 'max_splits': 20,
                'localization': {'minimum_pixels': 3, 'probability_threshold': .25,
                'present': [2]*5, 'detected': [2]*5, 'absent': [0]*5,
                'false_positive_on_absent': [0]*5, 'sensor_pos_present': [2]*5,
                'sensor_pos_detected': [2]*5, 'sensor_pos_recall_at_mm': recall,
                'label_centroid_recall_at_px': recall}}

    def test_validation_exact_ids_integer_bounds_monotonic_hits_and_geometry(self):
        result = self.evaluation()
        self.assertEqual(m.evaluation_summary(result, self.frames, self.config['monitor'])['macro_f1_at_15mm'], .5)
        for mutate in [lambda x: x['frames'].reverse(),
                       lambda x: x['localization']['sensor_pos_present'].__setitem__(0, .5),
                       lambda x: x['localization']['false_positive_on_absent'].__setitem__(0, 3),
                       lambda x: x['localization']['sensor_pos_recall_at_mm']['10']['hits'].__setitem__(0, 2),
                       lambda x: x['frames'][0]['fingertips'][0]['xyz'].__setitem__(2, float('nan'))]:
            bad = copy.deepcopy(result); mutate(bad)
            with self.assertRaises(ValueError): m.evaluation_summary(bad, self.frames, self.config['monitor'])

    def test_checkpoint_binary_validator_and_invalid_node(self):
        # Real TeaTree reference on a tiny native tree. No CUDA or training.
        sys.path.insert(0, str(SCRIPT.parent))
        from tea_model import TeaTree, NODE_DTYPE, LEAF_DTYPE
        import numpy as np
        coords = self.bank / 'Coords.co'; np.zeros((2,4), dtype='<f4').tofile(coords)
        expected = dict(self.config['monitor'], coords_sha256=m.sha(coords))
        nodes = np.zeros(1, dtype=NODE_DTYPE)
        leaves = np.zeros(3, dtype=LEAF_DTYPE); leaves['maximum'] = 1
        hist = np.zeros((3,8), dtype=np.uint8); hist[:,0] = 255
        checkpoint = self.bank / 'Tree_0000.depth00.tree'
        checkpoint.write_bytes(struct.pack('<4i',0,8,1,3)+nodes.tobytes()+leaves.tobytes()+hist.tobytes())
        report = m.checkpoint_summary(self.bank, expected, TeaTree)
        self.assertEqual(report['latest']['decisions'], 1)
        self.assertTrue(report['latest']['finite_normalized_probabilities'])
        nodes['feature'] = 2
        checkpoint.write_bytes(struct.pack('<4i',0,8,1,3)+nodes.tobytes()+leaves.tobytes()+hist.tobytes())
        with self.assertRaises(ValueError): m.checkpoint_summary(self.bank, expected, TeaTree)

    def test_process_stat_comm_spaces_and_replacement_discovery(self):
        proc = self.root / 'fake-proc'; p = proc / '123'; p.mkdir(parents=True)
        fields = ['S','1'] + ['0']*20; fields[11]='5'; fields[12]='6'; fields[19]='789'
        (p/'stat').write_text('123 (name with spaces) '+' '.join(fields))
        (p/'cmdline').write_bytes((str(self.repo/'Tea')+'\0').encode())
        (p/'cgroup').write_text('0::/synthetic-own-unit')
        value = m.process_snapshot(123, proc)
        self.assertEqual(value['start_ticks'], '789'); self.assertEqual(value['cpu_ticks'], 11)
        self.assertEqual(m.discover_processes(self.root, self.worker, proc)[0]['pid'], 123)

    def test_service_timeout_parser_rejects_infinity(self):
        self.assertEqual(m.duration_seconds('1min 30s'), 90)
        with self.assertRaises(ValueError): m.duration_seconds('infinity')


if __name__ == '__main__':
    unittest.main()
