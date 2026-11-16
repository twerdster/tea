"""Independent comparison-worker refusal checks; no operational jobs are used.

The CLI is invoked in-process through its argument parser and main entrypoint.
Production context, filesystem publication and subprocess launch are tripwires.
All supplied approval/review/monitor documents are UNIT TEST ONLY fixtures.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKER_PATH = HERE.parent / 'scripts' / 'run_handnet_comparison.py'
SPEC = importlib.util.spec_from_file_location(
    'handnet_comparison_refusal_independent', WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
# Bind this test run to one coherent source snapshot.
WORKER_SOURCE = WORKER_PATH.read_bytes()
TESTED_WORKER_SHA256 = hashlib.sha256(WORKER_SOURCE).hexdigest()
exec(compile(WORKER_SOURCE, str(WORKER_PATH), 'exec'), worker.__dict__)

TEST_SHA = 'UNIT TEST ONLY worker binding; no deployment or authorization'
TEST_RUNTIME_SHA = '2' * 64
TEST_REFERENCE = 'UNIT TEST ONLY; not an operational user approval'


def fixture(now=None):
    now = time.time() if now is None else now
    revision = '1' * 40
    auth = {
        'schema': 'tea-comparison-authorization-v1',
        'status': 'approved', 'scope': 'one_extra',
        'requested_runs': [{'run_id': worker.RUN_ID, 'seed': 204}],
        'existing_original_slots': ['final-tree-0', 'final-tree-1', 'final-tree-2'],
        'reserved_additional_ids': [], 'max_total_final_slots': 4,
        'additional_pilots': 0, 'max_aggregate_active_seconds': 172800,
        'max_active_seconds_per_new_run': 60,
        'user_instruction_reference': TEST_REFERENCE,
        'approved_at': '2026-01-01T00:00:00+00:00',
    }
    review = {
        'source_is_published': True, 'checks_passed': True,
        'worker_sha256': TEST_SHA, 'runtime_config_sha256': TEST_RUNTIME_SHA,
        'published_revision': revision,
        'published_source_url': 'https://github.com/twerdster/tea/blob/'
                                + revision + '/scripts/UNIT-TEST-ONLY.py',
        'root_review_reference': TEST_REFERENCE,
    }
    monitor = {
        'run_id': worker.RUN_ID, 'worker_sha256': TEST_SHA,
        'runtime_config_sha256': TEST_RUNTIME_SHA,
        'status': 'ACTIVE', 'rrule': 'FREQ=HOURLY;INTERVAL=2',
        'covers_run': True, 'verified_unix': now,
        'integration_review_passed': True,
        'aggregates_original_and_residual_once': True,
        'tracks_live_guard_once': True,
        'root_review_reference': TEST_REFERENCE,
    }
    return [auth, review, monitor]


def refusal_cases():
    """Each mutation targets a distinct operational admission requirement."""
    cases = []

    def add(name, component, field, value, message):
        values = fixture()
        if field is None:
            values[component] = value
        else:
            values[component][field] = value
        cases.append((name, values, message))

    add('empty_authorization', 0, None, {}, 'authorization missing/invalid')
    add('pending_scope', 0, 'status', 'pending_scope', 'explicit one_extra')
    add('rejected_authorization', 0, 'status', 'rejected', 'explicit one_extra')
    add('three_extra_scope', 0, 'scope', 'three_extra', 'batch unsupported')
    add('three_new_runs', 0, 'requested_runs', [
        {'run_id': 'UNIT-TEST-ONLY-' + str(s), 'seed': s}
        for s in (204, 205, 206)], 'only seed204')
    add('different_single_seed', 0, 'requested_runs', [
        {'run_id': worker.RUN_ID, 'seed': 205}], 'only seed204')
    add('unreconciled_original_three_slot_cap', 0, 'max_total_final_slots', 3,
        'total-four')
    add('unauthorized_six_slot_cap', 0, 'max_total_final_slots', 6, 'total-four')
    add('reclassified_original_slots', 0, 'existing_original_slots', [],
        'original slots must remain counted')
    add('extra_reserved_seed', 0, 'reserved_additional_ids', ['seed205'],
        'extra pilots/reservations unsupported')
    add('additional_pilot', 0, 'additional_pilots', 1,
        'extra pilots/reservations unsupported')
    add('expanded_aggregate_budget', 0, 'max_aggregate_active_seconds', 172801,
        '48h ceiling cannot expand')
    add('expanded_per_run_budget', 0, 'max_active_seconds_per_new_run', 57601,
        'per-run ceiling')
    add('zero_run_budget', 0, 'max_active_seconds_per_new_run', 0,
        'per-run ceiling')
    add('boolean_run_budget', 0, 'max_active_seconds_per_new_run', True,
        'per-run ceiling')
    add('missing_instruction_reference', 0, 'user_instruction_reference', '',
        'user instruction reference')
    add('naive_approval_timestamp', 0, 'approved_at', '2026-01-01T00:00:00',
        'approval timestamp')
    add('absent_source_review', 1, None, None, 'published-source/hash review')
    add('unpublished_source', 1, 'source_is_published', False,
        'published-source/hash review')
    add('unreviewed_source', 1, 'checks_passed', False,
        'published-source/hash review')
    add('different_worker_bytes', 1, 'worker_sha256', 'DIFFERENT UNIT TEST SHA',
        'exact published worker bytes')
    add('invalid_source_revision', 1, 'published_revision', 'main',
        'exact published worker bytes')
    add('different_repository_review_url', 1, 'published_source_url',
        'https://example.invalid/UNIT-TEST-ONLY', 'public source revision')
    add('absent_root_review', 1, 'root_review_reference', '',
        'root review reference')
    add('missing_review_runtime_binding', 1, 'runtime_config_sha256', None,
        'runtime')
    add('different_review_runtime_binding', 1, 'runtime_config_sha256', '3' * 64,
        'runtime')
    add('absent_monitor', 2, None, None, 'integration review receipt')
    add('unreviewed_monitor_integration', 2, 'integration_review_passed', False,
        'integration review receipt')
    add('monitor_double_counts_budget', 2, 'aggregates_original_and_residual_once',
        False, 'integration review receipt')
    add('monitor_omits_live_guard', 2, 'tracks_live_guard_once', False,
        'integration review receipt')
    add('monitor_lacks_root_review', 2, 'root_review_reference', '',
        'integration review receipt')
    add('wrong_monitor_run', 2, 'run_id', 'UNIT-TEST-ONLY-other-run',
        'monitor acknowledgment')
    add('wrong_monitor_worker', 2, 'worker_sha256', 'DIFFERENT UNIT TEST SHA',
        'monitor acknowledgment')
    add('missing_monitor_runtime_binding', 2, 'runtime_config_sha256', None,
        'runtime')
    add('different_monitor_runtime_binding', 2, 'runtime_config_sha256', '3' * 64,
        'runtime')
    add('inactive_monitor', 2, 'status', 'PAUSED', 'active two-hour monitor')
    add('different_monitor_interval', 2, 'rrule', 'FREQ=HOURLY;INTERVAL=24',
        'active two-hour monitor')
    add('monitor_does_not_cover_run', 2, 'covers_run', False,
        'active two-hour monitor')
    add('stale_monitor', 2, 'verified_unix', time.time() - 10801,
        'fresh monitor acknowledgment')
    add('future_monitor', 2, 'verified_unix', time.time() + 3600,
        'fresh monitor acknowledgment')
    return cases


def temporary_inventory(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob('*')}


class IndependentRefusalTests(unittest.TestCase):
    def invoke_refusal(self, args, root, expected_message=None,
                       include_runtime=True):
        runtime_path = root / 'runtime-UNIT-TEST-ONLY.json'
        if include_runtime:
            # Hashing is mocked; parsing these bytes must remain unreachable.
            runtime_path.write_text('UNIT TEST ONLY; invalid runtime JSON')
            args = ['--runtime-config', str(runtime_path)] + args
        before = temporary_inventory(root)
        worker_before = WORKER_PATH.read_bytes()
        stdout, stderr = io.StringIO(), io.StringIO()
        original_read = worker.read

        def read_document(path):
            if Path(path) == runtime_path:
                raise AssertionError('UNEXPECTED runtime config read')
            return original_read(path)

        def reviewed_hash(path):
            if Path(path) == runtime_path:
                return TEST_RUNTIME_SHA
            if Path(path).resolve() == WORKER_PATH:
                return TEST_SHA
            raise AssertionError('UNEXPECTED hash: ' + str(path))

        with contextlib.ExitStack() as stack:
            mocks = {}
            for owner, name in [
                (worker, 'production'), (worker, 'runtime_config'),
                (worker.Backend, 'preflight'),
                (worker.Runner, 'run'), (worker, 'publish'),
                (worker.subprocess, 'Popen'), (worker.subprocess, 'run'),
                (Path, 'mkdir'),
            ]:
                mocks[name + ':' + str(id(owner))] = stack.enter_context(
                    patch.object(owner, name,
                                 side_effect=AssertionError('UNEXPECTED ' + name)))
            # Neither fixture constitutes reviewed source or runtime config.
            stack.enter_context(patch.object(worker, 'sha', side_effect=reviewed_hash))
            stack.enter_context(patch.object(worker, 'read', side_effect=read_document))
            stack.enter_context(patch.object(sys, 'argv', ['run_handnet_comparison.py'] + args))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            try:
                code = worker.main()
            except SystemExit as error:
                code = error.code
        self.assertEqual(code, 2, stdout.getvalue() + stderr.getvalue())
        if expected_message:
            self.assertIn(expected_message, stderr.getvalue())
        for mock in mocks.values():
            mock.assert_not_called()
        self.assertEqual(temporary_inventory(root), before)
        self.assertEqual(WORKER_PATH.read_bytes(), worker_before)

    def test_missing_authorization_cli_refuses_all_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'preserve-original-ledger.txt').write_text('UNIT TEST SENTINEL')
            for mode in ([], ['--check'], ['--run']):
                with self.subTest(mode=mode):
                    self.invoke_refusal(mode, root, 'explicit authorization')

    def test_nonexistent_and_malformed_authorization_cli_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / 'malformed-UNIT-TEST-ONLY.json'
            bad.write_text('{ not JSON')
            for path in (root / 'missing.json', bad):
                for mode in ('--check', '--run'):
                    with self.subTest(path=path.name, mode=mode):
                        self.invoke_refusal(['--authorization', str(path), mode], root)

    def test_invalid_documents_cli_refuse_before_context_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values, message in refusal_cases():
                paths = [root / (label + '-UNIT-TEST-ONLY.json')
                         for label in ('authorization', 'review', 'monitor')]
                for path, value in zip(paths, values):
                    path.write_text(json.dumps(value))
                arguments = ['--authorization', str(paths[0]),
                             '--source-review', str(paths[1]),
                             '--monitor-ack', str(paths[2])]
                for mode in ('--check', '--run'):
                    with self.subTest(refusal=name, mode=mode):
                        self.invoke_refusal(arguments + [mode], root, message)

    def test_missing_optional_document_flags_do_not_bypass_published_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, r, _ = fixture()
            auth, review = root / 'auth.json', root / 'review.json'
            auth.write_text(json.dumps(a)); review.write_text(json.dumps(r))
            for args, message in [
                (['--authorization', str(auth)], 'published-source/hash review'),
                (['--authorization', str(auth), '--source-review', str(review)],
                 'integration review receipt'),
            ]:
                self.invoke_refusal(args + ['--run'], root, message)

    def test_check_and_run_are_mutually_exclusive_before_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / (str(i) + '-UNIT-TEST-ONLY.json') for i in range(3)]
            for path, value in zip(paths, fixture()):
                path.write_text(json.dumps(value))
            self.invoke_refusal([
                '--authorization', str(paths[0]), '--source-review', str(paths[1]),
                '--monitor-ack', str(paths[2]), '--check', '--run'], root)

    def test_missing_runtime_config_cli_refuses_before_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / (str(i) + '-UNIT-TEST-ONLY.json') for i in range(3)]
            for path, value in zip(paths, fixture()):
                path.write_text(json.dumps(value))
            arguments = ['--authorization', str(paths[0]),
                         '--source-review', str(paths[1]),
                         '--monitor-ack', str(paths[2])]
            for mode in ('--check', '--run'):
                self.invoke_refusal(arguments + [mode], root, 'runtime',
                                    include_runtime=False)

    def test_pure_gate_refusals_preserve_all_supplied_documents(self):
        for name, values, message in refusal_cases():
            with self.subTest(refusal=name):
                before = copy.deepcopy(values)
                with self.assertRaisesRegex(worker.Refused, message):
                    worker.authorization(*values, TEST_SHA, TEST_RUNTIME_SHA)
                self.assertEqual(values, before)

    def test_fixture_validates_only_the_pure_gate(self):
        # No production/Worker invocation: this cannot authorize an operational run.
        now = time.time()
        values = fixture(now)
        before = copy.deepcopy(values)
        self.assertEqual(worker.authorization(*values, TEST_SHA, TEST_RUNTIME_SHA, now=now), 60.0)
        self.assertEqual(values, before)


if __name__ == '__main__':
    print(json.dumps({'tested_worker_sha256': TESTED_WORKER_SHA256,
                      'independent_refusal_document_cases': len(refusal_cases()),
                      'cli_refusal_invocations': 12 + 2 * len(refusal_cases()),
                      'operational_commands_launched': 0}))
    unittest.main(verbosity=2)
