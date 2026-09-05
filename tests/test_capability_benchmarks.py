import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from modules.capability_benchmarks import BenchmarkManager, EVENT_PREFIX, save_json, validate_config


def config(**changes):
    values = dict(task='gsm8k', endpoint='http://127.0.0.1:5000/v1', model='qwen', limit=10,
                  max_tokens=8192, temperature=0, seed=42, thinking='Server default')
    values.update(changes)
    return validate_config(**values)


class BenchmarkTests(unittest.TestCase):
    def test_rejects_credentials_invalid_numbers_and_unknown_tasks(self):
        for changes in [dict(endpoint='http://key:secret@host/v1'), dict(endpoint='file:///tmp/model'),
                        dict(endpoint='https://host/v1?api_key=secret'), dict(limit=0), dict(limit=1.5),
                        dict(temperature=float('nan')), dict(task='arbitrary/script'), dict(model='None')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_worker_events_persist_and_secrets_are_excluded_from_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(directory)
            run_dir = Path(directory) / 'runs' / 'test-run'
            run_dir.mkdir(parents=True)
            record = dict(id='test-run', status='starting', config=config(), samples=[], completed=0, errors=0, truncated=0)
            manager.active = record
            sample = dict(id='1', error=None, truncated=True, answer='hello')
            events = 'secret-key debug\n' + EVENT_PREFIX + json.dumps(dict(kind='sample', sample=sample)) + '\n'
            events += EVENT_PREFIX + json.dumps(dict(status='completed', metrics=[dict(metric='accuracy', value=1)])) + '\n'
            process = Mock(stdout=io.StringIO(events))
            process.wait.return_value = 0
            with patch.object(manager, '_spawn', return_value=process) as spawn:
                manager._run(run_dir, record, 'secret-key')
            stored = manager.read('test-run')
            self.assertEqual(stored['completed'], 1)
            self.assertEqual(stored['truncated'], 1)
            self.assertEqual(stored['status'], 'completed')
            self.assertNotIn('answer', stored['samples'][0])
            self.assertEqual(manager.sample('test-run', 0)['answer'], 'hello')
            self.assertIsNone(manager.active)
            command, env = spawn.call_args.args
            self.assertNotIn('secret-key', str(command))
            self.assertEqual(env['TEXTGEN_BENCH_API_KEY'], 'secret-key')
            with zipfile.ZipFile(manager.export('test-run')) as archive:
                for name in archive.namelist():
                    self.assertNotIn(b'secret-key', archive.read(name))

    def test_stop_is_scoped_to_current_run_and_export_rejects_active_run(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(directory)
            run_dir = Path(directory) / 'runs' / 'test-run'
            run_dir.mkdir(parents=True)
            manager.active = dict(id='test-run')
            save_json(run_dir / 'summary.json', manager.active)
            manager.stop()
            self.assertTrue((run_dir / 'stop').is_file())
            with self.assertRaises(ValueError):
                manager.export('test-run')
            self.assertIsNone(manager.read('../../outside'))

    def test_error_exit_is_not_reported_as_a_completed_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = BenchmarkManager(directory)
            run_dir = Path(directory) / 'runs' / 'test-run'
            run_dir.mkdir(parents=True)
            record = dict(id='test-run', status='running')
            process = Mock(stdout=io.StringIO('download failed\n'))
            process.wait.return_value = 1
            with patch.object(manager, '_spawn', return_value=process):
                manager._run(run_dir, record, '')
            self.assertEqual(manager.read('test-run')['status'], 'error')


if __name__ == '__main__':
    unittest.main()
