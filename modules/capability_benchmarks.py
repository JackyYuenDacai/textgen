"""Capability evaluation process management; no eval dependencies in TextGen's interpreter."""

import json
import math
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


TASKS = {
    'gsm8k': 'GSM8K · mathematical reasoning',
    'humaneval': 'HumanEval · Python programming (Docker)',
    'ifeval': 'IFEval · instruction following',
}
PACKAGES = [
    'inspect-ai==0.3.263', 'openai==3.1.0', 'inspect-evals[ifeval]==0.19.0',
    'instruction-following-eval @ git+https://github.com/josejg/instruction_following_eval.git@0c495b2f95155e8b10acb919ae283bfb4d5be6e2',
]
EVENT_PREFIX = '@@TEXTGEN_BENCHMARK '
WORKER = Path(__file__).resolve().parents[1] / 'scripts' / 'capability_benchmark_worker.py'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_config(task, endpoint, model, limit, max_tokens, temperature, seed, thinking):
    if task not in TASKS:
        raise ValueError('Choose a supported benchmark.')
    endpoint = endpoint.strip().rstrip('/')
    url = urlsplit(endpoint)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) API base URL without credentials, query parameters, or fragments.')
    if not model.strip() or model.strip() == 'None':
        raise ValueError('Enter the model ID served by the endpoint.')
    values = {}
    for name, value, low, high in [('limit', limit, 1, 10000), ('max_tokens', max_tokens, 128, 65536), ('seed', seed, 0, 2147483647)]:
        if not math.isfinite(float(value)) or int(value) != float(value) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer between {low} and {high}.')
        values[name] = int(value)
    temperature = float(temperature)
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('Temperature must be between 0 and 2.')
    if thinking not in ('Server default', 'Enabled', 'Disabled'):
        raise ValueError('Invalid thinking option.')
    return dict(task=task, endpoint=endpoint, model=model.strip(), temperature=temperature, thinking=thinking, **values)


class BenchmarkManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.active = None
        self.setup_running = False
        self.console = deque(maxlen=80)

    @property
    def python(self):
        return self.root / 'env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    @property
    def ready(self):
        return self.python.is_file() and (self.root / 'ready.json').is_file()

    def _spawn(self, command, env):
        process = subprocess.Popen(command, cwd=WORKER.parents[1], env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace',
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        from modules.windows_subprocess import bind_to_parent_lifetime
        bind_to_parent_lifetime(process.pid)
        return process

    def _env(self):
        env = os.environ.copy()
        # Keep eval caches and optional NLTK downloads with the benchmark installation.
        env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HOME=str(self.root / 'cache' / 'huggingface'),
                   NLTK_DATA=str(self.root / 'cache' / 'nltk'), INSPECT_LOG_DIR=str(self.root / 'logs'))
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        return env

    def install(self):
        with self.lock:
            if self.active or self.setup_running:
                return 'A benchmark or installation is already running.'
            self.root.mkdir(parents=True, exist_ok=True)
            self.setup_running = True
            self.console.clear()
            threading.Thread(target=self._install, daemon=True).start()
        return 'Installing the isolated benchmark environment. Follow setup output below.'

    def _install(self):
        try:
            commands = []
            if not self.python.exists():
                commands.append([sys.executable, '-X', 'utf8', '-m', 'venv', str(self.root / 'env')])
            commands.extend([
                [str(self.python), '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', *PACKAGES],
                [str(self.python), str(WORKER), '--check'],
            ])
            for command in commands:
                process = self._spawn(command, self._env())
                for line in process.stdout:
                    with self.lock:
                        self.console.append(line.rstrip())
                if process.wait() != 0:
                    raise RuntimeError('Setup failed. See the output below; fix the error and retry installation.')
            save_json(self.root / 'ready.json', {'packages': PACKAGES, 'installed': timestamp()})
            with self.lock:
                self.console.append('Benchmark environment ready.')
        except Exception as error:
            with self.lock:
                self.console.append(str(error))
        finally:
            with self.lock:
                self.setup_running = False

    def start(self, config, api_key):
        with self.lock:
            if self.active or self.setup_running:
                raise ValueError('A benchmark or installation is already running.')
            if not self.ready:
                raise ValueError('Install the benchmark environment first.')
            # IDs and filesystem paths are generated locally, never supplied by the browser.
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
            run_dir = self.root / 'runs' / run_id
            run_dir.mkdir(parents=True)
            record = dict(id=run_id, status='starting', started=timestamp(), config=config,
                          completed=0, errors=0, truncated=0, total=config['limit'], metrics=[], samples=[])
            save_json(run_dir / 'config.json', config)
            save_json(run_dir / 'summary.json', record)
            self.active = record
            self.console.clear()
            threading.Thread(target=self._run, args=(run_dir, record, api_key), daemon=True).start()
        return f'Started {TASKS[config["task"]]}. Run {run_id}.'

    def _run(self, run_dir, record, api_key):
        env = self._env()
        env['TEXTGEN_BENCH_API_KEY'] = api_key or 'textgen-local'
        try:
            process = self._spawn([str(self.python), str(WORKER), str(run_dir)], env)
            with (run_dir / 'console.log').open('w', encoding='utf-8') as output:
                for line in process.stdout:
                    line = line.replace(api_key, '[REDACTED]') if api_key else line
                    output.write(line)
                    output.flush()
                    with self.lock:
                        if line.startswith(EVENT_PREFIX):
                            event = json.loads(line[len(EVENT_PREFIX):])
                            if event.get('kind') == 'sample':
                                sample_dir = run_dir / 'samples'
                                sample_dir.mkdir(exist_ok=True)
                                save_json(sample_dir / f'{len(record["samples"]):06d}.json', event['sample'])
                                record['samples'].append({key: event['sample'][key] for key in ('id', 'error', 'truncated')})
                                record['completed'] += 1
                                record['errors'] += bool(event['sample']['error'])
                                record['truncated'] += bool(event['sample']['truncated'])
                            else:
                                record.update({k: v for k, v in event.items() if k != 'kind'})
                            save_json(run_dir / 'summary.json', record)
                        else:
                            self.console.append(line.rstrip())
            code = process.wait()
            with self.lock:
                if record['status'] not in ('completed', 'cancelled', 'error'):
                    record['status'] = 'error'
                    record['message'] = f'Worker exited ({code}). See console output for details.'
        except Exception as error:
            with self.lock:
                record['status'] = 'error'
                record['message'] = str(error).replace(api_key, '[REDACTED]') if api_key else str(error)
        finally:
            with self.lock:
                record['finished'] = timestamp()
                save_json(run_dir / 'summary.json', record)
                self.active = None

    def stop(self):
        with self.lock:
            if not self.active:
                return 'No benchmark is running.'
            (self.root / 'runs' / self.active['id'] / 'stop').touch()
        return 'Stop requested. The worker will cancel evaluation and clean up its sandboxes; dataset loading may need to finish first.'

    def runs(self):
        return sorted((self.root / 'runs').glob('*/summary.json'), reverse=True)[:100]

    def read(self, run_id=None):
        if run_id:
            candidates = [p for p in self.runs() if p.parent.name == run_id]
        else:
            candidates = self.runs()[:1]
        if candidates:
            try:
                return json.loads(candidates[0].read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
        return None

    def export(self, run_id):
        import zipfile

        record = self.read(run_id)
        if not record:
            raise ValueError('Select a saved run first.')
        with self.lock:
            if self.active and self.active['id'] == record['id']:
                raise ValueError('Wait for the run to finish or stop it before exporting.')
        run_dir = self.root / 'runs' / record['id']
        # Only include this run's data, never environment variables or credentials.
        destination = self.root / 'exports' / f'{record["id"]}.zip'
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in run_dir.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.eval', '.log'):
                    archive.write(path, path.relative_to(run_dir))
        return str(destination)

    def sample(self, run_id, index):
        record = self.read(run_id)
        if not record or not 0 <= index < len(record.get('samples', [])):
            return None
        path = self.root / 'runs' / record['id'] / 'samples' / f'{index:06d}.json'
        return json.loads(path.read_text(encoding='utf-8'))
