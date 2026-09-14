"""Isolate native tokenizer encoding from the model server on Windows.

Only CPU token IDs cross this boundary. A native access violation terminates
the helper, fails the current request, and permits a fresh helper next time.
"""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class TokenizerWorkerError(RuntimeError):
    pass


class EncodedIds:
    """The encoding interface consumed by ExLlamaV3 (ids and token count)."""
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)


class IsolatedTokenizerEncoder:
    def __init__(self, tokenizer, timeout=120):
        self._tokenizer = tokenizer
        self._definition = tokenizer.to_str()
        self._timeout = timeout
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tokenizer-ipc')
        self._process = None
        self._closed = False
        atexit.register(self.close)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    @property
    def encode_special_tokens(self):
        return self._tokenizer.encode_special_tokens

    @encode_special_tokens.setter
    def encode_special_tokens(self, value):
        self._tokenizer.encode_special_tokens = value

    def _start(self):
        environment = dict(os.environ, TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='-1')
        self._process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )

    def _exchange(self, request, initialize):
        if initialize:
            request['tokenizer'] = self._definition
        process = self._process
        process.stdin.write(json.dumps(request, ensure_ascii=True).encode('utf-8') + b'\n')
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            code = process.wait()
            raise TokenizerWorkerError(
                f'Native tokenizer worker exited (code 0x{code & 0xffffffff:08X}). '
                'The request failed; TextGen is still running. The next request will start a new worker.')
        result = json.loads(line)
        if 'error' in result:
            raise TokenizerWorkerError('Native tokenizer encoding failed: ' + result['error'])
        return EncodedIds(result['ids'])

    def encode(self, text, add_special_tokens=True):
        with self._lock:
            if self._closed:
                raise TokenizerWorkerError('Tokenizer is unloaded.')
            initialize = self._process is None or self._process.poll() is not None
            if initialize:
                self._stop()
                self._start()
            pending = self._executor.submit(self._exchange, {
                'text': text, 'add_special_tokens': add_special_tokens,
                'encode_special_tokens': self.encode_special_tokens,
            }, initialize)
            try:
                return pending.result(timeout=self._timeout)
            except Exception as error:
                # Kill first to unblock a pending pipe write/read, then join
                # the I/O task before allowing another call to reuse state.
                self._stop(kill_only=True)
                try:
                    pending.result()
                except Exception:
                    pass
                self._stop()
                if isinstance(error, TimeoutError):
                    raise TokenizerWorkerError(
                        f'Native tokenizer exceeded {self._timeout:g}s; its worker was stopped.') from None
                if isinstance(error, TokenizerWorkerError):
                    raise
                raise TokenizerWorkerError('Native tokenizer worker communication failed.') from error

    def _stop(self, kill_only=False):
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        process.wait()
        if not kill_only:
            process.stdin.close()
            process.stdout.close()
            self._process = None

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop()
            self._executor.shutdown(wait=True)
        atexit.unregister(self.close)


def worker():
    # Never import TextGen, Torch, or a model in this helper.
    from tokenizers import Tokenizer

    tokenizer = None
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            if tokenizer is None:
                tokenizer = Tokenizer.from_str(request['tokenizer'])
            tokenizer.encode_special_tokens = request['encode_special_tokens']
            ids = tokenizer.encode(request['text'], add_special_tokens=request['add_special_tokens']).ids
            result = {'ids': ids}
        except Exception as error:
            # Avoid logging user prompts in exception messages.
            result = {'error': type(error).__name__}
        sys.stdout.buffer.write(json.dumps(result).encode('utf-8') + b'\n')
        sys.stdout.buffer.flush()


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    worker()
