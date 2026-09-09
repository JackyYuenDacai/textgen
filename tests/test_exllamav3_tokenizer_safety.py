"""CPU-only regression checks for shared ExLlamaV3 tokenizer state."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer as NativeTokenizer, models
from exllamav3.tokenizer import Tokenizer


class TokenizerSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        directory = Path(self.directory.name)
        native = NativeTokenizer(models.WordLevel({'[UNK]': 0, 'hello': 1, 't': 2}, unk_token='[UNK]'))
        native.add_special_tokens(['<special>'])
        native.save(str(directory / 'tokenizer.json'))
        (directory / 'tokenizer_config.json').write_text(json.dumps({
            'added_tokens_decoder': {'3': {'content': '<special>', 'special': True}}
        }), encoding='utf-8')
        self.tokenizer = Tokenizer(SimpleNamespace(
            directory=str(directory), bos_token_id=None, eos_token_id=None,
            pad_token_id=None, eos_token_id_list=[],
        ))

    def test_special_mode_is_request_local_and_restored(self):
        tokenizer = self.tokenizer
        for previous in (False, True):
            tokenizer.tokenizer.encode_special_tokens = previous
            self.assertEqual(tokenizer.encode_part_base('<special>', True), [3])
            self.assertEqual(tokenizer.tokenizer.encode_special_tokens, previous)
            self.assertNotIn(3, tokenizer.encode_part_base('<special>', False))
            self.assertEqual(tokenizer.tokenizer.encode_special_tokens, previous)

    def test_mode_restored_after_native_exception(self):
        native = self.tokenizer.tokenizer
        class FailingBackend:
            encode_special_tokens = False

            def encode(self, *args, **kwargs):
                raise ValueError('encoding failed')

        backend = FailingBackend()
        self.tokenizer.tokenizer = backend
        with self.assertRaisesRegex(ValueError, 'encoding failed'):
            self.tokenizer.encode_part_base('hello', False)
        self.assertFalse(backend.encode_special_tokens)
        self.tokenizer.tokenizer = native
        self.assertEqual(self.tokenizer.encode_part_base('<special>', True), [3])

    def test_encode_excludes_competing_encode_decode_and_count(self):
        # Pause inside a real native encode boundary. Competing calls must wait
        # without mutating its mode or touching its native decoder/count path.
        for operation in ('encode', 'decode', 'count'):
            with self.subTest(operation=operation):
                tokenizer = self.tokenizer
                native = tokenizer.tokenizer
                entered, release, competitor_started = (threading.Event() for _ in range(3))

                class PausingBackend:
                    @property
                    def encode_special_tokens(self):
                        return native.encode_special_tokens

                    @encode_special_tokens.setter
                    def encode_special_tokens(self, value):
                        native.encode_special_tokens = value

                    def encode(self, text, **kwargs):
                        if text == '<special>':
                            entered.set()
                            if not release.wait(5):
                                raise TimeoutError('test did not release encode')
                        return native.encode(text, **kwargs)

                    def decode(self, *args, **kwargs):
                        return native.decode(*args, **kwargs)

                tokenizer.tokenizer = PausingBackend()

                def compete():
                    competitor_started.set()
                    if operation == 'encode':
                        return tokenizer.encode_part_base('hello', False)
                    if operation == 'decode':
                        return tokenizer.decode(torch.tensor([1]))
                    return tokenizer.num_tokens('hello')

                try:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(tokenizer.encode_part_base, '<special>', True)
                        try:
                            self.assertTrue(entered.wait(5))
                            second = pool.submit(compete)
                            self.assertTrue(competitor_started.wait(5))
                            with self.assertRaises(TimeoutError):
                                second.result(timeout=0.1)
                        finally:
                            release.set()
                        self.assertEqual(first.result(timeout=5), [3])
                        self.assertEqual(second.result(timeout=5), {
                            'encode': [1], 'decode': 'hello', 'count': 1,
                        }[operation])
                finally:
                    tokenizer.tokenizer = native

    def test_concurrent_modes_match_serial_results(self):
        text = 'hello <special> ' * 128
        expected = {mode: self.tokenizer.encode(text, encode_special_tokens=mode).tolist()
                    for mode in (False, True)}
        modes = [False, True] * 32
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(
                lambda mode: self.tokenizer.encode(text, encode_special_tokens=mode).tolist(), modes,
            ))
        for mode, result in zip(modes, actual):
            self.assertEqual(result, expected[mode])


if __name__ == '__main__':
    unittest.main()
