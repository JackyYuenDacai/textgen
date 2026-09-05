import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.exllamav3_drafting import drafting_options


def modern_generator(dynamic_draft_tokens=False, draft_confidence=0.4):
    pass


class DraftingTests(unittest.TestCase):
    def args(self, **changes):
        return SimpleNamespace(**dict(dict(exl3_dynamic_draft=True, exl3_draft_confidence=0.4), **changes))

    def test_passes_enabled_settings(self):
        self.assertEqual(drafting_options(self.args(), modern_generator),
                         dict(dynamic_draft_tokens=True, draft_confidence=0.4))

    def test_off_preserves_older_backend_compatibility(self):
        self.assertEqual(drafting_options(self.args(exl3_dynamic_draft=False), lambda: None), {})

    def test_unsupported_backend_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'does not support'):
            drafting_options(self.args(), lambda **kwargs: None)

    def test_invalid_settings_are_rejected(self):
        for value in (0, 1, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                drafting_options(self.args(exl3_draft_confidence=value), modern_generator)
        with self.assertRaises(ValueError):
            drafting_options(self.args(exl3_dynamic_draft='false'), modern_generator)

    def test_controls_are_persisted_and_exl3_only(self):
        with patch('sys.argv', ['textgen-tests']):
            from modules import loaders, shared
        for key in ('exl3_dynamic_draft', 'exl3_draft_confidence'):
            self.assertIn(key, loaders.list_model_elements())
            self.assertIn(key, loaders.loaders_and_params['ExLlamav3'])
            self.assertNotIn(key, loaders.loaders_and_params['llama.cpp'])
            self.assertTrue(hasattr(shared.args, key))

    def test_benchmark_aggregate_is_token_weighted(self):
        from scripts.benchmark_drafting import summarize
        result = summarize([
            dict(new_tokens=100, emitted_tokens=100, time_generate=1, request_seconds=2),
            dict(new_tokens=300, emitted_tokens=300, time_generate=6, request_seconds=8),
        ])
        self.assertAlmostEqual(result['decode_tps'], 400 / 7)
        self.assertEqual(result['end_to_end_tps'], 40)


if __name__ == '__main__':
    unittest.main()
