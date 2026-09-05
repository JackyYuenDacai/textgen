import copy
import queue
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

with patch('sys.argv', ['textgen-tests']):
    from modules import shared
    from modules.exllamav3 import Exllamav3Model
from modules.exllamav3_cache import ImageEmbeddingCache
from modules.prompt_utils import stable_tool_definitions


def embedding():
    return SimpleNamespace(embeddings=torch.zeros(2, 4), token_string=torch.tensor([[1, 2]]),
                           deepstack_embeddings=None, text_alias='<$EMB_1000000000$>')


class ImageCacheTests(unittest.TestCase):
    def test_reuses_complete_object_and_invalidates_on_pixels_or_preprocessing(self):
        cache = ImageEmbeddingCache(1024)
        image = Image.new('RGB', (8, 8), 'red')
        create = Mock(side_effect=embedding)
        first, hit = cache.get_or_create(image, 'model-a', create)
        self.assertFalse(hit)
        second, hit = cache.get_or_create(image.copy(), 'model-a', create)
        self.assertTrue(hit)
        self.assertIs(first, second)
        image.putpixel((0, 0), (0, 0, 0))
        self.assertFalse(cache.get_or_create(image, 'model-a', create)[1])
        self.assertFalse(cache.get_or_create(image, 'different-preprocessing', create)[1])
        self.assertEqual(create.call_count, 3)

    def test_palette_and_dimensions_are_part_of_identity(self):
        cache = ImageEmbeddingCache(1024)
        create = Mock(side_effect=embedding)
        image = Image.new('P', (4, 4))
        image.putpalette([255, 0, 0] + [0] * 765)
        cache.get_or_create(image, None, create)
        image.putpalette([0, 255, 0] + [0] * 765)
        self.assertFalse(cache.get_or_create(image, None, create)[1])
        self.assertFalse(cache.get_or_create(image.resize((2, 8)), None, create)[1])

    def test_lru_is_bounded_and_eviction_does_not_modify_inflight_embedding(self):
        size = ImageEmbeddingCache.embedding_bytes(embedding())
        cache = ImageEmbeddingCache(size * 2)
        images = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue', 'green']]
        create = Mock(side_effect=embedding)
        first, _ = cache.get_or_create(images[0], None, create)
        evicted, _ = cache.get_or_create(images[1], None, create)
        cache.get_or_create(images[0], None, create)
        cache.get_or_create(images[2], None, create)
        self.assertTrue(cache.get_or_create(images[0], None, create)[1])
        self.assertEqual(cache.info()['bytes'], size * 2)
        self.assertFalse(cache.get_or_create(images[1], None, create)[1])
        self.assertEqual(evicted.token_string.tolist(), [[1, 2]])
        self.assertEqual(first.text_alias, '<$EMB_1000000000$>')
        cache.clear()
        self.assertEqual(cache.info()['bytes'], 0)
        self.assertEqual(cache.info()['entries'], 0)

    def test_disabled_oversize_and_failed_encodes_are_not_retained(self):
        image = Image.new('RGB', (2, 2))
        for budget in [0, 1]:
            cache = ImageEmbeddingCache(budget)
            create = Mock(side_effect=embedding)
            for _ in range(2):
                self.assertFalse(cache.get_or_create(image, None, create)[1])
            self.assertEqual(cache.info()['entries'], 0)
            self.assertEqual(create.call_count, 2)
        cache = ImageEmbeddingCache(1024)
        with self.assertRaises(RuntimeError):
            cache.get_or_create(image, None, Mock(side_effect=RuntimeError('encoder failed')))
        self.assertFalse(cache.get_or_create(image, None, embedding)[1])

    def test_simultaneous_requests_encode_once(self):
        cache = ImageEmbeddingCache(1024)
        image = Image.new('RGB', (2, 2))
        create = Mock(side_effect=embedding)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: cache.get_or_create(image, None, create), range(8)))
        self.assertEqual(create.call_count, 1)
        self.assertTrue(all(result[0] is results[0][0] for result in results))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.model = Exllamav3Model()
        self.model.tokenizer = SimpleNamespace(encode=lambda *a, **k: torch.tensor([[1, 2, 3]]))
        self.model.config = SimpleNamespace(eos_token_id_list=[9])
        self.model.generator = SimpleNamespace(max_chunk_size=2048)
        self.model.parallel_generator = Mock()
        self.model._capture_logprobs = Mock()
        self.state = dict(shared.settings, temperature=0, max_new_tokens=4, truncation_length=1024,
                          auto_max_new_tokens=False, add_bos_token=False, ban_eos_token=False,
                          skip_special_tokens=True, seed=42, logprobs=1)
        self.patches = [patch.object(shared, 'is_multimodal', False),
                        patch.object(shared, 'stop_everything', False),
                        patch('modules.exllamav3.Job', return_value=SimpleNamespace(serial_number=17))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def events(self, *events):
        q = queue.Queue()
        for event in events:
            q.put(event)
        self.model.parallel_generator.submit.return_value = q

    def test_final_event_text_tokens_logprobs_and_metrics_are_preserved(self):
        self.events({'text': 'Hello', 'token_ids': torch.tensor([[1, 2]])},
                    {'eos': True, 'text': ' world', 'token_ids': torch.tensor([[3, 4]]),
                     'new_tokens': 4, 'time_generate': 2.0, 'time_prefill': 0.5,
                     'time_enqueued': 0.1, 'cached_tokens': 768, 'prompt_tokens': 1024,
                     'accepted_draft_tokens': 3, 'rejected_draft_tokens': 1})
        self.assertEqual(list(self.model.generate_with_streaming('prompt', self.state)), ['Hello', 'Hello world'])
        self.assertEqual(self.model.last_completion_token_count, 4)
        self.assertEqual(self.model._capture_logprobs.call_count, 2)
        metrics = self.model.get_performance_stats()['recent_requests'][-1]
        self.assertEqual(metrics['decode_tokens_per_second'], 2.0)
        self.assertEqual(metrics['draft_acceptance'], 0.75)
        self.assertEqual(metrics['cached_tokens'], 768)
        self.assertTrue(metrics['completed'])
        self.model.parallel_generator.cancel.assert_called_once()

    def test_one_final_event_and_zero_timing_are_safe(self):
        self.events({'eos': True, 'text': 'x', 'token_ids': torch.tensor([[1]]),
                     'new_tokens': 1, 'time_generate': 0})
        self.assertEqual(self.model.generate('prompt', self.state), 'x')
        stats = self.model.get_performance_stats()['recent_requests'][-1]
        self.assertIsNone(stats['decode_tokens_per_second'])
        self.assertIsNone(stats['draft_acceptance'])

    def test_cancelled_request_does_not_inherit_completed_metrics(self):
        self.events({'eos': True, 'text': 'done', 'new_tokens': 4, 'time_generate': 1})
        self.model.generate('prompt', self.state)
        self.events({'text': 'partial', 'token_ids': torch.tensor([[1, 2]])})
        stream = self.model.generate_with_streaming('prompt', self.state)
        self.assertEqual(next(stream), 'partial')
        stream.close()
        recent = self.model.get_performance_stats()['recent_requests']
        self.assertTrue(recent[-2]['completed'])
        self.assertFalse(recent[-1]['completed'])
        self.assertNotIn('time_generate', recent[-1])
        self.assertEqual(recent[-1]['emitted_tokens'], 2)

    def test_image_wrapper_reuses_alias_and_invalidates_after_processor_change(self):
        self.model.vision_model = SimpleNamespace(
            config=SimpleNamespace(vision_pp=SimpleNamespace(max_pixels=1000)),
            get_image_embeddings=Mock(side_effect=lambda **kwargs: embedding()))
        image = Image.new('RGB', (8, 8), 'red')
        first_prompt, first = self.model._process_images_for_generation('<__media__> question', {'raw_images': [image]})
        metrics = {}
        next_prompt, second = self.model._process_images_for_generation('<__media__> question', {'raw_images': [image.copy()]}, metrics)
        self.assertEqual(first_prompt, next_prompt)
        self.assertIs(first[0], second[0])
        self.assertEqual(metrics['image_cache_hits'], 1)
        self.model.vision_model.config.vision_pp.max_pixels = 500
        self.model._process_images_for_generation('<__media__>', {'raw_images': [image]}, metrics)
        self.assertEqual(metrics['image_cache_misses'], 1)

    def test_invalid_cache_format_fails_before_model_allocation(self):
        for invalid in ['fp8', 'nvfp4', 'q8_0', 'q4_0', 'q1', 'q9', 'q4_q16']:
            with self.assertRaisesRegex(ValueError, 'Unsupported ExLlamaV3 cache type'):
                self.model._cache_settings(invalid)
        self.assertEqual(self.model._cache_settings('Q4_Q8')[1], {'k_bits': 4, 'v_bits': 8})


class PromptTests(unittest.TestCase):
    def test_equivalent_tools_render_same_prefix_without_mutating_input(self):
        from modules.chat import generate_chat_prompt
        tools = [{'type': 'function', 'function': {'name': 'z', 'parameters': {'enum': ['z', 'a']}}},
                 {'function': {'name': 'a', 'description': 'Example'}, 'type': 'function'}]
        original = copy.deepcopy(tools)
        reordered = [dict(reversed(list(t.items()))) for t in reversed(tools)]
        template = '{% for tool in tools %}{{ tool | tojson }}{% endfor %}{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}'
        state = dict(shared.settings, tools=tools, mode='instruct', history={'internal': [], 'visible': []},
                     instruction_template_str=template, chat_template_str=template)
        with patch.object(shared, 'tokenizer', None):
            a = generate_chat_prompt('hello', state)
            b = generate_chat_prompt('hello', dict(state, tools=reordered))
        self.assertEqual(a, b)
        self.assertEqual(tools, original)
        by_name = {t['function']['name']: t for t in stable_tool_definitions(tools)}
        self.assertEqual(by_name['z']['function']['parameters']['enum'], ['z', 'a'])

    def test_context_presets_respect_loaded_limit(self):
        from modules.ui_parameters import context_limit
        with patch.object(shared, 'model', SimpleNamespace(max_tokens=128000)):
            self.assertEqual(context_limit(65536), 65536)
            self.assertEqual(context_limit(), 128000)
        with patch.object(shared, 'model', SimpleNamespace(max_tokens=8192)):
            self.assertEqual(context_limit(32768), 8192)


if __name__ == '__main__':
    unittest.main()
