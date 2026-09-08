import copy
import queue
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from exllamav3.generator.pagetable import tensor_hash_checksum
from exllamav3.tokenizer.mm_embedding import MMEmbedding

with patch('sys.argv', ['textgen-tests']):
    from modules import shared
    from modules.exllamav3 import ConcurrentGenerator, Exllamav3Model
from modules.exllamav3_cache import ImageEmbeddingCache
from modules.prompt_utils import stable_tool_definitions


def embedding(length=2, grid=(1, 2, 4)):
    return MMEmbedding(embeddings=torch.zeros(length, 4), token_string=torch.tensor([[10] + [-1] * length + [11]]),
                       deepstack_embeddings=[torch.ones(length, 4)], grid_thw=grid, mrope_merge_size=2)


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
        first_alias, evicted_tokens = first.text_alias, evicted.token_string.clone()
        cache.get_or_create(images[0], None, create)
        cache.get_or_create(images[2], None, create)
        self.assertTrue(cache.get_or_create(images[0], None, create)[1])
        self.assertEqual(cache.info()['bytes'], size * 2)
        self.assertFalse(cache.get_or_create(images[1], None, create)[1])
        self.assertTrue(torch.equal(evicted.token_string, evicted_tokens))
        self.assertEqual(first.text_alias, first_alias)
        cache.clear()
        self.assertEqual(cache.info()['bytes'], 0)
        self.assertEqual(cache.info()['entries'], 0)
        self.assertEqual(cache.info()['token_identities'], 0)

    def test_evicted_image_keeps_real_exllama_ids_alias_and_prefix_hash(self):
        cache = ImageEmbeddingCache(ImageEmbeddingCache.embedding_bytes(embedding()))
        red, blue = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue']]
        first, _ = cache.get_or_create(red, 'processor', embedding)
        original = first.token_string.clone()
        other, _ = cache.get_or_create(blue, 'processor', embedding)
        second, hit = cache.get_or_create(red, 'processor', embedding)
        self.assertFalse(hit)  # Re-encoded pixels; identity restoration isn't a tensor hit.
        self.assertIsNot(first, second)
        self.assertEqual((first.first_index, first.last_index, first.text_alias),
                         (second.first_index, second.last_index, second.text_alias))
        self.assertTrue(torch.equal(first.token_string, original))
        self.assertTrue(torch.equal(second.token_string, original))
        self.assertEqual(second.token_list, original[0].tolist())
        self.assertNotEqual(other.first_index, first.first_index)
        prefix = torch.tensor([[123, 456]])
        suffix = torch.tensor([[789, 123]])
        old_hash = tensor_hash_checksum(torch.cat([prefix, original, suffix], dim=-1), None)
        new_hash = tensor_hash_checksum(torch.cat([prefix, second.token_string, suffix], dim=-1), None)
        self.assertEqual(old_hash, new_hash)
        self.assertTrue(torch.equal(first.deepstack_embeddings[0], second.deepstack_embeddings[0]))
        self.assertEqual(second.grid_thw, first.grid_thw)
        # Identity records must not keep image/embedding tensors alive after eviction.
        self.assertTrue(all(isinstance(value, (int, str, bytes))
                            for identity in cache.identities.values() for value in vars(identity).values()))

    def test_request_larger_than_budget_does_not_evict_later_hits(self):
        size = ImageEmbeddingCache.embedding_bytes(embedding())
        cache = ImageEmbeddingCache(size * 2)
        images = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue', 'green', 'yellow']]
        # Reproduce the failure: the oldest image misses before still-cached images.
        for image in images:
            cache.get_or_create(image, None, embedding)
        create = Mock(side_effect=lambda image: embedding())
        first = cache.get_or_create_many(images, None, create)
        second = cache.get_or_create_many(images, None, create)
        self.assertEqual([hit for _, hit in first], [False, False, True, True])
        self.assertEqual([hit for _, hit in second], [False, False, True, True])
        self.assertEqual(create.call_count, 4)
        self.assertEqual(cache.info()['bytes'], size * 2)
        for (old, _), (new, _) in zip(first, second):
            self.assertEqual(old.token_list, new.token_list)
        self.assertIs(first[2][0], second[2][0])

    def test_entry_limit_and_duplicate_oversize_images_are_request_aware(self):
        images = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue']]
        for budget, entries in [(1, 32), (0, 32), (10000, 0), (10000, 1)]:
            with self.subTest(budget=budget, entries=entries):
                cache = ImageEmbeddingCache(budget, max_entries=entries)
                create = Mock(side_effect=lambda image: embedding())
                first = cache.get_or_create_many([*images, images[0].copy()], None, create)
                self.assertEqual(create.call_count, 2)
                self.assertIs(first[0][0], first[2][0])
                second = cache.get_or_create_many(images, None, create)
                self.assertEqual(first[0][0].token_list, second[0][0].token_list)
                self.assertEqual(first[1][0].token_list, second[1][0].token_list)
                self.assertLessEqual(cache.info()['bytes'], budget)
                self.assertLessEqual(cache.info()['entries'], entries)

    def test_layout_change_and_clear_never_reuse_incompatible_identity(self):
        cache = ImageEmbeddingCache(0)
        image = Image.new('RGB', (2, 2))
        first, _ = cache.get_or_create(image, None, embedding)
        # Same token count but changed position geometry must also invalidate.
        changed, _ = cache.get_or_create(image, None, lambda: embedding(grid=(1, 4, 2)))
        longer, _ = cache.get_or_create(image, None, lambda: embedding(length=3, grid=(1, 4, 2)))
        self.assertEqual(len({e.first_index for e in [first, changed, longer]}), 3)
        cache.clear()
        fresh, _ = cache.get_or_create(image, None, embedding)
        self.assertNotEqual(first.first_index, fresh.first_index)

    def test_preprocessing_and_pixel_changes_do_not_share_ids(self):
        cache = ImageEmbeddingCache(0)
        image = Image.new('RGB', (2, 2))
        first, _ = cache.get_or_create(image, 'model-a', embedding)
        other_model, _ = cache.get_or_create(image, 'model-b', embedding)
        image.putpixel((0, 0), (255, 0, 0))
        other_image, _ = cache.get_or_create(image, 'model-a', embedding)
        self.assertEqual(len({e.first_index for e in [first, other_model, other_image]}), 3)

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

    def test_concurrent_over_budget_requests_keep_same_ids(self):
        cache = ImageEmbeddingCache(ImageEmbeddingCache.embedding_bytes(embedding()))
        images = [Image.new('RGB', (2, 2), color) for color in ['red', 'blue', 'green']]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: cache.get_or_create_many(images, None, lambda image: embedding()), range(8)))
        baseline = [item.token_list for item, _ in results[0]]
        self.assertTrue(all([item.token_list for item, _ in result] == baseline for result in results))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.model = Exllamav3Model()
        self.model.tokenizer = SimpleNamespace(encode=lambda *a, **k: torch.tensor([[1, 2, 3]]))
        self.model.config = SimpleNamespace(eos_token_id_list=[9])
        self.model.generator = SimpleNamespace(max_chunk_size=2048, max_total_tokens=256000, num_draft_tokens=0)
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

    def test_full_cache_output_reserves_generation_headroom(self):
        for auto_max in (False, True):
            for draft_tokens in (0, 4):
                with self.subTest(auto_max=auto_max, draft_tokens=draft_tokens):
                    self.model.generator.num_draft_tokens = draft_tokens
                    self.model.tokenizer.encode = Mock(return_value=torch.ones((1, 526), dtype=torch.long))
                    self.state.update(truncation_length=256000, max_new_tokens=255474,
                                      auto_max_new_tokens=auto_max)
                    self.events({'eos': True, 'text': '', 'new_tokens': 0})
                    with patch('modules.exllamav3.Job') as job:
                        self.model.generate('prompt', self.state)
                    self.assertEqual(job.call_args.kwargs['input_ids'].shape[-1], 526)
                    self.assertEqual(job.call_args.kwargs['max_new_tokens'], 255473 - draft_tokens)

    def test_request_context_larger_than_cache_is_clamped(self):
        self.model.generator.max_total_tokens = 1024
        self.model.generator.num_draft_tokens = 4
        self.model.tokenizer.encode = Mock(return_value=torch.ones((1, 1500), dtype=torch.long))
        self.state.update(truncation_length=4096, auto_max_new_tokens=True)
        self.events({'eos': True, 'text': '', 'new_tokens': 0})
        with patch('modules.exllamav3.Job') as job:
            self.model.generate('prompt', self.state)
        self.assertEqual(job.call_args.kwargs['input_ids'].shape[-1], 1018)
        self.assertEqual(job.call_args.kwargs['max_new_tokens'], 1)

    def test_small_output_limit_is_preserved(self):
        self.events({'eos': True, 'text': '', 'new_tokens': 0})
        with patch('modules.exllamav3.Job') as job:
            self.model.generate('prompt', self.state)
        self.assertEqual(job.call_args.kwargs['max_new_tokens'], 4)

    def test_cache_without_room_for_generation_fails_before_submit(self):
        self.model.generator.max_total_tokens = 6
        self.model.generator.num_draft_tokens = 4
        with self.assertRaisesRegex(ValueError, 'generation headroom'):
            self.model.generate('prompt', self.state)
        self.model.parallel_generator.submit.assert_not_called()

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

    def test_concurrent_generator_cancel_does_not_wait_for_prefill(self):
        """A stop request must not deadlock behind a long native iterate call."""
        import threading
        import time

        class BlockingGenerator:
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self.cancelled = threading.Event()

            def enqueue(self, job):
                self.job = job

            def iterate(self):
                self.started.set()
                self.release.wait(timeout=2)
                return []

            def cancel(self, job):
                self.cancelled.set()

            def clear_queue(self):
                pass

        native = BlockingGenerator()
        concurrent = ConcurrentGenerator(native)
        self.addCleanup(concurrent.stop)
        job = object()
        concurrent.submit(job)
        self.assertTrue(native.started.wait(timeout=1))

        started = time.monotonic()
        concurrent.cancel(job)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.2)

        native.release.set()
        self.assertTrue(native.cancelled.wait(timeout=1))

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

    def test_image_wrapper_preserves_full_prompt_after_over_budget_history(self):
        self.model.image_cache = ImageEmbeddingCache(ImageEmbeddingCache.embedding_bytes(embedding()) * 2)
        self.model.vision_model = SimpleNamespace(
            config=SimpleNamespace(vision_pp=SimpleNamespace(max_pixels=1000)),
            get_image_embeddings=Mock(side_effect=lambda **kwargs: embedding()))
        images = [Image.new('RGB', (8, 8), color) for color in ['red', 'green', 'blue']]
        metrics = {}
        prompt = 'system <__media__> first <__media__> second <__media__> question'
        first_prompt, first = self.model._process_images_for_generation(prompt, {'raw_images': images}, metrics)
        second_prompt, second = self.model._process_images_for_generation(prompt, {'raw_images': images}, metrics)
        self.assertEqual(first_prompt, second_prompt)
        self.assertEqual(metrics, {'image_cache_hits': 2, 'image_cache_misses': 1})
        self.assertEqual([e.token_list for e in first], [e.token_list for e in second])
        self.assertIsNot(first[2], second[2])

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
