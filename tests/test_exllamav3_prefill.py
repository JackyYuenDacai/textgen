import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from exllamav3.generator import Job as NativeJob
from exllamav3.util.tensor import SeqTensor

from modules.exllamav3_prefill import PrefillJobMixin


class RecordingJob:
    def prefill(self, results):
        self.inspect(results)
        return 'prefilled'


class BoundedJob(PrefillJobMixin, RecordingJob):
    pass


class NativeBoundedJob(PrefillJobMixin, NativeJob):
    pass


def sequence(position=0, length=139935, pages=1000, complete=False):
    return SimpleNamespace(kv_position=position, sequence_ids=range(length), prefill_complete=complete,
                           block_index_tensor=torch.arange(pages, dtype=torch.int32).view(1, -1))


class PrefillWorkspaceTests(unittest.TestCase):
    def make_job(self, sequences, embeddings=None, atomic=False, chunk_size=2048):
        job = BoundedJob()
        job.sequences = sequences
        job.embeddings = embeddings
        job.generator = SimpleNamespace(max_chunk_size=chunk_size, model=SimpleNamespace(caps={'atomic_mm_prefill': atomic}))
        job.inspect = Mock()
        return job

    def test_first_chunk_does_not_expose_future_output_pages(self):
        current = sequence()
        original = current.block_index_tensor
        job = self.make_job([current])
        def inspect(results):
            self.assertEqual(current.block_index_tensor.shape, (1, 8))
            self.assertTrue(torch.equal(current.block_index_tensor, original[:, :8]))
            self.assertEqual(len(current.sequence_ids), 139935)
        job.inspect.side_effect = inspect
        self.assertEqual(job.prefill([]), 'prefilled')
        self.assertIs(current.block_index_tensor, original)
        full_scratch = 2 * 1024 * 256 * 4 * 256 * 2
        chunk_scratch = 2 * 8 * 256 * 4 * 256 * 2
        self.assertEqual(full_scratch // chunk_scratch, 128)

    def test_long_prefix_tail_and_replay_use_enough_pages(self):
        for position, length, expected_pages in ((0, 526, 3), (139776, 139935, 547),
                                                  (255, 10000, 9), (4096, 8193, 24), (0, 1, 1)):
            with self.subTest(position=position, length=length):
                current = sequence(position=position, length=length)
                original = current.block_index_tensor
                job = self.make_job([current])
                job.inspect.side_effect = lambda results: self.assertEqual(current.block_index_tensor.shape[-1], expected_pages)
                job.prefill([])
                self.assertIs(current.block_index_tensor, original)

    def test_multimodal_atomic_chunks_keep_the_full_table(self):
        current = sequence()
        original = current.block_index_tensor
        job = self.make_job([current], embeddings=[object()], atomic=True)
        job.inspect.side_effect = lambda results: self.assertIs(current.block_index_tensor, original)
        job.prefill([])

    def test_non_atomic_multimodal_and_plain_atomic_model_can_use_bounds(self):
        for embeddings, atomic in (([object()], False), (None, True)):
            with self.subTest(embeddings=embeddings, atomic=atomic):
                current = sequence()
                job = self.make_job([current], embeddings=embeddings, atomic=atomic)
                job.inspect.side_effect = lambda results: self.assertEqual(current.block_index_tensor.shape[-1], 8)
                job.prefill([])

    def test_completed_missing_and_short_tables_are_left_alone(self):
        completed, missing, short = sequence(complete=True), sequence(), sequence(pages=4)
        missing.block_index_tensor = None
        originals = [item.block_index_tensor for item in (completed, missing, short)]
        job = self.make_job([completed, missing, short])
        def inspect(results):
            for item, original in zip(job.sequences, originals):
                self.assertIs(item.block_index_tensor, original)
        job.inspect.side_effect = inspect
        job.prefill([])

    def test_tables_are_restored_after_failure_for_all_sequences(self):
        sequences = [sequence(), sequence(position=4096)]
        originals = [item.block_index_tensor for item in sequences]
        job = self.make_job(sequences)
        job.inspect.side_effect = RuntimeError('prefill failed')
        with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
            job.prefill([])
        for item, original in zip(sequences, originals):
            self.assertIs(item.block_index_tensor, original)

    def make_native_job(self, job_class, mtp=False, recurrent=False):
        job = object.__new__(job_class)
        input_ids = torch.arange(5001).view(1, -1)
        current = sequence(length=5001, pages=1000)
        current.sequence_ids = SeqTensor.from_tensor(input_ids, -1)
        current.allocated_pages = [SimpleNamespace(page_index=page, kv_position=0, phash=page, prev_hash=None,
                                                   sequence=torch.zeros((1, 256), dtype=torch.long), can_revert=False)
                                   for page in range(1000)]
        current.mtp_carry_hidden = None
        job.sequences = [current]
        job.time_first_prefill = None
        job.recurrent_state = SimpleNamespace(position=0) if recurrent else None
        job.embeddings = None
        job.cached_pages = 0
        job.cached_tokens = 0
        job.pagetable = SimpleNamespace(all_pages=[])
        job.alt_rope_freqs = None
        job.identifier = None
        job.serial_number = 7
        job.mtp_last_hidden = None
        job.maybe_stash_recurrent = Mock()
        writes = []
        widths = []
        draft_calls = []
        def forward(input_ids, params):
            table = params['block_table']
            start = int(params['cache_seqlens'][0])
            positions = torch.arange(start, start + input_ids.shape[-1])
            physical_pages = table[0, positions // 256]
            writes.append((start, input_ids.clone(), physical_pages.clone()))
            widths.append(table.shape[-1])
            if recurrent:
                job.recurrent_state.position = start + input_ids.shape[-1]
            if mtp:
                params['export_states'] = [input_ids.float().unsqueeze(-1)]
        def draft_prefill(input_ids, params):
            draft_calls.append((input_ids.clone(), params['target_hidden'].clone(), params['block_table'].shape[-1]))
        model = SimpleNamespace(caps={}, prefill=forward, forward=forward)
        draft = SimpleNamespace(draft_verifier_params={}, prefill=draft_prefill) if mtp else None
        job.generator = SimpleNamespace(max_chunk_size=2048, model=model, draft_model=draft,
                                        recurrent_cache=object() if recurrent else None, cache=object(),
                                        draft_cache=object(), mtp_draft=mtp, dflash_draft=False)
        return job, writes, widths, draft_calls

    def test_native_prefill_preserves_tokens_cache_writes_progress_and_mtp(self):
        for mtp, recurrent in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(mtp=mtp, recurrent=recurrent):
                original, original_writes, original_widths, original_draft = self.make_native_job(NativeJob, mtp, recurrent)
                bounded, bounded_writes, bounded_widths, bounded_draft = self.make_native_job(NativeBoundedJob, mtp, recurrent)
                original_progress, bounded_progress = [], []
                saved_table = bounded.sequences[0].block_index_tensor
                for job, progress in ((original, original_progress), (bounded, bounded_progress)):
                    for iteration in range(10):
                        job.prefill(progress)
                        if job.sequences[0].prefill_complete:
                            break
                    self.assertTrue(job.sequences[0].prefill_complete)
                self.assertIs(bounded.sequences[0].block_index_tensor, saved_table)
                self.assertEqual([entry['curr_progress'] for entry in original_progress],
                                 [entry['curr_progress'] for entry in bounded_progress])
                self.assertEqual(len(original_writes), len(bounded_writes))
                for expected, actual in zip(original_writes, bounded_writes):
                    self.assertEqual(expected[0], actual[0])
                    self.assertTrue(torch.equal(expected[1], actual[1]))
                    self.assertTrue(torch.equal(expected[2], actual[2]))
                self.assertLess(max(bounded_widths), min(original_widths))
                self.assertEqual(len(original_draft), len(bounded_draft))
                for expected, actual in zip(original_draft, bounded_draft):
                    self.assertTrue(torch.equal(expected[0], actual[0]))
                    self.assertTrue(torch.equal(expected[1], actual[1]))
                for expected, actual in zip(original.sequences[0].allocated_pages, bounded.sequences[0].allocated_pages):
                    self.assertEqual(expected.kv_position, actual.kv_position)
                    self.assertTrue(torch.equal(expected.sequence, actual.sequence))


if __name__ == '__main__':
    unittest.main()
