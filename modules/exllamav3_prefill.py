"""Bound attention workspaces without changing a job's KV allocation or output budget."""


class PrefillJobMixin:
    prefill_page_size = 256

    def prefill(self, results):
        if self.embeddings and self.generator.model.caps.get('atomic_mm_prefill'):
            return super().prefill(results)

        original_tables = []
        try:
            for sequence in self.sequences:
                if sequence.prefill_complete:
                    continue
                table = sequence.block_index_tensor
                if table is None:
                    continue
                end = min(sequence.kv_position + self.generator.max_chunk_size, len(sequence.sequence_ids) - 1)
                pages = max(1, (end + self.prefill_page_size - 1) // self.prefill_page_size)
                if pages < table.shape[-1]:
                    original_tables.append((sequence, table))
                    sequence.block_index_tensor = table[:, :pages]
            return super().prefill(results)
        finally:
            for sequence, table in original_tables:
                sequence.block_index_tensor = table
