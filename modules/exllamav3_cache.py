"""Bounded reuse of complete multimodal embeddings, including their token IDs."""
import hashlib
import threading
from collections import OrderedDict


class ImageEmbeddingCache:
    def __init__(self, max_bytes, max_entries=32):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()

    @staticmethod
    def image_key(image, processing_key):
        # Include decoded colors (including a palette/alpha), dimensions and mode.
        # Container metadata and base64 encoding do not affect a cache hit.
        digest = hashlib.sha256()
        digest.update(repr((processing_key, image.mode, image.size)).encode())
        digest.update(image.convert('RGBA').tobytes())
        return digest.digest()

    @staticmethod
    def embedding_bytes(embedding):
        tensors = [embedding.embeddings, getattr(embedding, 'token_string', None)]
        tensors.extend(getattr(embedding, 'deepstack_embeddings', None) or [])
        # Count backing allocations, not just views, and don't count shared storage twice.
        storages = {}
        for tensor in tensors:
            if tensor is not None:
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def get_or_create(self, image, processing_key, create):
        if not self.max_bytes or not self.max_entries:
            return create(), False

        key = self.image_key(image, processing_key)
        # Serialize misses too: simultaneous requests for the same image must reuse IDs.
        with self.lock:
            if key in self.entries:
                self.entries.move_to_end(key)
                return self.entries[key][0], True

            embedding = create()
            size = self.embedding_bytes(embedding)
            if size <= self.max_bytes:
                while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
                    _, (_, evicted_size) = self.entries.popitem(last=False)
                    self.bytes -= evicted_size
                self.entries[key] = (embedding, size)
                self.bytes += size
            return embedding, False

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = 0

    def info(self):
        with self.lock:
            return {'entries': len(self.entries), 'bytes': self.bytes, 'max_bytes': self.max_bytes}
