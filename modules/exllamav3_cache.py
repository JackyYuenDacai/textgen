"""Bounded embedding tensors with model-lifetime image token identities."""
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageTokenIdentity:
    first_index: int
    last_index: int
    text_alias: str
    layout: bytes


class ImageEmbeddingCache:
    def __init__(self, max_bytes, max_entries=32):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max_entries
        self.entries = OrderedDict()
        # Keep only small scalar/hash records after tensor eviction. Never recycle IDs:
        # old prompt KV pages or in-flight jobs may still refer to them until unload.
        self.identities = {}
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
        return self.get_or_create_many([image], processing_key, lambda _: create())[0]

    def _restore_identity(self, key, embedding):
        first, last = embedding.first_index, embedding.last_index
        # Normalize only the synthetic IDs, preserving delimiters, ordering and MRoPE
        # geometry. A different encoder layout must not reuse incompatible KV pages.
        tokens = embedding.token_string[0].tolist()
        normalized = tuple(-(token - first + 1) if first <= token < last else token for token in tokens)
        layout = hashlib.sha256(repr((embedding.mm_length, embedding.full_length,
                                     embedding.grid_thw, embedding.mrope_merge_size, normalized)).encode()).digest()
        identity = self.identities.get(key)
        if identity is not None and identity.layout == layout:
            # Only touch the newly encoded object's token tensor. An evicted object
            # can still be in use by generation, including its deepstack tensors.
            token_string = embedding.token_string.clone()
            mask = (token_string >= first) & (token_string < last)
            token_string[mask] += identity.first_index - first
            embedding.token_string = token_string
            embedding.token_list = token_string[0].tolist()
            embedding.first_index = identity.first_index
            embedding.last_index = identity.last_index
            embedding.text_alias = identity.text_alias
        else:
            self.identities[key] = ImageTokenIdentity(first, last, embedding.text_alias, layout)

    def get_or_create_many(self, images, processing_key, create):
        """Resolve a whole request without evicting any of its resident images.

        The returned objects belong to the request even when its working set exceeds
        the retained-cache budget. Duplicate images encode once per request. A hit
        means tensor reuse, not merely restoration of synthetic token IDs.
        """
        images = list(images)
        keys = [self.image_key(image, processing_key) for image in images]
        protected = set(keys)
        # Serialize misses too: simultaneous requests for the same image reuse IDs.
        with self.lock:
            resolved = {}
            results = []
            for image, key in zip(images, keys):
                if key in resolved:
                    results.append((resolved[key], True))
                    continue
                if key in self.entries:
                    self.entries.move_to_end(key)
                    embedding = self.entries[key][0]
                    resolved[key] = embedding
                    results.append((embedding, True))
                    continue

                embedding = create(image)
                self._restore_identity(key, embedding)
                resolved[key] = embedding
                results.append((embedding, False))
                size = self.embedding_bytes(embedding)
                if self.max_bytes and self.max_entries > 0 and size <= self.max_bytes:
                    while self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries:
                        victim = next((entry for entry in self.entries if entry not in protected), None)
                        if victim is None:
                            break  # Needed by this request: retain it, don't churn the LRU.
                        _, evicted_size = self.entries.pop(victim)
                        self.bytes -= evicted_size
                    else:
                        self.entries[key] = (embedding, size)
                        self.bytes += size
            return results

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.identities.clear()
            self.bytes = 0

    def info(self):
        with self.lock:
            return {'entries': len(self.entries), 'bytes': self.bytes, 'max_bytes': self.max_bytes,
                    'token_identities': len(self.identities)}
