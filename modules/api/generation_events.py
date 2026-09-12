"""Semantic events shared by protocol adapters (no wire envelopes)."""
from dataclasses import dataclass


class GenerationEvent:
    pass


@dataclass
class StartedEvent(GenerationEvent):
    model: str


@dataclass
class TextDelta(GenerationEvent):
    text: str


@dataclass
class ReasoningDelta(GenerationEvent):
    text: str


@dataclass
class ToolCallDelta(GenerationEvent):
    call_id: str
    name: str
    arguments_delta: str
    index: int = 0


@dataclass
class DoneEvent(GenerationEvent):
    finish_reason: str


@dataclass
class UsageEvent(GenerationEvent):
    usage: dict


@dataclass
class EventBatch:
    """One backend step; preserves token/logprob boundaries for serializers."""
    events: list[GenerationEvent]
    logprobs: dict | None = None


class ChatSerializer:
    def __init__(self, stream=False, is_legacy=False, include_usage=False):
        import time
        self.stream = stream
        self.include_usage = include_usage
        self.key = 'data' if is_legacy else 'choices'
        self.base = dict(id='chatcmpl-%d' % time.time_ns(),
                         object='chat.completion.chunk' if stream else 'chat.completion',
                         created=int(time.time()), model=None, system_fingerprint=None)

    def process(self, batch):
        delta, finish, usage = {}, None, None
        for event in batch.events:
            if isinstance(event, StartedEvent):
                self.base['model'] = event.model
                delta.update(role='assistant', refusal=None)
            elif isinstance(event, TextDelta):
                delta['content'] = delta.get('content', '') + event.text
            elif isinstance(event, ReasoningDelta):
                delta['reasoning_content'] = delta.get('reasoning_content', '') + event.text
            elif isinstance(event, ToolCallDelta):
                call = {'id': event.call_id, 'type': 'function',
                        'function': {'name': event.name, 'arguments': event.arguments_delta}}
                if self.stream:
                    call['index'] = event.index
                delta.setdefault('tool_calls', []).append(call)
            elif isinstance(event, DoneEvent):
                finish = event.finish_reason
            elif isinstance(event, UsageEvent):
                usage = event.usage
        if self.stream and usage is not None and not delta and finish is None:
            return [{**self.base, self.key: [], 'usage': usage}] if self.include_usage else []
        if not self.stream:
            delta.setdefault('content', None)
        result = {**self.base, self.key: [dict(index=0, finish_reason=finish,
                   **{'delta' if self.stream else 'message': delta}, logprobs=batch.logprobs)]}
        if not self.stream or self.include_usage:
            result['usage'] = usage
        return [result]
