"""GPU-free tests of the actual pre-delivery recovery coordinator."""
import copy
import os
import threading
import unittest
from unittest.mock import patch

from modules.api.canonical_items import from_chat_messages
from modules.api.generation_events import EventBatch, TextDelta, ToolCallDelta, DoneEvent, UsageEvent
from modules.api.generation_recovery import recover, malformed_output, progress_only
from modules.api.responses_tools import ToolOutputError
from modules.tool_parsing import parse_tool_call

CALL = '<tool_call>{"name":"fetch","arguments":{}}</tool_call>'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, TEXTGEN_TOOL_REPAIR_ATTEMPTS='2', TEXTGEN_REPAIR_PROGRESS='1')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.body = {'items': from_chat_messages([{'role': 'user', 'content': 'Fetch both pages.'}]),
                     'tools': [{'type': 'function', 'function': {'name': 'fetch'}}]}
        self.seen = []
        self.closed = []

    def source(self, candidates, budget=100, cancel=None):
        def generate(body, **kwargs):
            index = len(self.seen)
            self.seen.append(copy.deepcopy(body))
            candidate = candidates[index]
            body['_generation_attempt'].update(raw=candidate.get('raw', candidate.get('text', '')),
                error=candidate.get('error'), output_budget=budget)
            try:
                events = [TextDelta(candidate.get('text', ''))]
                if candidate.get('call'):
                    events.append(ToolCallDelta('call_' + str(index), 'fetch', candidate.get('arguments', '{}')))
                events.extend([DoneEvent(candidate.get('finish', 'tool_calls' if candidate.get('call') else 'stop')),
                               UsageEvent({'prompt_tokens': 10, 'completion_tokens': candidate.get('tokens', 5),
                                           'total_tokens': 10 + candidate.get('tokens', 5)})])
                yield EventBatch(events)
                if cancel:
                    cancel.set()
            finally:
                self.closed.append(index)
        return generate

    def events(self, source, **kwargs):
        return [e for b in recover(source, self.body, **kwargs) for e in b.events]

    def test_bad_call_repaired_without_leaking_and_usage_aggregated(self):
        before = copy.deepcopy(self.body)
        events = self.events(self.source([{'raw': '<tool_call>{', 'text': 'Fetching:', 'error': 'bad'}, {'call': True}]))
        self.assertEqual([e.call_id for e in events if isinstance(e, ToolCallDelta)], ['call_1'])
        self.assertNotIn('Fetching:', ''.join(e.text for e in events if isinstance(e, TextDelta)))
        usage = next(e.usage for e in events if isinstance(e, UsageEvent))
        self.assertEqual(usage, {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30})
        self.assertEqual(self.seen[1]['_generation_output_ceiling'], 95)
        self.assertEqual(self.body, before)
        self.assertEqual(self.closed, [0, 1])

    def test_progress_only_repaired(self):
        events = self.events(self.source([{'text': 'I will now process the remaining batches.'}, {'call': True}]))
        self.assertTrue(any(isinstance(e, ToolCallDelta) for e in events))
        self.assertEqual(len(self.seen), 2)

    def test_complete_answer_not_retried(self):
        self.events(self.source([{'text': 'Both pages have been fetched.'}]))
        self.assertEqual(len(self.seen), 1)

    def test_blocker_not_retried(self):
        self.events(self.source([{'text': 'I cannot access the file. Please provide it.'}]))
        self.assertEqual(len(self.seen), 1)

    def test_repeat_stops_early(self):
        with self.assertRaisesRegex(ToolOutputError, 'after 1 repair'):
            self.events(self.source([{'raw': 'x', 'error': 'bad'}] * 2))
        self.assertEqual(self.closed, [0, 1])

    def test_attempt_cap(self):
        with self.assertRaisesRegex(ToolOutputError, 'after 2 repair'):
            self.events(self.source([{'raw': str(i), 'error': 'bad'} for i in range(3)]))
        self.assertEqual(len(self.seen), 3)

    def test_exhausted_budget_never_retries_or_delivers_calls(self):
        events = self.events(self.source([{'call': True, 'error': 'bad', 'tokens': 10}], budget=10))
        self.assertFalse(any(isinstance(e, ToolCallDelta) for e in events))
        self.assertEqual(next(e.finish_reason for e in events if isinstance(e, DoneEvent)), 'length')
        self.assertEqual(len(self.seen), 1)

    def test_backend_length_never_retries(self):
        events = self.events(self.source([{'error': 'bad', 'finish': 'length'}]))
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(next(e.finish_reason for e in events if isinstance(e, DoneEvent)), 'length')

    def test_cancel_before_attempt(self):
        stop = threading.Event(); stop.set()
        self.assertEqual(self.events(self.source([]), stop_event=stop), [])
        self.assertEqual(self.seen, [])

    def test_cancel_during_attempt(self):
        stop = threading.Event()
        self.assertEqual(self.events(self.source([{'error': 'bad'}], cancel=stop), stop_event=stop), [])
        self.assertEqual(self.closed, [0])

    def test_schema_error_repaired_before_delivery(self):
        def validate(calls):
            if calls[0]['arguments'] != '{"ok":true}':
                raise ToolOutputError('schema mismatch')
        self.body['_validate_generation_calls'] = validate
        events = self.events(self.source([{'call': True}, {'call': True, 'arguments': '{"ok":true}'}]))
        self.assertEqual([e.arguments_delta for e in events if isinstance(e, ToolCallDelta)], ['{"ok":true}'])

    def test_zero_attempt_setting_fails_honestly(self):
        with patch.dict(os.environ, TEXTGEN_TOOL_REPAIR_ATTEMPTS='0'):
            with self.assertRaises(ToolOutputError):
                self.events(self.source([{'error': 'bad'}]))
        self.assertEqual(len(self.seen), 1)

    def test_partial_parallel_batch_detected(self):
        raw = CALL + '\n<tool_call>{'
        self.assertIsNotNone(malformed_output(raw, parse_tool_call(raw, ['fetch']), ['fetch'], None, parse_tool_call))

    def test_unknown_tool_detected(self):
        raw = CALL.replace('fetch', 'missing')
        self.assertIsNotNone(malformed_output(raw, [], ['fetch'], None, parse_tool_call))

    def test_reasoning_and_markdown_examples_not_errors(self):
        for raw in ['<think>' + CALL + '</think>Done.', 'Example: `<tool_call>{`', '```xml\n<tool_call>{\n```']:
            self.assertIsNone(malformed_output(raw, [], ['fetch'], None, parse_tool_call))

    def test_progress_detector_is_conservative(self):
        for text in ['I will now check the remaining files.', 'Let me search the documentation.', '我会继续处理剩余项目。']:
            self.assertTrue(progress_only(text), text)
        for text in ['I will need your password.', 'I cannot continue.', 'Should I continue?', 'All 12 items are complete.',
                     'Example: "I will check the file."', 'I will check it. The answer is 42.']:
            self.assertFalse(progress_only(text), text)


if __name__ == '__main__':
    unittest.main()
