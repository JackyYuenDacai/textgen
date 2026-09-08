import copy
import unittest

from modules.prompt_utils import relocate_current_time


class CurrentTimeCacheTests(unittest.TestCase):
    def test_changing_time_leaves_instructions_and_history_identical(self):
        def request(timestamp):
            return [
                {'role': 'system', 'content': f'Workspace instructions\n<current_time>\n{timestamp}\n</current_time>\nRules'},
                {'role': 'user', 'content': 'Earlier question'},
                {'role': 'assistant', 'content': 'Earlier answer'},
                {'role': 'user', 'content': 'Next question'},
            ]

        first = request('2026-09-08 10:00:00')
        original = copy.deepcopy(first)
        a = relocate_current_time(first)
        b = relocate_current_time(request('2026-09-08 10:05:00'))
        self.assertEqual(a[:-1], b[:-1])
        self.assertEqual(a[0]['content'], 'Workspace instructions\n\nRules')
        self.assertEqual(a[1:-1], first[1:])
        self.assertEqual(a[-1], {'role': 'system', 'content': '<current_time>\n2026-09-08 10:00:00\n</current_time>'})
        self.assertIn('10:05:00', b[-1]['content'])
        self.assertEqual(first, original)

    def test_tool_call_and_result_remain_adjacent_and_unchanged(self):
        messages = [
            {'role': 'system', 'content': '<current_time>now</current_time>'},
            {'role': 'user', 'content': 'Check files'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1'}]},
            {'role': 'tool', 'content': '<current_time>file contents</current_time>', 'tool_call_id': 'call-1'},
        ]
        result = relocate_current_time(messages)
        self.assertEqual(result[1:-1], messages[1:])
        self.assertEqual(result[0]['content'], '')
        self.assertEqual(result[-1]['content'], '<current_time>now</current_time>')

    def test_assistant_prefill_remains_last(self):
        partial = {'role': 'assistant', 'content': '', 'reasoning_content': 'Let me think'}
        messages = [
            {'role': 'system', 'content': 'Rules <current_time>now</current_time>'},
            {'role': 'user', 'content': 'Question'},
            partial,
        ]
        result = relocate_current_time(messages)
        self.assertEqual(result[-1], partial)
        self.assertEqual(result[-2]['role'], 'system')
        self.assertEqual(result[-2]['content'], '<current_time>now</current_time>')

    def test_preserves_roles_and_multiple_blocks(self):
        messages = [
            {'role': 'system', 'content': 'A<current_time>one</current_time>B<current_time>two</current_time>C'},
            {'role': 'developer', 'content': 'D<current_time>three</current_time>E'},
            {'role': 'user', 'content': 'Question'},
        ]
        result = relocate_current_time(messages)
        self.assertEqual([m['content'] for m in result[:2]], ['ABC', 'DE'])
        self.assertEqual(result[-2], {'role': 'system', 'content': '<current_time>one</current_time>\n<current_time>two</current_time>'})
        self.assertEqual(result[-1], {'role': 'developer', 'content': '<current_time>three</current_time>'})

    def test_ignores_other_tags_incomplete_tags_and_nested_markup(self):
        for content in (
            '<time>now</time>', '<current_time>unfinished',
            '<current_time zone="UTC">now</current_time>',
            '<current_time><b>now</b></current_time>',
            '<CURRENT_TIME>now</CURRENT_TIME>',
        ):
            with self.subTest(content=content):
                messages = [{'role': 'system', 'content': content}, {'role': 'user', 'content': 'Hello'}]
                self.assertEqual(relocate_current_time(messages), messages)

    def test_leaves_nonleading_system_and_user_assistant_tool_text_alone(self):
        messages = [
            {'role': role, 'content': '<current_time>example</current_time>'}
            for role in ('user', 'assistant', 'tool', 'system', 'developer')
        ]
        self.assertEqual(relocate_current_time(messages), messages)

    def test_empty_system_only_and_structured_content_are_unchanged(self):
        for messages in (
            [],
            [{'role': 'system', 'content': '<current_time>now</current_time>'}],
            [{'role': 'system', 'content': [{'type': 'text', 'text': '<current_time>now</current_time>'}]},
             {'role': 'user', 'content': 'Hello'}],
        ):
            with self.subTest(messages=messages):
                self.assertEqual(relocate_current_time(messages), messages)


if __name__ == '__main__':
    unittest.main()
