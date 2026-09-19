import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class LlamaServerPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real startup method without importing GPU/UI dependencies.
        tree = ast.parse(Path('modules/llama_cpp_server.py').read_text(encoding='utf-8'))
        server = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlamaServer')
        method = next(n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == '_start_server')
        cls.code = compile(ast.Module(body=[method], type_ignores=[]), '<startup>', 'exec')

    def invoke(self, path, ik=False, explicit=None):
        args = SimpleNamespace(llama_server_path=path, ik=ik)
        log = Mock()
        # Stop after selection, before building arguments or starting a process.
        log.info.side_effect = InterruptedError('selected')
        scope = {'Path': Path, 'shared': SimpleNamespace(args=args), 'logger': log}
        exec(self.code, scope)
        server = SimpleNamespace(server_path=explicit)
        return scope['_start_server'], server, log

    def test_custom_executable_selected(self):
        method, server, log = self.invoke(__file__)
        with self.assertRaises(InterruptedError):
            method(server)
        self.assertEqual(server.server_path, str(Path(__file__).resolve()))

    def test_missing_executable_fails(self):
        method, server, _ = self.invoke('does-not-exist/llama-server.exe')
        with self.assertRaises(FileNotFoundError):
            method(server)

    def test_ik_conflict_fails(self):
        method, server, _ = self.invoke(__file__, ik=True)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            method(server)

    def test_explicit_constructor_path_takes_precedence(self):
        method, server, _ = self.invoke('does-not-exist', explicit='explicit-server')
        with self.assertRaises(InterruptedError):
            method(server)
        self.assertEqual(server.server_path, 'explicit-server')


if __name__ == '__main__':
    unittest.main()
