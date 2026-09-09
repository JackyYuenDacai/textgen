import importlib.util
import json
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import Mock, patch


helper_path = Path(__file__).resolve().parents[1] / 'exllamav3/exllamav3/util/optional_extensions.py'
spec = importlib.util.spec_from_file_location('optional_extensions_under_test', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class OptionalExtensionTests(unittest.TestCase):
    def test_build_mismatch_is_detected_without_importing_native_package(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / 'cpp_lib.json'
            metadata_path.write_text(json.dumps({'version': {'torch': '2.9.0+cu128'}}), encoding='utf-8')
            package = Mock()
            package.locate_file.return_value = metadata_path
            with patch.object(helper, 'distribution', return_value=package):
                self.assertIn('2.9.0+cu128', helper.xformers_unavailable_reason('2.10.0+cu128'))
                self.assertIsNone(helper.xformers_unavailable_reason('2.9.0+cu128'))
                metadata_path.write_text('{invalid', encoding='utf-8')
                self.assertIn('could not be read', helper.xformers_unavailable_reason('2.10.0+cu128'))
                package.locate_file.return_value = Path(directory) / 'absent.json'
                self.assertIsNone(helper.xformers_unavailable_reason('2.10.0+cu128'))

    def test_missing_optional_package_is_unavailable(self):
        with patch.object(helper, 'distribution', side_effect=PackageNotFoundError):
            self.assertEqual(helper.xformers_unavailable_reason('2.10.0'), 'xFormers is not installed')
