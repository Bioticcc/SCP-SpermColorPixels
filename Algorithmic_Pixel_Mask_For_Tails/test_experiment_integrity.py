import tempfile
from pathlib import Path
import unittest

from PIL import Image

from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance


class SnapshotIntegrityTests(unittest.TestCase):
    def test_new_input_changed_config_and_changed_code_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'inputs'
            inputs.mkdir()
            Image.new('RGB', (10, 10), 'white').save(inputs/'one.png')
            code, config = root/'code.py', root/'config.json'
            code.write_text('x = 1\n')
            config.write_text('{}')
            snapshot = build_provenance(input_root=inputs, project_root=root, code_paths=[code], config_paths=[config])
            self.assertTrue(verify_snapshot(snapshot, root)['passed'])
            Image.new('RGB', (10, 10), 'black').save(inputs/'two.png')
            config.write_text('{"changed": true}')
            code.write_text('x = 2\n')
            result = verify_snapshot(snapshot, root)
            self.assertFalse(result['passed'])
            self.assertEqual(result['added_inputs'], ['two.png'])
            self.assertEqual(result['changed_source_files'], ['code.py'])
            self.assertEqual(result['changed_config_files'], ['config.json'])

    def test_removed_and_changed_inputs_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (10, 10), 'white').save(root/'one.png')
            Image.new('RGB', (10, 10), 'white').save(root/'two.png')
            snapshot = build_provenance(input_root=root, project_root=root, code_paths=[])
            (root/'one.png').unlink()
            Image.new('RGB', (10, 10), 'black').save(root/'two.png')
            result = verify_snapshot(snapshot, root)
            self.assertEqual(result['missing_inputs'], ['one.png'])
            self.assertEqual(result['changed_inputs'], ['two.png'])


if __name__ == '__main__':
    unittest.main()
