"""A partial report may resume only from verified image artifacts."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_r1 import _checkpoint_files, load_image_checkpoint, validate_config


class CheckpointTests(unittest.TestCase):
    def test_only_complete_unchanged_matching_image_checkpoint_can_load(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.assertIsNone(load_image_checkpoint(directory, 'a.tif'))
            for name in ('routes.json', 'index.html', 'original.jpg', 'baseline.jpg'):
                (directory/name).write_bytes(b'fixture artifact')
            saved = {'summary': {'relative_image': 'a.tif'}, 'sha256': _checkpoint_files(directory)}
            (directory/'image_completion.json').write_text(json.dumps(saved))
            self.assertEqual(load_image_checkpoint(directory, 'a.tif'), saved['summary'])
            with self.assertRaises(ValueError):
                load_image_checkpoint(directory, 'other.tif')
            (directory/'routes.json').write_text('changed')
            with self.assertRaises(ValueError):
                load_image_checkpoint(directory, 'a.tif')

    def test_invalid_search_limits_fail_before_run(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root/'configs/overlap_r1.json').read_text())
        validate_config(config)
        for update in ({'k': 0}, {'max_expansions': True}, {'diagnostic_k': 1}, {'unexpected': 1}):
            with self.assertRaises(ValueError):
                validate_config({**config, **update})


if __name__ == '__main__':
    unittest.main()
