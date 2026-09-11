"""R3 configuration and multi-asset resume checks."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_r3 import validate_config
from experiments.run_r2 import image_hashes, load_checkpoint


class R3RunnerTests(unittest.TestCase):
    def test_configuration_records_effective_width_parameters(self):
        config = json.loads((Path(__file__).resolve().parents[2]/'configs/overlap_r3.json').read_text())
        validate_config(config)
        for changed in ({'reconstruction': None}, {'reconstruction': {}}, {'fixture_seed': True}, {'expected_image_count': 0}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_config({**config, **changed})

    def test_changed_mask_invalidates_completed_image_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'reconstruction.json').write_text('{}')
            (directory/'tail_mask.png').write_bytes(b'original mask bytes')
            summary = {'relative_image': 'a.tif'}
            (directory/'image_completion.json').write_text(json.dumps({'summary': summary, 'sha256': image_hashes(directory)}))
            self.assertEqual(load_checkpoint(directory, 'a.tif'), summary)
            (directory/'tail_mask.png').write_bytes(b'changed mask bytes')
            with self.assertRaises(ValueError):
                load_checkpoint(directory, 'a.tif')


if __name__ == '__main__':
    unittest.main()
