"""R2 cache safety and original-image diagnostic coordinates."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_r2 import compare_to_baseline, image_hashes, load_checkpoint, validate_config


class R2RunnerTests(unittest.TestCase):
    def test_checkpoint_refuses_changed_assignment_or_viewer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'assignment.json').write_text('{}')
            (root/'index.html').write_text('<p>first</p>')
            summary = {'relative_image': 'source.tif'}
            (root/'image_completion.json').write_text(json.dumps({'summary': summary, 'sha256': image_hashes(root)}))
            self.assertEqual(load_checkpoint(root, 'source.tif'), summary)
            with self.assertRaises(ValueError):
                load_checkpoint(root, 'different.tif')
            (root/'index.html').write_text('<p>changed</p>')
            with self.assertRaises(ValueError):
                load_checkpoint(root, 'source.tif')

    def test_baseline_comparison_uses_source_coordinates_and_retains_status(self):
        record = {'components': [{'tail_id': 7, 'roi_xyxy': [100, 200, 130, 230],
                  'baseline': [{'head_id': 'h', 'crop_id': 'crop', 'status': 'ambiguous', 'pixels_xy': [[3, 4], [4, 4]]}]}]}
        items = [{'id': 'chosen', 'termination': 'baseline_control', 'metadata': {
                  'roi_xyxy': [102, 202, 110, 210], 'pixels_xy_unordered': [[1, 2], [2, 2]]}}]
        assignment = {'selected_by_head': {'h': 'chosen'}, 'changed_head_ids': []}
        row = compare_to_baseline(record, items, assignment)[0]
        self.assertTrue(row['baseline_comparisons'][0]['exact_same_source_pixels'])
        self.assertEqual(row['baseline_comparisons'][0]['status'], 'ambiguous')
        self.assertEqual(row['selected_pixel_count'], 2)

    def test_config_rejects_invalid_bounds_and_untracked_settings(self):
        path = Path(__file__).resolve().parents[2]/'configs/overlap_r2.json'
        config = json.loads(path.read_text())
        validate_config(config)
        for key, bad in [('solver_max_expansions', True), ('k', 0), ('null_cost', float('nan')),
                         ('partial_cost', -1), ('fixture_seed', 1.5)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_config({**config, key: bad})
        with self.assertRaises(ValueError):
            validate_config({**config, 'undocumented': 1})


if __name__ == '__main__':
    unittest.main()
