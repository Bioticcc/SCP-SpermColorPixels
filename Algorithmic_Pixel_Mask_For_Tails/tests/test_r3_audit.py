"""Audit permissions are pairwise, including inside overlapping declared regions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.audit_r3 import audit, check_pairwise_ownership


class R3AuditTests(unittest.TestCase):
    def test_pair_permissions_do_not_authorize_third_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask = np.zeros((4, 4), np.uint8); mask[1, 1] = 255
            instances = []
            for key in ('a', 'b', 'c'):
                Image.fromarray(mask).save(root/f'{key}.png')
                instances.append({'id': key, 'roi_xyxy': [10, 20, 14, 24], 'files': {'instance_mask': f'{key}.png'}})
            Image.fromarray(np.full((4, 4), 255, np.uint8)).save(root/'crossing.png')
            component = {'source_roi_xyxy': [10, 20, 14, 24], 'crossing_regions': [
                {'track_ids': ['a', 'b'], 'mask_file': 'crossing.png'},
                {'track_ids': ['b', 'c'], 'mask_file': 'crossing.png'}]}
            result = check_pairwise_ownership(root, {'instances': instances, 'components': [component]})
            self.assertEqual(result['shared_pair_pixels'], 3)
            self.assertEqual(result['forbidden_pair_pixels'], 1)
            component['crossing_regions'].append({'track_ids': ['a', 'c'], 'mask_file': 'crossing.png'})
            self.assertEqual(check_pairwise_ownership(root, {'instances': instances, 'components': [component]})['forbidden_pair_pixels'], 0)

    def test_missing_run_is_failed_audit_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            result = audit(Path(temp))
            self.assertFalse(result['passed'])
            self.assertIn('required artifact', result['errors'][0])


if __name__ == '__main__':
    unittest.main()
