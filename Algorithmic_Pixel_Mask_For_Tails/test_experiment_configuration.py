import copy
import json
from pathlib import Path
import unittest

from experiments.configuration import baseline_arguments


class BaselineConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/overlap_demo_baseline.json").read_text())

    def test_effective_cli_matches_frozen_values(self):
        argv, args = baseline_arguments(self.config, Path('/tmp/input with spaces'), Path('/tmp/new output'))
        self.assertEqual(args.input_root, Path('/tmp/input with spaces'))
        self.assertFalse(args.overwrite)
        for name, value in self.config['arguments'].items():
            self.assertEqual(getattr(args, name), value)
        self.assertNotIn('--overwrite', argv)

    def test_missing_or_unknown_setting_is_not_silently_defaulted(self):
        for name, value in [('missing', None), ('unknown', 10)]:
            config = copy.deepcopy(self.config)
            if name == 'missing':
                del config['arguments']['head_contact_radius']
            else:
                config['arguments']['unknown'] = value
            with self.assertRaises(ValueError):
                baseline_arguments(config, Path('/tmp/in'), Path('/tmp/out'))

    def test_overwrite_and_partial_run_are_rejected(self):
        for name, value in [('overwrite', True), ('limit', 2)]:
            config = copy.deepcopy(self.config)
            config['arguments'][name] = value
            with self.assertRaises(ValueError):
                baseline_arguments(config, Path('/tmp/in'), Path('/tmp/out'))


if __name__ == '__main__':
    unittest.main()
