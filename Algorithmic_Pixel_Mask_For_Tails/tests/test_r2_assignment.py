"""Image grouping must preserve head/resource constraints and runner-up costs."""
import random
import unittest

from experiments.r2_assignment import assign_pool, conflict_groups
from global_assignment import Hypothesis, solve_assignment


class GroupAssignmentTests(unittest.TestCase):
    def test_disconnected_group_composition_matches_monolithic_top_two(self):
        items = [Hypothesis('a:null', 'a', 3, termination='null'),
                 Hypothesis('a:first', 'a', 0, ('tail1:s',)),
                 Hypothesis('a:second', 'a', 2, ('tail2:s',)),
                 Hypothesis('b:null', 'b', 3, termination='null'),
                 Hypothesis('b:first', 'b', 0, ('tail3:s',)),
                 Hypothesis('b:second', 'b', 1, ('tail4:s',))]
        self.assertEqual(len(conflict_groups(items)), 2)
        result = assign_pool(items)
        exact = solve_assignment(items)
        self.assertEqual(result['selected_by_head'], exact['solutions'][0]['selected_by_head'])
        self.assertEqual(result['runner_up']['selected_by_head'], exact['solutions'][1]['selected_by_head'])
        self.assertEqual(result['runner_up']['cost'], exact['solutions'][1]['cost'])

    def test_same_head_links_components_and_resource_conflict_links_heads(self):
        items = [Hypothesis('a:null', 'a', 3, termination='null'),
                 Hypothesis('a:tail1', 'a', 0, ('tail1:s',)),
                 Hypothesis('a:tail2', 'a', 0, ('tail2:s',)),
                 Hypothesis('b:null', 'b', 3, termination='null'),
                 Hypothesis('b:tail2', 'b', 0, ('tail2:s',))]
        self.assertEqual(len(conflict_groups(items)), 1)
        result = assign_pool(items)
        self.assertEqual(len(result['selected_by_head']), 2)
        self.assertFalse(result['joint_conflicts'])

    def test_empty_pool_has_empty_complete_assignment(self):
        result = assign_pool([])
        self.assertEqual(result['selected_by_head'], {})
        self.assertEqual(result['status'], 'optimal')
        self.assertIsNone(result['runner_up'])

    def test_duplicate_ids_across_disconnected_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'globally unique'):
            assign_pool([Hypothesis('same', 'a', 0, termination='null'),
                         Hypothesis('same', 'b', 0, termination='null')])

    def test_tied_interleaved_groups_match_complete_solver(self):
        rng = random.Random(271)
        for _ in range(30):
            items = []
            for head in ('a', 'b', 'c', 'd'):
                group = 'ac' if head in ('a', 'c') else 'bd'
                items.append(Hypothesis(head+':null', head, 3, termination='null'))
                for option in (0, 1):
                    items.append(Hypothesis(f'{head}:{option}', head, rng.choice([-1, 0, 0, 1]),
                                            (f'{group}:{option}',)))
            result, exact = assign_pool(items), solve_assignment(items)
            self.assertEqual(result['selected_by_head'], exact['solutions'][0]['selected_by_head'])
            self.assertEqual(result['runner_up']['selected_by_head'], exact['solutions'][1]['selected_by_head'])


if __name__ == '__main__':
    unittest.main()
