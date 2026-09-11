"""Small exhaustive checks for the bounded R2 joint-assignment solver."""
from __future__ import annotations

from itertools import product
import json
import random
import unittest

from global_assignment import Hypothesis, solve_assignment, solve_with_margins


def null(head: str, cost: float = 0.0) -> Hypothesis:
    return Hypothesis(f"{head}:null", head, cost, termination="null")


def exhaustive(items: list[Hypothesis]) -> list[tuple[float, tuple[str, ...]]]:
    heads = sorted({item.head_id for item in items})
    groups = [[item for item in items if item.head_id == head] for head in heads]
    answers = []
    for selected in product(*groups):
        segments, endpoints, compatible = set(), set(), True
        for item in selected:
            if set(item.exclusive_segments) & segments or (item.endpoint_id is not None and item.endpoint_id in endpoints):
                compatible = False
                break
            segments.update(item.exclusive_segments)
            if item.endpoint_id is not None:
                endpoints.add(item.endpoint_id)
        if compatible:
            answers.append((sum(item.cost for item in selected), tuple(item.id for item in selected)))
    return sorted(answers, key=lambda item: (item[0], item[1]))


class GlobalAssignmentTests(unittest.TestCase):
    def test_matches_exhaustive_random_small_cases(self) -> None:
        rng = random.Random(113)
        for case in range(60):
            items = []
            for head_index in range(3):
                head = f"h{head_index}"
                items.append(null(head, rng.choice([-1.0, 0.0, 1.0])))
                for option in range(2):
                    segment = () if rng.random() < 0.25 else (f"s{rng.randrange(4)}",)
                    endpoint = None if rng.random() < 0.6 else f"e{rng.randrange(3)}"
                    termination = "partial" if endpoint is None and rng.random() < 0.2 else "endpoint"
                    items.append(Hypothesis(f"{head}:{option}", head, rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0]), segment, endpoint, termination))
            expected = exhaustive(items)[:2]
            actual = solve_assignment(items, max_expansions=10000)
            self.assertEqual("optimal", actual["status"], case)
            self.assertEqual(expected, [(item["cost"], tuple(item["selected_by_head"][head] for head in sorted(item["selected_by_head"]))) for item in actual["solutions"]], case)

    def test_endpoint_and_segment_conflicts_but_crossings_can_share(self) -> None:
        items = [
            null("a", 5), Hypothesis("a:cross", "a", 0, ("component_1:branch_a",)), Hypothesis("a:end", "a", 1, (), "end"),
            null("b", 5), Hypothesis("b:cross", "b", 0, ("component_1:branch_b",)), Hypothesis("b:end", "b", 1, (), "end"),
        ]
        answer = solve_assignment(items)
        self.assertEqual({"a": "a:cross", "b": "b:cross"}, answer["solutions"][0]["selected_by_head"])
        items[4] = Hypothesis("b:reuse", "b", 0, ("component_1:branch_a",))
        answer = solve_assignment(items)
        self.assertNotEqual({"a": "a:cross", "b": "b:reuse"}, answer["solutions"][0]["selected_by_head"])

    def test_endpoint_collision_and_components_are_not_head_scoped(self) -> None:
        items = [
            null("a", 10), Hypothesis("a:x", "a", 0, ("component_a:s",), "shared-end"),
            null("b", 10), Hypothesis("b:x", "b", 0, ("component_b:s",), "shared-end"),
            # This available segment is intentionally unused; assignment does
            # not try to explain every observed tail fragment.
            Hypothesis("b:unused-evidence-control", "b", 20, ("component_b:unclaimed",)),
        ]
        selected = solve_assignment(items)["solutions"][0]["selected_by_head"]
        self.assertEqual(1, sum(1 for value in selected.values() if value.endswith(":null")))

    def test_null_is_required_and_forbidding_it_can_be_infeasible(self) -> None:
        with self.assertRaises(ValueError):
            solve_assignment([Hypothesis("a:x", "a", 0)])
        items = [null("a"), Hypothesis("a:x", "a", 2), null("b"), Hypothesis("b:x", "b", 0, ("s",)), Hypothesis("b:y", "b", 0, ("s",))]
        result = solve_assignment(items, forbidden_ids=frozenset({"a:null", "a:x"}))
        self.assertEqual("infeasible", result["status"])
        self.assertIsNone(result["lower_bound"])
        json.dumps(result, allow_nan=False)

    def test_top_two_ties_are_deterministic(self) -> None:
        items = [null("b"), Hypothesis("b:z", "b", 0), null("a"), Hypothesis("a:z", "a", 0)]
        first = solve_assignment(items)
        second = solve_assignment(list(reversed(items)))
        self.assertEqual(first["solutions"], second["solutions"])
        self.assertEqual(2, len(first["solutions"]))

    def test_limit_has_valid_lower_bound_and_never_claims_optimal(self) -> None:
        items = [null("a", 2), Hypothesis("a:x", "a", -3), null("b", 2), Hypothesis("b:x", "b", -3)]
        exact = exhaustive(items)[0][0]
        limited = solve_assignment(items, max_expansions=1)
        self.assertEqual("limit", limited["status"])
        self.assertLessEqual(limited["lower_bound"], exact)
        self.assertIsNotNone(limited["upper_bound"])

    def test_cutoff_after_a_cheap_incumbent_keeps_global_lower_bound(self) -> None:
        items = [
            null("a", 100), Hypothesis("a:cheap", "a", 0), Hypothesis("a:expensive", "a", 10),
            null("b", 100), Hypothesis("b:cheap", "b", 0), Hypothesis("b:expensive", "b", 10),
        ]
        exact = exhaustive(items)[0][0]
        limited = solve_assignment(items, max_expansions=3)
        self.assertEqual("limit", limited["status"])
        self.assertLessEqual(limited["lower_bound"], exact)
        self.assertLessEqual(exact, limited["upper_bound"])
        self.assertEqual(0, limited["lower_bound"])

    def test_random_limits_have_valid_global_bounds_and_json_output(self) -> None:
        rng = random.Random(991)
        for _ in range(40):
            items = []
            for head_index in range(3):
                head = f"h{head_index}"
                items.extend([null(head, rng.choice([0, 2])), Hypothesis(f"{head}:a", head, rng.choice([-2, 0, 3])), Hypothesis(f"{head}:b", head, rng.choice([-2, 0, 3]))])
            exact = exhaustive(items)[0][0]
            limited = solve_assignment(items, max_expansions=rng.randrange(1, 8))
            self.assertLessEqual(limited["lower_bound"], exact)
            self.assertLessEqual(exact, limited["upper_bound"])
            json.dumps(limited, allow_nan=False)

    def test_twenty_head_problem_proves_top_two_with_bound_pruning(self) -> None:
        items = []
        for index in range(20):
            head = f"h{index:02d}"
            items.extend([null(head, 5), Hypothesis(f"{head}:best", head, 0), Hypothesis(f"{head}:second", head, 1)])
        result = solve_assignment(items, max_expansions=300)
        self.assertEqual("optimal", result["status"])
        self.assertEqual([0, 1], [item["cost"] for item in result["solutions"]])
        self.assertGreater(result["diagnostics"]["pruned_by_bound"], 0)
        self.assertLess(result["diagnostics"]["expansions"], 300)
        self.assertLessEqual(result["diagnostics"]["max_frontier"], 4096)

    def test_per_head_counterfactual_can_differ_from_component_runner_up(self) -> None:
        items = [
            null("a", 10), Hypothesis("a:best", "a", 0), Hypothesis("a:alt", "a", 10),
            null("b", 10), Hypothesis("b:best", "b", 0), Hypothesis("b:alt", "b", 1),
        ]
        answer = solve_with_margins(items)
        self.assertEqual("b:alt", answer["component_runner_up"]["selected_by_head"]["b"])
        self.assertEqual(10, answer["head_alternatives"]["a"]["score_margin"])
        self.assertEqual(1, answer["head_alternatives"]["b"]["score_margin"])

    def test_empty_problem_is_exact(self) -> None:
        result = solve_assignment([])
        self.assertEqual("optimal", result["status"])
        self.assertEqual([{"selected_by_head": {}, "cost": 0.0}], result["solutions"])


if __name__ == "__main__":
    unittest.main()
