import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures import case_names, generate_case, load_case, render_identity_solution


class SyntheticFixtureTests(unittest.TestCase):
    def test_generation_is_reproducible_for_same_seed(self) -> None:
        for name in case_names():
            first = generate_case(name, seed=17)
            second = load_case(name, seed=17)
            self.assertEqual(first.to_truth_dict(), second.to_truth_dict())
            np.testing.assert_array_equal(first.rgb, second.rgb)
            np.testing.assert_array_equal(first.tail_mask, second.tail_mask)
            np.testing.assert_array_equal(first.head_mask, second.head_mask)

    def test_truth_export_is_json_serializable_and_excludes_evidence_arrays(self) -> None:
        fixture = generate_case("curved_crossing")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "truth.json"
            fixture.export_truth(destination)
            exported = json.loads(destination.read_text())
        self.assertEqual(exported, fixture.to_truth_dict())
        self.assertNotIn("rgb", exported)
        self.assertNotIn("tail_mask", exported)

    def test_graph_paths_reference_valid_edges_and_geometry(self) -> None:
        for name in case_names():
            fixture = generate_case(name)
            graph = fixture.truth["graph"]
            nodes = {node["id"]: node for node in graph["nodes"]}
            edges = {edge["id"]: edge for edge in graph["edges"]}
            self.assertTrue(fixture.tail_mask.any(), name)
            self.assertTrue(fixture.head_mask.any(), name)
            for edge in edges.values():
                self.assertIn(edge["start"], nodes, name)
                self.assertIn(edge["end"], nodes, name)
                self.assertGreaterEqual(len(edge["points"]), 2, name)
                self.assertEqual(edge["points"][0], nodes[edge["start"]]["xy"], name)
                self.assertEqual(edge["points"][-1], nodes[edge["end"]]["xy"], name)
            all_paths = list(fixture.truth["true_edge_paths"])
            for solution in fixture.truth["identity_solutions"]:
                all_paths.extend(solution["paths"])
            for path in all_paths:
                self.assertEqual(len(path["edge_ids"]), len(path["node_ids"]) - 1, name)
                self.assertTrue(set(path["edge_ids"]).issubset(edges), name)
                for index, edge_id in enumerate(path["edge_ids"]):
                    edge = edges[edge_id]
                    self.assertEqual(edge["start"], path["node_ids"][index], name)
                    self.assertEqual(edge["end"], path["node_ids"][index + 1], name)
            for instance in fixture.truth["instances"]:
                self.assertGreaterEqual(len(instance["centerline_xy"]), 2, name)
                self.assertIn(instance["head_id"], {head["id"] for head in fixture.truth["heads"]}, name)

    def test_expected_determinacy_is_explicit_and_mixed(self) -> None:
        fixtures = [generate_case(name) for name in case_names()]
        determinate = [fixture for fixture in fixtures if fixture.truth["expected_behavior"]["determinate"]]
        indeterminate = [fixture for fixture in fixtures if not fixture.truth["expected_behavior"]["determinate"]]
        self.assertGreaterEqual(len(determinate), 7)
        self.assertEqual([fixture.name for fixture in indeterminate], ["indeterminate_identity"])
        self.assertEqual(indeterminate[0].truth["expected_behavior"]["outcome"], "abstain_with_identity_alternatives")

    def test_self_loop_revisits_junction_without_reusing_an_ordinary_edge(self) -> None:
        fixture = generate_case("self_loop")
        path = fixture.truth["true_edge_paths"][0]
        self.assertGreater(path["node_ids"].count("j"), 1)
        self.assertEqual(len(path["edge_ids"]), len(set(path["edge_ids"])))

    def test_indeterminate_identity_solutions_are_distinct_but_render_same_evidence(self) -> None:
        fixture = generate_case("indeterminate_identity", seed=3)
        solutions = fixture.truth["identity_solutions"]
        self.assertGreaterEqual(len(solutions), 2)
        first_paths = solutions[0]["paths"]
        second_paths = solutions[1]["paths"]
        self.assertNotEqual(first_paths, second_paths)
        solution_edge_sets = [sorted(edge_id for path in solution["paths"] for edge_id in path["edge_ids"]) for solution in solutions]
        self.assertEqual(solution_edge_sets[0], solution_edge_sets[1])
        first_rgb, first_mask = render_identity_solution(fixture, solutions[0]["id"])
        second_rgb, second_mask = render_identity_solution(fixture, solutions[1]["id"])
        np.testing.assert_array_equal(first_rgb, second_rgb)
        np.testing.assert_array_equal(first_mask, second_mask)
        self.assertEqual(
            fixture.truth["expected_behavior"]["shared_corridor_edge_ids"],
            ["corridor_1", "corridor_2", "corridor_3"],
        )


if __name__ == "__main__":
    unittest.main()
