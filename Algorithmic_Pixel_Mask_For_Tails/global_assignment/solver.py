"""Dependency-free bounded branch-and-bound for R2 hypothesis selection.

Costs are intentionally uncalibrated path/assignment costs.  A result marked
``limit`` describes only the explored state space; its lower bound includes
discarded frontier states and remains valid when hypothesis costs are negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class Hypothesis:
    """One selectable tail-track proposal.

    A null choice is represented by ``termination='null'``.  Each head must
    supply exactly one such choice; its cost is supplied by the route producer.
    Ordinary segments and non-null endpoint IDs are exclusive resources.
    """

    id: str
    head_id: str
    cost: float
    exclusive_segments: tuple[str, ...] = ()
    endpoint_id: str | None = None
    termination: str = "endpoint"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_null(self) -> bool:
        return self.termination == "null"


def _assignment_key(head_ids: tuple[str, ...], selected: dict[str, Hypothesis]) -> tuple[str, ...]:
    return tuple(selected[head_id].id for head_id in head_ids)


def _validate(hypotheses: Iterable[Hypothesis], forbidden_ids: frozenset[str]) -> tuple[tuple[str, ...], dict[str, tuple[Hypothesis, ...]]]:
    by_head: dict[str, list[Hypothesis]] = {}
    ids: set[str] = set()
    for item in hypotheses:
        if not isinstance(item, Hypothesis):
            raise TypeError("hypotheses must contain Hypothesis records")
        if not item.id or not item.head_id:
            raise ValueError("hypothesis id and head_id must be non-empty")
        if item.id in ids:
            raise ValueError(f"hypothesis id is not globally unique: {item.id!r}")
        ids.add(item.id)
        if not math.isfinite(item.cost):
            raise ValueError(f"hypothesis {item.id!r} has a non-finite cost")
        if len(set(item.exclusive_segments)) != len(item.exclusive_segments):
            raise ValueError(f"hypothesis {item.id!r} repeats an exclusive segment")
        if any(not segment for segment in item.exclusive_segments):
            raise ValueError(f"hypothesis {item.id!r} has an empty exclusive segment id")
        if item.is_null and (item.exclusive_segments or item.endpoint_id is not None):
            raise ValueError(f"null hypothesis {item.id!r} cannot claim resources")
        if item.termination == "partial" and item.endpoint_id is not None:
            raise ValueError(f"partial hypothesis {item.id!r} cannot claim an endpoint")
        by_head.setdefault(item.head_id, []).append(item)

    for head_id, choices in by_head.items():
        null_count = sum(item.is_null for item in choices)
        if null_count != 1:
            raise ValueError(f"head {head_id!r} must have exactly one explicit null hypothesis")
    head_ids = tuple(sorted(by_head))
    allowed = {
        head_id: tuple(sorted((item for item in choices if item.id not in forbidden_ids), key=lambda item: (item.cost, item.id)))
        for head_id, choices in by_head.items()
    }
    return head_ids, allowed


def solve_assignment(
    hypotheses: list[Hypothesis],
    max_expansions: int = 100000,
    forbidden_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return the two cheapest mutually compatible complete assignments.

    The search chooses exactly one proposal per head.  It permits unused tail
    evidence, shares no ordinary segment, and makes every non-null endpoint
    exclusive.  The null route is a normal choice, so excluded null routes can
    make a constrained problem infeasible.
    """
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, int) or max_expansions < 1:
        raise ValueError("max_expansions must be a positive integer")
    if not isinstance(forbidden_ids, frozenset):
        forbidden_ids = frozenset(forbidden_ids)
    started = time.perf_counter()
    head_ids, choices = _validate(hypotheses, forbidden_ids)
    allowed_counts = {head_id: len(choices[head_id]) for head_id in head_ids}

    def result(status: str, solutions: list[dict[str, Any]], lower: float | None, diagnostics: dict[str, Any]) -> dict[str, Any]:
        elapsed = time.perf_counter() - started
        upper = solutions[0]["cost"] if solutions else None
        return {
            "status": status,
            "solutions": solutions,
            "lower_bound": lower,
            "upper_bound": upper,
            "diagnostics": {**diagnostics, "runtime_seconds": elapsed},
        }

    if not head_ids:
        empty = {"selected_by_head": {}, "cost": 0.0}
        return result("optimal", [empty], 0.0, {"expansions": 0, "frontier_count": 0, "discarded_frontier_count": 0, "max_expansions": max_expansions, "max_frontier": max_expansions})
    if any(not choices[head_id] for head_id in head_ids):
        return result("infeasible", [], None, {"expansions": 0, "frontier_count": 0, "discarded_frontier_count": 0, "max_expansions": max_expansions, "max_frontier": min(4096, max_expansions), "allowed_choice_counts": allowed_counts})

    # Prefix minima make a valid relaxation even for negative costs: they
    # ignore resource conflicts but retain the cheapest remaining choice.
    min_cost = [min(item.cost for item in choices[head_id]) for head_id in head_ids]
    suffix_min = [0.0] * (len(head_ids) + 1)
    for index in range(len(head_ids) - 1, -1, -1):
        suffix_min[index] = suffix_min[index + 1] + min_cost[index]
    lex_suffix = [()] * (len(head_ids) + 1)
    for index in range(len(head_ids) - 1, -1, -1):
        lex_suffix[index] = (min(item.id for item in choices[head_ids[index]]),) + lex_suffix[index + 1]

    # A feasible all-null incumbent gives a useful upper bound immediately.
    all_null: dict[str, Hypothesis] = {}
    for head_id in head_ids:
        null = next((item for item in choices[head_id] if item.is_null), None)
        if null is None:
            all_null = {}
            break
        all_null[head_id] = null
    found: dict[tuple[str, ...], float] = {}
    if all_null:
        found[_assignment_key(head_ids, all_null)] = sum(item.cost for item in all_null.values())

    # (relaxed lower bound, selected-ID prefix, depth, cost, segments, endpoints)
    frontier: list[tuple[float, tuple[str, ...], int, float, frozenset[str], frozenset[str]]] = [
        (suffix_min[0], (), 0, 0.0, frozenset(), frozenset())
    ]
    # The frontier is bounded independently of the work limit, so a caller
    # cannot accidentally reserve one state slot per permitted expansion.
    max_frontier = min(4096, max_expansions)
    discarded_best: tuple[float, tuple[str, ...]] | None = None
    discarded_count = 0
    expansions = generated = incompatible = pruned_by_bound = 0

    def top_two() -> list[tuple[tuple[str, ...], float]]:
        return sorted(found.items(), key=lambda pair: (pair[1], pair[0]))[:2]

    def can_improve_top_two(lower: float, optimistic_key: tuple[str, ...]) -> bool:
        """Whether this relaxation can enter the deterministic top two."""
        ranked = top_two()
        if len(ranked) < 2:
            return True
        second_key, second_cost = ranked[1]
        return (lower, optimistic_key) < (second_cost, second_key)

    def retain(state: tuple[float, tuple[str, ...], int, float, frozenset[str], frozenset[str]]) -> None:
        nonlocal discarded_best, discarded_count, pruned_by_bound
        optimistic = state[1] + lex_suffix[state[2]]
        if not can_improve_top_two(state[0], optimistic):
            pruned_by_bound += 1
            return
        if len(frontier) < max_frontier:
            heapq.heappush(frontier, state)
            return
        # Preserve the best relaxation states deterministically.  A discarded
        # state is still represented in the reported lower bound.
        worst_index = max(
            range(len(frontier)),
            key=lambda index: (frontier[index][0], frontier[index][1] + lex_suffix[frontier[index][2]]),
        )
        worst = frontier[worst_index]
        worst_optimistic = worst[1] + lex_suffix[worst[2]]
        if (state[0], optimistic) < (worst[0], worst_optimistic):
            candidate = (worst[0], worst_optimistic)
            discarded_best = candidate if discarded_best is None else min(discarded_best, candidate)
            discarded_count += 1
            frontier[worst_index] = state
            heapq.heapify(frontier)
        else:
            candidate = (state[0], optimistic)
            discarded_best = candidate if discarded_best is None else min(discarded_best, candidate)
            discarded_count += 1

    while frontier and expansions < max_expansions:
        _, prefix, depth, cost, used_segments, used_endpoints = heapq.heappop(frontier)
        expansions += 1
        if not can_improve_top_two(cost + suffix_min[depth], prefix + lex_suffix[depth]):
            pruned_by_bound += 1
            continue
        if depth == len(head_ids):
            found[prefix] = cost
            continue
        for item in choices[head_ids[depth]]:
            segments = frozenset(item.exclusive_segments)
            endpoint = item.endpoint_id if not item.is_null else None
            if segments & used_segments or (endpoint is not None and endpoint in used_endpoints):
                incompatible += 1
                continue
            child_cost = cost + item.cost
            child_prefix = prefix + (item.id,)
            child = (
                child_cost + suffix_min[depth + 1], child_prefix, depth + 1, child_cost,
                used_segments | segments,
                used_endpoints | ({endpoint} if endpoint is not None else set()),
            )
            generated += 1
            retain(child)

    ordered = top_two()
    solutions = [
        {"selected_by_head": dict(zip(head_ids, selection)), "cost": cost}
        for selection, cost in ordered
    ]
    unresolved = [(state[0], state[1] + lex_suffix[state[2]]) for state in frontier]
    if discarded_best is not None:
        unresolved.append(discarded_best)
    unresolved_lower = min((item[0] for item in unresolved), default=None)
    # A retained or evicted state may remain in memory/history, yet it cannot
    # affect the ordered top two once its relaxation loses to the runner-up.
    unresolved_can_improve = any(can_improve_top_two(lower, key) for lower, key in unresolved)
    exhaustive = not frontier and discarded_best is None
    proved = (len(ordered) >= 2 and not unresolved_can_improve) or (len(ordered) < 2 and exhaustive)
    status = "optimal" if proved else "limit"
    if exhaustive and not solutions:
        status = "infeasible"
    # This is a bound on the global optimum, not merely unexplored work.  A
    # feasible incumbent therefore always caps it from above, including after
    # a cutoff that leaves only more-expensive branches in the frontier.
    lower = None if status == "infeasible" else min(
        [solutions[0]["cost"]] + ([unresolved_lower] if unresolved_lower is not None else [])
        if solutions else ([unresolved_lower] if unresolved_lower is not None else [])
    )
    second_lower = None
    if len(solutions) > 1:
        second_lower = min([solutions[1]["cost"]] + ([unresolved_lower] if unresolved_lower is not None else []))
    elif unresolved_lower is not None:
        second_lower = unresolved_lower
    return result(status, solutions, lower, {
        "expansions": expansions,
        "generated_states": generated,
        "incompatible_choices": incompatible,
        "frontier_count": len(frontier),
        "discarded_frontier_count": discarded_count,
        "discarded_frontier_lower_bound": discarded_best[0] if discarded_best is not None else None,
        "pruned_by_bound": pruned_by_bound,
        "second_best_lower_bound": second_lower,
        "max_expansions": max_expansions,
        "max_frontier": max_frontier,
        "allowed_choice_counts": allowed_counts,
        "complete_search": proved,
    })


def solve_with_margins(
    hypotheses: list[Hypothesis],
    max_expansions: int = 100000,
    forbidden_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Solve once, then constrain each selected head choice to obtain its margin."""
    base = solve_assignment(hypotheses, max_expansions=max_expansions, forbidden_ids=forbidden_ids)
    output = dict(base)
    output["component_runner_up"] = base["solutions"][1] if len(base["solutions"]) > 1 else None
    alternatives: dict[str, dict[str, Any]] = {}
    if not base["solutions"]:
        output["head_alternatives"] = alternatives
        return output
    selected = base["solutions"][0]["selected_by_head"]
    base_cost = base["solutions"][0]["cost"]
    for head_id in sorted(selected):
        constrained = solve_assignment(
            hypotheses,
            max_expansions=max_expansions,
            forbidden_ids=frozenset(set(forbidden_ids) | {selected[head_id]}),
        )
        alternative = constrained["solutions"][0] if constrained["solutions"] else None
        exact = base["status"] == "optimal" and constrained["status"] == "optimal" and alternative is not None
        lower_cost, upper_cost = constrained["lower_bound"], constrained["upper_bound"]
        entry: dict[str, Any] = {
            "selected_id": selected[head_id],
            "status": constrained["status"],
            "alternative": alternative,
            "alternative_cost_lower_bound": lower_cost,
            "alternative_cost_upper_bound": upper_cost,
            "score_margin": alternative["cost"] - base_cost if exact else None,
        }
        # When the base problem is limited, its selected cost is only an
        # incumbent.  These are differences from that incumbent, not claims
        # about the (unknown) optimal assignment margin.
        if not exact and all(value is not None for value in (lower_cost, upper_cost, base["lower_bound"], base["upper_bound"])):
            entry["difference_from_incumbent_bounds"] = {
                "lower": lower_cost - base["upper_bound"],
                "upper": upper_cost - base["lower_bound"],
            }
        alternatives[head_id] = entry
    output["head_alternatives"] = alternatives
    return output
