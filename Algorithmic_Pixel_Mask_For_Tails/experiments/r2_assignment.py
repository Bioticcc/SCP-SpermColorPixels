"""Factor image-wide assignment by actual hypothesis conflicts and head identity."""
from __future__ import annotations

from collections import defaultdict

from global_assignment import Hypothesis, solve_with_margins


def conflict_groups(hypotheses: list[Hypothesis]) -> list[list[Hypothesis]]:
    """Split only when neither a head nor any exclusive resource crosses groups."""
    choices = defaultdict(list)
    owners = defaultdict(set)
    for item in hypotheses:
        choices[item.head_id].append(item)
        for segment in item.exclusive_segments:
            owners[('segment', segment)].add(item.head_id)
        if item.endpoint_id is not None:
            owners[('endpoint', item.endpoint_id)].add(item.head_id)
    neighbors = {head: set() for head in choices}
    for heads in owners.values():
        for head in heads:
            neighbors[head].update(heads - {head})
    unseen, groups = set(choices), []
    while unseen:
        start = min(unseen)
        pending, group = [start], set()
        while pending:
            head = pending.pop()
            if head in group:
                continue
            group.add(head)
            pending.extend(sorted(neighbors[head] - group, reverse=True))
        unseen.difference_update(group)
        groups.append([item for head in sorted(group) for item in sorted(choices[head], key=lambda h: h.id)])
    return groups


def conflicting_resources(selected: list[Hypothesis]) -> list[dict]:
    resources = defaultdict(list)
    for item in selected:
        for segment in item.exclusive_segments:
            resources[('segment', segment)].append(item.id)
        if item.endpoint_id is not None:
            resources[('endpoint', item.endpoint_id)].append(item.id)
    return [{'kind': kind, 'resource': resource, 'hypotheses': ids}
            for (kind, resource), ids in sorted(resources.items()) if len(ids) > 1]


def assign_pool(hypotheses: list[Hypothesis], max_expansions: int = 100000) -> dict:
    choices = defaultdict(list)
    by_id = {item.id: item for item in hypotheses}
    if len(by_id) != len(hypotheses):
        raise ValueError('Hypothesis IDs must be globally unique before conflict decomposition')
    for item in hypotheses:
        choices[item.head_id].append(item)
    independent = {head: min(items, key=lambda h: (h.cost, h.id)).id for head, items in sorted(choices.items())}
    groups = []
    selected, total_cost = {}, 0.0
    lower_bound, complete = 0.0, True
    for number, group in enumerate(conflict_groups(hypotheses), 1):
        solution = solve_with_margins(group, max_expansions=max_expansions)
        groups.append({'id': f'group_{number:03d}', 'head_ids': sorted({h.head_id for h in group}),
                       'hypothesis_ids': [h.id for h in group], 'assignment': solution})
        if not solution['solutions']:
            raise RuntimeError('An unconstrained group with explicit null choices has no feasible incumbent')
        selected.update(solution['solutions'][0]['selected_by_head'])
        total_cost += solution['solutions'][0]['cost']
        lower_bound += solution['lower_bound']
        complete = complete and solution['status'] == 'optimal'
    # Independent groups permit exact top-two composition by changing one group
    # to its own second solution. Multiple changed groups cannot be cheaper.
    alternatives = []
    for group in groups:
        local = group['assignment']
        if len(local['solutions']) > 1:
            alternative = dict(selected)
            alternative.update(local['solutions'][1]['selected_by_head'])
            alternative_cost = sum(by_id[choice].cost for choice in alternative.values())
            alternatives.append({'selected_by_head': alternative, 'cost': alternative_cost,
                                 'changed_group': group['id']})
    alternatives.sort(key=lambda row: (row['cost'], tuple(row['selected_by_head'][head] for head in sorted(row['selected_by_head']))))
    conflicts = conflicting_resources([by_id[value] for value in selected.values()])
    if conflicts:
        raise RuntimeError('Conflict decomposition produced an incompatible joint assignment')
    return {'status': 'optimal' if complete else 'limit', 'selected_by_head': selected,
            'cost': total_cost, 'lower_bound': lower_bound, 'upper_bound': total_cost,
            'runner_up': alternatives[0] if alternatives else None,
            'runner_up_proven': complete,
            'independent_by_head': independent,
            'independent_conflicts': conflicting_resources([by_id[value] for value in independent.values()]),
            'joint_conflicts': conflicts, 'groups': groups,
            'changed_head_ids': [head for head in sorted(selected) if selected[head] != independent[head]],
            'scope': 'Optimization over the supplied finite hypothesis pool; route search can remain incomplete.'}
