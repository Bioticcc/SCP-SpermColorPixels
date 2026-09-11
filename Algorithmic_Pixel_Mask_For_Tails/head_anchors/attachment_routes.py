"""R4a optional continuations through incidental foreign head anchors."""
from __future__ import annotations
from dataclasses import replace
import copy
from path_hypotheses import search_hypotheses

def extend_attachment_routes(graph, head_id, baseline_search, additional_k=5, max_expansions=20000):
    if head_id not in graph.anchors: raise KeyError(head_id)
    if type(additional_k) is not int or additional_k < 0 or type(max_expansions) is not int or max_expansions < 1: raise ValueError("invalid limits")
    base = copy.deepcopy(baseline_search)
    hypotheses = base.get("hypotheses")
    if not isinstance(hypotheses,list) or base.get("head_id") != head_id: raise ValueError("invalid baseline search")
    ids=[item.get("id") for item in hypotheses]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate baseline hypothesis IDs")
    if additional_k == 0 or len(graph.anchors) <= 1:
        base["attachment_extension"]={"added_count":0,"extension_search":None,"reason":"isolated_or_disabled"}; return base
    reduced=replace(graph, anchors={head_id:graph.anchors[head_id]})
    extension=search_hypotheses(reduced,head_id,k=additional_k+len(hypotheses),max_expansions=max_expansions)
    foreign={key:value for key,value in graph.anchors.items() if key != head_id}; known={(tuple(x.get("segment_ids",())),tuple(x.get("directions",())),x.get("termination")) for x in hypotheses}
    added=[]
    for route in extension["hypotheses"]:
        if route["termination"] == "null": continue
        nodes=route.get("node_ids",[]); passed=sorted(key for key,node in foreign.items() if node in nodes[1:-1])
        if not passed: continue
        changed=copy.deepcopy(route)
        if nodes and nodes[-1] in foreign.values():
            changed["termination"]="partial"; costs=changed["costs"]; costs["termination"]=0.25; costs["total"]=sum(costs[key] for key in ("direction","curvature","evidence","termination")); changed["score"]=costs["total"]
        key=(tuple(changed.get("segment_ids",())),tuple(changed.get("directions",())),changed.get("termination"))
        if key in known: continue
        changed["id"]=f"{head_id}:r4_attachment:{len(added)+1}"
        changed["metadata"]={"passed_foreign_head_ids":passed}
        if changed["id"] in ids: raise ValueError("attachment route ID collision")
        ids.append(changed["id"]); known.add(key); added.append(changed)
        if len(added) >= additional_k: break
    base["hypotheses"] = hypotheses + added
    base["attachment_extension"]={"added_count":len(added),"extension_search":extension.get("diagnostics"),"preserved_baseline_count":len(hypotheses)}
    return base
