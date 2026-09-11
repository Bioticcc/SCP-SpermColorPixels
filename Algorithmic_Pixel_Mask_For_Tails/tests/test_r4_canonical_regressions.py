"""Optional attachment routes leave the existing canonical corpus unchanged."""
import json
from pathlib import Path
import unittest
from tests.fixtures import case_names,generate_case
from experiments.r1_benchmark import _rendered_graph
from experiments.r4_benchmark import _record,_frozen,_searches
from experiments.r4_adapter import resolve_image
from graph_construction import from_explicit_graph

class R4CanonicalRegressions(unittest.TestCase):
 def test_existing_explicit_and_rendered_selections_unchanged(self):
  config=json.loads((Path(__file__).resolve().parents[2]/'configs/overlap_r4a.json').read_text())
  for name in case_names():
   case=generate_case(name,seed=0)
   for mode in ('explicit','rendered'):
    with self.subTest(case=name,mode=mode):
     graph=from_explicit_graph(case.truth['graph'],shape=case.tail_mask.shape) if mode=='explicit' else _rendered_graph(case)[0]
     record=_record(case,graph,_searches(graph,config),name);old=_frozen(record,config);_,new=resolve_image(record,old,config)
     self.assertEqual(sum(row['added_count'] for row in new['extension_diagnostics']),0)
     self.assertEqual(new['assignment']['selected_by_head'],old['assignment']['selected_by_head'])
