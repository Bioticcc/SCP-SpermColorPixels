import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.run_r4a import validate_config
from experiments.audit_r4a import audit,check_routes,deterministic_assignment
from experiments.r4_adapter import extend_image
from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses

class R4AuditTests(unittest.TestCase):
 def test_config_requires_explicit_controls(self):
  p=Path(__file__).resolve().parents[2]/'configs/overlap_r4a.json';config=json.loads(p.read_text());validate_config(config)
  for key in config:
   changed=dict(config);changed.pop(key)
   with self.assertRaises(ValueError):validate_config(changed)
 def test_routes_preserved_supported_and_tamper_detected(self):
  graph=from_explicit_graph({'nodes':[{'id':'a','xy':[1,1],'kind':'head_anchor'},{'id':'b','xy':[5,1],'kind':'head_anchor'},{'id':'e','xy':[9,1],'kind':'endpoint'}],'edges':[{'id':'ab','start':'a','end':'b','points':[[1,1],[5,1]]},{'id':'be','start':'b','end':'e','points':[[5,1],[9,1]]}]})
  record={'relative_image':'x','width':12,'height':12,'baseline_candidates':0,'components':[{'tail_id':1,'roi_xyxy':[0,0,12,12],'graph':graph.to_dict(),'baseline':[],'searches':{h:search_hypotheses(graph,h) for h in graph.anchors}}]}
  extended,_=extend_image(record,{'additional_k':5,'search_max_expansions':20000})
  self.assertGreater(check_routes(record,extended,5)['added_routes'],0)
  changed=copy.deepcopy(extended);changed['components'][0]['searches']['a']['hypotheses'][-1]['points_xy'][0][0]+=1
  with self.assertRaisesRegex(ValueError,'geometry'):check_routes(record,changed,5)
  changed=copy.deepcopy(extended);changed['components'][0]['searches']['a']['hypotheses'][0]['score']=9
  with self.assertRaisesRegex(ValueError,'Original route'):check_routes(record,changed,5)
 def test_missing_release_is_failed(self):
  with tempfile.TemporaryDirectory() as temp:self.assertFalse(audit(Path(temp))['passed'])

 def test_determinism_excludes_only_recorded_wall_clock_time(self):
  a={'cost':1.2,'diagnostics':{'runtime_seconds':.1,'expansions':5}}
  b=copy.deepcopy(a);b['diagnostics']['runtime_seconds']=.4
  self.assertEqual(deterministic_assignment(a),deterministic_assignment(b))
  b['cost']=1.3
  self.assertNotEqual(deterministic_assignment(a),deterministic_assignment(b))
