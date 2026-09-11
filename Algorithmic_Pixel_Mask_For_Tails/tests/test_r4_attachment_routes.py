import unittest
from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses
from head_anchors import extend_attachment_routes

class AttachmentRoutes(unittest.TestCase):
 def test_foreign_anchor_continuation_preserves_baseline(self):
  g=from_explicit_graph({'nodes':[{'id':'a','xy':[1,1],'kind':'head_anchor'},{'id':'b','xy':[5,1],'kind':'head_anchor'},{'id':'e','xy':[9,1],'kind':'endpoint'}],'edges':[{'id':'ab','start':'a','end':'b','points':[[1,1],[5,1]]},{'id':'be','start':'b','end':'e','points':[[5,1],[9,1]]}]})
  base=search_hypotheses(g,'a',k=5); out=extend_attachment_routes(g,'a',base)
  self.assertEqual(base['hypotheses'],out['hypotheses'][:len(base['hypotheses'])])
  added=out['hypotheses'][len(base['hypotheses']):]
  self.assertTrue(any(x['segment_ids']==['ab','be'] for x in added))
  self.assertTrue(all(x['metadata']['passed_foreign_head_ids']==['b'] for x in added))
 def test_deterministic_and_isolated(self):
  g=from_explicit_graph({'nodes':[{'id':'a','xy':[1,1],'kind':'head_anchor'},{'id':'e','xy':[5,1],'kind':'endpoint'}],'edges':[{'id':'ae','start':'a','end':'e','points':[[1,1],[5,1]]}]})
  b=search_hypotheses(g,'a'); self.assertEqual(b['hypotheses'],extend_attachment_routes(g,'a',b)['hypotheses'])

class AttachmentSafety(unittest.TestCase):
 def graph(self):
  return from_explicit_graph({'nodes':[{'id':'a','xy':[1,1],'kind':'head_anchor'},{'id':'b','xy':[5,1],'kind':'head_anchor'},{'id':'c','xy':[9,1],'kind':'endpoint'}], 'edges':[{'id':'ab','start':'a','end':'b','points':[[1,1],[5,1]]},{'id':'bc','start':'b','end':'c','points':[[5,1],[9,1]]}]}, anchors={'a':'a','b':'b','c':'c'})
 def test_foreign_terminal_stays_partial_and_inputs_unchanged(self):
  import copy
  g=self.graph(); base=search_hypotheses(g,'a'); before=copy.deepcopy(base); geometry=g.to_dict()
  out=extend_attachment_routes(g,'a',base); extra=out['hypotheses'][len(base['hypotheses']):]
  self.assertEqual(len(extra),1); self.assertEqual(extra[0]['termination'],'partial')
  self.assertEqual(extra[0]['metadata']['passed_foreign_head_ids'],['b'])
  self.assertEqual(extra[0]['costs']['termination'],.25)
  self.assertEqual(base,before);self.assertEqual(g.to_dict(),geometry)
  self.assertEqual(out,extend_attachment_routes(g,'a',base))
 def test_budget_and_invalid_baseline(self):
  import copy
  g=self.graph(); base=search_hypotheses(g,'a')
  limited=extend_attachment_routes(g,'a',base,max_expansions=1)
  self.assertTrue(limited['attachment_extension']['extension_search']['truncated'])
  self.assertEqual(limited['hypotheses'],base['hypotheses'])
  for kwargs in ({'additional_k':True},{'additional_k':-1},{'max_expansions':0}):
   with self.assertRaises(ValueError):extend_attachment_routes(g,'a',base,**kwargs)
  altered=copy.deepcopy(base);altered['hypotheses'].append(altered['hypotheses'][0])
  with self.assertRaises(ValueError):extend_attachment_routes(g,'a',altered)
 def test_collocated_foreign_start_is_not_passage(self):
  from dataclasses import replace
  g=replace(self.graph(),anchors={'a':'a','b':'a'}); base=search_hypotheses(g,'a')
  out=extend_attachment_routes(g,'a',base)
  self.assertEqual(out['hypotheses'],base['hypotheses'])
