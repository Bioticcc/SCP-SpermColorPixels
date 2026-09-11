import unittest,numpy as np
from graph_construction import from_explicit_graph
from head_anchors.candidates import propose_head_anchors
class Anchors(unittest.TestCase):
 def test_baseline_and_ranked_endpoints(self):
  g=from_explicit_graph({'nodes':[{'id':'a','xy':[2,5],'kind':'endpoint'},{'id':'b','xy':[8,5],'kind':'endpoint'},{'id':'j','xy':[5,5],'kind':'junction'}],'edges':[{'id':'aj','start':'a','end':'j','points':[[2,5],[5,5]]},{'id':'bj','start':'b','end':'j','points':[[8,5],[5,5]]}]})
  m=np.zeros((12,12),bool);m[4:7,4:7]=1;s=np.zeros((12,12),np.float32);s[5,2:9]=.5
  out=propose_head_anchors(g,m,s,['a']);self.assertEqual('a',out[0]['node_id']);self.assertTrue(any(x['node_id']=='b' for x in out))
 def test_empty_head(self):
  g=from_explicit_graph({'nodes':[],'edges':[]});self.assertEqual([],propose_head_anchors(g,np.zeros((2,2),bool),np.zeros((2,2),np.float32),[]))
