import unittest
from experiments.r4b_pool import extend_pool
class Pool(unittest.TestCase):
 def test_preserves_and_adds(self):
  old={'width':10,'height':10,'hypotheses':[{'id':'null','head_id':'h','cost':1.5,'exclusive_segments':(), 'endpoint_id':None,'termination':'null','metadata':{}}]}
  r={'head_id':'h','tail_id':'t','segment_ids':['t:s'],'points_xy':[[1,1],[8,1]],'termination':'endpoint','endpoint_id':'end','costs':{'direction':0,'curvature':0,'evidence':0}}
  o=extend_pool(old,[r],{'null_cost':1.5,'partial_cost':.85,'coverage_weight':1});self.assertEqual(old['hypotheses'][0],o['hypotheses'][0]);self.assertEqual(('t:s',),o['added'][0].exclusive_segments)
