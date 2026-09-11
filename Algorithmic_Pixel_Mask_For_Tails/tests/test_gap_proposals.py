import copy
import unittest
import numpy as np
from tail_evidence.gaps import propose_gap_edges

class GapProposalTests(unittest.TestCase):
 def case(self):
  score=np.zeros((64,96),np.float32);score[30:33,10:86]=1;score[30:33,44:52]=.4
  endpoints=[{'id':'left','component_id':1,'point_xy':[43,31],'outward_xy':[1,0],'radius_px':2},
             {'id':'right','component_id':2,'point_xy':[52,31],'outward_xy':[-1,0],'radius_px':2}]
  return score,endpoints
 def test_faint_supported_gap_is_inspectable_and_deterministic(self):
  score,points=self.case();before=score.copy();original=copy.deepcopy(points)
  a=propose_gap_edges(score,points);b=propose_gap_edges(score,list(reversed(points)))
  self.assertEqual(a,b);self.assertEqual(len(a['candidates']),1)
  gap=a['candidates'][0];self.assertEqual(gap['component_ids'],[1,2]);self.assertEqual(gap['unsupported_distance_px'],0)
  self.assertEqual(gap['points_xy'],[[43.,31.],[52.,31.]])
  self.assertEqual(points,original);self.assertTrue(np.array_equal(score,before))
 def test_blank_gap_and_wrong_direction_rejected(self):
  score,points=self.case();score[30:33,44:52]=0
  self.assertEqual(propose_gap_edges(score,points)['candidates'],[])
  score,points=self.case();points[1]['outward_xy']=[1,0]
  self.assertEqual(propose_gap_edges(score,points)['diagnostics']['geometry_eligible_pairs'],0)
 def test_head_intersection_is_not_a_tail_gap(self):
  score,points=self.case();heads=np.zeros(score.shape,np.int32);heads[31,48]=3
  out=propose_gap_edges(score,points,head_labels=heads)
  self.assertEqual(out['candidates'],[]);self.assertEqual(out['diagnostics']['rejected']['detected_head_intersection'],1)
 def test_uniform_color_is_not_ridge_support(self):
  score,points=self.case();score[:]=.8
  out=propose_gap_edges(score,points);self.assertEqual(out['candidates'],[])
  self.assertEqual(out['diagnostics']['rejected']['ridge_support'],1)
 def test_invalid_geometry_and_fields(self):
  score,points=self.case()
  with self.assertRaises(ValueError):propose_gap_edges(score,[points[0],points[0]])
  with self.assertRaises(ValueError):propose_gap_edges(score,points,{'max_evaluations':True})
  with self.assertRaises(ValueError):propose_gap_edges(score.astype(float),points)
  self.assertEqual(propose_gap_edges(score,[])['candidates'],[])
