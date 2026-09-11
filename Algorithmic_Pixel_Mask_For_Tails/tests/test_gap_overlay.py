from __future__ import annotations
import unittest
from graph_construction.gap_overlay import connected_subgraphs,merge_source_graphs,overlay_gap_edges
from path_hypotheses import search_hypotheses

def component(tail,roi,head,start,end,start_kind="head_anchor",end_kind="endpoint",y=5):
 width=roi[2]-roi[0]
 return {"tail_id":tail,"roi_xyxy":roi,"graph":{"shape":[roi[3]-roi[1],width],
  "nodes":[{"id":"start","xy":[1,y],"kind":start_kind,"pixels_xy":[[1,y]]},{"id":"end","xy":[width-2,y],"kind":end_kind,"pixels_xy":[[width-2,y]]}],
  "segments":[{"id":"stem","start":"start","end":"end","points_xy":[[1,y],[width-2,y]],"radii":[1,1]}],"anchors":{head:"start"}}}

class GapOverlayTests(unittest.TestCase):
 def setUp(self):
  self.components=[component("4",[10,10,30,25],"18","start","end"),component("9",[36,10,56,25],"18","start","end")]
  self.graph,self.provenance=merge_source_graphs(self.components,(50,80))
  self.gap={"id":"gap:4:end->9:end","endpoint_ids":["4:end","9:end"],"points_xy":[[28,15],[54,15]],"radius_px":1.25,"mean_support":.72,"distance_px":26}
 def test_source_offsets_resource_ids_and_reused_head_aliases(self):
  self.assertEqual(self.graph.nodes["4:end"].xy,(28.,15.)); self.assertEqual(self.graph.nodes["9:start"].pixels_xy,((37,15),))
  self.assertEqual(set(self.graph.segments),{"4:stem","9:stem"})
  self.assertEqual(self.provenance["original_segment_maps"]["4"]["stem"],"4:stem")
  self.assertEqual(set(self.graph.anchors),{"4@18","9@18"})
  self.assertEqual(self.provenance["anchor_aliases"]["9@18"],{"actual_head_id":"18","node_id":"9:start","tail_id":"9"})
 def test_gap_route_becomes_available_without_mutating_old_graph(self):
  before=self.graph.to_dict(); baseline=search_hypotheses(self.graph,"4@18",k=10,max_expansions=1000)
  self.assertFalse(any("9:stem" in row["segment_ids"] for row in baseline["hypotheses"]))
  overlaid=overlay_gap_edges(self.graph,[self.gap]); after=search_hypotheses(overlaid,"4@18",k=10,max_expansions=1000)
  self.assertTrue(any(self.gap["id"] in row["segment_ids"] and "9:stem" in row["segment_ids"] for row in after["hypotheses"]))
  self.assertEqual(self.graph.to_dict(),before); self.assertEqual(overlaid.nodes["4:end"].kind,"ordinary")
  segment=overlaid.segments[self.gap["id"]]; self.assertEqual(segment.points_xy[0],(28.,15.)); self.assertEqual(segment.points_xy[-1],(54.,15.))
  self.assertTrue(all(radius==1.25 for radius in segment.radii)); self.assertEqual(segment.evidence,.72)
  self.assertFalse(overlaid.diagnostics["gap_overlay"]["ownership_capacity_relaxed"])
 def test_invalid_missing_node_coordinates_anchor_boundary_and_duplicates(self):
  variants=[]
  missing={**self.gap,"endpoint_ids":["4:end","missing"]}; variants.append((self.graph,[missing]))
  mismatch={**self.gap,"points_xy":[[27,15],[54,15]]}; variants.append((self.graph,[mismatch]))
  anchor={**self.gap,"endpoint_ids":["4:start","9:start"],"points_xy":[[11,15],[37,15]]}; variants.append((self.graph,[anchor]))
  boundary_components=[component("4",[10,10,30,25],"18","start","end",end_kind="boundary"),self.components[1]]
  boundary_graph,_=merge_source_graphs(boundary_components,(50,80)); variants.append((boundary_graph,[self.gap]))
  duplicate={**self.gap,"id":"another"}; variants.append((self.graph,[self.gap,duplicate]))
  variants.append((self.graph,[self.gap,{**self.gap}]))
  for graph,gaps in variants:
   with self.subTest(gaps=gaps):
    with self.assertRaises(ValueError): overlay_gap_edges(graph,gaps)
 def test_multiple_gap_choices_share_base_endpoint_but_search_stays_bounded(self):
  third=component("12",[36,28,56,43],"22","start","end",y=5)
  graph,_=merge_source_graphs([self.components[0],self.components[1],third],(50,80))
  gaps=[self.gap,{"id":"gap:4:end->12:end","endpoint_ids":["4:end","12:end"],"points_xy":[[28,15],[54,33]],"radius_px":1.,"mean_support":.6}]
  overlaid=overlay_gap_edges(graph,gaps); result=search_hypotheses(overlaid,"4@18",k=10,max_expansions=200)
  routes=result["hypotheses"]; self.assertLessEqual(len(routes),10); self.assertFalse(result["diagnostics"]["truncated"])
  self.assertTrue(any(gaps[0]["id"] in row["segment_ids"] for row in routes)); self.assertTrue(any(gaps[1]["id"] in row["segment_ids"] for row in routes))
 def test_merge_validation(self):
  with self.assertRaises(ValueError): merge_source_graphs(self.components,(0,80))
  with self.assertRaises(ValueError): merge_source_graphs(self.components+[self.components[0]],(50,80))
 def test_connected_subgraphs_preserve_resources_aliases_and_isolated_nodes(self):
  from graph_construction import Node,SegmentGraph
  nodes=dict(self.graph.nodes); nodes["zz_isolated"]=Node("zz_isolated",(70.,45.),"ordinary")
  graph=SegmentGraph(nodes,self.graph.segments,self.graph.anchors,self.graph.shape,self.graph.diagnostics)
  views=connected_subgraphs(graph)
  self.assertEqual([min(view.nodes) for view in views],["4:end","9:end","zz_isolated"])
  self.assertEqual({key for view in views for key in view.segments},set(graph.segments))
  self.assertEqual(sum(len(view.segments) for view in views),len(graph.segments))
  self.assertTrue(any(set(view.anchors)=={"4@18"} for view in views))
  self.assertTrue(any(set(view.nodes)=={"zz_isolated"} and not view.segments for view in views))

if __name__=="__main__": unittest.main()
