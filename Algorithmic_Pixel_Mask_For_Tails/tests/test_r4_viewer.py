import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.r4_viewer import write_image_viewer,write_index
class Viewer(unittest.TestCase):
 def test_pages(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); r1=root/'r1'; r1.mkdir(); Image.fromarray(np.zeros((9,9,3),np.uint8)).save(r1/'original.jpg')
   item={'id':'a:r4_attachment:1','head_id':'a','termination':'endpoint','metadata':{'points_xy_ordered':[[1,1],[7,1]],'roi_xyxy':[0,0,9,9]}}
   p={'relative_image':'x','width':9,'height':9,'hypotheses':[item],'assignment':{'selected_by_head':{'a':item['id']}},'frozen_r2':{'assignment':{'selected_by_head':{'a':item['id']}}}}
   out=root/'images'/'x'; s=write_image_viewer(out,r1,p); self.assertTrue((out/'index.html').is_file()); write_index(root,[s],{'cases':[]},{'images':[]}); self.assertTrue((root/'index.html').is_file())
