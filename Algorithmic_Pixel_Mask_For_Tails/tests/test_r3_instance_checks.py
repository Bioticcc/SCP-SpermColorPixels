import tempfile, unittest
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.r3_instance_checks import check_instance

class TestChecks(unittest.TestCase):
 def make(self):
  d=Path(tempfile.mkdtemp()); rgb=np.zeros((9,9,3),np.uint8); rgb[:,:,0]=np.arange(9); head=np.zeros((9,9),int); head[1:3,1:3]=2; tail=np.zeros((9,9),int); tail[2,2:7]=4; roi=[1,1,8,5]; box=[1,1,8,5]; p={'id':'x','head_id':'2','tail_id':'4','termination':'endpoint','metadata':{'tail_id':'4','points_xy_ordered':[[1,1],[5,1]]}}; f={}
  arrays={'original_crop':rgb[1:5,1:8],'tail_mask':(tail[1:5,1:8]==4)*255,'head_mask':(head[1:5,1:8]==2)*255,'instance_mask':((tail[1:5,1:8]==4)|(head[1:5,1:8]==2))*255,'centerline':np.zeros((4,7),np.uint8),'highlighted_crop':rgb[1:5,1:8]}; arrays['centerline'][1,1:6]=255
  for k,a in arrays.items(): Image.fromarray(a.astype(np.uint8)).save(d/(k+'.png')); f[k]=k+'.png'
  return d,{'id':'x','head_id':'2','tail_id':'4','termination':'endpoint','roi_xyxy':box,'files':f},rgb,head,tail,p,{'tail_id':4,'roi_xyxy':roi}
 def test_valid_and_tamper(self):
  a=self.make(); self.assertEqual(5,check_instance(*a)['centerline_pixels']); Image.fromarray(np.ones((4,7,3),np.uint8)).save(a[0]/'original_crop.png');
  with self.assertRaises(ValueError): check_instance(*a)
 def test_missing(self):
  a=self.make(); (a[0]/'centerline.png').unlink()
  with self.assertRaises(ValueError): check_instance(*a)
 def test_v2_head_evidence_and_tamper(self):
  a=self.make(); d,ins,rgb,heads,tails,hyp,comp=a
  raw=(heads[1:5,1:8]==2); z=np.zeros((4,7),np.uint8)
  for key,value in [('head_evidence_mask',raw*255),('ownership_uncertainty_mask',z),('head_priority_instance_mask',((raw|(tails[1:5,1:8]==4))*255))]:
   Image.fromarray(value.astype(np.uint8)).save(d/(key+'.png')); ins['files'][key]=key+'.png'
  counts=np.zeros((9,9),int); counts[2,2:7]=1
  self.assertEqual(4,check_instance(*a,selected_centerline_counts=counts,selected_head_ids=[2])['head_evidence_pixels'])
  Image.fromarray(np.ones((4,7),np.uint8)*255).save(d/'ownership_uncertainty_mask.png')
  with self.assertRaises(ValueError): check_instance(*a,selected_centerline_counts=counts,selected_head_ids=[2])

 def test_head_outside_crop_is_not_silently_lost(self):
  a=self.make(); a[3][8,8]=2
  with self.assertRaisesRegex(ValueError, 'head label was truncated'): check_instance(*a)
 def test_centerline_outside_crop_is_not_silently_lost(self):
  a=self.make(); a[1]['roi_xyxy'][2]=6
  for filename in a[1]['files'].values():
   path=a[0]/filename; pixels=np.asarray(Image.open(path)); Image.fromarray(pixels[:,:5]).save(path)
  with self.assertRaisesRegex(ValueError, 'centerline was truncated'): check_instance(*a)

 def make_v2_conflict(self):
  """Build expected ownership masks independently of the production helper."""
  d=Path(tempfile.mkdtemp()); shape=(12,16); rgb=np.zeros((*shape,3),np.uint8); rgb[:,:,1]=np.arange(shape[1])
  heads=np.zeros(shape,int); heads[5:8,1:4]=2; heads[5:8,10:13]=3
  tails=np.zeros(shape,int); tails[6,3:13]=4; tails[4:9,8:13]=4
  line=np.zeros(shape,bool); line[6,3:13]=True
  foreign_line=np.zeros(shape,bool); foreign_line[5:8,2]=True
  counts=line.astype(int)+foreign_line.astype(int)
  raw_head=heads==2; foreign_heads=heads==3
  # The default tail keeps pinned pixels but drops all other growth in B's head.
  candidate_tail=tails==4
  logical_tail=candidate_tail & ~(foreign_heads & ~line)
  logical_head=raw_head & ~(foreign_line)
  uncertainty=(raw_head & foreign_line)|(line & foreign_heads)
  head_priority=raw_head|(logical_tail & ~foreign_heads)
  arrays={'original_crop':rgb,'tail_mask':logical_tail*255,'head_mask':logical_head*255,
   'instance_mask':(logical_tail|logical_head)*255,'centerline':line*255,'highlighted_crop':rgb,
   'head_evidence_mask':raw_head*255,'ownership_uncertainty_mask':uncertainty*255,
   'head_priority_instance_mask':head_priority*255}
  files={}
  for key,value in arrays.items():
   Image.fromarray(value.astype(np.uint8)).save(d/(key+'.png')); files[key]=key+'.png'
  instance={'id':'conflict','head_id':'2','tail_id':'4','termination':'endpoint','roi_xyxy':[0,0,16,12],'files':files}
  hypothesis={'id':'conflict','head_id':'2','tail_id':'4','termination':'endpoint',
   'metadata':{'tail_id':'4','points_xy_ordered':[[3,6],[12,6]]}}
  component={'tail_id':4,'roi_xyxy':[0,0,16,12]}
  return d,instance,rgb,heads,tails,hypothesis,component,counts,[2,3],uncertainty

 def test_v2_two_head_conflicts_and_exact_nine_exports_succeed(self):
  a=self.make_v2_conflict(); d,instance,*rest=a
  self.assertEqual(len(instance['files']),9)
  _rgb,heads,tails,_hypothesis,_component,counts,_ids,uncertainty=rest
  self.assertEqual(int(((heads==2)&(counts>0)&~(np.asarray(Image.open(d/'centerline.png')).astype(bool))).sum()),3)
  self.assertEqual(int(((heads==3)&np.asarray(Image.open(d/'centerline.png')).astype(bool)).sum()),3)
  self.assertEqual(tails[5,10],4); self.assertEqual(counts[5,10],0)  # foreign-head growth, not a pin
  self.assertEqual(np.asarray(Image.open(d/'tail_mask.png'))[5,10],0)
  result=check_instance(d,instance,*rest[:5],selected_centerline_counts=rest[5],selected_head_ids=rest[6])
  self.assertEqual(result['pinned_tail_pixels_in_foreign_head'],3)
  self.assertEqual(result['head_evidence_pixels'],9)
  self.assertEqual(int(uncertainty.sum()),6)  # three pixels for each candidate-owner conflict

 def test_v2_tampering_each_ownership_export_is_caught(self):
  for key,point in [('head_mask',(5,2)),('ownership_uncertainty_mask',(0,0)),
                    ('tail_mask',(5,10)),('head_priority_instance_mask',(0,0))]:
   with self.subTest(key=key):
    a=self.make_v2_conflict(); d,instance,*rest=a
    path=d/instance['files'][key]; pixels=np.asarray(Image.open(path)).copy(); pixels[point]=255 if pixels[point]==0 else 0
    Image.fromarray(pixels).save(path)
    with self.assertRaises(ValueError):
     check_instance(d,instance,*rest[:5],selected_centerline_counts=rest[5],selected_head_ids=rest[6])
