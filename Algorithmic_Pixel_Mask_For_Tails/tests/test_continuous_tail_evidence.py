import unittest,numpy as np,cv2
from tail_evidence import continuous_scp_evidence,sample_orientation_peaks
class Evidence(unittest.TestCase):
 def test_score_and_validation(self):
  a=np.full((20,20,3),240,np.uint8); a[8:12,:,]=[120,80,40]; out=continuous_scp_evidence(a,{'sat_min':20,'local_dark_min':1,'value_max':250}); self.assertEqual(np.float32,out['score'].dtype); self.assertTrue((out['score']>=0).all())
  with self.assertRaises(ValueError):continuous_scp_evidence(a,{'sat_min':0,'local_dark_min':1,'value_max':250})
 def test_orientation(self):
  s=np.zeros((40,40),np.float32); s[19:22,:]=1; peaks=sample_orientation_peaks(s,[[20,20]],[2],angles=12); self.assertTrue(peaks[0]); self.assertLess(abs(peaks[0][0]['angle_radians_mod_pi']),.3)
  self.assertEqual([],sample_orientation_peaks(np.ones((20,20),np.float32),[[10,10]],[1])[0])
 def test_border_no_peak(self):
  s=np.zeros((20,20),np.float32);s[:,0:2]=1;self.assertEqual([],sample_orientation_peaks(s,[[0,10]],[2])[0])
