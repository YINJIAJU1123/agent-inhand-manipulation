import unittest,torch
from sensors import object_regions

class SensorEvidenceTests(unittest.TestCase):
    def test_visible_rgb_region_contains_object_and_preserves_depth(self):
        rgb=torch.zeros(1,128,128,3,dtype=torch.uint8);rgb[:,40:80,50:90]=torch.tensor([217,13,13],dtype=torch.uint8)
        out=object_regions(rgb,torch.full((1,128,128,1),.6))
        self.assertTrue(bool(out['roi_found'][0]))
        self.assertEqual(tuple(out['roi_rgb'].shape),(1,3,96,96))
        x0,y0,x1,y1=out['roi_box'][0].tolist()
        self.assertLessEqual(x0,50/128);self.assertGreaterEqual(x1,90/128)
        self.assertLessEqual(y0,40/128);self.assertGreaterEqual(y1,80/128)
        self.assertTrue(torch.allclose(out['roi_depth'].float(),torch.full((1,1,96,96),.3),atol=1e-3))
    def test_missing_color_uses_disclosed_full_view(self):
        out=object_regions(torch.zeros(1,128,128,3,dtype=torch.uint8),torch.full((1,128,128,1),float('nan')))
        self.assertFalse(bool(out['roi_found'][0]))
        self.assertEqual(out['roi_box'].tolist(),[[0.,0.,1.,1.]])
        self.assertTrue(bool(torch.isfinite(out['roi_depth']).all()))
if __name__=='__main__':unittest.main()
