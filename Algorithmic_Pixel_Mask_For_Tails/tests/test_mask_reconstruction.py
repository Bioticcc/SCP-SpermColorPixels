import unittest
import numpy as np
import cv2

from mask_reconstruction import CrossingRegion, Track, reconstruct_tracks


class ReconstructionTests(unittest.TestCase):
    def test_thin_diagonal_uses_evidence_raster_convention(self):
        ev = np.zeros((31, 31), np.uint8)
        cv2.line(ev, (2, 2), (22, 12), 1, 1, lineType=cv2.LINE_8)
        result = reconstruct_tracks(ev, [Track('a', 'a', [(2, 2), (22, 12)])])
        self.assertTrue(np.array_equal(ev > 0, result['masks']['a']))
        self.assertEqual(result['diagnostics']['unsupported_centerline_pixels']['a'], 0)

    def test_unequal_width_crossing_recovers_both_masks_without_union_inflation(self):
        thin = np.zeros((45, 45), np.uint8)
        thick = thin.copy()
        cv2.line(thin, (2, 22), (42, 22), 1, 1)
        cv2.line(thick, (22, 2), (22, 42), 1, 7)
        region = np.zeros_like(thin, bool)
        region[16:29, 16:29] = True
        tracks = [Track('thin', 'a', [(2, 22), (42, 22)]), Track('thick', 'b', [(22, 2), (22, 42)])]
        out = reconstruct_tracks((thin | thick) > 0, tracks, [CrossingRegion('x', region, ('thin', 'thick'))])
        self.assertTrue(np.array_equal(out['masks']['thin'], thin > 0))
        self.assertTrue(np.array_equal(out['masks']['thick'], thick > 0))
        self.assertAlmostEqual(max(out['radii']['thin']), 0.5)
        self.assertGreater(out['diagnostics']['border_truncated_width_samples']['thick'], 0)

    def test_pairwise_permissions_cannot_authorize_an_unrelated_third_owner(self):
        ev = np.ones((13, 13), bool)
        tracks = [Track('a', 'a', [(5, 6)]), Track('b', 'b', [(6, 6)]), Track('c', 'c', [(7, 6)])]
        regions = [CrossingRegion('ab', ev, ('a', 'b')), CrossingRegion('bc', ev, ('b', 'c'))]
        result = reconstruct_tracks(ev, tracks, regions, {'min_radius_px': 2.0})
        self.assertFalse(np.any(result['masks']['a'] & result['masks']['c']))
        self.assertTrue(result['masks']['a'][6, 5])
        self.assertTrue(result['masks']['b'][6, 6])
        self.assertTrue(result['masks']['c'][6, 7])
        reverse = reconstruct_tracks(ev, list(reversed(tracks)), list(reversed(regions)), {'min_radius_px': 2.0})
        for key in result['masks']:
            self.assertTrue(np.array_equal(result['masks'][key], reverse['masks'][key]))

    def test_unsupported_and_outside_samples_are_reported_and_do_not_seed_tubes(self):
        ev = np.zeros((9, 9), bool)
        ev[4, 0:7] = True
        ev[4, 3] = False
        result = reconstruct_tracks(ev, [Track('a', 'a', [(-2, 4), (6, 4)])])
        self.assertEqual(result['diagnostics']['unsupported_centerline_pixels']['a'], 1)
        self.assertEqual(result['diagnostics']['out_of_bounds_centerline_samples']['a'], 2)
        self.assertFalse(result['masks']['a'][4, 3])

    def test_no_visible_width_evidence_is_explicitly_unknown_and_bounded(self):
        ev = np.ones((21, 21), bool)
        result = reconstruct_tracks(ev, [Track('a', 'a', [(5, 10), (15, 10)])],
                                    [CrossingRegion('j', ev, ('a',))])
        self.assertEqual(result['diagnostics']['width_unknown_crossing_samples']['a'], 11)
        self.assertEqual(result['diagnostics']['width_interpolated_samples']['a'], 0)
        self.assertEqual(max(result['radii']['a']), 0.5)
        self.assertEqual(int(result['masks']['a'].sum()), 11)

    def test_configuration_rejects_silent_coercions_and_unknown_keys(self):
        for config in ({'crossing_exclusion_px': 1.5}, {'crossing_exclusion_px': True},
                       {'min_radius_px': True}, {'tie_epsilon': float('nan')}, {'unused': 2}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    reconstruct_tracks(np.ones((3, 3), bool), [], config=config)

    def test_crossing_shares_only_declared_region_and_keeps_centerlines(self):
        ev=np.zeros((31,31),np.uint8); cv2.line(ev,(2,15),(28,15),1,3); cv2.line(ev,(15,2),(15,28),1,3); ev=ev.astype(bool)
        region=np.zeros_like(ev); region[13:18,13:18]=1
        out=reconstruct_tracks(ev,[Track('h','h',[(2,15),(28,15)]),Track('v','v',[(15,2),(15,28)])],[CrossingRegion('x',region,('h','v'))])
        self.assertTrue(np.all(out['masks']['h'][out['centerlines']['h']]))
        self.assertTrue(np.all(out['masks']['v'][out['centerlines']['v']]))
        self.assertEqual(0,out['diagnostics']['ordinary_duplicate_pixels'])

    def test_duplicate_centerline_outside_crossing_rejected(self):
        ev=np.zeros((10,10),bool); ev[5,1:9]=1
        with self.assertRaises(ValueError): reconstruct_tracks(ev,[Track('a','a',[(1,5),(8,5)]),Track('b','b',[(1,5),(8,5)])])

    def test_unordered_does_not_invent_chord_or_unsupported_pixels(self):
        ev=np.zeros((12,12),bool); ev[2,2]=ev[9,9]=1
        out=reconstruct_tracks(ev,[Track('u','u',[(2,2),(9,9)],ordered=False)])
        self.assertEqual(2,int(out['centerlines']['u'].sum()))
        self.assertFalse(out['masks']['u'][5,5])
        self.assertTrue(np.all(out['masks']['u'] <= ev))

    def test_null_boundary_and_determinism(self):
        ev=np.zeros((12,12),bool); ev[0:8,0]=1
        tracks=[Track('empty','e',[]),Track('edge','h',[(0,0),(0,7)])]
        a=reconstruct_tracks(ev,tracks); b=reconstruct_tracks(ev,tracks)
        self.assertFalse(a['masks']['empty'].any())
        self.assertTrue(np.array_equal(a['masks']['edge'],b['masks']['edge']))

    def test_single_track_crossing_region_is_width_exclusion_not_sharing(self):
        ev=np.zeros((15,15),bool); ev[7,1:14]=1
        region=np.zeros_like(ev); region[6:9,6:9]=1
        out=reconstruct_tracks(ev,[Track('a','a',[(1,7),(13,7)])],[CrossingRegion('j',region,('a',))])
        self.assertEqual(0,out['diagnostics']['ordinary_duplicate_pixels'])
        self.assertTrue(out['masks']['a'][7,7])


if __name__ == '__main__': unittest.main()
