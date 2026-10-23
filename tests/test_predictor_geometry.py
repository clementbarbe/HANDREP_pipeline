"""Geometry checks with deterministic dummy networks; no training weights needed."""

import unittest
import numpy as np

try:
    import torch
    from pipeline.segmentation.predictor import Predictor
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional Torch dependency is unavailable")
class PredictorGeometryTests(unittest.TestCase):
    def predictor(self, heatmap_xy, offset):
        predictor = object.__new__(Predictor)
        predictor.device = torch.device("cpu")
        def coarse(_):
            heatmap = torch.zeros((1, 1, 128, 128))
            heatmap[0, 0, heatmap_xy[1], heatmap_xy[0]] = 10
            return heatmap
        predictor.coarse = coarse
        predictor.refine = lambda _: torch.tensor([offset], dtype=torch.float32)
        return predictor

    def test_zero_offset_returns_actual_crop_center_near_border(self):
        predictor = self.predictor((0, 0), (0., 0.))
        self.assertEqual(predictor.predict(np.zeros((1080, 1920, 3), np.uint8)), (64, 64))

    def test_crop_offset_maps_to_original_image(self):
        predictor = self.predictor((0, 0), (.5, -.5))
        self.assertEqual(predictor.predict(np.zeros((1080, 1920, 3), np.uint8)), (96, 32))


if __name__ == "__main__":
    unittest.main()
