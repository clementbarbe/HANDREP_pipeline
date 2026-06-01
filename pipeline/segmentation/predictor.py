"""
Two-stage inference: coarse localisation then crop-based refinement.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.config.settings import NN_IMG_SIZE, NN_HEATMAP_SIZE, NN_CROP_SIZE
from pipeline.segmentation.models import SimpleHRNet, RefineNet


def soft_argmax(hm, beta=100):
    """
    Differentiable argmax via spatial softmax.

    Parameters
    ----------
    hm : Tensor (B, 1, H, W)

    Returns
    -------
    (x, y) : float
    """
    B, _, H, W = hm.shape
    flat = F.softmax(hm.view(B, -1) * beta, dim=1)
    idx = torch.arange(H * W, device=hm.device, dtype=torch.float32)
    cx = (flat * (idx % W)).sum(dim=1)
    cy = (flat * (idx // W)).sum(dim=1)
    return cx[0].item(), cy[0].item()


class Predictor:
    """
    Loads trained weights and runs two-stage prediction.

    Parameters
    ----------
    coarse_path, refine_path : str
        Paths to ``.pth`` weight files.
    device : torch.device, optional
    """

    def __init__(self, coarse_path, refine_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.coarse = SimpleHRNet().to(self.device)
        self.coarse.load_state_dict(
            torch.load(coarse_path, map_location=self.device, weights_only=True)
        )
        self.coarse.eval()

        self.refine = RefineNet().to(self.device)
        self.refine.load_state_dict(
            torch.load(refine_path, map_location=self.device, weights_only=True)
        )
        self.refine.eval()

    # ── helpers ────────────────────────────────────────────
    def _to_tensor(self, img_rgb, size):
        img = cv2.resize(img_rgb, size)
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    # ── public API ────────────────────────────────────────
    @torch.no_grad()
    def predict(self, img_bgr):
        """
        Predict the (x, y) keypoint in original pixel coordinates.

        Parameters
        ----------
        img_bgr : ndarray (H, W, 3) — OpenCV BGR image.

        Returns
        -------
        (int, int)
        """
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # Coarse
        tensor = self._to_tensor(img, (NN_IMG_SIZE, NN_IMG_SIZE))
        hm = self.coarse(tensor)
        hx, hy = soft_argmax(hm, beta=100)
        coarse_x = hx * w_orig / NN_HEATMAP_SIZE
        coarse_y = hy * h_orig / NN_HEATMAP_SIZE

        # Crop
        cx, cy = int(coarse_x), int(coarse_y)
        half = NN_CROP_SIZE // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w_orig, x1 + NN_CROP_SIZE)
        y2 = min(h_orig, y1 + NN_CROP_SIZE)
        x1, y1 = max(0, x2 - NN_CROP_SIZE), max(0, y2 - NN_CROP_SIZE)

        crop = img[y1:y2, x1:x2]
        crop_t = self._to_tensor(crop, (NN_CROP_SIZE, NN_CROP_SIZE))

        # Refine
        offset = self.refine(crop_t)[0].cpu().numpy()
        final_x = coarse_x + float(offset[0])
        final_y = coarse_y + float(offset[1])

        return int(round(final_x)), int(round(final_y))