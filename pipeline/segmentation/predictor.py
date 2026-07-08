"""
Two-stage inference: coarse localisation then crop-based refinement.

RefineNet outputs a **Tanh-normalised** offset in [-1, 1].
The pixel offset is recovered by multiplying by ``CROP_SIZE / 2``.
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
        Coordinates in heatmap space.
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

    Coarse → soft-argmax → crop → RefineNet (Tanh) → denormalise → final (x, y).

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

        # ── Coarse (SimpleHRNet) ──────────────────────────
        self.coarse = SimpleHRNet().to(self.device)
        self.coarse.load_state_dict(
            torch.load(coarse_path, map_location=self.device, weights_only=True)
        )
        self.coarse.eval()

        # ── Refine (RefineNet with Tanh) ──────────────────
        self.refine = RefineNet().to(self.device)
        self.refine.load_state_dict(
            torch.load(refine_path, map_location=self.device, weights_only=True)
        )
        self.refine.eval()

        print(f"[Predictor] Models loaded on {self.device}")
        print(f"[Predictor] Crop size = {NN_CROP_SIZE}, "
              f"RefineNet output = Tanh (denorm × {NN_CROP_SIZE // 2})")

    # ── Helpers ───────────────────────────────────────────

    def _to_tensor(self, img_rgb, size):
        """
        Resize and convert to normalised [0, 1] tensor.
        No ImageNet normalisation — consistent with training.
        """
        img = cv2.resize(img_rgb, size)
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    # ── Public API ────────────────────────────────────────

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
            Predicted pixel coordinates in the original image.
        """
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        half = NN_CROP_SIZE // 2

        # ── Stage 1: Coarse ───────────────────────────────
        tensor = self._to_tensor(img, (NN_IMG_SIZE, NN_IMG_SIZE))
        heatmap = self.coarse(tensor)                 # (1, 1, 128, 128)
        hx, hy = soft_argmax(heatmap, beta=100)

        # Heatmap coords → 512-space → original-image space
        coarse_x = hx * NN_IMG_SIZE / NN_HEATMAP_SIZE  # heatmap → 512
        coarse_y = hy * NN_IMG_SIZE / NN_HEATMAP_SIZE
        coarse_x = coarse_x * w_orig / NN_IMG_SIZE     # 512 → original
        coarse_y = coarse_y * h_orig / NN_IMG_SIZE

        # ── Stage 2: Crop ─────────────────────────────────
        cx, cy = int(coarse_x), int(coarse_y)

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w_orig, cx + half)
        y2 = min(h_orig, cy + half)

        # Ensure exact crop size even near borders
        if x2 - x1 < NN_CROP_SIZE:
            if x1 == 0:
                x2 = min(w_orig, NN_CROP_SIZE)
            else:
                x1 = max(0, x2 - NN_CROP_SIZE)
        if y2 - y1 < NN_CROP_SIZE:
            if y1 == 0:
                y2 = min(h_orig, NN_CROP_SIZE)
            else:
                y1 = max(0, y2 - NN_CROP_SIZE)

        crop = img[y1:y2, x1:x2]
        crop_t = self._to_tensor(crop, (NN_CROP_SIZE, NN_CROP_SIZE))

        # ── Stage 3: Refine (Tanh output) ─────────────────
        pred_offset = self.refine(crop_t)[0].cpu().numpy()   # [dx_norm, dy_norm]
        dx_norm = float(pred_offset[0])
        dy_norm = float(pred_offset[1])

        # Denormalise: Tanh output ∈ [-1, 1] × half → pixel offset
        final_x = coarse_x + dx_norm * half
        final_y = coarse_y + dy_norm * half

        return int(round(final_x)), int(round(final_y))