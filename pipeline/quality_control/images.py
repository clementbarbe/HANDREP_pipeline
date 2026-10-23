"""Check that available acquisition images use the calibration resolution."""

from pathlib import Path
import cv2
import numpy as np


def check_source_images(predictions, raw_dir, width, height):
    out = predictions.copy()
    available, widths, heights = [], [], []
    for filename in out.filename:
        path = raw_dir / Path(filename).name
        exists = path.is_file()
        available.append(exists)
        if not exists:
            widths.append(np.nan)
            heights.append(np.nan)
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode acquisition image: {path}")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise ValueError(f"Acquisition/calibration resolution mismatch for {filename}: {(actual_width, actual_height)} != {(width, height)}")
        widths.append(actual_width)
        heights.append(actual_height)
    out["source_image_available"] = available
    out["source_width"], out["source_height"] = widths, heights
    return out
