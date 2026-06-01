"""
Image discovery for the neural-network inference step.
"""

from pathlib import Path
from pipeline.config.settings import IMAGE_GLOBS


def collect_images(folder):
    """
    Return sorted, deduplicated JPEG paths from *folder*,
    excluding any filename containing ``ref`` (reference images).
    """
    folder = Path(folder)
    raw = []
    for pattern in IMAGE_GLOBS:
        raw.extend(folder.glob(pattern))

    seen = set()
    result = []
    for p in sorted(raw, key=lambda x: x.name.lower()):
        if "ref" in p.name.lower():
            continue
        if p.name not in seen:
            seen.add(p.name)
            result.append(p)
    return result