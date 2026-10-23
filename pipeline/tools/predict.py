"""Reproducible batch pointer inference without overwriting earlier predictions.

python -m pipeline.tools.predict --output-dir data/nn_outputs_scientific
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd
import torch

from pipeline.config.settings import NN_IMG_SIZE, NN_HEATMAP_SIZE, NN_CROP_SIZE
from pipeline.preprocessing.images import collect_images
from pipeline.io.discovery import parse_image_filename, parse_subject_session_from_path
from pipeline.segmentation.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/nn_outputs_scientific")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    weights = [Path(args.weights_dir) / "hrnet_weights.pth", Path(args.weights_dir) / "refine_weights.pth"]
    predictor = Predictor(*(str(p) for p in weights))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder in sorted((Path(args.data_dir) / "raw").glob("sub-*/ses-*")):
        subject, session = parse_subject_session_from_path(folder)
        if args.subjects and subject not in args.subjects:
            continue
        images = collect_images(folder)
        if not images:
            continue
        rows = []
        for path in images:
            _, trial, finger, zone = parse_image_filename(path.name)
            if trial < 1 or finger == "UNK" or zone not in {"Z1", "Z2"}:
                raise ValueError(f"Unrecognized acquisition filename: {path}")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode image: {path}")
            x, y = predictor.predict(image)
            rows.append(dict(subject=subject, session=session, trial=trial, finger=finger, zone=zone,
                             x_est=x, y_est=y, filename=path.name))
        table = pd.DataFrame(rows).sort_values("trial")
        if table.trial.duplicated().any():
            raise ValueError(f"Duplicate trial identifiers: {folder}")
        path = output / f"{subject}_{session}_predictions.csv"
        table.to_csv(path, index=False)
        provenance = dict(subject=subject, session=session, n_predictions=len(table),
                          inference="two_stage_actual_crop_centre_v2", torch_version=torch.__version__,
                          device=str(predictor.device), seed=42, cpu_threads=args.cpu_threads,
                          dimensions=dict(input=NN_IMG_SIZE, heatmap=NN_HEATMAP_SIZE, crop=NN_CROP_SIZE),
                          weights_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},
                          input_images_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
                          prediction_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          predictor_source_sha256=hashlib.sha256((Path(__file__).parents[1] / "segmentation" / "predictor.py").read_bytes()).hexdigest(),
                          independent_detection_accuracy="not_established_by_inference")
        (output / f"{subject}_{session}_prediction_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        count += len(table)
        print(f"{subject}/{session}: {len(table)} predictions -> {path}", flush=True)
    if not count:
        raise ValueError("No acquisition images selected")


if __name__ == "__main__":
    main()
