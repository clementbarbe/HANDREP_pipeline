# Propriohand

Automated pipeline for hand proprioception analysis.

## Quick start

```bash
pip install -r requirements.txt

# 1. Place raw images in data/raw/sub-S1/ses-1/
# 2. Annotate reference images
python -m pipeline.tools.annotate

# 3. Run NN predictions (GUI)
python -m pipeline.gui.app

# 4. Run the full analysis pipeline
python run.py
```

## CLI options

``` css
python run.py --help
python run.py --no-checkerboard --allow-fallback
python run.py --subjects S1 S3 --skip-figures
```

## Project structure

    Directory	Role
    pipeline/config	Settings, path resolution
    pipeline/io	File discovery, loading, writing
    pipeline/preprocessing	Image collection, data validation
    pipeline/registration	Checkerboard calibration, correction
    pipeline/segmentation	NN architectures, inference
    pipeline/transforms	Merging, error computation
    pipeline/quality_control	Integrity checks
    pipeline/analysis	Statistics, biomechanics, TPS
    pipeline/visualization	All figure generation
    pipeline/export	CSV report exports
    pipeline/utils	Logging, console output
    pipeline/gui	PyQt6 interactive interface
    pipeline/tools	Annotation tool
    
## Data layout

```csharp
data/
├── raw/              # Acquisition images (sub-S1/ses-1/*.jpg)
├── ref/              # Reference annotations (*_ref.csv)
├── calibration/      # table.png + plateau.png
├── nn_outputs/       # NN prediction CSVs
└── processed/        # Pipeline outputs (per subject)
```
## Calibration

Place two photos of the same checkerboard in data/calibration/:

    table.png — checkerboard at hand level
    plateau.png — checkerboard at pointing-board level

The pipeline computes the homography automatically.
Adjust CHECKERBOARD_SIZE in pipeline/config/settings.py to match
your checkerboard's internal corner count.

## Scientific analysis version 2

The analysis now uses the same distance-between-centroids estimator in both
coupling conditions, paired bootstrap resampling, corrected rigid alignment,
and metric rectification of both references and responses. Original image
coordinates are preserved. Analysis pixel coordinates now refer to a rectified
plane and must not be overlaid on the original camera image.

In the existing environment:

```bash
conda activate MOTORPLANNING
python -m pipeline.tools.predict --output-dir data/nn_outputs_scientific
python run.py --prediction-dir data/nn_outputs_scientific --output-dir data/processed_scientific
python -m unittest discover -s tests -v
```

For a new installation, `requirements.txt` is sufficient for analysis; install
`requirements-gui.txt` for neural-network prediction and its GUI.

Analyses are always recalculated; `--force` remains accepted for compatibility.
Failed calibration stops analysis unless `--allow-fallback` is explicitly set.
Fallback outputs are exploratory pixels and have no centimetre conversion.

Optional acquisition designs belong in
`data/design/S1_ses-1_design.csv` with columns `trial,condition,pair_id`.
Without a recorded design, adjacency comparisons are explicitly exploratory.
Independent manual pointer annotations can be placed in
`data/validation/S1_ses-1_pointer.csv` with columns
`filename,x_manual,y_manual`.

See [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md) for reference selection,
uncertainty assumptions, data-quality controls and the output dictionary.
