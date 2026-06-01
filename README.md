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
python run.py --force --no-checkerboard
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