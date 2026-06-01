"""
Figure generation — separated from scientific computation.

Each sub-module exposes one ``plot_*`` function that receives
pre-computed analysis results and writes a figure to disk.
"""

from pathlib import Path


def generate_all_figures(analysis, output_dir, cfg):
    """
    Convenience function: generate every figure for one (subject, session).

    Parameters
    ----------
    analysis : dict
        Output of the analysis stage (see ``workflow._process_pair``).
    output_dir : Path
        Directory for figure PNG files.
    cfg : dict
        Pipeline configuration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .positions import plot_estimated_positions
    from .error_analysis import plot_error_analysis
    from .morphometry import plot_finger_lengths_widths, plot_shape_index
    from .temporal import plot_error_over_time
    from .deformation import plot_deformation_maps
    from .correction_effect import plot_correction_effect

    plot_estimated_positions(analysis, output_dir / "01_estimated_positions.png", cfg)
    plot_error_analysis(analysis, output_dir / "02_error_analysis.png", cfg)
    plot_finger_lengths_widths(analysis, output_dir / "03_finger_lengths_widths.png", cfg)
    plot_error_over_time(analysis, output_dir / "04_error_over_time.png", cfg)
    plot_deformation_maps(analysis, output_dir, cfg)  # writes 05 + 06
    plot_shape_index(analysis, output_dir / "07_shape_index.png", cfg)
    plot_correction_effect(analysis, output_dir / "08_correction_effect.png", cfg)