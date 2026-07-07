"""
Save all output CSVs for one (subject, session) pair.
"""

from pathlib import Path
from pipeline.io.writer import save_trial_csv, save_summary_csv, save_biomech_csv


def export_all_csvs(analysis, out_dir, subject, session):
    """
    Write trials, summary, biomechanics, finger lengths,
    knuckle widths, and coupled/uncoupled breakdowns.

    Parameters
    ----------
    analysis : dict
        Must contain ``df``, ``summary``, ``biomech``,
        ``finger_lengths``, ``knuckle_widths``, and optionally
        ``finger_lengths_coupled``, ``finger_lengths_uncoupled``,
        ``couple_distances``.
    out_dir : Path
        Subject output directory.
    subject, session : str
    """
    out_dir = Path(out_dir)
    prefix = f"{subject}_{session}"

    # ── Core outputs ──────────────────────────────────────
    save_trial_csv(analysis["df"], out_dir / f"{prefix}_trials.csv")
    save_summary_csv(analysis["summary"], out_dir / f"{prefix}_summary.csv")
    save_biomech_csv(analysis["biomech"], out_dir / f"{prefix}_biomech.csv")

    # ── Detailed notebook-style CSVs ──────────────────────
    analysis["df"].to_csv(
        out_dir / f"{prefix}_results_all_trials.csv", index=False, sep=";",
    )

    stats = analysis.get("stats_landmark")
    if stats is not None:
        stats.to_csv(
            out_dir / f"{prefix}_results_stats_per_landmark.csv",
            index=False, sep=";",
        )

    fl = analysis.get("finger_lengths")
    if fl is not None and not fl.empty:
        fl.to_csv(
            out_dir / f"{prefix}_results_finger_lengths.csv",
            index=False, sep=";",
        )

    kw = analysis.get("knuckle_widths")
    if kw is not None and not kw.empty:
        kw.to_csv(
            out_dir / f"{prefix}_results_knuckle_widths.csv",
            index=False, sep=";",
        )

    # ── Coupled / uncoupled breakdowns ────────────────────
    fl_c = analysis.get("finger_lengths_coupled")
    if fl_c is not None and not fl_c.empty:
        fl_c.to_csv(
            out_dir / f"{prefix}_results_finger_lengths_coupled.csv",
            index=False, sep=";",
        )

    fl_u = analysis.get("finger_lengths_uncoupled")
    if fl_u is not None and not fl_u.empty:
        fl_u.to_csv(
            out_dir / f"{prefix}_results_finger_lengths_uncoupled.csv",
            index=False, sep=";",
        )

    cd = analysis.get("couple_distances")
    if cd is not None and not cd.empty:
        cd.to_csv(
            out_dir / f"{prefix}_results_couple_distances.csv",
            index=False, sep=";",
        )