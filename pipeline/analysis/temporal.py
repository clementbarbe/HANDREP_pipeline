"""Within-session exploratory regressions; no pseudo-replicated population tests."""

import numpy as np
import pandas as pd


def compute_temporal_models(df):
    ordered = df.sort_values("trial").copy()
    # Only the actual preceding acquisition trial is eligible, including QC barriers.
    previous = ordered.shift(1)
    consecutive = ordered.trial.sub(previous.trial).eq(1)
    if "block" in ordered:
        consecutive &= ordered.block.eq(previous.block)
    for axis in ("ru", "pd"):
        ordered["previous_target_delta_" + axis] = (previous["target_" + axis] - ordered["target_" + axis]).where(consecutive)
    ordered["same_previous_finger"] = ordered.finger.eq(previous.finger).where(consecutive)
    ordered["previous_zone"] = previous.zone.where(consecutive)
    results = []
    cm = ordered.cm_per_px.iloc[0] if "cm_per_px" in ordered else None
    factor = float(cm) if cm is not None and pd.notna(cm) else 1.
    units = "cm" if cm is not None and pd.notna(cm) else "approximate_px"
    for axis in ("ru", "pd"):
        # Landmark intercepts prevent task order from masquerading as drift.
        yname = "error_" + axis
        landmarks = ordered.finger.astype(str) + "_" + ordered.zone.astype(str)
        intercepts = pd.get_dummies(landmarks, dtype=float)
        trial_center = ordered.trial - ordered.trial.mean()
        base = pd.concat([intercepts, trial_center.rename("trial_slope"),
                          ordered.is_coupled.astype(float).rename("coupled_offset")], axis=1)
        if ordered.is_coupled.nunique() < 2:
            base = base.drop(columns="coupled_offset")
        for model, design in (("drift_adjusted_for_landmark", base),
                              ("previous_target_association", pd.concat([base, ordered[["previous_target_delta_" + axis]] * factor], axis=1))):
            good = ordered[yname].notna() & design.notna().all(axis=1)
            x, y = design.loc[good].to_numpy(float), ordered.loc[good, yname].to_numpy(float) * factor
            rank = np.linalg.matrix_rank(x) if len(x) else 0
            if len(y) <= x.shape[1] or rank != x.shape[1]:
                results.append(dict(axis=axis, model=model, term="model", coefficient=np.nan,
                                    n_trials=len(y), status="insufficient_data_or_collinearity"))
                continue
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            for name, coefficient in zip(design.columns, coefficients):
                if name in intercepts.columns:
                    continue
                results.append(dict(axis=axis, model=model, term=name, coefficient=coefficient,
                                    coefficient_unit=(units + "/trial" if name == "trial_slope" else "dimensionless" if name.startswith("previous_target_delta_") else units),
                                    n_trials=len(y), residual_rms=np.sqrt(np.mean((y - x @ coefficients)**2)),
                                    residual_unit=units, status="exploratory_association_no_causal_or_population_inference"))
    return ordered, pd.DataFrame(results)
