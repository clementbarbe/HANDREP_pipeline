"""
Merge neural-network predictions with manually annotated references.
"""

import pandas as pd


def merge_predictions_and_references(pred_df, ref_df):
    """
    Left-join predictions onto reference landmarks.

    Join keys: ``(subject, session, finger, zone)``.
    """
    keep = ["subject", "session", "finger", "zone",
            "landmark", "x_real", "y_real", "image"]
    keep = [c for c in keep if c in ref_df.columns]

    return pd.merge(
        pred_df,
        ref_df[keep],
        on=["subject", "session", "finger", "zone"],
        how="left",
        validate="many_to_one",
    )