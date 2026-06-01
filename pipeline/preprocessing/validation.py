"""
Schema and consistency checks on loaded DataFrames.
"""


def validate_subject_session(pred_df, ref_df, subject, session):
    """
    Assert that both DataFrames contain exactly the expected
    *subject* and *session* identifiers.

    Raises
    ------
    ValueError
        On any mismatch.
    """
    for label, df in [("Predictions", pred_df), ("Reference", ref_df)]:
        subjs = set(df["subject"].dropna().astype(str).unique())
        sesses = set(df["session"].dropna().astype(str).unique())
        if subjs != {subject}:
            raise ValueError(
                f"{label} subject mismatch: expected {{{subject}}}, got {subjs}"
            )
        if sesses != {session}:
            raise ValueError(
                f"{label} session mismatch: expected {{{session}}}, got {sesses}"
            )