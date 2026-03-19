"""Data quality checks for pair labels, as flagged by Clint.

Some pairs (~0.1%) have PFAM_PLUS values that match but were labelled -1
(different antigen) instead of 0.2 (same antigen, different epitope).  This
script identifies and optionally fixes those pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def fix_mislabelled_pairs(pair_csv: Path, output_csv: Path | None = None) -> pl.DataFrame:
    """Identify and fix pairs where PFAM_PLUS overlaps but label is -1.

    These should be labelled 0.2 (same antigen, different epitope) rather
    than -1 (different antigen).

    Args:
        pair_csv: Path to the original pairs CSV.
        output_csv: If provided, write the corrected CSV here.

    Returns:
        Corrected pairs DataFrame.
    """
    df = pl.read_csv(pair_csv)

    if "PFAM_PLUS_I" not in df.columns or "PFAM_PLUS_J" not in df.columns:
        log.warning("Pair CSV has no PFAM_PLUS_I/J columns — skipping label QC.")
        return df

    # Find pairs labelled -1 where PFAM_PLUS families actually overlap.
    # The boolean columns SAME_PFAM_DIF_EP / OVERLAPPING_EP may themselves be
    # incorrect for these cases, so check via direct string overlap.
    def _pfams_overlap(pfam_i: str, pfam_j: str) -> bool:
        return bool(set(pfam_i.split(";")) & set(pfam_j.split(";")))

    has_overlap = [
        _pfams_overlap(pi, pj)
        for pi, pj in zip(df["PFAM_PLUS_I"].to_list(), df["PFAM_PLUS_J"].to_list())
    ]
    overlap_series = pl.Series("_overlap", has_overlap)

    mislabelled = df.filter((pl.col("LABEL") == -1.0) & overlap_series)

    if len(mislabelled) == 0:
        log.info("No mislabelled pairs found in %s", pair_csv)
        return df

    n_total = len(df)
    n_bad = len(mislabelled)
    pct = 100.0 * n_bad / n_total
    log.warning(
        "Found %d / %d (%.2f%%) pairs with matching PFAM but label=-1. Correcting to 0.2.",
        n_bad,
        n_total,
        pct,
    )

    # Fix: set label to 0.2 where PFAM families overlap and label is -1
    df = df.with_columns(
        pl.when((pl.col("LABEL") == -1.0) & overlap_series)
        .then(pl.lit(0.2))
        .otherwise(pl.col("LABEL"))
        .alias("LABEL")
    )

    if output_csv is not None:
        df.write_csv(output_csv)
        log.info("Corrected pairs written to %s", output_csv)

    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m esm_pdb1.label_qc <pair_csv> [output_csv]")
        sys.exit(1)
    pair_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    fix_mislabelled_pairs(pair_csv, output_csv)
