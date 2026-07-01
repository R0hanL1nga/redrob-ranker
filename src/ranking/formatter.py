"""
src/ranking/formatter.py

Writes the final ranked DataFrame to a CSV file that satisfies the
challenge's validator (validate_submission.py):

  Row 1 : candidate_id,rank,score,reasoning  (exact header)
  Rows 2-101 : exactly 100 data rows
  - candidate_id : CAND_XXXXXXX pattern
  - rank : integer 1-100 (unique, no gaps)
  - score : float, non-increasing by rank; tie-break = candidate_id ASC
  - reasoning : free text (non-empty)
  Encoding : UTF-8
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


class SubmissionFormatter:
    """
    Serialise the ranked DataFrame to a challenge-compliant CSV.
    """

    def write(self, ranked_df: pd.DataFrame, output_path: Path) -> None:
        """
        Write submission CSV.

        Parameters
        ----------
        ranked_df:
            Must contain columns: candidate_id, rank, score, reasoning.
            Expected to be sorted by rank ascending (rank 1 first).
        output_path:
            Destination path for the CSV file.

        Raises
        ------
        ValueError
            If the DataFrame does not contain the required 100 rows.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate before writing
        self._validate(ranked_df)

        with output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(REQUIRED_HEADER)

            for _, row in ranked_df.iterrows():
                writer.writerow([
                    row["candidate_id"],
                    int(row["rank"]),
                    f"{row['score']:.4f}",
                    row["reasoning"],
                ])

        logger.info(
            "Submission written: %s (%d rows)", output_path, len(ranked_df)
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Lightweight pre-write validation to catch obvious errors early."""
        n = len(df)
        if n < 1:
            raise ValueError("Submission must contain at least 1 row.")
        if n > 100:
            raise ValueError(
                f"Submission must contain at most 100 rows, got {n}."
            )
        # Warn (not error) if fewer than 100 rows — sample dataset has only 50
        if n != 100:
            import warnings
            warnings.warn(
                f"Submission has {n} rows; challenge requires exactly 100. "
                "This is expected when running on sample_candidates.json (50 candidates).",
                stacklevel=3,
            )

        missing_cols = [c for c in REQUIRED_HEADER if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        ranks = sorted(df["rank"].tolist())
        expected = list(range(1, n + 1))
        if ranks != expected:
            raise ValueError(f"Ranks must be unique integers 1-{n}.")

        # Check non-increasing scores
        sorted_scores = df.sort_values("rank")["score"].tolist()
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] < sorted_scores[i + 1]:
                raise ValueError(
                    f"Score at rank {i+1} ({sorted_scores[i]:.4f}) is less than "
                    f"score at rank {i+2} ({sorted_scores[i+1]:.4f}). "
                    "Scores must be non-increasing."
                )

        logger.debug("Pre-write validation passed ✓")
