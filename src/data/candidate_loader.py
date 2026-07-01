"""
src/data/loader.py

Handles loading candidate data from both .jsonl (full dataset) and
.json (sample) formats. Returns a list of raw candidate dicts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CandidateLoader:
    """
    Loads candidate profiles from disk.

    Supports:
    - ``.jsonl`` — one JSON object per line (production dataset, ~487 MB)
    - ``.json``  — a JSON array (sample_candidates.json)
    """

    def load(self, path: Path) -> list[dict[str, Any]]:
        """
        Load candidates from a file.

        Parameters
        ----------
        path:
            Absolute or relative path to the candidate data file.

        Returns
        -------
        list[dict]
            A list of raw candidate dictionaries conforming to
            candidate_schema.json.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the file extension is not ``.jsonl`` or ``.json``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Candidate data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix == ".json":
            return self._load_json(path)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                "Expected '.jsonl' or '.json'."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Stream-read a JSONL file line by line (memory-efficient)."""
        candidates: list[dict[str, Any]] = []
        errors = 0

        with path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    errors += 1
                    logger.warning(
                        "Skipping malformed JSON on line %d: %s", line_num, exc
                    )
                if line_num % 10_000 == 0:
                    logger.debug("  … read %d lines", line_num)

        if errors:
            logger.warning("Skipped %d malformed lines in %s", errors, path)

        logger.info("Loaded %d candidates from %s", len(candidates), path.name)
        return candidates

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        """Load a JSON array from file (sample dataset)."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {path.name}, "
                f"got {type(data).__name__}."
            )

        logger.info("Loaded %d candidates from %s", len(data), path.name)
        return data
