#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob Candidate Ranking Pipeline.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./data/sample_candidates.json --out ./submission.csv --top_n 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.data.loader import CandidateLoader
from src.features.engineer import FeatureEngineer
from src.ranking.ranker import CandidateRanker
from src.ranking.formatter import SubmissionFormatter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ranking_run.log", mode="w"),
    ],
)
logger = logging.getLogger("rank")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Discovery & Ranking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidates.jsonl or sample_candidates.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scoring_weights.yaml"),
        help="Path to scoring weights YAML config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("submission.csv"),
        help="Output CSV path for final ranked submission",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100,
        help="Number of top candidates to include in submission",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate the end-to-end ranking pipeline."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Redrob Candidate Ranking Pipeline — Starting")
    logger.info("=" * 60)
    logger.info("Input  : %s", args.candidates)
    logger.info("Config : %s", args.config)
    logger.info("Output : %s", args.out)
    logger.info("Top-N  : %d", args.top_n)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1: Load candidates
    # ------------------------------------------------------------------ #
    logger.info("[1/4] Loading candidates …")
    loader = CandidateLoader()
    candidates = loader.load(args.candidates)
    logger.info("      Loaded %d candidates", len(candidates))

    # ------------------------------------------------------------------ #
    # Step 2: Feature engineering
    # ------------------------------------------------------------------ #
    logger.info("[2/4] Engineering features …")
    engineer = FeatureEngineer(config_path=args.config)
    feature_df = engineer.transform(candidates)
    logger.info("      Feature matrix shape: %s", feature_df.shape)

    # ------------------------------------------------------------------ #
    # Step 3: Rank candidates
    # ------------------------------------------------------------------ #
    logger.info("[3/4] Ranking candidates …")
    ranker = CandidateRanker(config_path=args.config)
    ranked_df = ranker.rank(feature_df, top_n=args.top_n)
    logger.info("      Top-%d candidates selected", args.top_n)

    # ------------------------------------------------------------------ #
    # Step 4: Write submission
    # ------------------------------------------------------------------ #
    logger.info("[4/4] Writing submission to %s …", args.out)
    formatter = SubmissionFormatter()
    formatter.write(ranked_df, output_path=args.out)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs  →  %s", elapsed, args.out)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
