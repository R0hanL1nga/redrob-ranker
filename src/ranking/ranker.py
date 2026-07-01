"""
src/ranking/ranker.py

Combines engineered features into a single composite score,
applies behavioral signal modifiers, and returns a ranked DataFrame.

Ranking Architecture
--------------------
Final Score = (Primary Score × Behavioral Modifier) − Penalty

Primary Score (weights from configs/scoring_weights.yaml):
  1. skill_match        — core AI skill depth + assessment scores
  2. career_relevance   — title + AI duration + product company ratio
  3. experience_fit     — yrs vs JD target range
  4. education          — institution tier
  5. certifications     — relevant AI/ML certifications

Behavioral Modifier (0.6 – 1.0 multiplicative):
  • Availability (recency, open-to-work)
  • Engagement (response rate, response speed)
  • Reliability (interview completion, offer acceptance)
  • Profile quality

Penalties:
  • keyword_stuffer_flag     → −0.20
  • High notice period        → built into notice_period_score
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default weights (overridden by YAML config if present)
DEFAULT_PRIMARY_WEIGHTS: dict[str, float] = {
    "skill_match": 0.35,
    "career_relevance": 0.30,
    "experience_fit": 0.15,
    "education": 0.10,
    "certifications": 0.10,
}

DEFAULT_BEHAVIORAL_WEIGHTS: dict[str, float] = {
    "availability_score": 0.35,
    "engagement_score": 0.30,
    "reliability_score": 0.20,
    "profile_quality_score": 0.15,
}

# Behavioral modifier is clamped to [0.60, 1.0]:
# even a fully disengaged but otherwise great candidate keeps 60% of score
BEHAVIORAL_MODIFIER_MIN = 0.60
BEHAVIORAL_MODIFIER_MAX = 1.00


class CandidateRanker:
    """
    Score and rank candidates from the feature DataFrame.

    Parameters
    ----------
    config_path:
        Path to ``configs/scoring_weights.yaml``.
    """

    def __init__(self, config_path: Path | str = "configs/scoring_weights.yaml") -> None:
        config = self._load_config(Path(config_path))
        self.primary_weights = config.get("primary_weights", DEFAULT_PRIMARY_WEIGHTS)
        self.behavioral_weights = config.get("behavioral_weights", DEFAULT_BEHAVIORAL_WEIGHTS)
        logger.info(
            "CandidateRanker initialised | primary_weights=%s",
            self.primary_weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, feature_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        """
        Compute composite scores and return top-N ranked candidates.

        Parameters
        ----------
        feature_df:
            Output of ``FeatureEngineer.transform()``.
        top_n:
            How many candidates to return (submission requires exactly 100).

        Returns
        -------
        pd.DataFrame
            Columns: candidate_id, rank, score, reasoning
            Sorted ascending by rank (rank 1 = best).
        """
        df = feature_df.copy()

        # 1. Compute primary score
        df["skill_match"] = self._skill_match(df)
        df["career_relevance"] = self._career_relevance(df)
        df["experience_fit"] = df["experience_score"]
        df["education"] = df["education_score"]
        df["certifications"] = df["certification_score"]

        df["primary_score"] = (
            self.primary_weights["skill_match"]         * df["skill_match"]
            + self.primary_weights["career_relevance"]  * df["career_relevance"]
            + self.primary_weights["experience_fit"]    * df["experience_fit"]
            + self.primary_weights["education"]         * df["education"]
            + self.primary_weights["certifications"]    * df["certifications"]
        )

        # 2. Behavioral modifier
        df["behavioral_modifier"] = self._behavioral_modifier(df)

        # 3. Combine
        df["raw_score"] = df["primary_score"] * df["behavioral_modifier"]

        # 4. Keyword stuffer penalty
        df["raw_score"] -= 0.20 * df.get("keyword_stuffer_flag", 0.0)

        # 5. Clamp to [0, 1]
        df["raw_score"] = df["raw_score"].clip(0.0, 1.0)

        # 6. Select top-N, assign rank
        ranked = (
            df.sort_values("raw_score", ascending=False)
            .head(top_n)
            .reset_index()
        )
        ranked["rank"] = range(1, len(ranked) + 1)

        # 7. Normalise scores to [0, 1] maintaining relative order
        ranked["score"] = self._normalise_scores(ranked["raw_score"])

        # 8. Generate human-readable reasoning
        ranked["reasoning"] = ranked.apply(self._generate_reasoning, axis=1)

        logger.info(
            "Top-3 scores: %s",
            ranked[["candidate_id", "score"]].head(3).to_dict("records"),
        )

        return ranked[["candidate_id", "rank", "score", "reasoning"]]

    # ------------------------------------------------------------------
    # Score components
    # ------------------------------------------------------------------

    def _skill_match(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine core skill score, depth, and assessment into one metric.
        Weights: 50% core_ai_skill_score, 30% skill_depth, 20% assessment.
        """
        return (
            0.50 * df["core_ai_skill_score"]
            + 0.30 * df["skill_depth_score"]
            + 0.20 * df["assessment_score"]
        )

    def _career_relevance(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine title score, product company ratio, tenure, AI duration.
        """
        return (
            0.40 * df["title_score"]
            + 0.25 * df["product_company_score"]
            + 0.20 * df["ai_career_duration_score"]
            + 0.15 * df["tenure_score"]
        )

    def _behavioral_modifier(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute [0.60, 1.00] modifier based on behavioral signals.
        """
        raw = (
            self.behavioral_weights["availability_score"]   * df["availability_score"]
            + self.behavioral_weights["engagement_score"]   * df["engagement_score"]
            + self.behavioral_weights["reliability_score"]  * df["reliability_score"]
            + self.behavioral_weights["profile_quality_score"] * df["profile_quality_score"]
        )
        # Map raw [0, 1] behavioral signal to [0.60, 1.00] modifier
        return raw.clip(0.0, 1.0) * (
            BEHAVIORAL_MODIFIER_MAX - BEHAVIORAL_MODIFIER_MIN
        ) + BEHAVIORAL_MODIFIER_MIN

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_scores(series: pd.Series) -> pd.Series:
        """
        Min-max normalise raw scores to [0, 1], keeping relative order.
        Ensures rank-1 candidate gets score 1.0, rank-N gets ≥ 0.2.
        """
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([1.0] * len(series), index=series.index)
        normalised = (series - mn) / (mx - mn)
        # Remap to [0.20, 1.00] so no candidate appears to score zero
        return (normalised * 0.80 + 0.20).round(4)

    @staticmethod
    def _generate_reasoning(row: pd.Series) -> str:
        """
        Build a compact, human-readable explanation for why a candidate
        was ranked at their position. Mirrors the format in sample_submission.csv.
        """
        parts: list[str] = []

        # Title / experience
        yoe = row.get("years_of_experience", 0)
        parts.append(f"{yoe:.1f} yrs exp")

        # Skill match
        core_count = int(row.get("core_ai_skill_count", 0))
        parts.append(f"{core_count} core AI skills")

        # Engagement signal
        engage = row.get("engagement_score", 0)
        parts.append(f"engagement={engage:.2f}")

        # Career signal
        prod = row.get("product_company_score", 0)
        if prod >= 0.8:
            parts.append("product-company background")
        elif prod < 0.4:
            parts.append("services-heavy career")

        # Availability
        avail = row.get("availability_score", 0)
        if avail >= 0.8:
            parts.append("actively available")
        elif avail < 0.4:
            parts.append("low recent activity")

        # Keyword stuffer flag
        if row.get("keyword_stuffer_flag", 0) > 0:
            parts.append("⚠ keyword-stuffer penalty applied")

        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            logger.warning("Ranker config not found at %s — using defaults", path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
