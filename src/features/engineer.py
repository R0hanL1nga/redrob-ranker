"""
src/features/engineer.py

Converts raw candidate dicts into a structured feature DataFrame.
Each row = one candidate. Columns = individual numeric features that
the ranker consumes.

Scoring philosophy (from job_description.docx):
  - The JD is for a Senior AI Engineer (embeddings, retrieval, ranking, LLMs)
  - "Keyword stuffer" trap: AI keywords in skills section != fit
  - Career history titles and product-company signal matter more than raw keywords
  - Behavioral signals are multipliers, not primary scores
  - Availability (recent activity, open-to-work) is a hard modifier
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AI/ML core skills the JD explicitly requires or values highly
# ---------------------------------------------------------------------------
CORE_AI_SKILLS: set[str] = {
    # Must-haves from JD
    "embeddings", "sentence transformers", "vector database", "vector search",
    "hybrid search", "semantic search", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "bm25",
    "information retrieval", "ranking", "learning to rank", "ndcg", "mrr",
    "a/b testing", "evaluation frameworks", "python",
    # Nice-to-haves from JD
    "fine-tuning llms", "lora", "qlora", "peft", "xgboost", "nlp",
    "large language models", "llm", "rag", "retrieval augmented generation",
    "transformers", "bert", "hugging face", "pytorch", "tensorflow",
    "scikit-learn", "ml engineering", "mlops",
    # Signals of deeper ML vs framework user
    "recommendation systems", "search ranking", "reranking",
    "sparse retrieval", "dense retrieval", "distributed systems",
    "apache spark", "kafka", "airflow",
}

# Title keywords that signal relevant ML/AI experience
AI_TITLE_KEYWORDS: list[str] = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "nlp engineer", "search engineer",
    "applied scientist", "research engineer", "recommendation",
    "backend engineer", "software engineer",  # weighted lower but not disqualifying
]

# Titles that are anti-signals for this specific JD
DISQUALIFYING_TITLES: set[str] = {
    "marketing manager", "hr manager", "accountant", "graphic designer",
    "content writer", "civil engineer", "mechanical engineer",
    "customer support", "sales executive", "operations manager",
    "project manager", "business analyst",
}

# Company-size categories (string → midpoint employees)
COMPANY_SIZE_MAP: dict[str, int] = {
    "1-10": 5, "11-50": 30, "51-200": 125, "201-500": 350,
    "501-1000": 750, "1001-5000": 3000, "5001-10000": 7500, "10001+": 20000,
}

EDUCATION_TIER_MAP: dict[str, float] = {
    "tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50,
    "tier_4": 0.30, "unknown": 0.40,
}

PROFICIENCY_MAP: dict[str, float] = {
    "beginner": 0.25, "intermediate": 0.50,
    "advanced": 0.75, "expert": 1.0,
}

TODAY = date.today()


class FeatureEngineer:
    """
    Transform raw candidate dicts → pd.DataFrame of numeric features.

    Parameters
    ----------
    config_path:
        Path to ``configs/scoring_weights.yaml``. Used to pull the JD
        target experience range and salary band for normalisation.
    """

    def __init__(self, config_path: Path | str = "configs/scoring_weights.yaml") -> None:
        self.config = self._load_config(Path(config_path))
        logger.info("FeatureEngineer initialised with config: %s", config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, candidates: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Build feature matrix from list of candidate dicts.

        Returns
        -------
        pd.DataFrame
            Index = candidate_id, columns = feature names.
        """
        rows = []
        for cand in candidates:
            try:
                rows.append(self._extract(cand))
            except Exception as exc:  # noqa: BLE001
                cid = cand.get("candidate_id", "UNKNOWN")
                logger.warning("Skipping %s due to extraction error: %s", cid, exc)

        df = pd.DataFrame(rows).set_index("candidate_id")
        logger.info("Feature matrix: %d candidates × %d features", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def _extract(self, c: dict[str, Any]) -> dict[str, Any]:
        """Extract all features for a single candidate dict."""
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        career = c.get("career_history", [])
        education = c.get("education", [])
        skills = c.get("skills", [])
        certs = c.get("certifications", [])
        signals = c.get("redrob_signals", {})

        row: dict[str, Any] = {"candidate_id": cid}

        # ---- Profile features ----------------------------------------
        row["years_of_experience"] = float(profile.get("years_of_experience", 0))
        row["experience_score"] = self._score_experience(
            row["years_of_experience"]
        )

        # ---- Skills features -----------------------------------------
        skill_scores = self._score_skills(skills, signals)
        row.update(skill_scores)

        # ---- Career history features ---------------------------------
        career_scores = self._score_career(career, profile)
        row.update(career_scores)

        # ---- Education features -------------------------------------
        row["education_score"] = self._score_education(education)

        # ---- Certification bonus ------------------------------------
        row["certification_score"] = self._score_certifications(certs)

        # ---- Behavioral / availability signals ----------------------
        signal_scores = self._score_signals(signals)
        row.update(signal_scores)

        # ---- Keyword stuffer detection (needs title + skills together) ---
        current_title = profile.get("current_title", "").lower()
        is_disqualifying_title = any(
            dis in current_title for dis in DISQUALIFYING_TITLES
        )
        core_count = int(row.get("core_ai_skill_count", 0))
        row["keyword_stuffer_flag"] = 1.0 if (
            is_disqualifying_title and core_count >= 6
        ) else 0.0

        return row

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _score_experience(self, years: float) -> float:
        """
        Score years of experience against JD target range (5-9 years).
        Peak = 1.0 at 6-8 yrs; decays outside range.
        """
        target_min = self.config.get("jd_experience_min", 5)
        target_max = self.config.get("jd_experience_max", 9)
        sweet_spot_min = self.config.get("jd_experience_sweet_min", 6)
        sweet_spot_max = self.config.get("jd_experience_sweet_max", 8)

        if sweet_spot_min <= years <= sweet_spot_max:
            return 1.0
        elif target_min <= years < sweet_spot_min:
            return 0.80
        elif sweet_spot_max < years <= target_max:
            return 0.85
        elif years < target_min:
            # Under-experienced: linear decay
            return max(0.1, 0.80 * (years / target_min))
        else:
            # Over-experienced (>9 yrs): slight penalty but still usable
            return max(0.50, 1.0 - 0.03 * (years - target_max))

    def _score_skills(
        self, skills: list[dict], signals: dict
    ) -> dict[str, float]:
        """
        Multi-dimensional skill scoring.

        Returns
        -------
        dict with keys:
            core_ai_skill_count     — raw count of matched core AI skills
            core_ai_skill_score     — normalised 0-1 match fraction
            skill_depth_score       — weighted by proficiency + endorsements
            assessment_score        — avg of Redrob in-platform assessment scores
            keyword_stuffer_flag    — 1 if skills list looks inflated vs career
        """
        skill_names_lower = [s.get("name", "").lower() for s in skills]
        matched = [
            s for s in skills
            if s.get("name", "").lower() in CORE_AI_SKILLS
        ]

        core_count = len(matched)
        # Normalise against a "perfect" candidate having ~10 core skills
        core_score = min(1.0, core_count / 10.0)

        # Depth score: proficiency + log(endorsements+1) + duration
        depth_total = 0.0
        for s in matched:
            prof = PROFICIENCY_MAP.get(s.get("proficiency", "beginner"), 0.25)
            end = math.log1p(s.get("endorsements", 0)) / math.log1p(100)  # norm to ~1
            dur = min(1.0, s.get("duration_months", 0) / 36)  # 3yr = full
            depth_total += (0.5 * prof + 0.3 * end + 0.2 * dur)
        depth_score = min(1.0, depth_total / max(1, core_count)) if matched else 0.0

        # In-platform assessment scores (average of all available)
        assessments = signals.get("skill_assessment_scores", {})
        if assessments:
            assessment_score = sum(assessments.values()) / len(assessments) / 100.0
        else:
            assessment_score = 0.0

        # Keyword stuffer detection: many AI skills but non-AI current title
        current_title_lower = ""  # passed via profile, not available here directly
        # We detect based on the skill list alone: ≥8 core AI skills is suspicious
        # The title check happens in _score_title; here we flag the signal
        # A non-AI title + many AI skills combo will be penalised by the low title_score
        # but we also set the flag for the explicit penalty in the ranker
        stuffer_flag = 0.0  # computed in ranker after title is known

        return {
            "core_ai_skill_count": float(core_count),
            "core_ai_skill_score": core_score,
            "skill_depth_score": depth_score,
            "assessment_score": assessment_score,
            "keyword_stuffer_flag": stuffer_flag,
        }

    def _score_career(
        self, career: list[dict], profile: dict
    ) -> dict[str, float]:
        """
        Score career history for role relevance, company quality,
        tenure, and consulting-firm penalty.

        Key signals:
        - Current/recent title relevance (AI/ML vs disqualifying)
        - Product company ratio (JD explicitly disfavours pure-services candidates)
        - Tenure stability (JD disfavours 1.5yr job-hoppers)
        - Total duration in AI-adjacent roles
        """
        current_title = profile.get("current_title", "").lower()

        # Title score
        title_score = self._score_title(current_title, career)

        # Product vs services ratio
        product_score = self._score_product_company_ratio(career)

        # Tenure score (penalise serial job-hoppers < 18 months avg)
        tenure_score = self._score_tenure(career)

        # AI role duration (months in ML/AI-adjacent roles)
        ai_months = self._ai_career_months(career)
        ai_duration_score = min(1.0, ai_months / 48.0)  # 4yr in AI = 1.0

        return {
            "title_score": title_score,
            "product_company_score": product_score,
            "tenure_score": tenure_score,
            "ai_career_duration_score": ai_duration_score,
        }

    def _score_title(
        self, current_title: str, career: list[dict]
    ) -> float:
        """Score based on current + career titles relevance to the JD."""
        # Hard disqualifier: current title is clearly non-technical
        for dis in DISQUALIFYING_TITLES:
            if dis in current_title:
                return 0.10  # very low, not zero (career pivot possible)

        # Positive signal: AI/ML title
        for kw in AI_TITLE_KEYWORDS[:6]:  # top 6 are strong positives
            if kw in current_title:
                return 1.0

        # Moderate signal: generic engineer titles
        if any(w in current_title for w in ["engineer", "developer", "scientist", "analyst"]):
            return 0.65

        return 0.40

    def _score_product_company_ratio(self, career: list[dict]) -> float:
        """
        Estimate fraction of career months at product companies vs
        pure consulting/outsourcing firms.
        """
        CONSULTING_KEYWORDS = {
            "tcs", "infosys", "wipro", "accenture", "cognizant",
            "capgemini", "hcl", "tech mahindra", "mphasis", "l&t infotech",
        }
        total_months = 0
        consulting_months = 0

        for job in career:
            dur = job.get("duration_months", 0)
            total_months += dur
            if any(kw in job.get("company", "").lower() for kw in CONSULTING_KEYWORDS):
                consulting_months += dur

        if total_months == 0:
            return 0.5
        consulting_ratio = consulting_months / total_months
        # > 80% consulting career = 0.2 score; 0% = 1.0
        return max(0.2, 1.0 - consulting_ratio)

    def _score_tenure(self, career: list[dict]) -> float:
        """
        Score average job tenure. Penalise < 18 months average.
        Ideal: 24-48 months average tenure.
        """
        durations = [
            job.get("duration_months", 0)
            for job in career
            if not job.get("is_current", False) and job.get("duration_months", 0) > 0
        ]
        if not durations:
            return 0.7  # can't penalise; probably early career

        avg_tenure = sum(durations) / len(durations)
        if avg_tenure >= 30:
            return 1.0
        elif avg_tenure >= 18:
            return 0.85
        elif avg_tenure >= 12:
            return 0.65
        else:
            return 0.40  # serial job-hopper

    def _ai_career_months(self, career: list[dict]) -> float:
        """Sum of months in AI/ML-adjacent roles across full career."""
        AI_ROLE_KEYWORDS = {
            "ml", "machine learning", "ai", "data science", "nlp",
            "deep learning", "computer vision", "research", "recommendation",
            "search", "ranking", "backend", "data engineer",
        }
        total = 0
        for job in career:
            title_lower = job.get("title", "").lower()
            desc_lower = job.get("description", "").lower()
            if any(kw in title_lower or kw in desc_lower for kw in AI_ROLE_KEYWORDS):
                total += job.get("duration_months", 0)
        return float(total)

    def _score_education(self, education: list[dict]) -> float:
        """Score highest education tier. Tier 1 = 1.0, unknown = 0.4."""
        if not education:
            return 0.30
        best_tier = max(
            (EDUCATION_TIER_MAP.get(e.get("tier", "unknown"), 0.40) for e in education),
            default=0.30,
        )
        return best_tier

    def _score_certifications(self, certs: list[dict]) -> float:
        """Bonus for relevant AI/ML certifications."""
        AI_CERT_KEYWORDS = {
            "aws", "gcp", "azure", "tensorflow", "pytorch", "google",
            "deeplearning.ai", "coursera", "databricks", "mlops",
        }
        relevant = sum(
            1 for c in certs
            if any(kw in c.get("name", "").lower() or kw in c.get("issuer", "").lower()
                   for kw in AI_CERT_KEYWORDS)
        )
        return min(1.0, relevant / 3.0)  # 3+ certs = 1.0

    def _score_signals(self, signals: dict) -> dict[str, float]:
        """
        Score all 23 Redrob behavioral signals into composites.

        Returns
        -------
        dict with keys:
            availability_score   — recency + open_to_work
            engagement_score     — response rate, activity
            reliability_score    — interview completion, offer acceptance
            profile_quality      — completeness, verifications
            github_score         — open-source activity
            notice_period_score  — JD prefers sub-30 day notice
        """
        # --- Availability ---
        last_active = self._days_since(signals.get("last_active_date"))
        recency_score = max(0.0, 1.0 - last_active / 180.0)  # 0 = 6+ months away
        otw = 1.0 if signals.get("open_to_work_flag", False) else 0.5
        availability = 0.6 * recency_score + 0.4 * otw

        # --- Engagement ---
        response_rate = float(signals.get("recruiter_response_rate", 0.0))
        avg_resp_hrs = float(signals.get("avg_response_time_hours", 48))
        # Fast responders score higher (< 4 hrs = 1.0, > 48 hrs = 0.2)
        resp_speed = max(0.2, 1.0 - math.log1p(avg_resp_hrs) / math.log1p(48))
        saved = math.log1p(signals.get("saved_by_recruiters_30d", 0)) / math.log1p(20)
        engagement = 0.5 * response_rate + 0.3 * resp_speed + 0.2 * min(1.0, saved)

        # --- Reliability ---
        icr = float(signals.get("interview_completion_rate", 0.5))
        oar_raw = signals.get("offer_acceptance_rate", -1)
        oar = float(oar_raw) if oar_raw >= 0 else 0.6  # no history → neutral
        reliability = 0.6 * icr + 0.4 * oar

        # --- Profile quality ---
        completeness = float(signals.get("profile_completeness_score", 50)) / 100.0
        verif = sum([
            signals.get("verified_email", False),
            signals.get("verified_phone", False),
            signals.get("linkedin_connected", False),
        ]) / 3.0
        profile_quality = 0.6 * completeness + 0.4 * verif

        # --- GitHub ---
        gh_raw = signals.get("github_activity_score", -1)
        github_score = float(gh_raw) / 100.0 if gh_raw >= 0 else 0.20

        # --- Notice period ---
        notice = int(signals.get("notice_period_days", 60))
        if notice <= 15:
            notice_score = 1.0
        elif notice <= 30:
            notice_score = 0.85
        elif notice <= 60:
            notice_score = 0.65
        elif notice <= 90:
            notice_score = 0.45
        else:
            notice_score = 0.25

        return {
            "availability_score": availability,
            "engagement_score": engagement,
            "reliability_score": reliability,
            "profile_quality_score": profile_quality,
            "github_score": github_score,
            "notice_period_score": notice_score,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _days_since(date_str: str | None) -> float:
        """Return float days since a date string (YYYY-MM-DD). 999 if None."""
        if not date_str:
            return 999.0
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            return max(0.0, float((TODAY - d).days))
        except ValueError:
            return 999.0

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            logger.warning("Config not found at %s — using defaults", path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
