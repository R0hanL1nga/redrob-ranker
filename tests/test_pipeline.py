"""
tests/test_pipeline.py

End-to-end and unit tests for the ranking pipeline.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import CandidateLoader
from src.features.engineer import FeatureEngineer
from src.ranking.formatter import SubmissionFormatter
from src.ranking.ranker import CandidateRanker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CANDIDATE = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Test Candidate",
        "headline": "ML Engineer | Embeddings & Retrieval",
        "summary": "5 years in production ML systems.",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 6.5,
        "current_title": "ML Engineer",
        "current_company": "TestCo",
        "current_company_size": "51-200",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            "company": "TestCo",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 29,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Built embeddings-based search using Qdrant and sentence-transformers.",
        }
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2015,
            "end_year": 2019,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {"name": "embeddings", "proficiency": "expert", "endorsements": 40, "duration_months": 48},
        {"name": "Python", "proficiency": "expert", "endorsements": 60, "duration_months": 72},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 35, "duration_months": 50},
    ],
    "certifications": [
        {"name": "AWS ML Specialty", "issuer": "AWS", "year": 2022}
    ],
    "redrob_signals": {
        "profile_completeness_score": 90,
        "signup_date": "2023-01-01",
        "last_active_date": "2025-05-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 15,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 2.5,
        "skill_assessment_scores": {"Python": 90, "NLP": 85},
        "connection_count": 200,
        "endorsements_received": 120,
        "notice_period_days": 15,
        "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 75,
        "search_appearance_30d": 25,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.95,
        "offer_acceptance_rate": 0.8,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}


def make_candidates(n: int = 100) -> list[dict]:
    """Generate n distinct fake candidates for testing."""
    candidates = []
    for i in range(1, n + 1):
        cand = json.loads(json.dumps(SAMPLE_CANDIDATE))  # deep copy
        cid = f"CAND_{i:07d}"
        cand["candidate_id"] = cid
        cand["profile"]["years_of_experience"] = 3.0 + (i % 10)
        candidates.append(cand)
    return candidates


# ---------------------------------------------------------------------------
# Tests: CandidateLoader
# ---------------------------------------------------------------------------


class TestCandidateLoader:
    def test_load_json(self, tmp_path):
        data = [SAMPLE_CANDIDATE]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        loader = CandidateLoader()
        result = loader.load(p)
        assert len(result) == 1
        assert result[0]["candidate_id"] == "CAND_0000001"

    def test_load_jsonl(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text(
            json.dumps(SAMPLE_CANDIDATE) + "\n" + json.dumps(SAMPLE_CANDIDATE),
            encoding="utf-8",
        )
        loader = CandidateLoader()
        result = loader.load(p)
        assert len(result) == 2

    def test_raises_on_missing_file(self, tmp_path):
        loader = CandidateLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.jsonl")

    def test_raises_on_bad_extension(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("[]")
        loader = CandidateLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(p)


# ---------------------------------------------------------------------------
# Tests: FeatureEngineer
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    def setup_method(self):
        self.engineer = FeatureEngineer(config_path="configs/scoring_weights.yaml")

    def test_transform_returns_dataframe(self):
        df = self.engineer.transform([SAMPLE_CANDIDATE])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_experience_score_peak(self):
        df = self.engineer.transform([SAMPLE_CANDIDATE])
        # 6.5 yrs is in the sweet spot → should be 1.0
        assert df.loc["CAND_0000001", "experience_score"] == 1.0

    def test_required_feature_columns(self):
        df = self.engineer.transform([SAMPLE_CANDIDATE])
        required = [
            "experience_score", "core_ai_skill_score", "skill_depth_score",
            "title_score", "availability_score", "engagement_score",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_scores_in_range(self):
        df = self.engineer.transform([SAMPLE_CANDIDATE])
        score_cols = [c for c in df.columns if c.endswith("_score") or c.endswith("_flag")]
        for col in score_cols:
            assert df[col].between(-1.01, 2.0).all(), f"Column {col} out of range"

    def test_keyword_stuffer_detection(self):
        """A marketing manager with 9 AI skills should be flagged."""
        stuffer = json.loads(json.dumps(SAMPLE_CANDIDATE))
        stuffer["profile"]["current_title"] = "Marketing Manager"
        stuffer["skills"] = [
            {"name": skill, "proficiency": "expert", "endorsements": 5, "duration_months": 6}
            for skill in [
                "Python", "NLP", "embeddings", "FAISS", "transformers",
                "RAG", "LLM", "Fine-tuning LLMs", "LoRA",
            ]
        ]
        df = self.engineer.transform([stuffer])
        assert df.loc["CAND_0000001", "keyword_stuffer_flag"] == 1.0


# ---------------------------------------------------------------------------
# Tests: CandidateRanker
# ---------------------------------------------------------------------------


class TestCandidateRanker:
    def setup_method(self):
        self.engineer = FeatureEngineer(config_path="configs/scoring_weights.yaml")
        self.ranker = CandidateRanker(config_path="configs/scoring_weights.yaml")

    def test_rank_output_shape(self):
        candidates = make_candidates(150)
        feature_df = self.engineer.transform(candidates)
        ranked = self.ranker.rank(feature_df, top_n=100)
        assert len(ranked) == 100

    def test_rank_columns(self):
        candidates = make_candidates(100)
        feature_df = self.engineer.transform(candidates)
        ranked = self.ranker.rank(feature_df)
        assert list(ranked.columns) == ["candidate_id", "rank", "score", "reasoning"]

    def test_ranks_are_sequential(self):
        candidates = make_candidates(100)
        feature_df = self.engineer.transform(candidates)
        ranked = self.ranker.rank(feature_df)
        assert sorted(ranked["rank"].tolist()) == list(range(1, 101))

    def test_scores_non_increasing(self):
        candidates = make_candidates(100)
        feature_df = self.engineer.transform(candidates)
        ranked = self.ranker.rank(feature_df).sort_values("rank")
        scores = ranked["score"].tolist()
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i+1} ({scores[i]:.4f}) < rank {i+2} ({scores[i+1]:.4f})"
            )

    def test_scores_in_valid_range(self):
        candidates = make_candidates(100)
        feature_df = self.engineer.transform(candidates)
        ranked = self.ranker.rank(feature_df)
        assert ranked["score"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Tests: SubmissionFormatter
# ---------------------------------------------------------------------------


class TestSubmissionFormatter:
    def _make_valid_df(self) -> pd.DataFrame:
        candidates = make_candidates(100)
        engineer = FeatureEngineer(config_path="configs/scoring_weights.yaml")
        ranker = CandidateRanker(config_path="configs/scoring_weights.yaml")
        feature_df = engineer.transform(candidates)
        return ranker.rank(feature_df)

    def test_write_creates_file(self, tmp_path):
        df = self._make_valid_df()
        out = tmp_path / "submission.csv"
        formatter = SubmissionFormatter()
        formatter.write(df, out)
        assert out.exists()

    def test_output_has_correct_header(self, tmp_path):
        df = self._make_valid_df()
        out = tmp_path / "submission.csv"
        SubmissionFormatter().write(df, out)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert header == "candidate_id,rank,score,reasoning"

    def test_output_has_100_data_rows(self, tmp_path):
        df = self._make_valid_df()
        out = tmp_path / "submission.csv"
        SubmissionFormatter().write(df, out)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 1 header + 100 data
        assert len(lines) == 101

    def test_raises_on_wrong_row_count(self, tmp_path):
        df = self._make_valid_df().head(50)
        # Reindex ranks 1-50 to make a plausible 50-row df
        df = df.copy()
        df["rank"] = range(1, 51)
        out = tmp_path / "submission.csv"
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SubmissionFormatter().write(df, out)
            assert any("50 rows" in str(warning.message) for warning in w)
