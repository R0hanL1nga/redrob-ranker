Redrob Intelligent Candidate Discovery & Ranking
> \*\*Hackathon submission\*\* for the \[Redrob India Runs Data \& AI Challenge](https://redrob.ai)  
> Ranks the top 100 candidates from a pool of 50K+ profiles against a Senior AI Engineer job description.
---
Table of Contents
Project Overview
Repository Structure
Ranking Methodology
Installation
How to Run
Configuration
Running Tests
Submission Output
---
Project Overview
This pipeline processes the `candidates.jsonl` dataset (487 MB, ~50 000+ profiles) and produces a `submission.csv` of the top 100 candidates ranked against a Senior AI Engineer role at a Series A startup.
Key design decisions driven by the job description:
Signal	Why it matters
Career title	The JD explicitly warns against keyword stuffers; title outweighs skill keywords
Product vs consulting company ratio	JD disqualifies candidates with 100% consulting-firm career
AI career duration	Months in ML/AI-adjacent roles, not just listed skills
Behavioral availability	Inactive or unresponsive candidates are deprioritised
Keyword stuffer penalty	Candidates with non-technical titles + many AI skills get −0.20
---
Repository Structure
```
redrob-ranker/
│
├── rank.py                         # Main entry point (CLI)
│
├── src/
│   ├── data/
│   │   └── loader.py               # Loads .jsonl and .json candidate files
│   ├── features/
│   │   └── engineer.py             # All feature extraction \& scoring
│   └── ranking/
│       ├── ranker.py               # Composite scoring + ranking logic
│       └── formatter.py            # Writes challenge-compliant CSV
│
├── tests/
│   └── test\_pipeline.py            # 18 unit + integration tests (pytest)
│
├── configs/
│   └── scoring\_weights.yaml        # All tunable weights (no code change needed)
│
├── data/                           # ← Put candidates.jsonl here (gitignored)
│   └── .gitkeep
│
├── scripts/
│   └── .gitkeep
│
├── submission\_metadata.yaml        # Fill in your team details
├── requirements.txt
└── README.md
```
---
Ranking Methodology
Architecture Overview
```
candidates.jsonl
      │
      ▼
\[1] CandidateLoader          – Stream-reads JSONL line by line (memory-efficient)
      │
      ▼
\[2] FeatureEngineer          – Extracts 19 numeric features per candidate
      │
      ▼
\[3] CandidateRanker          – Combines features into composite score
      │
      ▼
\[4] SubmissionFormatter      – Writes ranked CSV with reasoning column
      │
      ▼
submission.csv
```
Feature Groups
Primary Score (weighted sum → 0–1)
Component	Weight	Description
`skill\_match`	35%	Core AI skill depth + Redrob assessment scores
`career\_relevance`	30%	Title score + product-company ratio + AI tenure months
`experience\_fit`	15%	Years of experience vs JD target range (5–9 yrs, sweet spot 6–8)
`education`	10%	Institution tier (tier_1 = 1.0 → unknown = 0.4)
`certifications`	10%	Relevant AI/ML certifications (AWS ML, GCP, Databricks, etc.)
Behavioral Modifier (× 0.60–1.00)
The primary score is multiplied by a behavioral modifier derived from Redrob platform signals:
Signal	Weight	Notes
`availability\_score`	35%	Recency (days since last login) + open-to-work flag
`engagement\_score`	30%	Recruiter response rate + response speed + saved-by-recruiters
`reliability\_score`	20%	Interview completion rate + historical offer acceptance rate
`profile\_quality\_score`	15%	Profile completeness + email/phone/LinkedIn verification
> A perfect-on-paper candidate who hasn't logged in for 6 months retains at most \*\*60%\*\* of their primary score.
Penalties
Penalty	Amount	Trigger
Keyword stuffer	−0.20	Non-technical current title + ≥6 core AI skills listed
Scoring Formula
```
primary = 0.35·skill\_match + 0.30·career\_relevance + 0.15·experience\_fit
          + 0.10·education + 0.10·certifications

behavioral\_modifier = clamp(
    0.35·availability + 0.30·engagement + 0.20·reliability + 0.15·profile\_quality,
    min=0.60, max=1.00
)

raw\_score = primary × behavioral\_modifier − 0.20·keyword\_stuffer\_flag
score     = min\_max\_normalise(raw\_score, range=\[0.20, 1.00])
```
Anti-Keyword-Stuffer Design
The JD explicitly states: "The right answer is NOT 'find candidates whose skills section contains the most AI keywords.'"
This pipeline addresses that in two ways:
Title scoring outweighs skill scoring. A Marketing Manager with 9 AI skills scores 0.10 on `title\_score` regardless of skills.
Keyword stuffer flag. Any candidate with a disqualifying current title AND ≥6 AI core skills in their profile receives an explicit −0.20 penalty, preventing skills from compensating for mismatched roles.
---
Installation
Requirements: Python 3.10+
```bash
# Clone the repository
git clone https://github.com/YOUR\_USERNAME/redrob-ranker.git
cd redrob-ranker

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```
---
How to Run
1. Place the dataset
Copy the challenge dataset file into the `data/` directory:
```bash
cp /path/to/candidates.jsonl data/candidates.jsonl
```
2. Run the pipeline
```bash
# Full dataset (produces 100-row submission)
python rank.py --candidates data/candidates.jsonl --out submission.csv

# Sample dataset (for quick verification — produces 50 rows)
python rank.py --candidates data/sample\_candidates.json --out submission\_sample.csv

# Debug mode (verbose logging)
python rank.py --candidates data/candidates.jsonl --out submission.csv --debug
```
3. Validate the submission
Use the official challenge validator:
```bash
python validate\_submission.py submission.csv
```
Expected output: `Submission is valid.`
Runtime
Dataset	Candidates	Approx. time (8-core CPU, 16 GB RAM)
`sample\_candidates.json`	50	< 1 second
`candidates.jsonl`	~50 000	~45–60 seconds
---
Configuration
All scoring weights live in `configs/scoring\_weights.yaml`. You can tune them without changing any Python code:
```yaml
primary\_weights:
  skill\_match: 0.35       # ← increase to favour technical skills more
  career\_relevance: 0.30
  experience\_fit: 0.15
  education: 0.10
  certifications: 0.10

behavioral\_modifier\_min: 0.60   # ← set to 1.0 to ignore behavioral signals
keyword\_stuffer\_penalty: 0.20   # ← increase/decrease the stuffer penalty
```
---
Running Tests
```bash
pytest tests/ -v
```
Expected: 18 passed
```
tests/test\_pipeline.py::TestCandidateLoader::test\_load\_json             PASSED
tests/test\_pipeline.py::TestCandidateLoader::test\_load\_jsonl            PASSED
...
tests/test\_pipeline.py::TestSubmissionFormatter::test\_output\_has\_100\_data\_rows PASSED
18 passed in 0.61s
```
---
Submission Output
The output `submission.csv` has exactly this structure:
```csv
candidate\_id,rank,score,reasoning
CAND\_0000031,1,1.0000,6.0 yrs exp; 7 core AI skills; engagement=0.69; product-company background; actively available
CAND\_0000001,2,0.8502,6.9 yrs exp; 4 core AI skills; engagement=0.34; product-company background; actively available
...
```
`rank` — integer 1–100 (unique, sequential)
`score` — float in [0.20, 1.00], non-increasing by rank
`reasoning` — human-readable explanation of the score components
---
License
MIT