# AGENTS.md — Working notes for this repo

Context for AI agents and new contributors. Read this before changing anything.

The user-facing documentation is `README.md`. This file is about **why the code
is shaped the way it is**, and the traps that are easy to reintroduce.

---

## What this is

ED-05 bias-resistant short-answer assessment. Grades student answers as
`correct` / `contradictory` / `incorrect` with calibrated probabilities, and
scores itself on the SemEval 2013 Task 7 benchmark.

---

## Architecture — what runs where

Nothing heavy runs on the user's laptop. Each task has one owner:

| Task | Runs on | Notes |
|---|---|---|
| Live grading `/evaluate` | **Gemini API** via Render backend | low volume, failover across models |
| OCR `/extract-text` | **OCR.Space API** | never bring back a local model |
| Benchmark scorecard | **local sklearn model** | zero API calls, ~2s for 7,093 items |
| Dashboard | **Streamlit Community Cloud** | thin client, proxies to Render |

**Do not couple these.** The benchmark deliberately avoids the API so the
scorecard stays reproducible and free.

---

## Critical constraints

### 1. Render's free tier caps RAM at 512 MB

This is why OCR moved to a cloud API. Local EasyOCR pulled in PyTorch (~480 MB)
and exhausted memory on the first image upload, taking the **entire backend**
down — the service returned 502 until Render auto-restarted it.

**Never add `torch`, `easyocr`, `opencv`, or any large transformer dependency to
`requirements.txt`.** They will OOM the deployed service. A transformer would
have to be exported to quantized ONNX before it could safely ship here.

This also rules out most GPU-model upgrades unless they go through ONNX
quantization first.

### 2. Three things in `frontend/app.py` that are easy to break

- **The warmup block** (~line 78-95). Render spins down after ~15 min idle and
  takes ~83s to wake. `threading.Thread(target=_warmup_backend, daemon=True)`
  starts that wake on page load so it overlaps with the user reading the page.
  Removing it means every cold visitor stares at a spinner for 83s.
- **`DEFAULT_BASE_URL = os.getenv("BACKEND_BASE_URL", ...)`.** Never hardcode the
  backend URL — the env var is what makes local testing possible.
- **Timeouts `API_TIMEOUT` / `OCR_TIMEOUT`** (150s / 90s). The old 60s value was
  below Render's measured 83s cold start and failed reliably.

### 3. `.streamlit/config.toml` must track the CSS in `frontend/app.py`

The colours there mirror the app's custom dark theme. Changing the CSS without
updating the TOML gives mismatched native widgets against the styled UI.

---

## The validation trap (most important thing in this file)

**A random train/validation split of the SemEval training data reports
Macro-F1 ≈ 0.77. The real test split scores ≈ 0.46.**

SemEval's official test set is built from *unseen questions* and *unseen
domains*. A random split leaks question context across the train/val boundary
and produces a number that is badly inflated.

Use **question-grouped** validation (`StratifiedGroupKFold` grouped by
`question_id`), as `train_model_v2.py` does. With grouped splits, validation
(0.4489) tracks test (0.4729) closely and model selection becomes meaningful.

Concrete example of the damage: `CalibratedClassifierCV(LinearSVC)` scored
**0.6155** on the leaky split and **0.2541** on unseen questions. The internal
`cv=3` splits randomly, so it leaks too. Trusting it would have shipped a broken
model.

**The official test split is never used for model selection.** It is reported
once, at the end.

---

## Other traps

- **`LLM_MODEL` was dead config for a long time.** `get_model_name()` indexed
  only a hardcoded list, so setting it in `.env` did nothing. It is now the
  primary candidate with the list as fallbacks. Don't reintroduce a hardcoded
  model that ignores configuration.

- **429 vs other errors.** A 429 is worth retrying the *same* model with
  backoff. A 401 or a malformed request is not — rotating models won't fix a
  bad API key, and retrying it three times just wastes the quota window.
  `_is_rate_limit()` is the only place that distinction should live.

- **Exhausted models need cooldown.** Without it, every new request restarts at
  index 0 and pays a failed round-trip against a permanently exhausted model
  before reaching a working one. `_model_cooldown_until` fixes this.

- **`baseline_model.joblib` is deliberately committed.** `.gitignore` excludes
  `*.joblib` but negates this one file. A grader running `git clone &&
  python local_evaluation.py` must not have to retrain first — reproducibility is
  5% of the score. Regenerate with `python train_baseline.py`.

---

## Running things

```bash
# backend
cd backend && uvicorn app.main:app --port 8000 --reload

# frontend
streamlit run frontend/app.py

# benchmark (default = local model, zero API calls)
python local_evaluation.py --data-dir "data/semeval-2013-task7/extracted_3way/test/3way"

# benchmark against the live API instead (~5 calls/item, will exhaust free quota)
python local_evaluation.py --no-model --data-dir ... --limit 10

# integration tests (needs backend running)
python test_backend.py

# CPU-only model search
python train_model_v2.py
```

---

## Current state

Benchmark on the official test split (7,093 items, ~2s):

| Metric | Score | Weighted |
|---|---|---|
| Macro-F1 | 0.4692 | 21.12 / 45 |
| CF1–CF3 consistency | 0.8904 | 26.71 / 30 |
| CF4 resistance | 0.9895 | 14.84 / 15 |
| Calibration | 0.7090 | 3.55 / 5 |
| **Total** | | **66.21 / 95** |

**Macro-F1 is the bottleneck** (~24 points available there). Six sparse-linear
configurations were tried in `train_model_v2.py`; the original TF-IDF +
LogReg baseline remains the best. CPU-only ceiling is ~0.47 for this task —
meaningful gains need a transformer, which is blocked by the 512 MB constraint
above unless exported to ONNX.

Expect ~83s cold starts on the Render backend. That is the free tier, not a bug.

---

## Deploying

Push to `main`; both Render and Streamlit Cloud redeploy automatically and
neither URL changes. Anything pushed straight from the GitHub web UI will cause
a non-fast-forward on the next local `git push` — `git pull --rebase` first.