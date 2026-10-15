"""
llm_eval.py — Core LLM evaluation logic for EduEval AI (ED-05).

Sends the question, reference answer, and student answer to an
OpenAI-compatible chat model with a strict system prompt that enforces
bias-resistant, semantic-only evaluation.  Returns raw JSON with exactly
three keys: ``category``, ``probabilities``, and ``reasoning``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.schemas import EvaluationRequest

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client (reads env vars loaded from .env or system environment)
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
)

# Priority order of available Gemini Flash models for resilient free-tier quota failover
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]

# ---------------------------------------------------------------------------
# Failover tuning
# ---------------------------------------------------------------------------

# How many times to retry the SAME model on a 429 before rotating away from it.
MAX_RETRIES_PER_MODEL = int(os.getenv("LLM_MAX_RETRIES", "3"))

# How long a quota-exhausted model is skipped. Without this every new request
# re-tries a permanently exhausted model and pays a failed round-trip first.
MODEL_COOLDOWN_SECONDS = int(os.getenv("LLM_COOLDOWN_SECONDS", "300"))

# Backoff base for 429 retries (doubles each attempt, capped).
RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))
RETRY_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0"))

# model name -> monotonic timestamp until which it is considered exhausted
_model_cooldown_until: Dict[str, float] = {}


def candidate_models() -> List[str]:
    """Models to try, in order: ``LLM_MODEL`` first, then the fallback list.

    ``LLM_MODEL`` used to be ignored entirely -- the rotation was hardcoded to
    FALLBACK_MODELS -- so changing it in ``.env`` had no effect whatsoever.
    """
    ordered: List[str] = []

    primary = (os.getenv("LLM_MODEL") or "").strip()
    if primary:
        ordered.append(primary)

    for model in FALLBACK_MODELS:
        if model not in ordered:
            ordered.append(model)

    return ordered


def models_in_cooldown() -> List[str]:
    """Models currently considered quota-exhausted."""
    now = time.monotonic()
    return [m for m, until in _model_cooldown_until.items() if until > now]


def mark_cooldown(model: str) -> None:
    _model_cooldown_until[model] = time.monotonic() + MODEL_COOLDOWN_SECONDS
    logger.warning("model %s exhausted; cooling down for %ds", model, MODEL_COOLDOWN_SECONDS)


def clear_cooldown(model: str) -> None:
    _model_cooldown_until.pop(model, None)


def _is_rate_limit(exc: Exception) -> bool:
    """True when the failure is a quota/rate limit worth retrying."""
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "ratelimit", "resource_exhausted", "quota exceeded")
    )


def get_model_name(attempt: int = 0) -> str:
    """Backwards-compatible accessor: the nth candidate model."""
    models = candidate_models()
    return models[attempt % len(models)]


async def _complete_with_failover(user_message: str) -> Tuple[Any, str]:
    """Call the chat model, retrying on 429 and rotating on sustained failure.

    Returns ``(response, model_used)``.
    """
    models = candidate_models()
    now = time.monotonic()
    available = [m for m in models if _model_cooldown_until.get(m, 0.0) <= now]
    skipped = [m for m in models if m not in available]

    if skipped:
        logger.warning("skipping models in cooldown: %s", skipped)

    if not available:
        # Everything is exhausted. Clear the cooldowns and try anyway rather
        # than failing outright -- quotas reset on a rolling window, so the
        # earliest-cooling model may well be usable again.
        logger.warning("all models in cooldown; resetting and retrying anyway")
        _model_cooldown_until.clear()
        available = models

    failures: List[str] = []

    for model in available:
        delay = RETRY_BASE_DELAY

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                logger.info(
                    "Calling LLM model: %s (attempt %d/%d)", model, attempt, MAX_RETRIES_PER_MODEL
                )
                response = await _client.chat.completions.create(
                    model=model,
                    temperature=0.0,  # 0.0 for deterministic, objective grading
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )
                clear_cooldown(model)
                return response, model

            except Exception as exc:
                if _is_rate_limit(exc):
                    if attempt < MAX_RETRIES_PER_MODEL:
                        wait = min(delay, RETRY_MAX_DELAY)
                        logger.warning(
                            "rate limited on %s (attempt %d/%d); retrying in %.1fs",
                            model, attempt, MAX_RETRIES_PER_MODEL, wait,
                        )
                        await asyncio.sleep(wait)
                        delay *= 2
                        continue

                    # Retries exhausted for this model -- stop using it briefly.
                    mark_cooldown(model)
                    failures.append(f"{model}: {exc}")
                else:
                    # Auth errors, malformed requests, network blips: rotating
                    # models will not help, but try the next one anyway rather
                    # than aborting the whole evaluation.
                    logger.warning("call to %s failed: %s", model, exc)
                    failures.append(f"{model}: {exc}")

                break  # move on to the next model

    detail = "; ".join(failures) if failures else "no models available"
    raise RuntimeError(f"LLM API error after trying {len(available)} model(s) -> {detail}")

# ---------------------------------------------------------------------------
# System prompt — ED-05 bias-resistant assessment rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are EduEval AI, a strict, bias-resistant short-answer assessment engine.

═══ EVALUATION PROTOCOL (ED-05) ═══

1. SEMANTIC-ONLY EVALUATION
   • Assess the student's answer based SOLELY on whether the underlying meaning
     is factually and conceptually aligned with the question (and the reference
     answer, if provided).
   • Two phrasings that express the same idea MUST be treated as equivalent —
     ignore differences in vocabulary, sentence structure, grammar, style,
     verbosity, or formatting.

2. BIAS-RESISTANCE RULES
   • IGNORE superficial keyword matching: a student who uses different but
     correct terminology is still correct.
   • RESIST keyword-stuffing attacks: an answer that repeats many relevant
     keywords but makes no coherent claim is INCORRECT.
   • DO NOT penalise for: spelling mistakes, minor grammatical errors,
     informal tone, use of abbreviations, or unusual but valid phrasing.
   • DO NOT reward or penalise based on answer length alone.
   • If no reference answer is provided, evaluate entirely on factual/semantic
     correctness of the student's answer with respect to the question.

3. CLASSIFICATION (choose exactly one)
   • "correct"       — The core meaning is factually right and addresses the
                        question.  Minor omissions that do not change
                        correctness are acceptable.
   • "contradictory" — The answer contains statements that directly oppose the
                        correct facts OR contains a mix of correct and
                        incorrect claims such that the overall answer cannot
                        be accepted as correct.
   • "incorrect"     — The answer is factually wrong, irrelevant, nonsensical,
                        or fails to address the question in any meaningful way.

4. CONFIDENCE PROBABILITIES
   • Assign a probability (0.0–1.0) for EACH of the three categories.
   • The three probabilities MUST sum to exactly 1.0.
   • The category with the highest probability MUST match your chosen
     ``category`` value.

5. REASONING
   • Provide a concise (2–4 sentence) semantic explanation of WHY the answer
     falls into the chosen category.  Cite specific conceptual matches or
     mismatches — never reference surface wording alone.

6. SEMEVAL 2013 TASK 7 BENCHMARK EXAMPLES (FEW-SHOT TRAINING EXAMPLES)
   • Example 1 (Ground Truth: "correct"):
     Question: Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.
     Reference: Terminal 1 and the positive terminal are separated by the gap
     Student: positive battery terminal is separated by a gap from terminal 1
     Category: "correct" (matches the core proposition that a gap exists between the terminals).

   • Example 2 (Ground Truth: "contradictory"):
     Question: Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.
     Reference: Terminal 1 and the positive terminal are separated by the gap
     Student: Because terminal 1 is connected to the positive battery terminal
     Category: "contradictory" (claims connection, directly opposing the physical fact of a gap).

   • Example 3 (Ground Truth: "incorrect"):
     Question: Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.
     Reference: Terminal 1 and the positive terminal are separated by the gap
     Student: the terminal is connected to the battery
     Category: "incorrect" (vague and misses the essential causal mechanism of the gap).

═══ OUTPUT FORMAT ═══

Return ONLY a single, raw JSON object.  No markdown, no code fences, no
explanation outside the JSON.  The object MUST contain exactly these keys:

{
  "category": "<correct | contradictory | incorrect>",
  "probabilities": {
    "correct": <float>,
    "contradictory": <float>,
    "incorrect": <float>
  },
  "reasoning": "<2–4 sentence semantic explanation>"
}
"""

# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


async def evaluate_answer_semantics(data: EvaluationRequest) -> dict[str, Any]:
    """Call the LLM and return the parsed evaluation dict.

    Parameters
    ----------
    data : EvaluationRequest
        Validated request payload containing ``question``,
        ``reference_answer`` (optional), and ``student_answer``.

    Returns
    -------
    dict[str, Any]
        A dictionary with keys ``category``, ``probabilities``, and
        ``reasoning`` — ready to be unpacked into an ``EvaluationResponse``.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed as valid JSON or is missing
        required keys.
    RuntimeError
        If the OpenAI API call itself fails (network, auth, rate-limit, etc.).
    """

    # ── Build the user message ──────────────────────────────────────────────
    user_parts: list[str] = [
        f"**Question:**\n{data.question}",
        f"**Student Answer:**\n{data.student_answer}",
    ]
    if data.reference_answer:
        user_parts.insert(
            1,
            f"**Reference Answer:**\n{data.reference_answer}",
        )
    user_message = "\n\n".join(user_parts)

    # ── Call the LLM with failover for resilient free-tier handling ─────────
    response, model_used = await _complete_with_failover(user_message)
    logger.info("Evaluation completed using model: %s", model_used)

    raw_text: str = (response.choices[0].message.content or "").strip()

    # ── Strip accidental code fences ────────────────────────────────────────
    if raw_text.startswith("```"):
        # Remove ```json ... ``` wrappers some models add despite instructions
        lines = raw_text.splitlines()
        # Drop opening fence line and closing fence line
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    # ── Parse & validate JSON ───────────────────────────────────────────────
    try:
        parsed: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned non-JSON: %s", raw_text[:300])
        raise ValueError(
            f"LLM returned invalid JSON. Raw output: {raw_text[:200]}"
        ) from exc

    required_keys = {"category", "probabilities", "reasoning"}
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(
            f"LLM JSON is missing required keys: {missing}. "
            f"Got: {list(parsed.keys())}"
        )

    # ── Normalise category to lowercase ─────────────────────────────────────
    parsed["category"] = str(parsed["category"]).strip().lower()

    valid_categories = {"correct", "contradictory", "incorrect"}
    if parsed["category"] not in valid_categories:
        raise ValueError(
            f"LLM returned invalid category '{parsed['category']}'. "
            f"Must be one of {valid_categories}."
        )

    # ── Validate probabilities dict ─────────────────────────────────────────
    probs = parsed.get("probabilities", {})
    prob_missing = {"correct", "contradictory", "incorrect"} - set(probs.keys())
    if prob_missing:
        raise ValueError(
            f"probabilities object is missing keys: {prob_missing}"
        )

    return parsed
