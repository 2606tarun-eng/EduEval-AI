import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.llm_eval import evaluate_answer_semantics
from app.schemas import (
    CategoryProbabilities,
    EvaluationCategory,
    EvaluationRequest,
    EvaluationResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EduEval AI Backend",
    description=(
        "AI-powered student answer evaluation API. "
        "Classifies answers as **correct**, **contradictory**, or **incorrect** "
        "with per-category confidence probabilities and semantic reasoning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "EduEval AI Team",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# CORS Middleware
# Allows the Streamlit frontend (any origin) to communicate freely.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Open for all origins (frontend, Streamlit, etc.)
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, PUT, DELETE, OPTIONS …
    allow_headers=["*"],       # Content-Type, Authorization …
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Health Check",
    description="Verify that the EduEval AI Backend service is running.",
    tags=["Health"],
    response_description="Service status and version information.",
)
async def health_check() -> dict:
    """
    Returns a simple JSON payload confirming the API is online.
    Use this endpoint for uptime monitoring or readiness checks.
    """
    return {
        "status": "online",
        "service": "EduEval AI Backend",
        "version": app.version,
    }


# ---------------------------------------------------------------------------
# Evaluation Route — LLM-powered semantic assessment (ED-05)
# ---------------------------------------------------------------------------

@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate Student Answer",
    description=(
        "Accepts a question, an optional reference answer, and the student's answer. "
        "Calls the LLM with a bias-resistant system prompt (ED-05) and returns a "
        "classification (`correct` / `contradictory` / `incorrect`), confidence "
        "probabilities, and semantic reasoning."
    ),
    tags=["Evaluation"],
)
async def evaluate_answer(request: EvaluationRequest) -> EvaluationResponse:
    """
    Core evaluation endpoint.

    1. Sends the payload to the LLM via ``evaluate_answer_semantics``.
    2. Parses the returned dict into a validated ``EvaluationResponse``.
    3. Returns structured JSON to the frontend.
    """

    # ── Call the LLM ────────────────────────────────────────────────────────
    try:
        llm_result: dict = await evaluate_answer_semantics(request)
    except RuntimeError as exc:
        # OpenAI API-level failures (network, auth, rate-limit)
        logger.error("LLM API failure: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM service unavailable: {exc}",
        ) from exc
    except ValueError as exc:
        # LLM returned non-JSON or missing keys
        logger.error("LLM response parsing failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned an invalid response: {exc}",
        ) from exc

    # ── Build the validated Pydantic response ───────────────────────────────
    try:
        response = EvaluationResponse(
            category=EvaluationCategory(llm_result["category"]),
            probabilities=CategoryProbabilities(**llm_result["probabilities"]),
            reasoning=llm_result["reasoning"],
        )
    except (KeyError, ValueError) as exc:
        logger.error("Failed to construct EvaluationResponse: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM output could not be mapped to the response schema: {exc}"
            ),
        ) from exc

    return response


# ---------------------------------------------------------------------------
# OCR — Extract Text from Image
# ---------------------------------------------------------------------------

@app.post(
    "/extract-text",
    summary="Extract text from an uploaded image (OCR)",
    description=(
        "Accepts a JPEG/PNG image and returns the extracted plain text using "
        "EasyOCR.  The endpoint validates the content-type and gracefully handles "
        "corrupted files."
    ),
    tags=["OCR"],
)
async def extract_text(
    file: UploadFile = File(..., description="Image file (jpeg/png)"),
) -> dict:
    """Read an uploaded image and return OCR-extracted text."""

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPEG or PNG image.",
        )

    image_bytes: bytes = await file.read()

    from app.ocr_util import extract_text_from_image

    extracted: str = extract_text_from_image(image_bytes)

    return {"extracted_text": extracted}
