"""
ocr_util.py — Cloud OCR extraction module for EduEval AI.

Uses the OCR.Space REST API instead of a local model (EasyOCR/PyTorch).

Why cloud and not local:
    A local EasyOCR reader pulls in PyTorch (~480 MB) plus OpenCV. Render's
    free tier caps instances at 512 MB RAM, so loading the OCR model exhausted
    memory and the whole backend crashed (observed: OCR request hung for
    240 s, then the service returned 502 until Render auto-restarted it).
    Moving OCR off-box keeps the Render instance small and stable.

API: https://ocr.space/parseimage  (POST, multipart/form-data)
Free tier: 25,000 requests/month, max upload size 1 MB (hence _compress).
"""

from __future__ import annotations

import io
import logging
import os

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OCR_API_URL = "https://api.ocr.space/parse/image"

# "helloworld" is OCR.Space's publicly documented sandbox key. Override it by
# setting OCR_API_KEY in backend/.env for production / higher limits.
OCR_API_KEY = os.getenv("OCR_API_KEY", "helloworld")

# Engine 2 is OCR.Space's strongest general engine and handles handwriting
# reasonably well on its own.
OCR_ENGINE = os.getenv("OCR_ENGINE", "2")
# The `isHandwritten` flag is only honoured on a real (paid-tier) API key -- the
# public "helloworld" sandbox key rejects it with HTTP 400. Leave it off unless
# you have upgraded your key.
OCR_IS_HANDWRITTEN = os.getenv("OCR_IS_HANDWRITTEN", "false").lower() == "true"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "60"))

# OCR.Space free tier rejects uploads larger than 1 MB.
_MAX_UPLOAD_BYTES = 1_000_000
# Absolute floor so a hopeless upload still produces *something* rather than a
# 10-pixel-tall image.
_MIN_DIMENSION = 1000


def _encode_jpeg(pil_img: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _prepare_image(image_bytes: bytes) -> tuple[str, bytes, str]:
    """Re-encode the image so it fits OCR.Space's 1 MB free-tier limit.

    Resolution is preserved whenever possible -- exam photos are high-res with
    small text, so shrinking aggressively destroys OCR accuracy. We only step
    down (quality first, then dimensions) when the payload is genuinely too big.

    Returns ``(filename, payload, content_type)``.
    """
    with Image.open(io.BytesIO(image_bytes)) as pil_img:
        pil_img = pil_img.convert("RGB")

        # 1) Try full resolution at good quality -- the common case.
        for quality in (90, 75):
            payload = _encode_jpeg(pil_img, quality)
            if len(payload) <= _MAX_UPLOAD_BYTES:
                return "scan.jpg", payload, "image/jpeg"

        # 2) Still too big: drop resolution progressively, keeping text legible.
        for max_dim in (3000, 2500, 2000, 1500, _MIN_DIMENSION):
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                resized = pil_img.resize(
                    (max(int(pil_img.width * ratio), _MIN_DIMENSION),
                     max(int(pil_img.height * ratio), _MIN_DIMENSION)),
                    Image.LANCZOS,
                )
            else:
                resized = pil_img

            for quality in (85, 70, 55, 40):
                payload = _encode_jpeg(resized, quality)
                if len(payload) <= _MAX_UPLOAD_BYTES:
                    return "scan.jpg", payload, "image/jpeg"

        return "scan.jpg", payload, "image/jpeg"


def ocr_is_configured() -> bool:
    """True when a real (non-sandbox) OCR key has been supplied."""
    return bool(OCR_API_KEY) and OCR_API_KEY != "helloworld"


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract plain text from an image using the OCR.Space API.

    Parameters
    ----------
    image_bytes : bytes
        Raw image data (e.g., the content of an uploaded ``UploadFile``).

    Returns
    -------
    str
        Concatenated OCR text, lines separated by newlines. Returns an empty
        string on failure so a bad upload degrades gracefully instead of
        crashing the request.
    """
    try:
        filename, payload, content_type = _prepare_image(image_bytes)
    except Exception as exc:
        logger.error("OCR: could not decode uploaded image: %s", exc)
        return ""

    if not payload:
        logger.error("OCR: image could not be compressed under the size limit")
        return ""

    payload_fields: dict[str, str] = {
        "apikey": OCR_API_KEY,
        "language": OCR_LANGUAGE,
        "isOverlayRequired": "false",
        "OCREngine": OCR_ENGINE,
        "scale": "true",
    }
    # Only send isHandwritten when explicitly enabled -- the sandbox key 400s on it.
    if OCR_IS_HANDWRITTEN:
        payload_fields["isHandwritten"] = "true"

    try:
        response = requests.post(
            OCR_API_URL,
            files={"file": (filename, payload, content_type)},
            data=payload_fields,
            timeout=OCR_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error("OCR: request to %s timed out after %ss", OCR_API_URL, OCR_TIMEOUT)
        return ""
    except requests.exceptions.RequestException as exc:
        logger.error("OCR: request to %s failed: %s", OCR_API_URL, exc)
        return ""
    except ValueError as exc:
        logger.error("OCR: non-JSON response from %s: %s", OCR_API_URL, exc)
        return ""

    # OCR.Space always returns 200 and reports failures inside the body.
    if data.get("IsErroredOnProcessing"):
        logger.error(
            "OCR: provider rejected the image: %s",
            data.get("ErrorMessage") or "unknown error",
        )
        return ""

    parsed_results = data.get("ParsedResults") or []
    if not parsed_results:
        logger.error("OCR: provider returned no parsed results")
        return ""

    text = (parsed_results[0].get("ParsedText") or "").strip()
    logger.info("OCR: extracted %d characters", len(text))
    return text