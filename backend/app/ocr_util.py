import io
from typing import List
import numpy as np
from PIL import Image
import easyocr

# Lazy loader for EasyOCR reader with verbose=False to prevent Windows cp1252 UnicodeEncodeError
_reader = None

def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract plain text from an image using EasyOCR.

    Parameters
    ----------
    image_bytes: bytes
        Raw image data (e.g., the content of an uploaded ``UploadFile``).

    Returns
    -------
    str
        Concatenated OCR result – each line separated by a newline.

    The function is wrapped in a ``try/except`` block so that
    corrupted or unsupported images do not crash the service.
    In case of failure, an empty string is returned.
    """
    try:
        # Load image via Pillow and convert to NumPy array expected by EasyOCR.
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Ensure image is in RGB mode.
            pil_img = pil_img.convert("RGB")
            img_np: np.ndarray = np.array(pil_img)
        reader = get_reader()
        results = reader.readtext(img_np, detail=0, paragraph=False)
        # Join all detected strings with newlines (preserve line order).
        return "\n".join(results)
    except Exception as exc:
        # Log the exception if needed (omitted here for brevity).
        # Return an empty string to indicate extraction failure.
        return ""
