import io
import math
import sys
import requests
from PIL import Image, ImageDraw

BASE_URL = "http://localhost:8000"


def print_status(test_name: str, passed: bool, details: str = ""):
    if passed:
        print(f"[SUCCESS] {test_name}" + (f" - {details}" if details else ""))
    else:
        print(f"[FAILURE] {test_name}" + (f" - {details}" if details else ""))


def create_dummy_image_bytes() -> bytes:
    """Create a simple in-memory PNG image with text drawn on it."""
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 40), "EduEval", fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_text():
    print("\n--- Running Test 1: /extract-text ---")
    url = f"{BASE_URL}/extract-text"
    img_bytes = create_dummy_image_bytes()

    files = {
        "file": ("test_image.png", img_bytes, "image/png")
    }

    try:
        response = requests.post(url, files=files, timeout=120)
    except requests.exceptions.RequestException as e:
        print_status("Endpoint Reachability", False, f"Could not connect to {url}: {e}")
        return False

    if response.status_code != 200:
        print_status("HTTP Status Code", False, f"Expected 200, got {response.status_code} ({response.text})")
        return False
    print_status("HTTP Status Code", True, "200 OK")

    try:
        data = response.json()
    except Exception as e:
        print_status("JSON Response Parsing", False, f"Failed to parse response as JSON: {e}")
        return False

    if "extracted_text" not in data:
        print_status("Extracted Text Field Check", False, "'extracted_text' key missing in response payload")
        return False
    
    print_status("Extracted Text Field Check", True, f"Found 'extracted_text' (Value: '{data['extracted_text']}')")
    return True


def test_evaluate():
    print("\n--- Running Test 2: /evaluate ---")
    url = f"{BASE_URL}/evaluate"
    payload = {
        "question": "What is the primary function of chlorophyll in plants?",
        "reference_answer": "Chlorophyll absorbs light energy, usually from the sun, to facilitate photosynthesis.",
        "student_answer": "It captures solar energy to help the plant synthesize food through photosynthesis."
    }

    response = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                break
            if response.status_code in (429, 502) and attempt < 2:
                import time
                print(f"[INFO] Free-tier rate limit hit. Waiting 25s before retry (attempt {attempt + 1}/3)...")
                time.sleep(25)
                continue
            break
        except requests.exceptions.RequestException as e:
            print_status("Endpoint Reachability", False, f"Could not connect to {url}: {e}")
            return False

    if response is None or response.status_code != 200:
        err_msg = response.text if response else "No response"
        print_status("HTTP Status Code", False, f"Expected 200, got {getattr(response, 'status_code', 'None')} ({err_msg})")
        return False
    print_status("HTTP Status Code", True, "200 OK")

    try:
        data = response.json()
    except Exception as e:
        print_status("JSON Response Parsing", False, f"Failed to parse JSON response: {e}")
        return False

    # 1. Category validation
    category = data.get("category")
    valid_categories = {"correct", "contradictory", "incorrect"}
    if category not in valid_categories:
        print_status("Category Validation", False, f"Invalid category '{category}'. Expected one of {valid_categories}")
        return False
    print_status("Category Validation", True, f"Category is valid ('{category}')")

    # 2. Probabilities dictionary validation
    probs = data.get("probabilities")
    if not isinstance(probs, dict):
        print_status("Probabilities Structure Check", False, "Field 'probabilities' is not a dictionary")
        return False

    required_prob_keys = {"correct", "contradictory", "incorrect"}
    if not required_prob_keys.issubset(probs.keys()):
        print_status("Probabilities Keys Check", False, f"Missing keys in probabilities. Expected {required_prob_keys}, got {set(probs.keys())}")
        return False
    print_status("Probabilities Keys Check", True, f"All 3 probability keys present: {list(probs.keys())}")

    # 3. Probabilities sum validation
    prob_values = [probs[k] for k in required_prob_keys]
    if not all(isinstance(v, (int, float)) for v in prob_values):
        print_status("Probabilities Type Check", False, "Probabilities values must be float/numeric")
        return False

    prob_sum = sum(prob_values)
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        print_status("Probabilities Sum Check", False, f"Probabilities must sum to ~1.0, got {prob_sum}")
        return False
    print_status("Probabilities Sum Check", True, f"Probabilities sum correctly to {prob_sum:.4f}")

    # 4. Reasoning string validation
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        print_status("Reasoning Validation", False, "'reasoning' must be a non-empty string")
        return False
    print_status("Reasoning Validation", True, f"Reasoning provided ({len(reasoning)} chars)")

    return True


def main():
    print("====================================================")
    print("      EduEval AI Backend Automated Test Suite       ")
    print(f"      Target Host: {BASE_URL}                      ")
    print("====================================================")

    ocr_passed = test_extract_text()
    eval_passed = test_evaluate()

    print("\n====================================================")
    print("                    TEST SUMMARY                    ")
    print("====================================================")
    print(f"/extract-text : {'PASSED' if ocr_passed else 'FAILED'}")
    print(f"/evaluate     : {'PASSED' if eval_passed else 'FAILED'}")
    print("====================================================")

    if not (ocr_passed and eval_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
