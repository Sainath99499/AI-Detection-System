import os
import httpx

MODEL_REPO = "sainathk07888/text-detector-model"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_REPO}"

HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
} if HF_TOKEN else {}


def predict_text(text):

    # =========================================
    # CALL HUGGING FACE INFERENCE API
    # =========================================

    payload = {
        "inputs": text
    }

    try:

        with httpx.Client(timeout=60.0) as client:

            response = client.post(
                API_URL,
                headers=HEADERS,
                json=payload
            )

            response.raise_for_status()

            result = response.json()

    except Exception as e:

        raise RuntimeError(
            f"Hugging Face inference failed: {e}"
        )

    # =========================================
    # PARSE RESPONSE
    # =========================================

    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(
            f"Hugging Face API error: {result.get('error')}"
        )

    if isinstance(result, list) and result and isinstance(result[0], list):
        predictions = result[0]
    elif isinstance(result, list):
        predictions = result
    else:
        raise RuntimeError(
            f"Unexpected Hugging Face response type: {type(result).__name__} {result}"
        )

    human_score = 0.0
    ai_score = 0.0

    for item in predictions:

        if not isinstance(item, dict) or "label" not in item or "score" not in item:
            continue

        label = str(item["label"]).lower()
        score = float(item["score"])

        if "human" in label or label in ["label_0", "0"]:
            human_score = score
        elif "ai" in label or "machine" in label or label in ["label_1", "1"]:
            ai_score = score

    if human_score == 0.0 and ai_score == 0.0:
        raise RuntimeError(
            f"Unable to parse prediction scores from response: {result}"
        )

    # =========================================
    # CONVERT TO PERCENTAGES
    # =========================================

    human_probability = round(human_score * 100, 2)

    ai_probability = round(ai_score * 100, 2)

    # =========================================
    # CONFIDENCE LOGIC
    # =========================================

    max_score = max(
        human_probability,
        ai_probability
    )

    if max_score > 80:
        confidence = "High"

    elif max_score > 50:
        confidence = "Medium"

    else:
        confidence = "Low"

    # =========================================
    # FINAL PREDICTION
    # =========================================

    prediction = (
        "AI Generated"
        if ai_probability > human_probability
        else "Human Created"
    )

    return {

        "content_type": "text",

        "prediction": prediction,

        "human_probability": human_probability,

        "ai_probability": ai_probability,

        "confidence": confidence
    }


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is transforming industries worldwide.
    """

    result = predict_text(sample_text)

    print(result)
