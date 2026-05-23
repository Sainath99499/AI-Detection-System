import os
import httpx

MODEL_REPO = "sainathk07888/text-detector-model"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_REPO}"

HF_TOKEN = os.environ.get("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def predict_text(text):

    # Call Hugging Face Inference API to avoid loading large models in-process
    payload = {"inputs": text}

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(API_URL, headers=HEADERS, json=payload)
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        raise RuntimeError(f"HF inference request failed: {e}")

    # Expected result: list of {label,score} for sequence-classification
    if isinstance(result, list) and len(result) >= 2:
        try:
            human_score = float(result[0].get("score", 0.0))
            ai_score = float(result[1].get("score", 0.0))
        except Exception:
            # Fallback: treat first as human, second as ai
            human_score = float(result[0]["score"])
            ai_score = float(result[1]["score"])
    else:
        # Unknown response format: attempt to parse
        raise RuntimeError(f"Unexpected HF inference response: {result}")

    # Convert to percentages
    human_probability = round(human_score * 100, 2)
    ai_probability = round(ai_score * 100, 2)

    # Confidence logic
    confidence = "Low"
    max_score = max(human_probability, ai_probability)
    if max_score > 80:
        confidence = "High"
    elif max_score > 50:
        confidence = "Medium"

    prediction = "AI Generated" if ai_probability > human_probability else "Human Created"

    return {
        "content_type": "text",
        "prediction": prediction,
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "confidence": confidence
    }
        "ai_probability": ai_probability,
        "confidence": confidence
    }

# TEST

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is transforming industries worldwide.
    """

    result = predict_text(sample_text)

    print(result)