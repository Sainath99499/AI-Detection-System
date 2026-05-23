from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "models" / "text_detector"

tokenizer = None
model = None


def _load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True
        )
        model.eval()


def predict_text(text):

    _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    human_score = float(probabilities[0])
    ai_score = float(probabilities[1])

    human_probability = round(human_score * 100, 2)
    ai_probability = round(ai_score * 100, 2)

    max_score = max(human_probability, ai_probability)
    if max_score > 80:
        confidence = "High"
    elif max_score > 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    prediction = "AI Generated" if ai_probability > human_probability else "Human Created"

    return {
        "content_type": "text",
        "prediction": prediction,
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "confidence": confidence
    }

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
