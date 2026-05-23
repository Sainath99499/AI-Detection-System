from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch

MODEL_PATH = "sainathk07888/text-detector-model"

print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    low_cpu_mem_usage=True
)

model.eval()

print("Model loaded successfully!")

def predict_text(text):

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

    probabilities = torch.softmax(logits, dim=1)

    probabilities = probabilities.cpu().numpy()[0]

    human_score = float(probabilities[0])
    ai_score = float(probabilities[1])

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

    # Determine prediction
    prediction = "AI Generated" if ai_probability > human_probability else "Human Created"

    return {
        "content_type": "text",
        "prediction": prediction,
        "human_probability": human_probability,
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