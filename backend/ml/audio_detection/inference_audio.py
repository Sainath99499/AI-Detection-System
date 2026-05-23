import torch
import librosa
import numpy as np

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model
)

# =========================================
# MODEL CONFIG
# =========================================

MODEL_NAME = "facebook/wav2vec2-base-960h"

print("Loading audio model...")

processor = Wav2Vec2Processor.from_pretrained(
    MODEL_NAME
)

model = Wav2Vec2Model.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Audio model loaded successfully!")

# =========================================
# AUDIO PREDICTION
# =========================================

def predict_audio(audio_path):

    # LOAD AUDIO

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000
    )

    # PROCESS AUDIO

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state

    # SIMPLE HEURISTIC MVP

    mean_energy = torch.mean(
        torch.abs(hidden_states)
    ).item()

    # SIMULATED AI SCORE

    ai_probability = round(
        min(mean_energy * 100, 99),
        2
    )

    human_probability = round(
        100 - ai_probability,
        2
    )

    prediction = "Human Voice"

    if ai_probability > human_probability:
        prediction = "AI Generated Voice"

    confidence = "Low"

    max_score = max(
        ai_probability,
        human_probability
    )

    if max_score > 80:
        confidence = "High"

    elif max_score > 50:
        confidence = "Medium"

    return {
        "content_type": "audio",
        "prediction": prediction,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "confidence": confidence
    }

# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    result = predict_audio(
        "sample.wav.ogg"
    )

    print(result)