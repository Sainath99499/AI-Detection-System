import os
from pathlib import Path
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================================
# LOAD MODEL
# =========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "models" / "image_detector.h5"

print("Loading image detection model...")

model = load_model(MODEL_PATH)

print("Image model loaded successfully!")

# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_image(img_path):

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {img_path.resolve()}"
        )

    img = image.load_img(
        str(img_path),
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)[0][0]

    ai_probability = round(float(prediction) * 100, 2)

    human_probability = round(
        100 - ai_probability,
        2
    )

    confidence = "Low"

    max_score = max(
        ai_probability,
        human_probability
    )

    if max_score > 80:
        confidence = "High"

    elif max_score > 50:
        confidence = "Medium"

    result = "AI Generated"

    if human_probability > ai_probability:
        result = "Human Created"

    return {
        "content_type": "image",
        "prediction": result,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "confidence": confidence
    }

# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    test_image = BASE_DIR / "dataset" / "ai" / "test.jpg"

    if not test_image.exists():
        available_images = (
            list(BASE_DIR.glob("dataset/**/*.jpg")) +
            list(BASE_DIR.glob("dataset/**/*.jpeg"))
        )
        if not available_images:
            raise FileNotFoundError(
                "No test image found in ml/image_detection/dataset. "
                "Please add an image to dataset/ai or dataset/human."
            )
        test_image = available_images[0]
        print(f"Using available image for test: {test_image}")

    result = predict_image(test_image)

    print(result)