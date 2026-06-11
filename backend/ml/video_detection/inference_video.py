import os
import cv2
import tempfile

from ml.image_detection.inference_image import predict_image

# =========================================
# VIDEO PREDICTION
# =========================================

def predict_video(video_path):

    frame_scores = []

    try:

        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Analyze every 15th frame
            if frame_count % 15 == 0:

                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False
                )

                frame_path = temp_file.name

                temp_file.close()

                cv2.imwrite(frame_path, frame)

                try:

                    result = predict_image(frame_path)

                    frame_scores.append(
                        result["ai_probability"]
                    )

                finally:

                    if os.path.exists(frame_path):
                        os.remove(frame_path)

        cap.release()

    except Exception as e:

        print(f"Video detection error: {e}")

        return {
            "content_type": "video",
            "prediction": "Unable to Analyze",
            "ai_probability": 0,
            "human_probability": 0,
            "confidence": "Low"
        }

    # =========================================
    # NO FRAMES FOUND
    # =========================================

    if len(frame_scores) == 0:

        return {
            "content_type": "video",
            "prediction": "Unable to Analyze",
            "ai_probability": 0,
            "human_probability": 0,
            "confidence": "Low"
        }

    # =========================================
    # FINAL SCORE
    # =========================================

    avg_ai_score = round(
        sum(frame_scores) / len(frame_scores),
        2
    )

    human_score = round(
        100 - avg_ai_score,
        2
    )

    prediction = (
        "AI Generated Video"
        if avg_ai_score > human_score
        else "Human Video"
    )

    max_score = max(
        avg_ai_score,
        human_score
    )

    if max_score >= 80:
        confidence = "High"
    elif max_score >= 60:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "content_type": "video",
        "prediction": prediction,
        "ai_probability": avg_ai_score,
        "human_probability": human_score,
        "confidence": confidence
    }

# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    result = predict_video(
        "sample.mp4"
    )

    print(result)