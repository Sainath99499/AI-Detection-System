import cv2
import random

# =========================================
# VIDEO PREDICTION
# =========================================

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_scores = []

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # =========================================
        # ANALYZE EVERY 30TH FRAME
        # =========================================

        if frame_count % 30 == 0:

            # MVP SIMULATION SCORE

            ai_score = random.uniform(
                40,
                95
            )

            frame_scores.append(ai_score)

    cap.release()

    # =========================================
    # NO FRAMES
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
    # FINAL AGGREGATED SCORE
    # =========================================

    avg_ai_score = round(
        sum(frame_scores) / len(frame_scores),
        2
    )

    human_score = round(
        100 - avg_ai_score,
        2
    )

    prediction = "Human Video"

    if avg_ai_score > human_score:

        prediction = "AI Generated Video"

    confidence = "Low"

    max_score = max(
        avg_ai_score,
        human_score
    )

    if max_score > 80:

        confidence = "High"

    elif max_score > 50:

        confidence = "Medium"

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