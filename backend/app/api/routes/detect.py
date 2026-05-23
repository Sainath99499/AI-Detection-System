from fastapi import (
    APIRouter,
    UploadFile,
    File
)
from fastapi.responses import JSONResponse

from pydantic import BaseModel

import shutil
import os

from ml.explanation_engine.explanation import (
    generate_explanation
)

# =========================================
# ROUTER
# =========================================

router = APIRouter()

# =========================================
# TEXT REQUEST MODEL
# =========================================

class TextRequest(BaseModel):

    text: str

# =========================================
# TEXT DETECTION
# =========================================

@router.post("/text")
async def detect_text(data: TextRequest):

    from ml.text_detection.inference import predict_text

    # RUN TEXT MODEL with error capture for debugging
    try:
        result = predict_text(
            data.text
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

    # GENERATE EXPLANATION

    explanation = generate_explanation(
        result["prediction"],
        result["ai_probability"]
    )

    # ADD EXPLANATION

    result["explanation"] = explanation

    return result

# =========================================
# IMAGE DETECTION
# =========================================

@router.post("/image")
async def detect_image(
    file: UploadFile = File(...)
):

    upload_dir = "uploads"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    file_path = (
        f"{upload_dir}/{file.filename}"
    )

    # SAVE FILE

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    from ml.image_detection.inference_image import predict_image

    # RUN IMAGE MODEL

    result = predict_image(
        file_path
    )

    # GENERATE EXPLANATION

    explanation = generate_explanation(
        result["prediction"],
        result["ai_probability"]
    )

    # ADD EXPLANATION

    result["explanation"] = explanation

    return result

# =========================================
# AUDIO DETECTION
# =========================================

@router.post("/audio")
async def detect_audio(
    file: UploadFile = File(...)
):

    upload_dir = "uploads"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    file_path = (
        f"{upload_dir}/{file.filename}"
    )

    # SAVE FILE

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    from ml.audio_detection.inference_audio import predict_audio

    # RUN AUDIO MODEL

    result = predict_audio(
        file_path
    )

    # GENERATE EXPLANATION

    explanation = generate_explanation(
        result["prediction"],
        result["ai_probability"]
    )

    # ADD EXPLANATION

    result["explanation"] = explanation

    return result

# =========================================
# VIDEO DETECTION
# =========================================

@router.post("/video")
async def detect_video(
    file: UploadFile = File(...)
):

    upload_dir = "uploads"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    file_path = (
        f"{upload_dir}/{file.filename}"
    )

    # SAVE FILE

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    from ml.video_detection.inference_video import predict_video

    # RUN VIDEO MODEL

    result = predict_video(
        file_path
    )

    # GENERATE EXPLANATION

    explanation = generate_explanation(
        result["prediction"],
        result["ai_probability"]
    )

    # ADD EXPLANATION

    result["explanation"] = explanation

    return result