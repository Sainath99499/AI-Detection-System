from fastapi import (
    APIRouter,
    UploadFile,
    File
)
from fastapi.responses import JSONResponse

from pydantic import BaseModel

import shutil
import os
import traceback
import uuid
import tempfile
import logging
from typing import Optional, Set

from ml.explanation_engine.explanation import (
    generate_explanation
)

logger = logging.getLogger("ai_detection.routes.detect")

# Upload limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_IMAGE_TYPES: Set[str] = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_AUDIO_TYPES: Set[str] = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/x-m4a", "audio/mp4"}
ALLOWED_VIDEO_TYPES: Set[str] = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}


async def save_upload_file(upload_file: UploadFile, dest_path: str, max_size: Optional[int] = None, allowed_types: Optional[Set[str]] = None) -> int:
    """Save an UploadFile to dest_path while enforcing size and content-type limits.

    Returns number of bytes written.
    Raises ValueError for validation errors.
    """
    content_type = upload_file.content_type
    if allowed_types and content_type not in allowed_types:
        raise ValueError(f"Disallowed content type: {content_type}")

    total = 0
    chunk_size = 1024 * 1024
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_size and total > max_size:
                    raise ValueError(f"File exceeds maximum allowed size of {max_size} bytes")
                out.write(chunk)
    finally:
        try:
            await upload_file.close()
        except Exception:
            pass

    return total

# =========================================
# ROUTER
# =========================================

router = APIRouter()


@router.post("/_debug_echo")
async def debug_echo(payload: dict):
    return {"ok": True, "received": payload}

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
    upload_dir = os.path.join(tempfile.gettempdir(), "ai_detection_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # sanitize filename and ensure a fallback
    filename = os.path.basename(file.filename) if file.filename else f"upload-{uuid.uuid4().hex}"
    file_path = os.path.join(upload_dir, filename)

    # SAVE FILE with validation
    try:
        bytes_written = await save_upload_file(file, file_path, max_size=MAX_IMAGE_SIZE, allowed_types=ALLOWED_IMAGE_TYPES)
        logger.info("Saved image upload", extra={"filename": file_path, "bytes": bytes_written, "content_type": file.content_type})
    except ValueError as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=400, content={"error": str(e), "trace": tb})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

    try:
        from ml.image_detection.inference_image import predict_image

        # RUN IMAGE MODEL
        result = predict_image(file_path)

        # GENERATE EXPLANATION
        explanation = generate_explanation(result["prediction"], result["ai_probability"])

        # ADD EXPLANATION
        result["explanation"] = explanation

        return result
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

# =========================================
# AUDIO DETECTION
# =========================================

@router.post("/audio")
async def detect_audio(
    file: UploadFile = File(...)
):
    upload_dir = os.path.join(tempfile.gettempdir(), "ai_detection_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filename = os.path.basename(file.filename) if file.filename else f"upload-{uuid.uuid4().hex}"
    file_path = os.path.join(upload_dir, filename)

    # SAVE FILE with validation
    try:
        bytes_written = await save_upload_file(file, file_path, max_size=MAX_AUDIO_SIZE, allowed_types=ALLOWED_AUDIO_TYPES)
        logger.info("Saved audio upload", extra={"filename": file_path, "bytes": bytes_written, "content_type": file.content_type})
    except ValueError as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=400, content={"error": str(e), "trace": tb})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

    try:
        from ml.audio_detection.inference_audio import predict_audio
        result = predict_audio(file_path)
        explanation = generate_explanation(result["prediction"], result["ai_probability"])
        result["explanation"] = explanation
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Audio inference failed")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

# =========================================
# VIDEO DETECTION
# =========================================

@router.post("/video")
async def detect_video(
    file: UploadFile = File(...)
):
    upload_dir = os.path.join(tempfile.gettempdir(), "ai_detection_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filename = os.path.basename(file.filename) if file.filename else f"upload-{uuid.uuid4().hex}"
    file_path = os.path.join(upload_dir, filename)

    # SAVE FILE with validation
    try:
        bytes_written = await save_upload_file(file, file_path, max_size=MAX_VIDEO_SIZE, allowed_types=ALLOWED_VIDEO_TYPES)
        logger.info("Saved video upload", extra={"filename": file_path, "bytes": bytes_written, "content_type": file.content_type})
    except ValueError as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=400, content={"error": str(e), "trace": tb})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})

    try:
        from ml.video_detection.inference_video import predict_video
        result = predict_video(file_path)
        explanation = generate_explanation(result["prediction"], result["ai_probability"])
        result["explanation"] = explanation
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Video inference failed")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})