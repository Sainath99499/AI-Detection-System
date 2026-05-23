from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.detect import router

import traceback
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_detection.main")

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info("incoming request", extra={"method": request.method, "path": request.url.path})
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("request handling failed")
        raise
    process_time = (time.time() - start) * 1000
    logger.info("request complete", extra={"method": request.method, "path": request.url.path, "status_code": response.status_code, "duration_ms": int(process_time)})
    return response


# =========================================
# CORS (Added last to be outermost)
# =========================================

origins = [
    "https://ai-detection-system-jbgv.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# GLOBAL EXCEPTION HANDLER
# =========================================


# Global exception handler to ensure JSON responses (keeps CORS headers)
@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    tb = traceback.format_exc()
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": tb})

@app.post("/health_post")
async def health_post(payload: dict):
    return {"ok": True, "received": payload}

@app.get("/health_post")
async def health_post_get():
    return {"ok": True, "message": "POST health endpoint available"}

# =========================================
# ROUTES
# =========================================

app.include_router(
    router,
    prefix="/detect",
    tags=["Detection"]
)

# =========================================
# ROOT
# =========================================

@app.get("/")
def root():

    return {
        "message": "AI Detection API Running"
    }