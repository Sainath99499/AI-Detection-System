from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.detect import router

import traceback

app = FastAPI()

# =========================================
# DEBUG MIDDLEWARE
# =========================================

@app.middleware("http")
async def log_requests(request: Request, call_next):

    try:
        response = await call_next(request)

        print(f"{request.method} {request.url.path} -> {response.status_code}")

        return response

    except Exception as e:

        print("\n========== BACKEND ERROR ==========")

        traceback.print_exc()

        print("===================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

# =========================================
# CORS
# =========================================

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://ai-detection-system-jbgv.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# ROUTES
# =========================================

app.include_router(
    router,
    prefix="/detect",
    tags=["Detection"]
)

# =========================================
# HEALTH ROUTES
# =========================================

@app.get("/")
def root():

    return {
        "message": "AI Detection API Running"
    }

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

@app.post("/health_post")
async def health_post(payload: dict):

    return {
        "ok": True,
        "received": payload
    }