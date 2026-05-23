from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.detect import router

app = FastAPI()

# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

@app.post("/health_post")
async def health_post(payload: dict):
    return {"ok": True, "received": payload}

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