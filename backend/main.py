from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routers import text, image, video, stats

app = FastAPI(
    title="DeepFakeShield API",
    description="Multimodal AI detection platform for deepfakes and misinformation",
    version="1.0.0",
)

# Allow the frontend from any origin (localhost or your network IP).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

app.include_router(text.router, prefix="/api/text", tags=["Text Analysis"])
app.include_router(image.router, prefix="/api/image", tags=["Image Analysis"])
app.include_router(video.router, prefix="/api/video", tags=["Video Analysis"])
app.include_router(stats.router, prefix="/api", tags=["Stats"])


@app.get("/")
async def root():
    return {
        "name": "DeepFakeShield API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}