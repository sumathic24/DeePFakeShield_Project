from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.history import history
from services.text_service import TextAnalysisService

router = APIRouter()
service = TextAnalysisService()


class TextRequest(BaseModel):
    text: str


@router.post("/analyze")
async def analyze_text(req: TextRequest):
    """Run AI-authorship and optional misinformation analysis."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = await service.analyze_ai_text(req.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    history.record("text", req.text, result)
    return result


@router.post("/ai-authorship")
async def analyze_ai_authorship(req: TextRequest):
    """Detect AI authorship without running the misinformation classifier."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        result = await service.analyze_ai_text(req.text, include_fake_news=False)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    history.record("text", req.text, result)
    return result


@router.post("/fake-news")
async def analyze_fake_news(req: TextRequest):
    """Detect misinformation / fake news using BERT-based classifier."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text must be at least 20 characters.")
    try:
        return await service.analyze_fake_news(req.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
