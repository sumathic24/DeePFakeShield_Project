from fastapi import APIRouter
from services.history import history

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Return recorded activity; no sample statistics are synthesized."""
    return history.stats()


@router.get("/stats/recent")
async def get_recent_analyses(limit: int = 5):
    return history.list(limit=max(1, min(limit, 20)))
