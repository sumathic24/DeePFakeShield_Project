import io
import os
import uuid
import warnings

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from services.history import history
from services.image_service import ImageAnalysisService

router = APIRouter()
service = ImageAnalysisService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 40_000_000


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Detect whether an image is AI-generated using the downloaded image detector."""
    extension = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_TYPES or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported image type.")

    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"The image is larger than {MAX_UPLOAD_MB} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="The image dimensions are too large.")
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from exc

    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{extension}")
    try:
        with open(filepath, "wb") as output:
            output.write(data)
        result = await service.analyze_image(filepath)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    history.record("image", file.filename or "uploaded image", result)
    return result