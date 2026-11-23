"""
Image analysis service.

Uses the pre-trained AI-image detector set by IMAGE_DETECTOR_MODEL
(default haywoodsloan/ai-image-detector-deploy), downloaded by download_models.py.

This is one model, so the EfficientNet and ViT cards on the page show the same
score. Grad-CAM is computed from its active classifier and does not localize edits.
"""
import asyncio
import os
import time
from typing import Any, Dict

from services.pretrained import IMAGE_MODEL_ID, ImageDetector, load_detector
from services.explainability import render_gradcam

AI_THRESHOLD = float(os.getenv("IMAGE_AI_THRESHOLD", "0.5"))
MIN_PREDICTION_CONFIDENCE = float(os.getenv("IMAGE_MIN_CONFIDENCE", "0.65"))
if not 0.0 <= AI_THRESHOLD <= 1.0:
    raise ValueError("IMAGE_AI_THRESHOLD must be between 0 and 1.")
if not 0.5 <= MIN_PREDICTION_CONFIDENCE <= 1.0:
    raise ValueError("IMAGE_MIN_CONFIDENCE must be between 0.5 and 1.")


class ImageAnalysisService:
    def __init__(self):
        # Returns None (with a printed hint) if the model isn't downloaded yet.
        self.detector = load_detector(ImageDetector, IMAGE_MODEL_ID, "ImageService")

    async def analyze_image(self, filepath: str) -> Dict[str, Any]:
        if self.detector is None:
            raise RuntimeError(
                "The image detector is not downloaded. Run `python download_models.py --image` "
                "in the backend folder, then restart the backend."
            )
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(None, self.detector.explain, filepath)
        ai_score = round(float(analysis["ai_confidence"]), 4)
        real_score = round(1.0 - ai_score, 4)
        confidence = max(ai_score, real_score)
        is_ai = ai_score > AI_THRESHOLD
        uncertain = confidence < MIN_PREDICTION_CONFIDENCE
        prediction = "Uncertain" if uncertain else "AI Generated" if is_ai else "Real Image"
        gradcam = render_gradcam(analysis["model_input_image"], analysis["gradcam"], include_original=True)

        detector_block = {
            "prediction": "Uncertain" if uncertain else "AI Generated" if is_ai else "Real",
            "confidence": confidence,
        }
        return {
            "prediction": prediction,
            "status": "uncertain" if uncertain else "classified",
            "confidence": confidence,
            "ai_confidence": ai_score,
            "real_confidence": real_score,
            "model_used": self.detector.model_id,
            "detector": detector_block,
            "gradcam_regions": [gradcam["region_summary"]],
            "explanation": {
                "method": "Grad-CAM",
                "available": True,
                "target_class": analysis["class_label"],
                "target_layer": analysis["gradcam_layer"],
                "region_summary": gradcam["region_summary"],
                "notice": "Highlighted regions show model activation, not a localized manipulation.",
            },
            "gradcam": {
                "heatmap": gradcam["heatmap"],
                "overlay": gradcam["overlay"],
                "model_input": gradcam["original"],
            },
            "artifacts_detected": [],
            "processing_time_ms": round((time.perf_counter() - t0) * 1000),
        }