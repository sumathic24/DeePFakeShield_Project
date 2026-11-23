"""
Video analysis service.

Samples evenly spaced frames from the video and averages the AI-probability of the
pre-trained video detector (VIDEO_DETECTOR_MODEL, default ducbao210/video-deepfake-detection).
"""
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from services.pretrained import VIDEO_MODEL_ID, VideoDetector, load_detector
from services.explainability import render_gradcam

FRAME_COUNT = 16
MAX_DURATION_SECONDS = 300
AI_THRESHOLD = 0.5
MIN_PREDICTION_CONFIDENCE = float(os.getenv("VIDEO_MIN_CONFIDENCE", "0.65"))
if not 0.5 <= MIN_PREDICTION_CONFIDENCE <= 1.0:
    raise ValueError("VIDEO_MIN_CONFIDENCE must be between 0.5 and 1.")


@dataclass(frozen=True)
class SampledVideoFrame:
    image: Image.Image
    frame_index: int
    timestamp_seconds: float
    duration_seconds: float


class VideoAnalysisService:
    def __init__(self):
        # Returns None (with a printed hint) if the checkpoint isn't downloaded yet.
        self.detector = load_detector(VideoDetector, VIDEO_MODEL_ID, "VideoService")

    def _sample_frames(self, filepath: str):
        capture = cv2.VideoCapture(filepath)
        if not capture.isOpened():
            raise ValueError("The uploaded file is not a readable video.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            duration = frame_count / fps if fps > 0 else 0
            if duration > MAX_DURATION_SECONDS:
                raise ValueError(f"The video is longer than the {MAX_DURATION_SECONDS}-second limit.")

            indices = np.linspace(0, max(frame_count - 1, 0), FRAME_COUNT, dtype=int)
            frames = []
            for index in np.unique(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(SampledVideoFrame(
                        image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        frame_index=int(index),
                        timestamp_seconds=float(index / fps) if fps > 0 else 0.0,
                        duration_seconds=float(duration),
                    ))
            if not frames:
                raise ValueError("The uploaded file contains no readable frames.")
            return frames
        finally:
            capture.release()

    async def analyze_video(self, filepath: str) -> Dict[str, Any]:
        if self.detector is None:
            raise RuntimeError(
                "The video detector is not downloaded. Run `python download_models.py --video` "
                "in the backend folder, then restart the backend."
            )
        started = time.perf_counter()
        frames = self._sample_frames(filepath)

        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(
            None,
            self.detector.explain_frames,
            [sample.image for sample in frames],
        )
        frame_results = []
        for sample, ai_score, real_score, activation in zip(
            frames,
            analysis["ai_confidences"],
            analysis["real_confidences"],
            analysis["gradcams"],
        ):
            visual = render_gradcam(sample.image, activation, include_original=True)
            frame_results.append({
                "frame_index": sample.frame_index,
                "timestamp_seconds": round(sample.timestamp_seconds, 3),
                "ai_confidence": round(float(ai_score), 4),
                "real_confidence": round(float(real_score), 4),
                "prediction": "AI-generated" if ai_score > AI_THRESHOLD else "Real",
                "is_suspicious": bool(ai_score > AI_THRESHOLD),
                "original": visual["original"],
                "heatmap": visual["heatmap"],
                "overlay": visual["overlay"],
                "region_summary": visual["region_summary"],
            })

        ai_confidence = round(float(np.mean(analysis["ai_confidences"])), 4)
        real_confidence = round(1.0 - ai_confidence, 4)
        confidence = max(ai_confidence, real_confidence)
        is_ai = ai_confidence > AI_THRESHOLD
        low_confidence = confidence < MIN_PREDICTION_CONFIDENCE

        return {
            "prediction": "AI-generated video" if is_ai else "Real video",
            "status": "classified",
            "confidence": confidence,
            "low_confidence": low_confidence,
            "ai_confidence": ai_confidence,
            "real_confidence": real_confidence,
            "duration_seconds": round(frames[0].duration_seconds, 3),
            "model_used": f"{self.detector.model_id} (ConvNeXt frame sampler)",
            "frame_count": len(frame_results),
            "frames": frame_results,
            "suspicious_frames": [
                frame["timestamp_seconds"] for frame in frame_results if frame["is_suspicious"]
            ],
            "explanation": {
                "method": "Frame-wise Grad-CAM",
                "available": True,
                "notice": "Frame highlights show model activation, not a localized manipulation.",
            },
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
        }