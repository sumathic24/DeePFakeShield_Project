import threading
import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from services.explainability import generate_gradcam
from services.text_service import TextAnalysisService


class ExplainabilityTests(unittest.TestCase):
    def test_gradcam_returns_a_spatial_map_from_model_gradients(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        ).eval()
        logits, maps = generate_gradcam(
            model,
            model[0],
            torch.rand(1, 3, 24, 24),
            model,
        )
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(maps.shape, (1, 24, 24))
        self.assertTrue(np.isfinite(maps).all())
        self.assertGreaterEqual(float(maps.min()), 0.0)
        self.assertLessEqual(float(maps.max()), 1.0)

    def test_text_shap_uses_the_active_sklearn_pipeline(self):
        model = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=7))
        model.fit(
            [
                "human personal story",
                "human casual writing",
                "synthetic generated output",
                "automated generated language",
            ],
            [0, 0, 1, 1],
        )
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.hc3_model = model
        service.text_detector = None
        service._shap_explainer = None
        service._shap_lock = threading.Lock()

        text = "A human personal story with generated language."
        result = service._explain_ai_text(text)

        self.assertEqual("".join(item["text"] for item in result["highlighted_text"]), text)
        self.assertEqual(result["explanation"]["method"], "SHAP")
        self.assertTrue(result["explanation"]["available"])
        self.assertTrue(result["shap_values"])


if __name__ == "__main__":
    unittest.main()
