import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from services.history import AnalysisHistory
from services.image_service import ImageAnalysisService
from services.text_service import TextAnalysisService
from services.video_service import SampledVideoFrame, VideoAnalysisService


class PredictionReliabilityTests(unittest.TestCase):
    def test_image_service_abstains_near_decision_boundary(self):
        for score, expected_status, expected_prediction in (
            (0.50, "uncertain", "Uncertain"),
            (0.90, "classified", "AI Generated"),
            (0.10, "classified", "Real Image"),
        ):
            with self.subTest(score=score):
                service = ImageAnalysisService.__new__(ImageAnalysisService)
                service.detector = SimpleNamespace(
                    explain=lambda _: {
                        "ai_confidence": score,
                        "class_label": "AI" if score > 0.5 else "Real",
                        "image": Image.new("RGB", (16, 16), "white"),
                        "model_input_image": Image.new("RGB", (16, 16), "white"),
                        "gradcam": np.zeros((16, 16), dtype=np.float32),
                        "gradcam_layer": "test.layer",
                    },
                    model_id="test-detector",
                )
                result = asyncio.run(service.analyze_image("unused"))
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["prediction"], expected_prediction)
                self.assertAlmostEqual(result["confidence"], max(score, 1 - score), places=4)

    def test_video_service_returns_top_class_near_decision_boundary(self):
        for ai_score, expected_prediction in (
            (0.635, "AI-generated video"),
            (0.365, "Real video"),
        ):
            with self.subTest(ai_score=ai_score):
                service = VideoAnalysisService.__new__(VideoAnalysisService)
                frame = SampledVideoFrame(Image.new("RGB", (16, 16), "white"), 8, 0.5, 2.0)
                service.detector = SimpleNamespace(
                    explain_frames=lambda _: {
                        "ai_confidences": [ai_score],
                        "real_confidences": [1.0 - ai_score],
                        "gradcams": [np.zeros((16, 16), dtype=np.float32)],
                    },
                    model_id="test-video-detector",
                )
                service._sample_frames = lambda _: [frame]
                result = asyncio.run(service.analyze_video("unused"))
                self.assertEqual(result["status"], "classified")
                self.assertEqual(result["prediction"], expected_prediction)
                self.assertTrue(result["low_confidence"])
                self.assertAlmostEqual(result["confidence"], 0.635)
                self.assertEqual(result["frames"][0]["timestamp_seconds"], 0.5)
                self.assertEqual(result["frame_count"], 1)

    def test_text_shap_alignment_preserves_the_original_input(self):
        text = "Human-written text, with punctuation!"
        spans = TextAnalysisService._align_shap_segments(
            text,
            ["Human-written ", "text", ", ", "with ", "punctuation"],
            [0.2, -0.1, 0.0, -0.2, 0.3],
        )
        self.assertEqual("".join(span["text"] for span in spans), text)
        self.assertEqual(spans[0]["shap"], 0.2)
        self.assertEqual(spans[-2]["shap"], 0.3)

    def test_fake_news_classifier_marks_boundary_scores_uncertain(self):
        uncertain = TextAnalysisService._classify_fake_news(0.52)
        fake = TextAnalysisService._classify_fake_news(0.9)
        credible = TextAnalysisService._classify_fake_news(0.1)

        self.assertEqual((uncertain["prediction"], uncertain["status"]), ("Uncertain", "uncertain"))
        self.assertEqual((fake["prediction"], fake["status"]), ("Fake News", "classified"))
        self.assertEqual((credible["prediction"], credible["status"]), ("Credible", "classified"))
        self.assertAlmostEqual(fake["fake_score"] + fake["real_score"], 1.0)

    def test_fake_news_label_index_uses_named_fake_class(self):
        config = SimpleNamespace(id2label={0: "FAKE", 1: "TRUE"})
        self.assertEqual(TextAnalysisService._fake_news_label_index(config), 0)
        self.assertEqual(
            TextAnalysisService._fake_news_label_index(SimpleNamespace(id2label={0: "LABEL_0", 1: "LABEL_1"})),
            1,
        )

    def test_pretrained_fake_news_model_receives_its_expected_input_format(self):
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.fakenews_model_id = "hamzab/roberta-fake-news-classification"
        self.assertEqual(
            service._prepare_fake_news_text("Example claim."),
            "<title> <content> Example claim. <end>",
        )

    def test_social_greeting_is_not_classified_as_misinformation(self):
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.fakenews_model = object()
        service.fakenews_tokenizer = object()
        service._infer_model = lambda *_: self.fail("A greeting should not reach the classifier.")

        result = asyncio.run(service.analyze_fake_news(
            "GOOD AFTERNOON I AM SUMATHI FOR THIS TOO"
        ))

        self.assertEqual(result["prediction"], "Not a news claim")
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["fake_score"])

    def test_short_news_claim_is_sent_to_the_classifier(self):
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.fakenews_model = object()
        service.fakenews_tokenizer = object()
        service._infer_model = lambda *_: 0.9

        result = asyncio.run(service.analyze_fake_news("India is not a country."))

        self.assertEqual(result["prediction"], "Fake News")
        self.assertEqual(result["fake_score"], 0.9)

    def test_ai_authorship_only_mode_skips_misinformation_inference(self):
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.hc3_model = None
        service.text_detector = SimpleNamespace(model_id="test-text-detector")
        service.fakenews_model = object()
        service.fakenews_tokenizer = object()
        service._predict_ai_probability = lambda _: np.asarray([0.2])
        service._infer_model = lambda *_: self.fail("Authorship-only mode must skip misinformation.")
        service._explain_ai_text = lambda _: {
            "token_importance": [],
            "shap_values": [],
            "highlighted_text": [],
            "explanation": {},
        }

        result = asyncio.run(service.analyze_ai_text(
            "A sufficiently long sample that is only checked for AI authorship.",
            include_fake_news=False,
        ))

        self.assertEqual(result["prediction"], "Human Written")
        self.assertIsNone(result["fake_news_prediction"])

    def test_text_analysis_includes_fake_news_assessment_when_model_is_loaded(self):
        service = TextAnalysisService.__new__(TextAnalysisService)
        service.hc3_model = None
        service.text_detector = SimpleNamespace(model_id="test-text-detector")
        service.fakenews_model = object()
        service.fakenews_tokenizer = object()
        service._predict_ai_probability = lambda _: np.asarray([0.2])
        service._infer_model = lambda *_: 0.9
        service._explain_ai_text = lambda _: {
            "token_importance": [],
            "shap_values": [],
            "highlighted_text": [],
            "explanation": {},
        }

        sample = (
            "A sufficiently long test claim for the classifier with enough additional words "
            "to meet the minimum context requirement for a news assessment."
        )
        result = asyncio.run(service.analyze_ai_text(sample))

        self.assertEqual(result["fake_news_prediction"], "Fake News")
        self.assertEqual(result["fake_news_status"], "classified")
        self.assertEqual(result["fake_news_score"], 0.9)
        self.assertEqual(result["real_news_score"], 0.1)

        service.fakenews_model = None
        result_without_classifier = asyncio.run(
            service.analyze_ai_text(sample)
        )
        self.assertIsNone(result_without_classifier["fake_news_prediction"])
        self.assertIsNone(result_without_classifier["fake_news_score"])

    def test_uncertain_results_are_not_counted_as_classified_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            history = AnalysisHistory(str(Path(directory) / "history.db"))
            history.record("image", "uncertain.png", {
                "prediction": "Uncertain", "status": "uncertain",
                "ai_confidence": 0.52, "confidence": 0.52,
            })
            history.record("image", "ai.png", {
                "prediction": "AI Generated", "status": "classified",
                "ai_confidence": 0.9, "confidence": 0.9,
            })
            stats = history.stats()
            self.assertEqual(stats["total_analyses"], 1)
            self.assertEqual(stats["uncertain_analyses"], 1)
            self.assertEqual(stats["breakdown"]["ai_image"], 1)
            self.assertEqual(history.list()["items"][0]["uncertain"], False)
            history._conn.close()


if __name__ == "__main__":
    unittest.main()
