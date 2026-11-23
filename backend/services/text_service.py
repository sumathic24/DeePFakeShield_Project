"""
Text Analysis Service
Runs Desklib's AI text detector for AI-generated text detection and keeps the
existing fake-news classifier path separate.
"""
import asyncio
import pickle
import time
import os
import re
import threading
from typing import List, Dict, Any

import numpy as np
import shap

# ── Try importing ML libraries (graceful fallback if not installed) ────────────
try:
    import torch
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
DESKLIB_MODEL_ID = os.getenv("DESKLIB_MODEL_ID", "desklib/ai-text-detector-v1.01")
ROBERTA_PATH = os.path.join(MODEL_DIR, "roberta_ai_detector")
LINGUA_PATH = os.path.join(MODEL_DIR, "linguaforge_detector")
FAKENEWS_PATH = os.path.join(MODEL_DIR, "fakenews_classifier")
HC3_PATH = os.path.join(MODEL_DIR, "hc3_detector.pkl")
MIN_PREDICTION_CONFIDENCE = float(os.getenv("TEXT_MIN_CONFIDENCE", "0.65"))
SOCIAL_GREETING_PATTERN = re.compile(
    r"(?:good (?:morning|afternoon|evening)|hello|hi|hey)"
    r"(?: (?:there|i am [\w'-]+|i'm [\w'-]+))?"
    r"(?: for this too)?"
)
if not 0.5 <= MIN_PREDICTION_CONFIDENCE <= 1.0:
    raise ValueError("TEXT_MIN_CONFIDENCE must be between 0.5 and 1.")


if TORCH_AVAILABLE:
    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return {"logits": self.classifier(pooled)}


class TextAnalysisService:
    def __init__(self):
        self.text_detector = None  # pre-trained AI-text detector (models/pretrained/)
        self.hc3_model = None  # scikit-learn detector trained on HC3 (models/saved/)
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.lingua_model = None
        self.lingua_tokenizer = None
        self.fakenews_model = None
        self.fakenews_tokenizer = None
        self.fakenews_fake_index = 1
        self.fakenews_model_id = None
        self._shap_explainer = None
        self._shap_lock = threading.Lock()
        self._load_models()

    def _load_models(self):
        """Load the pre-trained AI-text detector and the optional local models."""
        if not TORCH_AVAILABLE:
            print("[TextService] PyTorch/Transformers unavailable - AI detection is unavailable.")
            return

        # Pre-trained detector downloaded by `python download_models.py --text`
        from services.pretrained import (
            MISINFO_MODEL_ID, TextDetector, TEXT_MODEL_ID, load_detector, model_path,
        )
        self.text_detector = load_detector(TextDetector, TEXT_MODEL_ID, "TextService")

        if os.path.exists(HC3_PATH):
            try:
                with open(HC3_PATH, "rb") as f:
                    self.hc3_model = pickle.load(f)
                print("[TextService] HC3 detector loaded (used as the primary text detector).")
            except Exception as e:
                print(f"[TextService] Could not load HC3 detector: {e}")

        try:
            if os.path.exists(ROBERTA_PATH):
                self.roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
                self.roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH)
                self.roberta_model.eval()
                print("[TextService] RoBERTa model loaded.")

            if os.path.exists(LINGUA_PATH):
                self.lingua_tokenizer = AutoTokenizer.from_pretrained(LINGUA_PATH)
                self.lingua_model = AutoModelForSequenceClassification.from_pretrained(LINGUA_PATH)
                self.lingua_model.eval()
                print("[TextService] LinguaForge-X model loaded.")

            pretrained_fake_news_path = (
                model_path(MISINFO_MODEL_ID) if MISINFO_MODEL_ID else None
            )
            if os.path.isfile(os.path.join(FAKENEWS_PATH, "config.json")):
                fake_news_path = FAKENEWS_PATH
                self.fakenews_model_id = "local fine-tuned classifier"
            elif pretrained_fake_news_path and os.path.isfile(
                os.path.join(pretrained_fake_news_path, "config.json")
            ):
                fake_news_path = pretrained_fake_news_path
                self.fakenews_model_id = MISINFO_MODEL_ID
            else:
                fake_news_path = None

            if fake_news_path:
                fake_news_config = AutoConfig.from_pretrained(
                    fake_news_path, local_files_only=True
                )
                self.fakenews_fake_index = self._fake_news_label_index(fake_news_config)
                self.fakenews_tokenizer = AutoTokenizer.from_pretrained(
                    fake_news_path, local_files_only=True
                )
                self.fakenews_model = AutoModelForSequenceClassification.from_pretrained(
                    fake_news_path, local_files_only=True
                )
                self.fakenews_model.eval()
                print(f"[TextService] Misinformation classifier loaded: {self.fakenews_model_id}")
            elif MISINFO_MODEL_ID:
                print(
                    "[TextService] Misinformation classifier is not downloaded. "
                    "Run `python download_models.py --misinformation` in the backend folder."
                )
        except Exception as e:
            print(f"[TextService] Local model load error: {e}")

    @staticmethod
    def _fake_news_label_index(config) -> int:
        for index, label in (config.id2label or {}).items():
            normalized = str(label).strip().lower()
            if any(term in normalized for term in ("fake", "false", "misinformation")):
                return int(index)
        return 1

    def _prepare_fake_news_text(self, text: str) -> str:
        if getattr(self, "fakenews_model_id", None) == "hamzab/roberta-fake-news-classification":
            return f"<title> <content> {text} <end>"
        return text

    @staticmethod
    def _is_social_greeting(text: str) -> bool:
        normalized = re.sub(r"[,!?.]+", " ", text.casefold())
        normalized = " ".join(normalized.split())
        return bool(SOCIAL_GREETING_PATTERN.fullmatch(normalized))

    @staticmethod
    def _insufficient_fake_news_context() -> Dict[str, Any]:
        return {
            "prediction": "Not a news claim",
            "status": "not_applicable",
            "confidence": None,
            "fake_score": None,
            "real_score": None,
            "reason": "The misinformation classifier assesses news claims, not greetings or personal introductions.",
        }

    def _infer_model(self, model, tokenizer, text: str, max_length: int = 512,
                     positive_index: int = 1) -> float:
        """Run a two-class classifier and return the selected class probability."""
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0][positive_index])

    def _infer_desklib(self, text: str) -> float:
        """Run Desklib's single-logit detector and return AI probability."""
        inputs = self.desklib_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self.desklib_model(**inputs)["logits"]
        return float(torch.sigmoid(logits).flatten()[0])

    def _predict_ai_probability(self, texts) -> np.ndarray:
        """Use the same configured AI detector as the primary prediction path."""
        values = [str(text) for text in texts]
        if self.hc3_model is not None:
            return np.asarray(self.hc3_model.predict_proba(values)[:, 1], dtype=np.float64)
        if self.text_detector is not None:
            return np.asarray(self.text_detector.predict(values), dtype=np.float64)
        raise RuntimeError("The configured AI text detector is unavailable.")

    @staticmethod
    def _classify_fake_news(fake_score: float) -> Dict[str, Any]:
        real_score = 1.0 - fake_score
        confidence = max(fake_score, real_score)
        uncertain = confidence < MIN_PREDICTION_CONFIDENCE
        return {
            "prediction": "Uncertain" if uncertain else "Fake News" if fake_score > 0.5 else "Credible",
            "status": "uncertain" if uncertain else "classified",
            "confidence": round(confidence, 4),
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
        }

    def _explain_ai_text(self, text: str) -> Dict[str, Any]:
        """Compute model-agnostic SHAP attributions over the active detector."""
        with self._shap_lock:
            if self._shap_explainer is None:
                tokenizer = self.text_detector.tokenizer if self.text_detector is not None else None
                masker = shap.maskers.Text(tokenizer)
                self._shap_explainer = shap.Explainer(
                    self._predict_ai_probability,
                    masker,
                    algorithm="partition",
                )
            explainer = self._shap_explainer
            token_segments = explainer.masker.token_segments(text)[0]
            feature_count = sum(bool(segment) for segment in token_segments)
            max_evals = min(max(2 * feature_count + 1, 16), 128)
            values = explainer([text], max_evals=max_evals)

        contributions = np.asarray(values.values[0], dtype=np.float64).reshape(-1)
        segments = list(values.data[0])
        max_abs = float(np.max(np.abs(contributions))) if contributions.size else 0.0
        highlights = self._align_shap_segments(text, segments, contributions)
        token_importance = [
            {
                "token": item["text"],
                "score": round(abs(item["shap"]) / max_abs, 4) if max_abs else 0.0,
                "shap": round(item["shap"], 6),
                "label": "ai" if item["shap"] > 0 else "human" if item["shap"] < 0 else "neutral",
            }
            for item in highlights
            if item["text"].strip()
        ]
        ranked = sorted(token_importance, key=lambda item: abs(item["shap"]), reverse=True)
        shap_values = [
            {"feature": item["token"].strip(), "shap": item["shap"]}
            for item in ranked[:12]
        ]
        ai_features = [item["feature"] for item in shap_values if item["shap"] > 0]
        human_features = [item["feature"] for item in shap_values if item["shap"] < 0]
        summary_parts = []
        if ai_features:
            summary_parts.append(f"{', '.join(ai_features[:3])} increased the AI score")
        if human_features:
            summary_parts.append(f"{', '.join(human_features[:3])} decreased it")
        summary = "; ".join(summary_parts) if summary_parts else "No individual token had a strong effect on the AI score."

        return {
            "token_importance": token_importance,
            "shap_values": shap_values,
            "highlighted_text": highlights,
            "explanation": {
                "method": "SHAP",
                "available": True,
                "base_value": round(float(np.asarray(values.base_values).reshape(-1)[0]), 6),
                "explained_output": "AI-generated probability",
                "summary": summary + ".",
                "notice": "Attributions explain this detector's score; they are not proof of authorship.",
            },
        }

    @staticmethod
    def _align_shap_segments(text: str, segments, contributions) -> List[Dict[str, Any]]:
        """Keep the original input intact while attaching values to SHAP token spans."""
        highlights = []
        cursor = 0
        for segment, contribution in zip(segments, contributions):
            segment = str(segment)
            if not segment:
                continue
            start = text.find(segment, cursor)
            end = start + len(segment) if start >= 0 else -1
            if start < 0:
                token = segment.strip()
                start = text.find(token, cursor) if token else -1
                if start < 0:
                    continue
                end = start + len(token)
            if start > cursor:
                highlights.append({"text": text[cursor:start], "shap": 0.0})
            highlights.append({"text": text[start:end], "shap": float(contribution)})
            cursor = end
        if cursor < len(text):
            highlights.append({"text": text[cursor:], "shap": 0.0})
        return highlights

    async def analyze_ai_text(self, text: str, include_fake_news: bool = True) -> Dict[str, Any]:
        """Analyze AI authorship, optionally adding the separate misinformation score."""
        t0 = time.time()

        # The pre-trained detector is the source of truth for AI-generated text.
        if self.hc3_model is None and self.text_detector is None:
            raise RuntimeError(
                "No AI text detector is available. Train the HC3 detector (training/train_hc3_tfidf.py) "
                "or run `python download_models.py --text`, then restart the backend."
            )
        loop = asyncio.get_running_loop()
        desklib_score = float((await loop.run_in_executor(
            None, self._predict_ai_probability, [text]
        ))[0])
        xai_result = await loop.run_in_executor(None, self._explain_ai_text, text)

        ensemble_score = desklib_score
        human_score = 1.0 - ensemble_score

        # Fake news score
        if include_fake_news and self.fakenews_model and not self._is_social_greeting(text):
            loop = asyncio.get_event_loop()
            fake_score = await loop.run_in_executor(
                None, self._infer_model, self.fakenews_model, self.fakenews_tokenizer,
                self._prepare_fake_news_text(text), 512,
                getattr(self, "fakenews_fake_index", 1))
            fake_news = self._classify_fake_news(fake_score)
        elif include_fake_news and self.fakenews_model:
            fake_news = self._insufficient_fake_news_context()
        else:
            fake_news = None
        confidence = max(ensemble_score, human_score)
        uncertain = confidence < MIN_PREDICTION_CONFIDENCE

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "prediction": "Uncertain" if uncertain else "AI Generated" if ensemble_score > 0.5 else "Human Written",
            "status": "uncertain" if uncertain else "classified",
            "confidence": round(confidence, 4),
            "model_used": "HC3 classifier" if self.hc3_model is not None else self.text_detector.model_id,
            "ai_confidence": round(ensemble_score, 4),
            "human_confidence": round(human_score, 4),
            "fake_news_prediction": fake_news["prediction"] if fake_news else None,
            "fake_news_status": fake_news["status"] if fake_news else None,
            "fake_news_confidence": fake_news["confidence"] if fake_news else None,
            "fake_news_score": fake_news["fake_score"] if fake_news else None,
            "real_news_score": fake_news["real_score"] if fake_news else None,
            "fake_news_message": fake_news.get("reason") if fake_news else None,
            "desklib": {
                "prediction": "Uncertain" if uncertain else "AI Generated" if desklib_score > 0.5 else "Human Written",
                "ai_confidence": round(desklib_score, 4),
            },
            "ensemble_weight": {"desklib": 1.0},
            "token_importance": xai_result["token_importance"],
            "shap_values": xai_result["shap_values"],
            "highlighted_text": xai_result["highlighted_text"],
            "explanation": xai_result["explanation"],
            "suspicious_phrases": [],
            "processing_time_ms": elapsed_ms,
        }

    async def analyze_fake_news(self, text: str) -> Dict[str, Any]:
        """Dedicated fake news / misinformation detection."""
        t0 = time.time()

        if self.fakenews_model:
            if self._is_social_greeting(text):
                classification = self._insufficient_fake_news_context()
            else:
                loop = asyncio.get_event_loop()
                fake_score = await loop.run_in_executor(
                    None, self._infer_model, self.fakenews_model, self.fakenews_tokenizer,
                    self._prepare_fake_news_text(text), 512,
                    getattr(self, "fakenews_fake_index", 1))
                classification = self._classify_fake_news(fake_score)
        else:
            raise RuntimeError(
                "No fake-news classifier is available. Train it with "
                "`training/train_fakenews.py` and configure the resulting model."
            )

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            **classification,
            "credibility_rating": (
                "Not enough context" if classification["status"] == "not_applicable"
                else "Uncertain" if classification["status"] == "uncertain"
                else "Low" if classification["fake_score"] > 0.7
                else "Medium" if classification["fake_score"] > 0.4
                else "High"
            ),
            "shap_values": [],
            "token_importance": [],
            "processing_time_ms": elapsed_ms,
        }