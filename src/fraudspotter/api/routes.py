from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from fraudspotter.api.schemas import (
    ExplanationResponse,
    HealthResponse,
    NewsTextRequest,
    PredictionResponse,
)
from fraudspotter.data.preprocessor import clean_text

router = APIRouter()

# Global state dictionary for loaded models
_MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "model_type": "None",
}


def set_active_model(model: Any, model_type: str) -> None:
    """Set the active serving model instance in application state."""
    _MODEL_STATE["model"] = model
    _MODEL_STATE["model_type"] = model_type


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        active_model=_MODEL_STATE["model_type"],
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: NewsTextRequest) -> PredictionResponse:
    """Classify news article text as FAKE or REAL."""
    model = _MODEL_STATE.get("model")
    model_type = _MODEL_STATE.get("model_type", "None")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No classification model is currently loaded in server state.",
        )

    combined_text = clean_text(f"{request.title} {request.text}")

    if model_type == "onnx":
        res = model.predict_one(combined_text)
        return PredictionResponse(
            label=res["prediction_label"],
            prediction_class=res["prediction_class"],
            probability_fake=res["probability_fake"],
            probability_real=res["probability_real"],
            confidence=res["confidence"],
            model_used="DistilBERT-ONNX-INT8",
        )
    elif model_type in ["classical", "pipeline"]:
        probs = model.predict_proba([combined_text])[0]
        prob_fake = float(probs[0])
        prob_real = float(probs[1])
        pred_class = int(prob_real >= 0.5)
        label = "REAL" if pred_class == 1 else "FAKE"

        return PredictionResponse(
            label=label,
            prediction_class=pred_class,
            probability_fake=prob_fake,
            probability_real=prob_real,
            confidence=max(prob_fake, prob_real),
            model_used="TF-IDF-LogisticRegression",
        )
    else:
        raise HTTPException(status_code=500, detail="Unknown active model type.")


@router.post("/explain", response_model=ExplanationResponse)
def explain(request: NewsTextRequest) -> ExplanationResponse:
    """Classify news article and generate token attribution explanation."""
    pred_res = predict(request)
    combined_text = clean_text(f"{request.title} {request.text}")
    model = _MODEL_STATE.get("model")
    model_type = _MODEL_STATE.get("model_type")

    # Local fallback explanation via feature weights or LIME
    attributions = []
    method = "FeatureWeights"

    if model_type in ["classical", "pipeline"]:
        # Extract features from pipeline
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            coef = clf.coef_[0]

            tokens = combined_text.split()
            found_tokens = {}
            for tok in tokens:
                tok_clean = tok.lower()
                if tok_clean in tfidf.vocabulary_:
                    idx = tfidf.vocabulary_[tok_clean]
                    weight = float(coef[idx])
                    found_tokens[tok_clean] = weight

            # Sort by absolute weight
            sorted_tokens = sorted(
                found_tokens.items(), key=lambda item: abs(item[1]), reverse=True
            )[:15]
            attributions = [{"token": tok, "weight": wt} for tok, wt in sorted_tokens]
            method = "TFIDF-LR-Weights"
        except Exception as e:
            attributions = [{"token": "unavailable", "weight": 0.0, "error": str(e)}]

    return ExplanationResponse(
        prediction=pred_res,
        method=method,
        attributions=attributions,
    )
