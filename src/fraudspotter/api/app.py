from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fraudspotter.api.routes import router, set_active_model
from fraudspotter.common.logger import setup_logger

logger = setup_logger("fraudspotter.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize models on startup."""
    logger.info("Initializing FraudSpotter API services...")

    # Attempt to load INT8 ONNX model if present
    onnx_path = Path("artifacts/models/distilbert_onnx/model_quantized.onnx")
    classical_path = Path("artifacts/models/classical/logreg.joblib")

    if onnx_path.is_file():
        try:
            from fraudspotter.models.onnx_inference import ONNXClassifier

            model = ONNXClassifier(str(onnx_path))
            set_active_model(model, "onnx")
            logger.info("Successfully loaded ONNX INT8 model into runtime.")
        except Exception as e:
            logger.warning("Could not load ONNX model (%s). Checking classical fallback...", e)

    elif classical_path.is_file():
        try:
            from fraudspotter.models.classical import load_model

            model = load_model(str(classical_path))
            set_active_model(model, "classical")
            logger.info("Successfully loaded classical Logistic Regression model.")
        except Exception as e:
            logger.warning("Could not load classical model: %s", e)
    else:
        logger.info(
            "No saved model artifacts found yet. Training pipeline needed before active serving."
        )

    yield
    logger.info("Shutting down FraudSpotter API services.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="FraudSpotter API",
        description="Explainable Fake News Classification API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
