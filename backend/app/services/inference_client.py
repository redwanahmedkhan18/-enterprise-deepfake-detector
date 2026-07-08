"""
Client for calling the AI Inference Service (the ai_engine, served separately —
e.g. as a FastAPI/Triton service — so it can scale and deploy independently of
this API backend). See Phase 3-5 of the roadmap for that service's own code.
"""
import httpx

from app.core.config import settings
from app.exceptions.custom_exceptions import InferenceServiceError


class InferenceClient:
    def __init__(self):
        self.base_url = settings.INFERENCE_SERVICE_URL
        self.timeout = settings.INFERENCE_TIMEOUT_SECONDS

    async def run_detection(self, media_url: str, media_type: str) -> dict:
        """
        Calls POST {INFERENCE_SERVICE_URL}/detect with a presigned media URL.
        Expected response shape:
        {
            "is_fake": bool,
            "confidence": float,
            "branch_scores": {"spatial": float, "temporal": float, "frequency": float, "physiological": float},
            "explainability_uri": str | null,
            "model_version": str,
            "processing_time_ms": int
        }
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/detect",
                    json={"media_url": media_url, "media_type": media_type},
                )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                raise InferenceServiceError(f"Inference service call failed: {exc}") from exc
