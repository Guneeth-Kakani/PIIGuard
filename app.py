"""
Phase 6: FastAPI integration.

Three detection paths, each a genuine tradeoff, not a "best" choice:

  /detect           ONNX quantized DistilBERT only. Fastest single
                     model, but known to fragment structured entities
                     (email, phone, PAN, etc.) since a neural token
                     classifier has no guaranteed regex-like precision
                     on deterministic formats.

  /detect/hybrid     Presidio (regex) for structured types + ONNX for
                     contextual types. Fixes the fragmentation problem
                     entirely for structured types, at a small latency
                     cost over ONNX-only (still much faster than the
                     full ensemble, since neither Presidio's regex nor
                     the quantized model are slow individually).
                     RECOMMENDED DEFAULT for most real traffic.

  /detect/ensemble   Presidio + GLiNER, the original two-model system.
                     Highest accuracy available in this project, but
                     slowest (GLiNER's forward pass dominates latency).
                     Best for offline audit or manual review, not the
                     live request path.

Run for real (not inside the notebook) with:
    uvicorn app:app --reload --port 8000
Then see http://127.0.0.1:8000/docs for interactive testing.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from detectors.onnx_detector import onnx_detector
from detectors.hybrid_detector import hybrid_detector
from detectors.ensemble_detector import ensemble_detector

app = FastAPI(
    title="PII Redactor Detection API",
    description="Detects PII spans in text for a privacy-preserving LLM proxy gateway.",
    version="1.1.0",
)


class DetectRequest(BaseModel):
    text: str


class Entity(BaseModel):
    type: str
    text: str
    start: int
    end: int


class DetectResponse(BaseModel):
    entities: list[Entity]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    """Fastest path: ONNX quantized DistilBERT only. Known limitation:
    can fragment structured entities (email, phone, PAN) -- prefer
    /detect/hybrid unless raw speed matters more than that tradeoff."""
    entities = onnx_detector(request.text)
    return {"entities": entities}


@app.post("/detect/hybrid", response_model=DetectResponse)
def detect_hybrid(request: DetectRequest):
    """RECOMMENDED DEFAULT: Presidio (regex) for structured types +
    ONNX (neural) for contextual types. Fixes structured-entity
    fragmentation while staying fast."""
    entities = hybrid_detector(request.text)
    return {"entities": entities}


@app.post("/detect/ensemble", response_model=DetectResponse)
def detect_ensemble(request: DetectRequest):
    """High-accuracy path: Presidio + GLiNER. Slower -- suited to
    offline audit passes or manual review queues, not the live
    request path."""
    entities = ensemble_detector(request.text)
    return {"entities": entities}
