"""
Hybrid detector: Presidio (fast regex) + ONNX quantized DistilBERT (fast
neural), instead of Presidio + GLiNER (slow neural). Same routing
philosophy as ensemble_detector.py -- structured, format-deterministic
types never need a neural model at all, and asking one to detect them
anyway is exactly what caused the email/phone/PAN fragmentation bugs
seen in the deployed ONNX-only /detect endpoint. Regex either matches
the whole span or it doesn't -- it cannot fragment mid-entity the way a
token classifier occasionally does.

This keeps DistilBERT/ONNX only for what it's actually needed for:
PERSON, ORGANIZATION, MEDICAL_CONDITION, ADDRESS, PROJECT_CODENAME --
the contextual types with no reliable regex signature.

Expected result: near-Presidio accuracy on every structured type (no more
fragmentation), while staying close to ONNX-only latency, since neither
Presidio's regex nor the quantized model are slow on their own -- the
ensemble's 0.377s mean latency was specifically GLiNER's cost, not an
inherent cost of routing between two sources.
"""

from detectors.presidio_baseline import presidio_baseline_detector, _resolve_overlaps
from detectors.onnx_detector import onnx_detector

# Structured, regex-owned types -- Presidio only, same list as
# ensemble_detector.py's ROUTING.
STRUCTURED_TYPES = {
    "EMAIL", "PHONE_NUMBER", "API_KEY", "PAN_NUMBER", "SSN",
    "IP_ADDRESS", "CREDIT_CARD", "AADHAAR_LIKE", "PROJECT_CODENAME",
}

# Contextual types -- no reliable regex shape, ONNX/DistilBERT only.
CONTEXTUAL_TYPES = {
    "PERSON", "ORGANIZATION", "MEDICAL_CONDITION", "ADDRESS",
}


def hybrid_detector(text):
    presidio_results = presidio_baseline_detector(text)
    onnx_results = onnx_detector(text)

    candidates = []
    # Presidio scored higher: when a neural false positive on a
    # structured-shaped span collides with a correct Presidio match
    # (e.g. ONNX mistagging a fragment of an email as PERSON), the
    # deterministic regex match should win the overlap.
    candidates += [{**e, "score": 0.95} for e in presidio_results if e["type"] in STRUCTURED_TYPES]
    candidates += [{**e, "score": 0.6} for e in onnx_results if e["type"] in CONTEXTUAL_TYPES]

    resolved = _resolve_overlaps(candidates)
    return [{"type": e["type"], "text": e["text"], "start": e["start"], "end": e["end"]}
            for e in resolved]


if __name__ == "__main__":
    sample = (
        "Rahul Verma's phone number is +91-98765-43210 and his email is "
        "rahul.v@company.in. He works at Infosys and is part of Project "
        "Nightingale. His PAN is ABCDE1234F."
    )
    for ent in hybrid_detector(sample):
        print(ent)
