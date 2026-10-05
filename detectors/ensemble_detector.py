"""
Ensemble detector — static per-type routing between Presidio (structured,
regex-friendly) and GLiNER (contextual, zero-shot), decided from the
per-type F1 comparison in evaluation/results_presidio_baseline.json vs.
evaluation/results_gliner.json.
"""

from detectors.presidio_baseline import presidio_baseline_detector, _resolve_overlaps
from detectors.gliner_detector import gliner_detector

# UPDATE "PERSON" and "ADDRESS" below based on Cell 1 / Cell 2 findings
ROUTING = {
    "EMAIL": "presidio", "PHONE_NUMBER": "presidio", "API_KEY": "presidio",
    "PAN_NUMBER": "presidio", "SSN": "presidio", "IP_ADDRESS": "presidio",
    "CREDIT_CARD": "presidio", "AADHAAR_LIKE": "presidio",
    "ORGANIZATION": "gliner", "MEDICAL_CONDITION": "gliner",
    "PROJECT_CODENAME": "presidio",
    "ADDRESS": "gliner",   # <-- confirm from Cell 2
    "PERSON": "gliner",    # <-- confirm from Cell 1
}


def ensemble_detector(text):
    presidio_results = presidio_baseline_detector(text)
    gliner_results = gliner_detector(text)

    candidates = []
    candidates += [{**e, "score": 0.95} for e in presidio_results if ROUTING.get(e["type"]) == "presidio"]
    candidates += [{**e, "score": 0.6} for e in gliner_results if ROUTING.get(e["type"]) == "gliner"]

    resolved = _resolve_overlaps(candidates)
    return [{"type": e["type"], "text": e["text"], "start": e["start"], "end": e["end"]}
            for e in resolved]


if __name__ == "__main__":
    sample = (
        "Rahul Verma's phone number is +91-98765-43210 and his email is "
        "rahul.v@company.in. He works at Infosys and is part of Project "
        "Nightingale. His PAN is ABCDE1234F."
    )
    for ent in ensemble_detector(sample):
        print(ent)
