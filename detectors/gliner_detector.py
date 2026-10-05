"""
GLiNER zero-shot NER detector — same detector_fn(text) -> list[dict]
contract as detectors/presidio_baseline.py, so it drops straight into
the existing evaluation harness.

ADDRESS boundary note: testing showed GLiNER reliably finds the START of
a street address correctly, but frequently splits the full comma-separated
address ("street, city, state zip") into 2-3 separate fragment entities
instead of one span.

  - Attempt 1: relabeling the prompt from "address" to "street address" —
    tested, did NOT help (confirmed via gold-vs-predicted comparison: the
    issue isn't label ambiguity, GLiNER splits full addresses into
    fragments regardless of wording). Reverted.
  - Attempt 2: an unbounded post-processing extension that grabbed every
    trailing ", <segment>" chunk — this OVER-extended into unrelated
    trailing clauses (e.g. ", attention Jane Doe"), which then lost
    overlap resolution against a separately-detected PERSON entity,
    making results worse than doing nothing (partial F1 dropped to
    0.045 from a baseline of ~0.39).
  - Attempt 3 (current): extend the span forward through comma-separated
    segments, but STOP as soon as a segment matches a US-style
    "STATE ZIPCODE" pattern (e.g. "KS 38439") — using the natural
    structural end of a US address as the stop signal, instead of an
    arbitrary word/segment count. This avoids swallowing unrelated
    trailing clauses while still capturing city + state + zip.
    KNOWN LIMITATION: this stopping rule is US-ZIP-shaped, so it won't
    have the same signal for non-US address formats (e.g. this project's
    Indian-context addresses, which have no state+zip suffix) — those
    still rely on the max_segments cap alone and may need a separate
    locale-specific rule if this remains a priority.
"""

import re
from gliner import GLiNER

_model = None

# Labels passed to GLiNER at inference time (zero-shot — no training).
# Only covering the CONTEXTUAL types here; structured types (EMAIL, PHONE,
# API_KEY, PAN, SSN, IP, CREDIT_CARD) stay regex-owned, GLiNER isn't
# expected to compete there.
LABELS = [
    "person",
    "organization",
    "medical condition",
    "address",
    "project codename",
]

GLINER_TO_TAXONOMY = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "medical condition": "MEDICAL_CONDITION",
    "address": "ADDRESS",
    "project codename": "PROJECT_CODENAME",
}

# Matches a US-style "STATE ZIPCODE" segment, e.g. "KS 38439" — used as
# the stopping signal for address-span extension (see module note above).
_STATE_ZIP_PATTERN = re.compile(r"^[A-Za-z]{2}\s+\d{4,6}$")

# Matches one comma-separated segment following an address span, e.g.
# ", Port Derrick" or ", KS 38439".
_ADDRESS_SEGMENT_PATTERN = re.compile(r",\s*([^,\.;]{1,40})")


def _extend_address_span(text, start, end, max_segments=3):
    """GLiNER reliably anchors the START of an address but often stops
    after the street portion. Extend the span forward through comma-
    separated segments (city, then state+zip), stopping as soon as a
    segment matches a US-style 'STATE ZIPCODE' pattern — the natural end
    of the address — or after max_segments, whichever comes first."""
    extended_end = end
    pos = 0
    remainder = text[end:]
    for _ in range(max_segments):
        m = _ADDRESS_SEGMENT_PATTERN.match(remainder, pos)
        if not m:
            break
        segment = m.group(1).strip()
        extended_end = end + m.end()
        pos = m.end()
        if _STATE_ZIP_PATTERN.match(segment):
            break  # reached the natural end of a US-style address
    return start, extended_end


def _get_model():
    global _model
    if _model is None:
        _model = GLiNER.from_pretrained("urchade/gliner_multi")
    return _model


def _resolve_overlaps(entities):
    """Same greedy longest/highest-confidence overlap resolution as the
    Presidio baseline, so results are comparable apples-to-apples."""
    ranked = sorted(
        entities, key=lambda e: (e.get("score", 0.5), e["end"] - e["start"]),
        reverse=True,
    )
    accepted = []
    for e in ranked:
        overlaps = any(
            e["start"] < a["end"] and a["start"] < e["end"] for a in accepted
        )
        if not overlaps:
            accepted.append(e)
    accepted.sort(key=lambda e: e["start"])
    return accepted


def gliner_detector(text, threshold=0.4):
    model = _get_model()
    preds = model.predict_entities(text, LABELS, threshold=threshold)

    candidates = []
    for p in preds:
        taxonomy_type = GLINER_TO_TAXONOMY.get(p["label"], p["label"].upper())
        start, end = p["start"], p["end"]

        if taxonomy_type == "ADDRESS":
            start, end = _extend_address_span(text, start, end)

        candidates.append({
            "type": taxonomy_type,
            "text": text[start:end],
            "start": start,
            "end": end,
            "score": p["score"],
        })

    resolved = _resolve_overlaps(candidates)
    return [{"type": e["type"], "text": e["text"], "start": e["start"], "end": e["end"]}
            for e in resolved]


if __name__ == "__main__":
    sample = (
        "Dr. Anjali Mehta prescribed metformin for Rohan's type 2 diabetes "
        "at Apollo Hospital. Project Nightingale's launch was delayed. "
        "Shipping to 49784 Jeffrey Rapid Apt. 034, Port Derrick, KS 38439, "
        "attention Joseph Bennett."
    )
    for ent in gliner_detector(sample):
        print(ent)
