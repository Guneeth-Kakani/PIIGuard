"""
Baseline PII detector: Presidio orchestration layer + custom recognizers
+ a direct spaCy pass for entity types Presidio's default pipeline
under-serves.

Design rationale (why this isn't just "import presidio, done"):

  1. Presidio's built-in recognizers cover EMAIL, PHONE, CREDIT_CARD, SSN,
     IP_ADDRESS, and (via its India-specific recognizer) PAN well out of
     the box — these are pattern-shaped entities, so we use Presidio's
     own PatternRecognizer machinery for them rather than reinventing regex.

  2. AADHAAR_LIKE is intentionally NOT handled via Presidio's built-in
     InAadhaarRecognizer. That recognizer runs a real Verhoeff checksum +
     "first digit >= 2" + non-palindrome validation in validate_result(),
     which is correct for verifying a genuine Aadhaar number but wrong for
     our purpose: a PII *redaction* gateway needs to mask anything SHAPED
     like an Aadhaar number, whether or not it happens to be checksum-valid
     (a real leaked number won't announce its own validity, and test/dummy
     numbers used in dev environments are routinely checksum-invalid too).
     So we register our own shape-only PatternRecognizer instead — this is
     a deliberate precision-for-recall tradeoff, not a workaround.

  3. Presidio's default SpacyRecognizer does NOT surface ORGANIZATION by
     default (it's disabled for precision reasons upstream) even though
     the underlying spaCy model detects ORG entities fine. Rather than
     fighting Presidio's recognizer config, we run spaCy directly as a
     second pass and pull ORG entities from its NER output, merging them
     in after Presidio's own PERSON detection.

  4. API_KEY and PROJECT_CODENAME have no built-in Presidio recognizer,
     so we add them as custom PatternRecognizers/standalone passes. Note
     explicitly: PROJECT_CODENAME here relies on structural regex
     ("Project/Operation/Initiative + Capitalized word"), which will do
     well on our synthetic data BECAUSE our synthetic data follows that
     structure. This is a known ceiling for the baseline — a codename
     that doesn't follow this naming convention (e.g. "Bluebird" used
     bare, without "Project" prefixed) will be missed. This exact gap is
     what Phase 3 (GLiNER zero-shot) is meant to close, so it's worth
     tracking as a baseline weakness rather than quietly working around it.

  5. MEDICAL_CONDITION and ADDRESS have no reliable regex signature and
     no dedicated Presidio recognizer. We use a deliberately weak
     keyword/heuristic recognizer here rather than skipping them, so the
     benchmark shows a real (low) score instead of hiding the gap. This
     is intentional: the baseline should honestly reveal what it can't
     do, since that's the evidence used to justify Phase 3/4.

  6. Presidio's AnalyzerEngine ships many recognizers (dates, locations,
     nationalities, UK/US-specific IDs, etc.) that aren't part of our
     taxonomy at all. Left unfiltered, these fire as false positives and
     can even win overlap-resolution against a real entity in our
     taxonomy (this happened concretely: a bare/grouped digit string was
     being claimed by the built-in DATE_TIME recognizer). We explicitly
     restrict output to our own taxonomy types, so any such stray
     recognizer output is dropped before it can compete for a span.

Merging: after collecting candidates from all sources, overlapping spans
are resolved by keeping the highest-confidence, longest span and dropping
whatever it overlaps with.
"""

import re
import spacy
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

_nlp = None
_analyzer = None

# Map Presidio's own entity names -> our project's taxonomy names
PRESIDIO_TO_TAXONOMY = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD",
    "US_SSN": "SSN",
    "IP_ADDRESS": "IP_ADDRESS",
    "AADHAAR_LIKE": "AADHAAR_LIKE",
    "PERSON": "PERSON",
}

# Our project's full entity taxonomy. Anything Presidio produces that maps
# outside this set (DATE_TIME, LOCATION, NRP, UK_NHS, US_BANK_NUMBER,
# US_DRIVER_LICENSE, etc. from Presidio's other built-in recognizers) is
# dropped before overlap resolution — see design note 6 above.
ALLOWED_TAXONOMY_TYPES = {
    "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS",
    "PAN_NUMBER", "AADHAAR_LIKE", "PERSON", "ORGANIZATION",
    "MEDICAL_CONDITION", "ADDRESS", "PROJECT_CODENAME", "API_KEY",
}

_TRAILING_STRIP_PATTERN = re.compile(r"('s|\u2019s|[.,;:])+$")

MEDICAL_KEYWORDS = [
    "type 2 diabetes", "type 1 diabetes", "hypertension", "asthma",
    "chronic migraine", "migraine", "seasonal allergies", "allergies",
    "anemia", "depression", "anxiety disorder", "epilepsy", "arthritis",
]

ADDRESS_HINT_PATTERN = re.compile(
    r"\b(?:Flat|Apartment|Apt\.?|House No\.?|Plot)\s+[\w\-]+,\s*[^,]+(?:,\s*[^,]+)*",
    re.IGNORECASE,
)

CODENAME_PATTERN = re.compile(r"\b(?:Project|Operation|Initiative)\s+[A-Z][a-z]+\b")

# Shape-only Aadhaar pattern: 12 digits grouped 4-4-4 with a space or dash
# separator. No checksum validation — see design note 2 above.
AADHAAR_PATTERN = re.compile(r"\b[0-9]{4}[- ][0-9]{4}[- ][0-9]{4}\b")


def _build_analyzer():
    """Set up Presidio's AnalyzerEngine with custom recognizers registered."""
    analyzer = AnalyzerEngine()

    # --- API_KEY: vendor-style prefixes ---
    api_key_patterns = [
        Pattern(name="anthropic_key", regex=r"\bsk-ant-[a-zA-Z0-9]{20,}\b", score=0.95),
        Pattern(name="openai_style_key", regex=r"\bsk-[a-zA-Z0-9]{20,}\b", score=0.9),
        Pattern(name="aws_access_key", regex=r"\bAKIA[A-Z0-9]{12,}\b", score=0.9),
    ]
    api_key_recognizer = PatternRecognizer(
        supported_entity="API_KEY", patterns=api_key_patterns
    )
    analyzer.registry.add_recognizer(api_key_recognizer)

    # --- AADHAAR_LIKE: shape-only, no checksum (see design note 2) ---
    aadhaar_patterns = [
        Pattern(name="aadhaar_grouped", regex=r"\b[0-9]{4}[- ][0-9]{4}[- ][0-9]{4}\b", score=0.86),
    ]
    aadhaar_recognizer = PatternRecognizer(
        supported_entity="AADHAAR_LIKE", patterns=aadhaar_patterns
    )
    analyzer.registry.add_recognizer(aadhaar_recognizer)

    # NOTE: PROJECT_CODENAME is intentionally NOT registered as a Presidio
    # PatternRecognizer. Presidio's Pattern class applies re.IGNORECASE
    # internally by default, which silently defeats a [A-Z][a-z]+
    # capitalization check (IGNORECASE case-folds character classes too,
    # not just literal text) — e.g. "the initiative is Operation Bluebird"
    # was matching the lowercase phrase "initiative is" as a false
    # positive codename. Handled instead as a standalone, case-sensitive
    # regex pass in _detect_codenames() below, where we control the flags.

    pan_patterns = [
    Pattern(name="pan_strict", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", score=0.9),
    ]
    pan_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns=pan_patterns)
    analyzer.registry.add_recognizer(pan_recognizer)

    analyzer.registry.remove_recognizer("UrlRecognizer")

    return analyzer


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = _build_analyzer()
    return _analyzer


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _trim_trailing(text, start, end):
    """Strip trailing possessive/punctuation characters a span may have
    swallowed (e.g. "Rahul Verma's" -> "Rahul Verma"), adjusting the end
    boundary. Returns (start, end)."""
    span = text[start:end]
    stripped = _TRAILING_STRIP_PATTERN.sub("", span)
    return start, start + len(stripped)


def _detect_organizations(text):
    """Direct spaCy pass for ORG entities, since Presidio's default
    SpacyRecognizer doesn't surface these. Also filters out common false
    positives: bare all-caps short tokens that are actually acronyms for
    other entity types in our taxonomy (e.g. "PAN", "SSN") rather than
    real organizations."""
    doc = _get_nlp()(text)
    known_non_org_acronyms = {"PAN", "SSN", "IP", "API"}
    results = []
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        if ent.text.upper() in known_non_org_acronyms:
            continue
        start, end = _trim_trailing(text, ent.start_char, ent.end_char)
        results.append({"type": "ORGANIZATION", "text": text[start:end],
                         "start": start, "end": end, "score": 0.75})
    return results


def _detect_medical_conditions(text):
    """Weak keyword-based heuristic — intentionally left simple to
    honestly expose baseline limitations rather than fake coverage."""
    results = []
    lowered = text.lower()
    for kw in MEDICAL_KEYWORDS:
        for m in re.finditer(re.escape(kw), lowered):
            results.append({
                "type": "MEDICAL_CONDITION", "text": text[m.start():m.end()],
                "start": m.start(), "end": m.end(), "score": 0.6,
            })
    return results


def _detect_codenames(text):
    """Standalone, genuinely case-sensitive regex pass — see module note
    on why this bypasses Presidio's Pattern class."""
    return [
        {"type": "PROJECT_CODENAME", "text": m.group(), "start": m.start(),
         "end": m.end(), "score": 0.85}
        for m in CODENAME_PATTERN.finditer(text)
    ]


def _detect_addresses(text):
    """Weak heuristic — flags lines with address-like keywords. Known gap."""
    results = []
    for m in ADDRESS_HINT_PATTERN.finditer(text):
        results.append({
            "type": "ADDRESS", "text": m.group(), "start": m.start(),
            "end": m.end(), "score": 0.5,
        })
    return results


def _resolve_overlaps(entities):
    """Greedily keep highest-confidence, longest spans; drop anything
    that overlaps with an already-accepted entity."""
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


def presidio_baseline_detector(text):
    """
    Main entry point — matches the detector_fn(text) -> list[dict]
    contract expected by evaluation/benchmark.py.
    """
    analyzer = _get_analyzer()
    candidates = []

    # 1. Presidio's built-in + our custom pattern recognizers
    presidio_results = analyzer.analyze(text=text, language="en")
    for r in presidio_results:
        taxonomy_type = PRESIDIO_TO_TAXONOMY.get(r.entity_type, r.entity_type)
        if taxonomy_type not in ALLOWED_TAXONOMY_TYPES:
            continue  # drop stray recognizer output outside our taxonomy
        start, end = _trim_trailing(text, r.start, r.end)
        candidates.append({
            "type": taxonomy_type,
            "text": text[start:end],
            "start": start,
            "end": end,
            "score": r.score,
        })

    # 2. Direct spaCy pass for ORG (Presidio doesn't surface this by default)
    candidates.extend(_detect_organizations(text))

    # 3. Weak heuristics for entity types with no reliable signature
    candidates.extend(_detect_medical_conditions(text))
    candidates.extend(_detect_addresses(text))
    candidates.extend(_detect_codenames(text))

    # 4. Resolve overlaps across all sources
    resolved = _resolve_overlaps(candidates)

    # strip internal "score" key to match the clean {type, text, start, end}
    # contract expected downstream — keep it available if callers want it
    return [{"type": e["type"], "text": e["text"], "start": e["start"], "end": e["end"]}
            for e in resolved]


if __name__ == "__main__":
    sample = (
        "My PAN is ABCDE1234F and Aadhaar number is 1234 5678 9012. "
        "Rahul Verma's phone number is +91-98765-43210 and his email is "
        "rahul.v@company.in. He works at Infosys and is part of Project "
        "Nightingale."
    )
    for ent in presidio_baseline_detector(sample):
        print(ent)
