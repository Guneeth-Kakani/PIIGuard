"""
Converts span-based entity annotations ({type, start, end}) into
per-token BIO tags, the format needed to fine-tune a token-classification
model like DistilBERT.

BIO tagging, in short: each TOKEN (not each character) gets one label.
  - "B-<TYPE>"  = this token BEGINS an entity of that type
  - "I-<TYPE>"  = this token continues (is INSIDE) an entity begun earlier
  - "O"         = this token is Outside any entity

Example:
  text:     "Rahul Verma emailed rahul@x.com"
  tokens:   ["Rahul", "Verma", "emailed", "rahul@x.com"]
  BIO tags: ["B-PERSON", "I-PERSON", "O", "B-EMAIL"]

This module is intentionally dependency-light (uses the tokenizer's own
offset mapping, not a separate tokenization scheme), so the exact same
tokenizer used for training later is what defines "a token" here — this
matters because BIO tags must line up with whatever tokenizer DistilBERT
actually uses at training time, not an arbitrary whitespace split.
"""

from transformers import AutoTokenizer

# Every entity type in our taxonomy — defines the full label set the
# token classifier will be trained to predict.
ENTITY_TYPES = [
    "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "SSN", "IP_ADDRESS",
    "PAN_NUMBER", "AADHAAR_LIKE", "PERSON", "ORGANIZATION",
    "MEDICAL_CONDITION", "ADDRESS", "PROJECT_CODENAME", "API_KEY",
]

# Build the full BIO label list: O, B-EMAIL, I-EMAIL, B-PERSON, I-PERSON, ...
LABEL_LIST = ["O"] + [f"{prefix}-{t}" for t in ENTITY_TYPES for prefix in ("B", "I")]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _tokenizer


def spans_to_bio(text, entities, tokenizer=None):
    """
    Convert a text + its span-based entities into a list of
    (token_string, bio_label) pairs, using the tokenizer's own subword
    offsets to guarantee alignment with what the model will actually see.

    `entities` should be a list of {"type": str, "start": int, "end": int}
    dicts (the "text" key, if present, is ignored — start/end drive
    everything, matching the detector_fn contract used elsewhere in this
    project).
    """
    tok = tokenizer or _get_tokenizer()
    encoding = tok(text, return_offsets_mapping=True, truncation=True, max_length=256)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    # Sort entities by start so overlapping/adjacent spans are handled
    # deterministically (shouldn't happen if fed from an already-resolved
    # ensemble output, but defensive regardless).
    entities = sorted(entities, key=lambda e: e["start"])

    bio_labels = []
    for token_id, (start, end) in zip(input_ids, offsets):
        # Special tokens ([CLS], [SEP], padding) have offset (0, 0) and
        # get a dedicated ignore label, handled at training time via
        # label -100 — see `align_labels_for_training` below.
        if start == end:
            bio_labels.append(None)  # placeholder for special tokens
            continue

        label = "O"
        for ent in entities:
            if start >= ent["start"] and start < ent["end"]:
                # Token begins exactly at (or starts within) the entity's
                # start -> B-, otherwise it's a continuation -> I-
                prefix = "B" if start == ent["start"] else "I"
                label = f"{prefix}-{ent['type']}"
                break

        bio_labels.append(label)

    tokens = tok.convert_ids_to_tokens(input_ids)
    return list(zip(tokens, bio_labels)), encoding


def align_labels_for_training(bio_labels):
    """
    Convert a list of BIO label strings (with None for special tokens)
    into label IDs, using -100 for special tokens — the standard
    HuggingFace convention that tells the loss function to ignore those
    positions entirely during training.
    """
    return [-100 if lbl is None else LABEL_TO_ID[lbl] for lbl in bio_labels]


if __name__ == "__main__":
    sample_text = "Rahul Verma's phone number is +91-98765-43210."
    sample_entities = [
        {"type": "PERSON", "start": 0, "end": 11},
        {"type": "PHONE_NUMBER", "start": 30, "end": 46},
    ]

    pairs, encoding = spans_to_bio(sample_text, sample_entities)
    print(f"{'TOKEN':<15}{'BIO LABEL'}")
    for token, label in pairs:
        print(f"{token:<15}{label}")

    label_ids = align_labels_for_training([lbl for _, lbl in pairs])
    print("\nLabel IDs (for training):", label_ids)
    print(f"\nTotal label set size: {len(LABEL_LIST)} labels")
