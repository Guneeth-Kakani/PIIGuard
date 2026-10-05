"""
Wraps the fine-tuned DistilBERT token-classification model behind the
same detector_fn(text) -> list[dict] contract as every other detector in
this project (presidio_baseline.py, gliner_detector.py, ensemble_detector.py).

This is the reverse operation of data/bio_tagging.py: instead of
converting spans -> per-token BIO labels for training, this converts the
model's per-token BIO *predictions* back into spans, by:
  1. Tokenizing the input and getting each token's character offsets
     (same tokenizer used at training time, for guaranteed alignment).
  2. Running the model to get a predicted label ID per token.
  3. Walking through predictions left to right, merging a B-<TYPE> token
     with any immediately following I-<TYPE> tokens of the SAME type into
     one entity span (using the first token's start and the last token's
     end as the span boundaries).
  4. Ignoring special tokens ([CLS], [SEP], padding) via their (0,0)
     offset, same convention used during training.

MODEL_DIR points at the root output directory train_distilbert.py saves
to directly (via trainer.save_model(...) + tokenizer.save_pretrained(...)
after training) — not a checkpoint-N/ subfolder.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "distilbert_pii_model"

_tokenizer = None
_model = None


def _get_model_and_tokenizer():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _model, _tokenizer


def distilbert_detector(text):
    model, tokenizer = _get_model_and_tokenizer()

    encoding = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=256, return_tensors="pt",
    )
    offsets = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        ).logits

    predicted_ids = torch.argmax(logits, dim=2)[0].tolist()
    predicted_labels = [model.config.id2label[i] for i in predicted_ids]

    entities = []
    current_entity = None  # {"type": str, "start": int, "end": int}

    for label, (start, end) in zip(predicted_labels, offsets):
        if start == end:
            continue  # special token ([CLS], [SEP], padding)

        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        prefix, _, entity_type = label.partition("-")

        if prefix == "B" or current_entity is None or current_entity["type"] != entity_type:
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": entity_type, "start": start, "end": end}
        else:  # "I" continuing the same entity type
            current_entity["end"] = end

    if current_entity:
        entities.append(current_entity)

    return [{"type": e["type"], "text": text[e["start"]:e["end"]],
             "start": e["start"], "end": e["end"]} for e in entities]


if __name__ == "__main__":
    sample = (
        "Rahul Verma's phone number is +91-98765-43210 and his email is "
        "rahul.v@company.in. He works at Infosys and is part of Project "
        "Nightingale. His PAN is ABCDE1234F."
    )
    for ent in distilbert_detector(sample):
        print(ent)
