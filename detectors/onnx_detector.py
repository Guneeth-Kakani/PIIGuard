"""
Wraps the quantized ONNX DistilBERT model behind the same
detector_fn(text) -> list[dict] contract as every other detector in this
project. Uses onnxruntime.InferenceSession directly (rather than
optimum's ORTModel wrapper) for a lighter-weight, more transparent
inference path -- and reuses the identical BIO-merging logic from
detectors/distilbert_detector.py, so the only thing that changes between
the two detectors is which runtime executes the forward pass.
"""

import json
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "distilbert_pii_model_onnx_quantized"

_session = None
_tokenizer = None
_id2label = None


def _get_session():
    global _session, _tokenizer, _id2label
    if _session is None:
        _session = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"))
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
        _id2label = {int(k): v for k, v in config["id2label"].items()}
    return _session, _tokenizer, _id2label


def onnx_detector(text):
    session, tokenizer, id2label = _get_session()

    encoding = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=256, return_tensors="np",
    )
    offsets = encoding["offset_mapping"][0].tolist()

    ort_inputs = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    # Some exported graphs also expect token_type_ids -- include only if
    # the ONNX graph actually declares that input.
    input_names = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in input_names and "token_type_ids" in encoding:
        ort_inputs["token_type_ids"] = encoding["token_type_ids"].astype(np.int64)

    logits = session.run(None, ort_inputs)[0]
    predicted_ids = np.argmax(logits, axis=2)[0].tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]

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
    for ent in onnx_detector(sample):
        print(ent)
