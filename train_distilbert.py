"""
DistilBERT fine-tuning for the token-classification distillation step
(Phase 4).

Fixes applied after the first training run:
  - `processing_class=tokenizer` passed to Trainer, so the tokenizer is
    saved alongside every checkpoint (the first run omitted this, which
    caused detectors/distilbert_detector.py to fail loading a tokenizer
    from the saved model directory).
  - `trainer.save_model(...)` + `tokenizer.save_pretrained(...)` called
    explicitly after training, writing the final model directly to the
    root `distilbert_pii_model/` folder — not just inside a
    `checkpoint-N/` subfolder — so downstream code can load from one
    predictable path.
"""

import sys
sys.path.insert(0, ".")

import json
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from datasets import Dataset

from data.bio_tagging import (
    spans_to_bio, align_labels_for_training, LABEL_LIST, LABEL_TO_ID, ID_TO_LABEL,
)
from detectors.ensemble_detector import ensemble_detector

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "distilbert_pii_model"


def build_training_examples(corpus_path):
    """
    Reads a corpus of {"text": str, ...} cases, runs the ensemble
    detector over each one to produce auto-labels (span-based), converts
    those to BIO-tagged, tokenizer-aligned label IDs, and returns a
    HuggingFace Dataset ready for the Trainer.

    NOTE: this is what makes it "distillation" rather than supervised
    training on ground truth — the labels here are the ensemble's own
    predictions, not independently verified truth. Any systematic error
    the ensemble makes (e.g. the known ADDRESS/MEDICAL_CONDITION gaps)
    will be learned by the student model too, not corrected by it.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(corpus_path) as f:
        cases = json.load(f)

    all_input_ids, all_attention_masks, all_labels = [], [], []

    for case in cases:
        text = case["text"]
        auto_entities = ensemble_detector(text)  # {type, text, start, end}

        pairs, encoding = spans_to_bio(text, auto_entities, tokenizer=tokenizer)
        label_ids = align_labels_for_training([lbl for _, lbl in pairs])

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_labels.append(label_ids)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


def compute_metrics(eval_pred):
    """Token-level accuracy, ignoring -100 (special token) positions.
    NOTE: this metric is dominated by the "O" (non-entity) label, since
    most tokens in any sentence aren't part of an entity — a high number
    here does NOT mean entity detection is accurate. Real entity-level
    precision/recall/F1 comes from run_distilbert_eval.py instead."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels_flat, pred_labels_flat = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_labels_flat.append(l)
            pred_labels_flat.append(p)

    correct = sum(p == t for p, t in zip(pred_labels_flat, true_labels_flat))
    total = len(true_labels_flat)
    return {"token_accuracy": correct / total if total else 0.0}


def build_model():
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def build_trainer(train_dataset, eval_dataset, tokenizer):
    model = build_model()
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Building the labeled training dataset (running the ensemble "
          "over all corpus cases — this takes a while)...")
    full_dataset = build_training_examples("data/phase4_corpus.json")
    print(f"Dataset built: {len(full_dataset)} examples")
    print(f"Label set size: {len(LABEL_LIST)}")
    print("Sample input_ids length:", len(full_dataset[0]["input_ids"]))
    print("Sample labels:", full_dataset[0]["labels"][:20])

    # Split: reuse the corpus's own examples for a quick train/eval split
    # here; NOTE this is separate from and should not be confused with
    # the project's sealed data/synthetic_heldout.json, which stays
    # untouched until Phase 5's final evaluation.
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    trainer = build_trainer(train_dataset, eval_dataset, tokenizer)
    print("\nTrainer built successfully. Starting fine-tuning...")

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel and tokenizer saved to {OUTPUT_DIR}/")
