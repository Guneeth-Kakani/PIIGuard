"""
THE FINAL, UNBIASED NUMBER.

data/synthetic_heldout.json has been sealed since Phase 1 -- generated
alongside synthetic_dev.json, but deliberately never looked at, never
used to tune a threshold, a routing decision, a template, or a training
label. Every fix in this project (Aadhaar/PAN patterns, ensemble
overlap-resolution, ADDRESS boundary extension, the SSN/CREDIT_CARD
corpus rebalancing) was validated against the DEV set, which means dev
performance is somewhat optimistic -- it's the set we iterated against.
This is the one moment we check against data the whole pipeline has
never seen, in any form, at any point.

Runs both the ensemble (the two-model system everything was distilled
from) and the quantized ONNX DistilBERT (the final, fast, deployable
model) against the held-out set, so the honest comparison is visible in
one place.
"""
import sys
sys.path.insert(0, ".")
from detectors.ensemble_detector import ensemble_detector
from detectors.onnx_detector import onnx_detector
from evaluation.benchmark import run_benchmark, print_report
import json

print("=" * 70)
print("HELD-OUT SET RESULTS -- never used for tuning until this run")
print("=" * 70)

ensemble_results = run_benchmark(
    ensemble_detector,
    ["data/synthetic_heldout.json"],
    mode="exact",
    detector_name="ENSEMBLE (held-out, exact match) -- FINAL",
)
print_report(ensemble_results)

ensemble_results_partial = run_benchmark(
    ensemble_detector,
    ["data/synthetic_heldout.json"],
    mode="partial",
    detector_name="ENSEMBLE (held-out, partial match) -- FINAL",
)
print_report(ensemble_results_partial)

onnx_results = run_benchmark(
    onnx_detector,
    ["data/synthetic_heldout.json"],
    mode="exact",
    detector_name="ONNX QUANTIZED DISTILBERT (held-out, exact match) -- FINAL",
)
print_report(onnx_results)

onnx_results_partial = run_benchmark(
    onnx_detector,
    ["data/synthetic_heldout.json"],
    mode="partial",
    detector_name="ONNX QUANTIZED DISTILBERT (held-out, partial match) -- FINAL",
)
print_report(onnx_results_partial)

with open("evaluation/results_heldout_FINAL.json", "w") as f:
    json.dump({
        "ensemble_exact": ensemble_results,
        "ensemble_partial": ensemble_results_partial,
        "onnx_exact": onnx_results,
        "onnx_partial": onnx_results_partial,
    }, f, indent=2)
print("\nSaved to evaluation/results_heldout_FINAL.json")
print("\nThis is the number that goes in the final report.")
