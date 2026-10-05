import sys
sys.path.insert(0, ".")
from detectors.distilbert_detector import distilbert_detector
from evaluation.benchmark import run_benchmark, print_report
import json

results_diag = run_benchmark(
    distilbert_detector,
    ["evaluation/test_cases.json"],
    mode="exact",
    detector_name="distilbert (diagnostic set)",
)
print_report(results_diag)

results_dev = run_benchmark(
    distilbert_detector,
    ["data/synthetic_dev.json"],
    mode="exact",
    detector_name="distilbert (synthetic dev, exact match)",
)
print_report(results_dev)

results_dev_partial = run_benchmark(
    distilbert_detector,
    ["data/synthetic_dev.json"],
    mode="partial",
    detector_name="distilbert (synthetic dev, partial match)",
)
print_report(results_dev_partial)

with open("evaluation/results_distilbert.json", "w") as f:
    json.dump({
        "diagnostic": results_diag,
        "synthetic_dev_exact": results_dev,
        "synthetic_dev_partial": results_dev_partial,
    }, f, indent=2)
print("Saved to evaluation/results_distilbert.json")
