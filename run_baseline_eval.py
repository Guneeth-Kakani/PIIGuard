import sys
sys.path.insert(0, ".")
from detectors.presidio_baseline import presidio_baseline_detector
from evaluation.benchmark import run_benchmark, print_report
import json

# Diagnostic set (16 hand-written)
results_diag = run_benchmark(
    presidio_baseline_detector,
    ["evaluation/test_cases.json"],
    mode="exact",
    detector_name="presidio_baseline (diagnostic set)",
)
print_report(results_diag)

# Synthetic dev set (800 cases) — NOT the held-out set, per Phase 5 discipline
results_dev = run_benchmark(
    presidio_baseline_detector,
    ["data/synthetic_dev.json"],
    mode="exact",
    detector_name="presidio_baseline (synthetic dev, exact match)",
)
print_report(results_dev)

# Same dev set, partial/overlap match — to see how much of the gap is
# boundary noise vs. true misses
results_dev_partial = run_benchmark(
    presidio_baseline_detector,
    ["data/synthetic_dev.json"],
    mode="partial",
    detector_name="presidio_baseline (synthetic dev, partial match)",
)
print_report(results_dev_partial)

with open("evaluation/results_presidio_baseline.json", "w") as f:
    json.dump({
        "diagnostic": results_diag,
        "synthetic_dev_exact": results_dev,
        "synthetic_dev_partial": results_dev_partial,
    }, f, indent=2)
print("Saved to evaluation/results_presidio_baseline.json")
