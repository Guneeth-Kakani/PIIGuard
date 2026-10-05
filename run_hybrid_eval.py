import sys
sys.path.insert(0, ".")
from detectors.hybrid_detector import hybrid_detector
from evaluation.benchmark import run_benchmark, print_report
import json

results_diag = run_benchmark(
    hybrid_detector,
    ["evaluation/test_cases.json"],
    mode="exact",
    detector_name="hybrid presidio+onnx (diagnostic set)",
)
print_report(results_diag)

results_dev = run_benchmark(
    hybrid_detector,
    ["data/synthetic_dev.json"],
    mode="exact",
    detector_name="hybrid presidio+onnx (synthetic dev, exact match)",
)
print_report(results_dev)

results_dev_partial = run_benchmark(
    hybrid_detector,
    ["data/synthetic_dev.json"],
    mode="partial",
    detector_name="hybrid presidio+onnx (synthetic dev, partial match)",
)
print_report(results_dev_partial)

results_heldout = run_benchmark(
    hybrid_detector,
    ["data/synthetic_heldout.json"],
    mode="exact",
    detector_name="hybrid presidio+onnx (HELD-OUT, exact match)",
)
print_report(results_heldout)

results_heldout_partial = run_benchmark(
    hybrid_detector,
    ["data/synthetic_heldout.json"],
    mode="partial",
    detector_name="hybrid presidio+onnx (HELD-OUT, partial match)",
)
print_report(results_heldout_partial)

with open("evaluation/results_hybrid.json", "w") as f:
    json.dump({
        "diagnostic": results_diag,
        "synthetic_dev_exact": results_dev,
        "synthetic_dev_partial": results_dev_partial,
        "heldout_exact": results_heldout,
        "heldout_partial": results_heldout_partial,
    }, f, indent=2)
print("Saved to evaluation/results_hybrid.json")
