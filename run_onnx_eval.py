import sys
sys.path.insert(0, ".")
from detectors.onnx_detector import onnx_detector
from evaluation.benchmark import run_benchmark, print_report
import json

results_diag = run_benchmark(
    onnx_detector,
    ["evaluation/test_cases.json"],
    mode="exact",
    detector_name="onnx quantized (diagnostic set)",
)
print_report(results_diag)

results_dev = run_benchmark(
    onnx_detector,
    ["data/synthetic_dev.json"],
    mode="exact",
    detector_name="onnx quantized (synthetic dev, exact match)",
)
print_report(results_dev)

results_dev_partial = run_benchmark(
    onnx_detector,
    ["data/synthetic_dev.json"],
    mode="partial",
    detector_name="onnx quantized (synthetic dev, partial match)",
)
print_report(results_dev_partial)

with open("evaluation/results_onnx.json", "w") as f:
    json.dump({
        "diagnostic": results_diag,
        "synthetic_dev_exact": results_dev,
        "synthetic_dev_partial": results_dev_partial,
    }, f, indent=2)
print("Saved to evaluation/results_onnx.json")
