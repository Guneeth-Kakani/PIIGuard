"""
Phase 5, part 1: export the fine-tuned DistilBERT model to ONNX format,
then apply dynamic INT8 quantization.

Why ONNX + quantization: the PyTorch model works, but the eventual goal
is a low-latency detection endpoint, not a full-precision research
model. Converting to ONNX (a portable, optimized inference graph) and
quantizing weights from FP32 down to INT8 typically shrinks model size
~4x and speeds up CPU inference meaningfully -- at some small, measurable
cost to accuracy, which we benchmark explicitly in the next few
sections rather than assuming it's negligible.

Uses HuggingFace's `optimum` library for the export step (handles
transformer graph tracing and opset selection reliably -- this is the
part that most commonly breaks with hand-rolled torch.onnx.export calls
on transformer models), then onnxruntime's own `quantize_dynamic`
utility for the INT8 conversion. Deliberately NOT using optimum's
CPU-architecture-specific quantization configs (e.g. AVX512/VNNI presets)
since those assume specific CPU instruction sets that may not exist on
every machine -- plain dynamic quantization makes no such assumption and
is the safer, more portable choice here.
"""

import os
import shutil
from optimum.onnxruntime import ORTModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

PYTORCH_MODEL_DIR = "distilbert_pii_model"
ONNX_DIR = "distilbert_pii_model_onnx"
QUANTIZED_DIR = "distilbert_pii_model_onnx_quantized"

os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(QUANTIZED_DIR, exist_ok=True)

print("Exporting PyTorch model to ONNX...")
ort_model = ORTModelForTokenClassification.from_pretrained(PYTORCH_MODEL_DIR, export=True)
ort_model.save_pretrained(ONNX_DIR)

tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_DIR)
tokenizer.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(QUANTIZED_DIR)  # quantized dir needs its own tokenizer copy too

raw_onnx_path = os.path.join(ONNX_DIR, "model.onnx")
quantized_onnx_path = os.path.join(QUANTIZED_DIR, "model.onnx")

print("Applying dynamic INT8 quantization...")
quantize_dynamic(
    model_input=raw_onnx_path,
    model_output=quantized_onnx_path,
    weight_type=QuantType.QInt8,
)

# config.json holds id2label/label2id -- needed at inference time, copy
# it into the quantized dir alongside the quantized model file.
shutil.copy(os.path.join(ONNX_DIR, "config.json"), os.path.join(QUANTIZED_DIR, "config.json"))

raw_size = os.path.getsize(raw_onnx_path) / (1024 * 1024)
quant_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)
print(f"\nRaw ONNX model size: {raw_size:.1f} MB")
print(f"Quantized ONNX model size: {quant_size:.1f} MB")
print(f"Size reduction: {(1 - quant_size / raw_size) * 100:.1f}%")
print(f"\nSaved to {QUANTIZED_DIR}/")
