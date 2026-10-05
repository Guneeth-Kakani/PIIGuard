"""
Evaluation harness for PII detectors.

Any detector is just a Python function with this signature:

    def detector_fn(text: str) -> list[dict]:
        # each dict: {"type": str, "text": str, "start": int, "end": int}
        return [...]

Feed it to `run_benchmark()` along with one or more test case files, and
get back per-entity-type precision/recall/F1 plus latency percentiles.

Design notes:
- Matching is EXACT SPAN MATCH (same type, same start, same end) by default,
  which is the standard strict metric for NER. A `partial` mode is also
  provided (overlap-based) since strict matching can be harsh for
  boundary-off-by-one errors — report both if boundaries are noisy.
- Metrics are computed manually here rather than via seqeval directly,
  because seqeval expects token-aligned BIO tags, which forces a specific
  tokenizer choice. Span-level exact match avoids that dependency and
  works identically regardless of what tokenizer a given detector uses
  internally — this matters because you'll be comparing regex, GLiNER,
  and DistilBERT detectors that may tokenize very differently.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class EntityMatch:
    entity_type: str
    matched: bool


def spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def match_entities(gold, pred, mode="exact"):
    """
    Returns (tp, fp, fn) lists of entities, each tagged with type.
    mode: "exact" (start/end/type must all match) or "partial" (overlap + type match)
    """
    gold_matched = [False] * len(gold)
    pred_matched = [False] * len(pred)

    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if gold_matched[gi]:
                continue
            if p["type"] != g["type"]:
                continue
            if mode == "exact":
                is_match = p["start"] == g["start"] and p["end"] == g["end"]
            else:
                is_match = spans_overlap(p["start"], p["end"], g["start"], g["end"])
            if is_match:
                gold_matched[gi] = True
                pred_matched[pi] = True
                break

    tp = [p for pi, p in enumerate(pred) if pred_matched[pi]]
    fp = [p for pi, p in enumerate(pred) if not pred_matched[pi]]
    fn = [g for gi, g in enumerate(gold) if not gold_matched[gi]]
    return tp, fp, fn


def load_test_cases(*paths):
    cases = []
    for path in paths:
        with open(path) as f:
            cases.extend(json.load(f))
    return cases


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def run_benchmark(detector_fn, test_case_paths, mode="exact", detector_name="detector"):
    cases = load_test_cases(*test_case_paths)

    per_type_tp = defaultdict(int)
    per_type_fp = defaultdict(int)
    per_type_fn = defaultdict(int)
    latencies = []
    failures = []  # cases with worst mismatches, for manual inspection

    for case in cases:
        text = case["text"]
        gold = case["entities"]

        start_t = time.perf_counter()
        try:
            pred = detector_fn(text)
        except Exception as e:
            failures.append({"id": case["id"], "error": str(e)})
            continue
        latencies.append(time.perf_counter() - start_t)

        tp, fp, fn = match_entities(gold, pred, mode=mode)

        for e in tp:
            per_type_tp[e["type"]] += 1
        for e in fp:
            per_type_fp[e["type"]] += 1
        for e in fn:
            per_type_fn[e["type"]] += 1

        if fp or fn:
            failures.append({
                "id": case["id"],
                "text": text,
                "false_positives": fp,
                "false_negatives": fn,
            })

    # per-type metrics
    all_types = sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
    per_type_metrics = {}
    total_tp = total_fp = total_fn = 0
    for etype in all_types:
        tp_n, fp_n, fn_n = per_type_tp[etype], per_type_fp[etype], per_type_fn[etype]
        total_tp += tp_n
        total_fp += fp_n
        total_fn += fn_n
        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) else 0.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_type_metrics[etype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp_n + fn_n,
        }

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) else 0.0)

    results = {
        "detector_name": detector_name,
        "mode": mode,
        "num_cases": len(cases),
        "num_failures_to_run": len([f for f in failures if "error" in f]),
        "overall": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
        "per_type": per_type_metrics,
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 5) if latencies else None,
            "p50": round(percentile(latencies, 50), 5),
            "p95": round(percentile(latencies, 95), 5),
            "p99": round(percentile(latencies, 99), 5),
            "max": round(max(latencies), 5) if latencies else None,
        },
        "failures_sample": [f for f in failures if "error" not in f][:10],
        "run_errors": [f for f in failures if "error" in f][:10],
    }
    return results


def print_report(results):
    print(f"\n{'=' * 60}")
    print(f"Detector: {results['detector_name']}  |  mode: {results['mode']}  |  cases: {results['num_cases']}")
    print(f"{'=' * 60}")
    o = results["overall"]
    print(f"OVERALL  precision={o['precision']:.3f}  recall={o['recall']:.3f}  f1={o['f1']:.3f}")
    print(f"\n{'Entity Type':<20}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<8}")
    for etype, m in results["per_type"].items():
        print(f"{etype:<20}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}{m['support']:<8}")
    lat = results["latency_seconds"]
    print(f"\nLatency (s)  mean={lat['mean']}  p50={lat['p50']}  p95={lat['p95']}  p99={lat['p99']}  max={lat['max']}")
    if results["run_errors"]:
        print(f"\n{len(results['run_errors'])} cases raised errors during detection (see 'run_errors' in output).")
    print()


if __name__ == "__main__":
    # Smoke test with a trivial regex-only detector, just to prove the
    # harness itself works before plugging in Presidio/GLiNER/DistilBERT.
    import re

    def toy_detector(text):
        entities = []
        for m in re.finditer(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
            entities.append({"type": "EMAIL", "text": m.group(), "start": m.start(), "end": m.end()})
        return entities

    results = run_benchmark(
        toy_detector,
        ["evaluation/test_cases.json"],
        mode="exact",
        detector_name="toy_regex_email_only",
    )
    print_report(results)

    with open("evaluation/results_toy.json", "w") as f:
        json.dump(results, f, indent=2)
