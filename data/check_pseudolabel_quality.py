import json
import sys
sys.path.insert(0, ".")
from detectors.ensemble_detector import ensemble_detector
from evaluation.benchmark import match_entities

with open("data/phase4_corpus.json") as f:
    corpus = json.load(f)

per_type_tp = {}
per_type_fp = {}
per_type_fn = {}

for case in corpus:
    gold = case["injected_entities"]
    pred = ensemble_detector(case["text"])
    tp, fp, fn = match_entities(gold, pred, mode="exact")
    for e in tp:
        per_type_tp[e["type"]] = per_type_tp.get(e["type"], 0) + 1
    for e in fp:
        per_type_fp[e["type"]] = per_type_fp.get(e["type"], 0) + 1
    for e in fn:
        per_type_fn[e["type"]] = per_type_fn.get(e["type"], 0) + 1

all_types = sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
print(f"{'Type':<20}{'Precision':<12}{'Recall':<12}{'Support':<8}")
for t in all_types:
    tp, fp, fn = per_type_tp.get(t, 0), per_type_fp.get(t, 0), per_type_fn.get(t, 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(f"{t:<20}{precision:<12.3f}{recall:<12.3f}{tp+fn:<8}")
