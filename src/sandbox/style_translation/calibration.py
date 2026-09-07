"""Haiku-vs-regex calibration on the FINAL corpus + a one-text-per-family spot-read dump."""
import json, sys, glob, statistics
sys.path.insert(0, '/workspace/function_vectors'); sys.path.insert(0, '/workspace/function_vectors/src')
from src.sandbox.style_translation.families import FAMILIES
D = '/workspace/function_vectors/dataset_files/style_translation'
print(f"{'family':14s} {'n':>3s} {'k_haiku med':>11s} {'k_regex med':>11s} {'|diff|<=1':>9s} {'haiku<regex':>11s} {'words med':>9s}")
for f in FAMILIES:
    recs = json.load(open(f'{D}/final/{f.name}.json'))
    kh = [r['verify']['k_found'] for r in recs]; kr = [r['regex_k'] for r in recs]
    close = sum(abs(a - b) <= 1 for a, b in zip(kh, kr)) / len(recs)
    lower = sum(a < b for a, b in zip(kh, kr)) / len(recs)
    print(f"{f.name:14s} {len(recs):3d} {statistics.median(kh):11.0f} {statistics.median(kr):11.0f} {close:9.2f} {lower:11.2f} {statistics.median(r['words'] for r in recs):9.0f}")
if len(sys.argv) > 1:
    print("\n===== one final text per family (index 7) =====")
    for f in FAMILIES:
        r = json.load(open(f'{D}/final/{f.name}.json'))[7]
        print(f"\n[{f.name}] k_haiku={r['verify']['k_found']} anchors={r['verify']['anchors'][:7]}\n{r['text_es']}")
