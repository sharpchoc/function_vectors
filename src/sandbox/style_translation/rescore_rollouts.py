#!/usr/bin/env python
"""Re-apply scoring (sentence cut + style decision) to stored rollout records from `tail_raw`.
Records whose cut tail changes get their judge verdict reset (the judge saw the old tail) so
`judge_rollouts.py` re-judges just those. usage: rescore_rollouts.py [--dir D] [--families ...] [--dry]"""
import argparse, json, sys
from pathlib import Path
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.scoring import cut_sentence, decide

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, default=ARTIFACTS_ROOT / "style_translation" / "rollouts")
ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
for fam in a.families:
    f = a.dir / f"{fam}.json"
    if not f.exists():
        continue
    recs = json.load(open(f)); n_tail = n_dec = 0
    for r in recs:
        cut, capped = cut_sentence(r["tail_raw"])
        dec = decide(r["family"], r["seg_prefix"], cut, r["next_nat"], r["next_alt"])
        if cut != r["tail"]:
            n_tail += 1
            if not a.dry:
                r["tail"], r["capped"], r["judge"] = cut, capped, None
        if dec != r.get("decision"):
            n_dec += 1
        if not a.dry:
            r["decision"], r["style_ok"] = dec, (dec == r["style"])
    if not a.dry:
        json.dump(recs, open(f, "w"), ensure_ascii=False)
    print(f"{fam:16s} records {len(recs):5d} | tail changed {n_tail:4d} | decision changed {n_dec:4d}{'  (dry)' if a.dry else ''}")
