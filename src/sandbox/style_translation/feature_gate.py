#!/usr/bin/env python
"""Phase 3/4 gate (user decision 2026-09-14): a family's write (cue-token) / read (evidence-token) feature "works" if the best
steered accuracy's Wilson CI excludes the unsteered rate, in BOTH directions. Reads <results>/steering/best_config.csv and
<results>/read_steer/best_config.csv, writes <results>/pool_gated.json and prints the table."""
import argparse, csv, json, sys
from pathlib import Path
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--pool", default=None)
ap.add_argument("--headroom", type=float, default=0.70, help="headroom-aware variant: a direction whose unsteered rate is already >= this is exempt (ceiling)")
args = ap.parse_args(); MP = model_paths(args.model); R = MP["results"]
pool = json.load(open(args.pool or (R / "pool.json")))["pool"]


def gate(path, key_dir):
    rows = list(csv.DictReader(open(path))) if Path(path).exists() else []
    out = {}
    for r in rows:
        f = r["family"]; d = r[key_dir]
        base = float(r.get("unsteered_accuracy", r.get("base_accuracy")))   # steering: base_accuracy; read_steer: unsteered_accuracy
        ok = float(r["ci_lo"]) > base
        out.setdefault(f, {})[d] = dict(ok=ok, acc=float(r["accuracy"]), unsteered=base, ci_lo=float(r["ci_lo"]), layer=r["layer"], alpha=r["alpha"])
    return out


w = gate(R / "steering" / "best_config.csv", "target"); rd = gate(R / "read_steer" / "best_config.csv", "direction")
res = {}
print(f"{'family':14s} {'write nat':>18s} {'write alt':>18s} {'read nat2alt':>18s} {'read alt2nat':>18s}  gate")
for f in pool:
    def cell(d, k):
        c = d.get(f, {}).get(k)
        return f"{c['unsteered']:.2f}→{c['acc']:.2f}{'✓' if c['ok'] else '✗'}" if c else "   —   "
    w_ok = f in w and len(w[f]) == 2 and all(c["ok"] for c in w[f].values())
    r_ok = f in rd and len(rd[f]) == 2 and all(c["ok"] for c in rd[f].values())
    hr = lambda d: f in d and len(d[f]) == 2 and all(c["ok"] or c["unsteered"] >= args.headroom for c in d[f].values())
    w_hr, r_hr = hr(w), hr(rd)
    res[f] = dict(write_ok=w_ok, read_ok=r_ok, in_map=w_ok and r_ok, in_map_headroom=w_hr and r_hr, write=w.get(f), read=rd.get(f))
    print(f"{f:14s} {cell(w,'nat'):>18s} {cell(w,'alt'):>18s} {cell(rd,'nat2alt'):>18s} {cell(rd,'alt2nat'):>18s}  strict {'IN ' if res[f]['in_map'] else 'out'} | headroom {'IN' if res[f]['in_map_headroom'] else 'out'}")
json.dump({"pool": pool, "gate": "Wilson CI of best steered accuracy excludes the unsteered rate, both directions, write AND read", "families": res,
           "map_pool": [f for f in pool if res[f]["in_map"]], "headroom": args.headroom,
           "map_pool_headroom": [f for f in pool if res[f]["in_map_headroom"]]}, open(R / "pool_gated.json", "w"), indent=1)
print(f"map pool strict: {sum(r['in_map'] for r in res.values())} of {len(pool)} | headroom-aware (unsteered >= {args.headroom:.2f} exempt): "
      f"{sum(r['in_map_headroom'] for r in res.values())} -> {R / 'pool_gated.json'}")
