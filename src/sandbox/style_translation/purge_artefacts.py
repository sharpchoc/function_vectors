#!/usr/bin/env python
"""Strip Gemini control-token artefacts (e.g. '<ctrl63>') from the code twins (user decision 2026-09-17), re-align the affected documents
(full opportunity list; the free-opportunity rule for the affected families), rewrite pairs + raw docs; then, for --rebuild families,
regenerate cues / prompts / evidence and report which (doc, style, k) prompts changed; records of changed prompts are removed from the
derived files (rollouts, steering confirm, read_steer confirm[_k1], prompt_pairs) so nothing stale survives."""
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_build import align
from src.sandbox.style_translation.code_free import filter_free, AFFECTED
ART = re.compile(r"<ctrl\d+>")
PY = sys.executable


def clean_doc(r, fam):
    nat, alt = ART.sub("", r["text_nat"]), ART.sub("", r["text_alt"])
    opps, shared = align(nat, alt)
    ok = opps is not None and len(opps) >= 5 and shared >= 0.6 and opps[4]["nat_span"][0] < 0.75 * len(nat) and all(o["nat"] != o["alt"] for o in opps)
    out = dict(r, text_nat=nat, text_alt=alt, opps=opps or [], k_en=len(opps or []), shared_fraction=round(shared, 3)); out.pop("opps_all", None); out.pop("free_filter", None)
    out["pass"] = bool(ok)
    if ok and fam in AFFECTED:
        fr = filter_free(out, fam); out["pass"] = fr is not None
        if fr: out.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--families", nargs="+", required=True); ap.add_argument("--rebuild", nargs="*", default=None, help="families whose cues/prompts/evidence to rebuild now (default: all given)")
    ap.add_argument("--model", default="qwen25_base"); args = ap.parse_args()
    MP = model_paths(args.model); PAIRS = STYLE_TRANSLATION_DATA / "pairs"; RAW = STYLE_TRANSLATION_DATA / "code"
    touched = {}
    for fam in args.families:
        pairs = json.load(open(PAIRS / f"{fam}.json")); ids = {r["doc_id"] for r in pairs}
        raw = json.load(open(RAW / f"{fam}.json")) if (RAW / f"{fam}.json").exists() else []
        changed = []
        for coll in (pairs, raw):
            for i, r in enumerate(coll):
                if ART.search(r.get("text_nat", "")) or ART.search(r.get("text_alt", "")):
                    coll[i] = clean_doc(r, fam); changed.append(r["doc_id"])
        pairs = [r for r in pairs if r["pass"]]
        json.dump(pairs, open(PAIRS / f"{fam}.json", "w"), ensure_ascii=False, indent=0)
        if raw: json.dump(raw, open(RAW / f"{fam}.json", "w"), ensure_ascii=False, indent=0)
        dropped = sorted(ids - {r["doc_id"] for r in pairs}); touched[fam] = (sorted(set(changed)), dropped)
        print(f"{fam:18s} stripped in {len(set(changed))} docs; still passing: {len(pairs)} (dropped {dropped})", flush=True)
    rebuild = args.families if args.rebuild is None else args.rebuild
    if not rebuild:
        return
    old = {}
    for fam in rebuild:
        old[fam] = {(p["doc_id"], p["style"], p["k"]): p["prompt_ids"] for p in json.load(open(MP["prompts"] / f"{fam}.json"))}
    D = "src/sandbox/style_translation"
    for script, extra in ((f"{D}/cue_tokens.py", []), (f"{D}/build_prompts.py", ["--K", "5"]), (f"{D}/evidence_tokens.py", [])):
        import os
        r = subprocess.run([PY, script, "--model", args.model, *extra, "--families", *rebuild], cwd=_BOOT, capture_output=True, text=True, env={**os.environ, "MKL_THREADING_LAYER": "GNU"})
        if r.returncode != 0:
            print(r.stderr[-2000:]); raise SystemExit(f"{script} failed")
    for fam in rebuild:
        new = {(p["doc_id"], p["style"], p["k"]): p["prompt_ids"] for p in json.load(open(MP["prompts"] / f"{fam}.json"))}
        # compare only the k values this rebuild produced (K = 5 -> k <= 4); items with other k (e.g. appended k = 8) are left alone
        ks = {k[2] for k in new}
        changed_keys = {k for k in new if old[fam].get(k) != new[k]} | {k for k in old[fam] if k[2] in ks and k not in new}
        docs = sorted({k[0] for k in changed_keys})
        print(f"{fam:18s} prompts changed or removed: {len(changed_keys)} (docs {docs})", flush=True)
        if not changed_keys:
            continue
        for sub in ("rollouts", "steering/confirm", "read_steer/confirm", "read_steer/confirm_k1"):
            p = MP["artifacts"] / sub / f"{fam}.json" if "artifacts" in MP else None
            p = (MP["rollouts"].parent / sub / f"{fam}.json")
            if p.exists():
                recs = json.load(open(p)); keep = [r for r in recs if r["doc_id"] not in docs]
                json.dump(keep, open(p, "w"), ensure_ascii=False); print(f"    {sub}: {len(recs) - len(keep)} records removed")
        pp = MP["prompt_pairs"] / f"{fam}.npz"
        if pp.exists():
            d = np.load(pp); m = ~np.isin(d["doc_id"], docs)
            np.savez_compressed(pp, **{k: (d[k][m] if d[k].shape and d[k].shape[0] == len(m) else d[k]) for k in d.files}); print(f"    prompt_pairs: {(~m).sum()} rows removed")


if __name__ == "__main__":
    main()
