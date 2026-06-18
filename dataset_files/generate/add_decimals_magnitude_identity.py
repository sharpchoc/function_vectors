"""
Double the magnitude/identity ambiguous datasets by appending DECIMAL examples (<=3 dp),
mirroring the existing integer structure. Overwrites the canonical
dataset_files/ambiguous/{magnitude,identity}.json after backing up the integer-only originals.

Structure (parallels build_magnitude_identity):
  existing 300 ints: positives 1..150 (overlap) + negatives -1..-150 (differentiator)
  appended 300 decimals: 150 positive decimals d (overlap) + their negatives -d (differentiator)
    overlap:        input "+d" -> magnitude "d",   identity "d"
    differentiator: input "-d" -> magnitude "d",   identity "-d"   (magnitude drops the sign)
Final: 600 entries/file, overlap=300 (150 int + 150 dec), differentiator=300.
"""
import json, os, random

AMBIG = os.path.join(os.path.dirname(__file__), "..", "ambiguous")
SEED = 12345
N_DEC = 150


def gen_decimals(n, rng):
    """n distinct positive decimals: integer part 1..150, 1-3 fractional digits, no trailing
    zero (canonical string), fractional part non-zero so they are genuinely decimals."""
    seen = set()
    out = []
    while len(out) < n:
        ipart = rng.randint(1, 150)
        ndp = rng.randint(1, 3)
        frac = "".join(str(rng.randint(0, 9)) for _ in range(ndp)).rstrip("0")
        if frac == "":                       # avoid x.0 / trailing-zero-only -> not a clean decimal
            continue
        s = f"{ipart}.{frac}"
        if s in seen:
            continue
        seen.add(s); out.append(s)
    return out


def main():
    mag_p = os.path.join(AMBIG, "magnitude.json")
    id_p = os.path.join(AMBIG, "identity.json")
    mag = json.load(open(mag_p))
    idn = json.load(open(id_p))
    assert len(mag) == len(idn) == 300, f"expected 300 int entries, got {len(mag)}/{len(idn)}"

    # back up integer-only originals (reversible)
    for src, name in [(mag, "magnitude"), (idn, "identity")]:
        bpath = os.path.join(AMBIG, f"{name}.int_only.json")
        if not os.path.exists(bpath):
            json.dump(src, open(bpath, "w"), indent=2)
            print("backed up ->", bpath)

    rng = random.Random(SEED)
    decs = gen_decimals(N_DEC, rng)                      # 150 positive decimal strings

    # overlap decimals (+d): both outputs = d ; differentiator decimals (-d): mag=d, id=-d
    pos_inputs = decs
    neg_inputs = ["-" + d for d in decs]
    dec_inputs = pos_inputs + neg_inputs                 # 300, overlap then differentiator

    for inp in dec_inputs:
        d_abs = inp.lstrip("-")
        mag.append({"input": inp, "output": d_abs})      # magnitude = absolute value
        idn.append({"input": inp, "output": inp})        # identity = the number itself

    json.dump(mag, open(mag_p, "w"), indent=2)
    json.dump(idn, open(id_p, "w"), indent=2)

    # ---- assertions ----
    assert [m["input"] for m in mag] == [i["input"] for i in idn], "inputs misaligned"
    overlap = [i for i, (a, b) in enumerate(zip(mag, idn)) if a["output"] == b["output"]]
    differ = [i for i, (a, b) in enumerate(zip(mag, idn)) if a["output"] != b["output"]]
    print(f"magnitude/identity: total={len(mag)} overlap={len(overlap)} differ={len(differ)}")
    print("decimal overlap sample:", mag[300], idn[300])
    print("decimal differ  sample:", mag[450], idn[450])
    assert len(mag) == 600 and len(overlap) == 300 and len(differ) == 300


if __name__ == "__main__":
    main()
