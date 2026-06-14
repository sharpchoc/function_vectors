"""
Generate periodic-table spatial-neighbour task pairs for `paired_tasks`.

The periodic table is a 2D grid the model knows — the chemistry analog of the
geography map. Two grid-neighbour relations:

    next_in_period:  element -> the element to its RIGHT  (same row/period, atomic_number+1)
    next_in_group:   element -> the element BELOW it      (same column/group, next period down)

Both are element->element (shared pool), so input AND output marginals match — a lone
element name reveals nothing about which direction (period-right vs group-below), exactly
like east/west for countries but on a known scientific grid.

Input set = elements that have BOTH a right and a below neighbour (so the two datasets
share an identical input set => matched input marginal). This is the main-group +
transition block (66 elements) — the f-block (lanthanides/actinides) has no group_id in
the source data and is excluded; those are also the elements the model knows worst.

Element names are mostly multi-token (only ~14 single-token); allowed, handled downstream
by first/last label-token scoring (like compound number-words / multi-token countries).

Data: `mendeleev` (offline, bundled). Run:
  python dataset_files/generate/create_periodic_table_neighbor_datasets.py
"""
import json
import math
import os

from mendeleev.fetch import fetch_table

OUT = "dataset_files/paired_tasks"


def main():
    df = fetch_table("elements")
    els = df[["name", "atomic_number", "period", "group_id"]].to_dict("records")
    els = [e for e in els if e["period"] == e["period"]]  # drop NaN period
    by_num = {int(e["atomic_number"]): e for e in els}

    def has_group(e):
        g = e["group_id"]
        return g is not None and not (isinstance(g, float) and math.isnan(g))

    def right(e):  # next element in the same period (to the right)
        nxt = by_num.get(int(e["atomic_number"]) + 1)
        return nxt if nxt and nxt["period"] == e["period"] else None

    def below(e):  # next element in the same group (down a period)
        if not has_group(e):
            return None
        cands = [x for x in els if has_group(x) and x["group_id"] == e["group_id"] and x["period"] > e["period"]]
        return min(cands, key=lambda x: x["period"]) if cands else None

    inputs = sorted([e for e in els if right(e) and below(e)],
                    key=lambda e: int(e["atomic_number"]))
    period_ds = [{"input": e["name"], "output": right(e)["name"]} for e in inputs]
    group_ds = [{"input": e["name"], "output": below(e)["name"]} for e in inputs]

    os.makedirs(OUT, exist_ok=True)
    for fname, ds in [("next_in_period.json", period_ds), ("next_in_group.json", group_ds)]:
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(ds, f, indent=2)
        print(f"wrote {fname:20s} {len(ds):4d} pairs   e.g. {ds[0]['input']!r} -> {ds[0]['output']!r}")
    assert [d["input"] for d in period_ds] == [d["input"] for d in group_ds]
    assert all(d["input"] != d["output"] for d in period_ds + group_ds)
    print(f"input sets identical across period/group: True  ({len(inputs)} elements); no self-pairs")


if __name__ == "__main__":
    main()
