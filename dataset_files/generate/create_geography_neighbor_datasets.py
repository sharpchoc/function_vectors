"""
Generate geographic spatial-neighbour task pairs for `paired_tasks`.

The matched-marginal criterion is hard in geography: attribute relations
(country->capital vs ->currency) have disjoint OUTPUT pools (output leak), and
entity-type->country relations (landmark vs park ->country) leak via the INPUT
(park names literally contain "Park"). The one family that survives is
place->place SPATIAL relations over a single shared pool — the geographic analog
of next/prev_number:

    east_neighbour:  country -> nearest country in its EAST quadrant
    west_neighbour:  country -> nearest country in its WEST quadrant

Both are country->country, so input AND output marginals match (a lone country
tells you nothing about which direction), and the relation is one the model knows.
Neighbours are computed deterministically from each country's most-populous-city
coordinates (geonamescache, offline): bearing quadrant + nearest great-circle
distance. Input set = countries that have BOTH an east and a west neighbour (so the
two datasets share an identical input set => matched input marginal).

Multi-token country names are allowed (handled downstream by first/last label-token
scoring, like compound number-words).

Run:  python dataset_files/generate/create_geography_neighbor_datasets.py
"""
import json
import math
import os

import geonamescache

OUT = "dataset_files/paired_tasks"


def haversine(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def bearing(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def main():
    gc = geonamescache.GeonamesCache()
    name = {cc: info["name"] for cc, info in gc.get_countries().items()}
    # anchor each country at its most-populous city
    anchor = {}
    for c in gc.get_cities().values():
        cc = c["countrycode"]
        pop = c["population"] or 0
        if cc not in anchor or pop > anchor[cc][2]:
            anchor[cc] = (c["latitude"], c["longitude"], pop)
    ccs = [cc for cc in anchor if cc in name]

    def nearest_in_quadrant(cc, lo, hi):
        a = anchor[cc][:2]
        best = None
        for o in ccs:
            if o == cc:
                continue
            br = bearing(a, anchor[o][:2])
            inq = (lo <= br < hi) if lo < hi else (br >= lo or br < hi)
            if not inq:
                continue
            d = haversine(a, anchor[o][:2])
            if best is None or d < best[1]:
                best = (o, d)
        return best[0] if best else None

    east = {cc: nearest_in_quadrant(cc, 45, 135) for cc in ccs}
    west = {cc: nearest_in_quadrant(cc, 225, 315) for cc in ccs}

    # shared input set: countries with BOTH neighbours defined -> matched input marginal
    inputs = sorted([cc for cc in ccs if east[cc] and west[cc]], key=lambda cc: name[cc])
    east_ds = [{"input": name[cc], "output": name[east[cc]]} for cc in inputs]
    west_ds = [{"input": name[cc], "output": name[west[cc]]} for cc in inputs]

    os.makedirs(OUT, exist_ok=True)
    for fname, ds in [("east_neighbor.json", east_ds), ("west_neighbor.json", west_ds)]:
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(ds, f, indent=2)
        print(f"wrote {fname:20s} {len(ds):4d} pairs   e.g. {ds[0]['input']!r} -> {ds[0]['output']!r}")
    # sanity: identical input sets
    assert [d["input"] for d in east_ds] == [d["input"] for d in west_ds]
    print(f"input sets identical across east/west: True  ({len(inputs)} countries)")


if __name__ == "__main__":
    main()
