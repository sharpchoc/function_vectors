#!/usr/bin/env python
"""Generate the neutral base corpus for the style-property study via OpenRouter.

~320 mundane-topic prose docs (~350 words, 10-14 sentences), written in the *nat*
convention for every property (US spelling, standard caps, straight quotes, single
space, serial comma, digits, % sign, contractions, em dashes) with an explicit
feature-mix instruction so every property has opportunity density. The togglers in
properties.py detect BOTH surface forms, so residual convention slips in the generated
text are absorbed at build time, not here.

Resumable: skips doc ids already present in the output file. Output:
dataset_files/style_properties/base_corpus.json  [{doc_id, topic, text}]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import REPO_ROOT

OUT_PATH = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus.json"
KEY_PATH = Path.home() / ".openrouter_key"
URL = "https://openrouter.ai/api/v1/chat/completions"

TOPICS = [
    "maintaining a bicycle", "how a local farmers market operates", "packing for a short trip",
    "the routine of a small-town library", "how bread is baked in a home kitchen",
    "organizing a community cleanup day", "the daily schedule of a ferry service",
    "how a neighborhood garden is planned", "repainting an old wooden fence",
    "sorting and storing winter clothes", "how a small museum catalogs its items",
    "planning a school science fair", "the workings of a municipal recycling center",
    "how a family plans a road trip", "keeping houseplants healthy through winter",
    "the process of moving to a new apartment", "how a corner bakery starts its morning",
    "training for a five-kilometer run", "how a public pool is maintained",
    "setting up a home office in a spare room", "the routine of a dog-walking service",
    "how a town prepares for a street festival", "restoring a secondhand desk",
    "how a food co-op divides its weekly orders", "learning to knit a simple scarf",
    "the operations of a bicycle repair shop", "how a community theater stages a play",
    "organizing a garage sale", "how mail is sorted at a small post office",
    "keeping a weather journal", "how a coffee roaster schedules its batches",
    "planning meals for a busy week", "how volunteers run a soup kitchen",
    "the upkeep of a hiking trail", "how an office plans its supply orders",
    "assembling flat-pack furniture", "how a swim team organizes practice",
    "the workflow of a small print shop", "how a landlord schedules building repairs",
    "preparing a vegetable bed in spring", "how a choir rehearses for a concert",
    "the routine of an early-morning fish market", "how students organize a study group",
    "maintaining a shared laundry room", "how a hardware store tracks inventory",
    "planning a picnic for a large family", "how a night market sets up its stalls",
    "the care of a saltwater aquarium", "how a courier service plans its routes",
    "repotting overgrown houseplants", "how a bookshop arranges its shelves",
    "the schedule of a community radio station", "how apples are pressed into cider",
    "organizing files in a shared office", "how a youth soccer league schedules games",
    "the maintenance of a public fountain", "how a bakery tests a new recipe",
    "preparing a house for house guests", "how a ceramics studio fires its kiln",
    "the routine of a park groundskeeper", "how neighbors share a tool library",
    "canning tomatoes at the end of summer", "how a small airline schedules its crews",
    "the process of digitizing old photographs", "how a climbing gym sets new routes",
    "keeping bees in a suburban backyard", "how a thrift store processes donations",
    "the workflow of a bicycle courier", "how a lake house is closed for winter",
    "organizing a neighborhood watch meeting", "how a cheese shop ages its stock",
    "the upkeep of a village clock tower", "how a research library repairs old books",
    "planning a surprise birthday dinner", "how a ski lodge prepares for the season",
    "the routine of a harbor master", "how a print newspaper is laid out",
    "organizing a charity fun run", "how a botanical garden labels its plants",
]

PROMPT = """Write a plain informative text of 10 to 14 sentences (about 350 words) about {topic}.

Style requirements (follow all of them exactly):
- Standard American English spelling and standard capitalization.
- Straight quotation marks ("), single space after every period, serial comma in lists.
- No headings, no markdown, no bullet points; one or two plain paragraphs.

Include ALL of the following, worked in naturally:
- one list of three or more items written as "X, Y, and Z"
- at least two whole numbers between 2 and 20 written as numerals
- one percentage written with the % sign
- one short direct quotation from a named person, with the comma or period inside the closing quote
- one aside set off by an em dash (—)
- one sentence that trails off with a three-dot ellipsis (...)
- at least three contractions (such as don't, it's, they're)
- the word "and" joining two nouns, at least twice
- one ordinal written like 3rd or 5th
- several of these words where natural: color, center, organize, realize, neighbor, favorite, traveled, gray, meter, behavior, recognize, learned, burned, while, among

Return only the text."""

# Seeded batches: extra density instructions for properties that are sparse in general
# prose (plan Stage A1 "lexical seeding"). Each batch serves a family of properties.
BATCHES = {
    "gen": "",
    "num": ("\nAdditionally, this text must be statistics-heavy: include at least 10 whole "
            "numbers between 2 and 20 written as numerals, at least 4 percentages written "
            "with the % sign, and at least 4 ordinals written like 3rd or 7th."),
    "dlg": ("\nAdditionally, this text must be quotation-heavy: include at least 6 short "
            "direct quotations from named people, each with the comma or period inside the "
            "closing quote, and at least 3 sentences that trail off with a three-dot "
            "ellipsis (...)."),
    "dsh": ("\nAdditionally: include at least 5 asides set off by em dashes (—) and at "
            "least 5 lists of exactly three items written as \"X, Y, and Z\"."),
    "msc": ("\nAdditionally: include at least 5 sentences that trail off with a three-dot "
            "ellipsis (...); use the past forms learned, spelled, burned, dreamed, leaped, "
            "leaned, spilled, and spoiled at least 6 times in total; and use the words "
            "while, among, and amid at least 6 times in total."),
    "ukv": ("\nAdditionally, use MANY of these words — at least 14 occurrences in total: "
            "color, colors, colorful, favorite, center, organize, organized, organization, "
            "realize, realized, recognize, recognized, neighbor, neighbors, neighborhood, "
            "traveled, traveling, labeled, gray, meter, meters, behavior, flavor, honor, "
            "learned, spelled, burned, dreamed, while, among, analyze, emphasize, "
            "summarize, theater, catalog, defense."),
}


def load_key():
    for line in KEY_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "OPENROUTER" in line.upper() and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
        if line and "=" not in line and not line.startswith("#"):
            return line
    raise RuntimeError(f"no key found in {KEY_PATH}")


def gen_one(key, model, topic, doc_id, variant, batch_extra=""):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(topic=topic) + batch_extra
                      + f"\n\n(Variation {variant} of 4 — take a different angle than other variations.)"}],
        "temperature": 1.0,
        "max_tokens": 900,
    }
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            txt = txt.replace(" ", " ").replace("\r", "")
            if len(txt.split()) < 150:
                raise ValueError(f"too short ({len(txt.split())} words)")
            return {"doc_id": doc_id, "topic": topic, "text": txt}
        except Exception as e:
            if attempt == 3:
                print(f"{doc_id} FAILED: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--per_topic", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", default="gen", choices=sorted(BATCHES))
    ap.add_argument("--n_topics", type=int, default=None,
                    help="use only the first N topics (seeded batches)")
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = {}
    if OUT_PATH.exists():
        docs = {d["doc_id"]: d for d in json.load(open(OUT_PATH))}
        print(f"resume: {len(docs)} docs already present", flush=True)

    key = load_key()
    todo = []
    prefix = "d" if args.batch == "gen" else args.batch
    topics = TOPICS[:args.n_topics] if args.n_topics else TOPICS
    for ti, topic in enumerate(topics):
        for v in range(args.per_topic):
            doc_id = f"{prefix}{ti:03d}_{v}"
            if doc_id not in docs:
                todo.append((topic, doc_id, v + 1))
    print(f"{len(todo)} docs to generate with {args.model} (batch={args.batch})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, key, args.model, t, d, v, BATCHES[args.batch]): d
                for t, d, v in todo}
        done = 0
        for f in as_completed(futs):
            rec = f.result()
            done += 1
            if rec:
                docs[rec["doc_id"]] = rec
            if done % 25 == 0:
                json.dump(sorted(docs.values(), key=lambda d: d["doc_id"]),
                          open(OUT_PATH, "w"), indent=0)
                print(f"{done}/{len(todo)} done ({len(docs)} total)", flush=True)

    json.dump(sorted(docs.values(), key=lambda d: d["doc_id"]), open(OUT_PATH, "w"), indent=0)
    print(f"final: {len(docs)} docs -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
